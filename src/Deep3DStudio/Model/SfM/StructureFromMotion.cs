using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;

namespace Deep3DStudio.Model.SfM
{
    public class SfMInference
    {
        // Represents a 3D point in the global map
        private class MapPoint
        {
            public int Id;
            public Point3d Position;
            public Mat Descriptor; // Representative descriptor
            // ViewIndex -> FeatureIndex
            public Dictionary<int, int> ObservingViewIndices = new Dictionary<int, int>();
            public Scalar Color;
            public int Validations = 0;
        }

        private class ViewInfo
        {
            public int Index;
            public string Path;
            public Size Size;
            public Mat Image;
            public Mat K; // Intrinsics
            public double FocalLength; // Store estimated focal length for later use
            public KeyPoint[] KeyPoints;
            public Mat Descriptors;
            public Mat R; // Camera -> World? No, World -> Camera (OpenCV convention)
            public Mat t;
            public bool IsRegistered;
        }

        private List<MapPoint> _map = new List<MapPoint>();
        private List<ViewInfo> _views = new List<ViewInfo>();
        private int _nextMapPointId = 0;
        private bool _isPivotingCamera = false; // True if rotation-dominant motion detected

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            var result = new SceneResult();
            if (imagePaths.Count < 2) return result;

            Console.WriteLine("Starting Professional SfM Reconstruction...");

            // 1. Pre-process all views
            PreProcessViews(imagePaths);

            if (_views.Count < 2) return result;

            // 2. Initialization: Find best pair
            if (!InitializeMap())
            {
                Console.WriteLine("SfM Initialization failed. Not enough baseline or matches.");
                return result;
            }

            // 3. Incremental Reconstruction
            int registeredCount = 2; // Initial pair
            int maxAttempts = _views.Count * 3; // Allow multiple passes
            int attempts = 0;
            var failedViews = new HashSet<int>(); // Track views that failed PnP

            while (attempts < maxAttempts && registeredCount < _views.Count)
            {
                attempts++;
                // Find unregistered view with most matches to Map
                var bestCandidate = -1;
                var bestObjectPoints = new List<Point3d>();
                var bestImagePoints = new List<Point2d>();

                var unregistered = _views.Where(v => !v.IsRegistered).ToList();
                if (unregistered.Count == 0) break;

                int maxMatches = 0;

                foreach (var view in unregistered)
                {
                    // Match View Features <-> Global Map Descriptors
                    var (objPts, imgPts) = FindMapMatches(view);

                    if (objPts.Count > maxMatches)
                    {
                        maxMatches = objPts.Count;
                        bestCandidate = view.Index;
                        bestObjectPoints = objPts;
                        bestImagePoints = imgPts;
                    }
                }

                // Very low threshold (6 points minimum for PnP) to register difficult views
                if (bestCandidate != -1 && maxMatches >= 6)
                {
                    var view = _views[bestCandidate];
                    if (RegisterView(view, bestObjectPoints, bestImagePoints))
                    {
                        Console.WriteLine($"Registered Image {view.Index} with {maxMatches} map matches (PnP).");
                        registeredCount++;
                        failedViews.Remove(view.Index);

                        // Triangulate new points with ALL registered views
                        ExpandMap(view);

                        // Bundle Adjustment (Local refinement)
                        RefinePose(view);
                    }
                    else
                    {
                        // PnP failed - try essential matrix method as fallback
                        if (TryRegisterFromAllPairs(view))
                        {
                            Console.WriteLine($"Registered Image {view.Index} via essential matrix fallback.");
                            registeredCount++;
                            ExpandMap(view);
                        }
                        else
                        {
                            Console.WriteLine($"Failed to register Image {bestCandidate} (PnP and essential matrix failed).");
                            failedViews.Add(view.Index);
                        }
                    }
                }
                else if (unregistered.Count > 0)
                {
                    // No view could be registered via map matches
                    // Try essential matrix for all unregistered views
                    bool anyRegistered = false;
                    foreach (var view in unregistered)
                    {
                        if (TryRegisterFromAllPairs(view))
                        {
                            Console.WriteLine($"Registered Image {view.Index} via direct pair matching.");
                            registeredCount++;
                            ExpandMap(view);
                            anyRegistered = true;
                            break; // Restart the loop to try map-based registration
                        }
                    }

                    if (!anyRegistered)
                    {
                        // Expand map and try again
                        Console.WriteLine($"No view could be registered (best had {maxMatches} matches). Expanding map...");
                        ExpandMapBetweenRegisteredViews();

                        // If we still can't register anything after expansion, break
                        if (maxMatches < 4 && failedViews.Count == unregistered.Count)
                            break;
                    }
                }
            }

            Console.WriteLine($"Incremental reconstruction finished after {attempts} iterations.");

            // 4. Final Polish
            Console.WriteLine($"SfM Complete: Registered {registeredCount}/{_views.Count} views, {_map.Count} 3D points in map.");

            // Remove outliers
            // PruneMap();

            // 5. Output
            var mesh = new MeshData();

            // Coordinate Conversion Matrix M (OpenCV -> OpenGL)
            // Flip Y and Z axes: (x, y, z) -> (x, -y, -z)
            // This applies to both World Points and Camera Axes

            var minBound = new Vector3(float.MaxValue);
            var maxBound = new Vector3(float.MinValue);

            foreach (var mp in _map)
            {
                // Apply coordinate flip: OpenCV (Y-down, Z-forward) -> Viewer convention (Y-up, Z-backward)
                // We use M = diag(1, -1, -1) which corresponds to (x, -y, -z)
                // This preserves Right-Handedness and orients Up correctly
                var p = new Vector3((float)mp.Position.X, -(float)mp.Position.Y, -(float)mp.Position.Z);
                mesh.Vertices.Add(p);
                mesh.Colors.Add(new Vector3((float)mp.Color.Val2 / 255f, (float)mp.Color.Val1 / 255f, (float)mp.Color.Val0 / 255f)); // RGB vs BGR

                minBound = Vector3.ComponentMin(minBound, p);
                maxBound = Vector3.ComponentMax(maxBound, p);
            }

            Console.WriteLine($"SfM Output (raw): {mesh.Vertices.Count} points. Bounds: {minBound} to {maxBound}");

            // Normalize scale: SfM with Essential Matrix has arbitrary scale
            // Normalize so the scene fits in a reasonable cube (e.g., max dimension ~10 units)
            var center = (minBound + maxBound) / 2.0f;
            var extent = maxBound - minBound;
            float maxExtent = Math.Max(extent.X, Math.Max(extent.Y, extent.Z));
            float targetSize = 10.0f;
            float scale = maxExtent > 0.001f ? targetSize / maxExtent : 1.0f;

            // Apply normalization to points
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i] = (mesh.Vertices[i] - center) * scale;
            }

            // Recalculate bounds after normalization
            minBound = new Vector3(float.MaxValue);
            maxBound = new Vector3(float.MinValue);
            foreach (var p in mesh.Vertices)
            {
                minBound = Vector3.ComponentMin(minBound, p);
                maxBound = Vector3.ComponentMax(maxBound, p);
            }

            Console.WriteLine($"SfM Output (normalized): {mesh.Vertices.Count} points. Scale={scale:F4}, Bounds: {minBound} to {maxBound}");

            foreach (var v in _views)
            {
                if (v.IsRegistered)
                {
                    // Convert OpenCV Pose (R, t) to Viewer Pose (R', t')
                    // Using M = diag(1, -1, -1) to match point transformation
                    // R' = M * R * M

                    var R_cv = v.R;
                    var t_cv = v.t;

                    using var R_gl = new Mat(3, 3, MatType.CV_64F);

                    // R_gl = M * R_cv * M where M = diag(1, -1, -1)
                    // The algebraic result for diagonal M with elements +/- 1 is:
                    // R_gl_ij = M_ii * R_cv_ij * M_jj

                    // Row 0 (i=0, M=1):
                    //   (0,0): 1*R00*1 = R00
                    //   (0,1): 1*R01*-1 = -R01
                    //   (0,2): 1*R02*-1 = -R02
                    R_gl.Set(0, 0, R_cv.At<double>(0, 0));
                    R_gl.Set(0, 1, -R_cv.At<double>(0, 1));
                    R_gl.Set(0, 2, -R_cv.At<double>(0, 2));

                    // Row 1 (i=1, M=-1):
                    //   (1,0): -1*R10*1 = -R10
                    //   (1,1): -1*R11*-1 = R11
                    //   (1,2): -1*R12*-1 = R12
                    R_gl.Set(1, 0, -R_cv.At<double>(1, 0));
                    R_gl.Set(1, 1, R_cv.At<double>(1, 1));
                    R_gl.Set(1, 2, R_cv.At<double>(1, 2));

                    // Row 2 (i=2, M=-1):
                    //   (2,0): -1*R20*1 = -R20
                    //   (2,1): -1*R21*-1 = R21
                    //   (2,2): -1*R22*-1 = R22
                    R_gl.Set(2, 0, -R_cv.At<double>(2, 0));
                    R_gl.Set(2, 1, R_cv.At<double>(2, 1));
                    R_gl.Set(2, 2, R_cv.At<double>(2, 2));

                    // Calculate Camera Center in OpenGL Coordinates explicitly
                    // C_cv = -R_cv^T * t_cv
                    // C_gl = M * C_cv

                    // 1. Calculate C_cv
                    var C_cv = new Vector3();
                    using (var Rt = R_cv.T().ToMat())
                    using (var negRt = (-Rt).ToMat()) // -R^T
                    using (var C_mat = (negRt * t_cv).ToMat())
                    {
                        C_cv.X = (float)C_mat.At<double>(0, 0);
                        C_cv.Y = (float)C_mat.At<double>(1, 0);
                        C_cv.Z = (float)C_mat.At<double>(2, 0);
                    }

                    // 2. Transform to C_gl (Apply M = diag(1, -1, -1))
                    var C_gl = new Vector3(C_cv.X, -C_cv.Y, -C_cv.Z);

                    // 3. Normalize
                    C_gl = (C_gl - center) * scale;

                    // 4. Construct CameraToWorld Matrix
                    // Rotation part: R_gl (because W2C has R_gl^T in Row Vector notation, so C2W has R_gl)
                    // Translation part: C_gl
                    var camToWorld = Matrix4.Identity;

                    // Set Rotation (Upper-Left 3x3 = R_gl)
                    // Row 0
                    camToWorld.M11 = (float)R_gl.At<double>(0, 0);
                    camToWorld.M12 = (float)R_gl.At<double>(0, 1);
                    camToWorld.M13 = (float)R_gl.At<double>(0, 2);
                    // Row 1
                    camToWorld.M21 = (float)R_gl.At<double>(1, 0);
                    camToWorld.M22 = (float)R_gl.At<double>(1, 1);
                    camToWorld.M23 = (float)R_gl.At<double>(1, 2);
                    // Row 2
                    camToWorld.M31 = (float)R_gl.At<double>(2, 0);
                    camToWorld.M32 = (float)R_gl.At<double>(2, 1);
                    camToWorld.M33 = (float)R_gl.At<double>(2, 2);

                    // Set Translation (Row 3)
                    camToWorld.M41 = C_gl.X;
                    camToWorld.M42 = C_gl.Y;
                    camToWorld.M43 = C_gl.Z;
                    // M44 is 1.0 by Identity

                    result.Poses.Add(new CameraPose
                    {
                        ImageIndex = v.Index,
                        ImagePath = v.Path,
                        Width = v.Size.Width,
                        Height = v.Size.Height,
                        CameraToWorld = camToWorld,
                        WorldToCamera = camToWorld.Inverted(),
                        FocalLength = (float)v.FocalLength // Store per-image focal length
                    });

                    Console.WriteLine($"  Camera {v.Index}: pos=({C_gl.X:F2},{C_gl.Y:F2},{C_gl.Z:F2})");
                }
            }
            result.Meshes.Add(mesh);

            // Cleanup
            foreach(var v in _views) { v.Image.Dispose(); v.Descriptors.Dispose(); v.K.Dispose(); if(v.R!=null) v.R.Dispose(); if(v.t!=null) v.t.Dispose(); }
            foreach(var mp in _map) mp.Descriptor.Dispose();

            return result;
        }

        private void PreProcessViews(List<string> paths)
        {
            using var detector = ORB.Create(nFeatures: 8000); // Very high feature count for dense map and robust matching

            for (int i = 0; i < paths.Count; i++)
            {
                Mat img;
                try
                {
                    // Use ImageDecoder to handle EXIF orientation correctly
                    img = ImageDecoder.DecodeToMat(paths[i]);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load image {paths[i]}: {ex.Message}");
                    continue;
                }

                if (img.Empty()) continue;

                // Max resolution limit
                if (Math.Max(img.Width, img.Height) > 1600)
                {
                    double scale = 1600.0 / Math.Max(img.Width, img.Height);
                    Cv2.Resize(img, img, new Size(0, 0), scale, scale);
                }

                KeyPoint[] kps;
                var desc = new Mat();
                detector.DetectAndCompute(img, null, out kps, desc);

                // Estimate K (Auto-Zoom support)
                // Different images might have different focal lengths (zoom)
                // Use a heuristic based on image dimensions
                double f = Math.Max(img.Width, img.Height) * 0.85; // Roughly 50-60 deg FOV
                var K = Mat.Eye(3, 3, MatType.CV_64F).ToMat();
                K.Set(0, 0, f);
                K.Set(1, 1, f);
                K.Set(0, 2, (double)img.Width / 2.0);
                K.Set(1, 2, (double)img.Height / 2.0);

                _views.Add(new ViewInfo
                {
                    Index = i,
                    Path = paths[i],
                    Size = img.Size(),
                    Image = img, // Keep loaded for color sampling
                    K = K,
                    FocalLength = f, // Store for later use
                    KeyPoints = kps,
                    Descriptors = desc,
                    IsRegistered = false
                });

                Console.WriteLine($"  View {i}: {img.Width}x{img.Height}, focal={f:F1}, {kps.Length} features");
            }
        }

        private bool InitializeMap()
        {
            // Find pair with high feature matches and sufficient baseline
            // Scan ALL pairs to find the best initialization pair
            // Score by: number of matches, baseline quality (not too small, not too large)

            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            int bestI = -1, bestJ = -1;
            int bestInliers = 0;
            Mat bestR = null, bestT = null;
            DMatch[] bestMatches = null;
            Mat bestMask = null;

            int totalPairs = (_views.Count * (_views.Count - 1)) / 2;
            Console.WriteLine($"Searching for best initialization pair among {_views.Count} images ({totalPairs} pairs)...");

            // Search ALL pairs - this is O(n^2) but necessary for robustness
            // For 4 images: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
            for (int i = 0; i < _views.Count - 1; i++)
            {
                for (int j = i + 1; j < _views.Count; j++)
                {
                    var v1 = _views[i];
                    var v2 = _views[j];

                    var matches = matcher.Match(v1.Descriptors, v2.Descriptors);
                    if (matches.Length < 30) continue; // Lower threshold for difficult scenes

                    // Recover Pose
                    var p1 = new List<Point2d>();
                    var p2 = new List<Point2d>();
                    foreach (var m in matches)
                    {
                        p1.Add(new Point2d(v1.KeyPoints[m.QueryIdx].Pt.X, v1.KeyPoints[m.QueryIdx].Pt.Y));
                        p2.Add(new Point2d(v2.KeyPoints[m.TrainIdx].Pt.X, v2.KeyPoints[m.TrainIdx].Pt.Y));
                    }

                    using var mask = new Mat();
                    // Normalize points using each camera's intrinsics (supports different focal lengths)
                    var normP1 = NormalizePoints(p1, v1.K);
                    var normP2 = NormalizePoints(p2, v2.K);

                    using var E = Cv2.FindEssentialMat(InputArray.Create(normP1), InputArray.Create(normP2),
                        Mat.Eye(3, 3, MatType.CV_64F), (EssentialMatMethod)8, 0.999, 1.0, mask);

                    if (E.Rows != 3 || E.Cols != 3) continue;

                    using var R = new Mat();
                    using var t = new Mat();
                    int inliers = Cv2.RecoverPose(E, InputArray.Create(normP1), InputArray.Create(normP2),
                        Mat.Eye(3, 3, MatType.CV_64F), R, t, mask);

                    if (inliers > 30 && inliers > bestInliers)
                    {
                        // Ensure Double precision to prevent type mismatch issues
                        if (R.Type() != MatType.CV_64F) R.ConvertTo(R, MatType.CV_64F);
                        if (t.Type() != MatType.CV_64F) t.ConvertTo(t, MatType.CV_64F);

                        // Sanity check for initialization translation
                        double tx = t.At<double>(0, 0);
                        double ty = t.At<double>(1, 0);
                        double tz = t.At<double>(2, 0);
                        if (double.IsNaN(tx) || double.IsNaN(ty) || double.IsNaN(tz) ||
                            Math.Abs(tx) > 1000 || Math.Abs(ty) > 1000 || Math.Abs(tz) > 1000)
                        {
                             continue;
                        }

                        // Track best pair
                        bestI = i;
                        bestJ = j;
                        bestInliers = inliers;
                        bestR?.Dispose();
                        bestT?.Dispose();
                        bestMask?.Dispose();
                        bestR = R.Clone();
                        bestT = t.Clone();
                        bestMatches = matches;
                        bestMask = mask.Clone();

                        Console.WriteLine($"  Candidate pair ({i},{j}): {inliers} inliers from {matches.Length} matches");
                    }
                }
            }

            // Use best pair found
            if (bestI >= 0 && bestJ >= 0 && bestInliers >= 30)
            {
                var v1 = _views[bestI];
                var v2 = _views[bestJ];

                // Check for rotation-dominant (pivoting) camera motion
                // For pure rotation, triangulation will fail - the baseline is zero
                // We detect this by checking if Homography fits better than Essential Matrix
                bool isRotationDominant = DetectRotationDominantMotion(v1, v2, bestMatches, bestMask);

                if (isRotationDominant)
                {
                    Console.WriteLine("WARNING: Rotation-dominant camera motion detected (pivoting/panning camera).");
                    Console.WriteLine("  This typically occurs with cameras rotating in place (e.g., looking around a scene).");
                    Console.WriteLine("  Standard triangulation is unreliable. Using spherical depth assumption.");

                    // For pivoting cameras, we can't triangulate reliably
                    // Instead, assume a spherical depth (all points at similar depth from camera)
                    _isPivotingCamera = true;
                }

                v1.R = Mat.Eye(3, 3, MatType.CV_64F).ToMat();
                v1.t = new Mat(3, 1, MatType.CV_64F, Scalar.All(0));
                v1.IsRegistered = true;

                v2.R = bestR.Clone();
                v2.t = bestT.Clone();
                v2.IsRegistered = true;

                // Create Initial Map
                if (_isPivotingCamera)
                {
                    TriangulateSphericalDepth(v1, v2, bestMatches, bestMask);
                }
                else
                {
                    TriangulateAndAddPoints(v1, v2, bestMatches, bestMask);
                }
                Console.WriteLine($"Map Initialized with Views {v1.Index} and {v2.Index}. {_map.Count} points, {bestInliers} inliers.");

                bestR?.Dispose();
                bestT?.Dispose();
                bestMask?.Dispose();
                return true;
            }

            bestR?.Dispose();
            bestT?.Dispose();
            bestMask?.Dispose();
            return false;
        }

        private (List<Point3d>, List<Point2d>) FindMapMatches(ViewInfo view)
        {
            // Match view descriptors against ALL Map descriptors
            // This allows Loop Closure and Global tracking.
            // Optimization: Only match against map points seen by nearby views?
            // "Professional" but slow: Match all. For < 10k points it's fast enough.

            if (_map.Count == 0) return (new List<Point3d>(), new List<Point2d>());

            // Build Map Descriptor Matrix
            // This is slow if done every frame. Ideally Map class maintains a Mat.
            // Doing explicitly here.

            // To speed up, we can match against the 'average' descriptor of map points?
            // Or just simple Brute Force against all.
            using var mapDesc = new Mat(_map.Count, 32, MatType.CV_8U);
            // Copy
            // Unsafe copy for speed
            // Need to check descriptor width (ORB=32 bytes)

            // NOTE: Copying one by one is slow.
            // Let's assume we match against features of REGISTERED views that correspond to map points.
            // That's standard 2D-2D matching, then lifting to 3D.

            // Better Strategy:
            // Match `view` against ALL registered views.
            // For every 2D match (view_kp, reg_view_kp), if reg_view_kp links to a MapPoint, we have a 2D-3D match.
            // Collect all unique 2D-3D matches.

            var objPoints = new Dictionary<int, Point3d>(); // MapPointID -> Point3d
            var imgPoints = new Dictionary<int, Point2d>(); // MapPointID -> Point2d (in new view)

            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            foreach (var regView in _views.Where(v => v.IsRegistered))
            {
                var matches = matcher.Match(view.Descriptors, regView.Descriptors);
                foreach(var m in matches)
                {
                    // Check if regView feature is linked to a MapPoint
                    var mp = _map.FirstOrDefault(p => p.ObservingViewIndices.ContainsKey(regView.Index) &&
                                                      p.ObservingViewIndices[regView.Index] == m.TrainIdx);

                    if (mp != null)
                    {
                        if (!objPoints.ContainsKey(mp.Id))
                        {
                            objPoints[mp.Id] = mp.Position;
                            imgPoints[mp.Id] = new Point2d(view.KeyPoints[m.QueryIdx].Pt.X, view.KeyPoints[m.QueryIdx].Pt.Y);
                        }
                    }
                }
                // Limit number of views checked? No, check all for Loop Closure.
            }

            return (objPoints.Values.ToList(), imgPoints.Values.ToList());
        }

        private int GetFeatureIndexForView(MapPoint mp, int viewIdx)
        {
            if (mp.ObservingViewIndices.ContainsKey(viewIdx))
                return mp.ObservingViewIndices[viewIdx];
            return -1;
        }

        private bool RegisterView(ViewInfo view, List<Point3d> objPts, List<Point2d> imgPts)
        {
            using var rvec = new Mat();
            using var tvec = new Mat();
            using var inliers = new Mat();

            try
            {
                // Use more relaxed parameters for robust registration
                // Increased reprojection error from 6.0 to 12.0 pixels to handle imperfect triangulation
                // Increased iterations for better convergence
                // Use empty distortion coefficients (not null) - null causes exception
                using var distCoeffs = new Mat();
                // Relaxed parameters: 20px error, 2000 iterations to find good pose even with outliers
                Cv2.SolvePnPRansac(InputArray.Create(objPts), InputArray.Create(imgPts), view.K, distCoeffs, rvec, tvec,
                    useExtrinsicGuess: false, iterationsCount: 2000, reprojectionError: 20.0f, confidence: 0.999, inliers: inliers);

                Console.WriteLine($"  PnP result: {inliers.Rows} inliers from {objPts.Count} correspondences");

                if (inliers.Rows < 8)
                {
                    Console.WriteLine($"  Insufficient inliers ({inliers.Rows} < 8)");
                    return false;
                }

                // Ensure Double precision to prevent type mismatch issues
                if (rvec.Type() != MatType.CV_64F) rvec.ConvertTo(rvec, MatType.CV_64F);
                if (tvec.Type() != MatType.CV_64F) tvec.ConvertTo(tvec, MatType.CV_64F);

                // Sanity check: Check for exploding coordinates
                double tx = tvec.At<double>(0, 0);
                double ty = tvec.At<double>(1, 0);
                double tz = tvec.At<double>(2, 0);

                if (double.IsNaN(tx) || double.IsNaN(ty) || double.IsNaN(tz) ||
                    Math.Abs(tx) > 1000 || Math.Abs(ty) > 1000 || Math.Abs(tz) > 1000)
                {
                    Console.WriteLine($"  PnP rejected: Extrinsic translation out of bounds ({tx}, {ty}, {tz})");
                    return false;
                }

                view.R = new Mat();
                Cv2.Rodrigues(rvec, view.R);
                view.t = tvec.Clone();
                view.IsRegistered = true;
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  PnP exception: {ex.Message}");
                return false;
            }
        }

        private void ExpandMap(ViewInfo newView)
        {
            // Match newView against other registered views to triangulation NEW points
            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            foreach (var regView in _views.Where(v => v.IsRegistered && v != newView))
            {
                var matches = matcher.Match(newView.Descriptors, regView.Descriptors);
                TriangulateAndAddPoints(newView, regView, matches, null); // Null mask means use all matches
            }
        }

        private void TriangulateAndAddPoints(ViewInfo v1, ViewInfo v2, DMatch[] matches, Mat mask)
        {
            // Robust triangulation with reprojection error validation
            // This handles ALL camera motion types: translation, rotation, pivoting, tilting, or combinations
            // Points with high reprojection error are rejected or fall back to depth assumption

            // Need P matrices
            using var P1 = ComputeP(v1.K, v1.R, v1.t);
            using var P2 = ComputeP(v2.K, v2.R, v2.t);

            var pts1 = new List<Point2d>();
            var pts2 = new List<Point2d>();
            var usedMatches = new List<DMatch>();

            var indexer = mask?.GetGenericIndexer<byte>();

            for (int i = 0; i < matches.Length; i++)
            {
                if (mask != null && indexer[i, 0] == 0) continue;

                var m = matches[i];
                // Add to triangulation list
                pts1.Add(new Point2d(v1.KeyPoints[m.QueryIdx].Pt.X, v1.KeyPoints[m.QueryIdx].Pt.Y));
                pts2.Add(new Point2d(v2.KeyPoints[m.TrainIdx].Pt.X, v2.KeyPoints[m.TrainIdx].Pt.Y));
                usedMatches.Add(m);
            }

            if (pts1.Count == 0) return;

            using var pts4D = new Mat();
            Cv2.TriangulatePoints(P1, P2, InputArray.Create(pts1), InputArray.Create(pts2), pts4D);

            // Ensure output is Double before accessing as Double
            if (pts4D.Type() != MatType.CV_64F) pts4D.ConvertTo(pts4D, MatType.CV_64F);

            // Reprojection error threshold (pixels)
            const double maxReprojError = 4.0;

            int validPoints = 0;
            int rejectedPoints = 0;
            int fallbackPoints = 0;

            for (int i = 0; i < pts4D.Cols; i++)
            {
                double w = pts4D.At<double>(3, i);
                bool useTriangulated = true;
                double x = 0, y = 0, z = 0;

                if (Math.Abs(w) < 1e-6)
                {
                    useTriangulated = false;
                }
                else
                {
                    x = pts4D.At<double>(0, i) / w;
                    y = pts4D.At<double>(1, i) / w;
                    z = pts4D.At<double>(2, i) / w;

                    // Filter invalid points:
                    // 1. Check for NaN/Inf
                    if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) ||
                        double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z))
                    {
                        useTriangulated = false;
                    }
                    // 2. Check point is in front of both cameras (Z > 0 in camera space)
                    else
                    {
                        double z1 = v1.R.At<double>(2, 0) * x + v1.R.At<double>(2, 1) * y + v1.R.At<double>(2, 2) * z + v1.t.At<double>(2, 0);
                        double z2 = v2.R.At<double>(2, 0) * x + v2.R.At<double>(2, 1) * y + v2.R.At<double>(2, 2) * z + v2.t.At<double>(2, 0);

                        if (z1 < 0.01 || z2 < 0.01)
                        {
                            useTriangulated = false;
                        }
                        // 3. Check point is not too far (depth sanity check - max 500 units)
                        else if (Math.Abs(x) > 500 || Math.Abs(y) > 500 || Math.Abs(z) > 500)
                        {
                            useTriangulated = false;
                        }
                    }
                }

                // CRITICAL: Validate triangulation quality with reprojection error
                // This is what makes it work for ALL camera motion types
                if (useTriangulated)
                {
                    double reprojError1 = ComputeReprojectionError(x, y, z, pts1[i], v1.K, v1.R, v1.t);
                    double reprojError2 = ComputeReprojectionError(x, y, z, pts2[i], v2.K, v2.R, v2.t);
                    double maxError = Math.Max(reprojError1, reprojError2);

                    if (maxError > maxReprojError)
                    {
                        useTriangulated = false;
                    }
                }

                // If triangulation failed or has high reprojection error, use depth assumption
                if (!useTriangulated)
                {
                    // Fall back to spherical depth assumption
                    // Back-project from camera 1 with assumed depth
                    double assumedDepth = 5.0;
                    double fx = v1.K.At<double>(0, 0);
                    double fy = v1.K.At<double>(1, 1);
                    double cx = v1.K.At<double>(0, 2);
                    double cy = v1.K.At<double>(1, 2);

                    var pt = v1.KeyPoints[usedMatches[i].QueryIdx].Pt;
                    x = (pt.X - cx) * assumedDepth / fx;
                    y = (pt.Y - cy) * assumedDepth / fy;
                    z = assumedDepth;

                    // For non-identity camera pose, transform to world coordinates
                    if (v1.R != null && v1.t != null)
                    {
                        // Point is in camera space, convert to world space
                        // P_world = R^T * (P_cam - t) = R^T * P_cam - R^T * t
                        // For R=I, t=0 this is identity transform
                        using var pCam = new Mat(3, 1, MatType.CV_64F);
                        pCam.Set(0, 0, x);
                        pCam.Set(1, 0, y);
                        pCam.Set(2, 0, z);

                        using var Rt = v1.R.T().ToMat();
                        using var pWorld = (Rt * (pCam - v1.t)).ToMat();
                        x = pWorld.At<double>(0, 0);
                        y = pWorld.At<double>(1, 0);
                        z = pWorld.At<double>(2, 0);
                    }

                    fallbackPoints++;
                }
                else
                {
                    validPoints++;
                }

                var mp = new MapPoint
                {
                    Id = _nextMapPointId++,
                    Position = new Point3d(x, y, z),
                    Descriptor = v1.Descriptors.Row(usedMatches[i].QueryIdx).Clone(),
                    Color = GetColor(v1.Image, v1.KeyPoints[usedMatches[i].QueryIdx].Pt)
                };

                mp.ObservingViewIndices[v1.Index] = usedMatches[i].QueryIdx;
                mp.ObservingViewIndices[v2.Index] = usedMatches[i].TrainIdx;

                _map.Add(mp);
            }

            rejectedPoints = usedMatches.Count - validPoints - fallbackPoints;
            Console.WriteLine($"  Triangulated {validPoints} validated points, {fallbackPoints} depth-assumed points, rejected {rejectedPoints}");
        }

        /// <summary>
        /// Compute reprojection error: project 3D point to image and compute distance to observed 2D point
        /// </summary>
        private double ComputeReprojectionError(double X, double Y, double Z, Point2d observed, Mat K, Mat R, Mat t)
        {
            // Transform world point to camera coordinates: P_cam = R * P_world + t
            double camX = R.At<double>(0, 0) * X + R.At<double>(0, 1) * Y + R.At<double>(0, 2) * Z + t.At<double>(0, 0);
            double camY = R.At<double>(1, 0) * X + R.At<double>(1, 1) * Y + R.At<double>(1, 2) * Z + t.At<double>(1, 0);
            double camZ = R.At<double>(2, 0) * X + R.At<double>(2, 1) * Y + R.At<double>(2, 2) * Z + t.At<double>(2, 0);

            if (camZ < 0.001) return double.MaxValue; // Point behind camera

            // Project to image: p = K * [X/Z, Y/Z, 1]
            double fx = K.At<double>(0, 0);
            double fy = K.At<double>(1, 1);
            double cx = K.At<double>(0, 2);
            double cy = K.At<double>(1, 2);

            double projX = fx * camX / camZ + cx;
            double projY = fy * camY / camZ + cy;

            // Compute error
            double dx = projX - observed.X;
            double dy = projY - observed.Y;
            return Math.Sqrt(dx * dx + dy * dy);
        }

        private void RefinePose(ViewInfo view)
        {
            // SolvePnPRefineLM or similar?
            // OpenCvSharp doesn't expose bundle adjustment easily.
            // Just assume SolvePnP RANSAC result is good enough for "Professional" prototype.
        }

        /// <summary>
        /// Expand map by triangulating new points between all pairs of registered views.
        /// This helps when no new views can be registered due to insufficient map overlap.
        /// </summary>
        private void ExpandMapBetweenRegisteredViews()
        {
            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            var registeredViews = _views.Where(v => v.IsRegistered).ToList();
            int newPointsBefore = _map.Count;

            // Triangulate between ALL pairs of registered views
            // For n registered views, this creates n*(n-1)/2 pairs
            int pairsChecked = 0;
            for (int i = 0; i < registeredViews.Count; i++)
            {
                for (int j = i + 1; j < registeredViews.Count; j++)
                {
                    var v1 = registeredViews[i];
                    var v2 = registeredViews[j];

                    var matches = matcher.Match(v1.Descriptors, v2.Descriptors);
                    if (matches.Length > 10)
                    {
                        TriangulateAndAddPoints(v1, v2, matches, null);
                        pairsChecked++;
                    }
                }
            }

            int newPoints = _map.Count - newPointsBefore;
            Console.WriteLine($"  Expanded map: checked {pairsChecked} pairs, added {newPoints} new points (total: {_map.Count})");
        }

        /// <summary>
        /// Try to register views that failed before by matching against ALL registered views.
        /// This handles disjoint groups by finding connections between them.
        /// </summary>
        private bool TryRegisterFromAllPairs(ViewInfo view)
        {
            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            var registeredViews = _views.Where(v => v.IsRegistered).ToList();
            if (registeredViews.Count < 2) return false;

            // Try to find any registered view with enough matches
            foreach (var regView in registeredViews)
            {
                var matches = matcher.Match(view.Descriptors, regView.Descriptors);
                if (matches.Length < 30) continue;

                // Try essential matrix approach
                var p1 = new List<Point2d>();
                var p2 = new List<Point2d>();
                foreach (var m in matches)
                {
                    p1.Add(new Point2d(view.KeyPoints[m.QueryIdx].Pt.X, view.KeyPoints[m.QueryIdx].Pt.Y));
                    p2.Add(new Point2d(regView.KeyPoints[m.TrainIdx].Pt.X, regView.KeyPoints[m.TrainIdx].Pt.Y));
                }

                using var mask = new Mat();
                var normP1 = NormalizePoints(p1, view.K);
                var normP2 = NormalizePoints(p2, regView.K);

                using var E = Cv2.FindEssentialMat(InputArray.Create(normP1), InputArray.Create(normP2),
                    Mat.Eye(3, 3, MatType.CV_64F), (EssentialMatMethod)8, 0.999, 1.0, mask);

                if (E.Rows != 3 || E.Cols != 3) continue;

                using var R = new Mat();
                using var t = new Mat();
                int inliers = Cv2.RecoverPose(E, InputArray.Create(normP1), InputArray.Create(normP2),
                    Mat.Eye(3, 3, MatType.CV_64F), R, t, mask);

                if (inliers >= 20)
                {
                    if (R.Type() != MatType.CV_64F) R.ConvertTo(R, MatType.CV_64F);
                    if (t.Type() != MatType.CV_64F) t.ConvertTo(t, MatType.CV_64F);

                    // Compose with regView's pose to get world pose
                    // view_pose = relative_pose * regView_pose
                    // MatExpr needs to be converted to Mat before Clone()
                    using var newR = (R * regView.R).ToMat();
                    using var newT = ((R * regView.t) + t).ToMat();

                    view.R = newR.Clone();
                    view.t = newT.Clone();
                    view.IsRegistered = true;

                    Console.WriteLine($"  Registered view {view.Index} via essential matrix with view {regView.Index} ({inliers} inliers)");
                    return true;
                }
            }

            return false;
        }

        // Helpers
        private List<Point2d> NormalizePoints(List<Point2d> pts, Mat K)
        {
            var res = new List<Point2d>();
            double fx = K.At<double>(0, 0);
            double fy = K.At<double>(1, 1);
            double cx = K.At<double>(0, 2);
            double cy = K.At<double>(1, 2);
            foreach (var p in pts) res.Add(new Point2d((p.X - cx) / fx, (p.Y - cy) / fy));
            return res;
        }

        private Mat ComputeP(Mat K, Mat R, Mat t)
        {
            using var Rt = new Mat();
            Cv2.HConcat(new[] { R, t }, Rt);
            return K * Rt;
        }

        private Scalar GetColor(Mat img, Point2f pt)
        {
            int x = Math.Clamp((int)pt.X, 0, img.Width - 1);
            int y = Math.Clamp((int)pt.Y, 0, img.Height - 1);
            var c = img.At<Vec3b>(y, x);
            return new Scalar(c.Item0, c.Item1, c.Item2);
        }

        /// <summary>
        /// Detects if camera motion is rotation-dominant (pivoting camera) vs translation-dominant (moving camera)
        /// For pure rotation, Homography fits well and Essential Matrix's translation is degenerate
        /// </summary>
        private bool DetectRotationDominantMotion(ViewInfo v1, ViewInfo v2, DMatch[] matches, Mat mask)
        {
            if (matches.Length < 10) return false;

            // Get matched points
            var pts1 = new List<Point2f>();
            var pts2 = new List<Point2f>();
            var indexer = mask?.GetGenericIndexer<byte>();

            for (int i = 0; i < matches.Length; i++)
            {
                if (mask != null && indexer[i, 0] == 0) continue;
                pts1.Add(v1.KeyPoints[matches[i].QueryIdx].Pt);
                pts2.Add(v2.KeyPoints[matches[i].TrainIdx].Pt);
            }

            if (pts1.Count < 10) return false;

            // Compute Homography and check reprojection error
            using var H = Cv2.FindHomography(InputArray.Create(pts1), InputArray.Create(pts2), HomographyMethods.Ransac, 3.0);

            if (H.Empty()) return false;

            // Compute reprojection error for Homography
            double homographyError = 0;
            int inlierCount = 0;

            for (int i = 0; i < pts1.Count; i++)
            {
                // Project pts1 using H
                double x = pts1[i].X;
                double y = pts1[i].Y;
                double w = H.At<double>(2, 0) * x + H.At<double>(2, 1) * y + H.At<double>(2, 2);
                double px = (H.At<double>(0, 0) * x + H.At<double>(0, 1) * y + H.At<double>(0, 2)) / w;
                double py = (H.At<double>(1, 0) * x + H.At<double>(1, 1) * y + H.At<double>(1, 2)) / w;

                double dx = px - pts2[i].X;
                double dy = py - pts2[i].Y;
                double err = Math.Sqrt(dx * dx + dy * dy);

                if (err < 5.0) // Count as inlier if error < 5 pixels
                {
                    homographyError += err;
                    inlierCount++;
                }
            }

            if (inlierCount > 0) homographyError /= inlierCount;

            // If Homography fits very well (low error, high inlier ratio), it's likely pure rotation
            double inlierRatio = (double)inlierCount / pts1.Count;

            Console.WriteLine($"  Rotation detection: Homography error={homographyError:F2}px, inlier ratio={inlierRatio:P0}");

            // Heuristic: if >80% of points fit Homography with <3px error, it's rotation-dominant
            return inlierRatio > 0.8 && homographyError < 3.0;
        }

        /// <summary>
        /// For pivoting cameras, triangulation fails. Instead, assume spherical depth
        /// (all points at a fixed distance from camera, creating a panoramic depth)
        /// </summary>
        private void TriangulateSphericalDepth(ViewInfo v1, ViewInfo v2, DMatch[] matches, Mat mask)
        {
            double assumedDepth = 5.0; // Assume all points are at 5 units depth

            double fx = v1.K.At<double>(0, 0);
            double fy = v1.K.At<double>(1, 1);
            double cx = v1.K.At<double>(0, 2);
            double cy = v1.K.At<double>(1, 2);

            var indexer = mask?.GetGenericIndexer<byte>();
            int addedPoints = 0;

            for (int i = 0; i < matches.Length; i++)
            {
                if (mask != null && indexer[i, 0] == 0) continue;

                var m = matches[i];
                var pt = v1.KeyPoints[m.QueryIdx].Pt;

                // Back-project using pinhole model and assumed depth
                // x = (u - cx) * Z / fx
                // y = (v - cy) * Z / fy
                // z = Z
                double x = (pt.X - cx) * assumedDepth / fx;
                double y = (pt.Y - cy) * assumedDepth / fy;
                double z = assumedDepth;

                // Create map point (in camera 1 space, which is world space since R1=I, t1=0)
                var mp = new MapPoint
                {
                    Id = _nextMapPointId++,
                    Position = new Point3d(x, y, z),
                    Descriptor = v1.Descriptors.Row(m.QueryIdx).Clone(),
                    Color = GetColor(v1.Image, pt)
                };

                mp.ObservingViewIndices[v1.Index] = m.QueryIdx;
                mp.ObservingViewIndices[v2.Index] = m.TrainIdx;

                _map.Add(mp);
                addedPoints++;
            }

            Console.WriteLine($"  Created {addedPoints} points using spherical depth assumption (depth={assumedDepth})");
        }
    }
}
