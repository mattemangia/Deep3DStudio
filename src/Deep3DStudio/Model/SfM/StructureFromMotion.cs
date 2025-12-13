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
                // Apply M to Point: (x, -y, -z)
                var p = new Vector3((float)mp.Position.X, -(float)mp.Position.Y, -(float)mp.Position.Z);
                mesh.Vertices.Add(p);
                mesh.Colors.Add(new Vector3((float)mp.Color.Val2 / 255f, (float)mp.Color.Val1 / 255f, (float)mp.Color.Val0 / 255f)); // RGB vs BGR

                minBound = Vector3.ComponentMin(minBound, p);
                maxBound = Vector3.ComponentMax(maxBound, p);
            }

            Console.WriteLine($"SfM Output: {mesh.Vertices.Count} points. Bounds: {minBound} to {maxBound}");

            foreach (var v in _views)
            {
                if (v.IsRegistered)
                {
                    // Convert OpenCV Pose (R, t) to OpenGL Pose (R', t')
                    // R' = M * R * M
                    // t' = M * t

                    var R_cv = v.R;
                    var t_cv = v.t;

                    using var R_gl = new Mat(3, 3, MatType.CV_64F);
                    using var t_gl = new Mat(3, 1, MatType.CV_64F);

                    // R_gl = M * R_cv * M
                    // Row 0: R00, -R01, -R02
                    R_gl.Set(0, 0, R_cv.At<double>(0, 0));
                    R_gl.Set(0, 1, -R_cv.At<double>(0, 1));
                    R_gl.Set(0, 2, -R_cv.At<double>(0, 2));

                    // Row 1: -R10, R11, R12
                    R_gl.Set(1, 0, -R_cv.At<double>(1, 0));
                    R_gl.Set(1, 1, R_cv.At<double>(1, 1));
                    R_gl.Set(1, 2, R_cv.At<double>(1, 2));

                    // Row 2: -R20, R21, R22
                    R_gl.Set(2, 0, -R_cv.At<double>(2, 0));
                    R_gl.Set(2, 1, R_cv.At<double>(2, 1));
                    R_gl.Set(2, 2, R_cv.At<double>(2, 2));

                    // t_gl = M * t_cv -> (tx, -ty, -tz)
                    t_gl.Set(0, 0, t_cv.At<double>(0, 0));
                    t_gl.Set(1, 0, -t_cv.At<double>(1, 0));
                    t_gl.Set(2, 0, -t_cv.At<double>(2, 0));

                    var camToWorld = CvPoseToOpenTK(R_gl, t_gl); // Returns M_gl.Inverted() (Camera -> World)

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
                var img = Cv2.ImRead(paths[i], ImreadModes.Color);
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

                v1.R = Mat.Eye(3, 3, MatType.CV_64F).ToMat();
                v1.t = new Mat(3, 1, MatType.CV_64F, Scalar.All(0));
                v1.IsRegistered = true;

                v2.R = bestR.Clone();
                v2.t = bestT.Clone();
                v2.IsRegistered = true;

                // Create Initial Map
                TriangulateAndAddPoints(v1, v2, bestMatches, bestMask);
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
            // Filter matches that are NOT already in map
            // If v1_feat or v2_feat is already in map, we merge?
            // Ideally yes. For simplicity, we only add if NEITHER is in map.

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
                // Check map existence
                // We need the Observation Dictionary in MapPoint.
                // Assuming we implemented that change.

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

            int validPoints = 0;
            int rejectedPoints = 0;

            for (int i = 0; i < pts4D.Cols; i++)
            {
                double w = pts4D.At<double>(3, i);
                if (Math.Abs(w) < 1e-6)
                {
                    rejectedPoints++;
                    continue;
                }

                double x = pts4D.At<double>(0, i) / w;
                double y = pts4D.At<double>(1, i) / w;
                double z = pts4D.At<double>(2, i) / w;

                // Filter invalid points:
                // 1. Check for NaN/Inf
                if (double.IsNaN(x) || double.IsNaN(y) || double.IsNaN(z) ||
                    double.IsInfinity(x) || double.IsInfinity(y) || double.IsInfinity(z))
                {
                    rejectedPoints++;
                    continue;
                }

                // 2. Check point is in front of both cameras (Z > 0 in camera space)
                // Transform point to camera 1 space
                double z1 = v1.R.At<double>(2, 0) * x + v1.R.At<double>(2, 1) * y + v1.R.At<double>(2, 2) * z + v1.t.At<double>(2, 0);
                double z2 = v2.R.At<double>(2, 0) * x + v2.R.At<double>(2, 1) * y + v2.R.At<double>(2, 2) * z + v2.t.At<double>(2, 0);

                if (z1 < 0.01 || z2 < 0.01)
                {
                    rejectedPoints++;
                    continue; // Point is behind one of the cameras
                }

                // 3. Check point is not too far (depth sanity check - max 1000 units)
                if (Math.Abs(x) > 1000 || Math.Abs(y) > 1000 || Math.Abs(z) > 1000)
                {
                    rejectedPoints++;
                    continue;
                }

                var mp = new MapPoint
                {
                    Id = _nextMapPointId++,
                    Position = new Point3d(x, y, z),
                    // Use v1 descriptor
                    Descriptor = v1.Descriptors.Row(usedMatches[i].QueryIdx).Clone(),
                    // Color from v1
                    Color = GetColor(v1.Image, v1.KeyPoints[usedMatches[i].QueryIdx].Pt)
                };

                // Add observations
                mp.ObservingViewIndices[v1.Index] = usedMatches[i].QueryIdx;
                mp.ObservingViewIndices[v2.Index] = usedMatches[i].TrainIdx;

                _map.Add(mp);
                validPoints++;
            }

            Console.WriteLine($"  Triangulated {validPoints} valid points, rejected {rejectedPoints} bad points");
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

        private Matrix4 CvPoseToOpenTK(Mat R, Mat t)
        {
            var M = Matrix4.Identity;
            M.M11 = (float)R.At<double>(0, 0); M.M12 = (float)R.At<double>(1, 0); M.M13 = (float)R.At<double>(2, 0);
            M.M21 = (float)R.At<double>(0, 1); M.M22 = (float)R.At<double>(1, 1); M.M23 = (float)R.At<double>(2, 1);
            M.M31 = (float)R.At<double>(0, 2); M.M32 = (float)R.At<double>(1, 2); M.M33 = (float)R.At<double>(2, 2);
            M.M41 = (float)t.At<double>(0, 0); M.M42 = (float)t.At<double>(1, 0); M.M43 = (float)t.At<double>(2, 0);
            return M.Inverted();
        }
    }
}
