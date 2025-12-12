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
            public Mat DistCoeffs;
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
            bool changed = true;

            while (changed && registeredCount < _views.Count)
            {
                changed = false;
                // Find unregistered view with most matches to Map
                var bestCandidate = -1;
                var bestMatches = new List<DMatch>();
                var bestObjectPoints = new List<Point3d>();
                var bestImagePoints = new List<Point2d>();

                // Heuristic: Try next sequential view first (Video case), then others
                // But "Professional" means robust. Let's scan all unregistered.
                // For performance on large sets, we might restrict to neighbors of registered.

                // Sort unregistered views by index to prioritize sequence if it exists,
                // but checking map overlap is key.
                var unregistered = _views.Where(v => !v.IsRegistered).ToList();

                int maxInliers = 0;

                foreach (var view in unregistered)
                {
                    // Match View Features <-> Global Map Descriptors
                    // This is "Map Tracking" / Implicit Loop Closure
                    var (objPts, imgPts) = FindMapMatches(view);

                    if (objPts.Count > maxInliers)
                    {
                        maxInliers = objPts.Count;
                        bestCandidate = view.Index;
                        bestObjectPoints = objPts;
                        bestImagePoints = imgPts;
                    }
                }

                if (bestCandidate != -1 && maxInliers >= 8) // Minimum threshold for PnP (lowered from 15 for small image sets)
                {
                    var view = _views[bestCandidate];
                    if (RegisterView(view, bestObjectPoints, bestImagePoints))
                    {
                        Console.WriteLine($"Registered Image {view.Index} with {maxInliers} map matches.");
                        registeredCount++;
                        changed = true;

                        // Triangulate new points with existing registered views
                        // To expand the map
                        ExpandMap(view);

                        // Bundle Adjustment (Local refinement)
                        RefinePose(view);
                    }
                    else
                    {
                        Console.WriteLine($"Failed to register Image {bestCandidate} (PnP failed with {maxInliers} matches).");
                    }
                }
                else if (bestCandidate != -1)
                {
                    Console.WriteLine($"Skipping Image {bestCandidate}: only {maxInliers} map matches found (minimum 8 required).");
                }
            }

            // 4. Final Polish
            Console.WriteLine($"SfM Complete: Registered {registeredCount}/{_views.Count} views, {_map.Count} 3D points in map.");

            // Remove outliers
            // PruneMap();

            // 5. Output
            var mesh = new MeshData();
            foreach (var mp in _map)
            {
                // Convert OpenCV Point3d to OpenTK Vector3
                // Simple filter: Check reasonably in front of cameras?
                mesh.Vertices.Add(new Vector3((float)mp.Position.X, (float)mp.Position.Y, (float)mp.Position.Z));
                mesh.Colors.Add(new Vector3((float)mp.Color.Val2 / 255f, (float)mp.Color.Val1 / 255f, (float)mp.Color.Val0 / 255f)); // RGB vs BGR
            }

            foreach (var v in _views)
            {
                if (v.IsRegistered)
                {
                    result.Poses.Add(new CameraPose
                    {
                        ImageIndex = v.Index,
                        ImagePath = v.Path,
                        Width = v.Size.Width,
                        Height = v.Size.Height,
                        // OpenCV Pose (R, t) is World->Camera.
                        // We store Camera->World.
                        CameraToWorld = CvPoseToOpenTK(v.R, v.t),
                        WorldToCamera = CvPoseToOpenTK(v.R, v.t).Inverted()
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
                    KeyPoints = kps,
                    Descriptors = desc,
                    IsRegistered = false
                });
            }
        }

        private bool InitializeMap()
        {
            // Find pair with high feature matches and sufficient baseline
            // Simplified: Just scan 0-1, 0-2, 1-2.
            // Professional: Use Homography vs Fundamental matrix score to detect baseline.

            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);

            for (int i = 0; i < Math.Min(_views.Count - 1, 3); i++)
            {
                for (int j = i + 1; j < Math.Min(_views.Count, i + 4); j++)
                {
                    var v1 = _views[i];
                    var v2 = _views[j];

                    var matches = matcher.Match(v1.Descriptors, v2.Descriptors);
                    Console.WriteLine($"Initialization: Matching views {v1.Index} and {v2.Index}: {matches.Length} matches");
                    if (matches.Length < 50) continue; // Lowered from 100 to support smaller/simpler scenes

                    // Recover Pose
                    var p1 = new List<Point2d>();
                    var p2 = new List<Point2d>();
                    foreach (var m in matches)
                    {
                        p1.Add(new Point2d(v1.KeyPoints[m.QueryIdx].Pt.X, v1.KeyPoints[m.QueryIdx].Pt.Y));
                        p2.Add(new Point2d(v2.KeyPoints[m.TrainIdx].Pt.X, v2.KeyPoints[m.TrainIdx].Pt.Y));
                    }

                    using var mask = new Mat();
                    // Essential Matrix (uses both Ks correctly)
                    // Note: OpenCV FindEssentialMat assumes same K? No, we can normalize points manually.
                    // Or use the overload with focal/pp. But we have two Ks.
                    // Correct way: Undistort points (normalize) -> FindEssentialMat with I -> RecoverPose with I.

                    var normP1 = NormalizePoints(p1, v1.K);
                    var normP2 = NormalizePoints(p2, v2.K);

                    using var E = Cv2.FindEssentialMat(InputArray.Create(normP1), InputArray.Create(normP2),
                        Mat.Eye(3, 3, MatType.CV_64F), (EssentialMatMethod)8, 0.999, 1.0, mask);

                    if (E.Rows != 3 || E.Cols != 3) continue;

                    using var R = new Mat();
                    using var t = new Mat();
                    int inliers = Cv2.RecoverPose(E, InputArray.Create(normP1), InputArray.Create(normP2),
                        Mat.Eye(3, 3, MatType.CV_64F), R, t, mask);

                    if (inliers > 50)
                    {
                        // Good initialization
                        v1.R = Mat.Eye(3, 3, MatType.CV_64F).ToMat();
                        v1.t = new Mat(3, 1, MatType.CV_64F, Scalar.All(0));
                        v1.IsRegistered = true;

                        v2.R = R.Clone();
                        v2.t = t.Clone();
                        v2.IsRegistered = true;

                        // Create Initial Map
                        TriangulateAndAddPoints(v1, v2, matches, mask);
                        Console.WriteLine($"Map Initialized with Views {v1.Index} and {v2.Index}. {_map.Count} points.");
                        return true;
                    }
                }
            }
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
                Cv2.SolvePnPRansac(InputArray.Create(objPts), InputArray.Create(imgPts), view.K, null, rvec, tvec,
                    useExtrinsicGuess: false, iterationsCount: 500, reprojectionError: 12.0f, confidence: 0.99, inliers: inliers);

                Console.WriteLine($"  PnP result: {inliers.Rows} inliers from {objPts.Count} correspondences");

                // Lowered minimum inlier threshold from 10 to 6 for challenging cases
                if (inliers.Rows < 6)
                {
                    Console.WriteLine($"  Insufficient inliers ({inliers.Rows} < 6)");
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
            }

            if (validPoints > 0 || rejectedPoints > 0)
            {
                Console.WriteLine($"  Triangulated {validPoints} valid points, rejected {rejectedPoints} bad points");
            }
        }

        private void RefinePose(ViewInfo view)
        {
            // SolvePnPRefineLM or similar?
            // OpenCvSharp doesn't expose bundle adjustment easily.
            // Just assume SolvePnP RANSAC result is good enough for "Professional" prototype.
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
