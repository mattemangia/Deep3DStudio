using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using OpenCvSharp;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;

namespace Deep3DStudio.Model.SfM
{
    public class SfMInference
    {
        // -----------------------------------------------------------------------
        // Internal Data Structures matching the C++ logic
        // -----------------------------------------------------------------------

        private class Point3D
        {
            public Point3d Pt;
            public Dictionary<int, int> IdxImage = new Dictionary<int, int>(); // ViewIndex -> FeatureIndex
            public Scalar Color;
        }

        private class ViewInfo
        {
            public int Index;
            public string Path;
            public Mat Image; // Color (BGR)
            public Mat Gray;  // Grayscale for feature detection
            public Size Size;
            public KeyPoint[] KeyPoints;
            public Mat Descriptors;
            public List<Point2d> Points2D; // Converted from KeyPoints
            public Mat P; // 3x4 Projection Matrix (World -> Camera)
            public Mat R; // 3x3 Rotation
            public Mat t; // 3x1 Translation
            public bool IsRegistered;
        }

        // -----------------------------------------------------------------------
        // Fields
        // -----------------------------------------------------------------------

        private List<ViewInfo> _views = new List<ViewInfo>();
        private List<Point3D> _reconstructionCloud = new List<Point3D>();
        // Optimization: Map (ViewIndex, FeatureIndex) to Point3D to avoid linear search
        private Dictionary<(int, int), Point3D> _featureToPointMap = new Dictionary<(int, int), Point3D>();

        private HashSet<int> _doneViews = new HashSet<int>();
        private HashSet<int> _goodViews = new HashSet<int>();

        private Mat _K; // Shared K
        private Mat _distCoef = new Mat(1, 5, MatType.CV_64F, Scalar.All(0));

        private const float NN_MATCH_RATIO = 0.8f;

        // -----------------------------------------------------------------------
        // Public Interface
        // -----------------------------------------------------------------------

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            var result = new SceneResult();
            if (imagePaths.Count < 2) return result;

            Console.WriteLine("************************************************");
            Console.WriteLine("              3D MAPPING (C# Port)              ");
            Console.WriteLine("************************************************");

            // Apply Computation Device Settings
            ConfigureOpenCV();

            _views.Clear();
            _reconstructionCloud.Clear();
            _featureToPointMap.Clear();
            _doneViews.Clear();
            _goodViews.Clear();

            // 1. Load Images & Extract Features
            if (!ImagesLoad(imagePaths)) return result;
            ExtractFeatures();

            // 2. Base Reconstruction
            if (!BaseReconstruction())
            {
                Console.Error.WriteLine("Could not find a good pair for initial reconstruction");
                return result;
            }

            // 3. Add More Views
            if (!AddMoreViews())
            {
                Console.Error.WriteLine("Could not add more views");
            }

            // 4. Final Global Bundle Adjustment
            PerformBundleAdjustment();

            // 5. Convert to SceneResult
            return ConvertToSceneResult();
        }

        private void ConfigureOpenCV()
        {
            var settings = IniSettings.Instance;
            if (settings.Device == ComputeDevice.GPU)
            {
                try
                {
                    // Attempt to enable OpenCL
                    Cv2.SetUseOptimized(true);
                // Cv2.Ocl class was removed/moved in newer OpenCVSharp or usage is different.
                // Assuming default optimization is sufficient or Ocl is not directly accessible.
                // Just use SetUseOptimized.
                Console.WriteLine("SfM: Optimization enabled.");
                }
                catch (Exception ex)
                {
                Console.WriteLine($"SfM: Failed to enable optimizations: {ex.Message}.");
                }
            }
            else
            {
                // Force CPU
                Cv2.SetUseOptimized(true);
            // Cv2.Ocl.SetUseOpenCL(false); // Removed
                Console.WriteLine("SfM: CPU mode enforced.");
            }
        }

        // -----------------------------------------------------------------------
        // Core Pipeline Steps
        // -----------------------------------------------------------------------

        private bool ImagesLoad(List<string> paths)
        {
            Console.WriteLine("Getting images...");
            _views.Clear();

            for (int i = 0; i < paths.Count; i++)
            {
                // Use ImageDecoder if possible or fallback to standard Cv2 load
                // The C++ code resizes large images.
                Mat image = Cv2.ImRead(paths[i], ImreadModes.Color);
                if (image.Empty()) continue;

                if (image.Rows > 1200 || image.Cols > 1200)
                {
                    double scale = 1200.0 / Math.Max(image.Rows, image.Cols);
                    Cv2.Resize(image, image, new Size(), scale, scale);
                }

                Mat gray = new Mat();
                Cv2.CvtColor(image, gray, ColorConversionCodes.BGR2GRAY);

                _views.Add(new ViewInfo
                {
                    Index = i,
                    Path = paths[i],
                    Image = image,
                    Gray = gray,
                    Size = image.Size(),
                    IsRegistered = false,
                    P = Mat.Eye(3, 4, MatType.CV_64F)
                });
            }

            if (_views.Count < 2)
            {
                Console.Error.WriteLine("Not enough images.");
                return false;
            }

            Console.WriteLine($"Loaded {_views.Count} images.");

            // Initialize Intrinsics (Approximation)
            var img = _views[0].Image;
            double f = Math.Max(img.Width, img.Height) * 0.85;
            double cx = img.Width / 2.0;
            double cy = img.Height / 2.0;

            _K = Mat.Eye(3, 3, MatType.CV_64F);
            _K.Set(0, 0, f);
            _K.Set(1, 1, f);
            _K.Set(0, 2, cx);
            _K.Set(1, 2, cy);

            Console.WriteLine($"Estimated K:\n{_K.Dump()}");

            return true;
        }

        private void ExtractFeatures()
        {
            Console.WriteLine("Extracting features from all images (ORB)...");
            using var detector = ORB.Create(nFeatures: 5000);

            foreach (var view in _views)
            {
                KeyPoint[] kps;
                Mat desc = new Mat();
                detector.DetectAndCompute(view.Gray, null, out kps, desc);

                view.KeyPoints = kps;
                view.Descriptors = desc;
                view.Points2D = kps.Select(k => new Point2d(k.Pt.X, k.Pt.Y)).ToList();

                Console.WriteLine($"Image {view.Index} --> {kps.Length} kps");
            }
        }

        private bool BaseReconstruction()
        {
            var bestViews = FindBestPair();
            if (bestViews.Count == 0)
            {
                Console.WriteLine("Could not obtain a good pair for baseline reconstruction.");
                return false;
            }

            var bestPairEntry = bestViews.Last();
            var score = bestPairEntry.Key;
            var pair = bestPairEntry.Value;
            int queryIdx = pair.Key;
            int trainIdx = pair.Value;

            Console.WriteLine($"Best pair: [{queryIdx}, {trainIdx}] Score: {score}");

            var matches = GetMatching(queryIdx, trainIdx);

            Mat Pleft = Mat.Eye(3, 4, MatType.CV_64F);
            Mat Pright = new Mat();

            Console.WriteLine("Estimating camera pose with Essential Matrix...");
            if (!GetCameraPose(_K, queryIdx, trainIdx, matches, _views[queryIdx].Points2D, _views[trainIdx].Points2D, ref Pleft, ref Pright))
            {
                Console.Error.WriteLine("Failed to get camera pose.");
                return false;
            }

            List<Point3D> pointcloud = new List<Point3D>();
            if (!TriangulateViews(_views[queryIdx].Points2D, _views[trainIdx].Points2D, Pleft, Pright, matches, _K, new KeyValuePair<int, int>(queryIdx, trainIdx), out pointcloud))
            {
                Console.Error.WriteLine("Could not triangulate initial pair.");
                return false;
            }

            _reconstructionCloud = pointcloud;
            _views[queryIdx].P = Pleft;
            _views[trainIdx].P = Pright;

            DecomposeP(Pleft, out _views[queryIdx].R, out _views[queryIdx].t);
            DecomposeP(Pright, out _views[trainIdx].R, out _views[trainIdx].t);

            _views[queryIdx].IsRegistered = true;
            _views[trainIdx].IsRegistered = true;

            _doneViews.Add(queryIdx);
            _doneViews.Add(trainIdx);
            _goodViews.Add(queryIdx);
            _goodViews.Add(trainIdx);

            PerformBundleAdjustment();

            return true;
        }

        private bool AddMoreViews()
        {
            while (_doneViews.Count != _views.Count)
            {
                HashSet<int> potentialViews = new HashSet<int>();
                foreach (int doneView in _doneViews)
                {
                    if (doneView > 0 && !_doneViews.Contains(doneView - 1)) potentialViews.Add(doneView - 1);
                    if (doneView < _views.Count - 1 && !_doneViews.Contains(doneView + 1)) potentialViews.Add(doneView + 1);
                }

                if (potentialViews.Count == 0)
                {
                    for(int i=0; i<_views.Count; i++) if(!_doneViews.Contains(i)) potentialViews.Add(i);
                }

                if (potentialViews.Count == 0) break;

                bool anyAdded = false;

                foreach (int newViewIdx in potentialViews)
                {
                    if (_doneViews.Contains(newViewIdx)) continue;

                    Console.WriteLine("\n====================================");
                    Console.WriteLine($"Processing view {newViewIdx}...");

                    List<Point3d> points3D = new List<Point3d>();
                    List<Point2d> points2D = new List<Point2d>();
                    DMatch[] bestMatches;
                    int doneViewRef;

                    Find2D3DMatches(newViewIdx, out points3D, out points2D, out bestMatches, out doneViewRef);

                    Console.WriteLine($"Found {points3D.Count} 2D-3D correspondences.");

                    Mat Pnew = new Mat();
                    if (!FindCameraPosePNP(_K, points3D, points2D, out Pnew))
                    {
                        Console.WriteLine("PNP failed or not enough inliers. Skipping view.");
                        continue;
                    }

                    _views[newViewIdx].P = Pnew;
                    DecomposeP(Pnew, out _views[newViewIdx].R, out _views[newViewIdx].t);
                    _views[newViewIdx].IsRegistered = true;
                    _doneViews.Add(newViewIdx);
                    anyAdded = true;

                    // Triangulate new points with existing good views
                    foreach (int goodViewIdx in _goodViews)
                    {
                        if (goodViewIdx == newViewIdx) continue;

                        int q = (newViewIdx < goodViewIdx) ? newViewIdx : goodViewIdx;
                        int t = (newViewIdx < goodViewIdx) ? goodViewIdx : newViewIdx;

                        var matches = GetMatching(q, t);
                        List<Point3D> newPoints;

                        if (TriangulateViews(_views[q].Points2D, _views[t].Points2D, _views[q].P, _views[t].P, matches, _K, new KeyValuePair<int, int>(q, t), out newPoints))
                        {
                            MergeNewPoints(newPoints);
                        }
                    }

                    _goodViews.Add(newViewIdx);
                    PerformBundleAdjustment();
                }

                if (!anyAdded) break;
            }

            Console.WriteLine($"\nImages processed: {_doneViews.Count}/{_views.Count}");
            Console.WriteLine($"PointCloud size: {_reconstructionCloud.Count}");
            return true;
        }

        private void PerformBundleAdjustment()
        {
            if (_reconstructionCloud.Count < 10) return;

            Console.WriteLine("Performing Bundle Adjustment...");
            var ba = new BundleAdjustment();

            // Add registered cameras
            // Keep track of mapping from ViewIndex to BA Camera Index
            var viewToCamIndex = new Dictionary<int, int>();
            int camCounter = 0;

            foreach (var view in _views)
            {
                if (view.IsRegistered)
                {
                    // Fix the first camera (index 0 usually) to anchor the world
                    // Or specifically check the first registered one.
                    // Assuming views[0] is the anchor if registered.
                    bool fix = (view.Index == _views.First(v => v.IsRegistered).Index);
                    ba.AddCamera(view.Index, view.R, view.t, _K, fix);
                    viewToCamIndex[view.Index] = camCounter++;
                }
            }

            // Add points and observations
            var baPoints = new Dictionary<int, Point3d>();
            int ptCounter = 0;

            // Rebuild point list for BA to have sequential indices
            // Map original Point3D object to BA index
            var ptToBAIndex = new Dictionary<Point3D, int>();

            for (int i = 0; i < _reconstructionCloud.Count; i++)
            {
                var pt = _reconstructionCloud[i];
                ba.AddPoint(i, pt.Pt); // ID is index in original list
                ptToBAIndex[pt] = i;
                baPoints[i] = pt.Pt;

                // Add observations for this point
                foreach (var obs in pt.IdxImage)
                {
                    int viewIdx = obs.Key;
                    int featIdx = obs.Value;

                    if (viewToCamIndex.ContainsKey(viewIdx))
                    {
                        var obs2d = _views[viewIdx].Points2D[featIdx];
                        ba.AddObservation(viewToCamIndex[viewIdx], i, obs2d);
                    }
                }
            }

            // Optimize
            ba.Optimize(10); // 10 iterations

            // Sync back results
            var rotations = new List<Mat>();
            var translations = new List<Mat>();

            // Prepare lists matching BA indices
            // We iterate registered views in the same order as added
            foreach (var view in _views)
            {
                if (view.IsRegistered)
                {
                    rotations.Add(view.R);
                    translations.Add(view.t);
                }
            }

            ba.UpdateResults(rotations, translations, baPoints);

            // Update P matrices for views
            foreach (var view in _views)
            {
                if (view.IsRegistered)
                {
                    // Update projection matrix P = [R|t] (Extrinsics 3x4)
                    // P is used in TriangulateViews with normalized coordinates (UndistortPoints).
                    view.P = new Mat(3, 4, MatType.CV_64F);
                    view.R.CopyTo(view.P.ColRange(0, 3));
                    view.t.CopyTo(view.P.ColRange(3, 4));
                }
            }

            // Update point cloud positions
            for(int i=0; i<_reconstructionCloud.Count; i++)
            {
                if(baPoints.ContainsKey(i))
                {
                    _reconstructionCloud[i].Pt = baPoints[i];
                }
            }
        }

        // -----------------------------------------------------------------------
        // Algorithms & Helpers
        // -----------------------------------------------------------------------

        private SortedDictionary<float, KeyValuePair<int, int>> FindBestPair()
        {
            Console.WriteLine("Getting best two views for baseline...");
            var numInliers = new SortedDictionary<float, KeyValuePair<int, int>>();

            int numImg = _views.Count;

            for (int q = 0; q < numImg - 1; q++)
            {
                for (int t = q + 1; t < numImg; t++)
                {
                    var matches = GetMatching(q, t);
                    if (matches.Length < 100) continue;

                    int hInliers = FindHomographyInliers(q, t, matches);
                    if (hInliers < 40) continue;

                    var pts1 = new List<Point2d>();
                    var pts2 = new List<Point2d>();
                    foreach(var m in matches)
                    {
                        pts1.Add(_views[q].Points2D[m.QueryIdx]);
                        pts2.Add(_views[t].Points2D[m.TrainIdx]);
                    }

                    Mat mask = new Mat();
                    using var m1 = CreateMatFromPoint2d(pts1);
                    using var m2 = CreateMatFromPoint2d(pts2);
                    Cv2.FindEssentialMat(m1, m2, _K, EssentialMatMethod.Ransac, 0.999, 1.0, mask);

                    int eInliers = Cv2.CountNonZero(mask);
                    float ratio = (float)eInliers / matches.Length;

                    Console.WriteLine($"Pair [{q},{t}] Matches: {matches.Length}, H-Inliers: {hInliers}, E-Inliers: {eInliers} ({ratio:P})");

                    while (numInliers.ContainsKey(ratio)) ratio += 0.00001f;
                    numInliers.Add(ratio, new KeyValuePair<int, int>(q, t));
                }
            }

            return numInliers;
        }

        private DMatch[] GetMatching(int q, int t)
        {
            using var matcher = new BFMatcher(NormTypes.Hamming, false);
            var knnMatches = matcher.KnnMatch(_views[q].Descriptors, _views[t].Descriptors, 2);

            var goodMatches = new List<DMatch>();
            foreach (var knn in knnMatches)
            {
                if (knn.Length >= 2 && knn[0].Distance <= NN_MATCH_RATIO * knn[1].Distance)
                {
                    goodMatches.Add(knn[0]);
                }
            }
            return goodMatches.ToArray();
        }

        private int FindHomographyInliers(int q, int t, DMatch[] matches)
        {
            if (matches.Length < 4) return 0;
            var pts1 = matches.Select(m => _views[q].Points2D[m.QueryIdx]).ToList();
            var pts2 = matches.Select(m => _views[t].Points2D[m.TrainIdx]).ToList();

            Mat mask = new Mat();
            using var m1 = CreateMatFromPoint2d(pts1);
            using var m2 = CreateMatFromPoint2d(pts2);
            Cv2.FindHomography((InputArray)m1, (InputArray)m2, HomographyMethods.Ransac, 3.0, mask);
            return Cv2.CountNonZero(mask);
        }

        private bool GetCameraPose(Mat K, int q, int t, DMatch[] matches, List<Point2d> pts1, List<Point2d> pts2, ref Mat Pleft, ref Mat Pright)
        {
            var aligned1 = new List<Point2d>();
            var aligned2 = new List<Point2d>();
            foreach(var m in matches)
            {
                aligned1.Add(pts1[m.QueryIdx]);
                aligned2.Add(pts2[m.TrainIdx]);
            }

            if (aligned1.Count <= 5) return false;

            Mat mask = new Mat();

            using var m1 = CreateMatFromPoint2d(aligned1);
            using var m2 = CreateMatFromPoint2d(aligned2);

            Mat E = Cv2.FindEssentialMat((InputArray)m1, (InputArray)m2, K, EssentialMatMethod.Ransac, 0.999, 1.0, mask);

            if (E.Rows != 3 || E.Cols != 3) return false;

            Mat R = new Mat();
            Mat T = new Mat();

            int valid = Cv2.RecoverPose((InputArray)E, (InputArray)m1, (InputArray)m2, K, R, T, mask);

            if (valid < 5) return false;

            if (R.Type() != MatType.CV_64F) R.ConvertTo(R, MatType.CV_64F);
            if (T.Type() != MatType.CV_64F) T.ConvertTo(T, MatType.CV_64F);

            Pright = new Mat(3, 4, MatType.CV_64F);
            R.CopyTo(Pright.ColRange(0, 3));
            T.CopyTo(Pright.ColRange(3, 4));

            Pleft = Mat.Eye(3, 4, MatType.CV_64F);

            return true;
        }

        private bool TriangulateViews(List<Point2d> pts1, List<Point2d> pts2, Mat P1, Mat P2, DMatch[] matches, Mat K, KeyValuePair<int, int> pair, out List<Point3D> pointcloud)
        {
            pointcloud = new List<Point3D>();

            var aligned1 = new List<Point2d>();
            var aligned2 = new List<Point2d>();
            foreach(var m in matches)
            {
                aligned1.Add(pts1[m.QueryIdx]);
                aligned2.Add(pts2[m.TrainIdx]);
            }

            if (aligned1.Count == 0) return false;

            // Undistort / Normalize
            Mat norm1 = new Mat();
            Mat norm2 = new Mat();

            // Convert to Point2f for UndistortPoints
            var aligned1f = aligned1.Select(p => new Point2f((float)p.X, (float)p.Y)).ToList();
            var aligned2f = aligned2.Select(p => new Point2f((float)p.X, (float)p.Y)).ToList();

            using var m1 = CreateMatFromPoint2f(aligned1f);
            using var m2 = CreateMatFromPoint2f(aligned2f);

            Cv2.UndistortPoints((InputArray)m1, norm1, K, _distCoef);
            Cv2.UndistortPoints((InputArray)m2, norm2, K, _distCoef);

            Mat pts4D = new Mat();
            Cv2.TriangulatePoints(P1, P2, norm1, norm2, pts4D);
            if (pts4D.Type() != MatType.CV_64F) pts4D.ConvertTo(pts4D, MatType.CV_64F);

            Mat pts3D = new Mat();
            Cv2.ConvertPointsFromHomogeneous(pts4D.T(), pts3D);

            // Reprojection check
            Mat rvec1 = new Mat(); Mat tvec1 = P1.ColRange(3, 4);
            Cv2.Rodrigues(P1.ColRange(0, 3), rvec1);

            Mat rvec2 = new Mat(); Mat tvec2 = P2.ColRange(3, 4);
            Cv2.Rodrigues(P2.ColRange(0, 3), rvec2);

            Mat proj1 = new Mat();
            Mat proj2 = new Mat();
            Cv2.ProjectPoints(pts3D, rvec1, tvec1, K, _distCoef, proj1);
            Cv2.ProjectPoints(pts3D, rvec2, tvec2, K, _distCoef, proj2);

            float minReprojError = 6.0f;

            for(int i = 0; i < pts3D.Rows; i++)
            {
                Point3d p3 = new Point3d(pts3D.At<double>(i, 0), pts3D.At<double>(i, 1), pts3D.At<double>(i, 2));

                Point2d p2_1 = proj1.At<Point2d>(i);
                Point2d p2_2 = proj2.At<Point2d>(i);

                double err1 = Math.Sqrt(Math.Pow(p2_1.X - aligned1[i].X, 2) + Math.Pow(p2_1.Y - aligned1[i].Y, 2));
                double err2 = Math.Sqrt(Math.Pow(p2_2.X - aligned2[i].X, 2) + Math.Pow(p2_2.Y - aligned2[i].Y, 2));

                if (err1 > minReprojError || err2 > minReprojError) continue;

                var pt = new Point3D
                {
                    Pt = p3,
                    Color = GetColor(_views[pair.Key].Image, aligned1[i])
                };
                pt.IdxImage[pair.Key] = matches[i].QueryIdx;
                pt.IdxImage[pair.Value] = matches[i].TrainIdx;

                pointcloud.Add(pt);
            }

            return pointcloud.Count > 0;
        }

        private void Find2D3DMatches(int newViewIdx, out List<Point3d> pts3D, out List<Point2d> pts2D, out DMatch[] bestMatches, out int doneViewRef)
        {
            pts3D = new List<Point3d>();
            pts2D = new List<Point2d>();
            bestMatches = new DMatch[0];
            doneViewRef = -1;

            int bestCount = 0;

            foreach (int doneIdx in _doneViews)
            {
                var matches = GetMatching(newViewIdx, doneIdx);
                if (matches.Length > bestCount)
                {
                    bestCount = matches.Length;
                    bestMatches = matches;
                    doneViewRef = doneIdx;
                }
            }

            if (doneViewRef == -1) return;

            // Use a local copy of doneViewRef to avoid closure error
            int currentDoneViewRef = doneViewRef;

            foreach (var m in bestMatches)
            {
                int doneFeatIdx = m.TrainIdx;
                int newFeatIdx = m.QueryIdx;

                // Fast Lookup
                if (_featureToPointMap.TryGetValue((currentDoneViewRef, doneFeatIdx), out var cloudPt))
                {
                    pts3D.Add(cloudPt.Pt);
                    pts2D.Add(_views[newViewIdx].Points2D[newFeatIdx]);
                }
            }
        }

        private bool FindCameraPosePNP(Mat K, List<Point3d> pts3D, List<Point2d> pts2D, out Mat P)
        {
            P = null;
            if (pts3D.Count < 6) return false;

            Mat rvec = new Mat();
            Mat tvec = new Mat();
            Mat inliers = new Mat();

            // SolvePnPRansac can take double (CV_64F)
            using var m3d = CreateMatFromPoint3d(pts3D);
            using var m2d = CreateMatFromPoint2d(pts2D);

            Cv2.SolvePnPRansac((InputArray)m3d, (InputArray)m2d, K, _distCoef, rvec, tvec,
                false, 1000, 8.0f, 0.99, inliers, SolvePnPFlags.EPNP);

            if (inliers.Rows < 8) return false;

            if (Cv2.Norm(tvec) > 500.0) return false;

            Mat R = new Mat();
            Cv2.Rodrigues(rvec, R);

            if (Math.Abs(Cv2.Determinant(R) - 1.0) > 1e-5) return false;

            P = new Mat(3, 4, MatType.CV_64F);
            if(R.Type() != MatType.CV_64F) R.ConvertTo(R, MatType.CV_64F);
            if(tvec.Type() != MatType.CV_64F) tvec.ConvertTo(tvec, MatType.CV_64F);

            R.CopyTo(P.ColRange(0, 3));
            tvec.CopyTo(P.ColRange(3, 4));

            return true;
        }

        private void MergeNewPoints(List<Point3D> newPoints)
        {
            float MERGE_DIST = 0.01f;

            foreach (var np in newPoints)
            {
                bool exists = false;

                // Check if point already exists in the cloud to merge.
                // Ensure Feature Map is updated on merge.

                foreach (var ep in _reconstructionCloud)
                {
                    double dist = Math.Sqrt(Math.Pow(np.Pt.X - ep.Pt.X, 2) + Math.Pow(np.Pt.Y - ep.Pt.Y, 2) + Math.Pow(np.Pt.Z - ep.Pt.Z, 2));
                    if (dist < MERGE_DIST)
                    {
                        exists = true;
                        foreach (var kv in np.IdxImage)
                        {
                            if (!ep.IdxImage.ContainsKey(kv.Key))
                            {
                                ep.IdxImage[kv.Key] = kv.Value;
                                // Update Map
                                _featureToPointMap[(kv.Key, kv.Value)] = ep;
                            }
                        }
                        break;
                    }
                }
                if (!exists)
                {
                    _reconstructionCloud.Add(np);
                    foreach(var kv in np.IdxImage)
                    {
                        _featureToPointMap[(kv.Key, kv.Value)] = np;
                    }
                }
            }
        }

        private void DecomposeP(Mat P, out Mat R, out Mat t)
        {
            R = P.ColRange(0, 3).Clone();
            t = P.ColRange(3, 4).Clone();
        }

        private Scalar GetColor(Mat img, Point2d pt)
        {
            int x = Math.Clamp((int)pt.X, 0, img.Width - 1);
            int y = Math.Clamp((int)pt.Y, 0, img.Height - 1);
            Vec3b c = img.At<Vec3b>(y, x);
            return new Scalar(c.Item0, c.Item1, c.Item2);
        }

        private Mat CreateMatFromPoint2d(IEnumerable<Point2d> points)
        {
            var arr = points.ToArray();
            var mat = new Mat(arr.Length, 1, MatType.CV_64FC2);
            for(int i=0; i<arr.Length; i++) mat.Set(i, 0, arr[i]);
            return mat;
        }

        private Mat CreateMatFromPoint2f(IEnumerable<Point2f> points)
        {
            var arr = points.ToArray();
            var mat = new Mat(arr.Length, 1, MatType.CV_32FC2);
            for(int i=0; i<arr.Length; i++) mat.Set(i, 0, arr[i]);
            return mat;
        }

        private Mat CreateMatFromPoint3d(IEnumerable<Point3d> points)
        {
            var arr = points.ToArray();
            var mat = new Mat(arr.Length, 1, MatType.CV_64FC3);
            for(int i=0; i<arr.Length; i++) mat.Set(i, 0, arr[i]);
            return mat;
        }

        // -----------------------------------------------------------------------
        // Output Conversion
        // -----------------------------------------------------------------------

        private SceneResult ConvertToSceneResult()
        {
            var result = new SceneResult();
            var mesh = new MeshData();

            foreach (var pt in _reconstructionCloud)
            {
                var v = new Vector3((float)pt.Pt.X, -(float)pt.Pt.Y, -(float)pt.Pt.Z);
                mesh.Vertices.Add(v);
                mesh.Colors.Add(new Vector3((float)pt.Color.Val2 / 255f, (float)pt.Color.Val1 / 255f, (float)pt.Color.Val0 / 255f));
            }
            result.Meshes.Add(mesh);

            foreach (var view in _views)
            {
                if (!view.IsRegistered) continue;

                // Use using statements to ensure proper disposal of temporary Mat objects
                using Mat R_wc = view.R.T();
                using Mat C_wc = -R_wc * view.t;

                Vector3 pos = new Vector3((float)C_wc.At<double>(0), -(float)C_wc.At<double>(1), -(float)C_wc.At<double>(2));

                using Mat R_gl_temp = new Mat(3, 3, MatType.CV_64F);
                R_gl_temp.Set(0, 0, view.R.At<double>(0, 0));
                R_gl_temp.Set(0, 1, -view.R.At<double>(0, 1));
                R_gl_temp.Set(0, 2, -view.R.At<double>(0, 2));
                R_gl_temp.Set(1, 0, -view.R.At<double>(1, 0));
                R_gl_temp.Set(1, 1, view.R.At<double>(1, 1));
                R_gl_temp.Set(1, 2, view.R.At<double>(1, 2));
                R_gl_temp.Set(2, 0, -view.R.At<double>(2, 0));
                R_gl_temp.Set(2, 1, view.R.At<double>(2, 1));
                R_gl_temp.Set(2, 2, view.R.At<double>(2, 2));

                using Mat R_gl = R_gl_temp.T();

                var m4 = Matrix4.Identity;
                m4.M11 = (float)R_gl.At<double>(0, 0); m4.M12 = (float)R_gl.At<double>(0, 1); m4.M13 = (float)R_gl.At<double>(0, 2);
                m4.M21 = (float)R_gl.At<double>(1, 0); m4.M22 = (float)R_gl.At<double>(1, 1); m4.M23 = (float)R_gl.At<double>(1, 2);
                m4.M31 = (float)R_gl.At<double>(2, 0); m4.M32 = (float)R_gl.At<double>(2, 1); m4.M33 = (float)R_gl.At<double>(2, 2);

                m4.M41 = pos.X;
                m4.M42 = pos.Y;
                m4.M43 = pos.Z;

                result.Poses.Add(new CameraPose
                {
                    ImageIndex = view.Index,
                    ImagePath = view.Path,
                    Width = view.Size.Width,
                    Height = view.Size.Height,
                    CameraToWorld = m4,
                    WorldToCamera = m4.Inverted(),
                    FocalLength = (float)_K.At<double>(0,0)
                });
            }

            // Clean up OpenCV Mat objects to prevent crashes during garbage collection
            CleanupResources();

            return result;
        }

        private void CleanupResources()
        {
            // Dispose all Mat objects in views
            foreach (var view in _views)
            {
                view.Image?.Dispose();
                view.Gray?.Dispose();
                view.Descriptors?.Dispose();
                view.P?.Dispose();
                view.R?.Dispose();
                view.t?.Dispose();
            }
            _views.Clear();

            // Dispose shared matrices
            _K?.Dispose();
            _distCoef?.Dispose();

            Console.WriteLine("[SfM] Resources cleaned up.");
        }

    }
}
