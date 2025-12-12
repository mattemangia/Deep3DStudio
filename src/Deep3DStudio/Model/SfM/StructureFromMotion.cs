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
        // Tracks 3D points across multiple views
        private class Track
        {
            public int TrackId;
            public Vector3 Point3D;
            public bool Has3D;
            // ViewIndex -> FeatureIndex
            public Dictionary<int, int> Observations = new Dictionary<int, int>();
            public Scalar Color;
        }

        private List<Track> _tracks = new List<Track>();
        private int _nextTrackId = 0;

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            var result = new SceneResult();
            if (imagePaths.Count < 2) return result;

            Console.WriteLine("Starting OpenCV SfM Reconstruction...");

            // 1. Feature Extraction (ORB)
            var keypointsList = new List<KeyPoint[]>();
            var descriptorsList = new List<Mat>();
            var images = new List<Mat>();
            var imageSizes = new List<Size>();
            var Ks = new List<Mat>(); // Per-image Intrinsics

            using var detector = ORB.Create(nFeatures: 5000);

            for (int i = 0; i < imagePaths.Count; i++)
            {
                var img = Cv2.ImRead(imagePaths[i], ImreadModes.Color);
                if (img.Empty()) continue;

                if (Math.Max(img.Width, img.Height) > 1200)
                {
                    double scale = 1200.0 / Math.Max(img.Width, img.Height);
                    Cv2.Resize(img, img, new Size(0,0), scale, scale);
                }

                images.Add(img);
                imageSizes.Add(img.Size());

                // Estimate K per image (handling Zoom)
                double f = Math.Max(img.Width, img.Height) * 0.8;
                var K = new Mat(3, 3, MatType.CV_64F, Scalar.All(0));
                K.Set(0, 0, f);
                K.Set(1, 1, f);
                K.Set(0, 2, img.Width / 2.0);
                K.Set(1, 2, img.Height / 2.0);
                K.Set(2, 2, 1.0);
                Ks.Add(K);

                KeyPoint[] kps;
                var desc = new Mat();
                detector.DetectAndCompute(img, null, out kps, desc);

                keypointsList.Add(kps);
                descriptorsList.Add(desc);
                Console.WriteLine($"Image {i}: {kps.Length} features");
            }

            if (keypointsList.Count < 2) return result;

            // Camera Poses (World -> Camera)
            var poses = new List<Matrix4>();
            var worldToCams = new List<(Mat R, Mat t)>();

            // 2. Initialize with Pair 0-1
            var matches01 = MatchFeatures(descriptorsList[0], descriptorsList[1]);

            // Normalize points for essential matrix (independent of K)
            var normPts0 = NormalizePoints(keypointsList[0], matches01.Select(m => m.QueryIdx).ToArray(), Ks[0]);
            var normPts1 = NormalizePoints(keypointsList[1], matches01.Select(m => m.TrainIdx).ToArray(), Ks[1]);

            if (normPts0.Count < 10) return result;

            // Recover Pose (from Normalized Coordinates -> E is essential)
            using var mask = new Mat();
            // Use Identity K because points are already normalized
            using var eye3 = Mat.Eye(3, 3, MatType.CV_64F).ToMat();

            // 8 = RANSAC
            using var E = Cv2.FindEssentialMat(InputArray.Create(normPts0), InputArray.Create(normPts1), eye3, (EssentialMatMethod)8, 0.999, 0.003, mask);

            var R = new Mat();
            var t = new Mat();
            Cv2.RecoverPose(E, InputArray.Create(normPts0), InputArray.Create(normPts1), eye3, R, t, mask);

            // Pose 0 is Identity
            worldToCams.Add((Mat.Eye(3, 3, MatType.CV_64F).ToMat(), new Mat(3, 1, MatType.CV_64F, Scalar.All(0))));
            poses.Add(Matrix4.Identity);

            // Pose 1
            worldToCams.Add((R.Clone(), t.Clone()));
            poses.Add(CvPoseToOpenTK(R, t));

            // Create Tracks 0-1
            TriangulateAndCreateTracks(0, 1, matches01, mask, worldToCams[0], worldToCams[1], Ks, keypointsList);

            Console.WriteLine($"Initialized with {normPts0.Count} matches. {_tracks.Count} 3D points.");

            // 3. Incremental SfM
            for (int i = 2; i < images.Count; i++)
            {
                var matches = MatchFeatures(descriptorsList[i-1], descriptorsList[i]);

                // Find 2D-3D Correspondences
                var objectPoints = new List<Point3d>();
                var imagePoints = new List<Point2d>();
                var featureIndices = new List<int>();

                foreach(var m in matches)
                {
                    int prevIdx = m.QueryIdx;
                    int currIdx = m.TrainIdx;

                    var track = _tracks.FirstOrDefault(t => t.Observations.ContainsKey(i-1) && t.Observations[i-1] == prevIdx && t.Has3D);

                    if (track != null)
                    {
                        objectPoints.Add(new Point3d(track.Point3D.X, track.Point3D.Y, track.Point3D.Z));
                        imagePoints.Add(new Point2d(keypointsList[i][currIdx].Pt.X, keypointsList[i][currIdx].Pt.Y));
                        featureIndices.Add(currIdx);
                    }
                }

                if (objectPoints.Count < 6)
                {
                    Console.WriteLine($"Image {i}: Tracking lost.");
                    poses.Add(poses.Last());
                    worldToCams.Add((worldToCams.Last().R.Clone(), worldToCams.Last().t.Clone()));
                    continue;
                }

                // Solve PnP using current camera's K
                using var rvec = new Mat();
                var tvec = new Mat();
                using var pnpInliers = new Mat();

                Cv2.SolvePnPRansac(InputArray.Create(objectPoints), InputArray.Create(imagePoints), Ks[i], null, rvec, tvec,
                    useExtrinsicGuess: false, iterationsCount: 100, reprojectionError: 8.0f, confidence: 0.99, inliers: pnpInliers);

                var R_curr = new Mat();
                Cv2.Rodrigues(rvec, R_curr);

                worldToCams.Add((R_curr, tvec));
                poses.Add(CvPoseToOpenTK(R_curr, tvec));

                // Update Tracks
                var indexer = pnpInliers.GetGenericIndexer<int>();
                var trackIds = new List<int>();
                foreach(var m in matches) {
                     var track = _tracks.FirstOrDefault(t => t.Observations.ContainsKey(i-1) && t.Observations[i-1] == m.QueryIdx && t.Has3D);
                     if(track != null) trackIds.Add(track.TrackId);
                }

                for(int k=0; k<pnpInliers.Rows; k++)
                {
                    int idx = indexer[k, 0];
                    int tId = trackIds[idx];
                    int fIdx = featureIndices[idx];

                    var track = _tracks.First(t => t.TrackId == tId);
                    track.Observations[i] = fIdx;
                }

                TriangulateNewPoints(i-1, i, matches, worldToCams[i-1], worldToCams[i], Ks, keypointsList);
            }

            // Build Mesh
            var mesh = new MeshData();
            foreach(var track in _tracks)
            {
                if (track.Has3D)
                {
                    mesh.Vertices.Add(track.Point3D);
                    mesh.Colors.Add(new Vector3(1,1,1));
                }
            }

            for (int i = 0; i < imagePaths.Count; i++)
            {
                result.Poses.Add(new CameraPose
                {
                    ImageIndex = i,
                    ImagePath = imagePaths[i],
                    Width = imageSizes[i].Width,
                    Height = imageSizes[i].Height,
                    CameraToWorld = poses[i],
                    WorldToCamera = poses[i].Inverted()
                });
            }
            result.Meshes.Add(mesh);

            // Clean up
            foreach(var img in images) img.Dispose();
            foreach(var d in descriptorsList) d.Dispose();
            foreach(var wc in worldToCams) { wc.R.Dispose(); wc.t.Dispose(); }
            foreach(var k in Ks) k.Dispose();

            return result;
        }

        private DMatch[] MatchFeatures(Mat d1, Mat d2)
        {
            using var matcher = new BFMatcher(NormTypes.Hamming, crossCheck: true);
            return matcher.Match(d1, d2);
        }

        private List<Point2d> NormalizePoints(KeyPoint[] kps, int[] indices, Mat K)
        {
            var pts = new List<Point2d>();
            double fx = K.At<double>(0, 0);
            double fy = K.At<double>(1, 1);
            double cx = K.At<double>(0, 2);
            double cy = K.At<double>(1, 2);

            foreach(var idx in indices)
            {
                double u = kps[idx].Pt.X;
                double v = kps[idx].Pt.Y;
                pts.Add(new Point2d((u - cx)/fx, (v - cy)/fy));
            }
            return pts;
        }

        private void TriangulateAndCreateTracks(int view1, int view2, DMatch[] matches, Mat mask,
            (Mat R, Mat t) pose1, (Mat R, Mat t) pose2, List<Mat> Ks, List<KeyPoint[]> kps)
        {
            // P = K * [R|t]
            Mat P1 = ComputeP(Ks[view1], pose1.R, pose1.t);
            Mat P2 = ComputeP(Ks[view2], pose2.R, pose2.t);

            var pts1 = new List<Point2d>();
            var pts2 = new List<Point2d>();
            var goodMatches = new List<DMatch>();

            var indexer = mask.GetGenericIndexer<byte>();
            for (int i = 0; i < matches.Length; i++)
            {
                if (indexer[i, 0] != 0)
                {
                    pts1.Add(new Point2d(kps[view1][matches[i].QueryIdx].Pt.X, kps[view1][matches[i].QueryIdx].Pt.Y));
                    pts2.Add(new Point2d(kps[view2][matches[i].TrainIdx].Pt.X, kps[view2][matches[i].TrainIdx].Pt.Y));
                    goodMatches.Add(matches[i]);
                }
            }

            if (pts1.Count == 0) { P1.Dispose(); P2.Dispose(); return; }

            using var pts4D = new Mat();
            Cv2.TriangulatePoints(P1, P2, InputArray.Create(pts1), InputArray.Create(pts2), pts4D);

            P1.Dispose(); P2.Dispose();

            for (int i = 0; i < pts4D.Cols; i++)
            {
                double w = pts4D.At<double>(3, i);
                if (Math.Abs(w) < 1e-6) continue;

                float x = (float)(pts4D.At<double>(0, i) / w);
                float y = (float)(pts4D.At<double>(1, i) / w);
                float z = (float)(pts4D.At<double>(2, i) / w);

                var track = new Track
                {
                    TrackId = _nextTrackId++,
                    Point3D = new Vector3(x, y, z),
                    Has3D = true
                };
                track.Observations[view1] = goodMatches[i].QueryIdx;
                track.Observations[view2] = goodMatches[i].TrainIdx;
                _tracks.Add(track);
            }
        }

        private void TriangulateNewPoints(int view1, int view2, DMatch[] matches,
             (Mat R, Mat t) pose1, (Mat R, Mat t) pose2, List<Mat> Ks, List<KeyPoint[]> kps)
        {
            var candidateMatches = new List<DMatch>();
            var pts1 = new List<Point2d>();
            var pts2 = new List<Point2d>();

            foreach(var m in matches)
            {
                bool tracked = _tracks.Any(t => t.Observations.ContainsKey(view1) && t.Observations[view1] == m.QueryIdx);
                if (!tracked)
                {
                    candidateMatches.Add(m);
                    pts1.Add(new Point2d(kps[view1][m.QueryIdx].Pt.X, kps[view1][m.QueryIdx].Pt.Y));
                    pts2.Add(new Point2d(kps[view2][m.TrainIdx].Pt.X, kps[view2][m.TrainIdx].Pt.Y));
                }
            }

            if (candidateMatches.Count == 0) return;

            Mat P1 = ComputeP(Ks[view1], pose1.R, pose1.t);
            Mat P2 = ComputeP(Ks[view2], pose2.R, pose2.t);
            using var pts4D = new Mat();
            Cv2.TriangulatePoints(P1, P2, InputArray.Create(pts1), InputArray.Create(pts2), pts4D);

            P1.Dispose(); P2.Dispose();

            for (int i = 0; i < pts4D.Cols; i++)
            {
                double w = pts4D.At<double>(3, i);
                if (Math.Abs(w) < 1e-5) continue;

                float x = (float)(pts4D.At<double>(0, i) / w);
                float y = (float)(pts4D.At<double>(1, i) / w);
                float z = (float)(pts4D.At<double>(2, i) / w);

                var track = new Track
                {
                    TrackId = _nextTrackId++,
                    Point3D = new Vector3(x, y, z),
                    Has3D = true
                };
                track.Observations[view1] = candidateMatches[i].QueryIdx;
                track.Observations[view2] = candidateMatches[i].TrainIdx;
                _tracks.Add(track);
            }
        }

        private Mat ComputeP(Mat K, Mat R, Mat t)
        {
            // P = K * [R|t]
            Mat P = new Mat(3, 4, MatType.CV_64F);
            Mat Rt = new Mat();
            Cv2.HConcat(new[] { R, t }, Rt);
            return K * Rt;
        }

        private Matrix4 CvPoseToOpenTK(Mat R, Mat t)
        {
            var M = Matrix4.Identity;
            M.M11 = (float)R.At<double>(0, 0); M.M12 = (float)R.At<double>(1, 0); M.M13 = (float)R.At<double>(2, 0);
            M.M21 = (float)R.At<double>(0, 1); M.M22 = (float)R.At<double>(1, 1); M.M23 = (float)R.At<double>(2, 1);
            M.M31 = (float)R.At<double>(0, 2); M.M32 = (float)R.At<double>(1, 2); M.M33 = (float)R.At<double>(2, 2);

            M.M41 = (float)t.At<double>(0, 0);
            M.M42 = (float)t.At<double>(1, 0);
            M.M43 = (float)t.At<double>(2, 0);

            return M.Inverted();
        }
    }
}
