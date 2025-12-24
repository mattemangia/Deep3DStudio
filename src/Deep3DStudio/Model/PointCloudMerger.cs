using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using MathNet.Numerics.LinearAlgebra;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.Model
{
    /// <summary>
    /// Intelligent point cloud merging and alignment system.
    /// Supports ICP, feature-based registration, and multi-cloud fusion.
    /// </summary>
    public class PointCloudMerger
    {
        public float VoxelSize { get; set; } = 0.02f;
        public int MaxIterations { get; set; } = 50;
        public float ConvergenceThreshold { get; set; } = 1e-6f;
        public float MaxCorrespondenceDistance { get; set; } = 0.1f;
        public float OutlierRejectionThreshold { get; set; } = 2.0f;

        private readonly Action<string>? _progressCallback;

        public PointCloudMerger(Action<string>? progressCallback = null)
        {
            _progressCallback = progressCallback;
        }

        /// <summary>
        /// Merge multiple point clouds into a single unified cloud.
        /// Uses pairwise ICP alignment then global optimization.
        /// </summary>
        public async Task<PointCloudData> MergePointCloudsAsync(List<PointCloudData> clouds)
        {
            if (clouds == null || clouds.Count == 0)
                throw new ArgumentException("No point clouds to merge");

            if (clouds.Count == 1)
                return clouds[0].Clone();

            // Ensure settings are respected (e.g., if GPU is selected, we might want to ensure we use Parallel loops effectively on CPU as fallback,
            // since native GPU ICP requires Compute Shaders or CUDA which is not fully implemented in C# here).
            // But we can check:
            var settings = IniSettings.Instance;
            bool useParallel = true;

            _progressCallback?.Invoke($"Merging {clouds.Count} point clouds (Device: {settings.Device})...");

            // Downsample all clouds for faster registration
            var downsampledClouds = new List<PointCloudData>();
            for (int i = 0; i < clouds.Count; i++)
            {
                _progressCallback?.Invoke($"Downsampling cloud {i + 1}/{clouds.Count}...");
                downsampledClouds.Add(await Task.Run(() => VoxelDownsample(clouds[i], VoxelSize)));
            }

            // Compute transforms via sequential pairwise ICP
            var transforms = new List<Matrix4>();
            transforms.Add(Matrix4.Identity); // First cloud stays in place

            for (int i = 1; i < clouds.Count; i++)
            {
                _progressCallback?.Invoke($"Aligning cloud {i + 1}/{clouds.Count} to reference...");

                // Find best overlap with existing merged cloud
                var mergedSoFar = MergeWithTransforms(downsampledClouds.Take(i).ToList(), transforms);

                // Run ICP to align current cloud to merged
                var (transform, fitness, rmse) = await ICPAlignAsync(downsampledClouds[i], mergedSoFar);

                _progressCallback?.Invoke($"Cloud {i + 1}: fitness={fitness:F3}, RMSE={rmse:F4}");
                transforms.Add(transform);
            }

            // Global refinement using pose graph optimization
            _progressCallback?.Invoke("Running global refinement...");
            transforms = await GlobalRefineAsync(downsampledClouds, transforms);

            // Merge all clouds with final transforms (using original resolution)
            _progressCallback?.Invoke("Merging aligned clouds...");
            var merged = MergeWithTransforms(clouds, transforms);

            // Remove duplicate points
            _progressCallback?.Invoke("Removing duplicates...");
            merged = await Task.Run(() => RemoveDuplicates(merged, VoxelSize * 0.5f));

            // Statistical outlier removal
            _progressCallback?.Invoke("Removing outliers...");
            merged = await Task.Run(() => RemoveStatisticalOutliers(merged, 20, OutlierRejectionThreshold));

            _progressCallback?.Invoke($"Merge complete: {merged.Points.Length} points");
            return merged;
        }

        /// <summary>
        /// Iterative Closest Point alignment between source and target clouds.
        /// Returns transform, fitness score, and RMSE.
        /// </summary>
        public async Task<(Matrix4 transform, float fitness, float rmse)> ICPAlignAsync(
            PointCloudData source, PointCloudData target)
        {
            return await Task.Run(() =>
            {
                var transform = Matrix4.Identity;
                var sourcePoints = source.Points.Select(p => new Vector3(p.X, p.Y, p.Z)).ToList();
                var targetKdTree = BuildKdTree(target.Points);

                float prevError = float.MaxValue;

                for (int iter = 0; iter < MaxIterations; iter++)
                {
                    // Transform source points
                    var transformedSource = sourcePoints.Select(p => GeometryUtils.TransformPoint(p, transform)).ToList();

                    // Find correspondences
                    var (srcCorr, tgtCorr, distances) = FindCorrespondences(transformedSource, target.Points, targetKdTree);

                    if (srcCorr.Count < 4)
                    {
                        break;
                    }

                    // Compute transform from correspondences
                    var deltaTransform = GeometryUtils.ComputeRigidTransformRANSAC(
                        srcCorr, tgtCorr, out int inliers, out float rmse,
                        maxIterations: 50, inlierThreshold: MaxCorrespondenceDistance * 0.5f);

                    transform = deltaTransform * transform;

                    // Check convergence
                    float error = distances.Average();
                    if (Math.Abs(prevError - error) < ConvergenceThreshold)
                        break;

                    prevError = error;
                }

                // Compute final metrics
                var finalTransformed = sourcePoints.Select(p => GeometryUtils.TransformPoint(p, transform)).ToList();
                var (_, _, finalDistances) = FindCorrespondences(finalTransformed, target.Points, targetKdTree);

                float fitness = finalDistances.Count(d => d < MaxCorrespondenceDistance) / (float)sourcePoints.Count;
                float finalRmse = finalDistances.Count > 0 ? (float)Math.Sqrt(finalDistances.Average(d => d * d)) : float.MaxValue;

                return (transform, fitness, finalRmse);
            });
        }

        /// <summary>
        /// Global refinement using pose graph optimization.
        /// Refines all transforms simultaneously for better consistency.
        /// </summary>
        private async Task<List<Matrix4>> GlobalRefineAsync(List<PointCloudData> clouds, List<Matrix4> initialTransforms)
        {
            return await Task.Run(() =>
            {
                var transforms = initialTransforms.ToList();
                int n = clouds.Count;

                // Build pose graph edges (pairwise constraints)
                var edges = new List<(int i, int j, Matrix4 relativeTransform, float weight)>();

                for (int i = 0; i < n; i++)
                {
                    for (int j = i + 1; j < n; j++)
                    {
                        // Check if clouds have significant overlap
                        var cloudI = TransformCloud(clouds[i], transforms[i]);
                        var cloudJ = TransformCloud(clouds[j], transforms[j]);

                        float overlap = ComputeOverlap(cloudI, cloudJ, MaxCorrespondenceDistance * 2);
                        if (overlap > 0.1f) // At least 10% overlap
                        {
                            // Compute relative transform
                            var (relTrans, fitness, rmse) = ICPAlignSync(cloudI, cloudJ);
                            if (fitness > 0.3f)
                            {
                                edges.Add((i, j, relTrans, fitness));
                            }
                        }
                    }
                }

                // Iterative refinement (simplified pose graph optimization)
                for (int iter = 0; iter < 10; iter++)
                {
                    bool changed = false;

                    foreach (var (i, j, relTrans, weight) in edges)
                    {
                        if (i == 0) continue; // Keep first cloud fixed

                        // Compute expected transform for i based on j and relative transform
                        var expectedI = relTrans.Inverted() * transforms[j];

                        // Blend towards expected (weighted by fitness)
                        transforms[i] = BlendTransforms(transforms[i], expectedI, weight * 0.1f);
                        changed = true;
                    }

                    if (!changed) break;
                }

                return transforms;
            });
        }

        private (Matrix4 transform, float fitness, float rmse) ICPAlignSync(PointCloudData source, PointCloudData target)
        {
            var transform = Matrix4.Identity;
            var sourcePoints = source.Points.Select(p => new Vector3(p.X, p.Y, p.Z)).ToList();
            var targetKdTree = BuildKdTree(target.Points);

            float prevError = float.MaxValue;

            for (int iter = 0; iter < MaxIterations / 2; iter++)
            {
                var transformedSource = sourcePoints.Select(p => GeometryUtils.TransformPoint(p, transform)).ToList();
                var (srcCorr, tgtCorr, distances) = FindCorrespondences(transformedSource, target.Points, targetKdTree);

                if (srcCorr.Count < 4) break;

                var deltaTransform = GeometryUtils.ComputeRigidTransform(srcCorr, tgtCorr);
                transform = deltaTransform * transform;

                float error = distances.Average();
                if (Math.Abs(prevError - error) < ConvergenceThreshold) break;
                prevError = error;
            }

            var finalTransformed = sourcePoints.Select(p => GeometryUtils.TransformPoint(p, transform)).ToList();
            var (_, _, finalDistances) = FindCorrespondences(finalTransformed, target.Points, targetKdTree);

            float fitness = finalDistances.Count(d => d < MaxCorrespondenceDistance) / (float)sourcePoints.Count;
            float finalRmse = finalDistances.Count > 0 ? (float)Math.Sqrt(finalDistances.Average(d => d * d)) : float.MaxValue;

            return (transform, fitness, finalRmse);
        }

        private Matrix4 BlendTransforms(Matrix4 a, Matrix4 b, float t)
        {
            // Simple linear blend of translation
            var transA = a.ExtractTranslation();
            var transB = b.ExtractTranslation();
            var blendedTrans = Vector3.Lerp(transA, transB, t);

            // Extract rotation as quaternion and slerp
            var quatA = a.ExtractRotation();
            var quatB = b.ExtractRotation();
            var blendedQuat = Quaternion.Slerp(quatA, quatB, t);

            return Matrix4.CreateFromQuaternion(blendedQuat) * Matrix4.CreateTranslation(blendedTrans);
        }

        private float ComputeOverlap(PointCloudData cloudA, PointCloudData cloudB, float threshold)
        {
            var kdTree = BuildKdTree(cloudB.Points);
            int matches = 0;

            foreach (var p in cloudA.Points)
            {
                var nearest = FindNearestPoint(new Vector3(p.X, p.Y, p.Z), cloudB.Points, kdTree);
                float dist = (new Vector3(p.X, p.Y, p.Z) - nearest).Length;
                if (dist < threshold) matches++;
            }

            return (float)matches / cloudA.Points.Length;
        }

        private PointCloudData TransformCloud(PointCloudData cloud, Matrix4 transform)
        {
            var result = new PointCloudData
            {
                Points = new System.Numerics.Vector3[cloud.Points.Length],
                Colors = cloud.Colors?.ToArray(),
                Normals = cloud.Normals?.ToArray()
            };

            for (int i = 0; i < cloud.Points.Length; i++)
            {
                var p = cloud.Points[i];
                var transformed = GeometryUtils.TransformPoint(new Vector3(p.X, p.Y, p.Z), transform);
                result.Points[i] = new System.Numerics.Vector3(transformed.X, transformed.Y, transformed.Z);
            }

            return result;
        }

        private PointCloudData MergeWithTransforms(List<PointCloudData> clouds, List<Matrix4> transforms)
        {
            var allPoints = new List<System.Numerics.Vector3>();
            var allColors = new List<System.Numerics.Vector3>();
            var allNormals = new List<System.Numerics.Vector3>();

            bool hasColors = clouds.All(c => c.Colors != null && c.Colors.Length > 0);
            bool hasNormals = clouds.All(c => c.Normals != null && c.Normals.Length > 0);

            for (int i = 0; i < clouds.Count; i++)
            {
                var transform = transforms[i];
                var cloud = clouds[i];

                for (int j = 0; j < cloud.Points.Length; j++)
                {
                    var p = cloud.Points[j];
                    var transformed = GeometryUtils.TransformPoint(new Vector3(p.X, p.Y, p.Z), transform);
                    allPoints.Add(new System.Numerics.Vector3(transformed.X, transformed.Y, transformed.Z));

                    if (hasColors && cloud.Colors != null)
                        allColors.Add(cloud.Colors[j]);

                    if (hasNormals && cloud.Normals != null)
                    {
                        var n = cloud.Normals[j];
                        // Transform normal (rotation only)
                        var normalMat = new Matrix3(transform);
                        var transformedNormal = new Vector3(n.X, n.Y, n.Z) * normalMat;
                        transformedNormal.Normalize();
                        allNormals.Add(new System.Numerics.Vector3(transformedNormal.X, transformedNormal.Y, transformedNormal.Z));
                    }
                }
            }

            return new PointCloudData
            {
                Points = allPoints.ToArray(),
                Colors = hasColors ? allColors.ToArray() : null,
                Normals = hasNormals ? allNormals.ToArray() : null
            };
        }

        /// <summary>
        /// Voxel grid downsampling for faster processing.
        /// </summary>
        public PointCloudData VoxelDownsample(PointCloudData cloud, float voxelSize)
        {
            var voxelMap = new Dictionary<(int, int, int), List<int>>();

            for (int i = 0; i < cloud.Points.Length; i++)
            {
                var p = cloud.Points[i];
                var key = (
                    (int)Math.Floor(p.X / voxelSize),
                    (int)Math.Floor(p.Y / voxelSize),
                    (int)Math.Floor(p.Z / voxelSize)
                );

                if (!voxelMap.ContainsKey(key))
                    voxelMap[key] = new List<int>();
                voxelMap[key].Add(i);
            }

            var points = new List<System.Numerics.Vector3>();
            var colors = cloud.Colors != null ? new List<System.Numerics.Vector3>() : null;
            var normals = cloud.Normals != null ? new List<System.Numerics.Vector3>() : null;

            foreach (var voxel in voxelMap.Values)
            {
                // Compute centroid
                var avgPoint = System.Numerics.Vector3.Zero;
                var avgColor = System.Numerics.Vector3.Zero;
                var avgNormal = System.Numerics.Vector3.Zero;

                foreach (int idx in voxel)
                {
                    avgPoint += cloud.Points[idx];
                    if (colors != null && cloud.Colors != null) avgColor += cloud.Colors[idx];
                    if (normals != null && cloud.Normals != null) avgNormal += cloud.Normals[idx];
                }

                points.Add(avgPoint / voxel.Count);
                colors?.Add(avgColor / voxel.Count);
                if (normals != null)
                {
                    avgNormal = System.Numerics.Vector3.Normalize(avgNormal);
                    normals.Add(avgNormal);
                }
            }

            return new PointCloudData
            {
                Points = points.ToArray(),
                Colors = colors?.ToArray(),
                Normals = normals?.ToArray()
            };
        }

        /// <summary>
        /// Remove duplicate points within a distance threshold.
        /// </summary>
        public PointCloudData RemoveDuplicates(PointCloudData cloud, float threshold)
        {
            var voxelSize = threshold;
            return VoxelDownsample(cloud, voxelSize);
        }

        /// <summary>
        /// Statistical outlier removal based on mean distance to k nearest neighbors.
        /// </summary>
        public PointCloudData RemoveStatisticalOutliers(PointCloudData cloud, int kNeighbors, float stdRatio)
        {
            int n = cloud.Points.Length;
            if (n <= kNeighbors) return cloud;

            var kdTree = BuildKdTree(cloud.Points);
            var meanDistances = new float[n];

            // Compute mean distance to k neighbors for each point
            Parallel.For(0, n, i =>
            {
                var p = cloud.Points[i];
                var point = new Vector3(p.X, p.Y, p.Z);
                var distances = FindKNearestDistances(point, cloud.Points, kdTree, kNeighbors + 1);

                // Skip first (self) and compute mean
                meanDistances[i] = distances.Skip(1).Average();
            });

            // Compute global statistics
            float globalMean = meanDistances.Average();
            float globalStd = (float)Math.Sqrt(meanDistances.Average(d => (d - globalMean) * (d - globalMean)));
            float threshold = globalMean + stdRatio * globalStd;

            // Filter points
            var points = new List<System.Numerics.Vector3>();
            var colors = cloud.Colors != null ? new List<System.Numerics.Vector3>() : null;
            var normals = cloud.Normals != null ? new List<System.Numerics.Vector3>() : null;

            for (int i = 0; i < n; i++)
            {
                if (meanDistances[i] <= threshold)
                {
                    points.Add(cloud.Points[i]);
                    if (colors != null && cloud.Colors != null) colors.Add(cloud.Colors[i]);
                    if (normals != null && cloud.Normals != null) normals.Add(cloud.Normals[i]);
                }
            }

            return new PointCloudData
            {
                Points = points.ToArray(),
                Colors = colors?.ToArray(),
                Normals = normals?.ToArray()
            };
        }

        /// <summary>
        /// Estimate normals for a point cloud using PCA on local neighborhoods.
        /// </summary>
        public PointCloudData EstimateNormals(PointCloudData cloud, int kNeighbors = 30)
        {
            int n = cloud.Points.Length;
            var normals = new System.Numerics.Vector3[n];
            var kdTree = BuildKdTree(cloud.Points);

            Parallel.For(0, n, i =>
            {
                var p = cloud.Points[i];
                var point = new Vector3(p.X, p.Y, p.Z);
                var neighbors = FindKNearestPoints(point, cloud.Points, kdTree, kNeighbors);

                // Compute covariance matrix
                var centroid = neighbors.Aggregate(Vector3.Zero, (acc, v) => acc + v) / neighbors.Count;
                var cov = Matrix<double>.Build.Dense(3, 3);

                foreach (var neighbor in neighbors)
                {
                    var d = neighbor - centroid;
                    cov[0, 0] += d.X * d.X; cov[0, 1] += d.X * d.Y; cov[0, 2] += d.X * d.Z;
                    cov[1, 0] += d.Y * d.X; cov[1, 1] += d.Y * d.Y; cov[1, 2] += d.Y * d.Z;
                    cov[2, 0] += d.Z * d.X; cov[2, 1] += d.Z * d.Y; cov[2, 2] += d.Z * d.Z;
                }

                // PCA: smallest eigenvector is normal
                var evd = cov.Evd();
                var eigenVectors = evd.EigenVectors;
                var eigenValues = evd.EigenValues.Select(c => c.Real).ToArray();

                int minIdx = Array.IndexOf(eigenValues, eigenValues.Min());
                var normal = new System.Numerics.Vector3(
                    (float)eigenVectors[0, minIdx],
                    (float)eigenVectors[1, minIdx],
                    (float)eigenVectors[2, minIdx]
                );

                // Consistent orientation (point towards centroid)
                var toCenter = -new System.Numerics.Vector3(point.X, point.Y, point.Z);
                if (System.Numerics.Vector3.Dot(normal, toCenter) < 0)
                    normal = -normal;

                normals[i] = System.Numerics.Vector3.Normalize(normal);
            });

            return new PointCloudData
            {
                Points = cloud.Points,
                Colors = cloud.Colors,
                Normals = normals
            };
        }

        #region KD-Tree Operations

        private class KdNode
        {
            public int PointIndex;
            public int SplitAxis;
            public float SplitValue;
            public KdNode? Left;
            public KdNode? Right;
        }

        private KdNode? BuildKdTree(System.Numerics.Vector3[] points, int depth = 0, int[]? indices = null)
        {
            indices ??= Enumerable.Range(0, points.Length).ToArray();
            if (indices.Length == 0) return null;

            int axis = depth % 3;
            var sorted = indices.OrderBy(i => GetAxis(points[i], axis)).ToArray();
            int mid = sorted.Length / 2;

            return new KdNode
            {
                PointIndex = sorted[mid],
                SplitAxis = axis,
                SplitValue = GetAxis(points[sorted[mid]], axis),
                Left = BuildKdTree(points, depth + 1, sorted.Take(mid).ToArray()),
                Right = BuildKdTree(points, depth + 1, sorted.Skip(mid + 1).ToArray())
            };
        }

        private float GetAxis(System.Numerics.Vector3 p, int axis) => axis switch
        {
            0 => p.X,
            1 => p.Y,
            _ => p.Z
        };

        private float GetAxis(Vector3 p, int axis) => axis switch
        {
            0 => p.X,
            1 => p.Y,
            _ => p.Z
        };

        private (List<Vector3> srcCorr, List<Vector3> tgtCorr, List<float> distances) FindCorrespondences(
            List<Vector3> source, System.Numerics.Vector3[] target, KdNode? kdTree)
        {
            var srcCorr = new List<Vector3>();
            var tgtCorr = new List<Vector3>();
            var distances = new List<float>();

            foreach (var p in source)
            {
                var nearest = FindNearestPoint(p, target, kdTree);
                float dist = (p - nearest).Length;

                if (dist < MaxCorrespondenceDistance)
                {
                    srcCorr.Add(p);
                    tgtCorr.Add(nearest);
                    distances.Add(dist);
                }
            }

            return (srcCorr, tgtCorr, distances);
        }

        private Vector3 FindNearestPoint(Vector3 query, System.Numerics.Vector3[] points, KdNode? node)
        {
            if (node == null || points.Length == 0)
                return Vector3.Zero;

            Vector3 best = new Vector3(points[node.PointIndex].X, points[node.PointIndex].Y, points[node.PointIndex].Z);
            float bestDist = (query - best).LengthSquared;

            SearchKdTree(query, points, node, ref best, ref bestDist);
            return best;
        }

        private void SearchKdTree(Vector3 query, System.Numerics.Vector3[] points, KdNode node, ref Vector3 best, ref float bestDist)
        {
            var p = new Vector3(points[node.PointIndex].X, points[node.PointIndex].Y, points[node.PointIndex].Z);
            float dist = (query - p).LengthSquared;

            if (dist < bestDist)
            {
                bestDist = dist;
                best = p;
            }

            float queryAxis = GetAxis(query, node.SplitAxis);
            float diff = queryAxis - node.SplitValue;

            var first = diff < 0 ? node.Left : node.Right;
            var second = diff < 0 ? node.Right : node.Left;

            if (first != null)
                SearchKdTree(query, points, first, ref best, ref bestDist);

            if (second != null && diff * diff < bestDist)
                SearchKdTree(query, points, second, ref best, ref bestDist);
        }

        private List<float> FindKNearestDistances(Vector3 query, System.Numerics.Vector3[] points, KdNode? node, int k)
        {
            var heap = new SortedList<float, int>();
            SearchKdTreeKNearest(query, points, node, k, heap);
            return heap.Keys.ToList();
        }

        private List<Vector3> FindKNearestPoints(Vector3 query, System.Numerics.Vector3[] points, KdNode? node, int k)
        {
            var heap = new SortedList<float, int>();
            SearchKdTreeKNearest(query, points, node, k, heap);
            return heap.Values.Select(i => new Vector3(points[i].X, points[i].Y, points[i].Z)).ToList();
        }

        private void SearchKdTreeKNearest(Vector3 query, System.Numerics.Vector3[] points, KdNode? node, int k, SortedList<float, int> heap)
        {
            if (node == null) return;

            var p = new Vector3(points[node.PointIndex].X, points[node.PointIndex].Y, points[node.PointIndex].Z);
            float dist = (query - p).Length;

            // Use a small offset to handle duplicate distances
            while (heap.ContainsKey(dist))
                dist += 0.0001f;

            if (heap.Count < k)
            {
                heap.Add(dist, node.PointIndex);
            }
            else if (dist < heap.Keys.Last())
            {
                heap.RemoveAt(heap.Count - 1);
                heap.Add(dist, node.PointIndex);
            }

            float queryAxis = GetAxis(query, node.SplitAxis);
            float diff = queryAxis - node.SplitValue;
            float maxDist = heap.Count < k ? float.MaxValue : heap.Keys.Last();

            var first = diff < 0 ? node.Left : node.Right;
            var second = diff < 0 ? node.Right : node.Left;

            if (first != null)
                SearchKdTreeKNearest(query, points, first, k, heap);

            maxDist = heap.Count < k ? float.MaxValue : heap.Keys.Last();
            if (second != null && Math.Abs(diff) < maxDist)
                SearchKdTreeKNearest(query, points, second, k, heap);
        }

        #endregion

        #region Point Cloud Features (FPFH-like)

        /// <summary>
        /// Compute local feature descriptors for keypoints.
        /// Simplified FPFH (Fast Point Feature Histograms).
        /// </summary>
        public float[][] ComputeFeatures(PointCloudData cloud, int kNeighbors = 30)
        {
            // Ensure normals exist
            if (cloud.Normals == null)
                cloud = EstimateNormals(cloud, kNeighbors);

            int n = cloud.Points.Length;
            var features = new float[n][];
            var kdTree = BuildKdTree(cloud.Points);

            Parallel.For(0, n, i =>
            {
                var p = cloud.Points[i];
                var normal = cloud.Normals![i];
                var point = new Vector3(p.X, p.Y, p.Z);
                var neighbors = FindKNearestPoints(point, cloud.Points, kdTree, kNeighbors);

                // Compute SPFH (Simplified Point Feature Histogram)
                var histogram = new float[33]; // 11 bins x 3 features

                foreach (var neighbor in neighbors)
                {
                    if ((neighbor - point).LengthSquared < 1e-10f) continue;

                    var diff = neighbor - point;
                    diff.Normalize();

                    // Compute features (alpha, phi, theta)
                    float alpha = (float)Math.Acos(Math.Clamp(Vector3.Dot(
                        new Vector3(normal.X, normal.Y, normal.Z), diff), -1, 1));
                    float phi = (float)Math.Atan2(diff.Y, diff.X);
                    float theta = (float)Math.Acos(Math.Clamp(diff.Z, -1, 1));

                    // Bin the features
                    int alphaIdx = Math.Clamp((int)(alpha / Math.PI * 10), 0, 10);
                    int phiIdx = Math.Clamp((int)((phi + Math.PI) / (2 * Math.PI) * 10), 0, 10);
                    int thetaIdx = Math.Clamp((int)(theta / Math.PI * 10), 0, 10);

                    histogram[alphaIdx]++;
                    histogram[11 + phiIdx]++;
                    histogram[22 + thetaIdx]++;
                }

                // Normalize
                float sum = histogram.Sum();
                if (sum > 0)
                    for (int j = 0; j < histogram.Length; j++)
                        histogram[j] /= sum;

                features[i] = histogram;
            });

            return features;
        }

        /// <summary>
        /// Feature-based registration using RANSAC on feature correspondences.
        /// </summary>
        public async Task<Matrix4> FeatureBasedRegistrationAsync(PointCloudData source, PointCloudData target)
        {
            return await Task.Run(() =>
            {
                _progressCallback?.Invoke("Computing features for source cloud...");
                var sourceFeatures = ComputeFeatures(source);

                _progressCallback?.Invoke("Computing features for target cloud...");
                var targetFeatures = ComputeFeatures(target);

                _progressCallback?.Invoke("Finding feature correspondences...");
                var correspondences = FindFeatureCorrespondences(source, target, sourceFeatures, targetFeatures);

                if (correspondences.Count < 4)
                {
                    _progressCallback?.Invoke("Not enough correspondences, falling back to ICP");
                    return Matrix4.Identity;
                }

                _progressCallback?.Invoke($"Found {correspondences.Count} correspondences, running RANSAC...");

                var srcPoints = correspondences.Select(c => new Vector3(
                    source.Points[c.srcIdx].X, source.Points[c.srcIdx].Y, source.Points[c.srcIdx].Z)).ToList();
                var tgtPoints = correspondences.Select(c => new Vector3(
                    target.Points[c.tgtIdx].X, target.Points[c.tgtIdx].Y, target.Points[c.tgtIdx].Z)).ToList();

                return GeometryUtils.ComputeRigidTransformRANSAC(srcPoints, tgtPoints, out _, out _,
                    maxIterations: 1000, inlierThreshold: MaxCorrespondenceDistance);
            });
        }

        private List<(int srcIdx, int tgtIdx, float distance)> FindFeatureCorrespondences(
            PointCloudData source, PointCloudData target, float[][] srcFeatures, float[][] tgtFeatures)
        {
            var correspondences = new List<(int srcIdx, int tgtIdx, float distance)>();

            for (int i = 0; i < srcFeatures.Length; i++)
            {
                float bestDist = float.MaxValue;
                float secondBest = float.MaxValue;
                int bestIdx = -1;

                for (int j = 0; j < tgtFeatures.Length; j++)
                {
                    float dist = FeatureDistance(srcFeatures[i], tgtFeatures[j]);
                    if (dist < bestDist)
                    {
                        secondBest = bestDist;
                        bestDist = dist;
                        bestIdx = j;
                    }
                    else if (dist < secondBest)
                    {
                        secondBest = dist;
                    }
                }

                // Lowe's ratio test
                if (bestIdx >= 0 && bestDist < 0.8f * secondBest)
                {
                    correspondences.Add((i, bestIdx, bestDist));
                }
            }

            return correspondences;
        }

        private float FeatureDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            return (float)Math.Sqrt(sum);
        }

        #endregion
    }

    /// <summary>
    /// Point cloud data structure.
    /// </summary>
    public class PointCloudData
    {
        public System.Numerics.Vector3[] Points { get; set; } = Array.Empty<System.Numerics.Vector3>();
        public System.Numerics.Vector3[]? Colors { get; set; }
        public System.Numerics.Vector3[]? Normals { get; set; }

        public PointCloudData Clone()
        {
            return new PointCloudData
            {
                Points = Points.ToArray(),
                Colors = Colors?.ToArray(),
                Normals = Normals?.ToArray()
            };
        }
    }
}
