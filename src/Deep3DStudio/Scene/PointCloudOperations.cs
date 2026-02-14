using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    public sealed class PointCloudBlueFilterOptions
    {
        public float MinBlue { get; set; } = 0.45f;
        public float MaxRed { get; set; } = 0.60f;
        public float MaxGreen { get; set; } = 0.75f;
        public float MinBlueDominance { get; set; } = 0.08f;
    }

    /// <summary>
    /// High-level point cloud editing helpers that operate on scene objects.
    /// </summary>
    public static class PointCloudOperations
    {
        public static int VoxelDownsample(PointCloudObject pointCloud, float voxelSize)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.VoxelDownsample(ToData(pointCloud), Math.Max(0.0001f, voxelSize));
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static async Task<int> VoxelDownsampleAsync(
            PointCloudObject pointCloud,
            float voxelSize,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                progress?.Report("Building voxel grid...");
                var merger = new PointCloudMerger();
                var before = pointCloud.Points.Count;
                var data = ToData(pointCloud);
                var filtered = merger.VoxelDownsample(data, Math.Max(0.0001f, voxelSize));
                ApplyData(pointCloud, filtered);
                progress?.Report($"Voxel downsample removed {before - pointCloud.Points.Count:N0} points");
                return before - pointCloud.Points.Count;
            }, cancellationToken);
        }

        public static int RemoveStatisticalOutliers(PointCloudObject pointCloud, int kNeighbors, float stdRatio)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.RemoveStatisticalOutliers(
                ToData(pointCloud),
                Math.Max(2, kNeighbors),
                Math.Max(0.1f, stdRatio));
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static async Task<int> RemoveStatisticalOutliersAsync(
            PointCloudObject pointCloud,
            int kNeighbors,
            float stdRatio,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                progress?.Report("Building KD-tree for outlier detection...");
                var merger = new PointCloudMerger();
                var before = pointCloud.Points.Count;
                var filtered = merger.RemoveStatisticalOutliers(
                    ToData(pointCloud),
                    Math.Max(2, kNeighbors),
                    Math.Max(0.1f, stdRatio));
                ApplyData(pointCloud, filtered);
                progress?.Report($"Outlier filter removed {before - pointCloud.Points.Count:N0} points");
                return before - pointCloud.Points.Count;
            }, cancellationToken);
        }

        public static int RemoveDuplicates(PointCloudObject pointCloud, float threshold)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.RemoveDuplicates(ToData(pointCloud), Math.Max(0.00001f, threshold));
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static async Task<int> RemoveDuplicatesAsync(
            PointCloudObject pointCloud,
            float threshold,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                progress?.Report("Finding duplicate points...");
                var merger = new PointCloudMerger();
                var before = pointCloud.Points.Count;
                var filtered = merger.RemoveDuplicates(ToData(pointCloud), Math.Max(0.00001f, threshold));
                ApplyData(pointCloud, filtered);
                progress?.Report($"Duplicate removal removed {before - pointCloud.Points.Count:N0} points");
                return before - pointCloud.Points.Count;
            }, cancellationToken);
        }

        public static void EstimateNormals(PointCloudObject pointCloud, int kNeighbors)
        {
            var merger = new PointCloudMerger();
            var withNormals = merger.EstimateNormals(ToData(pointCloud), Math.Max(3, kNeighbors));
            ApplyData(pointCloud, withNormals);
        }

        public static async Task EstimateNormalsAsync(
            PointCloudObject pointCloud,
            int kNeighbors,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            await Task.Run(() =>
            {
                progress?.Report("Building KD-tree for normal estimation...");
                var merger = new PointCloudMerger();
                var withNormals = merger.EstimateNormals(ToData(pointCloud), Math.Max(3, kNeighbors));
                ApplyData(pointCloud, withNormals);
                progress?.Report($"Estimated normals for {pointCloud.Points.Count:N0} points");
            }, cancellationToken);
        }

        public static int PassThroughAxis(PointCloudObject pointCloud, int axis, float minValue, float maxValue)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.PassThroughAxis(ToData(pointCloud), axis, minValue, maxValue);
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static async Task<int> PassThroughAxisAsync(
            PointCloudObject pointCloud,
            int axis,
            float minValue,
            float maxValue,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                progress?.Report("Filtering points by axis...");
                var merger = new PointCloudMerger();
                var before = pointCloud.Points.Count;
                var filtered = merger.PassThroughAxis(ToData(pointCloud), axis, minValue, maxValue);
                ApplyData(pointCloud, filtered);
                progress?.Report($"Pass-through removed {before - pointCloud.Points.Count:N0} points");
                return before - pointCloud.Points.Count;
            }, cancellationToken);
        }

        public static int RadiusCrop(PointCloudObject pointCloud, Vector3 center, float radius)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.RadiusCrop(ToData(pointCloud), center, radius);
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static async Task<int> RadiusCropAsync(
            PointCloudObject pointCloud,
            Vector3 center,
            float radius,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                progress?.Report("Filtering points by radius...");
                var merger = new PointCloudMerger();
                var before = pointCloud.Points.Count;
                var filtered = merger.RadiusCrop(ToData(pointCloud), center, radius);
                ApplyData(pointCloud, filtered);
                progress?.Report($"Radius crop removed {before - pointCloud.Points.Count:N0} points");
                return before - pointCloud.Points.Count;
            }, cancellationToken);
        }

        public static int Densify(PointCloudObject pointCloud, float neighborRadius, int pointsPerSeed)
        {
            if (pointCloud.Points.Count < 2)
                return 0;

            neighborRadius = Math.Max(0.0001f, neighborRadius);
            pointsPerSeed = Math.Clamp(pointsPerSeed, 1, 8);

            var sourcePoints = pointCloud.Points.ToArray();
            var sourceColors = pointCloud.Colors.Count == sourcePoints.Length ? pointCloud.Colors.ToArray() : null;
            var sourceNormals = pointCloud.Normals.Count == sourcePoints.Length ? pointCloud.Normals.ToArray() : null;
            var sourceConfidence = pointCloud.Confidence.Count == sourcePoints.Length ? pointCloud.Confidence.ToArray() : null;

            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            for (int i = 0; i < sourcePoints.Length; i++)
            {
                min = Vector3.ComponentMin(min, sourcePoints[i]);
                max = Vector3.ComponentMax(max, sourcePoints[i]);
            }

            float diagonal = (max - min).Length;
            if (diagonal > 0f)
            {
                // Estimate average spacing from volumetric density
                float volume = (max.X - min.X) * (max.Y - min.Y) * (max.Z - min.Z);
                float estimatedSpacing = volume > 0 ? (float)Math.Pow(volume / sourcePoints.Length, 1.0 / 3.0) : diagonal * 0.01f;

                // Use the larger of: 1% of diagonal or 2x estimated spacing
                float adaptiveMinRadius = Math.Max(0.0001f, Math.Max(diagonal * 0.01f, estimatedSpacing * 2.0f));
                if (neighborRadius < adaptiveMinRadius)
                {
                    neighborRadius = adaptiveMinRadius;
                }
            }

            float cellSize = neighborRadius;
            var buckets = new Dictionary<(int x, int y, int z), List<int>>();
            for (int i = 0; i < sourcePoints.Length; i++)
            {
                var p = sourcePoints[i];
                var key = (
                    (int)Math.Floor(p.X / cellSize),
                    (int)Math.Floor(p.Y / cellSize),
                    (int)Math.Floor(p.Z / cellSize));
                if (!buckets.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    buckets[key] = list;
                }
                list.Add(i);
            }

            var outPoints = new List<Vector3>(sourcePoints.Length * (pointsPerSeed + 1));
            var outColors = sourceColors != null ? new List<Vector3>(sourcePoints.Length * (pointsPerSeed + 1)) : null;
            var outNormals = sourceNormals != null ? new List<Vector3>(sourcePoints.Length * (pointsPerSeed + 1)) : null;
            var outConfidence = new List<float>(sourcePoints.Length * (pointsPerSeed + 1));

            for (int i = 0; i < sourcePoints.Length; i++)
            {
                outPoints.Add(sourcePoints[i]);
                if (outColors != null && sourceColors != null) outColors.Add(sourceColors[i]);
                if (outNormals != null && sourceNormals != null) outNormals.Add(sourceNormals[i]);
                outConfidence.Add(sourceConfidence != null ? sourceConfidence[i] : 1.0f);
            }

            int maxAdded = sourcePoints.Length * pointsPerSeed;
            int added = 0;

            for (int i = 0; i < sourcePoints.Length && added < maxAdded; i++)
            {
                var p = sourcePoints[i];
                int bx = (int)Math.Floor(p.X / cellSize);
                int by = (int)Math.Floor(p.Y / cellSize);
                int bz = (int)Math.Floor(p.Z / cellSize);

                var nearest = new List<(int idx, float dist2)>();
                for (int dz = -1; dz <= 1; dz++)
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            var key = (bx + dx, by + dy, bz + dz);
                            if (!buckets.TryGetValue(key, out var candidates))
                                continue;

                            foreach (var c in candidates)
                            {
                                if (c == i) continue;
                                var d = sourcePoints[c] - p;
                                float d2 = d.LengthSquared;
                                if (d2 <= 1e-12f || d2 > neighborRadius * neighborRadius) continue;
                                nearest.Add((c, d2));
                            }
                        }
                    }
                }

                if (nearest.Count == 0)
                    continue;

                nearest.Sort((a, b) => a.dist2.CompareTo(b.dist2));
                // Take up to 3 nearest neighbors, create multiple interpolated points per pair
                int neighborsTake = Math.Min(3, nearest.Count);
                int pointsPerNeighbor = Math.Max(1, pointsPerSeed / Math.Max(1, neighborsTake));
                for (int n = 0; n < neighborsTake && added < maxAdded; n++)
                {
                    int j = nearest[n].idx;
                    var q = sourcePoints[j];

                    // Create uniformly distributed interpolated points along the segment
                    for (int s = 0; s < pointsPerNeighbor && added < maxAdded; s++)
                    {
                        float t = (s + 1.0f) / (pointsPerNeighbor + 1.0f);
                        var interpolated = p + t * (q - p);

                        outPoints.Add(interpolated);
                        if (outColors != null && sourceColors != null)
                        {
                            outColors.Add(sourceColors[i] * (1.0f - t) + sourceColors[j] * t);
                        }
                        if (outNormals != null && sourceNormals != null)
                        {
                            var avg = sourceNormals[i] * (1.0f - t) + sourceNormals[j] * t;
                            if (avg.LengthSquared > 1e-10f)
                                avg.Normalize();
                            outNormals.Add(avg);
                        }
                        outConfidence.Add(sourceConfidence != null ? sourceConfidence[i] * (1.0f - t) + sourceConfidence[j] * t : 1.0f);
                        added++;
                    }
                }
            }

            pointCloud.Points = outPoints;
            if (outColors != null)
            {
                pointCloud.Colors = outColors;
            }
            else
            {
                pointCloud.Colors = new List<Vector3>(pointCloud.Points.Count);
                for (int i = 0; i < pointCloud.Points.Count; i++)
                    pointCloud.Colors.Add(new Vector3(0.85f));
            }

            if (outNormals != null)
            {
                pointCloud.Normals = outNormals;
            }
            else
            {
                pointCloud.Normals = new List<Vector3>();
            }
            pointCloud.Confidence = outConfidence;
            pointCloud.UpdateBounds();
            return added;
        }

        public static async Task<int> DensifyAsync(
            PointCloudObject pointCloud,
            float neighborRadius,
            int pointsPerSeed,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null,
            IProgress<float>? progressFactor = null)
        {
            return await Task.Run(() =>
            {
                if (pointCloud.Points.Count < 2)
                    return 0;

                progress?.Report("Computing scene bounds...");
                progressFactor?.Report(0.05f);
                neighborRadius = Math.Max(0.0001f, neighborRadius);
                pointsPerSeed = Math.Clamp(pointsPerSeed, 1, 8);

                var sourcePoints = pointCloud.Points.ToArray();
                var sourceColors = pointCloud.Colors.Count == sourcePoints.Length ? pointCloud.Colors.ToArray() : null;
                var sourceNormals = pointCloud.Normals.Count == sourcePoints.Length ? pointCloud.Normals.ToArray() : null;
                var sourceConfidence = pointCloud.Confidence.Count == sourcePoints.Length ? pointCloud.Confidence.ToArray() : null;

                var min = new Vector3(float.MaxValue);
                var max = new Vector3(float.MinValue);
                for (int i = 0; i < sourcePoints.Length; i++)
                {
                    min = Vector3.ComponentMin(min, sourcePoints[i]);
                    max = Vector3.ComponentMax(max, sourcePoints[i]);
                }

                float diagonal = (max - min).Length;
                if (diagonal > 0f)
                {
                    float adaptiveMinRadius = Math.Max(0.0001f, diagonal * 0.005f);
                    if (neighborRadius < adaptiveMinRadius)
                    {
                        neighborRadius = adaptiveMinRadius;
                    }
                }

                progress?.Report("Building spatial grid...");
                progressFactor?.Report(0.1f);
                float cellSize = neighborRadius;
                var buckets = new ConcurrentDictionary<(int x, int y, int z), ConcurrentBag<int>>();

                Parallel.For(0, sourcePoints.Length, new ParallelOptions { CancellationToken = cancellationToken }, i =>
                {
                    var p = sourcePoints[i];
                    var key = (
                        (int)Math.Floor(p.X / cellSize),
                        (int)Math.Floor(p.Y / cellSize),
                        (int)Math.Floor(p.Z / cellSize));
                    buckets.AddOrUpdate(key, _ => new ConcurrentBag<int> { i }, (_, bag) => { bag.Add(i); return bag; });
                });

                progress?.Report("Creating dense points...");
                progressFactor?.Report(0.2f);
                var outPoints = new ConcurrentBag<Vector3>();
                var outColors = sourceColors != null ? new ConcurrentBag<Vector3>() : null;
                var outNormals = sourceNormals != null ? new ConcurrentBag<Vector3>() : null;
                var outConfidence = new ConcurrentBag<float>();

                var reportInterval = Math.Max(1, sourcePoints.Length / 100);
                int lastReport = 0;
                int processedCount = 0;

                Parallel.For(0, sourcePoints.Length, new ParallelOptions { CancellationToken = cancellationToken }, i =>
                {
                    var p = sourcePoints[i];
                    int bx = (int)Math.Floor(p.X / cellSize);
                    int by = (int)Math.Floor(p.Y / cellSize);
                    int bz = (int)Math.Floor(p.Z / cellSize);

                    var nearest = new List<(int idx, float dist2)>();
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                var key = (bx + dx, by + dy, bz + dz);
                                if (!buckets.TryGetValue(key, out var candidates))
                                    continue;

                                foreach (var c in candidates)
                                {
                                    if (c == i) continue;
                                    var d = sourcePoints[c] - p;
                                    float d2 = d.LengthSquared;
                                    if (d2 <= 1e-12f || d2 > neighborRadius * neighborRadius) continue;
                                    nearest.Add((c, d2));
                                }
                            }
                        }
                    }

                    outPoints.Add(p);
                    if (outColors != null && sourceColors != null) outColors.Add(sourceColors[i]);
                    if (outNormals != null && sourceNormals != null) outNormals.Add(sourceNormals[i]);
                    outConfidence.Add(sourceConfidence != null ? sourceConfidence[i] : 1.0f);

                    if (nearest.Count > 0)
                    {
                        nearest.Sort((a, b) => a.dist2.CompareTo(b.dist2));
                        int take = Math.Min(pointsPerSeed, nearest.Count);
                        for (int n = 0; n < take; n++)
                        {
                            int j = nearest[n].idx;
                            var q = sourcePoints[j];
                            var mid = (p + q) * 0.5f;

                            outPoints.Add(mid);
                            if (outColors != null && sourceColors != null)
                            {
                                outColors.Add((sourceColors[i] + sourceColors[j]) * 0.5f);
                            }
                            if (outNormals != null && sourceNormals != null)
                            {
                                var avg = sourceNormals[i] + sourceNormals[j];
                                if (avg.LengthSquared > 1e-10f)
                                    avg.Normalize();
                                outNormals.Add(avg);
                            }
                            outConfidence.Add(sourceConfidence != null ? (sourceConfidence[i] + sourceConfidence[j]) * 0.5f : 1.0f);
                        }
                    }

                    int currentProcessed = Interlocked.Increment(ref processedCount);
                    if (currentProcessed % reportInterval == 0)
                    {
                        float pFactor = 0.2f + (currentProcessed / (float)sourcePoints.Length) * 0.75f;
                        progressFactor?.Report(pFactor);
                        progress?.Report($"Densifying... {currentProcessed * 100 / sourcePoints.Length}% ({outPoints.Count:N0} total points)");
                    }
                });

                progress?.Report("Finalizing point cloud...");
                progressFactor?.Report(0.95f);
                var finalPoints = outPoints.ToList();
                var finalColors = outColors?.ToList();
                var finalNormals = outNormals?.ToList();
                var finalConfidence = outConfidence.ToList();

                pointCloud.Points = finalPoints;
                if (finalColors != null)
                {
                    pointCloud.Colors = finalColors;
                }
                else
                {
                    pointCloud.Colors = new List<Vector3>(pointCloud.Points.Count);
                    for (int i = 0; i < pointCloud.Points.Count; i++)
                        pointCloud.Colors.Add(new Vector3(0.85f));
                }

                if (finalNormals != null)
                {
                    pointCloud.Normals = finalNormals;
                }
                else
                {
                    pointCloud.Normals = new List<Vector3>();
                }
                pointCloud.Confidence = finalConfidence;
                pointCloud.UpdateBounds();

                progressFactor?.Report(1.0f);
                progress?.Report($"Dense cloud added {finalPoints.Count - sourcePoints.Length:N0} points");
                return finalPoints.Count - sourcePoints.Length;
            }, cancellationToken);
        }

        public static int RemoveBlueDominantPoints(PointCloudObject pointCloud, PointCloudBlueFilterOptions options)
        {
            if (pointCloud.Points.Count == 0)
                return 0;

            options ??= new PointCloudBlueFilterOptions();

            float minBlue = Math.Clamp(options.MinBlue, 0.0f, 1.0f);
            float maxRed = Math.Clamp(options.MaxRed, 0.0f, 1.0f);
            float maxGreen = Math.Clamp(options.MaxGreen, 0.0f, 1.0f);
            float minBlueDominance = Math.Max(0.0f, options.MinBlueDominance);

            int before = pointCloud.Points.Count;
            bool hasAlignedNormals = pointCloud.Normals.Count == before;
            bool hasAlignedConfidence = pointCloud.Confidence.Count == before;

            var outPoints = new List<Vector3>(before);
            var outColors = new List<Vector3>(before);
            var outNormals = hasAlignedNormals ? new List<Vector3>(before) : new List<Vector3>();
            var outConfidence = new List<float>(before);

            int removed = 0;

            for (int i = 0; i < before; i++)
            {
                bool hasColor = i < pointCloud.Colors.Count;
                var color = hasColor ? pointCloud.Colors[i] : new Vector3(0.85f);

                bool remove = false;
                if (hasColor)
                {
                    float blue = color.Z;
                    float red = color.X;
                    float green = color.Y;
                    float dominance = blue - Math.Max(red, green);

                    remove = blue >= minBlue &&
                             red <= maxRed &&
                             green <= maxGreen &&
                             dominance >= minBlueDominance;
                }

                if (remove)
                {
                    removed++;
                    continue;
                }

                outPoints.Add(pointCloud.Points[i]);
                outColors.Add(color);

                if (hasAlignedNormals)
                    outNormals.Add(pointCloud.Normals[i]);

                if (hasAlignedConfidence)
                    outConfidence.Add(pointCloud.Confidence[i]);
                else
                    outConfidence.Add(1.0f);
            }

            pointCloud.Points = outPoints;
            pointCloud.Colors = outColors;
            pointCloud.Normals = outNormals;
            pointCloud.Confidence = outConfidence;
            pointCloud.UpdateBounds();
            return removed;
        }

        public static async Task<int> RemoveBlueDominantPointsAsync(
            PointCloudObject pointCloud,
            PointCloudBlueFilterOptions options,
            CancellationToken cancellationToken = default,
            IProgress<string>? progress = null)
        {
            return await Task.Run(() =>
            {
                if (pointCloud.Points.Count == 0)
                    return 0;

                progress?.Report("Filtering blue-dominant points...");
                options ??= new PointCloudBlueFilterOptions();

                float minBlue = Math.Clamp(options.MinBlue, 0.0f, 1.0f);
                float maxRed = Math.Clamp(options.MaxRed, 0.0f, 1.0f);
                float maxGreen = Math.Clamp(options.MaxGreen, 0.0f, 1.0f);
                float minBlueDominance = Math.Max(0.0f, options.MinBlueDominance);

                int before = pointCloud.Points.Count;
                bool hasAlignedNormals = pointCloud.Normals.Count == before;
                bool hasAlignedConfidence = pointCloud.Confidence.Count == before;

                var outPoints = new ConcurrentBag<Vector3>();
                var outColors = new ConcurrentBag<Vector3>();
                var outNormals = hasAlignedNormals ? new ConcurrentBag<Vector3>() : null;
                var outConfidence = new ConcurrentBag<float>();

                var reportInterval = Math.Max(1, before / 20);
                int removed = 0;
                int lastReport = 0;

                Parallel.For(0, before, new ParallelOptions { CancellationToken = cancellationToken }, i =>
                {
                    bool hasColor = i < pointCloud.Colors.Count;
                    var color = hasColor ? pointCloud.Colors[i] : new Vector3(0.85f);

                    bool remove = false;
                    if (hasColor)
                    {
                        float blue = color.Z;
                        float red = color.X;
                        float green = color.Y;
                        float dominance = blue - Math.Max(red, green);

                        remove = blue >= minBlue &&
                                 red <= maxRed &&
                                 green <= maxGreen &&
                                 dominance >= minBlueDominance;
                    }

                    if (!remove)
                    {
                        outPoints.Add(pointCloud.Points[i]);
                        outColors.Add(color);

                        if (hasAlignedNormals && outNormals != null)
                            outNormals.Add(pointCloud.Normals[i]);

                        if (hasAlignedConfidence)
                            outConfidence.Add(pointCloud.Confidence[i]);
                        else
                            outConfidence.Add(1.0f);
                    }
                    else
                    {
                        Interlocked.Increment(ref removed);
                    }

                    if (Interlocked.CompareExchange(ref lastReport, i, lastReport) == lastReport && i % reportInterval == 0)
                    {
                        progress?.Report($"Filtering blue-dominant points... {i * 100 / before}%");
                    }
                });

                var finalPoints = outPoints.ToList();
                var finalColors = outColors.ToList();
                var finalNormals = outNormals?.ToList();
                var finalConfidence = outConfidence.ToList();

                pointCloud.Points = finalPoints;
                pointCloud.Colors = finalColors;
                pointCloud.Normals = finalNormals ?? new List<Vector3>();
                pointCloud.Confidence = finalConfidence;
                pointCloud.UpdateBounds();

                progress?.Report($"Sky/blue filter removed {removed:N0} points");
                return removed;
            }, cancellationToken);
        }

        public static PointCloudData ToData(PointCloudObject pointCloud)
        {
            return new PointCloudData
            {
                Points = pointCloud.Points.ConvertAll(ToNumerics).ToArray(),
                Colors = pointCloud.Colors.Count == pointCloud.Points.Count
                    ? pointCloud.Colors.ConvertAll(ToNumerics).ToArray()
                    : null,
                Normals = pointCloud.Normals.Count == pointCloud.Points.Count
                    ? pointCloud.Normals.ConvertAll(ToNumerics).ToArray()
                    : null
            };
        }

        public static MeshData ToMeshData(PointCloudObject pointCloud, bool visibleOnly = false)
        {
            var mesh = new MeshData();
            int totalPoints = pointCloud.Points.Count;
            if (totalPoints == 0)
                return mesh;

            int visibleCount = visibleOnly ? pointCloud.GetVisiblePointCount() : totalPoints;
            if (visibleCount <= 0)
                return mesh;

            bool fullCloud = visibleCount >= totalPoints;
            bool hasFullColors = pointCloud.Colors.Count >= totalPoints;

            bool hasFullNormals = pointCloud.Normals.Count >= totalPoints;

            if (fullCloud)
            {
                mesh.Vertices.AddRange(pointCloud.Points);
                if (hasFullColors)
                {
                    mesh.Colors.AddRange(pointCloud.Colors.Take(totalPoints));
                }
                else
                {
                    for (int i = 0; i < totalPoints; i++)
                    {
                        if (i < pointCloud.Colors.Count) mesh.Colors.Add(pointCloud.Colors[i]);
                        else mesh.Colors.Add(new Vector3(1f, 1f, 1f));
                    }
                }
                if (hasFullNormals)
                {
                    mesh.Normals.AddRange(pointCloud.Normals.Take(totalPoints));
                }
                if (pointCloud.Confidence.Count >= totalPoints)
                {
                    mesh.Confidence.AddRange(pointCloud.Confidence.Take(totalPoints));
                }
                else
                {
                    for (int i = 0; i < totalPoints; i++)
                        mesh.Confidence.Add(1.0f);
                }

                return mesh;
            }

            for (int i = 0; i < visibleCount; i++)
            {
                int sourceIndex = pointCloud.GetSourcePointIndex(i, visibleCount);
                if (sourceIndex < 0 || sourceIndex >= totalPoints)
                    continue;

                mesh.Vertices.Add(pointCloud.Points[sourceIndex]);
                if (sourceIndex < pointCloud.Colors.Count) mesh.Colors.Add(pointCloud.Colors[sourceIndex]);
                else mesh.Colors.Add(new Vector3(1f, 1f, 1f));
                if (hasFullNormals && sourceIndex < pointCloud.Normals.Count)
                    mesh.Normals.Add(pointCloud.Normals[sourceIndex]);
                if (sourceIndex < pointCloud.Confidence.Count) mesh.Confidence.Add(pointCloud.Confidence[sourceIndex]);
                else mesh.Confidence.Add(1.0f);
            }

            return mesh;
        }

        public static void ApplyData(PointCloudObject pointCloud, PointCloudData data)
        {
            pointCloud.Points = new List<Vector3>(Array.ConvertAll(data.Points, ToOpenTk));

            if (data.Colors != null && data.Colors.Length == data.Points.Length)
            {
                pointCloud.Colors = new List<Vector3>(Array.ConvertAll(data.Colors, ToOpenTk));
            }
            else
            {
                pointCloud.Colors = new List<Vector3>(pointCloud.Points.Count);
                for (int i = 0; i < pointCloud.Points.Count; i++)
                    pointCloud.Colors.Add(new Vector3(0.85f));
            }

            if (data.Normals != null && data.Normals.Length == data.Points.Length)
            {
                pointCloud.Normals = new List<Vector3>(Array.ConvertAll(data.Normals, ToOpenTk));
            }
            else
            {
                pointCloud.Normals = new List<Vector3>();
            }
            pointCloud.Confidence = new List<float>(pointCloud.Points.Count);
            for (int i = 0; i < pointCloud.Points.Count; i++)
                pointCloud.Confidence.Add(1.0f);

            pointCloud.UpdateBounds();
        }

        private static System.Numerics.Vector3 ToNumerics(Vector3 v) => new System.Numerics.Vector3(v.X, v.Y, v.Z);
        private static Vector3 ToOpenTk(System.Numerics.Vector3 v) => new Vector3(v.X, v.Y, v.Z);
    }
}
