using System;
using System.Collections.Generic;
using System.Linq;
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

        public static int RemoveDuplicates(PointCloudObject pointCloud, float threshold)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.RemoveDuplicates(ToData(pointCloud), Math.Max(0.00001f, threshold));
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static void EstimateNormals(PointCloudObject pointCloud, int kNeighbors)
        {
            var merger = new PointCloudMerger();
            var withNormals = merger.EstimateNormals(ToData(pointCloud), Math.Max(3, kNeighbors));
            ApplyData(pointCloud, withNormals);
        }

        public static int PassThroughAxis(PointCloudObject pointCloud, int axis, float minValue, float maxValue)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.PassThroughAxis(ToData(pointCloud), axis, minValue, maxValue);
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
        }

        public static int RadiusCrop(PointCloudObject pointCloud, Vector3 center, float radius)
        {
            var merger = new PointCloudMerger();
            var before = pointCloud.Points.Count;
            var filtered = merger.RadiusCrop(ToData(pointCloud), center, radius);
            ApplyData(pointCloud, filtered);
            return before - pointCloud.Points.Count;
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

            // Auto-scale radius to scene size so dense cloud works on large-coordinate reconstructions.
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
                int take = Math.Min(pointsPerSeed, nearest.Count);
                for (int n = 0; n < take && added < maxAdded; n++)
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
                    added++;
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
