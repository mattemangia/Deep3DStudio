using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public static class VoxelizationUtils
    {
        public static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) Voxelize(List<MeshData> meshes, int maxRes, Action<string, float>? progress = null)
        {
            int totalCount = 0;
            foreach (var m in meshes)
                totalCount += m.Vertices.Count;

            if (totalCount == 0)
                return (null, null, Vector3.Zero, 0.02f);

            var points = new Vector3[totalCount];
            var colors = new Vector3[totalCount];
            Vector3[]? normals = null;

            // Check if any mesh has normals
            int totalNormals = 0;
            foreach (var m in meshes)
                totalNormals += m.Normals.Count;

            if (totalNormals >= totalCount)
                normals = new Vector3[totalCount];

            int idx = 0;
            foreach (var m in meshes)
            {
                bool hasColors = m.Colors.Count >= m.Vertices.Count;
                bool hasNormals = normals != null && m.Normals.Count >= m.Vertices.Count;
                for (int i = 0; i < m.Vertices.Count; i++)
                {
                    points[idx] = m.Vertices[i];
                    colors[idx] = hasColors ? m.Colors[i] : new Vector3(0.8f, 0.8f, 0.8f);
                    if (hasNormals)
                        normals![idx] = m.Normals[i];
                    idx++;
                }
            }

            return VoxelizeCoreSDF(points, normals, colors, maxRes, progress);
        }

        public static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) Voxelize(IList<Vector3> points, int maxRes, Action<string, float>? progress = null)
        {
            if (points.Count == 0)
                return (null, null, Vector3.Zero, 0.02f);

            var arr = new Vector3[points.Count];
            for (int i = 0; i < points.Count; i++)
                arr[i] = points[i];

            return VoxelizeCoreSDF(arr, null, null, maxRes, progress);
        }

        private static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) VoxelizeCoreSDF(
            Vector3[] points, Vector3[]? normals, Vector3[]? colors, int maxRes, Action<string, float>? progress = null)
        {
            if (points.Length == 0 || maxRes < 16)
                return (null, null, Vector3.Zero, 0.02f);

            // 1. Compute AABB
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            foreach (var v in points)
            {
                min = Vector3.ComponentMin(min, v);
                max = Vector3.ComponentMax(max, v);
            }

            var size = max - min;
            float maxDim = Math.Max(size.X, Math.Max(size.Y, size.Z));
            if (maxDim <= 0)
                return (null, null, min, 0.02f);

            // 2. Compute voxel size
            float voxelSize = maxDim / Math.Max(2, maxRes - 10);

            // Add padding
            var padding = new Vector3(voxelSize * 5);
            min -= padding;
            max += padding;
            size = max - min;

            int w = Math.Min(maxRes, (int)Math.Ceiling(size.X / voxelSize) + 1);
            int h = Math.Min(maxRes, (int)Math.Ceiling(size.Y / voxelSize) + 1);
            int d = Math.Min(maxRes, (int)Math.Ceiling(size.Z / voxelSize) + 1);

            if (w <= 2 || h <= 2 || d <= 2)
                return (null, null, min, voxelSize);

            Console.WriteLine($"[Voxelize-SDF] Grid: {w}x{h}x{d}, voxelSize={voxelSize:F6}, points={points.Length}");
            progress?.Invoke("Estimating normals...", 0.02f);

            // 3. Estimate normals if not provided
            if (normals == null || normals.Length < points.Length)
            {
                normals = EstimateNormalsPCA(points, 15, progress);
            }

            progress?.Invoke("Building spatial hash...", 0.15f);

            // 4. Build spatial hash grid for fast nearest-neighbor lookup
            float cellSize = voxelSize * 2.0f;
            var spatialHash = BuildSpatialHash(points, min, cellSize);

            // 5. Compute SDF: for each voxel, find nearby points and compute signed distance
            //    using weighted average of normals from multiple nearby points to avoid
            //    sign flickering from inconsistent normals.
            progress?.Invoke("Computing signed distance field...", 0.20f);
            var grid = new float[w, h, d];
            float bandwidth = voxelSize * 3.0f;
            float searchRadiusDist = cellSize * 4.0f; // max search distance
            float searchRadiusDistSq = searchRadiusDist * searchRadiusDist;
            int searchCells = 4;

            // Color accumulation per voxel
            Vector3[,,]? colorSum = colors != null ? new Vector3[w, h, d] : null;
            int[,,]? colorCount = colors != null ? new int[w, h, d] : null;

            // First, accumulate colors from points
            if (colorSum != null && colorCount != null && colors != null)
            {
                for (int i = 0; i < points.Length; i++)
                {
                    var v = points[i];
                    int gx = (int)((v.X - min.X) / voxelSize);
                    int gy = (int)((v.Y - min.Y) / voxelSize);
                    int gz = (int)((v.Z - min.Z) / voxelSize);

                    if (gx >= 0 && gx < w && gy >= 0 && gy < h && gz >= 0 && gz < d)
                    {
                        colorSum[gx, gy, gz] += colors[i];
                        colorCount[gx, gy, gz]++;
                    }
                }
            }

            // Compute SDF in parallel using weighted normal averaging
            Parallel.For(0, w, x =>
            {
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        float vx = min.X + (x + 0.5f) * voxelSize;
                        float vy = min.Y + (y + 0.5f) * voxelSize;
                        float vz = min.Z + (z + 0.5f) * voxelSize;
                        var voxelCenter = new Vector3(vx, vy, vz);

                        // Find nearest point and compute weighted sign from nearby points
                        var (nearestIdx, weightedSign) = FindNearestWithWeightedSign(
                            voxelCenter, points, normals, spatialHash, min, cellSize, searchCells);

                        if (nearestIdx < 0)
                        {
                            grid[x, y, z] = 1.0f; // fully outside
                            continue;
                        }

                        float dist = (voxelCenter - points[nearestIdx]).Length;

                        // If the nearest point is beyond the bandwidth, this voxel is
                        // far from the surface - mark as outside to avoid spurious geometry
                        if (dist > bandwidth)
                        {
                            grid[x, y, z] = 1.0f;
                            continue;
                        }

                        float sdf = weightedSign * dist;

                        // Clamp to bandwidth and normalize to [0, 1]
                        float clamped = Math.Clamp(sdf, -bandwidth, bandwidth);
                        grid[x, y, z] = (clamped + bandwidth) / (2.0f * bandwidth);
                    }
                }
            });

            progress?.Invoke("SDF computed. Smoothing...", 0.55f);

            // 6. Smoothing (2 passes) for clean connected isosurface
            var current = grid;
            var next = new float[w, h, d];
            for (int pass = 0; pass < 2; pass++)
            {
                progress?.Invoke($"Smoothing SDF (pass {pass + 1}/2)...", 0.55f + 0.10f * pass);
                Parallel.For(1, w - 1, bx =>
                {
                    for (int by = 1; by < h - 1; by++)
                    {
                        for (int bz = 1; bz < d - 1; bz++)
                        {
                            float sum = 0;
                            for (int ddx = -1; ddx <= 1; ddx++)
                                for (int ddy = -1; ddy <= 1; ddy++)
                                    for (int ddz = -1; ddz <= 1; ddz++)
                                        sum += current[bx + ddx, by + ddy, bz + ddz];
                            next[bx, by, bz] = sum / 27.0f;
                        }
                    }
                });
                // Copy boundaries
                for (int bx = 0; bx < w; bx++)
                    for (int by = 0; by < h; by++)
                    {
                        next[bx, by, 0] = current[bx, by, 0];
                        next[bx, by, d - 1] = current[bx, by, d - 1];
                    }
                for (int bx = 0; bx < w; bx++)
                    for (int bz = 0; bz < d; bz++)
                    {
                        next[bx, 0, bz] = current[bx, 0, bz];
                        next[bx, h - 1, bz] = current[bx, h - 1, bz];
                    }
                for (int by = 0; by < h; by++)
                    for (int bz = 0; bz < d; bz++)
                    {
                        next[0, by, bz] = current[0, by, bz];
                        next[w - 1, by, bz] = current[w - 1, by, bz];
                    }
                var temp = current;
                current = next;
                next = temp;
            }
            var smoothed = current;

            // 7. Build color grid
            progress?.Invoke("Building color grid...", 0.75f);
            Vector3[,,]? colorGrid = null;
            if (colorSum != null && colorCount != null)
            {
                colorGrid = new Vector3[w, h, d];
                Parallel.For(0, w, x =>
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            if (colorCount[x, y, z] > 0)
                                colorGrid[x, y, z] = colorSum[x, y, z] / colorCount[x, y, z];
                            else
                                colorGrid[x, y, z] = FindNearestColor(colorSum, colorCount, x, y, z, w, h, d);
                        }
                    }
                });
            }

            // Count surface voxels (where field is near 0.5)
            int surfaceVoxels = 0;
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < d; z++)
                    {
                        float val = smoothed[x, y, z];
                        if (val > 0.3f && val < 0.7f) surfaceVoxels++;
                    }
            Console.WriteLine($"[Voxelize-SDF] Surface voxels (0.3-0.7): {surfaceVoxels}");
            progress?.Invoke("Voxelization complete.", 1.0f);

            return (smoothed, colorGrid, min, voxelSize);
        }

        /// <summary>
        /// Estimate normals using PCA on k-nearest neighbors.
        /// </summary>
        private static Vector3[] EstimateNormalsPCA(Vector3[] points, int k, Action<string, float>? progress)
        {
            int n = points.Length;
            var normals = new Vector3[n];
            k = Math.Min(k, n - 1);

            if (k < 3)
            {
                // Too few points for PCA, use default up normal
                for (int i = 0; i < n; i++)
                    normals[i] = Vector3.UnitY;
                return normals;
            }

            // Compute centroid for consistent normal orientation
            var centroid = Vector3.Zero;
            foreach (var p in points)
                centroid += p;
            centroid /= n;

            progress?.Invoke("Building KD-grid for normals...", 0.03f);

            // Build spatial hash for neighbor finding
            // Use a cell size that gives roughly k neighbors per cell neighborhood
            float volume = 1.0f;
            {
                var pmin = new Vector3(float.MaxValue);
                var pmax = new Vector3(float.MinValue);
                foreach (var p in points)
                {
                    pmin = Vector3.ComponentMin(pmin, p);
                    pmax = Vector3.ComponentMax(pmax, p);
                }
                var sz = pmax - pmin;
                volume = Math.Max(sz.X * sz.Y * sz.Z, 1e-10f);
            }
            float avgSpacing = (float)Math.Pow(volume / n, 1.0 / 3.0);
            float cellSize = avgSpacing * 2.5f;
            var hash = new Dictionary<long, List<int>>();
            for (int i = 0; i < n; i++)
            {
                long key = SpatialKey(points[i], Vector3.Zero, cellSize);
                if (!hash.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    hash[key] = list;
                }
                list.Add(i);
            }

            progress?.Invoke("Estimating normals (PCA)...", 0.05f);

            // Estimate normals in parallel
            int reportInterval = Math.Max(1, n / 20);
            Parallel.For(0, n, i =>
            {
                // Find k nearest neighbors using spatial hash
                var neighbors = FindKNearest(points[i], i, points, hash, cellSize, k);

                if (neighbors.Count < 3)
                {
                    normals[i] = Vector3.UnitY;
                    return;
                }

                // Compute centroid of neighbors
                var neighCentroid = Vector3.Zero;
                foreach (int ni in neighbors)
                    neighCentroid += points[ni];
                neighCentroid /= neighbors.Count;

                // Build covariance matrix
                float cxx = 0, cxy = 0, cxz = 0;
                float cyy = 0, cyz = 0, czz = 0;
                foreach (int ni in neighbors)
                {
                    float dx = points[ni].X - neighCentroid.X;
                    float dy = points[ni].Y - neighCentroid.Y;
                    float dz = points[ni].Z - neighCentroid.Z;
                    cxx += dx * dx;
                    cxy += dx * dy;
                    cxz += dx * dz;
                    cyy += dy * dy;
                    cyz += dy * dz;
                    czz += dz * dz;
                }

                // Find smallest eigenvector using power iteration on inverse
                // Approximate: use cross product of two largest eigenvectors
                // Simpler approach: iterate to find the eigenvector with smallest eigenvalue
                var normal = SmallestEigenvector(cxx, cxy, cxz, cyy, cyz, czz);

                // Orient normal to point away from centroid
                var toCenter = centroid - points[i];
                if (Vector3.Dot(normal, toCenter) > 0)
                    normal = -normal;

                normals[i] = normal;
            });

            progress?.Invoke("Normals estimated.", 0.14f);
            return normals;
        }

        /// <summary>
        /// Find the eigenvector corresponding to the smallest eigenvalue of a 3x3 symmetric matrix.
        /// Uses the analytical method for 3x3 symmetric matrices.
        /// </summary>
        private static Vector3 SmallestEigenvector(float a00, float a01, float a02, float a11, float a12, float a22)
        {
            // Use inverse power iteration to find smallest eigenvector
            // Start with a random-ish vector
            var v = new Vector3(1.0f, 0.5f, 0.25f);
            v.Normalize();

            // Add a small shift to make matrix positive definite for inversion
            // This helps when eigenvalues are near zero
            float shift = (a00 + a11 + a22) * 0.001f + 1e-6f;
            float s00 = a00 + shift, s11 = a11 + shift, s22 = a22 + shift;

            for (int iter = 0; iter < 30; iter++)
            {
                // Solve (A + shift*I) * w = v using Cramer's rule for 3x3
                float det = s00 * (s11 * s22 - a12 * a12)
                          - a01 * (a01 * s22 - a12 * a02)
                          + a02 * (a01 * a12 - s11 * a02);

                if (Math.Abs(det) < 1e-15f)
                    break;

                float invDet = 1.0f / det;

                float wx = ((s11 * s22 - a12 * a12) * v.X + (a02 * a12 - a01 * s22) * v.Y + (a01 * a12 - a02 * s11) * v.Z) * invDet;
                float wy = ((a12 * a02 - a01 * s22) * v.X + (s00 * s22 - a02 * a02) * v.Y + (a01 * a02 - s00 * a12) * v.Z) * invDet;
                float wz = ((a01 * a12 - a02 * s11) * v.X + (a01 * a02 - s00 * a12) * v.Y + (s00 * s11 - a01 * a01) * v.Z) * invDet;

                float len = MathF.Sqrt(wx * wx + wy * wy + wz * wz);
                if (len < 1e-15f)
                    break;

                v = new Vector3(wx / len, wy / len, wz / len);
            }

            return v;
        }

        private static List<int> FindKNearest(Vector3 point, int selfIdx, Vector3[] points,
            Dictionary<long, List<int>> hash, float cellSize, int k)
        {
            var result = new List<(int idx, float distSq)>();
            int cx = (int)MathF.Floor(point.X / cellSize);
            int cy = (int)MathF.Floor(point.Y / cellSize);
            int cz = (int)MathF.Floor(point.Z / cellSize);

            for (int radius = 1; radius <= 3; radius++)
            {
                result.Clear();
                for (int dx = -radius; dx <= radius; dx++)
                {
                    for (int dy = -radius; dy <= radius; dy++)
                    {
                        for (int dz = -radius; dz <= radius; dz++)
                        {
                            long key = PackKey(cx + dx, cy + dy, cz + dz);
                            if (!hash.TryGetValue(key, out var list))
                                continue;
                            foreach (int i in list)
                            {
                                if (i == selfIdx) continue;
                                float distSq = (points[i] - point).LengthSquared;
                                result.Add((i, distSq));
                            }
                        }
                    }
                }

                if (result.Count >= k)
                    break;
            }

            // Sort and take k nearest
            result.Sort((a, b) => a.distSq.CompareTo(b.distSq));
            var kNearest = new List<int>(Math.Min(k, result.Count));
            for (int i = 0; i < Math.Min(k, result.Count); i++)
                kNearest.Add(result[i].idx);
            return kNearest;
        }

        private static Dictionary<long, List<int>> BuildSpatialHash(Vector3[] points, Vector3 origin, float cellSize)
        {
            var hash = new Dictionary<long, List<int>>(points.Length / 4);
            for (int i = 0; i < points.Length; i++)
            {
                long key = SpatialKey(points[i], origin, cellSize);
                if (!hash.TryGetValue(key, out var list))
                {
                    list = new List<int>();
                    hash[key] = list;
                }
                list.Add(i);
            }
            return hash;
        }

        private static long SpatialKey(Vector3 point, Vector3 origin, float cellSize)
        {
            int cx = (int)MathF.Floor((point.X - origin.X) / cellSize);
            int cy = (int)MathF.Floor((point.Y - origin.Y) / cellSize);
            int cz = (int)MathF.Floor((point.Z - origin.Z) / cellSize);
            return PackKey(cx, cy, cz);
        }

        private static long PackKey(int x, int y, int z)
        {
            // Pack 3 ints into a long for hash key (21 bits each)
            return ((long)(x & 0x1FFFFF)) | ((long)(y & 0x1FFFFF) << 21) | ((long)(z & 0x1FFFFF) << 42);
        }

        /// <summary>
        /// Find nearest point and compute a weighted sign from multiple nearby points.
        /// This avoids sign flickering from single-point normal inconsistencies.
        /// Returns (nearestIdx, weightedSign) where weightedSign is +1 or -1.
        /// </summary>
        private static (int nearestIdx, float weightedSign) FindNearestWithWeightedSign(
            Vector3 voxelCenter, Vector3[] points, Vector3[] normals,
            Dictionary<long, List<int>> spatialHash, Vector3 origin, float cellSize, int searchRadius)
        {
            int cx = (int)MathF.Floor((voxelCenter.X - origin.X) / cellSize);
            int cy = (int)MathF.Floor((voxelCenter.Y - origin.Y) / cellSize);
            int cz = (int)MathF.Floor((voxelCenter.Z - origin.Z) / cellSize);

            int bestIdx = -1;
            float bestDistSq = float.MaxValue;
            float weightedDotSum = 0;
            float weightSum = 0;

            for (int dx = -searchRadius; dx <= searchRadius; dx++)
            {
                for (int dy = -searchRadius; dy <= searchRadius; dy++)
                {
                    for (int dz = -searchRadius; dz <= searchRadius; dz++)
                    {
                        long key = PackKey(cx + dx, cy + dy, cz + dz);
                        if (!spatialHash.TryGetValue(key, out var list))
                            continue;

                        foreach (int i in list)
                        {
                            var diff = points[i] - voxelCenter;
                            float distSq = diff.LengthSquared;

                            if (distSq < bestDistSq)
                            {
                                bestDistSq = distSq;
                                bestIdx = i;
                            }

                            // Accumulate weighted sign from all nearby points
                            // Weight = 1 / (distSq + epsilon) so closer points have more influence
                            float weight = 1.0f / (distSq + 1e-8f);
                            float dot = Vector3.Dot(voxelCenter - points[i], normals[i]);
                            weightedDotSum += dot * weight;
                            weightSum += weight;
                        }
                    }
                }
            }

            if (bestIdx < 0)
                return (-1, 1.0f);

            // Use weighted average of dot products to determine sign
            float avgDot = weightedDotSum / weightSum;
            float sign = avgDot >= 0 ? 1.0f : -1.0f;

            return (bestIdx, sign);
        }

        private static Vector3 FindNearestColor(Vector3[,,] colorSum, int[,,] colorCount, int cx, int cy, int cz, int w, int h, int d)
        {
            for (int r = 1; r <= 3; r++)
            {
                for (int dx = -r; dx <= r; dx++)
                {
                    for (int dy = -r; dy <= r; dy++)
                    {
                        for (int dz = -r; dz <= r; dz++)
                        {
                            int nx = cx + dx, ny = cy + dy, nz = cz + dz;
                            if (nx >= 0 && nx < w && ny >= 0 && ny < h && nz >= 0 && nz < d)
                            {
                                if (colorCount[nx, ny, nz] > 0)
                                    return colorSum[nx, ny, nz] / colorCount[nx, ny, nz];
                            }
                        }
                    }
                }
            }

            return new Vector3(0.8f, 0.8f, 0.8f);
        }
    }
}
