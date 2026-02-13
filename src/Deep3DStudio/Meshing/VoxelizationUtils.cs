using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public static class VoxelizationUtils
    {
        public static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) Voxelize(List<MeshData> meshes, int maxRes)
        {
            int totalCount = 0;
            foreach (var m in meshes)
                totalCount += m.Vertices.Count;

            if (totalCount == 0)
                return (null, null, Vector3.Zero, 0.02f);

            var points = new Vector3[totalCount];
            var colors = new Vector3[totalCount];
            int idx = 0;
            foreach (var m in meshes)
            {
                bool hasColors = m.Colors.Count >= m.Vertices.Count;
                for (int i = 0; i < m.Vertices.Count; i++)
                {
                    points[idx] = m.Vertices[i];
                    colors[idx] = hasColors ? m.Colors[i] : new Vector3(0.8f, 0.8f, 0.8f);
                    idx++;
                }
            }

            return VoxelizeCore(points, colors, maxRes);
        }

        public static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) Voxelize(IList<Vector3> points, int maxRes)
        {
            if (points.Count == 0)
                return (null, null, Vector3.Zero, 0.02f);

            var arr = new Vector3[points.Count];
            for (int i = 0; i < points.Count; i++)
                arr[i] = points[i];

            return VoxelizeCore(arr, null, maxRes);
        }

        private static (float[,,]? grid, Vector3[,,]? colorGrid, Vector3 origin, float voxelSize) VoxelizeCore(Vector3[] points, Vector3[]? colors, int maxRes)
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

            // 2. Compute voxel size and grid dimensions
            float voxelSize = maxDim / Math.Max(2, maxRes - 10);

            // Add padding so points aren't at boundary (critical for gradient-based methods)
            var padding = new Vector3(voxelSize * 5);
            min -= padding;
            max += padding;
            size = max - min;

            int w = Math.Min(maxRes, (int)Math.Ceiling(size.X / voxelSize) + 1);
            int h = Math.Min(maxRes, (int)Math.Ceiling(size.Y / voxelSize) + 1);
            int d = Math.Min(maxRes, (int)Math.Ceiling(size.Z / voxelSize) + 1);

            if (w <= 2 || h <= 2 || d <= 2)
                return (null, null, min, voxelSize);

            // 3. Estimate point spacing and compute splat radius
            float volume = size.X * size.Y * size.Z;
            float estimatedSpacing = (float)Math.Pow(volume / points.Length, 1.0 / 3.0);
            int splatRadius = Math.Max(2, Math.Min(5, (int)Math.Ceiling(estimatedSpacing / voxelSize * 1.5f)));
            float sigma = splatRadius * voxelSize * 0.4f;
            float twoSigmaSq = 2.0f * sigma * sigma;

            // 4. Gaussian splatting: each point contributes a Gaussian blob to the grid
            var grid = new float[w, h, d];

            // Color accumulation grids (only if colors provided)
            Vector3[,,]? colorAccum = colors != null ? new Vector3[w, h, d] : null;
            float[,,]? weightAccum = colors != null ? new float[w, h, d] : null;

            Parallel.For(0, points.Length, i =>
            {
                var v = points[i];
                float fx = (v.X - min.X) / voxelSize;
                float fy = (v.Y - min.Y) / voxelSize;
                float fz = (v.Z - min.Z) / voxelSize;

                int cx = (int)fx;
                int cy = (int)fy;
                int cz = (int)fz;

                int xMin = Math.Max(0, cx - splatRadius);
                int xMax = Math.Min(w - 1, cx + splatRadius);
                int yMin = Math.Max(0, cy - splatRadius);
                int yMax = Math.Min(h - 1, cy + splatRadius);
                int zMin = Math.Max(0, cz - splatRadius);
                int zMax = Math.Min(d - 1, cz + splatRadius);

                Vector3 pointColor = colors != null ? colors[i] : default;

                for (int x = xMin; x <= xMax; x++)
                {
                    float dx = (x - fx) * voxelSize;
                    float dxSq = dx * dx;
                    for (int y = yMin; y <= yMax; y++)
                    {
                        float dy = (y - fy) * voxelSize;
                        float dySq = dy * dy;
                        for (int z = zMin; z <= zMax; z++)
                        {
                            float dz = (z - fz) * voxelSize;
                            float distSq = dxSq + dySq + dz * dz;
                            float weight = (float)Math.Exp(-distSq / twoSigmaSq);

                            // Acceptable minor races for additive accumulation
                            grid[x, y, z] += weight;

                            if (colorAccum != null && weightAccum != null)
                            {
                                colorAccum[x, y, z] += weight * pointColor;
                                weightAccum[x, y, z] += weight;
                            }
                        }
                    }
                }
            });

            // 5. Box blur smoothing (2 passes) for density only (not colors)
            int smoothPasses = 2;
            var current = grid;
            var next = new float[w, h, d];

            for (int iter = 0; iter < smoothPasses; iter++)
            {
                Parallel.For(1, w - 1, x =>
                {
                    for (int y = 1; y < h - 1; y++)
                    {
                        for (int z = 1; z < d - 1; z++)
                        {
                            float sum = 0;
                            for (int dx2 = -1; dx2 <= 1; dx2++)
                                for (int dy2 = -1; dy2 <= 1; dy2++)
                                    for (int dz2 = -1; dz2 <= 1; dz2++)
                                        sum += current[x + dx2, y + dy2, z + dz2];
                            next[x, y, z] = sum / 27.0f;
                        }
                    }
                });

                var temp = current;
                current = next;
                next = temp;
            }

            // 6. Normalize density to [0, 1]
            float maxVal = 0;
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < d; z++)
                        if (current[x, y, z] > maxVal)
                            maxVal = current[x, y, z];

            if (maxVal > 0)
            {
                float invMax = 1.0f / maxVal;
                Parallel.For(0, w, x =>
                {
                    for (int y = 0; y < h; y++)
                        for (int z = 0; z < d; z++)
                            current[x, y, z] *= invMax;
                });
            }

            // 7. Normalize color grid by accumulated weights
            Vector3[,,]? colorGrid = null;
            if (colorAccum != null && weightAccum != null)
            {
                colorGrid = new Vector3[w, h, d];
                Parallel.For(0, w, x =>
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            if (weightAccum[x, y, z] > 1e-6f)
                                colorGrid[x, y, z] = colorAccum[x, y, z] / weightAccum[x, y, z];
                            else
                                colorGrid[x, y, z] = new Vector3(0.8f, 0.8f, 0.8f);
                        }
                    }
                });
            }

            return (current, colorGrid, min, voxelSize);
        }
    }
}
