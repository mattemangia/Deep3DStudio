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

            // 2. Compute voxel size from the largest dimension so the grid always
            //    fills maxRes voxels along that axis, regardless of coordinate scale.
            float voxelSize = maxDim / Math.Max(2, maxRes - 10);

            // Add padding so points aren't at the grid boundary (needed for blur/dilation)
            var padding = new Vector3(voxelSize * 5);
            min -= padding;
            max += padding;
            size = max - min;

            int w = Math.Min(maxRes, (int)Math.Ceiling(size.X / voxelSize) + 1);
            int h = Math.Min(maxRes, (int)Math.Ceiling(size.Y / voxelSize) + 1);
            int d = Math.Min(maxRes, (int)Math.Ceiling(size.Z / voxelSize) + 1);

            if (w <= 2 || h <= 2 || d <= 2)
                return (null, null, min, voxelSize);

            Console.WriteLine($"[Voxelize] Grid: {w}x{h}x{d}, voxelSize={voxelSize:F6}, points={points.Length}");

            // 3. Binary voxelization: mark occupied voxels
            var grid = new float[w, h, d];

            // Color accumulation per voxel
            Vector3[,,]? colorSum = colors != null ? new Vector3[w, h, d] : null;
            int[,,]? colorCount = colors != null ? new int[w, h, d] : null;

            for (int i = 0; i < points.Length; i++)
            {
                var v = points[i];
                int gx = (int)((v.X - min.X) / voxelSize);
                int gy = (int)((v.Y - min.Y) / voxelSize);
                int gz = (int)((v.Z - min.Z) / voxelSize);

                if (gx >= 0 && gx < w && gy >= 0 && gy < h && gz >= 0 && gz < d)
                {
                    grid[gx, gy, gz] = 1.0f;

                    if (colorSum != null && colorCount != null && colors != null)
                    {
                        colorSum[gx, gy, gz] += colors[i];
                        colorCount[gx, gy, gz]++;
                    }
                }
            }

            // Count occupied voxels
            int occupied = 0;
            for (int x = 0; x < w; x++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < d; z++)
                        if (grid[x, y, z] > 0) occupied++;
            Console.WriteLine($"[Voxelize] Occupied voxels: {occupied}");

            // 4. Dilation: expand occupied region to bridge small gaps.
            //    Use adaptive radius but cap at 3 to avoid filling the entire grid.
            float volume = size.X * size.Y * size.Z;
            float avgSpacing = volume > 0
                ? (float)Math.Pow(volume / points.Length, 1.0 / 3.0)
                : voxelSize;
            int dilationRadius = Math.Clamp((int)Math.Ceiling(avgSpacing / voxelSize * 0.5f), 1, 3);
            Console.WriteLine($"[Voxelize] avgSpacing={avgSpacing:F4}, dilationRadius={dilationRadius}");

            var current = grid;
            for (int round = 0; round < dilationRadius; round++)
            {
                var dilated = new float[w, h, d];
                Parallel.For(1, w - 1, x =>
                {
                    for (int y = 1; y < h - 1; y++)
                    {
                        for (int z = 1; z < d - 1; z++)
                        {
                            if (current[x, y, z] > 0)
                            {
                                dilated[x, y, z] = 1.0f;
                                dilated[x + 1, y, z] = 1.0f;
                                dilated[x - 1, y, z] = 1.0f;
                                dilated[x, y + 1, z] = 1.0f;
                                dilated[x, y - 1, z] = 1.0f;
                                dilated[x, y, z + 1] = 1.0f;
                                dilated[x, y, z - 1] = 1.0f;
                            }
                        }
                    }
                });
                current = dilated;
            }

            // 5. Box blur smoothing (3 passes) to create smooth gradient for MC.
            //    Binary 0/1 → smooth [0,1]. MC at isoLevel=0.5 extracts the surface.
            var next = new float[w, h, d];
            for (int pass = 0; pass < 3; pass++)
            {
                Parallel.For(1, w - 1, x =>
                {
                    for (int y = 1; y < h - 1; y++)
                    {
                        for (int z = 1; z < d - 1; z++)
                        {
                            float sum = 0;
                            for (int ddx = -1; ddx <= 1; ddx++)
                                for (int ddy = -1; ddy <= 1; ddy++)
                                    for (int ddz = -1; ddz <= 1; ddz++)
                                        sum += current[x + ddx, y + ddy, z + ddz];
                            next[x, y, z] = sum / 27.0f;
                        }
                    }
                });

                var temp = current;
                current = next;
                next = temp;
            }

            // 6. Build color grid from accumulated colors
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
                        float val = current[x, y, z];
                        if (val > 0.1f && val < 0.9f) surfaceVoxels++;
                    }
            Console.WriteLine($"[Voxelize] Surface voxels (0.1-0.9): {surfaceVoxels}");

            return (current, colorGrid, min, voxelSize);
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
