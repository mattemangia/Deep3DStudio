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
            float voxelSize = 0.02f;
            int w = (int)(size.X / voxelSize) + 5;
            int h = (int)(size.Y / voxelSize) + 5;
            int d = (int)(size.Z / voxelSize) + 5;

            // Scale voxelSize if grid exceeds maxRes
            if (w > maxRes)
            {
                voxelSize *= (w / (float)maxRes);
                w = maxRes;
                h = (int)(size.Y / voxelSize) + 5;
                d = (int)(size.Z / voxelSize) + 5;
            }
            if (h > maxRes) h = maxRes;
            if (d > maxRes) d = maxRes;

            if (w <= 2 || h <= 2 || d <= 2)
                return (null, null, min, voxelSize);

            // 3. Compute adaptive dilation radius based on point spacing.
            //    Points that are far apart (relative to voxelSize) need more dilation
            //    rounds so their expanded regions overlap, forming a connected surface.
            float volume = size.X * size.Y * size.Z;
            float avgSpacing = volume > 0
                ? (float)Math.Pow(volume / points.Length, 1.0 / 3.0)
                : voxelSize;
            int dilationRadius = Math.Clamp((int)Math.Ceiling(avgSpacing / voxelSize), 1, 8);

            // 4. Binary voxelization: mark occupied voxels
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

            // 5. Multiple rounds of 6-connected dilation.
            //    Each round expands the occupied region by 1 voxel in each axis direction.
            //    This bridges gaps between sparse points so MC produces a connected surface.
            var current = grid;
            for (int round = 0; round < dilationRadius; round++)
            {
                var dilated = new float[w, h, d];
                for (int x = 1; x < w - 1; x++)
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
                }
                current = dilated;
            }

            // 6. Box blur smoothing (3 passes) to create a smooth field for MC.
            //    After blur, the binary 0/1 field becomes a smooth [0,1] gradient.
            //    MC at isoLevel=0.5 naturally extracts the surface at the boundary
            //    of the dilated region with smooth vertex interpolation.
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
                            for (int dx = -1; dx <= 1; dx++)
                                for (int dy = -1; dy <= 1; dy++)
                                    for (int dz = -1; dz <= 1; dz++)
                                        sum += current[x + dx, y + dy, z + dz];
                            next[x, y, z] = sum / 27.0f;
                        }
                    }
                });

                var temp = current;
                current = next;
                next = temp;
            }

            // 7. Build color grid from accumulated colors
            Vector3[,,]? colorGrid = null;
            if (colorSum != null && colorCount != null)
            {
                colorGrid = new Vector3[w, h, d];
                for (int x = 0; x < w; x++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        for (int z = 0; z < d; z++)
                        {
                            if (colorCount[x, y, z] > 0)
                            {
                                colorGrid[x, y, z] = colorSum[x, y, z] / colorCount[x, y, z];
                            }
                            else
                            {
                                colorGrid[x, y, z] = FindNearestColor(colorSum, colorCount, x, y, z, w, h, d);
                            }
                        }
                    }
                }
            }

            return (current, colorGrid, min, voxelSize);
        }

        private static Vector3 FindNearestColor(Vector3[,,] colorSum, int[,,] colorCount, int cx, int cy, int cz, int w, int h, int d)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
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

            return new Vector3(0.8f, 0.8f, 0.8f);
        }
    }
}
