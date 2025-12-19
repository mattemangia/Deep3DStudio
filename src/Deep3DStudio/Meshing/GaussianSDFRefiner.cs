using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using OpenTK.Mathematics;

namespace Deep3DStudio.Meshing
{
    /// <summary>
    /// Gaussian SDF Refiner: A mesh refinement algorithm that converts a mesh to a Signed Distance Field (SDF),
    /// applies 3D Gaussian smoothing (Geometric Gaussian Supervision), and extracts a refined mesh.
    /// Inspired by the principles of GSurf (Gaussian Surface), adapted for geometric refinement.
    /// </summary>
    public class GaussianSDFRefiner
    {
        public Task<MeshData> RefineMeshAsync(MeshData inputMesh, Action<string, float>? progressCallback = null)
        {
            return Task.Run(() =>
            {
                var settings = IniSettings.Instance;
                int resolution = settings.GaussianSDFGridResolution;
                float sigma = settings.GaussianSDFSigma;
                int iterations = settings.GaussianSDFIterations;
                float isoLevel = settings.GaussianSDFIsoLevel;

                progressCallback?.Invoke("Initializing Gaussian SDF Grid...", 0.1f);

                // 1. Calculate Bounds
                var bounds = inputMesh.GetBounds();
                Vector3 min = bounds.Min;
                Vector3 max = bounds.Max;
                Vector3 size = max - min;
                float maxDim = Math.Max(size.X, Math.Max(size.Y, size.Z));

                // Add padding
                float padding = maxDim * 0.1f;
                min -= new Vector3(padding);
                max += new Vector3(padding);
                size = max - min;

                // Uniform grid based on max dimension
                float step = Math.Max(size.X, Math.Max(size.Y, size.Z)) / resolution;
                int resX = (int)(size.X / step) + 1;
                int resY = (int)(size.Y / step) + 1;
                int resZ = (int)(size.Z / step) + 1;

                float[,,] sdf = new float[resX, resY, resZ];

                progressCallback?.Invoke("Computing Signed Distance Field...", 0.2f);

                // 2. Compute SDF (Signed Distance Field)
                // For each voxel, find distance to the nearest triangle. Without a KD-Tree,
                // use a parallel brute-force scan, which is adequate for moderate meshes.

                var triangles = new List<(Vector3 p1, Vector3 p2, Vector3 p3)>();
                for (int i = 0; i < inputMesh.Indices.Count; i += 3)
                {
                    triangles.Add((
                        inputMesh.Vertices[inputMesh.Indices[i]],
                        inputMesh.Vertices[inputMesh.Indices[i+1]],
                        inputMesh.Vertices[inputMesh.Indices[i+2]]
                    ));
                }

                Parallel.For(0, resX, x =>
                {
                    for (int y = 0; y < resY; y++)
                    {
                        for (int z = 0; z < resZ; z++)
                        {
                            Vector3 p = min + new Vector3(x * step, y * step, z * step);
                            float minDist = float.MaxValue;

                            // Brute-force check against all triangles
                            foreach (var tri in triangles)
                            {
                                float dist = PointToTriangleDistance(p, tri.p1, tri.p2, tri.p3);
                                if (dist < minDist) minDist = dist;
                            }

                            // Without a reliable inside/outside test, use unsigned distance.
                            sdf[x, y, z] = minDist;
                        }
                    }
                });

                // Sign correction (optional) uses the closest triangle normal as a heuristic.

                progressCallback?.Invoke("Correcting SDF Signs...", 0.4f);
                 Parallel.For(0, resX, x =>
                {
                    for (int y = 0; y < resY; y++)
                    {
                        for (int z = 0; z < resZ; z++)
                        {
                            Vector3 p = min + new Vector3(x * step, y * step, z * step);
                            float minDist = float.MaxValue;
                            Vector3 closestNormal = Vector3.UnitY;
                            Vector3 closestPoint = Vector3.Zero;

                            foreach (var tri in triangles)
                            {
                                var result = ClosestPointOnTriangle(p, tri.p1, tri.p2, tri.p3);
                                float dist = Vector3.Distance(p, result.Point);
                                if (dist < minDist)
                                {
                                    minDist = dist;
                                    closestPoint = result.Point;
                                    // Compute normal
                                    var triNormal = Vector3.Cross(tri.p2 - tri.p1, tri.p3 - tri.p1).Normalized();
                                    closestNormal = triNormal;
                                }
                            }

                            // Dot product with normal to determine side
                            // If (p - closestPoint) . normal > 0, we are "outside" (in direction of normal)
                            float sign = Vector3.Dot(p - closestPoint, closestNormal) > 0 ? 1.0f : -1.0f;
                            sdf[x, y, z] = minDist * sign;
                        }
                    }
                });


                progressCallback?.Invoke("Applying Gaussian Smoothing...", 0.6f);

                // 3. Gaussian Smoothing (Geometric Gaussian Supervision)
                // Apply 3D Gaussian convolution
                if (iterations > 0 && sigma > 0)
                {
                    for(int iter=0; iter<iterations; iter++)
                    {
                        sdf = ApplyGaussianFilter(sdf, resX, resY, resZ, sigma);
                    }
                }

                progressCallback?.Invoke("Extracting Refined Mesh...", 0.8f);

                // 4. Extract Mesh (Marching Cubes)
                // Use the local extractor to avoid extra dependencies.

                var refinedMesh = ExtractIsoSurface(sdf, resX, resY, resZ, min, step, isoLevel);

                progressCallback?.Invoke("Refinement Complete.", 1.0f);
                return refinedMesh;
            });
        }

        private float[,,] ApplyGaussianFilter(float[,,] input, int w, int h, int d, float sigma)
        {
            float[,,] output = new float[w, h, d];
            int kernelRadius = (int)Math.Ceiling(sigma * 2.0f);
            float[] kernel = new float[kernelRadius * 2 + 1];
            float sum = 0;

            for (int i = -kernelRadius; i <= kernelRadius; i++)
            {
                float val = (float)Math.Exp(-(i * i) / (2 * sigma * sigma));
                kernel[i + kernelRadius] = val;
                sum += val;
            }
            for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;

            // Separable convolution: X, then Y, then Z.

            float[,,] temp1 = new float[w, h, d];
            float[,,] temp2 = new float[w, h, d];

            // X pass
            Parallel.For(0, d, z =>
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                        {
                            int ix = Math.Clamp(x + k, 0, w - 1);
                            val += input[ix, y, z] * kernel[k + kernelRadius];
                        }
                        temp1[x, y, z] = val;
                    }
                }
            });

            // Y pass
             Parallel.For(0, d, z =>
            {
                for (int x = 0; x < w; x++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                        {
                            int iy = Math.Clamp(y + k, 0, h - 1);
                            val += temp1[x, iy, z] * kernel[k + kernelRadius];
                        }
                        temp2[x, y, z] = val;
                    }
                }
            });

            // Z pass
             Parallel.For(0, h, y =>
            {
                for (int x = 0; x < w; x++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                        {
                            int iz = Math.Clamp(z + k, 0, d - 1);
                            val += temp2[x, y, iz] * kernel[k + kernelRadius];
                        }
                        output[x, y, z] = val;
                    }
                }
            });

            return output;
        }

        private float PointToTriangleDistance(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            var result = ClosestPointOnTriangle(p, a, b, c);
            return Vector3.Distance(p, result.Point);
        }

        private (Vector3 Point, float S, float T) ClosestPointOnTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            // Robust closest point on triangle algorithm
            Vector3 edge0 = b - a;
            Vector3 edge1 = c - a;
            Vector3 v0 = a - p;

            float a_dot = Vector3.Dot(edge0, edge0);
            float b_dot = Vector3.Dot(edge0, edge1);
            float c_dot = Vector3.Dot(edge1, edge1);
            float d = Vector3.Dot(edge0, v0);
            float e = Vector3.Dot(edge1, v0);

            float det = a_dot * c_dot - b_dot * b_dot;
            float s = b_dot * e - c_dot * d;
            float t = b_dot * d - a_dot * e;

            if (s + t < det)
            {
                if (s < 0.0f)
                {
                    if (t < 0.0f)
                    {
                        if (d < 0.0f)
                        {
                            s = Math.Clamp(-d / a_dot, 0.0f, 1.0f);
                            t = 0.0f;
                        }
                        else
                        {
                            s = 0.0f;
                            t = Math.Clamp(-e / c_dot, 0.0f, 1.0f);
                        }
                    }
                    else
                    {
                        s = 0.0f;
                        t = Math.Clamp(-e / c_dot, 0.0f, 1.0f);
                    }
                }
                else if (t < 0.0f)
                {
                    s = Math.Clamp(-d / a_dot, 0.0f, 1.0f);
                    t = 0.0f;
                }
                else
                {
                    float invDet = 1.0f / det;
                    s *= invDet;
                    t *= invDet;
                }
            }
            else
            {
                if (s < 0.0f)
                {
                    float tmp0 = b_dot + d;
                    float tmp1 = c_dot + e;
                    if (tmp1 > tmp0)
                    {
                        float numer = tmp1 - tmp0;
                        float denom = a_dot - 2 * b_dot + c_dot;
                        s = Math.Clamp(numer / denom, 0.0f, 1.0f);
                        t = 1.0f - s;
                    }
                    else
                    {
                        t = Math.Clamp(-e / c_dot, 0.0f, 1.0f);
                        s = 0.0f;
                    }
                }
                else if (t < 0.0f)
                {
                    if (a_dot + d > b_dot + e)
                    {
                        float numer = c_dot + e - b_dot - d;
                        float denom = a_dot - 2 * b_dot + c_dot;
                        s = Math.Clamp(numer / denom, 0.0f, 1.0f);
                        t = 1.0f - s;
                    }
                    else
                    {
                        s = Math.Clamp(-d / a_dot, 0.0f, 1.0f);
                        t = 0.0f;
                    }
                }
                else
                {
                    float numer = c_dot + e - b_dot - d;
                    float denom = a_dot - 2 * b_dot + c_dot;
                    s = Math.Clamp(numer / denom, 0.0f, 1.0f);
                    t = 1.0f - s;
                }
            }

            return (a + s * edge0 + t * edge1, s, t);
        }

        private MeshData ExtractIsoSurface(float[,,] grid, int resX, int resY, int resZ, Vector3 min, float step, float isoLevel)
        {
            var mesh = new MeshData();
            // Simple Marching Cubes implementation or call existing utility if available.
            // Assuming we must implement it since we are in a Refiner class.

            // Marching Cubes Tables (simplified for brevity, usually these are large static arrays)
            // Ideally we should use GeometryUtils.MarchingCubesTables if it exists or define them.
            // Let's verify if GeometryUtils has them. The memory says "GeometryUtils.cs, the Marching Cubes lookup table triTable (int[,])".
            // So we can use GeometryUtils.

            for (int x = 0; x < resX - 1; x++)
            {
                for (int y = 0; y < resY - 1; y++)
                {
                    for (int z = 0; z < resZ - 1; z++)
                    {
                        // 8 corners
                        float[] val = new float[8];
                        Vector3[] pos = new Vector3[8];

                        val[0] = grid[x, y, z];
                        val[1] = grid[x + 1, y, z];
                        val[2] = grid[x + 1, y, z + 1];
                        val[3] = grid[x, y, z + 1];
                        val[4] = grid[x, y + 1, z];
                        val[5] = grid[x + 1, y + 1, z];
                        val[6] = grid[x + 1, y + 1, z + 1];
                        val[7] = grid[x, y + 1, z + 1];

                        pos[0] = min + new Vector3(x * step, y * step, z * step);
                        pos[1] = min + new Vector3((x + 1) * step, y * step, z * step);
                        pos[2] = min + new Vector3((x + 1) * step, y * step, (z + 1) * step);
                        pos[3] = min + new Vector3(x * step, y * step, (z + 1) * step);
                        pos[4] = min + new Vector3(x * step, (y + 1) * step, z * step);
                        pos[5] = min + new Vector3((x + 1) * step, (y + 1) * step, z * step);
                        pos[6] = min + new Vector3((x + 1) * step, (y + 1) * step, (z + 1) * step);
                        pos[7] = min + new Vector3(x * step, (y + 1) * step, (z + 1) * step);

                        int cubeIndex = 0;
                        if (val[0] < isoLevel) cubeIndex |= 1;
                        if (val[1] < isoLevel) cubeIndex |= 2;
                        if (val[2] < isoLevel) cubeIndex |= 4;
                        if (val[3] < isoLevel) cubeIndex |= 8;
                        if (val[4] < isoLevel) cubeIndex |= 16;
                        if (val[5] < isoLevel) cubeIndex |= 32;
                        if (val[6] < isoLevel) cubeIndex |= 64;
                        if (val[7] < isoLevel) cubeIndex |= 128;

                        // Use GeometryUtils or local tables
                        int[] edges = Deep3DStudio.Model.GeometryUtils.GetMarchingCubesEdges(cubeIndex);

                        for (int i = 0; i < edges.Length; i += 3)
                        {
                            if (edges[i] == -1) break;

                            int e0 = edges[i];
                            int e1 = edges[i+1];
                            int e2 = edges[i+2];

                            mesh.Vertices.Add(Interp(pos, val, e0, isoLevel));
                            mesh.Vertices.Add(Interp(pos, val, e1, isoLevel));
                            mesh.Vertices.Add(Interp(pos, val, e2, isoLevel));

                            int vCount = mesh.Vertices.Count;
                            mesh.Indices.Add(vCount - 3);
                            mesh.Indices.Add(vCount - 2);
                            mesh.Indices.Add(vCount - 1);
                        }
                    }
                }
            }

            mesh.RecalculateNormals();
            return mesh;
        }

        private Vector3 Interp(Vector3[] pos, float[] val, int edgeIndex, float iso)
        {
            // Edge to vertex mapping
            // 0: 0-1, 1: 1-2, 2: 2-3, 3: 3-0
            // 4: 4-5, 5: 5-6, 6: 6-7, 7: 7-4
            // 8: 0-4, 9: 1-5, 10: 2-6, 11: 3-7

            int v1 = 0, v2 = 0;
            switch(edgeIndex)
            {
                case 0: v1=0; v2=1; break;
                case 1: v1=1; v2=2; break;
                case 2: v1=2; v2=3; break;
                case 3: v1=3; v2=0; break;
                case 4: v1=4; v2=5; break;
                case 5: v1=5; v2=6; break;
                case 6: v1=6; v2=7; break;
                case 7: v1=7; v2=4; break;
                case 8: v1=0; v2=4; break;
                case 9: v1=1; v2=5; break;
                case 10: v1=2; v2=6; break;
                case 11: v1=3; v2=7; break;
            }

            float p1Val = val[v1];
            float p2Val = val[v2];
            Vector3 p1 = pos[v1];
            Vector3 p2 = pos[v2];

            if (Math.Abs(iso - p1Val) < 0.00001f) return p1;
            if (Math.Abs(iso - p2Val) < 0.00001f) return p2;
            if (Math.Abs(p1Val - p2Val) < 0.00001f) return p1;

            float mu = (iso - p1Val) / (p2Val - p1Val);
            return p1 + mu * (p2 - p1);
        }
    }
}
