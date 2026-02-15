using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    /// <summary>
    /// Reconstructs a mesh from multi-view normal maps and color images (e.g. from Wonder3D).
    /// Uses TSDF (Truncated Signed Distance Function) fusion with normal-based depth integration.
    /// Pure C# implementation — no Python/NeuS dependency required.
    /// </summary>
    public class NormalFusionMesher
    {
        private int _resolution = 128;
        private float _smoothSigma = 1.5f;
        private int _smoothIterations = 2;

        public NormalFusionMesher(int resolution = 128, float smoothSigma = 1.5f, int smoothIterations = 2)
        {
            _resolution = resolution;
            _smoothSigma = smoothSigma;
            _smoothIterations = smoothIterations;
        }

        /// <summary>
        /// Reconstruct a mesh from multi-view normal maps and color images.
        /// </summary>
        public MeshData Reconstruct(
            List<MultiViewData> views,
            Action<string, float>? progress = null,
            CancellationToken cancellationToken = default)
        {
            if (views == null || views.Count < 6)
                throw new ArgumentException("Need at least 6 views for normal fusion");

            progress?.Invoke("Normal fusion: integrating depth maps...", 0.0f);

            // Step 1: Integrate normals to depth maps for each view
            var depthMaps = new float[views.Count][];
            for (int v = 0; v < views.Count; v++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progress?.Invoke($"Normal fusion: integrating depth for view {v + 1}/{views.Count}...", (float)v / views.Count * 0.3f);
                depthMaps[v] = IntegrateNormalsToDepth(views[v]);
            }

            // Step 2: Create TSDF volume and carve from all views
            progress?.Invoke("Normal fusion: building TSDF volume...", 0.3f);
            int res = _resolution;
            float extent = 1.2f; // slightly larger than [-1,1] to capture edges
            float voxelSize = (2.0f * extent) / res;
            var origin = new Vector3(-extent, -extent, -extent);

            var tsdf = new float[res, res, res];
            var weights = new float[res, res, res];

            // Initialize TSDF to positive (outside)
            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                    for (int z = 0; z < res; z++)
                        tsdf[x, y, z] = 1.0f;

            for (int v = 0; v < views.Count; v++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                progress?.Invoke($"Normal fusion: carving view {v + 1}/{views.Count}...", 0.3f + (float)v / views.Count * 0.3f);
                CarveTSDF(tsdf, weights, views[v], depthMaps[v], origin, voxelSize, res);
            }

            // Normalize TSDF by weights
            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                    for (int z = 0; z < res; z++)
                    {
                        if (weights[x, y, z] > 0)
                            tsdf[x, y, z] /= weights[x, y, z];
                    }

            // Step 3: Smooth the TSDF
            progress?.Invoke("Normal fusion: smoothing volume...", 0.6f);
            var smoothed = tsdf;
            for (int i = 0; i < _smoothIterations; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                smoothed = ApplyGaussianFilter(smoothed, res, res, res, _smoothSigma, cancellationToken);
            }

            // Step 4: Extract mesh via Marching Cubes
            progress?.Invoke("Normal fusion: extracting mesh...", 0.75f);

            // Build a neutral color grid (will be overwritten by projection)
            var colorGrid = new Vector3[res, res, res];
            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                    for (int z = 0; z < res; z++)
                        colorGrid[x, y, z] = new Vector3(0.7f, 0.7f, 0.7f);

            var mesh = GeometryUtils.MarchingCubes(
                smoothed, colorGrid, origin, new Vector3(voxelSize),
                0.0f, // iso-level at zero crossing
                (msg, p) => progress?.Invoke($"Normal fusion: {msg}", 0.75f + p * 0.15f));

            if (mesh.Vertices.Count == 0)
            {
                progress?.Invoke("Normal fusion: no surface found, retrying with adaptive threshold...", 0.85f);
                // Try adaptive iso-level
                float isoLevel = ComputeAdaptiveIsoLevel(smoothed, res);
                mesh = GeometryUtils.MarchingCubes(
                    smoothed, colorGrid, origin, new Vector3(voxelSize),
                    isoLevel,
                    (msg, p) => progress?.Invoke($"Normal fusion: {msg}", 0.85f + p * 0.05f));
            }

            // Step 5: Project colors from views onto mesh vertices
            if (mesh.Vertices.Count > 0)
            {
                progress?.Invoke("Normal fusion: projecting colors...", 0.9f);
                ProjectColors(mesh, views);
                mesh.RecalculateNormals();
            }

            progress?.Invoke($"Normal fusion complete: {mesh.Vertices.Count} vertices, {mesh.Indices.Count / 3} triangles", 1.0f);
            return mesh;
        }

        /// <summary>
        /// Integrate a normal map to produce a depth map using iterative Poisson integration.
        /// Normal map encodes surface orientation; depth is recovered by integrating gradients.
        /// </summary>
        private float[] IntegrateNormalsToDepth(MultiViewData view)
        {
            int w = view.Width;
            int h = view.Height;
            float[] normals = view.Normals;
            float[] depth = new float[w * h];

            if (normals.Length < w * h * 3) return depth;

            // Extract depth gradients from normals: dz/dx = -nx/nz, dz/dy = -ny/nz
            float[] p = new float[w * h]; // dz/dx
            float[] q = new float[w * h]; // dz/dy

            for (int y = 0; y < h; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * 3;
                    float nx = normals[idx];
                    float ny = normals[idx + 1];
                    float nz = normals[idx + 2];

                    // Avoid division by very small nz
                    if (MathF.Abs(nz) > 0.01f)
                    {
                        p[y * w + x] = -nx / nz;
                        q[y * w + x] = -ny / nz;
                    }
                }
            }

            // Compute divergence of the gradient field
            float[] divergence = new float[w * h];
            for (int y = 1; y < h - 1; y++)
            {
                for (int x = 1; x < w - 1; x++)
                {
                    float dpx = (p[y * w + x + 1] - p[y * w + x - 1]) * 0.5f;
                    float dqy = (q[(y + 1) * w + x] - q[(y - 1) * w + x]) * 0.5f;
                    divergence[y * w + x] = dpx + dqy;
                }
            }

            // Solve Laplacian(depth) = divergence using Jacobi iteration
            float[] current = new float[w * h];
            float[] next = new float[w * h];
            int iterations = Math.Max(200, Math.Max(w, h) * 4);

            for (int iter = 0; iter < iterations; iter++)
            {
                for (int y = 1; y < h - 1; y++)
                {
                    for (int x = 1; x < w - 1; x++)
                    {
                        int idx = y * w + x;
                        next[idx] = 0.25f * (
                            current[idx - 1] + current[idx + 1] +
                            current[idx - w] + current[idx + w] -
                            divergence[idx]);
                    }
                }
                // Swap
                (current, next) = (next, current);
            }

            // Center the depth around zero
            float mean = 0;
            int count = 0;
            for (int i = 0; i < current.Length; i++)
            {
                if (current[i] != 0) { mean += current[i]; count++; }
            }
            if (count > 0) mean /= count;
            for (int i = 0; i < current.Length; i++)
                depth[i] = current[i] - mean;

            return depth;
        }

        /// <summary>
        /// Carve the TSDF volume using a single view's depth map.
        /// </summary>
        private void CarveTSDF(float[,,] tsdf, float[,,] weights,
            MultiViewData view, float[] depthMap,
            Vector3 origin, float voxelSize, int res)
        {
            int w = view.Width;
            int h = view.Height;
            var rot = view.Rotation;
            // Inverse rotation to go from world to view space
            var invRot = Matrix3.Transpose(rot);
            float truncation = voxelSize * 3.0f;

            Parallel.For(0, res, z =>
            {
                for (int y = 0; y < res; y++)
                {
                    for (int x = 0; x < res; x++)
                    {
                        // Voxel center in world space
                        var worldPos = origin + new Vector3(
                            (x + 0.5f) * voxelSize,
                            (y + 0.5f) * voxelSize,
                            (z + 0.5f) * voxelSize);

                        // Transform to view space (orthographic)
                        var viewPos = invRot * worldPos;

                        // Project to pixel coordinates (orthographic: u,v map to [-1,1])
                        float u = viewPos.X;
                        float v = viewPos.Y;
                        float worldDepth = viewPos.Z;

                        // Map from [-1,1] to pixel coords
                        int px = (int)((u + 1.0f) * 0.5f * w);
                        int py = (int)((v + 1.0f) * 0.5f * h);

                        if (px < 0 || px >= w || py < 0 || py >= h) continue;

                        float surfaceDepth = depthMap[py * w + px];

                        // Check normal validity at this pixel
                        int nIdx = (py * w + px) * 3;
                        if (nIdx + 2 >= view.Normals.Length) continue;
                        float nz = MathF.Abs(view.Normals[nIdx + 2]);
                        if (nz < 0.05f) continue; // skip grazing angles

                        // Signed distance: positive outside, negative inside
                        float sd = worldDepth - surfaceDepth;

                        // Truncate
                        if (MathF.Abs(sd) > truncation) continue;

                        float truncatedSd = MathF.Max(-1.0f, MathF.Min(1.0f, sd / truncation));
                        float weight = nz; // weight by normal confidence

                        tsdf[x, y, z] += truncatedSd * weight;
                        weights[x, y, z] += weight;
                    }
                }
            });
        }

        /// <summary>
        /// Project colors from multi-view images onto mesh vertices.
        /// </summary>
        private void ProjectColors(MeshData mesh, List<MultiViewData> views)
        {
            mesh.Colors.Clear();

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var vertex = mesh.Vertices[i];
                var bestColor = new Vector3(0.7f, 0.7f, 0.7f);
                float bestWeight = 0;

                // Compute vertex normal for view selection
                var vertNormal = i < mesh.Normals.Count ? mesh.Normals[i] : Vector3.UnitZ;

                for (int v = 0; v < views.Count; v++)
                {
                    var view = views[v];
                    var invRot = Matrix3.Transpose(view.Rotation);
                    var viewPos = invRot * vertex;

                    // Orthographic projection
                    float u = viewPos.X;
                    float vCoord = viewPos.Y;

                    int px = (int)((u + 1.0f) * 0.5f * view.Width);
                    int py = (int)((vCoord + 1.0f) * 0.5f * view.Height);

                    if (px < 0 || px >= view.Width || py < 0 || py >= view.Height) continue;

                    // View direction is the Z axis of the rotation
                    var viewDir = new Vector3(view.Rotation.M13, view.Rotation.M23, view.Rotation.M33);

                    // Weight by alignment of vertex normal with view direction
                    float alignment = MathF.Abs(Vector3.Dot(vertNormal, viewDir));
                    if (alignment < 0.1f) continue;

                    int cIdx = (py * view.Width + px) * 3;
                    if (cIdx + 2 >= view.Colors.Length) continue;

                    float r = Math.Clamp(view.Colors[cIdx], 0, 1);
                    float g = Math.Clamp(view.Colors[cIdx + 1], 0, 1);
                    float b = Math.Clamp(view.Colors[cIdx + 2], 0, 1);

                    if (alignment > bestWeight)
                    {
                        bestWeight = alignment;
                        bestColor = new Vector3(r, g, b);
                    }
                }

                mesh.Colors.Add(bestColor);
            }
        }

        /// <summary>
        /// Compute adaptive iso-level using Otsu's method on the TSDF values.
        /// </summary>
        private float ComputeAdaptiveIsoLevel(float[,,] volume, int res)
        {
            // Histogram of values
            int bins = 256;
            int[] histogram = new int[bins];
            float minVal = float.MaxValue, maxVal = float.MinValue;

            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                    for (int z = 0; z < res; z++)
                    {
                        float v = volume[x, y, z];
                        if (v < minVal) minVal = v;
                        if (v > maxVal) maxVal = v;
                    }

            float range = maxVal - minVal;
            if (range < 1e-6f) return 0;

            int total = res * res * res;
            for (int x = 0; x < res; x++)
                for (int y = 0; y < res; y++)
                    for (int z = 0; z < res; z++)
                    {
                        int bin = Math.Clamp((int)((volume[x, y, z] - minVal) / range * (bins - 1)), 0, bins - 1);
                        histogram[bin]++;
                    }

            // Otsu's threshold
            float bestVariance = 0;
            int bestThreshold = 0;
            float sumAll = 0, sumB = 0;
            int wB = 0;

            for (int i = 0; i < bins; i++) sumAll += i * histogram[i];

            for (int t = 0; t < bins; t++)
            {
                wB += histogram[t];
                if (wB == 0) continue;
                int wF = total - wB;
                if (wF == 0) break;

                sumB += t * histogram[t];
                float mB = sumB / wB;
                float mF = (sumAll - sumB) / wF;
                float variance = (float)wB * wF * (mB - mF) * (mB - mF);

                if (variance > bestVariance)
                {
                    bestVariance = variance;
                    bestThreshold = t;
                }
            }

            return minVal + (bestThreshold / (float)(bins - 1)) * range;
        }

        /// <summary>
        /// Separable 3D Gaussian filter (adapted from GaussianSDFRefiner).
        /// </summary>
        private float[,,] ApplyGaussianFilter(float[,,] input, int w, int h, int d, float sigma, CancellationToken ct)
        {
            int kernelRadius = (int)MathF.Ceiling(sigma * 2.0f);
            float[] kernel = new float[kernelRadius * 2 + 1];
            float sum = 0;
            for (int i = -kernelRadius; i <= kernelRadius; i++)
            {
                float val = MathF.Exp(-(i * i) / (2 * sigma * sigma));
                kernel[i + kernelRadius] = val;
                sum += val;
            }
            for (int i = 0; i < kernel.Length; i++) kernel[i] /= sum;

            float[,,] temp1 = new float[w, h, d];
            float[,,] temp2 = new float[w, h, d];
            float[,,] output = new float[w, h, d];

            // X pass
            Parallel.For(0, d, z =>
            {
                ct.ThrowIfCancellationRequested();
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                            val += input[Math.Clamp(x + k, 0, w - 1), y, z] * kernel[k + kernelRadius];
                        temp1[x, y, z] = val;
                    }
            });

            // Y pass
            Parallel.For(0, d, z =>
            {
                ct.ThrowIfCancellationRequested();
                for (int x = 0; x < w; x++)
                    for (int y = 0; y < h; y++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                            val += temp1[x, Math.Clamp(y + k, 0, h - 1), z] * kernel[k + kernelRadius];
                        temp2[x, y, z] = val;
                    }
            });

            // Z pass
            Parallel.For(0, h, y =>
            {
                ct.ThrowIfCancellationRequested();
                for (int x = 0; x < w; x++)
                    for (int z = 0; z < d; z++)
                    {
                        float val = 0;
                        for (int k = -kernelRadius; k <= kernelRadius; k++)
                            val += temp2[x, y, Math.Clamp(z + k, 0, d - 1)] * kernel[k + kernelRadius];
                        output[x, y, z] = val;
                    }
            });

            return output;
        }
    }
}
