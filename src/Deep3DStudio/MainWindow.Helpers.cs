using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private MeshData? GetSelectedMesh()
        {
            var selected = _sceneGraph.GetSelectedObjects();
            foreach (var obj in selected)
            {
                if (obj is Scene.MeshObject meshObj)
                {
                    return meshObj.MeshData;
                }
            }
            return null;
        }

        private void ShowMessage(string title, string message)
        {
            var dialog = new MessageDialog(this, DialogFlags.Modal, MessageType.Info, ButtonsType.Ok, message);
            dialog.Title = title;
            dialog.Run();
            dialog.Destroy();
        }

        private void ShowMessage(string message)
        {
            var md = new MessageDialog(this, DialogFlags.Modal, MessageType.Info, ButtonsType.Ok, message);
            md.Run();
            md.Destroy();
        }

        private void PopulateDepthData(SceneResult result)
        {
            if (result.Poses.Count == 0 || result.Meshes.Count == 0) return;

            // SfM produces one combined mesh with all points, so use the same mesh for all cameras
            // Dust3r might produce per-camera meshes, so handle both cases
            var combinedMesh = result.Meshes[0];
            if (result.Meshes.Count > 1)
            {
                // Combine all meshes if there are multiple
                combinedMesh = new MeshData();
                foreach (var m in result.Meshes)
                {
                    combinedMesh.Vertices.AddRange(m.Vertices);
                    combinedMesh.Colors.AddRange(m.Colors);
                }
            }

            for (int i = 0; i < result.Poses.Count; i++)
            {
                var pose = result.Poses[i];
                // Pass camera-specific focal length for accurate depth projection
                var depthMap = ExtractDepthMap(combinedMesh, pose.Width, pose.Height, pose.WorldToCamera, pose.GetEffectiveFocalLength());
                _imageBrowser.SetDepthData(i, depthMap);
            }

            _imageBrowser.QueueDraw();
        }

        private IMesher GetMesher(MeshingAlgorithm algo)
        {
            switch (algo)
            {
                case MeshingAlgorithm.Poisson: return new PoissonMesher();
                case MeshingAlgorithm.GreedyMeshing: return new GreedyMesher();
                case MeshingAlgorithm.SurfaceNets: return new SurfaceNetsMesher();
                case MeshingAlgorithm.Blocky: return new BlockMesher();
                case MeshingAlgorithm.DeepMeshPrior:
                case MeshingAlgorithm.TripoSF:
                case MeshingAlgorithm.LGM:
                    Console.WriteLine($"Meshing algorithm {algo} is AI-driven; falling back to MarchingCubes for geometry extraction.");
                    return new MarchingCubesMesher();
                case MeshingAlgorithm.MarchingCubes:
                default:
                    return new MarchingCubesMesher();
            }
        }

        private float[,] ExtractDepthMap(MeshData mesh, int width, int height, OpenTK.Mathematics.Matrix4 worldToCamera, float focalLength = 0)
        {
            float[,] depthMap = new float[width, height];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    depthMap[x, y] = -1.0f; // Initialize as invalid/background

            if (mesh.PixelToVertexIndex != null && mesh.PixelToVertexIndex.Length == width * height)
            {
                // Dense mesh logic (Dust3r)
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pIdx = y * width + x;
                        int vertIdx = mesh.PixelToVertexIndex[pIdx];
                        if (vertIdx >= 0 && vertIdx < mesh.Vertices.Count)
                        {
                            var v = mesh.Vertices[vertIdx];
                            var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                            depthMap[x, y] = Math.Abs(vCam.Z);
                        }
                    }
                }
            }
            else
            {
                // Sparse Point Cloud Logic (SfM)
                // Use provided focal length if available, otherwise estimate
                float focal = focalLength > 0 ? focalLength : Math.Max(width, height) * 0.85f;
                float cx = width / 2.0f;
                float cy = height / 2.0f;

                int projectedCount = 0;
                int splatRadius = 3; // Splatting to densify sparse points

                foreach (var v in mesh.Vertices)
                {
                    // Transform to camera space
                    var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                    float depth = Math.Abs(vCam.Z);

                    // Check if point is roughly in front (ignoring very close clipping)
                    if (depth < 0.1f) continue;

                    // Project to image plane
                    // Handle both +Z and -Z conventions by checking sign
                    int px, py;

                    if (vCam.Z < 0) // OpenGL style (-Z fwd)
                    {
                        px = (int)(-focal * vCam.X / vCam.Z + cx);
                        py = (int)(-focal * vCam.Y / vCam.Z + cy);
                    }
                    else // OpenCV style (+Z fwd)
                    {
                        px = (int)(focal * vCam.X / vCam.Z + cx);
                        py = (int)(focal * vCam.Y / vCam.Z + cy);
                    }

                    // Splatting loop
                    for (int dy = -splatRadius; dy <= splatRadius; dy++)
                    {
                        for (int dx = -splatRadius; dx <= splatRadius; dx++)
                        {
                            // Circular kernel
                            if (dx * dx + dy * dy > splatRadius * splatRadius) continue;

                            int nx = px + dx;
                            int ny = py + dy;

                            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                            {
                                // Z-Buffer test (keeping smallest depth)
                                if (depthMap[nx, ny] < 0 || depth < depthMap[nx, ny])
                                {
                                    depthMap[nx, ny] = depth;
                                    projectedCount++;
                                }
                            }
                        }
                    }
                }

                // Fill gaps to make it look solid
                FillDepthMapGaps(depthMap, width, height);
                Console.WriteLine($"  Depth map generated (Sparse -> Dense).");
            }

            return depthMap;
        }

        /// <summary>
        /// Extract depth map without gap filling - only real sparse point depths.
        /// Uses larger splatting radius to create small patches around each point.
        /// </summary>
        private float[,] ExtractDepthMapNoFill(MeshData mesh, int width, int height, OpenTK.Mathematics.Matrix4 worldToCamera, float focalLength = 0)
        {
            float[,] depthMap = new float[width, height];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    depthMap[x, y] = -1.0f; // Initialize as invalid

            float focal = focalLength > 0 ? focalLength : Math.Max(width, height) * 0.85f;
            float cx = width / 2.0f;
            float cy = height / 2.0f;

            int splatRadius = 5; // Larger radius to create more coverage from sparse points
            int projectedCount = 0;

            foreach (var v in mesh.Vertices)
            {
                // Transform to camera space
                var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                float depth = Math.Abs(vCam.Z);

                if (depth < 0.1f) continue;

                // Project to image plane
                int px, py;
                if (vCam.Z < 0) // OpenGL style (-Z fwd)
                {
                    px = (int)(-focal * vCam.X / vCam.Z + cx);
                    py = (int)(-focal * vCam.Y / vCam.Z + cy);
                }
                else // OpenCV style (+Z fwd)
                {
                    px = (int)(focal * vCam.X / vCam.Z + cx);
                    py = (int)(focal * vCam.Y / vCam.Z + cy);
                }

                // Splatting with larger radius
                for (int dy = -splatRadius; dy <= splatRadius; dy++)
                {
                    for (int dx = -splatRadius; dx <= splatRadius; dx++)
                    {
                        if (dx * dx + dy * dy > splatRadius * splatRadius) continue;

                        int nx = px + dx;
                        int ny = py + dy;

                        if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                        {
                            if (depthMap[nx, ny] < 0 || depth < depthMap[nx, ny])
                            {
                                depthMap[nx, ny] = depth;
                                projectedCount++;
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"  Depth map (no fill): {projectedCount} pixels from {mesh.Vertices.Count} points");
            return depthMap;
        }

        private void FillDepthMapGaps(float[,] depthMap, int width, int height)
        {
            // Pyramid Filling (Push-Pull) for dense coverage
            // 1. Build MipMaps (Downsampling)
            int levels = 6; // Enough to bridge large gaps (2^6 = 64 pixels)
            var pyramid = new List<float[,]>();
            pyramid.Add(depthMap);

            for (int l = 1; l < levels; l++)
            {
                int prevW = pyramid[l - 1].GetLength(0);
                int prevH = pyramid[l - 1].GetLength(1);
                int newW = (prevW + 1) / 2;
                int newH = (prevH + 1) / 2;

                float[,] mip = new float[newW, newH];

                Parallel.For(0, newH, y =>
                {
                    for (int x = 0; x < newW; x++)
                    {
                        // Downsample: Use Min Positive Depth (Foreground) to preserve edges
                        float minPos = float.MaxValue;
                        bool found = false;

                        for (int dy = 0; dy < 2; dy++)
                        {
                            for (int dx = 0; dx < 2; dx++)
                            {
                                int sx = x * 2 + dx;
                                int sy = y * 2 + dy;
                                if (sx < prevW && sy < prevH)
                                {
                                    float val = pyramid[l - 1][sx, sy];
                                    if (val > 0)
                                    {
                                        if (val < minPos) minPos = val;
                                        found = true;
                                    }
                                }
                            }
                        }
                        mip[x, y] = found ? minPos : -1.0f;
                    }
                });
                pyramid.Add(mip);
            }

            // 2. Upsample and Fill Gaps
            for (int l = levels - 1; l > 0; l--)
            {
                float[,] current = pyramid[l];
                float[,] target = pyramid[l - 1];

                int tw = target.GetLength(0);
                int th = target.GetLength(1);
                int cw = current.GetLength(0);
                int ch = current.GetLength(1);

                Parallel.For(0, th, y =>
                {
                    for (int x = 0; x < tw; x++)
                    {
                        if (target[x, y] <= 0) // If gap
                        {
                            // Nearest neighbor from coarse level
                            int cx = x / 2;
                            int cy = y / 2;
                            if (cx < cw && cy < ch)
                            {
                                float val = current[cx, cy];
                                if (val > 0) target[x, y] = val;
                            }
                        }
                    }
                });
            }

            // 3. Final Smoothing Pass - Edge-preserving bilateral-like filter
            // Only smooth pixels that are close in depth to their neighbors
            // This preserves depth discontinuities at object boundaries
            float[,] smoothed = (float[,])depthMap.Clone();
            float depthThreshold = 0.15f; // Only blend depths within 15% of each other

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    float centerDepth = depthMap[x, y];
                    if (centerDepth <= 0) continue;

                    float sum = centerDepth;
                    float weight = 1.0f;

                    for (int dy = -1; dy <= 1; dy++)
                    {
                        int ny = y + dy;
                        if (ny < 0 || ny >= height) continue;
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            int nx = x + dx;
                            if (nx < 0 || nx >= width) continue;
                            float v = depthMap[nx, ny];
                            if (v <= 0) continue;

                            // Only include neighbors with similar depth (edge-preserving)
                            float depthDiff = Math.Abs(v - centerDepth) / centerDepth;
                            if (depthDiff < depthThreshold)
                            {
                                // Weight by inverse distance
                                float w = (dx == 0 || dy == 0) ? 1.0f : 0.7f;
                                sum += v * w;
                                weight += w;
                            }
                        }
                    }
                    smoothed[x, y] = sum / weight;
                }
            });

            // Copy back
            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++) depthMap[x, y] = smoothed[x, y];
            });
        }

        private MeshData GenerateDensePointCloud(SceneResult result)
        {
            var denseMesh = new MeshData();
            var lockObj = new object();

            // Use the sparse mesh to generate depth maps first
            if (result.Meshes.Count == 0) return denseMesh;
            var sparseMesh = result.Meshes[0];

            // Calculate sparse point cloud bounds for validation
            var sparseMin = new OpenTK.Mathematics.Vector3(float.MaxValue);
            var sparseMax = new OpenTK.Mathematics.Vector3(float.MinValue);
            foreach (var v in sparseMesh.Vertices)
            {
                sparseMin = OpenTK.Mathematics.Vector3.ComponentMin(sparseMin, v);
                sparseMax = OpenTK.Mathematics.Vector3.ComponentMax(sparseMax, v);
            }
            var sparseExtent = sparseMax - sparseMin;
            float margin = Math.Max(sparseExtent.X, Math.Max(sparseExtent.Y, sparseExtent.Z)) * 0.5f;

            Parallel.ForEach(result.Poses, pose =>
            {
                // 1. Generate depth map from sparse points (NO gap filling - we want only real geometry)
                float focal = pose.GetEffectiveFocalLength();
                var depthMap = ExtractDepthMapNoFill(sparseMesh, pose.Width, pose.Height, pose.WorldToCamera, focal);

                // 2. Load Image for Colors
                if (!System.IO.File.Exists(pose.ImagePath)) return;

                using var img = SkiaSharp.SKBitmap.Decode(pose.ImagePath);
                if (img == null) return;

                float scaleX = (float)img.Width / pose.Width;
                float scaleY = (float)img.Height / pose.Height;

                // 3. Back-project only pixels with valid depth (no interpolated/filled values)
                float cx = pose.Width / 2.0f;
                float cy = pose.Height / 2.0f;

                var localVerts = new List<OpenTK.Mathematics.Vector3>();
                var localColors = new List<OpenTK.Mathematics.Vector3>();

                // Use smaller stride for denser output where we have valid depth
                int stride = 2;

                for (int y = 0; y < pose.Height; y += stride)
                {
                    for (int x = 0; x < pose.Width; x += stride)
                    {
                        float d = depthMap[x, y];
                        if (d <= 0) continue; // No valid depth at this pixel

                        // Back-project to camera space
                        // SfM poses are converted to an OpenGL-style frame (Y-up, -Z forward).
                        // Pixel coordinates remain in image space (Y-down), so we need to flip
                        // the Y component to match the camera pose convention; otherwise the
                        // reconstructed cloud appears upside down.
                        float z_cam = -d; // Camera looks down -Z after conversion
                        float x_cam = (x - cx) * d / focal;
                        float y_cam = -(y - cy) * d / focal; // Flip Y to convert from image to camera coords

                        var pCam = new OpenTK.Mathematics.Vector3(x_cam, y_cam, z_cam);
                        var pWorld = OpenTK.Mathematics.Vector3.TransformPosition(pCam, pose.CameraToWorld);

                        // Bounds check: only add points within reasonable distance of sparse cloud
                        if (pWorld.X < sparseMin.X - margin || pWorld.X > sparseMax.X + margin ||
                            pWorld.Y < sparseMin.Y - margin || pWorld.Y > sparseMax.Y + margin ||
                            pWorld.Z < sparseMin.Z - margin || pWorld.Z > sparseMax.Z + margin)
                        {
                            continue;
                        }

                        // Color sampling from original image
                        int imgX = (int)(x * scaleX);
                        int imgY = (int)(y * scaleY);
                        imgX = Math.Clamp(imgX, 0, img.Width - 1);
                        imgY = Math.Clamp(imgY, 0, img.Height - 1);

                        var color = img.GetPixel(imgX, imgY);

                        localVerts.Add(pWorld);
                        localColors.Add(new OpenTK.Mathematics.Vector3(color.Red / 255f, color.Green / 255f, color.Blue / 255f));
                    }
                }

                lock (lockObj)
                {
                    denseMesh.Vertices.AddRange(localVerts);
                    denseMesh.Colors.AddRange(localColors);
                }
            });

            // Log dense cloud stats
            if (denseMesh.Vertices.Count > 0)
            {
                var denseMin = new OpenTK.Mathematics.Vector3(float.MaxValue);
                var denseMax = new OpenTK.Mathematics.Vector3(float.MinValue);
                foreach (var v in denseMesh.Vertices)
                {
                    denseMin = OpenTK.Mathematics.Vector3.ComponentMin(denseMin, v);
                    denseMax = OpenTK.Mathematics.Vector3.ComponentMax(denseMax, v);
                }
                Console.WriteLine($"Dense cloud: {denseMesh.Vertices.Count} points, bounds ({denseMin.X:F2},{denseMin.Y:F2},{denseMin.Z:F2}) to ({denseMax.X:F2},{denseMax.Y:F2},{denseMax.Z:F2})");
            }

            return denseMesh;
        }

        private (float[,,], OpenTK.Mathematics.Vector3, float) VoxelizePoints(List<MeshData> meshes, int maxRes = 200)
        {
            var min = new OpenTK.Mathematics.Vector3(float.MaxValue);
            var max = new OpenTK.Mathematics.Vector3(float.MinValue);
            foreach (var m in meshes)
            {
                foreach (var v in m.Vertices)
                {
                    min = OpenTK.Mathematics.Vector3.ComponentMin(min, v);
                    max = OpenTK.Mathematics.Vector3.ComponentMax(max, v);
                }
            }

            float voxelSize = 0.02f;
            int w = (int)((max.X - min.X) / voxelSize) + 5;
            int h = (int)((max.Y - min.Y) / voxelSize) + 5;
            int d = (int)((max.Z - min.Z) / voxelSize) + 5;

            if (w > maxRes)
            {
                voxelSize *= (w / (float)maxRes);
                w = maxRes;
                h = (int)((max.Y - min.Y) / voxelSize) + 5;
                d = (int)((max.Z - min.Z) / voxelSize) + 5;
            }

            float[,,] grid = new float[w, h, d];

            foreach (var m in meshes)
            {
                foreach (var v in m.Vertices)
                {
                    int x = (int)((v.X - min.X) / voxelSize);
                    int y = (int)((v.Y - min.Y) / voxelSize);
                    int z = (int)((v.Z - min.Z) / voxelSize);
                    if (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d)
                    {
                        grid[x, y, z] = 1.0f;
                    }
                }
            }

            float[,,] smooth = new float[w, h, d];
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 1; y++)
                    for (int z = 1; z < d - 1; z++)
                    {
                        if (grid[x, y, z] > 0)
                        {
                            smooth[x, y, z] = 1;
                            smooth[x + 1, y, z] = 1; smooth[x - 1, y, z] = 1;
                            smooth[x, y + 1, z] = 1; smooth[x, y - 1, z] = 1;
                            smooth[x, y, z + 1] = 1; smooth[x, y, z - 1] = 1;
                        }
                    }

            return (smooth, min, voxelSize);
        }
    }
}
