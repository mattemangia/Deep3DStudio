using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using SkiaSharp;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;

namespace Deep3DStudio.Texturing
{
    /// <summary>
    /// Comprehensive texture baking system for projecting camera images onto 3D meshes
    /// </summary>
    public class TextureBaker
    {
        #region Settings

        /// <summary>
        /// Texture resolution for the baked texture atlas
        /// </summary>
        public int TextureSize { get; set; } = 2048;

        /// <summary>
        /// Margin between UV islands (in pixels)
        /// </summary>
        public int IslandMargin { get; set; } = 4;

        /// <summary>
        /// Blending mode for overlapping projections
        /// </summary>
        public TextureBlendMode BlendMode { get; set; } = TextureBlendMode.ViewAngleWeighted;

        /// <summary>
        /// Minimum view angle cosine for projection (faces pointing away are skipped)
        /// </summary>
        public float MinViewAngleCosine { get; set; } = 0.1f;

        /// <summary>
        /// Enable texture seam blending
        /// </summary>
        public bool BlendSeams { get; set; } = true;

        /// <summary>
        /// Number of dilation passes for filling holes
        /// </summary>
        public int DilationPasses { get; set; } = 4;

        #endregion

        #region Public Methods

        /// <summary>
        /// Generates UVs for a mesh using automatic unwrapping
        /// </summary>
        public UVData GenerateUVs(MeshData mesh, UVUnwrapMethod method = UVUnwrapMethod.SmartProject)
        {
            return method switch
            {
                UVUnwrapMethod.SmartProject => GenerateSmartProjectUVs(mesh),
                UVUnwrapMethod.LightmapPack => GenerateLightmapPackUVs(mesh),
                UVUnwrapMethod.BoxProject => GenerateBoxProjectUVs(mesh),
                UVUnwrapMethod.CylindricalProject => GenerateCylindricalProjectUVs(mesh),
                UVUnwrapMethod.SphericalProject => GenerateSphericalProjectUVs(mesh),
                _ => GenerateSmartProjectUVs(mesh)
            };
        }

        /// <summary>
        /// Bakes textures from camera images onto mesh
        /// </summary>
        public BakedTextureResult BakeTextures(MeshData mesh, UVData uvData, List<CameraObject> cameras,
            IProgress<float>? progress = null)
        {
            var result = new BakedTextureResult
            {
                TextureSize = TextureSize,
                DiffuseMap = new SKBitmap(TextureSize, TextureSize, SKColorType.Rgba8888, SKAlphaType.Premul),
                WeightMap = new float[TextureSize, TextureSize]
            };

            // Initialize with transparent
            using (var canvas = new SKCanvas(result.DiffuseMap))
            {
                canvas.Clear(SKColors.Transparent);
            }

            int totalCameras = cameras.Count;
            int processedCameras = 0;

            // Project each camera
            foreach (var camera in cameras.Where(c => c.Pose != null && !string.IsNullOrEmpty(c.ImagePath)))
            {
                ProjectCameraOntoMesh(mesh, uvData, camera, result);

                processedCameras++;
                progress?.Report((float)processedCameras / totalCameras * 0.8f);
            }

            // Post-processing
            if (BlendSeams)
            {
                BlendTextureSeams(result);
            }

            progress?.Report(0.9f);

            // Fill holes with dilation
            DilateTexture(result.DiffuseMap, DilationPasses);

            progress?.Report(1.0f);

            return result;
        }

        /// <summary>
        /// Bakes vertex colors from mesh to texture
        /// </summary>
        public SKBitmap BakeVertexColorsToTexture(MeshData mesh, UVData uvData)
        {
            var texture = new SKBitmap(TextureSize, TextureSize, SKColorType.Rgba8888, SKAlphaType.Premul);

            using (var canvas = new SKCanvas(texture))
            {
                canvas.Clear(SKColors.Transparent);
            }

            // Rasterize each triangle
            for (int t = 0; t < mesh.Indices.Count / 3; t++)
            {
                int i0 = mesh.Indices[t * 3];
                int i1 = mesh.Indices[t * 3 + 1];
                int i2 = mesh.Indices[t * 3 + 2];

                var uv0 = uvData.UVs[i0];
                var uv1 = uvData.UVs[i1];
                var uv2 = uvData.UVs[i2];

                var c0 = mesh.Colors[i0];
                var c1 = mesh.Colors[i1];
                var c2 = mesh.Colors[i2];

                RasterizeTriangle(texture, uv0, uv1, uv2, c0, c1, c2);
            }

            // Dilate to fill gaps
            DilateTexture(texture, DilationPasses);

            return texture;
        }

        #endregion

        #region UV Generation Methods

        private UVData GenerateSmartProjectUVs(MeshData mesh)
        {
            var uvData = new UVData
            {
                UVs = new List<Vector2>()
            };

            // Group faces by normal direction
            var faceGroups = GroupFacesByNormal(mesh, 45.0f); // 45 degree threshold

            // Pack each group as an island
            var islands = new List<UVIsland>();

            foreach (var group in faceGroups)
            {
                var island = ProjectGroupToIsland(mesh, group);
                islands.Add(island);
            }

            // Pack islands into atlas
            PackIslands(islands, out var packedUVs);

            // Build final UV list
            uvData.UVs = new List<Vector2>(new Vector2[mesh.Vertices.Count]);
            foreach (var island in islands)
            {
                foreach (var kvp in island.VertexToUV)
                {
                    uvData.UVs[kvp.Key] = kvp.Value;
                }
            }

            uvData.Islands = islands;
            return uvData;
        }

        private UVData GenerateLightmapPackUVs(MeshData mesh)
        {
            // Lightmap packing treats each triangle as a separate island
            var uvData = new UVData
            {
                UVs = new List<Vector2>(new Vector2[mesh.Vertices.Count])
            };

            var islands = new List<UVIsland>();

            for (int t = 0; t < mesh.Indices.Count / 3; t++)
            {
                int i0 = mesh.Indices[t * 3];
                int i1 = mesh.Indices[t * 3 + 1];
                int i2 = mesh.Indices[t * 3 + 2];

                // Compute 3D area for proportional UV area
                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                float area3D = Vector3.Cross(v1 - v0, v2 - v0).Length * 0.5f;

                // Create island for this triangle
                var island = new UVIsland
                {
                    TriangleIndices = new List<int> { t },
                    VertexToUV = new Dictionary<int, Vector2>
                    {
                        [i0] = new Vector2(0, 0),
                        [i1] = new Vector2(1, 0),
                        [i2] = new Vector2(0.5f, 1)
                    },
                    BoundsMin = Vector2.Zero,
                    BoundsMax = Vector2.One,
                    Area = area3D
                };

                islands.Add(island);
            }

            // Pack islands
            PackIslands(islands, out _);

            // Copy UVs
            foreach (var island in islands)
            {
                foreach (var kvp in island.VertexToUV)
                {
                    uvData.UVs[kvp.Key] = kvp.Value;
                }
            }

            uvData.Islands = islands;
            return uvData;
        }

        private UVData GenerateBoxProjectUVs(MeshData mesh)
        {
            var uvData = new UVData
            {
                UVs = new List<Vector2>()
            };

            // Compute bounding box
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            foreach (var v in mesh.Vertices)
            {
                min = Vector3.ComponentMin(min, v);
                max = Vector3.ComponentMax(max, v);
            }

            var size = max - min;
            float maxSize = Math.Max(size.X, Math.Max(size.Y, size.Z));

            // Compute normals for each triangle and project based on dominant axis
            var normals = ComputeFaceNormals(mesh);

            foreach (var v in mesh.Vertices)
            {
                // For each vertex, use box projection
                var normalized = (v - min) / maxSize;

                // Simple planar projection from top (Y-up)
                uvData.UVs.Add(new Vector2(normalized.X, normalized.Z));
            }

            return uvData;
        }

        private UVData GenerateCylindricalProjectUVs(MeshData mesh)
        {
            var uvData = new UVData
            {
                UVs = new List<Vector2>()
            };

            // Compute centroid
            var centroid = Vector3.Zero;
            foreach (var v in mesh.Vertices)
                centroid += v;
            centroid /= mesh.Vertices.Count;

            // Compute bounding box height
            float minY = mesh.Vertices.Min(v => v.Y);
            float maxY = mesh.Vertices.Max(v => v.Y);
            float height = maxY - minY;

            foreach (var v in mesh.Vertices)
            {
                var relative = v - centroid;

                // Cylindrical projection
                float angle = MathF.Atan2(relative.Z, relative.X);
                float u = (angle + MathF.PI) / (2 * MathF.PI);
                float vCoord = (v.Y - minY) / height;

                uvData.UVs.Add(new Vector2(u, vCoord));
            }

            return uvData;
        }

        private UVData GenerateSphericalProjectUVs(MeshData mesh)
        {
            var uvData = new UVData
            {
                UVs = new List<Vector2>()
            };

            // Compute centroid
            var centroid = Vector3.Zero;
            foreach (var v in mesh.Vertices)
                centroid += v;
            centroid /= mesh.Vertices.Count;

            foreach (var v in mesh.Vertices)
            {
                var dir = (v - centroid).Normalized();

                // Spherical projection
                float u = 0.5f + MathF.Atan2(dir.Z, dir.X) / (2 * MathF.PI);
                float vCoord = 0.5f - MathF.Asin(dir.Y) / MathF.PI;

                uvData.UVs.Add(new Vector2(u, vCoord));
            }

            return uvData;
        }

        #endregion

        #region Projection Methods

        private void ProjectCameraOntoMesh(MeshData mesh, UVData uvData, CameraObject camera, BakedTextureResult result)
        {
            if (camera.Pose == null || string.IsNullOrEmpty(camera.ImagePath)) return;

            // Load camera image
            SKBitmap? cameraImage;
            try
            {
                cameraImage = SKBitmap.Decode(camera.ImagePath);
                if (cameraImage == null) return;
            }
            catch
            {
                return;
            }

            // Get camera matrices
            var viewMatrix = camera.Pose.WorldToCamera;
            var projMatrix = CreatePerspectiveProjection(camera.FieldOfView, camera.AspectRatio, 0.1f, 1000f);
            var cameraPos = camera.Pose.CameraToWorld.ExtractTranslation();

            // Process each triangle
            var faceNormals = ComputeFaceNormals(mesh);

            for (int t = 0; t < mesh.Indices.Count / 3; t++)
            {
                int i0 = mesh.Indices[t * 3];
                int i1 = mesh.Indices[t * 3 + 1];
                int i2 = mesh.Indices[t * 3 + 2];

                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                // Check if face is visible from camera
                var faceNormal = faceNormals[t];
                var faceCenter = (v0 + v1 + v2) / 3;
                var toCamera = (cameraPos - faceCenter).Normalized();

                float viewAngle = Vector3.Dot(faceNormal, toCamera);
                if (viewAngle < MinViewAngleCosine) continue;

                // Project vertices to camera space
                var uv0_cam = ProjectToCamera(v0, viewMatrix, projMatrix, cameraImage.Width, cameraImage.Height);
                var uv1_cam = ProjectToCamera(v1, viewMatrix, projMatrix, cameraImage.Width, cameraImage.Height);
                var uv2_cam = ProjectToCamera(v2, viewMatrix, projMatrix, cameraImage.Width, cameraImage.Height);

                if (!uv0_cam.HasValue || !uv1_cam.HasValue || !uv2_cam.HasValue) continue;

                // Get texture UVs
                var uv0_tex = uvData.UVs[i0];
                var uv1_tex = uvData.UVs[i1];
                var uv2_tex = uvData.UVs[i2];

                // Rasterize triangle
                float weight = ComputeProjectionWeight(viewAngle, BlendMode);
                RasterizeProjectedTriangle(result, cameraImage,
                    uv0_tex, uv1_tex, uv2_tex,
                    uv0_cam.Value, uv1_cam.Value, uv2_cam.Value,
                    weight);
            }

            cameraImage.Dispose();
        }

        private Vector2? ProjectToCamera(Vector3 point, Matrix4 view, Matrix4 proj, int imageWidth, int imageHeight)
        {
            var clipSpace = new Vector4(point, 1.0f) * view * proj;

            if (clipSpace.W <= 0) return null;

            clipSpace /= clipSpace.W;

            // Check if point is within normalized device coordinates
            if (clipSpace.X < -1 || clipSpace.X > 1 ||
                clipSpace.Y < -1 || clipSpace.Y > 1 ||
                clipSpace.Z < -1 || clipSpace.Z > 1)
            {
                return null;
            }

            // Convert to image coordinates
            float u = (clipSpace.X + 1) * 0.5f;
            float v = (1 - clipSpace.Y) * 0.5f; // Flip Y

            return new Vector2(u, v);
        }

        private void RasterizeProjectedTriangle(BakedTextureResult result, SKBitmap sourceImage,
            Vector2 uv0_tex, Vector2 uv1_tex, Vector2 uv2_tex,
            Vector2 uv0_cam, Vector2 uv1_cam, Vector2 uv2_cam,
            float weight)
        {
            // Convert UV to pixel coordinates
            int texSize = result.TextureSize;

            var p0 = new Vector2(uv0_tex.X * texSize, uv0_tex.Y * texSize);
            var p1 = new Vector2(uv1_tex.X * texSize, uv1_tex.Y * texSize);
            var p2 = new Vector2(uv2_tex.X * texSize, uv2_tex.Y * texSize);

            // Compute bounding box
            int minX = Math.Max(0, (int)Math.Min(p0.X, Math.Min(p1.X, p2.X)));
            int maxX = Math.Min(texSize - 1, (int)Math.Max(p0.X, Math.Max(p1.X, p2.X)));
            int minY = Math.Max(0, (int)Math.Min(p0.Y, Math.Min(p1.Y, p2.Y)));
            int maxY = Math.Min(texSize - 1, (int)Math.Max(p0.Y, Math.Max(p1.Y, p2.Y)));

            // Precompute barycentric coefficients
            float denom = (p1.Y - p2.Y) * (p0.X - p2.X) + (p2.X - p1.X) * (p0.Y - p2.Y);
            if (Math.Abs(denom) < 1e-6f) return;

            // Rasterize
            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    var p = new Vector2(x + 0.5f, y + 0.5f);

                    // Compute barycentric coordinates
                    float w0 = ((p1.Y - p2.Y) * (p.X - p2.X) + (p2.X - p1.X) * (p.Y - p2.Y)) / denom;
                    float w1 = ((p2.Y - p0.Y) * (p.X - p2.X) + (p0.X - p2.X) * (p.Y - p2.Y)) / denom;
                    float w2 = 1 - w0 - w1;

                    // Check if inside triangle
                    if (w0 < -0.001f || w1 < -0.001f || w2 < -0.001f) continue;

                    // Interpolate camera UV
                    var camUV = uv0_cam * w0 + uv1_cam * w1 + uv2_cam * w2;

                    // Sample camera image
                    int srcX = (int)(camUV.X * sourceImage.Width);
                    int srcY = (int)(camUV.Y * sourceImage.Height);

                    if (srcX < 0 || srcX >= sourceImage.Width ||
                        srcY < 0 || srcY >= sourceImage.Height)
                        continue;

                    var srcColor = sourceImage.GetPixel(srcX, srcY);

                    // Blend with existing pixel
                    BlendPixel(result, x, y, srcColor, weight);
                }
            }
        }

        private void BlendPixel(BakedTextureResult result, int x, int y, SKColor srcColor, float weight)
        {
            float existingWeight = result.WeightMap[x, y];
            var existingColor = result.DiffuseMap.GetPixel(x, y);

            float totalWeight = existingWeight + weight;

            if (totalWeight > 0)
            {
                byte r = (byte)((existingColor.Red * existingWeight + srcColor.Red * weight) / totalWeight);
                byte g = (byte)((existingColor.Green * existingWeight + srcColor.Green * weight) / totalWeight);
                byte b = (byte)((existingColor.Blue * existingWeight + srcColor.Blue * weight) / totalWeight);
                byte a = 255;

                result.DiffuseMap.SetPixel(x, y, new SKColor(r, g, b, a));
            }

            result.WeightMap[x, y] = totalWeight;
        }

        #endregion

        #region UV Island Packing

        private List<HashSet<int>> GroupFacesByNormal(MeshData mesh, float angleThreshold)
        {
            var normals = ComputeFaceNormals(mesh);
            int faceCount = mesh.Indices.Count / 3;

            var groups = new List<HashSet<int>>();
            var visited = new bool[faceCount];

            // Build face adjacency
            var adjacency = BuildFaceAdjacency(mesh);

            for (int f = 0; f < faceCount; f++)
            {
                if (visited[f]) continue;

                var group = new HashSet<int>();
                var queue = new Queue<int>();
                queue.Enqueue(f);

                while (queue.Count > 0)
                {
                    int face = queue.Dequeue();
                    if (visited[face]) continue;

                    visited[face] = true;
                    group.Add(face);

                    // Check adjacent faces
                    foreach (int adj in adjacency[face])
                    {
                        if (visited[adj]) continue;

                        // Check if normals are similar
                        float dot = Vector3.Dot(normals[face], normals[adj]);
                        if (dot >= MathF.Cos(MathHelper.DegreesToRadians(angleThreshold)))
                        {
                            queue.Enqueue(adj);
                        }
                    }
                }

                if (group.Count > 0)
                    groups.Add(group);
            }

            return groups;
        }

        private UVIsland ProjectGroupToIsland(MeshData mesh, HashSet<int> faceGroup)
        {
            var island = new UVIsland
            {
                TriangleIndices = faceGroup.ToList(),
                VertexToUV = new Dictionary<int, Vector2>()
            };

            // Compute average normal for projection
            var avgNormal = Vector3.Zero;
            foreach (int f in faceGroup)
            {
                var v0 = mesh.Vertices[mesh.Indices[f * 3]];
                var v1 = mesh.Vertices[mesh.Indices[f * 3 + 1]];
                var v2 = mesh.Vertices[mesh.Indices[f * 3 + 2]];

                avgNormal += Vector3.Cross(v1 - v0, v2 - v0);
            }
            avgNormal = avgNormal.Normalized();

            // Create projection basis
            var up = Math.Abs(avgNormal.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX;
            var right = Vector3.Cross(avgNormal, up).Normalized();
            up = Vector3.Cross(right, avgNormal).Normalized();

            // Collect unique vertices and project
            var uniqueVertices = new HashSet<int>();
            foreach (int f in faceGroup)
            {
                uniqueVertices.Add(mesh.Indices[f * 3]);
                uniqueVertices.Add(mesh.Indices[f * 3 + 1]);
                uniqueVertices.Add(mesh.Indices[f * 3 + 2]);
            }

            // Project vertices
            var minUV = new Vector2(float.MaxValue);
            var maxUV = new Vector2(float.MinValue);

            foreach (int v in uniqueVertices)
            {
                var pos = mesh.Vertices[v];
                var uv = new Vector2(Vector3.Dot(pos, right), Vector3.Dot(pos, up));

                island.VertexToUV[v] = uv;
                minUV = Vector2.ComponentMin(minUV, uv);
                maxUV = Vector2.ComponentMax(maxUV, uv);
            }

            // Normalize UVs to 0-1 range
            var uvSize = maxUV - minUV;
            if (uvSize.X > 0 && uvSize.Y > 0)
            {
                foreach (int v in uniqueVertices)
                {
                    var uv = island.VertexToUV[v];
                    island.VertexToUV[v] = (uv - minUV) / uvSize;
                }
            }

            island.BoundsMin = Vector2.Zero;
            island.BoundsMax = Vector2.One;
            island.Area = uvSize.X * uvSize.Y;

            return island;
        }

        private void PackIslands(List<UVIsland> islands, out Dictionary<int, Vector2> packedUVs)
        {
            packedUVs = new Dictionary<int, Vector2>();

            // Sort islands by area (largest first)
            islands = islands.OrderByDescending(i => i.Area).ToList();

            // Simple shelf packing algorithm
            float currentX = 0;
            float currentY = 0;
            float rowHeight = 0;
            float margin = IslandMargin / (float)TextureSize;

            foreach (var island in islands)
            {
                var size = island.BoundsMax - island.BoundsMin;

                // Scale island to fit well
                float scale = Math.Min(0.3f, 1.0f / Math.Max(size.X, size.Y) * 0.2f);
                size *= scale;

                // Check if fits in current row
                if (currentX + size.X + margin > 1.0f)
                {
                    currentX = 0;
                    currentY += rowHeight + margin;
                    rowHeight = 0;
                }

                // Position island
                var offset = new Vector2(currentX + margin, currentY + margin);

                // Transform island UVs
                foreach (var kvp in island.VertexToUV.ToList())
                {
                    island.VertexToUV[kvp.Key] = kvp.Value * scale + offset;
                }

                island.BoundsMin = offset;
                island.BoundsMax = offset + size;

                currentX += size.X + margin;
                rowHeight = Math.Max(rowHeight, size.Y);
            }
        }

        #endregion

        #region Helper Methods

        private List<HashSet<int>> BuildFaceAdjacency(MeshData mesh)
        {
            int faceCount = mesh.Indices.Count / 3;
            var adjacency = new List<HashSet<int>>();

            for (int i = 0; i < faceCount; i++)
                adjacency.Add(new HashSet<int>());

            // Build edge -> faces map
            var edgeToFaces = new Dictionary<(int, int), List<int>>();

            for (int f = 0; f < faceCount; f++)
            {
                int i0 = mesh.Indices[f * 3];
                int i1 = mesh.Indices[f * 3 + 1];
                int i2 = mesh.Indices[f * 3 + 2];

                AddEdgeFace(edgeToFaces, i0, i1, f);
                AddEdgeFace(edgeToFaces, i1, i2, f);
                AddEdgeFace(edgeToFaces, i2, i0, f);
            }

            // Build adjacency from shared edges
            foreach (var faces in edgeToFaces.Values)
            {
                for (int i = 0; i < faces.Count; i++)
                {
                    for (int j = i + 1; j < faces.Count; j++)
                    {
                        adjacency[faces[i]].Add(faces[j]);
                        adjacency[faces[j]].Add(faces[i]);
                    }
                }
            }

            return adjacency;
        }

        private void AddEdgeFace(Dictionary<(int, int), List<int>> edgeToFaces, int v1, int v2, int face)
        {
            var key = v1 < v2 ? (v1, v2) : (v2, v1);
            if (!edgeToFaces.ContainsKey(key))
                edgeToFaces[key] = new List<int>();
            edgeToFaces[key].Add(face);
        }

        private List<Vector3> ComputeFaceNormals(MeshData mesh)
        {
            var normals = new List<Vector3>();

            for (int f = 0; f < mesh.Indices.Count / 3; f++)
            {
                var v0 = mesh.Vertices[mesh.Indices[f * 3]];
                var v1 = mesh.Vertices[mesh.Indices[f * 3 + 1]];
                var v2 = mesh.Vertices[mesh.Indices[f * 3 + 2]];

                var normal = Vector3.Cross(v1 - v0, v2 - v0);
                if (normal.LengthSquared > 0)
                    normal = normal.Normalized();

                normals.Add(normal);
            }

            return normals;
        }

        private float ComputeProjectionWeight(float viewAngle, TextureBlendMode mode)
        {
            return mode switch
            {
                TextureBlendMode.Replace => 1.0f,
                TextureBlendMode.Average => 1.0f,
                TextureBlendMode.ViewAngleWeighted => viewAngle * viewAngle, // Favor perpendicular views
                TextureBlendMode.DistanceWeighted => 1.0f,
                _ => 1.0f
            };
        }

        private Matrix4 CreatePerspectiveProjection(float fov, float aspect, float near, float far)
        {
            return Matrix4.CreatePerspectiveFieldOfView(
                MathHelper.DegreesToRadians(fov), aspect, near, far);
        }

        private void RasterizeTriangle(SKBitmap texture, Vector2 uv0, Vector2 uv1, Vector2 uv2,
            Vector3 c0, Vector3 c1, Vector3 c2)
        {
            int texSize = texture.Width;

            var p0 = new Vector2(uv0.X * texSize, uv0.Y * texSize);
            var p1 = new Vector2(uv1.X * texSize, uv1.Y * texSize);
            var p2 = new Vector2(uv2.X * texSize, uv2.Y * texSize);

            int minX = Math.Max(0, (int)Math.Min(p0.X, Math.Min(p1.X, p2.X)));
            int maxX = Math.Min(texSize - 1, (int)Math.Max(p0.X, Math.Max(p1.X, p2.X)));
            int minY = Math.Max(0, (int)Math.Min(p0.Y, Math.Min(p1.Y, p2.Y)));
            int maxY = Math.Min(texSize - 1, (int)Math.Max(p0.Y, Math.Max(p1.Y, p2.Y)));

            float denom = (p1.Y - p2.Y) * (p0.X - p2.X) + (p2.X - p1.X) * (p0.Y - p2.Y);
            if (Math.Abs(denom) < 1e-6f) return;

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    var p = new Vector2(x + 0.5f, y + 0.5f);

                    float w0 = ((p1.Y - p2.Y) * (p.X - p2.X) + (p2.X - p1.X) * (p.Y - p2.Y)) / denom;
                    float w1 = ((p2.Y - p0.Y) * (p.X - p2.X) + (p0.X - p2.X) * (p.Y - p2.Y)) / denom;
                    float w2 = 1 - w0 - w1;

                    if (w0 < -0.001f || w1 < -0.001f || w2 < -0.001f) continue;

                    var color = c0 * w0 + c1 * w1 + c2 * w2;
                    byte r = (byte)(Math.Clamp(color.X, 0, 1) * 255);
                    byte g = (byte)(Math.Clamp(color.Y, 0, 1) * 255);
                    byte b = (byte)(Math.Clamp(color.Z, 0, 1) * 255);

                    texture.SetPixel(x, y, new SKColor(r, g, b, 255));
                }
            }
        }

        private void BlendTextureSeams(BakedTextureResult result)
        {
            // Simple seam blending using weighted blur at edges
            int texSize = result.TextureSize;
            var blurred = new SKBitmap(texSize, texSize, SKColorType.Rgba8888, SKAlphaType.Premul);

            using (var canvas = new SKCanvas(blurred))
            {
                canvas.Clear(SKColors.Transparent);
            }

            // Apply 3x3 blur to low-weight areas
            for (int y = 1; y < texSize - 1; y++)
            {
                for (int x = 1; x < texSize - 1; x++)
                {
                    if (result.WeightMap[x, y] < 0.1f)
                    {
                        // Average surrounding pixels
                        int r = 0, g = 0, b = 0, count = 0;

                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (result.WeightMap[x + dx, y + dy] > 0.1f)
                                {
                                    var c = result.DiffuseMap.GetPixel(x + dx, y + dy);
                                    r += c.Red;
                                    g += c.Green;
                                    b += c.Blue;
                                    count++;
                                }
                            }
                        }

                        if (count > 0)
                        {
                            blurred.SetPixel(x, y, new SKColor(
                                (byte)(r / count),
                                (byte)(g / count),
                                (byte)(b / count),
                                255));
                        }
                    }
                    else
                    {
                        blurred.SetPixel(x, y, result.DiffuseMap.GetPixel(x, y));
                    }
                }
            }

            // Copy back
            for (int y = 0; y < texSize; y++)
            {
                for (int x = 0; x < texSize; x++)
                {
                    result.DiffuseMap.SetPixel(x, y, blurred.GetPixel(x, y));
                }
            }

            blurred.Dispose();
        }

        private void DilateTexture(SKBitmap texture, int passes)
        {
            int w = texture.Width;
            int h = texture.Height;

            for (int pass = 0; pass < passes; pass++)
            {
                var temp = new SKBitmap(w, h, SKColorType.Rgba8888, SKAlphaType.Premul);

                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        var current = texture.GetPixel(x, y);

                        if (current.Alpha < 128)
                        {
                            // Find valid neighbor
                            int r = 0, g = 0, b = 0, count = 0;

                            for (int dy = -1; dy <= 1; dy++)
                            {
                                for (int dx = -1; dx <= 1; dx++)
                                {
                                    int nx = x + dx;
                                    int ny = y + dy;

                                    if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                                    {
                                        var neighbor = texture.GetPixel(nx, ny);
                                        if (neighbor.Alpha >= 128)
                                        {
                                            r += neighbor.Red;
                                            g += neighbor.Green;
                                            b += neighbor.Blue;
                                            count++;
                                        }
                                    }
                                }
                            }

                            if (count > 0)
                            {
                                temp.SetPixel(x, y, new SKColor(
                                    (byte)(r / count),
                                    (byte)(g / count),
                                    (byte)(b / count),
                                    255));
                            }
                            else
                            {
                                temp.SetPixel(x, y, current);
                            }
                        }
                        else
                        {
                            temp.SetPixel(x, y, current);
                        }
                    }
                }

                // Copy back
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        texture.SetPixel(x, y, temp.GetPixel(x, y));
                    }
                }

                temp.Dispose();
            }
        }

        #endregion
    }

    #region Supporting Types

    public enum UVUnwrapMethod
    {
        SmartProject,
        LightmapPack,
        BoxProject,
        CylindricalProject,
        SphericalProject
    }

    public enum TextureBlendMode
    {
        Replace,
        Average,
        ViewAngleWeighted,
        DistanceWeighted
    }

    public class UVData
    {
        public List<Vector2> UVs { get; set; } = new List<Vector2>();
        public List<UVIsland> Islands { get; set; } = new List<UVIsland>();
    }

    public class UVIsland
    {
        public List<int> TriangleIndices { get; set; } = new List<int>();
        public Dictionary<int, Vector2> VertexToUV { get; set; } = new Dictionary<int, Vector2>();
        public Vector2 BoundsMin { get; set; }
        public Vector2 BoundsMax { get; set; }
        public float Area { get; set; }
    }

    public class BakedTextureResult : IDisposable
    {
        public int TextureSize { get; set; }
        public SKBitmap DiffuseMap { get; set; } = null!;
        public float[,] WeightMap { get; set; } = null!;
        public SKBitmap? NormalMap { get; set; }
        public SKBitmap? AmbientOcclusionMap { get; set; }

        public void Dispose()
        {
            DiffuseMap?.Dispose();
            NormalMap?.Dispose();
            AmbientOcclusionMap?.Dispose();
        }
    }

    #endregion
}
