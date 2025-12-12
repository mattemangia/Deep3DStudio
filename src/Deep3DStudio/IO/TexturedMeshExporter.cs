using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OpenTK.Mathematics;
using SkiaSharp;
using Deep3DStudio.Model;
using Deep3DStudio.Texturing;

namespace Deep3DStudio.IO
{
    /// <summary>
    /// Export formats for textured meshes
    /// </summary>
    public enum TexturedMeshFormat
    {
        OBJ,        // Wavefront OBJ with MTL
        GLTF,       // glTF 2.0 (separate files)
        GLB,        // glTF 2.0 binary
        FBX_ASCII,  // Autodesk FBX ASCII
        PLY         // PLY with UV attributes
    }

    /// <summary>
    /// Texture export format
    /// </summary>
    public enum TextureFormat
    {
        PNG,
        JPEG,
        TGA,
        BMP
    }

    /// <summary>
    /// Options for mesh export
    /// </summary>
    public class MeshExportOptions
    {
        public TexturedMeshFormat Format { get; set; } = TexturedMeshFormat.OBJ;
        public TextureFormat TextureFormat { get; set; } = TextureFormat.PNG;
        public bool ExportNormals { get; set; } = true;
        public bool ExportColors { get; set; } = true;
        public bool ExportUVs { get; set; } = true;
        public bool ExportTextures { get; set; } = true;
        public int JpegQuality { get; set; } = 90;
        public bool EmbedTextures { get; set; } = false;
        public float Scale { get; set; } = 1.0f;
        public bool SwapYZ { get; set; } = false;
        public bool FlipNormals { get; set; } = false;
    }

    /// <summary>
    /// Exports textured meshes to various formats
    /// </summary>
    public static class TexturedMeshExporter
    {
        #region Public Methods

        /// <summary>
        /// Exports a mesh with optional UVs and textures
        /// </summary>
        public static void Export(string filePath, MeshData mesh, UVData? uvData,
            BakedTextureResult? texture, MeshExportOptions options)
        {
            switch (options.Format)
            {
                case TexturedMeshFormat.OBJ:
                    ExportOBJ(filePath, mesh, uvData, texture, options);
                    break;
                case TexturedMeshFormat.GLTF:
                    ExportGLTF(filePath, mesh, uvData, texture, options, false);
                    break;
                case TexturedMeshFormat.GLB:
                    ExportGLTF(filePath, mesh, uvData, texture, options, true);
                    break;
                case TexturedMeshFormat.FBX_ASCII:
                    ExportFBXAscii(filePath, mesh, uvData, texture, options);
                    break;
                case TexturedMeshFormat.PLY:
                    ExportPLY(filePath, mesh, uvData, options);
                    break;
            }
        }

        /// <summary>
        /// Exports only the texture
        /// </summary>
        public static void ExportTexture(string filePath, SKBitmap texture, TextureFormat format, int jpegQuality = 90)
        {
            using var stream = File.OpenWrite(filePath);

            var encodeFormat = format switch
            {
                TextureFormat.PNG => SKEncodedImageFormat.Png,
                TextureFormat.JPEG => SKEncodedImageFormat.Jpeg,
                TextureFormat.BMP => SKEncodedImageFormat.Bmp,
                _ => SKEncodedImageFormat.Png
            };

            texture.Encode(stream, encodeFormat, jpegQuality);
        }

        #endregion

        #region OBJ Export

        private static void ExportOBJ(string filePath, MeshData mesh, UVData? uvData,
            BakedTextureResult? texture, MeshExportOptions options)
        {
            string directory = Path.GetDirectoryName(filePath) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(filePath);
            string mtlFileName = baseName + ".mtl";
            string textureFileName = baseName + "_diffuse" + GetTextureExtension(options.TextureFormat);

            var sb = new StringBuilder();
            var culture = CultureInfo.InvariantCulture;

            // Header
            sb.AppendLine("# Deep3DStudio OBJ Export");
            sb.AppendLine($"# Vertices: {mesh.Vertices.Count}");
            sb.AppendLine($"# Triangles: {mesh.Indices.Count / 3}");
            sb.AppendLine();

            if (texture != null && options.ExportTextures)
            {
                sb.AppendLine($"mtllib {mtlFileName}");
                sb.AppendLine();
            }

            // Compute normals if needed
            Vector3[]? normals = null;
            if (options.ExportNormals)
            {
                normals = ComputeVertexNormals(mesh);
            }

            // Vertices
            sb.AppendLine("# Vertices");
            foreach (var v in mesh.Vertices)
            {
                var scaled = v * options.Scale;
                if (options.SwapYZ)
                {
                    sb.AppendLine(string.Format(culture, "v {0:F6} {1:F6} {2:F6}", scaled.X, scaled.Z, scaled.Y));
                }
                else
                {
                    sb.AppendLine(string.Format(culture, "v {0:F6} {1:F6} {2:F6}", scaled.X, scaled.Y, scaled.Z));
                }
            }
            sb.AppendLine();

            // Texture coordinates
            if (uvData != null && options.ExportUVs)
            {
                sb.AppendLine("# Texture Coordinates");
                foreach (var uv in uvData.UVs)
                {
                    sb.AppendLine(string.Format(culture, "vt {0:F6} {1:F6}", uv.X, 1.0f - uv.Y)); // Flip V
                }
                sb.AppendLine();
            }

            // Normals
            if (normals != null)
            {
                sb.AppendLine("# Normals");
                foreach (var n in normals)
                {
                    var normal = options.FlipNormals ? -n : n;
                    if (options.SwapYZ)
                    {
                        sb.AppendLine(string.Format(culture, "vn {0:F6} {1:F6} {2:F6}", normal.X, normal.Z, normal.Y));
                    }
                    else
                    {
                        sb.AppendLine(string.Format(culture, "vn {0:F6} {1:F6} {2:F6}", normal.X, normal.Y, normal.Z));
                    }
                }
                sb.AppendLine();
            }

            // Vertex colors as comments
            if (options.ExportColors && mesh.Colors.Count == mesh.Vertices.Count)
            {
                sb.AppendLine("# Vertex Colors (as comments for reference)");
                for (int i = 0; i < mesh.Colors.Count; i++)
                {
                    var c = mesh.Colors[i];
                    sb.AppendLine($"# vc {i + 1} {c.X:F4} {c.Y:F4} {c.Z:F4}");
                }
                sb.AppendLine();
            }

            // Use material
            if (texture != null && options.ExportTextures)
            {
                sb.AppendLine($"usemtl {baseName}_material");
                sb.AppendLine();
            }

            // Faces
            sb.AppendLine("# Faces");
            bool hasUVs = uvData != null && options.ExportUVs;
            bool hasNormals = normals != null;

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i] + 1;     // OBJ uses 1-based indices
                int i1 = mesh.Indices[i + 1] + 1;
                int i2 = mesh.Indices[i + 2] + 1;

                if (hasUVs && hasNormals)
                {
                    sb.AppendLine($"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}");
                }
                else if (hasUVs)
                {
                    sb.AppendLine($"f {i0}/{i0} {i1}/{i1} {i2}/{i2}");
                }
                else if (hasNormals)
                {
                    sb.AppendLine($"f {i0}//{i0} {i1}//{i1} {i2}//{i2}");
                }
                else
                {
                    sb.AppendLine($"f {i0} {i1} {i2}");
                }
            }

            File.WriteAllText(filePath, sb.ToString());

            // Write MTL file
            if (texture != null && options.ExportTextures)
            {
                WriteMTL(Path.Combine(directory, mtlFileName), baseName, textureFileName);

                // Write texture
                string texturePath = Path.Combine(directory, textureFileName);
                ExportTexture(texturePath, texture.DiffuseMap, options.TextureFormat, options.JpegQuality);
            }
        }

        private static void WriteMTL(string filePath, string materialName, string textureFileName)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# Deep3DStudio MTL Export");
            sb.AppendLine();
            sb.AppendLine($"newmtl {materialName}_material");
            sb.AppendLine("Ka 1.000 1.000 1.000");
            sb.AppendLine("Kd 1.000 1.000 1.000");
            sb.AppendLine("Ks 0.000 0.000 0.000");
            sb.AppendLine("Ns 10.000");
            sb.AppendLine("d 1.000");
            sb.AppendLine("illum 2");
            sb.AppendLine($"map_Kd {textureFileName}");

            File.WriteAllText(filePath, sb.ToString());
        }

        #endregion

        #region GLTF Export

        private static void ExportGLTF(string filePath, MeshData mesh, UVData? uvData,
            BakedTextureResult? texture, MeshExportOptions options, bool binary)
        {
            string directory = Path.GetDirectoryName(filePath) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(filePath);

            // Build GLTF JSON structure
            var gltf = new Dictionary<string, object>
            {
                ["asset"] = new Dictionary<string, string>
                {
                    ["version"] = "2.0",
                    ["generator"] = "Deep3DStudio"
                }
            };

            // Prepare binary buffer
            using var bufferStream = new MemoryStream();
            using var writer = new BinaryWriter(bufferStream);

            int vertexBufferOffset = 0;
            int vertexBufferLength = 0;

            // Write vertices
            var minPos = new Vector3(float.MaxValue);
            var maxPos = new Vector3(float.MinValue);

            foreach (var v in mesh.Vertices)
            {
                var scaled = v * options.Scale;
                if (options.SwapYZ)
                {
                    writer.Write(scaled.X);
                    writer.Write(scaled.Z);
                    writer.Write(scaled.Y);
                    minPos = Vector3.ComponentMin(minPos, new Vector3(scaled.X, scaled.Z, scaled.Y));
                    maxPos = Vector3.ComponentMax(maxPos, new Vector3(scaled.X, scaled.Z, scaled.Y));
                }
                else
                {
                    writer.Write(scaled.X);
                    writer.Write(scaled.Y);
                    writer.Write(scaled.Z);
                    minPos = Vector3.ComponentMin(minPos, scaled);
                    maxPos = Vector3.ComponentMax(maxPos, scaled);
                }
            }
            vertexBufferLength = (int)bufferStream.Position;

            // Write normals
            int normalBufferOffset = (int)bufferStream.Position;
            int normalBufferLength = 0;
            var normals = options.ExportNormals ? ComputeVertexNormals(mesh) : null;

            if (normals != null)
            {
                foreach (var n in normals)
                {
                    var normal = options.FlipNormals ? -n : n;
                    if (options.SwapYZ)
                    {
                        writer.Write(normal.X);
                        writer.Write(normal.Z);
                        writer.Write(normal.Y);
                    }
                    else
                    {
                        writer.Write(normal.X);
                        writer.Write(normal.Y);
                        writer.Write(normal.Z);
                    }
                }
                normalBufferLength = (int)bufferStream.Position - normalBufferOffset;
            }

            // Write UVs
            int uvBufferOffset = (int)bufferStream.Position;
            int uvBufferLength = 0;

            if (uvData != null && options.ExportUVs)
            {
                foreach (var uv in uvData.UVs)
                {
                    writer.Write(uv.X);
                    writer.Write(1.0f - uv.Y);
                }
                uvBufferLength = (int)bufferStream.Position - uvBufferOffset;
            }

            // Write indices
            int indexBufferOffset = (int)bufferStream.Position;
            foreach (int idx in mesh.Indices)
            {
                writer.Write((uint)idx);
            }
            int indexBufferLength = (int)bufferStream.Position - indexBufferOffset;

            // Pad buffer to 4-byte alignment
            while (bufferStream.Position % 4 != 0)
                writer.Write((byte)0);

            byte[] bufferData = bufferStream.ToArray();

            // Build accessor list
            var accessors = new List<Dictionary<string, object>>();
            var bufferViews = new List<Dictionary<string, object>>();

            // Position accessor
            bufferViews.Add(new Dictionary<string, object>
            {
                ["buffer"] = 0,
                ["byteOffset"] = vertexBufferOffset,
                ["byteLength"] = vertexBufferLength,
                ["target"] = 34962 // ARRAY_BUFFER
            });

            accessors.Add(new Dictionary<string, object>
            {
                ["bufferView"] = 0,
                ["componentType"] = 5126, // FLOAT
                ["count"] = mesh.Vertices.Count,
                ["type"] = "VEC3",
                ["min"] = new float[] { minPos.X, minPos.Y, minPos.Z },
                ["max"] = new float[] { maxPos.X, maxPos.Y, maxPos.Z }
            });

            int accessorIndex = 1;
            int bufferViewIndex = 1;

            // Normal accessor
            if (normalBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = normalBufferOffset,
                    ["byteLength"] = normalBufferLength,
                    ["target"] = 34962
                });

                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIndex++,
                    ["componentType"] = 5126,
                    ["count"] = mesh.Vertices.Count,
                    ["type"] = "VEC3"
                });
                accessorIndex++;
            }

            // UV accessor
            int uvAccessorIndex = -1;
            if (uvBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = uvBufferOffset,
                    ["byteLength"] = uvBufferLength,
                    ["target"] = 34962
                });

                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIndex++,
                    ["componentType"] = 5126,
                    ["count"] = uvData!.UVs.Count,
                    ["type"] = "VEC2"
                });
                uvAccessorIndex = accessorIndex++;
            }

            // Index accessor
            bufferViews.Add(new Dictionary<string, object>
            {
                ["buffer"] = 0,
                ["byteOffset"] = indexBufferOffset,
                ["byteLength"] = indexBufferLength,
                ["target"] = 34963 // ELEMENT_ARRAY_BUFFER
            });

            int indexAccessorIndex = accessorIndex;
            accessors.Add(new Dictionary<string, object>
            {
                ["bufferView"] = bufferViewIndex,
                ["componentType"] = 5125, // UNSIGNED_INT
                ["count"] = mesh.Indices.Count,
                ["type"] = "SCALAR"
            });

            // Build primitives
            var primitiveAttributes = new Dictionary<string, int>
            {
                ["POSITION"] = 0
            };

            if (normalBufferLength > 0)
                primitiveAttributes["NORMAL"] = 1;

            if (uvAccessorIndex >= 0)
                primitiveAttributes["TEXCOORD_0"] = uvAccessorIndex;

            var primitive = new Dictionary<string, object>
            {
                ["attributes"] = primitiveAttributes,
                ["indices"] = indexAccessorIndex,
                ["mode"] = 4 // TRIANGLES
            };

            // Add material if texture exists
            var materials = new List<Dictionary<string, object>>();
            var textures = new List<Dictionary<string, object>>();
            var images = new List<Dictionary<string, object>>();

            if (texture != null && options.ExportTextures)
            {
                primitive["material"] = 0;

                string textureFileName = baseName + "_diffuse" + GetTextureExtension(options.TextureFormat);

                if (!binary || !options.EmbedTextures)
                {
                    // External texture
                    images.Add(new Dictionary<string, object>
                    {
                        ["uri"] = textureFileName
                    });

                    // Save texture file
                    string texturePath = Path.Combine(directory, textureFileName);
                    ExportTexture(texturePath, texture.DiffuseMap, options.TextureFormat, options.JpegQuality);
                }
                else
                {
                    // Embedded texture (data URI)
                    using var texStream = new MemoryStream();
                    texture.DiffuseMap.Encode(texStream, SKEncodedImageFormat.Png, 100);
                    string base64 = Convert.ToBase64String(texStream.ToArray());

                    images.Add(new Dictionary<string, object>
                    {
                        ["uri"] = $"data:image/png;base64,{base64}"
                    });
                }

                textures.Add(new Dictionary<string, object>
                {
                    ["source"] = 0
                });

                materials.Add(new Dictionary<string, object>
                {
                    ["pbrMetallicRoughness"] = new Dictionary<string, object>
                    {
                        ["baseColorTexture"] = new Dictionary<string, int>
                        {
                            ["index"] = 0
                        },
                        ["metallicFactor"] = 0.0,
                        ["roughnessFactor"] = 1.0
                    }
                });
            }

            // Assemble GLTF
            gltf["buffers"] = new List<Dictionary<string, object>>
            {
                binary
                    ? new Dictionary<string, object> { ["byteLength"] = bufferData.Length }
                    : new Dictionary<string, object>
                    {
                        ["uri"] = baseName + ".bin",
                        ["byteLength"] = bufferData.Length
                    }
            };

            gltf["bufferViews"] = bufferViews;
            gltf["accessors"] = accessors;

            gltf["meshes"] = new List<Dictionary<string, object>>
            {
                new Dictionary<string, object>
                {
                    ["name"] = baseName,
                    ["primitives"] = new List<Dictionary<string, object>> { primitive }
                }
            };

            gltf["nodes"] = new List<Dictionary<string, object>>
            {
                new Dictionary<string, object>
                {
                    ["name"] = baseName,
                    ["mesh"] = 0
                }
            };

            gltf["scenes"] = new List<Dictionary<string, object>>
            {
                new Dictionary<string, object>
                {
                    ["nodes"] = new List<int> { 0 }
                }
            };

            gltf["scene"] = 0;

            if (materials.Count > 0)
                gltf["materials"] = materials;
            if (textures.Count > 0)
                gltf["textures"] = textures;
            if (images.Count > 0)
                gltf["images"] = images;

            // Write output
            if (binary)
            {
                WriteGLB(filePath, gltf, bufferData);
            }
            else
            {
                // Write JSON
                string json = System.Text.Json.JsonSerializer.Serialize(gltf,
                    new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(filePath, json);

                // Write binary buffer
                string binPath = Path.Combine(directory, baseName + ".bin");
                File.WriteAllBytes(binPath, bufferData);
            }
        }

        private static void WriteGLB(string filePath, Dictionary<string, object> gltf, byte[] bufferData)
        {
            string json = System.Text.Json.JsonSerializer.Serialize(gltf);

            // Pad JSON to 4-byte alignment
            while (json.Length % 4 != 0)
                json += " ";

            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

            using var stream = File.Create(filePath);
            using var writer = new BinaryWriter(stream);

            // Header
            writer.Write(0x46546C67); // "glTF"
            writer.Write(2);          // Version
            writer.Write(12 + 8 + jsonBytes.Length + 8 + bufferData.Length); // Total length

            // JSON chunk
            writer.Write(jsonBytes.Length);
            writer.Write(0x4E4F534A); // "JSON"
            writer.Write(jsonBytes);

            // BIN chunk
            writer.Write(bufferData.Length);
            writer.Write(0x004E4942); // "BIN\0"
            writer.Write(bufferData);
        }

        #endregion

        #region FBX Export

        private static void ExportFBXAscii(string filePath, MeshData mesh, UVData? uvData,
            BakedTextureResult? texture, MeshExportOptions options)
        {
            var sb = new StringBuilder();
            var culture = CultureInfo.InvariantCulture;

            string baseName = Path.GetFileNameWithoutExtension(filePath);
            string directory = Path.GetDirectoryName(filePath) ?? ".";

            // FBX Header
            sb.AppendLine("; FBX 7.4 project file");
            sb.AppendLine("; Deep3DStudio Export");
            sb.AppendLine($"; Vertices: {mesh.Vertices.Count}");
            sb.AppendLine($"; Triangles: {mesh.Indices.Count / 3}");
            sb.AppendLine();

            sb.AppendLine("FBXHeaderExtension:  {");
            sb.AppendLine("\tFBXHeaderVersion: 1003");
            sb.AppendLine("\tFBXVersion: 7400");
            sb.AppendLine("}");
            sb.AppendLine();

            // Definitions
            sb.AppendLine("Definitions:  {");
            sb.AppendLine("\tVersion: 100");
            sb.AppendLine("\tCount: 3");
            sb.AppendLine("\tObjectType: \"Model\" { Count: 1 }");
            sb.AppendLine("\tObjectType: \"Geometry\" { Count: 1 }");
            sb.AppendLine("\tObjectType: \"Material\" { Count: 1 }");
            sb.AppendLine("}");
            sb.AppendLine();

            // Objects
            sb.AppendLine("Objects:  {");

            // Geometry
            sb.AppendLine($"\tGeometry: 1000, \"Geometry::{baseName}\", \"Mesh\" {{");

            // Vertices
            sb.Append("\t\tVertices: ");
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i] * options.Scale;
                if (i > 0) sb.Append(",");
                if (options.SwapYZ)
                    sb.AppendFormat(culture, "{0},{1},{2}", v.X, v.Z, v.Y);
                else
                    sb.AppendFormat(culture, "{0},{1},{2}", v.X, v.Y, v.Z);
            }
            sb.AppendLine();

            // Indices
            sb.Append("\t\tPolygonVertexIndex: ");
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                if (i > 0) sb.Append(",");
                sb.AppendFormat("{0},{1},{2}", mesh.Indices[i], mesh.Indices[i + 1], ~mesh.Indices[i + 2]);
            }
            sb.AppendLine();

            // UVs
            if (uvData != null && options.ExportUVs)
            {
                sb.AppendLine("\t\tLayerElementUV: 0 {");
                sb.AppendLine("\t\t\tVersion: 101");
                sb.AppendLine("\t\t\tName: \"UVChannel_1\"");
                sb.AppendLine("\t\t\tMappingInformationType: \"ByPolygonVertex\"");
                sb.AppendLine("\t\t\tReferenceInformationType: \"IndexToDirect\"");

                sb.Append("\t\t\tUV: ");
                for (int i = 0; i < uvData.UVs.Count; i++)
                {
                    if (i > 0) sb.Append(",");
                    sb.AppendFormat(culture, "{0},{1}", uvData.UVs[i].X, 1.0f - uvData.UVs[i].Y);
                }
                sb.AppendLine();

                sb.Append("\t\t\tUVIndex: ");
                for (int i = 0; i < mesh.Indices.Count; i++)
                {
                    if (i > 0) sb.Append(",");
                    sb.Append(mesh.Indices[i]);
                }
                sb.AppendLine();

                sb.AppendLine("\t\t}");
            }

            sb.AppendLine("\t}");

            // Model
            sb.AppendLine($"\tModel: 2000, \"Model::{baseName}\", \"Mesh\" {{");
            sb.AppendLine("\t\tVersion: 232");
            sb.AppendLine("\t\tProperties70:  {");
            sb.AppendLine("\t\t}");
            sb.AppendLine("\t}");

            // Material
            sb.AppendLine($"\tMaterial: 3000, \"Material::{baseName}_Material\", \"\" {{");
            sb.AppendLine("\t\tVersion: 102");
            sb.AppendLine("\t\tShadingModel: \"phong\"");
            sb.AppendLine("\t}");

            sb.AppendLine("}");

            // Connections
            sb.AppendLine("Connections:  {");
            sb.AppendLine("\tC: \"OO\",1000,2000");
            sb.AppendLine("\tC: \"OO\",3000,2000");
            sb.AppendLine("\tC: \"OO\",2000,0");
            sb.AppendLine("}");

            File.WriteAllText(filePath, sb.ToString());

            // Export texture separately
            if (texture != null && options.ExportTextures)
            {
                string textureFileName = baseName + "_diffuse" + GetTextureExtension(options.TextureFormat);
                string texturePath = Path.Combine(directory, textureFileName);
                ExportTexture(texturePath, texture.DiffuseMap, options.TextureFormat, options.JpegQuality);
            }
        }

        #endregion

        #region PLY Export

        private static void ExportPLY(string filePath, MeshData mesh, UVData? uvData, MeshExportOptions options)
        {
            var sb = new StringBuilder();
            var culture = CultureInfo.InvariantCulture;

            // PLY Header
            sb.AppendLine("ply");
            sb.AppendLine("format ascii 1.0");
            sb.AppendLine($"comment Deep3DStudio Export");
            sb.AppendLine($"element vertex {mesh.Vertices.Count}");
            sb.AppendLine("property float x");
            sb.AppendLine("property float y");
            sb.AppendLine("property float z");

            if (options.ExportColors && mesh.Colors.Count == mesh.Vertices.Count)
            {
                sb.AppendLine("property uchar red");
                sb.AppendLine("property uchar green");
                sb.AppendLine("property uchar blue");
            }

            if (uvData != null && options.ExportUVs)
            {
                sb.AppendLine("property float s");
                sb.AppendLine("property float t");
            }

            sb.AppendLine($"element face {mesh.Indices.Count / 3}");
            sb.AppendLine("property list uchar int vertex_indices");
            sb.AppendLine("end_header");

            // Vertices
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i] * options.Scale;
                if (options.SwapYZ)
                    sb.AppendFormat(culture, "{0} {1} {2}", v.X, v.Z, v.Y);
                else
                    sb.AppendFormat(culture, "{0} {1} {2}", v.X, v.Y, v.Z);

                if (options.ExportColors && i < mesh.Colors.Count)
                {
                    var c = mesh.Colors[i];
                    sb.AppendFormat(" {0} {1} {2}",
                        (byte)(Math.Clamp(c.X, 0, 1) * 255),
                        (byte)(Math.Clamp(c.Y, 0, 1) * 255),
                        (byte)(Math.Clamp(c.Z, 0, 1) * 255));
                }

                if (uvData != null && options.ExportUVs && i < uvData.UVs.Count)
                {
                    sb.AppendFormat(culture, " {0} {1}", uvData.UVs[i].X, uvData.UVs[i].Y);
                }

                sb.AppendLine();
            }

            // Faces
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                sb.AppendLine($"3 {mesh.Indices[i]} {mesh.Indices[i + 1]} {mesh.Indices[i + 2]}");
            }

            File.WriteAllText(filePath, sb.ToString());
        }

        #endregion

        #region Helper Methods

        private static Vector3[] ComputeVertexNormals(MeshData mesh)
        {
            var normals = new Vector3[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                var faceNormal = Vector3.Cross(v1 - v0, v2 - v0);

                normals[i0] += faceNormal;
                normals[i1] += faceNormal;
                normals[i2] += faceNormal;
            }

            for (int i = 0; i < normals.Length; i++)
            {
                if (normals[i].LengthSquared > 0.0001f)
                    normals[i] = normals[i].Normalized();
            }

            return normals;
        }

        private static string GetTextureExtension(TextureFormat format)
        {
            return format switch
            {
                TextureFormat.PNG => ".png",
                TextureFormat.JPEG => ".jpg",
                TextureFormat.TGA => ".tga",
                TextureFormat.BMP => ".bmp",
                _ => ".png"
            };
        }

        #endregion
    }
}
