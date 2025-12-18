using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OpenTK.Mathematics;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;
using Deep3DStudio.Texturing;

namespace Deep3DStudio.IO
{
    /// <summary>
    /// Export options for rigged meshes
    /// </summary>
    public class RiggedMeshExportOptions : MeshExportOptions
    {
        public bool ExportSkeleton { get; set; } = true;
        public bool ExportSkinningWeights { get; set; } = true;
        public int MaxBonesPerVertex { get; set; } = 4;
    }

    /// <summary>
    /// Exports rigged meshes with skeleton data to various formats
    /// </summary>
    public static class RiggedMeshExporter
    {
        /// <summary>
        /// Export a rigged mesh with skeleton
        /// </summary>
        public static void Export(string filePath, MeshData mesh, SkeletonData skeleton,
            UVData? uvData, BakedTextureResult? texture, RiggedMeshExportOptions options)
        {
            switch (options.Format)
            {
                case TexturedMeshFormat.FBX_ASCII:
                    ExportFBXWithSkeleton(filePath, mesh, skeleton, uvData, texture, options);
                    break;
                case TexturedMeshFormat.GLTF:
                    ExportGLTFWithSkeleton(filePath, mesh, skeleton, uvData, texture, options, false);
                    break;
                case TexturedMeshFormat.GLB:
                    ExportGLTFWithSkeleton(filePath, mesh, skeleton, uvData, texture, options, true);
                    break;
                default:
                    // Fall back to non-rigged export for unsupported formats
                    TexturedMeshExporter.Export(filePath, mesh, uvData, texture, options);
                    Console.WriteLine($"Warning: Format {options.Format} does not support skeleton export. Exporting mesh only.");
                    break;
            }
        }

        #region FBX Export with Skeleton

        private static void ExportFBXWithSkeleton(string filePath, MeshData mesh, SkeletonData skeleton,
            UVData? uvData, BakedTextureResult? texture, RiggedMeshExportOptions options)
        {
            var sb = new StringBuilder();
            var culture = CultureInfo.InvariantCulture;

            string baseName = Path.GetFileNameWithoutExtension(filePath);
            string directory = Path.GetDirectoryName(filePath) ?? ".";

            var joints = skeleton.GetJointsHierarchical().ToList();
            int jointCount = joints.Count;

            // FBX Header
            sb.AppendLine("; FBX 7.4 project file");
            sb.AppendLine("; Deep3DStudio Rigged Mesh Export");
            sb.AppendLine($"; Vertices: {mesh.Vertices.Count}");
            sb.AppendLine($"; Triangles: {mesh.Indices.Count / 3}");
            sb.AppendLine($"; Joints: {jointCount}");
            sb.AppendLine();

            sb.AppendLine("FBXHeaderExtension:  {");
            sb.AppendLine("\tFBXHeaderVersion: 1003");
            sb.AppendLine("\tFBXVersion: 7400");
            sb.AppendLine("\tCreationTimeStamp:  {");
            sb.AppendLine("\t\tVersion: 1000");
            sb.AppendLine($"\t\tYear: {DateTime.Now.Year}");
            sb.AppendLine($"\t\tMonth: {DateTime.Now.Month}");
            sb.AppendLine($"\t\tDay: {DateTime.Now.Day}");
            sb.AppendLine("\t}");
            sb.AppendLine("\tCreator: \"Deep3DStudio\"");
            sb.AppendLine("}");
            sb.AppendLine();

            // Global Settings
            sb.AppendLine("GlobalSettings:  {");
            sb.AppendLine("\tVersion: 1000");
            sb.AppendLine("\tProperties70:  {");
            sb.AppendLine("\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\", 1");
            sb.AppendLine("\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\", 1");
            sb.AppendLine("\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\", 2");
            sb.AppendLine("\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\", 1");
            sb.AppendLine("\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\", 0");
            sb.AppendLine("\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\", 1");
            sb.AppendLine("\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\", 1");
            sb.AppendLine("\t}");
            sb.AppendLine("}");
            sb.AppendLine();

            // Definitions
            int deformerCount = options.ExportSkinningWeights && skeleton.SkinningWeights != null ? jointCount + 1 : 0;
            sb.AppendLine("Definitions:  {");
            sb.AppendLine("\tVersion: 100");
            sb.AppendLine($"\tCount: {3 + jointCount + deformerCount}");
            sb.AppendLine($"\tObjectType: \"Model\" {{ Count: {1 + jointCount} }}");
            sb.AppendLine("\tObjectType: \"Geometry\" { Count: 1 }");
            sb.AppendLine("\tObjectType: \"Material\" { Count: 1 }");
            if (deformerCount > 0)
            {
                sb.AppendLine($"\tObjectType: \"Deformer\" {{ Count: {deformerCount} }}");
            }
            sb.AppendLine("}");
            sb.AppendLine();

            // Objects
            sb.AppendLine("Objects:  {");

            // Build joint index mapping
            var jointToIndex = new Dictionary<Joint, int>();
            for (int i = 0; i < joints.Count; i++)
            {
                jointToIndex[joints[i]] = i;
            }

            // Skeleton joints (as Null/LimbNode models)
            long jointBaseId = 10000;
            for (int i = 0; i < joints.Count; i++)
            {
                var joint = joints[i];
                long jointId = jointBaseId + i;

                sb.AppendLine($"\tModel: {jointId}, \"Model::{joint.Name}\", \"LimbNode\" {{");
                sb.AppendLine("\t\tVersion: 232");
                sb.AppendLine("\t\tProperties70:  {");

                var pos = joint.Position * options.Scale;
                if (options.SwapYZ)
                {
                    sb.AppendLine(string.Format(culture, "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\", {0}, {1}, {2}",
                        pos.X, pos.Z, pos.Y));
                }
                else
                {
                    sb.AppendLine(string.Format(culture, "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\", {0}, {1}, {2}",
                        pos.X, pos.Y, pos.Z));
                }

                sb.AppendLine("\t\t}");
                sb.AppendLine("\t}");
            }

            // Geometry
            sb.AppendLine($"\tGeometry: 1000, \"Geometry::{baseName}\", \"Mesh\" {{");

            // Vertices
            sb.Append("\t\tVertices: *").Append(mesh.Vertices.Count * 3).Append(" {\n\t\t\ta: ");
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i] * options.Scale;
                if (i > 0) sb.Append(",");
                if (options.SwapYZ)
                    sb.AppendFormat(culture, "{0},{1},{2}", v.X, v.Z, v.Y);
                else
                    sb.AppendFormat(culture, "{0},{1},{2}", v.X, v.Y, v.Z);
            }
            sb.AppendLine("\n\t\t}");

            // Indices
            sb.Append("\t\tPolygonVertexIndex: *").Append(mesh.Indices.Count).Append(" {\n\t\t\ta: ");
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                if (i > 0) sb.Append(",");
                sb.AppendFormat("{0},{1},{2}", mesh.Indices[i], mesh.Indices[i + 1], ~mesh.Indices[i + 2]);
            }
            sb.AppendLine("\n\t\t}");

            // Normals
            var normals = ComputeVertexNormals(mesh);
            sb.AppendLine("\t\tLayerElementNormal: 0 {");
            sb.AppendLine("\t\t\tVersion: 101");
            sb.AppendLine("\t\t\tName: \"\"");
            sb.AppendLine("\t\t\tMappingInformationType: \"ByVertice\"");
            sb.AppendLine("\t\t\tReferenceInformationType: \"Direct\"");
            sb.Append("\t\t\tNormals: *").Append(normals.Length * 3).Append(" {\n\t\t\t\ta: ");
            for (int i = 0; i < normals.Length; i++)
            {
                var n = options.FlipNormals ? -normals[i] : normals[i];
                if (i > 0) sb.Append(",");
                if (options.SwapYZ)
                    sb.AppendFormat(culture, "{0},{1},{2}", n.X, n.Z, n.Y);
                else
                    sb.AppendFormat(culture, "{0},{1},{2}", n.X, n.Y, n.Z);
            }
            sb.AppendLine("\n\t\t\t}");
            sb.AppendLine("\t\t}");

            // UVs
            if (uvData != null && options.ExportUVs)
            {
                sb.AppendLine("\t\tLayerElementUV: 0 {");
                sb.AppendLine("\t\t\tVersion: 101");
                sb.AppendLine("\t\t\tName: \"UVChannel_1\"");
                sb.AppendLine("\t\t\tMappingInformationType: \"ByVertice\"");
                sb.AppendLine("\t\t\tReferenceInformationType: \"Direct\"");
                sb.Append("\t\t\tUV: *").Append(uvData.UVs.Count * 2).Append(" {\n\t\t\t\ta: ");
                for (int i = 0; i < uvData.UVs.Count; i++)
                {
                    if (i > 0) sb.Append(",");
                    sb.AppendFormat(culture, "{0},{1}", uvData.UVs[i].X, 1.0f - uvData.UVs[i].Y);
                }
                sb.AppendLine("\n\t\t\t}");
                sb.AppendLine("\t\t}");
            }

            // Layer
            sb.AppendLine("\t\tLayer: 0 {");
            sb.AppendLine("\t\t\tVersion: 100");
            sb.AppendLine("\t\t\tLayerElement:  {");
            sb.AppendLine("\t\t\t\tType: \"LayerElementNormal\"");
            sb.AppendLine("\t\t\t\tTypedIndex: 0");
            sb.AppendLine("\t\t\t}");
            if (uvData != null && options.ExportUVs)
            {
                sb.AppendLine("\t\t\tLayerElement:  {");
                sb.AppendLine("\t\t\t\tType: \"LayerElementUV\"");
                sb.AppendLine("\t\t\t\tTypedIndex: 0");
                sb.AppendLine("\t\t\t}");
            }
            sb.AppendLine("\t\t}");
            sb.AppendLine("\t}");

            // Mesh Model
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

            // Skin deformer (if skinning weights available)
            long skinDeformerId = 20000;
            if (options.ExportSkinningWeights && skeleton.SkinningWeights != null)
            {
                sb.AppendLine($"\tDeformer: {skinDeformerId}, \"Deformer::\", \"Skin\" {{");
                sb.AppendLine("\t\tVersion: 101");
                sb.AppendLine("\t\tLink_DeformAcuracy: 50");
                sb.AppendLine("\t}");

                // Sub-deformers (clusters) for each joint
                long clusterBaseId = 30000;
                for (int j = 0; j < jointCount; j++)
                {
                    var joint = joints[j];
                    long clusterId = clusterBaseId + j;

                    // Collect vertices influenced by this joint
                    var influences = new List<(int vertex, float weight)>();
                    for (int v = 0; v < mesh.Vertices.Count; v++)
                    {
                        float weight = skeleton.SkinningWeights[v, j];
                        if (weight > 0.001f)
                        {
                            influences.Add((v, weight));
                        }
                    }

                    if (influences.Count > 0)
                    {
                        sb.AppendLine($"\tDeformer: {clusterId}, \"SubDeformer::{joint.Name}\", \"Cluster\" {{");
                        sb.AppendLine("\t\tVersion: 100");

                        // Indexes
                        sb.Append("\t\tIndexes: *").Append(influences.Count).Append(" {\n\t\t\ta: ");
                        for (int i = 0; i < influences.Count; i++)
                        {
                            if (i > 0) sb.Append(",");
                            sb.Append(influences[i].vertex);
                        }
                        sb.AppendLine("\n\t\t}");

                        // Weights
                        sb.Append("\t\tWeights: *").Append(influences.Count).Append(" {\n\t\t\ta: ");
                        for (int i = 0; i < influences.Count; i++)
                        {
                            if (i > 0) sb.Append(",");
                            sb.AppendFormat(culture, "{0}", influences[i].weight);
                        }
                        sb.AppendLine("\n\t\t}");

                        // Transform matrix (identity for bind pose)
                        sb.AppendLine("\t\tTransform: *16 {");
                        sb.AppendLine("\t\t\ta: 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1");
                        sb.AppendLine("\t\t}");

                        sb.AppendLine("\t}");
                    }
                }
            }

            sb.AppendLine("}");

            // Connections
            sb.AppendLine();
            sb.AppendLine("Connections:  {");

            // Geometry to mesh
            sb.AppendLine("\tC: \"OO\",1000,2000");

            // Material to mesh
            sb.AppendLine("\tC: \"OO\",3000,2000");

            // Mesh to root
            sb.AppendLine("\tC: \"OO\",2000,0");

            // Joint hierarchy
            for (int i = 0; i < joints.Count; i++)
            {
                var joint = joints[i];
                long jointId = jointBaseId + i;

                if (joint.Parent != null && jointToIndex.ContainsKey(joint.Parent))
                {
                    long parentId = jointBaseId + jointToIndex[joint.Parent];
                    sb.AppendLine($"\tC: \"OO\",{jointId},{parentId}");
                }
                else
                {
                    // Root joint to scene
                    sb.AppendLine($"\tC: \"OO\",{jointId},0");
                }
            }

            // Skin deformer connections
            if (options.ExportSkinningWeights && skeleton.SkinningWeights != null)
            {
                sb.AppendLine($"\tC: \"OO\",{skinDeformerId},1000");

                long clusterBaseId = 30000;
                for (int j = 0; j < jointCount; j++)
                {
                    long clusterId = clusterBaseId + j;
                    long jointId = jointBaseId + j;
                    sb.AppendLine($"\tC: \"OO\",{clusterId},{skinDeformerId}");
                    sb.AppendLine($"\tC: \"OO\",{jointId},{clusterId}");
                }
            }

            sb.AppendLine("}");

            File.WriteAllText(filePath, sb.ToString());

            // Export texture separately
            if (texture != null && options.ExportTextures)
            {
                string textureFileName = baseName + "_diffuse" + GetTextureExtension(options.TextureFormat);
                string texturePath = Path.Combine(directory, textureFileName);
                TexturedMeshExporter.ExportTexture(texturePath, texture.DiffuseMap, options.TextureFormat, options.JpegQuality);
            }

            Console.WriteLine($"Exported rigged mesh to {filePath} with {jointCount} joints");
        }

        #endregion

        #region GLTF Export with Skeleton

        private static void ExportGLTFWithSkeleton(string filePath, MeshData mesh, SkeletonData skeleton,
            UVData? uvData, BakedTextureResult? texture, RiggedMeshExportOptions options, bool binary)
        {
            string directory = Path.GetDirectoryName(filePath) ?? ".";
            string baseName = Path.GetFileNameWithoutExtension(filePath);

            var joints = skeleton.GetJointsHierarchical().ToList();
            int jointCount = joints.Count;

            // Build joint index mapping
            var jointToIndex = new Dictionary<Joint, int>();
            for (int i = 0; i < joints.Count; i++)
            {
                jointToIndex[joints[i]] = i;
            }

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

            // Write vertices
            int vertexBufferOffset = 0;
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
            int vertexBufferLength = (int)bufferStream.Position;

            // Write normals
            int normalBufferOffset = (int)bufferStream.Position;
            var normals = options.ExportNormals ? ComputeVertexNormals(mesh) : null;
            int normalBufferLength = 0;

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

            // Write joint indices (JOINTS_0)
            int jointsBufferOffset = (int)bufferStream.Position;
            int jointsBufferLength = 0;

            if (options.ExportSkinningWeights && skeleton.SkinningWeights != null)
            {
                for (int v = 0; v < mesh.Vertices.Count; v++)
                {
                    // Get top 4 joints for this vertex
                    var influences = new List<(int joint, float weight)>();
                    for (int j = 0; j < jointCount; j++)
                    {
                        float w = skeleton.SkinningWeights[v, j];
                        if (w > 0.001f)
                        {
                            influences.Add((j, w));
                        }
                    }

                    influences.Sort((a, b) => b.weight.CompareTo(a.weight));

                    // Write 4 joint indices (as unsigned shorts)
                    for (int i = 0; i < 4; i++)
                    {
                        ushort jointIdx = i < influences.Count ? (ushort)influences[i].joint : (ushort)0;
                        writer.Write(jointIdx);
                    }
                }
                jointsBufferLength = (int)bufferStream.Position - jointsBufferOffset;
            }

            // Write weights (WEIGHTS_0)
            int weightsBufferOffset = (int)bufferStream.Position;
            int weightsBufferLength = 0;

            if (options.ExportSkinningWeights && skeleton.SkinningWeights != null)
            {
                for (int v = 0; v < mesh.Vertices.Count; v++)
                {
                    // Get top 4 weights for this vertex
                    var influences = new List<(int joint, float weight)>();
                    for (int j = 0; j < jointCount; j++)
                    {
                        float w = skeleton.SkinningWeights[v, j];
                        if (w > 0.001f)
                        {
                            influences.Add((j, w));
                        }
                    }

                    influences.Sort((a, b) => b.weight.CompareTo(a.weight));

                    // Normalize weights
                    float totalWeight = 0;
                    for (int i = 0; i < Math.Min(4, influences.Count); i++)
                    {
                        totalWeight += influences[i].weight;
                    }

                    // Write 4 weights
                    for (int i = 0; i < 4; i++)
                    {
                        float w = i < influences.Count ? influences[i].weight / totalWeight : 0;
                        writer.Write(w);
                    }
                }
                weightsBufferLength = (int)bufferStream.Position - weightsBufferOffset;
            }

            // Write inverse bind matrices
            int ibmBufferOffset = (int)bufferStream.Position;
            int ibmBufferLength = 0;

            if (options.ExportSkeleton)
            {
                foreach (var joint in joints)
                {
                    // Write identity matrix for bind pose (simplified)
                    var ibm = Matrix4.Identity;
                    var pos = joint.GetWorldPosition() * options.Scale;
                    ibm = Matrix4.CreateTranslation(-pos);

                    // Write column-major
                    writer.Write(ibm.M11); writer.Write(ibm.M21); writer.Write(ibm.M31); writer.Write(ibm.M41);
                    writer.Write(ibm.M12); writer.Write(ibm.M22); writer.Write(ibm.M32); writer.Write(ibm.M42);
                    writer.Write(ibm.M13); writer.Write(ibm.M23); writer.Write(ibm.M33); writer.Write(ibm.M43);
                    writer.Write(ibm.M14); writer.Write(ibm.M24); writer.Write(ibm.M34); writer.Write(ibm.M44);
                }
                ibmBufferLength = (int)bufferStream.Position - ibmBufferOffset;
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

            // Build accessor and buffer view lists
            var accessors = new List<Dictionary<string, object>>();
            var bufferViews = new List<Dictionary<string, object>>();
            int accessorIdx = 0;
            int bufferViewIdx = 0;

            // Position accessor
            bufferViews.Add(new Dictionary<string, object>
            {
                ["buffer"] = 0,
                ["byteOffset"] = vertexBufferOffset,
                ["byteLength"] = vertexBufferLength,
                ["target"] = 34962
            });
            int positionAccessor = accessorIdx++;
            accessors.Add(new Dictionary<string, object>
            {
                ["bufferView"] = bufferViewIdx++,
                ["componentType"] = 5126,
                ["count"] = mesh.Vertices.Count,
                ["type"] = "VEC3",
                ["min"] = new float[] { minPos.X, minPos.Y, minPos.Z },
                ["max"] = new float[] { maxPos.X, maxPos.Y, maxPos.Z }
            });

            // Normal accessor
            int normalAccessor = -1;
            if (normalBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = normalBufferOffset,
                    ["byteLength"] = normalBufferLength,
                    ["target"] = 34962
                });
                normalAccessor = accessorIdx++;
                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIdx++,
                    ["componentType"] = 5126,
                    ["count"] = mesh.Vertices.Count,
                    ["type"] = "VEC3"
                });
            }

            // Joints accessor
            int jointsAccessor = -1;
            if (jointsBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = jointsBufferOffset,
                    ["byteLength"] = jointsBufferLength,
                    ["target"] = 34962
                });
                jointsAccessor = accessorIdx++;
                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIdx++,
                    ["componentType"] = 5123, // UNSIGNED_SHORT
                    ["count"] = mesh.Vertices.Count,
                    ["type"] = "VEC4"
                });
            }

            // Weights accessor
            int weightsAccessor = -1;
            if (weightsBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = weightsBufferOffset,
                    ["byteLength"] = weightsBufferLength,
                    ["target"] = 34962
                });
                weightsAccessor = accessorIdx++;
                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIdx++,
                    ["componentType"] = 5126,
                    ["count"] = mesh.Vertices.Count,
                    ["type"] = "VEC4"
                });
            }

            // Inverse bind matrices accessor
            int ibmAccessor = -1;
            if (ibmBufferLength > 0)
            {
                bufferViews.Add(new Dictionary<string, object>
                {
                    ["buffer"] = 0,
                    ["byteOffset"] = ibmBufferOffset,
                    ["byteLength"] = ibmBufferLength
                });
                ibmAccessor = accessorIdx++;
                accessors.Add(new Dictionary<string, object>
                {
                    ["bufferView"] = bufferViewIdx++,
                    ["componentType"] = 5126,
                    ["count"] = jointCount,
                    ["type"] = "MAT4"
                });
            }

            // Index accessor
            bufferViews.Add(new Dictionary<string, object>
            {
                ["buffer"] = 0,
                ["byteOffset"] = indexBufferOffset,
                ["byteLength"] = indexBufferLength,
                ["target"] = 34963
            });
            int indexAccessor = accessorIdx++;
            accessors.Add(new Dictionary<string, object>
            {
                ["bufferView"] = bufferViewIdx,
                ["componentType"] = 5125,
                ["count"] = mesh.Indices.Count,
                ["type"] = "SCALAR"
            });

            // Build primitive attributes
            var primitiveAttributes = new Dictionary<string, int>
            {
                ["POSITION"] = positionAccessor
            };

            if (normalAccessor >= 0)
                primitiveAttributes["NORMAL"] = normalAccessor;
            if (jointsAccessor >= 0)
                primitiveAttributes["JOINTS_0"] = jointsAccessor;
            if (weightsAccessor >= 0)
                primitiveAttributes["WEIGHTS_0"] = weightsAccessor;

            var primitive = new Dictionary<string, object>
            {
                ["attributes"] = primitiveAttributes,
                ["indices"] = indexAccessor,
                ["mode"] = 4
            };

            // Build nodes for skeleton
            var nodes = new List<Dictionary<string, object>>();
            var jointNodeIndices = new List<int>();

            // Add joint nodes
            for (int i = 0; i < joints.Count; i++)
            {
                var joint = joints[i];
                var nodeDict = new Dictionary<string, object>
                {
                    ["name"] = joint.Name
                };

                var pos = joint.Position * options.Scale;
                if (options.SwapYZ)
                {
                    nodeDict["translation"] = new float[] { pos.X, pos.Z, pos.Y };
                }
                else
                {
                    nodeDict["translation"] = new float[] { pos.X, pos.Y, pos.Z };
                }

                // Add children
                var childIndices = new List<int>();
                foreach (var child in joint.Children)
                {
                    if (jointToIndex.ContainsKey(child))
                    {
                        childIndices.Add(jointToIndex[child]);
                    }
                }
                if (childIndices.Count > 0)
                {
                    nodeDict["children"] = childIndices;
                }

                jointNodeIndices.Add(nodes.Count);
                nodes.Add(nodeDict);
            }

            // Add mesh node
            int meshNodeIndex = nodes.Count;
            var meshNode = new Dictionary<string, object>
            {
                ["name"] = baseName,
                ["mesh"] = 0
            };

            if (options.ExportSkeleton && jointCount > 0)
            {
                meshNode["skin"] = 0;
            }

            nodes.Add(meshNode);

            // Build skins
            var skins = new List<Dictionary<string, object>>();
            if (options.ExportSkeleton && jointCount > 0)
            {
                var skin = new Dictionary<string, object>
                {
                    ["joints"] = jointNodeIndices
                };

                if (ibmAccessor >= 0)
                {
                    skin["inverseBindMatrices"] = ibmAccessor;
                }

                // Find root joint (skeleton root)
                var rootJoint = skeleton.RootJoint;
                if (rootJoint != null && jointToIndex.ContainsKey(rootJoint))
                {
                    skin["skeleton"] = jointNodeIndices[jointToIndex[rootJoint]];
                }

                skins.Add(skin);
            }

            // Build scene nodes (root joints + mesh)
            var sceneNodes = new List<int>();
            foreach (var joint in joints.Where(j => j.Parent == null))
            {
                sceneNodes.Add(jointNodeIndices[jointToIndex[joint]]);
            }
            sceneNodes.Add(meshNodeIndex);

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
            gltf["nodes"] = nodes;

            gltf["meshes"] = new List<Dictionary<string, object>>
            {
                new Dictionary<string, object>
                {
                    ["name"] = baseName,
                    ["primitives"] = new List<Dictionary<string, object>> { primitive }
                }
            };

            if (skins.Count > 0)
            {
                gltf["skins"] = skins;
            }

            gltf["scenes"] = new List<Dictionary<string, object>>
            {
                new Dictionary<string, object>
                {
                    ["nodes"] = sceneNodes
                }
            };

            gltf["scene"] = 0;

            // Write output
            if (binary)
            {
                WriteGLB(filePath, gltf, bufferData);
            }
            else
            {
                string json = System.Text.Json.JsonSerializer.Serialize(gltf,
                    new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(filePath, json);

                string binPath = Path.Combine(directory, baseName + ".bin");
                File.WriteAllBytes(binPath, bufferData);
            }

            Console.WriteLine($"Exported rigged mesh to {filePath} with {jointCount} joints");
        }

        private static void WriteGLB(string filePath, Dictionary<string, object> gltf, byte[] bufferData)
        {
            string json = System.Text.Json.JsonSerializer.Serialize(gltf);

            while (json.Length % 4 != 0)
                json += " ";

            byte[] jsonBytes = Encoding.UTF8.GetBytes(json);

            using var stream = File.Create(filePath);
            using var writer = new BinaryWriter(stream);

            writer.Write(0x46546C67);
            writer.Write(2);
            writer.Write(12 + 8 + jsonBytes.Length + 8 + bufferData.Length);

            writer.Write(jsonBytes.Length);
            writer.Write(0x4E4F534A);
            writer.Write(jsonBytes);

            writer.Write(bufferData.Length);
            writer.Write(0x004E4942);
            writer.Write(bufferData);
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
