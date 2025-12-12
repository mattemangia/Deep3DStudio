using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.IO
{
    public static class MeshImporter
    {
        public static MeshData Load(string filepath)
        {
            string ext = Path.GetExtension(filepath).ToLower();
            switch (ext)
            {
                case ".obj": return LoadObj(filepath);
                case ".stl": return LoadStl(filepath);
                case ".ply": return LoadPly(filepath);
                default: throw new NotSupportedException($"File format {ext} not supported for mesh import.");
            }
        }

        public static MeshData LoadObj(string filepath)
        {
            var mesh = new MeshData();
            var rawVertices = new List<Vector3>();
            var rawUVs = new List<Vector2>();

            // Unique vertex tracking: (vIdx, vtIdx) -> newIndex
            var uniqueVertices = new Dictionary<(int, int), int>();

            foreach (var line in File.ReadLines(filepath))
            {
                var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length == 0) continue;

                if (parts[0] == "v")
                {
                    if (parts.Length >= 4)
                    {
                        float x = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float y = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        float z = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        rawVertices.Add(new Vector3(x, y, z));
                    }
                }
                else if (parts[0] == "vt")
                {
                    if (parts.Length >= 3)
                    {
                        float u = float.Parse(parts[1], CultureInfo.InvariantCulture);
                        float v = float.Parse(parts[2], CultureInfo.InvariantCulture);
                        rawUVs.Add(new Vector2(u, v));
                    }
                }
                else if (parts[0] == "f")
                {
                    if (parts.Length >= 4)
                    {
                        // Triangulate fan
                        for (int i = 2; i < parts.Length - 1; i++)
                        {
                            AddObjVertex(parts[1], rawVertices, rawUVs, mesh, uniqueVertices);
                            AddObjVertex(parts[i], rawVertices, rawUVs, mesh, uniqueVertices);
                            AddObjVertex(parts[i + 1], rawVertices, rawUVs, mesh, uniqueVertices);
                        }
                    }
                }
            }

            // If no indices were generated (no faces), it might be a point cloud.
            // In that case, add all raw vertices.
            if (mesh.Indices.Count == 0 && rawVertices.Count > 0)
            {
                mesh.Vertices = rawVertices;
                for (int i = 0; i < rawVertices.Count; i++)
                {
                    mesh.Colors.Add(new Vector3(0.8f));
                    mesh.UVs.Add(Vector2.Zero);
                }
            }

            // Try to load texture material if mtl file exists
            // This is a simplified check, ideally we parse mtllib
            string mtlPath = Path.ChangeExtension(filepath, ".mtl");
            if (File.Exists(mtlPath))
            {
                // Very basic parser to find map_Kd
                foreach(var line in File.ReadLines(mtlPath))
                {
                     var trimmedLine = line.Trim();
                     if (trimmedLine.StartsWith("map_Kd"))
                     {
                         var parts = trimmedLine.Split(new[]{' '}, 2, StringSplitOptions.RemoveEmptyEntries);
                         if (parts.Length >= 2)
                         {
                             string texPath = parts[1].Trim();
                             if (!Path.IsPathRooted(texPath))
                                 texPath = Path.Combine(Path.GetDirectoryName(filepath) ?? "", texPath);

                             if (File.Exists(texPath))
                             {
                                 try {
                                     mesh.Texture = ImageDecoder.DecodeBitmap(texPath);
                                 } catch (Exception ex) {
                                     Console.WriteLine("Failed to load texture: " + ex.Message);
                                 }
                             }
                             break; // Only load first texture found
                         }
                     }
                }
            }
            // Also check for image with same name
            else
            {
                 string[] exts = { ".png", ".jpg", ".jpeg" };
                 string baseName = Path.Combine(Path.GetDirectoryName(filepath) ?? "", Path.GetFileNameWithoutExtension(filepath));
                 foreach(var e in exts)
                 {
                     if (File.Exists(baseName + e))
                     {
                         try {
                             mesh.Texture = ImageDecoder.DecodeBitmap(baseName + e);
                             break;
                         } catch {}
                     }
                 }
            }

            return mesh;
        }

        private static void AddObjVertex(string facePart, List<Vector3> rawV, List<Vector2> rawVT, MeshData mesh, Dictionary<(int, int), int> uniqueCache)
        {
            var parts = facePart.Split('/');

            // 1. Vertex Index
            int vIdx = int.Parse(parts[0]);
            if (vIdx < 0) vIdx += rawV.Count;
            else vIdx -= 1;

            // 2. UV Index
            int vtIdx = -1;
            if (parts.Length > 1 && !string.IsNullOrEmpty(parts[1]))
            {
                vtIdx = int.Parse(parts[1]);
                if (vtIdx < 0) vtIdx += rawVT.Count;
                else vtIdx -= 1;
            }

            var key = (vIdx, vtIdx);

            if (uniqueCache.TryGetValue(key, out int existingIdx))
            {
                mesh.Indices.Add(existingIdx);
            }
            else
            {
                int newIdx = mesh.Vertices.Count;
                mesh.Vertices.Add(rawV[vIdx]);
                mesh.Colors.Add(new Vector3(0.8f)); // Default gray

                if (vtIdx >= 0 && vtIdx < rawVT.Count)
                    mesh.UVs.Add(rawVT[vtIdx]);
                else
                    mesh.UVs.Add(Vector2.Zero);

                mesh.Indices.Add(newIdx);
                uniqueCache[key] = newIdx;
            }
        }

        public static MeshData LoadStl(string filepath)
        {
            using (var fs = File.OpenRead(filepath))
            {
                var buffer = new byte[5];
                fs.Read(buffer, 0, 5);
                string start = System.Text.Encoding.ASCII.GetString(buffer);
                fs.Position = 0;

                if (start.ToLower() == "solid")
                {
                    return LoadStlAscii(filepath);
                }
                else
                {
                    return LoadStlBinary(fs);
                }
            }
        }

        private static MeshData LoadStlAscii(string filepath)
        {
             var mesh = new MeshData();
             var vertices = new List<Vector3>();
             var indices = new List<int>();
             var vertexMap = new Dictionary<Vector3, int>();

             foreach (var line in File.ReadLines(filepath))
             {
                 var trimmed = line.Trim();
                 if (trimmed.StartsWith("vertex"))
                 {
                     var parts = trimmed.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                     float x = float.Parse(parts[1], CultureInfo.InvariantCulture);
                     float y = float.Parse(parts[2], CultureInfo.InvariantCulture);
                     float z = float.Parse(parts[3], CultureInfo.InvariantCulture);
                     var v = new Vector3(x, y, z);

                     if (!vertexMap.TryGetValue(v, out int idx))
                     {
                         idx = vertices.Count;
                         vertexMap[v] = idx;
                         vertices.Add(v);
                         mesh.Colors.Add(new Vector3(0.8f));
                     }
                     indices.Add(idx);
                 }
             }

             mesh.Vertices = vertices;
             mesh.Indices = indices;
             return mesh;
        }

        private static MeshData LoadStlBinary(Stream stream)
        {
             var mesh = new MeshData();
             var vertexMap = new Dictionary<Vector3, int>();

             using (var br = new BinaryReader(stream))
             {
                 br.ReadBytes(80); // Header
                 uint triangleCount = br.ReadUInt32();

                 for (int i = 0; i < triangleCount; i++)
                 {
                     // Normal
                     br.ReadSingle(); br.ReadSingle(); br.ReadSingle();

                     // Vertices
                     for(int v=0; v<3; v++)
                     {
                         float x = br.ReadSingle();
                         float y = br.ReadSingle();
                         float z = br.ReadSingle();
                         var vec = new Vector3(x, y, z);

                         if (!vertexMap.TryGetValue(vec, out int idx))
                         {
                             idx = mesh.Vertices.Count;
                             vertexMap[vec] = idx;
                             mesh.Vertices.Add(vec);
                             mesh.Colors.Add(new Vector3(0.8f));
                         }
                         mesh.Indices.Add(idx);
                     }

                     br.ReadUInt16(); // Attribute byte count
                 }
             }
             return mesh;
        }

        public static MeshData LoadPly(string filepath)
        {
            var plyData = PlyParser.Parse(filepath);
            var mesh = new MeshData
            {
                Vertices = plyData.Vertices,
                Colors = plyData.Colors,
                Indices = plyData.Indices,
                UVs = plyData.UVs
            };

            // Try load texture for PLY
             string[] exts = { ".png", ".jpg", ".jpeg" };
             string baseName = Path.Combine(Path.GetDirectoryName(filepath) ?? "", Path.GetFileNameWithoutExtension(filepath));
             foreach(var e in exts)
             {
                 if (File.Exists(baseName + e))
                 {
                     try {
                         mesh.Texture = ImageDecoder.DecodeBitmap(baseName + e);
                         break;
                     } catch {}
                 }
             }

             return mesh;
        }
    }
}
