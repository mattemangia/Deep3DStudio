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
            var vertices = new List<Vector3>();
            var indices = new List<int>();

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
                        vertices.Add(new Vector3(x, y, z));
                        mesh.Colors.Add(new Vector3(0.8f)); // Default gray
                    }
                }
                else if (parts[0] == "f")
                {
                    if (parts.Length >= 4)
                    {
                        // Triangulate fan
                        for (int i = 2; i < parts.Length - 1; i++)
                        {
                            indices.Add(ParseObjIndex(parts[1], vertices.Count));
                            indices.Add(ParseObjIndex(parts[i], vertices.Count));
                            indices.Add(ParseObjIndex(parts[i + 1], vertices.Count));
                        }
                    }
                }
            }

            mesh.Vertices = vertices;
            mesh.Indices = indices;
            return mesh;
        }

        private static int ParseObjIndex(string str, int vertexCount)
        {
            var parts = str.Split('/');
            int idx = int.Parse(parts[0]);
            if (idx < 0) idx += vertexCount;
            else idx -= 1; // OBJ is 1-based
            return idx;
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
            return new MeshData
            {
                Vertices = plyData.Vertices,
                Colors = plyData.Colors,
                Indices = plyData.Indices
            };
        }
    }
}
