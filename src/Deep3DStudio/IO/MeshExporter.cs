using System;
using System.IO;
using Deep3DStudio.Model;

namespace Deep3DStudio.IO
{
    public static class MeshExporter
    {
        public static void Save(string filePath, MeshData mesh)
        {
            var options = new MeshExportOptions();

            // Determine format from extension
            string ext = Path.GetExtension(filePath).ToLower();
            switch (ext)
            {
                case ".obj":
                    options.Format = TexturedMeshFormat.OBJ;
                    break;
                case ".gltf":
                    options.Format = TexturedMeshFormat.GLTF;
                    break;
                case ".glb":
                    options.Format = TexturedMeshFormat.GLB;
                    break;
                case ".ply":
                    options.Format = TexturedMeshFormat.PLY;
                    break;
                case ".fbx":
                    options.Format = TexturedMeshFormat.FBX_ASCII;
                    break;
                default:
                    options.Format = TexturedMeshFormat.OBJ;
                    break;
            }

            TexturedMeshExporter.Export(filePath, mesh, null, null, options);
        }
    }
}
