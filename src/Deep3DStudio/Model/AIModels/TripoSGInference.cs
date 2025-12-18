using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model.AIModels
{
    public class TripoSGInference : BasePythonInference
    {
        public TripoSGInference() : base("triposg") { }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();
            if (!_isLoaded) return mesh;

            try
            {
                byte[] imgBytes = File.ReadAllBytes(imagePath);
                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    dynamic output = _bridgeModule.infer_triposg(imgBytes.ToPython());
                    if (output != null)
                    {
                        // Parsing logic
                        dynamic vertices = output["vertices"];
                        dynamic faces = output["faces"];
                        dynamic colors = output["colors"];

                        long vCount = (long)vertices.shape[0];
                        long fCount = (long)faces.shape[0];

                        for(int i=0; i<vCount; i++) {
                            mesh.Vertices.Add(new Vector3((float)vertices[i][0], (float)vertices[i][1], (float)vertices[i][2]));
                            mesh.Colors.Add(new Vector3((float)colors[i][0], (float)colors[i][1], (float)colors[i][2]));
                        }
                        for(int i=0; i<fCount; i++) {
                            mesh.Indices.Add((int)faces[i][0]);
                            mesh.Indices.Add((int)faces[i][1]);
                            mesh.Indices.Add((int)faces[i][2]);
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"TripoSG Inference failed: {ex.Message}");
            }
            return mesh;
        }
    }
}
