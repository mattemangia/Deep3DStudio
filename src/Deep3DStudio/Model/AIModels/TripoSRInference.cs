using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using Deep3DStudio.Configuration;
using OpenTK.Mathematics;
using System.Collections.Generic;

namespace Deep3DStudio.Model.AIModels
{
    public class TripoSRInference : BasePythonInference
    {
        public TripoSRInference() : base("triposr") { }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();
            if (!_isLoaded) return mesh;

            try
            {
                byte[] imgBytes = File.ReadAllBytes(imagePath);

                // Get settings for model parameters
                var settings = IniSettings.Instance;
                int resolution = settings.TripoSRResolution;
                int mcResolution = settings.TripoSRMarchingCubesRes;

                Deep3DStudio.Python.PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    // Pass configured parameters to Python
                    dynamic output = _bridgeModule.infer_triposr(
                        imgBytes.ToPython(),
                        resolution,
                        mcResolution
                    );
                    if (output != null)
                    {
                        // Parse output dictionary from Python
                        // 'vertices': np array (N, 3)
                        // 'faces': np array (M, 3)
                        // 'colors': np array (N, 3)

                        dynamic vertices = output["vertices"];
                        dynamic faces = output["faces"];
                        dynamic colors = output["colors"];

                        long vCount = (long)vertices.shape[0];
                        long fCount = (long)faces.shape[0];

                        for(int i=0; i<vCount; i++) {
                            float x = (float)vertices[i][0];
                            float y = (float)vertices[i][1];
                            float z = (float)vertices[i][2];
                            mesh.Vertices.Add(new Vector3(x, y, z));

                            float r = (float)colors[i][0];
                            float g = (float)colors[i][1];
                            float b = (float)colors[i][2];
                            mesh.Colors.Add(new Vector3(r, g, b));
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
                Console.WriteLine($"TripoSR Inference failed: {ex.Message}");
            }
            return mesh;
        }
    }
}
