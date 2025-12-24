using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Python.Runtime;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using SkiaSharp;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model
{
    public class Dust3rInference : IDisposable
    {
        private bool _isLoaded = false;
        private dynamic _bridgeModule;
        private bool _disposed = false;

        public Dust3rInference()
        {
            // Initialization is lazy
        }

        public bool IsLoaded => _isLoaded;

        private string GetDeviceString()
        {
            var settings = IniSettings.Instance;
            return settings.AIDevice switch
            {
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.DirectML => "directml",
                _ => "cpu"
            };
        }

        private void Initialize()
        {
            if (_isLoaded) return;

            try
            {
                PythonService.Instance.Initialize();

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    dynamic sys = Py.Import("sys");
                    if (sys.modules.Contains("deep3dstudio_bridge"))
                    {
                        _bridgeModule = sys.modules["deep3dstudio_bridge"];
                    }
                    else
                    {
                        var assembly = Assembly.GetExecutingAssembly();
                        var resourceName = "Deep3DStudio.Embedded.Python.inference_bridge.py";
                        string scriptContent;
                        using (Stream stream = assembly.GetManifestResourceStream(resourceName))
                        using (StreamReader reader = new StreamReader(stream))
                        {
                            scriptContent = reader.ReadToEnd();
                        }

                        dynamic types = Py.Import("types");
                        _bridgeModule = types.ModuleType("deep3dstudio_bridge");
                        PythonEngine.Exec(scriptContent, _bridgeModule.__dict__);
                        sys.modules["deep3dstudio_bridge"] = _bridgeModule;
                    }

                    string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");
                    string weightsPath = Path.Combine(modelsDir, "dust3r_weights.pth");
                    string device = GetDeviceString();

                    // Load the model with configured device
                    _bridgeModule.load_model("dust3r", weightsPath, device);
                });

                _isLoaded = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing Dust3r: {ex.Message}");
            }
        }

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            Initialize();
            var result = new SceneResult();

            if (!_isLoaded) return result;

            try
            {
                List<byte[]> imagesBytes = new List<byte[]>();
                foreach (var path in imagePaths)
                    imagesBytes.Add(File.ReadAllBytes(path));

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    using(var pyList = new PyList())
                    {
                        foreach(var b in imagesBytes)
                            pyList.Append(b.ToPython());

                        dynamic output = _bridgeModule.infer_dust3r(pyList);

                        int len = (int)output.__len__();
                        for (int i = 0; i < len; i++)
                        {
                            dynamic item = output[i];
                            var mesh = new MeshData();

                            dynamic vertices = item["vertices"];
                            dynamic colors = item["colors"];
                            dynamic faces = item["faces"];

                            long vCount = (long)vertices.shape[0];
                            long fCount = (long)faces.shape[0];

                            for(int v=0; v<vCount; v++) {
                                mesh.Vertices.Add(new Vector3((float)vertices[v][0], (float)vertices[v][1], (float)vertices[v][2]));
                                mesh.Colors.Add(new Vector3((float)colors[v][0], (float)colors[v][1], (float)colors[v][2]));
                            }
                            for(int f=0; f<fCount; f++) {
                                mesh.Indices.Add((int)faces[f][0]);
                                mesh.Indices.Add((int)faces[f][1]);
                                mesh.Indices.Add((int)faces[f][2]);
                            }

                            result.Meshes.Add(mesh);
                            result.Poses.Add(new CameraPose { ImageIndex = i, ImagePath = imagePaths[i] });
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Inference failed: {ex.Message}");
            }

            return result;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;
                // Cleanup logic if needed
            }
        }
    }
}
