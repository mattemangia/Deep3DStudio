using System;
using System.IO;
using System.Reflection;
using Python.Runtime;
using Deep3DStudio.Python;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.Model
{
    public abstract class BasePythonInference : IDisposable
    {
        protected dynamic _bridgeModule;
        protected bool _isLoaded = false;
        protected string _modelName;
        protected bool _disposed = false;

        public BasePythonInference(string modelName)
        {
            _modelName = modelName;
        }

        public bool IsLoaded => _isLoaded;

        protected string GetDeviceString()
        {
            var settings = IniSettings.Instance;
            return settings.AIDevice switch
            {
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.DirectML => "directml",
                AIComputeDevice.MPS => "mps",
                _ => "cpu"
            };
        }

        protected string GetModelPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = _modelName switch
            {
                "triposr" => settings.TripoSRModelPath,
                "lgm" => settings.LGMModelPath,
                "triposf" => settings.TripoSFModelPath,
                "wonder3d" => settings.Wonder3DModelPath,
                "unirig" => settings.UniRigModelPath,
                _ => $"models/{_modelName}"
            };

            // If path is relative, make it absolute from app directory
            if (!Path.IsPathRooted(configuredPath))
            {
                configuredPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, configuredPath);
            }

            // Return the weights file path
            return Path.Combine(configuredPath, $"{_modelName}_weights.pth");
        }

        protected void Initialize()
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

                    string weightsPath = GetModelPath();
                    string device = GetDeviceString();

                    // Load the model with configured device
                    _bridgeModule.load_model(_modelName, weightsPath, device);
                });

                _isLoaded = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error initializing {_modelName}: {ex.Message}");
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Cleanup managed resources if any
                }
                _disposed = true;
            }
        }
    }
}
