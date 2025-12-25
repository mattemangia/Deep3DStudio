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

        // Progress callback: (stage, progress, message)
        public event Action<string, float, string>? OnLoadProgress;

        public BasePythonInference(string modelName)
        {
            _modelName = modelName;
        }

        public bool IsLoaded => _isLoaded;

        /// <summary>
        /// Reports loading progress to any registered listeners
        /// </summary>
        protected void ReportProgress(string stage, float progress, string message)
        {
            OnLoadProgress?.Invoke(stage, progress, message);
        }

        protected string GetDeviceString()
        {
            var settings = IniSettings.Instance;
            return settings.AIDevice switch
            {
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.ROCm => "rocm",
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
                ReportProgress("init", 0.0f, $"Initializing Python for {_modelName}...");
                PythonService.Instance.Initialize();

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    ReportProgress("init", 0.05f, "Loading inference bridge...");
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

                    // Set up progress callback from Python to C#
                    SetupPythonProgressCallback();

                    string weightsPath = GetModelPath();
                    string device = GetDeviceString();

                    ReportProgress("load", 0.1f, $"Loading {_modelName} model...");

                    // Load the model with configured device
                    bool success = _bridgeModule.load_model(_modelName, weightsPath, device);

                    if (!success)
                    {
                        throw new Exception($"Failed to load model {_modelName}");
                    }
                });

                _isLoaded = true;
                ReportProgress("load", 1.0f, $"{_modelName} loaded successfully");
            }
            catch (Exception ex)
            {
                ReportProgress("error", 0.0f, $"Error initializing {_modelName}: {ex.Message}");
                Console.WriteLine($"Error initializing {_modelName}: {ex.Message}");
            }
        }

        /// <summary>
        /// Sets up a Python callback to receive progress updates
        /// </summary>
        private void SetupPythonProgressCallback()
        {
            try
            {
                // Create a Python-callable delegate
                Action<string, float, string> progressAction = (stage, progress, message) =>
                {
                    ReportProgress(stage, progress, message);
                };

                // Use PyObject to wrap the delegate - this requires the callback to be invoked within GIL
                // For now, progress will be reported via print statements in Python
                // and we'll report key stages from C# side
            }
            catch
            {
                // Callback setup is optional - continue without it
            }
        }

        /// <summary>
        /// Unload the model to free GPU memory
        /// </summary>
        public void Unload()
        {
            if (!_isLoaded || _bridgeModule == null) return;

            try
            {
                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    _bridgeModule.unload_model(_modelName);
                });
                _isLoaded = false;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error unloading {_modelName}: {ex.Message}");
            }
        }

        /// <summary>
        /// Get current GPU memory usage info
        /// </summary>
        public (float UsedMB, float TotalMB) GetGPUMemoryInfo()
        {
            if (_bridgeModule == null) return (0, 0);

            try
            {
                (float, float) result = (0, 0);
                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    dynamic memInfo = _bridgeModule.get_gpu_memory_info();
                    result = ((float)memInfo[0], (float)memInfo[1]);
                });
                return result;
            }
            catch
            {
                return (0, 0);
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
