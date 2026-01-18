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
    /// <summary>
    /// MASt3R (Matching And Stereo 3D Reconstruction) inference handler.
    /// MASt3R builds upon DUSt3R with metric pointmaps and dense feature maps for better matching.
    /// Best for: 2+ images, metric reconstruction, image matching scenarios.
    /// </summary>
    public class Mast3rInference : IDisposable
    {
        private bool _isLoaded = false;
        private dynamic _bridgeModule;
        private bool _disposed = false;
        private Action<string>? _logCallback;
        private string? _lastError;

        public Mast3rInference()
        {
            // Initialization is lazy
        }

        public bool IsLoaded => _isLoaded;

        /// <summary>
        /// Gets the last error message if initialization failed.
        /// </summary>
        public string? LastError => _lastError;

        /// <summary>
        /// Sets a callback for log messages.
        /// </summary>
        public Action<string>? LogCallback
        {
            set => _logCallback = value;
        }

        private void Log(string message)
        {
            Console.WriteLine(message);
            _logCallback?.Invoke(message);
        }

        private string GetDeviceString()
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

        private void Initialize()
        {
            if (_isLoaded) return;

            try
            {
                PythonService.Instance.Initialize();

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    dynamic sys = Py.Import("sys");
                    if (sys.modules.__contains__("deep3dstudio_bridge"))
                    {
                        _bridgeModule = sys.modules["deep3dstudio_bridge"];
                    }
                    else
                    {
                        // Try to find the embedded resource from any loaded assembly
                        string scriptContent = null;
                        string[] possibleNames = new[]
                        {
                            "Deep3DStudio.Embedded.Python.inference_bridge.py",
                            "Deep3DStudio.Cross.Embedded.Python.inference_bridge.py"
                        };

                        var assemblyList = new List<Assembly>();
                        if (Assembly.GetEntryAssembly() != null) assemblyList.Add(Assembly.GetEntryAssembly());
                        if (Assembly.GetExecutingAssembly() != null) assemblyList.Add(Assembly.GetExecutingAssembly());
                        if (Assembly.GetCallingAssembly() != null) assemblyList.Add(Assembly.GetCallingAssembly());
                        assemblyList.AddRange(AppDomain.CurrentDomain.GetAssemblies()
                            .Where(a => a.FullName?.Contains("Deep3DStudio") == true));

                        foreach (var assembly in assemblyList.Distinct())
                        {
                            if (assembly == null) continue;

                            var allResources = assembly.GetManifestResourceNames();
                            Console.WriteLine($"[Mast3r] Checking assembly '{assembly.GetName().Name}' with {allResources.Length} resources");

                            foreach (var resourceName in possibleNames)
                            {
                                if (allResources.Contains(resourceName))
                                {
                                    Console.WriteLine($"[Mast3r] Found exact match: {resourceName}");
                                    using (Stream stream = assembly.GetManifestResourceStream(resourceName))
                                    {
                                        if (stream != null)
                                        {
                                            using (StreamReader reader = new StreamReader(stream))
                                            {
                                                scriptContent = reader.ReadToEnd();
                                                break;
                                            }
                                        }
                                    }
                                }
                            }

                            if (scriptContent != null) break;

                            var match = allResources.FirstOrDefault(r => r.EndsWith("inference_bridge.py"));
                            if (match != null)
                            {
                                Console.WriteLine($"[Mast3r] Found by suffix: {match}");
                                using (Stream stream = assembly.GetManifestResourceStream(match))
                                {
                                    if (stream != null)
                                    {
                                        using (StreamReader reader = new StreamReader(stream))
                                        {
                                            scriptContent = reader.ReadToEnd();
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        if (string.IsNullOrEmpty(scriptContent))
                        {
                            throw new FileNotFoundException("Could not find embedded resource 'inference_bridge.py' in any assembly.");
                        }

                        dynamic types = Py.Import("types");
                        _bridgeModule = types.ModuleType("deep3dstudio_bridge");
                        dynamic builtins = Py.Import("builtins");
                        builtins.exec(scriptContent, _bridgeModule.__dict__);
                        sys.modules["deep3dstudio_bridge"] = _bridgeModule;
                    }

                    var settings = IniSettings.Instance;
                    string configuredPath = settings.Mast3rModelPath;
                    string weightsPath;

                    if (configuredPath.Contains("/") && !Path.IsPathRooted(configuredPath) && !configuredPath.StartsWith("models"))
                    {
                        weightsPath = configuredPath;
                    }
                    else if (Path.IsPathRooted(configuredPath))
                    {
                        weightsPath = configuredPath;
                    }
                    else
                    {
                        string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, configuredPath);

                        if (Directory.Exists(basePath))
                        {
                            string pthFile = Path.Combine(basePath, "mast3r_weights.pth");
                            if (File.Exists(pthFile))
                                weightsPath = pthFile;
                            else
                                weightsPath = basePath;
                        }
                        else if (File.Exists(basePath))
                        {
                            weightsPath = basePath;
                        }
                        else
                        {
                            weightsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "mast3r_weights.pth");
                        }
                    }

                    string device = GetDeviceString();
                    Log($"[Mast3r] Loading model from: {weightsPath}");

                    _bridgeModule.load_model("mast3r", weightsPath, device);
                });

                _isLoaded = true;
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Mast3r: {ex.Message}";
                Log(_lastError);
            }
        }

        /// <summary>
        /// Reconstruct 3D scene from multiple images using MASt3R.
        /// MASt3R provides metric pointmaps and better feature matching than DUSt3R.
        /// </summary>
        /// <param name="imagePaths">List of image file paths</param>
        /// <param name="useRetrieval">If true, use retrieval model for optimal pairing of unordered images</param>
        public SceneResult ReconstructScene(List<string> imagePaths, bool useRetrieval = true)
        {
            Initialize();
            var result = new SceneResult();

            if (!_isLoaded)
            {
                if (_lastError != null)
                {
                    Log($"Mast3r not loaded. {_lastError}");
                }
                return result;
            }

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

                        // Pass use_retrieval parameter for optimal pairing of unordered images
                        dynamic output = _bridgeModule.infer_mast3r(pyList, use_retrieval: useRetrieval);

                        int len = (int)output.__len__();
                        for (int i = 0; i < len; i++)
                        {
                            dynamic item = output[i];
                            var mesh = new MeshData();

                            dynamic vertices = item["vertices"];
                            dynamic colors = item["colors"];
                            dynamic faces = item["faces"];
                            int imageIndex = i;
                            try
                            {
                                if (item.__contains__("image_index"))
                                {
                                    imageIndex = (int)item["image_index"];
                                }
                            }
                            catch (Exception)
                            {
                                imageIndex = i;
                            }

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
                            if (imageIndex >= 0 && imageIndex < imagePaths.Count)
                            {
                                result.Poses.Add(new CameraPose { ImageIndex = imageIndex, ImagePath = imagePaths[imageIndex] });
                            }
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                _lastError = $"Mast3r inference failed: {ex.Message}";
                Log(_lastError);
                Log($"Stack trace: {ex.StackTrace}");
            }

            return result;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _disposed = true;

                try
                {
                    _bridgeModule = null;
                    _isLoaded = false;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Mast3r] Dispose warning: {ex.Message}");
                }
            }
        }
    }
}
