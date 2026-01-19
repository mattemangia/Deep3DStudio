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
        private Action<string>? _logCallback;
        private string? _lastError;

        public Dust3rInference()
        {
            // Initialization is lazy
        }

        public bool IsLoaded => _isLoaded;

        /// <summary>
        /// Gets the last error message if initialization failed.
        /// </summary>
        public string? LastError => _lastError;

        /// <summary>
        /// Sets a callback for log messages. Messages will be sent to both console and this callback.
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

                        // Try all loaded assemblies, starting with entry and executing
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
                            Console.WriteLine($"[Dust3r] Checking assembly '{assembly.GetName().Name}' with {allResources.Length} resources");

                            // Try known names first
                            foreach (var resourceName in possibleNames)
                            {
                                if (allResources.Contains(resourceName))
                                {
                                    Console.WriteLine($"[Dust3r] Found exact match: {resourceName}");
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

                            // Fallback: search by filename
                            var match = allResources.FirstOrDefault(r => r.EndsWith("inference_bridge.py"));
                            if (match != null)
                            {
                                Console.WriteLine($"[Dust3r] Found by suffix: {match}");
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

                            // Log all resources for debugging
                            if (allResources.Length > 0 && allResources.Length < 20)
                            {
                                Console.WriteLine($"[Dust3r] Available resources in {assembly.GetName().Name}: {string.Join(", ", allResources)}");
                            }
                        }

                        if (string.IsNullOrEmpty(scriptContent))
                        {
                            throw new FileNotFoundException("Could not find embedded resource 'inference_bridge.py' in any assembly. Check that it's included as EmbeddedResource in the .csproj file.");
                        }

                        dynamic types = Py.Import("types");
                        _bridgeModule = types.ModuleType("deep3dstudio_bridge");
                        // Use Python's built-in exec instead of PythonEngine.Exec to avoid protection level issues
                        dynamic builtins = Py.Import("builtins");
                        builtins.exec(scriptContent, _bridgeModule.__dict__);
                        sys.modules["deep3dstudio_bridge"] = _bridgeModule;
                    }

                    var settings = IniSettings.Instance;
                    string configuredPath = settings.Dust3rModelPath;
                    string weightsPath;

                    // Check if it's a relative path, absolute path, or HuggingFace Hub identifier
                    if (configuredPath.Contains("/") && !Path.IsPathRooted(configuredPath) && !configuredPath.StartsWith("models"))
                    {
                        // Looks like a HuggingFace Hub identifier (e.g., "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt")
                        weightsPath = configuredPath;
                    }
                    else if (Path.IsPathRooted(configuredPath))
                    {
                        // Absolute path
                        weightsPath = configuredPath;
                    }
                    else
                    {
                        // Relative path - resolve from app directory
                        string basePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, configuredPath);

                        // Check if it's a directory containing dust3r_weights.pth
                        if (Directory.Exists(basePath))
                        {
                            string pthFile = Path.Combine(basePath, "dust3r_weights.pth");
                            if (File.Exists(pthFile))
                                weightsPath = pthFile;
                            else
                                weightsPath = basePath; // Maybe it's a HF-style directory
                        }
                        else if (File.Exists(basePath))
                        {
                            // It's a direct file path
                            weightsPath = basePath;
                        }
                        else
                        {
                            // Fallback: try models/dust3r_weights.pth
                            weightsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "dust3r_weights.pth");
                        }
                    }

                    string device = GetDeviceString();
                    // CRITICAL: Don't call Log inside GIL - Console is safe inside GIL
                    Console.WriteLine($"[Dust3r] Loading model from: {weightsPath}");

                    // Load the model with configured device
                    _bridgeModule.load_model("dust3r", weightsPath, device);
                });

                // Log AFTER GIL is released to prevent GTK/GIL interaction issues
                Log($"[Dust3r] Model loaded successfully");
                _isLoaded = true;
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Dust3r: {ex.Message}";
                Log(_lastError);
            }
        }

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            Initialize();
            var result = new SceneResult();

            if (!_isLoaded)
            {
                // Log the last error if we have one
                if (_lastError != null)
                {
                    Log($"Dust3r not loaded. {_lastError}");
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

                        // CRITICAL: Use PyObject explicitly so we can dispose it properly
                        PyObject outputObj = _bridgeModule.infer_dust3r(pyList);
                        try
                        {
                            int len = (int)outputObj.Length();
                            for (int i = 0; i < len; i++)
                            {
                                PyObject item = outputObj[i];
                                try
                                {
                                    var mesh = new MeshData();

                                    int imageIndex = i;

                                    // Extract data immediately to primitive types, disposing PyObjects ASAP
                                    using (PyObject verticesObj = item["vertices"])
                                    using (PyObject colorsObj = item["colors"])
                                    using (PyObject facesObj = item["faces"])
                                    {
                                        try
                                        {
                                            using (PyObject containsResult = item.InvokeMethod("__contains__", new PyString("image_index")))
                                            {
                                                if (containsResult.IsTrue())
                                                {
                                                    using (PyObject idxObj = item["image_index"])
                                                    {
                                                        imageIndex = idxObj.As<int>();
                                                    }
                                                }
                                            }
                                        }
                                        catch (Exception)
                                        {
                                            imageIndex = i;
                                        }

                                        // Get shapes as primitives immediately
                                        long vCount, fCount;
                                        using (PyObject vShapeObj = verticesObj.GetAttr("shape"))
                                        {
                                            vCount = vShapeObj[0].As<long>();
                                        }
                                        using (PyObject fShapeObj = facesObj.GetAttr("shape"))
                                        {
                                            fCount = fShapeObj[0].As<long>();
                                        }

                                        // Convert numpy arrays to Python lists for safe extraction
                                        // numpy scalars don't convert properly with As<float>()
                                        dynamic builtins = Py.Import("builtins");

                                        // Extract vertex and color data
                                        for(int v=0; v<vCount; v++)
                                        {
                                            using (PyObject vRow = verticesObj[v])
                                            using (PyObject cRow = colorsObj[v])
                                            {
                                                // Use Python's float() to convert numpy scalars to native float
                                                float vx = (float)(double)builtins.@float(vRow[0]);
                                                float vy = (float)(double)builtins.@float(vRow[1]);
                                                float vz = (float)(double)builtins.@float(vRow[2]);
                                                float cx = (float)(double)builtins.@float(cRow[0]);
                                                float cy = (float)(double)builtins.@float(cRow[1]);
                                                float cz = (float)(double)builtins.@float(cRow[2]);
                                                mesh.Vertices.Add(new Vector3(vx, vy, vz));
                                                mesh.Colors.Add(new Vector3(cx, cy, cz));
                                            }
                                        }

                                        // Extract face indices
                                        for(int f=0; f<fCount; f++)
                                        {
                                            using (PyObject fRow = facesObj[f])
                                            {
                                                mesh.Indices.Add((int)(long)builtins.@int(fRow[0]));
                                                mesh.Indices.Add((int)(long)builtins.@int(fRow[1]));
                                                mesh.Indices.Add((int)(long)builtins.@int(fRow[2]));
                                            }
                                        }
                                    }

                                    result.Meshes.Add(mesh);
                                    if (imageIndex >= 0 && imageIndex < imagePaths.Count)
                                    {
                                        result.Poses.Add(new CameraPose { ImageIndex = imageIndex, ImagePath = imagePaths[imageIndex] });
                                    }
                                }
                                finally
                                {
                                    item.Dispose();
                                }
                            }
                        }
                        finally
                        {
                            outputObj.Dispose();
                        }

                        // CRITICAL: Force garbage collection INSIDE the GIL block
                        // This ensures all PyObject finalizers run while we still hold the GIL
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        GC.Collect();
                    }
                });
            }
            catch (Exception ex)
            {
                _lastError = $"Dust3r inference failed: {ex.Message}";
                Log(_lastError);

                // Log the full stack trace for debugging
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
                    // Clean up Python object references within GIL context
                    // This is critical to prevent AccessViolationException
                    if (_bridgeModule != null && PythonService.Instance.IsInitialized)
                    {
                        try
                        {
                            PythonService.Instance.ExecuteWithGIL((scope) =>
                            {
                                // Release the reference within GIL context
                                _bridgeModule = null;
                            });
                        }
                        catch (Exception)
                        {
                            // If GIL acquisition fails, just clear the reference
                            _bridgeModule = null;
                        }
                    }
                    else
                    {
                        _bridgeModule = null;
                    }

                    _isLoaded = false;

                    // Note: We don't unload the Python module from sys.modules
                    // as that could cause issues if other code references it.
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Dust3r] Dispose warning: {ex.Message}");
                }
            }
        }
    }
}
