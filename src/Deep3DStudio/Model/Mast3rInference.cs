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
                    // CRITICAL: Don't call Log inside GIL - store for logging after GIL release
                    string logMsg = $"[Mast3r] Loading model from: {weightsPath}";
                    Console.WriteLine(logMsg); // Console is safe inside GIL

                    _bridgeModule.load_model("mast3r", weightsPath, device);
                });

                // Log AFTER GIL is released to prevent GTK/GIL interaction issues
                Log($"[Mast3r] Model loaded successfully");
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
                if (imagePaths == null || imagePaths.Count == 0)
                {
                    Log("[Mast3r] ReconstructScene called with no images.");
                    return result;
                }

                // Load image bytes - kept alive during Python call to prevent GC issues
                List<byte[]> imagesBytes = new List<byte[]>();
                List<string> validImagePaths = new List<string>();
                foreach (var path in imagePaths)
                {
                    if (string.IsNullOrWhiteSpace(path))
                    {
                        Log("[Mast3r] Skipping empty image path.");
                        continue;
                    }

                    if (!File.Exists(path))
                    {
                        Log($"[Mast3r] Skipping missing image path: {path}");
                        continue;
                    }

                    imagesBytes.Add(File.ReadAllBytes(path));
                    validImagePaths.Add(path);
                }

                if (imagesBytes.Count == 0)
                {
                    Log("[Mast3r] No valid images to process.");
                    return result;
                }

                Log($"[Mast3r] Starting inference for {imagesBytes.Count} image(s). Use retrieval: {useRetrieval}.");

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    using(var pyList = new PyList())
                    {
                        // Convert each image to Python - keep C# references alive until Python call completes
                        // to prevent GC from collecting data that Python might still be copying
                        var pyImages = new List<PyObject>();
                        try
                        {
                            foreach(var b in imagesBytes)
                            {
                                var pyBytes = b.ToPython();
                                pyImages.Add(pyBytes);
                                pyList.Append(pyBytes);
                            }

                            // Pass use_retrieval parameter for optimal pairing of unordered images
                            // CRITICAL: Use PyObject explicitly so we can dispose it properly
                            PyObject outputObj = _bridgeModule.infer_mast3r(pyList, use_retrieval: useRetrieval);
                            try
                            {
                                int len = (int)outputObj.Length();
                                Log($"[Mast3r] Python returned {len} mesh result(s).");
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
                                        if (imageIndex >= 0 && imageIndex < validImagePaths.Count)
                                        {
                                            result.Poses.Add(new CameraPose { ImageIndex = imageIndex, ImagePath = validImagePaths[imageIndex] });
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
                        finally
                        {
                            foreach (var pyImage in pyImages)
                            {
                                pyImage.Dispose();
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
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Mast3r] Dispose warning: {ex.Message}");
                }
            }
        }
    }
}
