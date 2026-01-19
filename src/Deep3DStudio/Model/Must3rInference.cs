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
    /// MUSt3R (Multi-view Network for Stereo 3D Reconstruction) inference handler.
    /// MUSt3R is optimized for multi-view reconstruction with many images.
    /// Supports video/streaming scenarios at 8-11 FPS.
    /// Best for: Many images (>2), video input, real-time scenarios.
    /// </summary>
    public class Must3rInference : IDisposable
    {
        private bool _isLoaded = false;
        private dynamic _bridgeModule;
        private bool _disposed = false;
        private Action<string>? _logCallback;
        private string? _lastError;

        public Must3rInference()
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
                            Console.WriteLine($"[Must3r] Checking assembly '{assembly.GetName().Name}' with {allResources.Length} resources");

                            foreach (var resourceName in possibleNames)
                            {
                                if (allResources.Contains(resourceName))
                                {
                                    Console.WriteLine($"[Must3r] Found exact match: {resourceName}");
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
                                Console.WriteLine($"[Must3r] Found by suffix: {match}");
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
                    string configuredPath = settings.Must3rModelPath;
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
                            string pthFile = Path.Combine(basePath, "must3r_weights.pth");
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
                            weightsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "must3r_weights.pth");
                        }
                    }

                    string device = GetDeviceString();
                    // CRITICAL: Don't call Log inside GIL - Console is safe inside GIL
                    Console.WriteLine($"[Must3r] Loading model from: {weightsPath}");

                    _bridgeModule.load_model("must3r", weightsPath, device);
                });

                // Log AFTER GIL is released to prevent GTK/GIL interaction issues
                Log($"[Must3r] Model loaded successfully");
                _isLoaded = true;
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Must3r: {ex.Message}";
                Log(_lastError);
            }
        }

        /// <summary>
        /// Reconstruct 3D scene from multiple images using MUSt3R.
        /// MUSt3R is optimized for many images with multi-layer memory mechanism.
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
                    Log($"Must3r not loaded. {_lastError}");
                }
                return result;
            }

            try
            {
                // Load image bytes - will be cleared after conversion to Python objects
                List<byte[]> imagesBytes = new List<byte[]>();
                foreach (var path in imagePaths)
                    imagesBytes.Add(File.ReadAllBytes(path));

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    using(var pyList = new PyList())
                    {
                        // Convert each image to Python and immediately clear the C# reference
                        // This prevents holding duplicate data in both C# and Python memory
                        for (int idx = 0; idx < imagesBytes.Count; idx++)
                        {
                            pyList.Append(imagesBytes[idx].ToPython());
                            imagesBytes[idx] = null; // Release C# reference immediately
                        }
                        imagesBytes.Clear(); // Clear the list to allow GC

                        // Pass use_memory=true and use_retrieval for optimal pairing of unordered images
                        // CRITICAL: Use PyObject explicitly so we can dispose it properly
                        PyObject outputObj = _bridgeModule.infer_must3r(pyList, use_memory: true, use_retrieval: useRetrieval);
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
                _lastError = $"Must3r inference failed: {ex.Message}";
                Log(_lastError);
                Log($"Stack trace: {ex.StackTrace}");
            }

            return result;
        }

        /// <summary>
        /// Reconstruct 3D scene from a video file using MUSt3R.
        /// MUSt3R is designed for video/streaming scenarios at 8-11 FPS.
        /// </summary>
        /// <param name="videoPath">Path to the video file</param>
        /// <param name="maxFrames">Maximum number of frames to extract (default: 100)</param>
        /// <param name="frameInterval">Extract every Nth frame (default: 5)</param>
        public SceneResult ReconstructFromVideo(string videoPath, int maxFrames = 100, int frameInterval = 5)
        {
            Initialize();
            var result = new SceneResult();

            if (!_isLoaded)
            {
                if (_lastError != null)
                {
                    Log($"Must3r not loaded. {_lastError}");
                }
                return result;
            }

            if (!File.Exists(videoPath))
            {
                _lastError = $"Video file not found: {videoPath}";
                Log(_lastError);
                return result;
            }

            try
            {
                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    // CRITICAL: Use PyObject explicitly so we can dispose it properly
                    PyObject outputObj = _bridgeModule.infer_must3r_video(videoPath, maxFrames, frameInterval);
                    try
                    {
                        int len = (int)outputObj.Length();
                        Log($"[Must3r] Video reconstruction produced {len} point clouds");

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
                                result.Poses.Add(new CameraPose { ImageIndex = imageIndex, ImagePath = $"frame_{imageIndex}" });
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
                });
            }
            catch (Exception ex)
            {
                _lastError = $"Must3r video inference failed: {ex.Message}";
                Log(_lastError);
                Log($"Stack trace: {ex.StackTrace}");
            }

            return result;
        }

        /// <summary>
        /// Check if video file format is supported.
        /// </summary>
        public static bool IsVideoSupported(string filePath)
        {
            if (string.IsNullOrEmpty(filePath)) return false;

            string ext = Path.GetExtension(filePath).ToLowerInvariant();
            return ext switch
            {
                ".mp4" => true,
                ".avi" => true,
                ".mov" => true,
                ".mkv" => true,
                ".webm" => true,
                ".wmv" => true,
                ".m4v" => true,
                _ => false
            };
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
                    Console.WriteLine($"[Must3r] Dispose warning: {ex.Message}");
                }
            }
        }
    }
}
