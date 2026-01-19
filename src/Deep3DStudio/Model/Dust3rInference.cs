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
        // CRITICAL: Use PyObject instead of dynamic to prevent GC from collecting
        // temporary PyObjects during method calls, which causes AccessViolationException
        private PyObject? _bridgeModule;
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
                    using var sys = Py.Import("sys");
                    using var modules = sys.GetAttr("modules");

                    // Check if module already exists using safe PyObject methods
                    bool moduleExists = false;
                    using (var containsMethod = modules.GetAttr("__contains__"))
                    using (var moduleName = new PyString("deep3dstudio_bridge"))
                    using (var containsResult = containsMethod.Invoke(new PyTuple(new PyObject[] { moduleName })))
                    {
                        moduleExists = containsResult.IsTrue();
                    }

                    if (moduleExists)
                    {
                        using var moduleName = new PyString("deep3dstudio_bridge");
                        _bridgeModule = modules[moduleName];
                    }
                    else
                    {
                        // Try to find the embedded resource from any loaded assembly
                        string? scriptContent = null;
                        string[] possibleNames = new[]
                        {
                            "Deep3DStudio.Embedded.Python.inference_bridge.py",
                            "Deep3DStudio.Cross.Embedded.Python.inference_bridge.py"
                        };

                        // Try all loaded assemblies, starting with entry and executing
                        var assemblyList = new List<Assembly>();
                        if (Assembly.GetEntryAssembly() != null) assemblyList.Add(Assembly.GetEntryAssembly()!);
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
                                    using (Stream? stream = assembly.GetManifestResourceStream(resourceName))
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
                                using (Stream? stream = assembly.GetManifestResourceStream(match))
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

                        // Create module using safe PyObject operations
                        using var types = Py.Import("types");
                        using var moduleTypeAttr = types.GetAttr("ModuleType");
                        using var moduleNameArg = new PyString("deep3dstudio_bridge");
                        _bridgeModule = moduleTypeAttr.Invoke(new PyTuple(new PyObject[] { moduleNameArg }));

                        using var builtins = Py.Import("builtins");
                        using var execFunc = builtins.GetAttr("exec");
                        using var scriptPy = new PyString(scriptContent);
                        using var moduleDict = _bridgeModule.GetAttr("__dict__");
                        execFunc.Invoke(new PyTuple(new PyObject[] { scriptPy, moduleDict }));

                        // Register in sys.modules
                        using var setItemArgs = new PyTuple(new PyObject[] { moduleNameArg, _bridgeModule });
                        using var setItemMethod = modules.GetAttr("__setitem__");
                        setItemMethod.Invoke(setItemArgs);
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

                    // CRITICAL: Use explicit PyObject method calls instead of dynamic
                    // to prevent GC from collecting temporary arguments
                    using var loadModelMethod = _bridgeModule.GetAttr("load_model");
                    using var modelNameArg = new PyString("dust3r");
                    using var weightsPathArg = new PyString(weightsPath);
                    using var deviceArg = new PyString(device);
                    using var loadArgs = new PyTuple(new PyObject[] { modelNameArg, weightsPathArg, deviceArg });
                    loadModelMethod.Invoke(loadArgs);
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
                if (imagePaths == null || imagePaths.Count == 0)
                {
                    Log("[Dust3r] ReconstructScene called with no images.");
                    return result;
                }

                // Load image bytes - kept alive during Python call to prevent GC issues
                List<byte[]> imagesBytes = new List<byte[]>();
                List<string> validImagePaths = new List<string>();
                foreach (var path in imagePaths)
                {
                    if (string.IsNullOrWhiteSpace(path))
                    {
                        Log("[Dust3r] Skipping empty image path.");
                        continue;
                    }

                    if (!File.Exists(path))
                    {
                        Log($"[Dust3r] Skipping missing image path: {path}");
                        continue;
                    }

                    imagesBytes.Add(File.ReadAllBytes(path));
                    validImagePaths.Add(path);
                }

                if (imagesBytes.Count == 0)
                {
                    Log("[Dust3r] No valid images to process.");
                    return result;
                }

                Log($"[Dust3r] Starting inference for {imagesBytes.Count} image(s).");

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

                            // CRITICAL: Use explicit PyObject method calls instead of dynamic
                            // to prevent GC from collecting temporary arguments
                            using var inferMethod = _bridgeModule!.GetAttr("infer_dust3r");
                            using var args = new PyTuple(new PyObject[] { pyList });

                            Console.WriteLine($"[Dust3r] Calling infer_dust3r...");
                            PyObject outputObj = inferMethod.Invoke(args);
                            try
                            {
                                int len = (int)outputObj.Length();
                                Log($"[Dust3r] Python returned {len} mesh result(s).");

                                // Import builtins once and keep reference alive
                                using var builtins = Py.Import("builtins");
                                using var floatFunc = builtins.GetAttr("float");
                                using var intFunc = builtins.GetAttr("int");

                                for (int i = 0; i < len; i++)
                                {
                                    using PyObject item = outputObj[i];
                                    var mesh = new MeshData();

                                    int imageIndex = i;

                                    // Extract data immediately to primitive types, disposing PyObjects ASAP
                                    using (PyObject verticesObj = item["vertices"])
                                    using (PyObject colorsObj = item["colors"])
                                    using (PyObject facesObj = item["faces"])
                                    {
                                        try
                                        {
                                            using (var keyStr = new PyString("image_index"))
                                            using (PyObject containsResult = item.InvokeMethod("__contains__", keyStr))
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

                                        // Extract vertex and color data using safe PyObject calls
                                        for(int v=0; v<vCount; v++)
                                        {
                                            using (PyObject vRow = verticesObj[v])
                                            using (PyObject cRow = colorsObj[v])
                                            {
                                                // Use Python's float() to convert numpy scalars to native float
                                                // Keep intermediate PyObjects in using blocks
                                                using var vx0 = vRow[0];
                                                using var vy0 = vRow[1];
                                                using var vz0 = vRow[2];
                                                using var cx0 = cRow[0];
                                                using var cy0 = cRow[1];
                                                using var cz0 = cRow[2];

                                                using var vxArgs = new PyTuple(new PyObject[] { vx0 });
                                                using var vyArgs = new PyTuple(new PyObject[] { vy0 });
                                                using var vzArgs = new PyTuple(new PyObject[] { vz0 });
                                                using var cxArgs = new PyTuple(new PyObject[] { cx0 });
                                                using var cyArgs = new PyTuple(new PyObject[] { cy0 });
                                                using var czArgs = new PyTuple(new PyObject[] { cz0 });

                                                using var vxPy = floatFunc.Invoke(vxArgs);
                                                using var vyPy = floatFunc.Invoke(vyArgs);
                                                using var vzPy = floatFunc.Invoke(vzArgs);
                                                using var cxPy = floatFunc.Invoke(cxArgs);
                                                using var cyPy = floatFunc.Invoke(cyArgs);
                                                using var czPy = floatFunc.Invoke(czArgs);

                                                float vx = (float)vxPy.As<double>();
                                                float vy = (float)vyPy.As<double>();
                                                float vz = (float)vzPy.As<double>();
                                                float cx = (float)cxPy.As<double>();
                                                float cy = (float)cyPy.As<double>();
                                                float cz = (float)czPy.As<double>();

                                                mesh.Vertices.Add(new Vector3(vx, vy, vz));
                                                mesh.Colors.Add(new Vector3(cx, cy, cz));
                                            }
                                        }

                                        // Extract face indices
                                        for(int f=0; f<fCount; f++)
                                        {
                                            using (PyObject fRow = facesObj[f])
                                            {
                                                using var f0 = fRow[0];
                                                using var f1 = fRow[1];
                                                using var f2 = fRow[2];

                                                using var f0Args = new PyTuple(new PyObject[] { f0 });
                                                using var f1Args = new PyTuple(new PyObject[] { f1 });
                                                using var f2Args = new PyTuple(new PyObject[] { f2 });

                                                using var i0Py = intFunc.Invoke(f0Args);
                                                using var i1Py = intFunc.Invoke(f1Args);
                                                using var i2Py = intFunc.Invoke(f2Args);

                                                mesh.Indices.Add((int)i0Py.As<long>());
                                                mesh.Indices.Add((int)i1Py.As<long>());
                                                mesh.Indices.Add((int)i2Py.As<long>());
                                            }
                                        }
                                    }

                                    result.Meshes.Add(mesh);
                                    if (imageIndex >= 0 && imageIndex < validImagePaths.Count)
                                    {
                                        result.Poses.Add(new CameraPose { ImageIndex = imageIndex, ImagePath = validImagePaths[imageIndex] });
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
                                // Dispose the PyObject within GIL context
                                _bridgeModule?.Dispose();
                                _bridgeModule = null;
                            });
                        }
                        catch (Exception)
                        {
                            // If GIL acquisition fails, try to dispose anyway
                            try { _bridgeModule?.Dispose(); } catch { }
                            _bridgeModule = null;
                        }
                    }
                    else
                    {
                        try { _bridgeModule?.Dispose(); } catch { }
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
