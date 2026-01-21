using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;

namespace Deep3DStudio.Python
{
    /// <summary>
    /// Subprocess-based inference that runs Python in a completely isolated process.
    /// This avoids all pythonnet memory corruption issues by using process isolation.
    /// </summary>
    public class SubprocessInference : IDisposable
    {
        private readonly string _pythonPath;
        private readonly string _scriptPath;
        private readonly string _modelName;
        private string? _weightsPath;
        private string _device = "cuda";
        private bool _isLoaded = false;
        private bool _disposed = false;
        private Process? _persistentProcess;
        private readonly object _processLock = new object();

        public event Action<string>? OnLog;
        public event Action<string, float, string>? OnProgress;

        public bool IsLoaded => _isLoaded;

        public SubprocessInference(string modelName)
        {
            _modelName = modelName;
            _pythonPath = GetPythonPath();
            _scriptPath = GetScriptPath();

            Log($"SubprocessInference initialized for {modelName}");
            Log($"Python path: {_pythonPath}");
            Log($"Script path: {_scriptPath}");
        }

        private void Log(string message)
        {
            Console.WriteLine($"[SubprocessInference:{_modelName}] {message}");
            OnLog?.Invoke(message);
        }

        private string GetPythonPath()
        {
            bool isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;

            // 1. First check LocalApplicationData/Deep3DStudio/python (where PythonService extracts it)
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string appDataPythonDir = Path.Combine(appData, "Deep3DStudio", "python");

            string? pythonExe = FindPythonExecutable(appDataPythonDir, isWindows);
            if (pythonExe != null)
            {
                Log($"Found Python in AppData: {pythonExe}");
                return pythonExe;
            }

            // 2. Check for local python directory (dev mode)
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string localPythonDir = Path.Combine(exeDir, "python");

            pythonExe = FindPythonExecutable(localPythonDir, isWindows);
            if (pythonExe != null)
            {
                Log($"Found local Python: {pythonExe}");
                return pythonExe;
            }

            // 3. Try system Python
            string[] systemPaths = isWindows
                ? new[] { "python.exe", "python3.exe" }
                : new[] { "/usr/bin/python3", "/usr/local/bin/python3" };

            foreach (var path in systemPaths)
            {
                if (File.Exists(path))
                {
                    Log($"Using system Python: {path}");
                    return path;
                }
            }

            // 4. Try using 'which' or 'where' to find python
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = isWindows ? "where" : "which",
                    Arguments = "python3",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var proc = Process.Start(psi);
                if (proc != null)
                {
                    string output = proc.StandardOutput.ReadLine() ?? "";
                    proc.WaitForExit();
                    if (!string.IsNullOrEmpty(output) && File.Exists(output))
                    {
                        Log($"Found Python via which/where: {output}");
                        return output;
                    }
                }
            }
            catch { }

            Log("Warning: Could not find Python. Please ensure python_env.zip is extracted.");
            return isWindows ? "python.exe" : "python3"; // Last resort
        }

        /// <summary>
        /// Recursively search for Python executable in the given directory.
        /// This handles the nested structure of indygreg python-build-standalone.
        /// </summary>
        private string? FindPythonExecutable(string rootDir, bool isWindows)
        {
            if (!Directory.Exists(rootDir)) return null;

            string exeName = isWindows ? "python.exe" : "python3";

            // Search for the executable using a DFS
            var stack = new Stack<string>();
            stack.Push(rootDir);
            int safetyCounter = 0;

            while (stack.Count > 0 && safetyCounter++ < 500)
            {
                string currentDir = stack.Pop();

                // Check common locations in this directory
                string[] candidates = isWindows
                    ? new[] { Path.Combine(currentDir, "python.exe") }
                    : new[] {
                        Path.Combine(currentDir, "bin", "python3"),
                        Path.Combine(currentDir, "bin", "python"),
                        Path.Combine(currentDir, "python3"),
                        Path.Combine(currentDir, "python")
                    };

                foreach (var candidate in candidates)
                {
                    if (File.Exists(candidate))
                        return candidate;
                }

                try
                {
                    foreach (string dir in Directory.GetDirectories(currentDir))
                    {
                        string dirName = Path.GetFileName(dir);
                        // Skip common non-python directories
                        if (dirName != "__pycache__" && dirName != "site-packages" && !dirName.StartsWith("."))
                            stack.Push(dir);
                    }
                }
                catch { /* Ignore access errors */ }
            }

            return null;
        }

        private string GetScriptPath()
        {
            // First check if it's deployed alongside the executable
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string[] possiblePaths = new[]
            {
                Path.Combine(exeDir, "subprocess_inference.py"),
                Path.Combine(exeDir, "Embedded", "Python", "subprocess_inference.py"),
                Path.Combine(exeDir, "..", "Embedded", "Python", "subprocess_inference.py"),
                // Development paths
                Path.Combine(exeDir, "..", "..", "..", "..", "Embedded", "Python", "subprocess_inference.py"),
                Path.Combine(exeDir, "..", "..", "..", "..", "..", "src", "Deep3DStudio", "Embedded", "Python", "subprocess_inference.py"),
            };

            foreach (var path in possiblePaths)
            {
                string fullPath = Path.GetFullPath(path);
                if (File.Exists(fullPath))
                {
                    return fullPath;
                }
            }

            // Extract from embedded resource to temp file
            return ExtractScriptToTemp();
        }

        private string ExtractScriptToTemp()
        {
            string tempPath = Path.Combine(Path.GetTempPath(), "deep3dstudio_subprocess_inference.py");

            // Check if we need to extract
            if (File.Exists(tempPath))
            {
                // Check if it's recent (within last hour)
                var fileInfo = new FileInfo(tempPath);
                if ((DateTime.Now - fileInfo.LastWriteTime).TotalHours < 1)
                {
                    return tempPath;
                }
            }

            // Try to extract from embedded resource
            var assembly = System.Reflection.Assembly.GetExecutingAssembly();
            string[] resourceNames = new[]
            {
                "Deep3DStudio.Embedded.Python.subprocess_inference.py",
                "Deep3DStudio.Cross.Embedded.Python.subprocess_inference.py"
            };

            foreach (var resourceName in resourceNames)
            {
                using var stream = assembly.GetManifestResourceStream(resourceName);
                if (stream != null)
                {
                    using var reader = new StreamReader(stream);
                    string content = reader.ReadToEnd();
                    File.WriteAllText(tempPath, content);
                    Log($"Extracted script to {tempPath}");
                    return tempPath;
                }
            }

            // Search all assemblies
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                try
                {
                    var resources = asm.GetManifestResourceNames();
                    foreach (var res in resources)
                    {
                        if (res.EndsWith("subprocess_inference.py"))
                        {
                            using var stream = asm.GetManifestResourceStream(res);
                            if (stream != null)
                            {
                                using var reader = new StreamReader(stream);
                                string content = reader.ReadToEnd();
                                File.WriteAllText(tempPath, content);
                                Log($"Extracted script from {asm.GetName().Name} to {tempPath}");
                                return tempPath;
                            }
                        }
                    }
                }
                catch { }
            }

            throw new FileNotFoundException("Could not find subprocess_inference.py embedded resource");
        }

        private (int exitCode, string stdout, string stderr) RunPythonCommand(string arguments, int timeoutMs = 300000, CancellationToken cancellationToken = default)
        {
            Log($"Running: {_pythonPath} {_scriptPath} {arguments}");

            cancellationToken.ThrowIfCancellationRequested();

            var psi = new ProcessStartInfo
            {
                FileName = _pythonPath,
                Arguments = $"\"{_scriptPath}\" {arguments}",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };

            // Set up environment - add Python directory to PATH
            string pythonDir = Path.GetDirectoryName(_pythonPath) ?? "";
            if (!string.IsNullOrEmpty(pythonDir))
            {
                string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                string pathSep = Environment.OSVersion.Platform == PlatformID.Win32NT ? ";" : ":";
                string scriptsDir = Environment.OSVersion.Platform == PlatformID.Win32NT
                    ? Path.Combine(pythonDir, "Scripts")
                    : pythonDir; // On Unix, scripts are in the same bin directory
                string libDir = Path.Combine(pythonDir, "Library", "bin");

                psi.Environment["PATH"] = $"{pythonDir}{pathSep}{scriptsDir}{pathSep}{libDir}{pathSep}{currentPath}";

                // Set PYTHONHOME for the standalone Python distribution
                string pythonHome = Path.GetDirectoryName(pythonDir) ?? pythonDir;
                if (Environment.OSVersion.Platform != PlatformID.Win32NT)
                {
                    // On Unix, python binary is in bin/, so go up one level
                    pythonHome = Path.GetDirectoryName(pythonDir) ?? pythonDir;
                }
                else
                {
                    pythonHome = pythonDir;
                }
                psi.Environment["PYTHONHOME"] = pythonHome;
            }

            using var process = new Process { StartInfo = psi };
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();

            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    stdout.AppendLine(e.Data);
                    // Log stdout in real-time so it appears in TUI
                    // Filter out some noise if necessary, but generally we want to see it
                    if (!e.Data.StartsWith("{") && !e.Data.Trim().StartsWith("{")) // Don't log JSON payloads if they are huge
                         Log($"[Py] {e.Data}");
                }
            };

            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    stderr.AppendLine(e.Data);
                    // Log stderr in real-time for debugging
                    Log($"[stderr] {e.Data}");
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            using var cancellation = cancellationToken.Register(() =>
            {
                try
                {
                    if (!process.HasExited)
                        process.Kill(true);
                }
                catch { }
            });

            bool exited = process.WaitForExit(timeoutMs);

            if (!exited)
            {
                try { process.Kill(); } catch { }
                throw new TimeoutException($"Python process timed out after {timeoutMs}ms");
            }

            // Wait for async reads to complete
            process.WaitForExit();

            cancellationToken.ThrowIfCancellationRequested();

            return (process.ExitCode, stdout.ToString().Trim(), stderr.ToString().Trim());
        }

        public bool Load(string weightsPath, string device = "cuda")
        {
            // Store weights path and device for use in Infer (since each subprocess is a new process)
            _weightsPath = weightsPath;
            _device = device;

            if (_isLoaded)
            {
                Log("Model already loaded");
                return true;
            }

            try
            {
                OnProgress?.Invoke("load", 0.1f, $"Loading {_modelName}...");

                string args = $"--command load --model {_modelName} --weights \"{weightsPath}\" --device {device}";
                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000); // 10 min timeout for loading

                if (exitCode == 0)
                {
                    try
                    {
                        var result = JsonSerializer.Deserialize<Dictionary<string, object>>(stdout);
                        if (result != null && result.TryGetValue("success", out var success))
                        {
                            if (success is JsonElement je && je.GetBoolean())
                            {
                                _isLoaded = true;
                                OnProgress?.Invoke("load", 1.0f, $"{_modelName} loaded successfully");
                                Log("Model loaded successfully");
                                return true;
                            }
                        }
                    }
                    catch (JsonException)
                    {
                        // stdout might have extra content, try to find JSON
                        int jsonStart = stdout.IndexOf('{');
                        if (jsonStart >= 0)
                        {
                            string jsonStr = stdout.Substring(jsonStart);
                            var result = JsonSerializer.Deserialize<Dictionary<string, object>>(jsonStr);
                            if (result != null && result.TryGetValue("success", out var success))
                            {
                                if (success is JsonElement je && je.GetBoolean())
                                {
                                    _isLoaded = true;
                                    OnProgress?.Invoke("load", 1.0f, $"{_modelName} loaded successfully");
                                    return true;
                                }
                            }
                        }
                    }
                }

                Log($"Failed to load model. Exit code: {exitCode}");
                Log($"stdout: {stdout}");
                Log($"stderr: {stderr}");
                return false;
            }
            catch (Exception ex)
            {
                Log($"Exception during load: {ex.Message}");
                return false;
            }
        }

        public void Unload()
        {
            if (!_isLoaded) return;

            try
            {
                string args = $"--command unload --model {_modelName}";
                RunPythonCommand(args, 30000);
                _isLoaded = false;
                Log("Model unloaded");
            }
            catch (Exception ex)
            {
                Log($"Error unloading: {ex.Message}");
            }
        }

        public List<MeshData> Infer(List<byte[]> imagesBytes, bool useRetrieval = true, CancellationToken cancellationToken = default)
        {
            var results = new List<MeshData>();

            if (!_isLoaded)
            {
                Log("Model not loaded");
                return results;
            }

            string inputPath = "";
            string outputPath = "";

            try
            {
                OnProgress?.Invoke("inference", 0.1f, "Preparing images...");

                // Create temp files for input/output
                inputPath = Path.GetTempFileName();
                outputPath = Path.GetTempFileName();

                // Prepare input JSON with base64-encoded images
                var images = new List<string>();
                foreach (var imgBytes in imagesBytes)
                {
                    images.Add(Convert.ToBase64String(imgBytes));
                }

                var inputData = new Dictionary<string, object>
                {
                    { "images", images }
                };

                File.WriteAllText(inputPath, JsonSerializer.Serialize(inputData));
                Log($"Wrote {images.Count} images to {inputPath}");

                OnProgress?.Invoke("inference", 0.2f, "Running inference...");

                // Run inference - pass weights and device since each subprocess is a new process
                string retrieval = useRetrieval ? "--use-retrieval" : "";
                string weightsArg = !string.IsNullOrEmpty(_weightsPath) ? $"--weights \"{_weightsPath}\"" : "";
                string deviceArg = $"--device {_device}";
                string args = $"--command infer --model {_modelName} --input \"{inputPath}\" --output \"{outputPath}\" {weightsArg} {deviceArg} {retrieval}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000, cancellationToken); // 10 min timeout

                OnProgress?.Invoke("inference", 0.8f, "Processing results...");

                if (exitCode == 0 && File.Exists(outputPath))
                {
                    string outputJson = File.ReadAllText(outputPath);
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson);

                    if (outputData.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
                    {
                        if (outputData.TryGetProperty("results", out var resultsProp))
                        {
                            foreach (var item in resultsProp.EnumerateArray())
                            {
                                var mesh = new MeshData();

                                if (item.TryGetProperty("vertices", out var vertsProp))
                                {
                                    foreach (var v in vertsProp.EnumerateArray())
                                    {
                                        var arr = v.EnumerateArray().ToArray();
                                        if (arr.Length >= 3)
                                        {
                                            mesh.Vertices.Add(new Vector3(
                                                (float)arr[0].GetDouble(),
                                                (float)arr[1].GetDouble(),
                                                (float)arr[2].GetDouble()
                                            ));
                                        }
                                    }
                                }

                                if (item.TryGetProperty("colors", out var colorsProp))
                                {
                                    foreach (var c in colorsProp.EnumerateArray())
                                    {
                                        var arr = c.EnumerateArray().ToArray();
                                        if (arr.Length >= 3)
                                        {
                                            mesh.Colors.Add(new Vector3(
                                                (float)arr[0].GetDouble(),
                                                (float)arr[1].GetDouble(),
                                                (float)arr[2].GetDouble()
                                            ));
                                        }
                                    }
                                }

                                if (item.TryGetProperty("faces", out var facesProp))
                                {
                                    foreach (var f in facesProp.EnumerateArray())
                                    {
                                        mesh.Indices.Add(f.GetInt32());
                                    }
                                }

                                results.Add(mesh);
                                Log($"Loaded mesh with {mesh.Vertices.Count} vertices");
                            }
                        }
                    }
                    else
                    {
                        string error = outputData.TryGetProperty("error", out var errProp) ? errProp.GetString() ?? "Unknown error" : "Unknown error";
                        Log($"Inference failed: {error}");
                    }
                }
                else
                {
                    Log($"Inference failed with exit code {exitCode}");
                }

                OnProgress?.Invoke("inference", 1.0f, "Complete");
            }
            catch (OperationCanceledException)
            {
                Log("Inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                Log($"Inference exception: {ex.Message}");
            }
            finally
            {
                // Clean up temp files
                try { if (!string.IsNullOrEmpty(inputPath) && File.Exists(inputPath)) File.Delete(inputPath); } catch { }
                try { if (!string.IsNullOrEmpty(outputPath) && File.Exists(outputPath)) File.Delete(outputPath); } catch { }
            }

            return results;
        }

        /// <summary>
        /// Run inference with a mesh file as input (for mesh refinement models like TripoSF).
        /// </summary>
        /// <param name="meshPath">Path to the input mesh file</param>
        /// <returns>List of output meshes</returns>
        public List<MeshData> InferMesh(string meshPath, CancellationToken cancellationToken = default)
        {
            var results = new List<MeshData>();

            if (!_isLoaded)
            {
                Log("Model not loaded");
                return results;
            }

            string outputPath = "";

            try
            {
                OnProgress?.Invoke("inference", 0.1f, "Preparing mesh input...");

                outputPath = Path.GetTempFileName();

                OnProgress?.Invoke("inference", 0.2f, "Running mesh refinement...");

                // Run inference with mesh path
                string weightsArg = !string.IsNullOrEmpty(_weightsPath) ? $"--weights \"{_weightsPath}\"" : "";
                string deviceArg = $"--device {_device}";
                string args = $"--command infer --model {_modelName} --mesh-input \"{meshPath}\" --output \"{outputPath}\" {weightsArg} {deviceArg}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000, cancellationToken); // 10 min timeout

                OnProgress?.Invoke("inference", 0.8f, "Processing results...");

                if (exitCode == 0 && File.Exists(outputPath))
                {
                    string outputJson = File.ReadAllText(outputPath);
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson);

                    if (outputData.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
                    {
                        if (outputData.TryGetProperty("results", out var resultsProp))
                        {
                            foreach (var item in resultsProp.EnumerateArray())
                            {
                                var mesh = new MeshData();

                                if (item.TryGetProperty("vertices", out var vertsProp))
                                {
                                    foreach (var v in vertsProp.EnumerateArray())
                                    {
                                        var arr = v.EnumerateArray().ToArray();
                                        if (arr.Length >= 3)
                                        {
                                            mesh.Vertices.Add(new Vector3(
                                                (float)arr[0].GetDouble(),
                                                (float)arr[1].GetDouble(),
                                                (float)arr[2].GetDouble()
                                            ));
                                        }
                                    }
                                }

                                if (item.TryGetProperty("colors", out var colorsProp))
                                {
                                    foreach (var c in colorsProp.EnumerateArray())
                                    {
                                        var arr = c.EnumerateArray().ToArray();
                                        if (arr.Length >= 3)
                                        {
                                            mesh.Colors.Add(new Vector3(
                                                (float)arr[0].GetDouble(),
                                                (float)arr[1].GetDouble(),
                                                (float)arr[2].GetDouble()
                                            ));
                                        }
                                    }
                                }

                                if (item.TryGetProperty("faces", out var facesProp))
                                {
                                    foreach (var f in facesProp.EnumerateArray())
                                    {
                                        mesh.Indices.Add(f.GetInt32());
                                    }
                                }

                                results.Add(mesh);
                                Log($"Refined mesh with {mesh.Vertices.Count} vertices, {mesh.Indices.Count / 3} triangles");
                            }
                        }
                    }
                    else
                    {
                        string error = outputData.TryGetProperty("error", out var errProp) ? errProp.GetString() ?? "Unknown error" : "Unknown error";
                        Log($"Mesh inference failed: {error}");
                    }
                }
                else
                {
                    Log($"Mesh inference failed with exit code {exitCode}");
                }

                OnProgress?.Invoke("inference", 1.0f, "Complete");
            }
            catch (OperationCanceledException)
            {
                Log("Mesh inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                Log($"Mesh inference exception: {ex.Message}");
            }
            finally
            {
                try { if (!string.IsNullOrEmpty(outputPath) && File.Exists(outputPath)) File.Delete(outputPath); } catch { }
            }

            return results;
        }

        /// <summary>
        /// Run UniRig inference with mesh data (vertices + faces) and return rigging result.
        /// </summary>
        public RigResult InferRig(MeshData mesh, int maxJoints, int maxBonesPerVertex, CancellationToken cancellationToken = default)
        {
            var result = new RigResult();

            if (!_isLoaded)
            {
                Log("Model not loaded");
                result.StatusMessage = "Model not loaded";
                return result;
            }

            string inputPath = "";
            string outputPath = "";

            try
            {
                OnProgress?.Invoke("inference", 0.1f, "Preparing UniRig mesh input...");

                inputPath = Path.GetTempFileName();
                outputPath = Path.GetTempFileName();

                // Prepare mesh payload
                var vertices = new List<float[]>(mesh.Vertices.Count);
                foreach (var v in mesh.Vertices)
                {
                    vertices.Add(new[] { v.X, v.Y, v.Z });
                }

                var inputData = new Dictionary<string, object>
                {
                    ["mesh"] = new Dictionary<string, object>
                    {
                        ["vertices"] = vertices,
                        ["faces"] = mesh.Indices.ToArray()
                    }
                };

                File.WriteAllText(inputPath, JsonSerializer.Serialize(inputData));

                OnProgress?.Invoke("inference", 0.2f, "Running UniRig inference...");

                string weightsArg = !string.IsNullOrEmpty(_weightsPath) ? $"--weights \"{_weightsPath}\"" : "";
                string deviceArg = $"--device {_device}";
                string args = $"--command infer --model {_modelName} --input \"{inputPath}\" --output \"{outputPath}\" {weightsArg} {deviceArg} --max-joints {maxJoints} --max-bones {maxBonesPerVertex}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000, cancellationToken);

                OnProgress?.Invoke("inference", 0.8f, "Processing UniRig results...");

                if (exitCode == 0 && File.Exists(outputPath))
                {
                    string outputJson = File.ReadAllText(outputPath);
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson);

                    if (outputData.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
                    {
                        if (outputData.TryGetProperty("rig_result", out var rigProp))
                        {
                            if (rigProp.TryGetProperty("joint_positions", out var jointsProp))
                            {
                                var joints = new List<Vector3>();
                                foreach (var j in jointsProp.EnumerateArray())
                                {
                                    var arr = j.EnumerateArray().ToArray();
                                    if (arr.Length >= 3)
                                    {
                                        joints.Add(new Vector3(
                                            (float)arr[0].GetDouble(),
                                            (float)arr[1].GetDouble(),
                                            (float)arr[2].GetDouble()
                                        ));
                                    }
                                }
                                result.JointPositions = joints.ToArray();
                            }

                            if (rigProp.TryGetProperty("parent_indices", out var parentsProp))
                            {
                                var parents = new List<int>();
                                foreach (var p in parentsProp.EnumerateArray())
                                {
                                    parents.Add(p.GetInt32());
                                }
                                result.ParentIndices = parents.ToArray();
                            }

                            if (rigProp.TryGetProperty("joint_names", out var namesProp))
                            {
                                var names = new List<string>();
                                foreach (var n in namesProp.EnumerateArray())
                                {
                                    names.Add(n.GetString() ?? "Joint");
                                }
                                result.JointNames = names.ToArray();
                            }

                            if (rigProp.TryGetProperty("skinning_weights", out var weightsProp))
                            {
                                int vertexCount = weightsProp.GetArrayLength();
                                int jointCount = result.JointPositions?.Length ?? 0;
                                var weights = new float[vertexCount, jointCount];

                                int vIdx = 0;
                                foreach (var row in weightsProp.EnumerateArray())
                                {
                                    int jIdx = 0;
                                    foreach (var val in row.EnumerateArray())
                                    {
                                        if (jIdx < jointCount)
                                            weights[vIdx, jIdx] = (float)val.GetDouble();
                                        jIdx++;
                                    }
                                    vIdx++;
                                }
                                result.SkinningWeights = weights;
                            }

                            result.Success = true;
                            result.StatusMessage = "UniRig rigging complete";
                        }
                    }
                    else
                    {
                        string error = outputData.TryGetProperty("error", out var errProp)
                            ? errProp.GetString() ?? "Unknown error"
                            : "Unknown error";
                        result.StatusMessage = error;
                        Log($"UniRig inference failed: {error}");
                    }
                }
                else
                {
                    result.StatusMessage = $"UniRig inference failed with exit code {exitCode}";
                    Log(result.StatusMessage);
                }

                OnProgress?.Invoke("inference", 1.0f, "Complete");
            }
            catch (OperationCanceledException)
            {
                result.StatusMessage = "UniRig inference cancelled";
                Log("UniRig inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                result.StatusMessage = ex.Message;
                Log($"UniRig inference exception: {ex.Message}");
            }
            finally
            {
                try { if (!string.IsNullOrEmpty(inputPath) && File.Exists(inputPath)) File.Delete(inputPath); } catch { }
                try { if (!string.IsNullOrEmpty(outputPath) && File.Exists(outputPath)) File.Delete(outputPath); } catch { }
            }

            return result;
        }

        /// <summary>
        /// Run UniRig inference with a mesh file path (for GLB/GLTF etc.) and return rigging result.
        /// </summary>
        public RigResult InferRigFromFile(string meshPath, int maxJoints, int maxBonesPerVertex, CancellationToken cancellationToken = default)
        {
            var result = new RigResult();

            if (!_isLoaded)
            {
                Log("Model not loaded");
                result.StatusMessage = "Model not loaded";
                return result;
            }

            if (!File.Exists(meshPath))
            {
                result.StatusMessage = $"Mesh file not found: {meshPath}";
                return result;
            }

            string outputPath = "";

            try
            {
                OnProgress?.Invoke("inference", 0.1f, "Preparing UniRig mesh file input...");
                outputPath = Path.GetTempFileName();

                string weightsArg = !string.IsNullOrEmpty(_weightsPath) ? $"--weights \"{_weightsPath}\"" : "";
                string deviceArg = $"--device {_device}";
                string args = $"--command infer --model {_modelName} --mesh-input \"{meshPath}\" --output \"{outputPath}\" {weightsArg} {deviceArg} --max-joints {maxJoints} --max-bones {maxBonesPerVertex}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000, cancellationToken);

                OnProgress?.Invoke("inference", 0.8f, "Processing UniRig results...");

                if (exitCode == 0 && File.Exists(outputPath))
                {
                    string outputJson = File.ReadAllText(outputPath);
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson);

                    if (outputData.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
                    {
                        if (outputData.TryGetProperty("rig_result", out var rigProp))
                        {
                            if (rigProp.TryGetProperty("joint_positions", out var jointsProp))
                            {
                                var joints = new List<Vector3>();
                                foreach (var j in jointsProp.EnumerateArray())
                                {
                                    var arr = j.EnumerateArray().ToArray();
                                    if (arr.Length >= 3)
                                    {
                                        joints.Add(new Vector3(
                                            (float)arr[0].GetDouble(),
                                            (float)arr[1].GetDouble(),
                                            (float)arr[2].GetDouble()
                                        ));
                                    }
                                }
                                result.JointPositions = joints.ToArray();
                            }

                            if (rigProp.TryGetProperty("parent_indices", out var parentsProp))
                            {
                                var parents = new List<int>();
                                foreach (var p in parentsProp.EnumerateArray())
                                {
                                    parents.Add(p.GetInt32());
                                }
                                result.ParentIndices = parents.ToArray();
                            }

                            if (rigProp.TryGetProperty("joint_names", out var namesProp))
                            {
                                var names = new List<string>();
                                foreach (var n in namesProp.EnumerateArray())
                                {
                                    names.Add(n.GetString() ?? "Joint");
                                }
                                result.JointNames = names.ToArray();
                            }

                            if (rigProp.TryGetProperty("skinning_weights", out var weightsProp))
                            {
                                int vertexCount = weightsProp.GetArrayLength();
                                int jointCount = result.JointPositions?.Length ?? 0;
                                var weights = new float[vertexCount, jointCount];

                                int vIdx = 0;
                                foreach (var row in weightsProp.EnumerateArray())
                                {
                                    int jIdx = 0;
                                    foreach (var val in row.EnumerateArray())
                                    {
                                        if (jIdx < jointCount)
                                            weights[vIdx, jIdx] = (float)val.GetDouble();
                                        jIdx++;
                                    }
                                    vIdx++;
                                }
                                result.SkinningWeights = weights;
                            }

                            result.Success = true;
                            result.StatusMessage = "UniRig rigging complete";
                        }
                    }
                    else
                    {
                        string error = outputData.TryGetProperty("error", out var errProp)
                            ? errProp.GetString() ?? "Unknown error"
                            : "Unknown error";
                        result.StatusMessage = error;
                        Log($"UniRig inference failed: {error}");
                    }
                }
                else
                {
                    result.StatusMessage = $"UniRig inference failed with exit code {exitCode}";
                    Log(result.StatusMessage);
                }

                OnProgress?.Invoke("inference", 1.0f, "Complete");
            }
            catch (OperationCanceledException)
            {
                result.StatusMessage = "UniRig inference cancelled";
                Log("UniRig inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                result.StatusMessage = ex.Message;
                Log($"UniRig inference exception: {ex.Message}");
            }
            finally
            {
                try { if (!string.IsNullOrEmpty(outputPath) && File.Exists(outputPath)) File.Delete(outputPath); } catch { }
            }

            return result;
        }

        public void Dispose()
        {
            if (_disposed) return;

            Unload();

            lock (_processLock)
            {
                if (_persistentProcess != null && !_persistentProcess.HasExited)
                {
                    try { _persistentProcess.Kill(); } catch { }
                    _persistentProcess.Dispose();
                    _persistentProcess = null;
                }
            }

            _disposed = true;
        }
    }
}
