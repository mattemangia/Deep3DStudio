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
            var settings = IniSettings.Instance;
            string condaEnv = settings.CondaEnvironment;

            // Try common locations
            string[] paths = new[]
            {
                // Windows conda
                $@"C:\Users\{Environment.UserName}\miniconda3\envs\{condaEnv}\python.exe",
                $@"C:\Users\{Environment.UserName}\anaconda3\envs\{condaEnv}\python.exe",
                $@"C:\ProgramData\miniconda3\envs\{condaEnv}\python.exe",
                $@"C:\ProgramData\anaconda3\envs\{condaEnv}\python.exe",
                // Linux/macOS conda
                $"/home/{Environment.UserName}/miniconda3/envs/{condaEnv}/bin/python",
                $"/home/{Environment.UserName}/anaconda3/envs/{condaEnv}/bin/python",
                $"/opt/conda/envs/{condaEnv}/bin/python",
                // Fallback to system python
                "python3",
                "python"
            };

            foreach (var path in paths)
            {
                if (File.Exists(path))
                {
                    return path;
                }
            }

            // Try using 'which' or 'where' to find python
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = Environment.OSVersion.Platform == PlatformID.Win32NT ? "where" : "which",
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
                        return output;
                    }
                }
            }
            catch { }

            return "python"; // Last resort
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

        private (int exitCode, string stdout, string stderr) RunPythonCommand(string arguments, int timeoutMs = 300000)
        {
            Log($"Running: {_pythonPath} {_scriptPath} {arguments}");

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

            // Set up environment
            var settings = IniSettings.Instance;
            if (!string.IsNullOrEmpty(settings.CondaEnvironment))
            {
                // Add conda environment to PATH
                string condaBase = Path.GetDirectoryName(Path.GetDirectoryName(_pythonPath)) ?? "";
                if (!string.IsNullOrEmpty(condaBase))
                {
                    string currentPath = Environment.GetEnvironmentVariable("PATH") ?? "";
                    psi.Environment["PATH"] = $"{condaBase};{Path.Combine(condaBase, "Scripts")};{Path.Combine(condaBase, "Library", "bin")};{currentPath}";
                }
            }

            using var process = new Process { StartInfo = psi };
            var stdout = new StringBuilder();
            var stderr = new StringBuilder();

            process.OutputDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    stdout.AppendLine(e.Data);
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

            bool exited = process.WaitForExit(timeoutMs);

            if (!exited)
            {
                try { process.Kill(); } catch { }
                throw new TimeoutException($"Python process timed out after {timeoutMs}ms");
            }

            // Wait for async reads to complete
            process.WaitForExit();

            return (process.ExitCode, stdout.ToString().Trim(), stderr.ToString().Trim());
        }

        public bool Load(string weightsPath, string device = "cuda")
        {
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

        public List<MeshData> Infer(List<byte[]> imagesBytes, bool useRetrieval = true)
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

                // Run inference
                string retrieval = useRetrieval ? "--use-retrieval" : "";
                string args = $"--command infer --model {_modelName} --input \"{inputPath}\" --output \"{outputPath}\" {retrieval}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000); // 10 min timeout

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
