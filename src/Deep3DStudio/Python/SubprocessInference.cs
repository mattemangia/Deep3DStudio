using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;
using Deep3DStudio.IO;

namespace Deep3DStudio.Python
{
    /// <summary>
    /// Subprocess-based inference that runs Python in a completely isolated process.
    /// This avoids all pythonnet memory corruption issues by using process isolation.
    /// </summary>
    public class SubprocessInference : IDisposable
    {
        private static readonly JsonSerializerOptions JsonOptions = new()
        {
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
        };
        private static readonly string[] ConfidenceJsonKeys = { "confidence", "confidences", "confidence_scores" };
        private readonly string _pythonPath;
        private readonly string _scriptPath;
        private readonly string _modelName;
        private string? _weightsPath;
        private string _device = "auto";
        private bool _isLoaded = false;
        public bool LastOpenMpShmError { get; private set; }
        public string? LastErrorMessage { get; private set; }
        private bool _disposed = false;
        private Process? _persistentProcess;
        private readonly object _processLock = new object();
        private readonly object _macRuntimeLock = new object();
        private readonly List<MacRuntimeProfile> _macRuntimeProfiles = new();
        private bool _macRuntimeProfilesLoaded = false;
        private int _activeMacRuntimeProfileIndex = -1;
        private bool _macHealthcheckAttempted = false;

        public event Action<string>? OnLog;
        public event Action<string, float, string>? OnProgress;

        public bool IsLoaded => _isLoaded;

        private sealed class MacRuntimeConfig
        {
            [JsonPropertyName("selected_profile_id")]
            public string? SelectedProfileId { get; set; }

            [JsonPropertyName("profiles")]
            public List<MacRuntimeConfigProfile>? Profiles { get; set; }
        }

        private sealed class MacRuntimeConfigProfile
        {
            [JsonPropertyName("id")]
            public string? Id { get; set; }

            [JsonPropertyName("description")]
            public string? Description { get; set; }

            [JsonPropertyName("valid")]
            public bool Valid { get; set; } = true;

            [JsonPropertyName("libomp_path")]
            public string? LibompPath { get; set; }

            [JsonPropertyName("env")]
            public Dictionary<string, string>? Env { get; set; }
        }

        private sealed class MacRuntimeProfile
        {
            public string Id { get; init; } = "system-default";
            public string Description { get; init; } = "System default runtime";
            public string? LibompPath { get; init; }
            public Dictionary<string, string> Env { get; init; } = new(StringComparer.OrdinalIgnoreCase);
        }

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
                    TemporaryFileManager.RegisterFile(tempPath);
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
                                TemporaryFileManager.RegisterFile(tempPath);
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
            LastOpenMpShmError = false;
            LastErrorMessage = null;

            bool macProfileSensitive = OperatingSystem.IsMacOS() && IsProfileSensitiveCommand(arguments);
            var profiles = GetMacProfilesForAttempt(macProfileSensitive);

            (int exitCode, string stdout, string stderr) last = (1, "", "No attempt executed");

            foreach (var profile in profiles)
            {
                var firstTry = RunPythonCommandInternal(
                    arguments,
                    timeoutMs,
                    cancellationToken,
                    safeOpenMpMode: false,
                    macProfile: profile);

                if (firstTry.exitCode == 0)
                {
                    SetActiveMacProfile(profile);
                    LastOpenMpShmError = false;
                    LastErrorMessage = null;
                    return firstTry;
                }

                last = firstTry;

                bool shouldRetrySafe = ShouldRetryWithSafeOpenMp(firstTry.exitCode, firstTry.stderr)
                    || (OperatingSystem.IsMacOS() && IsMacRuntimeBootstrapError(firstTry.exitCode, firstTry.stderr));

                if (shouldRetrySafe)
                {
                    Log("Detected macOS runtime/bootstrap failure. Retrying with safe OpenMP settings...");
                    var retry = RunPythonCommandInternal(
                        arguments,
                        timeoutMs,
                        cancellationToken,
                        safeOpenMpMode: true,
                        macProfile: profile);

                    if (retry.exitCode == 0)
                    {
                        SetActiveMacProfile(profile);
                        LastOpenMpShmError = false;
                        LastErrorMessage = null;
                        return retry;
                    }

                    last = retry;
                }

                // Only cycle mac profiles for bootstrap-sensitive commands.
                if (!macProfileSensitive)
                    break;
            }

            LastOpenMpShmError = IsOpenMpShmError(last.exitCode, last.stderr);
            LastErrorMessage = BuildErrorMessage(last.exitCode, last.stderr);
            return last;
        }

        private static bool ShouldRetryWithSafeOpenMp(int exitCode, string stderr)
        {
            if (!OperatingSystem.IsMacOS())
                return false;

            return IsOpenMpShmError(exitCode, stderr);
        }

        private static bool IsOpenMpShmError(int exitCode, string stderr)
        {
            if (exitCode != 134 || string.IsNullOrWhiteSpace(stderr))
                return false;

            return stderr.Contains("OMP: Error #179", StringComparison.OrdinalIgnoreCase)
                || stderr.Contains("Can't open SHM2", StringComparison.OrdinalIgnoreCase)
                || stderr.Contains("OMP: System error #2", StringComparison.OrdinalIgnoreCase);
        }

        private static string BuildErrorMessage(int exitCode, string stderr)
        {
            if (IsOpenMpShmError(exitCode, stderr))
            {
                return "OpenMP shared-memory initialization failed on macOS (OMP #179 / SHM2).";
            }
            if (IsTorchvisionNmsDuplicateRegistrationError(stderr))
            {
                return "torchvision NMS operator registration conflicted on macOS runtime startup.";
            }
            return $"Python subprocess failed with exit code {exitCode}.";
        }

        private string? GetPythonHomeDirectory()
        {
            try
            {
                var pythonDir = Path.GetDirectoryName(_pythonPath);
                if (string.IsNullOrWhiteSpace(pythonDir))
                    return null;

                var parent = Directory.GetParent(pythonDir);
                return parent?.FullName;
            }
            catch
            {
                return null;
            }
        }

        private static Dictionary<string, string> BuildDefaultMacRuntimeEnv()
        {
            return new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                ["KMP_USE_SHM"] = "0",
                ["KMP_DUPLICATE_LIB_OK"] = "TRUE",
                ["KMP_INIT_AT_FORK"] = "FALSE",
                ["KMP_AFFINITY"] = "disabled",
                ["OMP_NUM_THREADS"] = "1",
                ["OMP_WAIT_POLICY"] = "PASSIVE",
                ["MKL_NUM_THREADS"] = "1",
                ["OPENBLAS_NUM_THREADS"] = "1",
                ["NUMEXPR_NUM_THREADS"] = "1",
                ["VECLIB_MAXIMUM_THREADS"] = "1",
                ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1",
                ["DEEP3D_TORCHVISION_NMS_STUB"] = "auto",
            };
        }

        private static void TryAddMacProfile(List<MacRuntimeProfile> profiles, string id, string description, string? libompPath, Dictionary<string, string>? env = null)
        {
            if (!string.IsNullOrWhiteSpace(libompPath) && !File.Exists(libompPath))
                return;

            if (profiles.Any(p =>
                    p.Id.Equals(id, StringComparison.OrdinalIgnoreCase) &&
                    string.Equals(p.LibompPath, libompPath, StringComparison.Ordinal)))
            {
                return;
            }

            var profileEnv = BuildDefaultMacRuntimeEnv();
            if (env != null)
            {
                foreach (var kv in env)
                {
                    if (!string.IsNullOrWhiteSpace(kv.Key) && !string.IsNullOrWhiteSpace(kv.Value))
                        profileEnv[kv.Key] = kv.Value;
                }
            }

            profiles.Add(new MacRuntimeProfile
            {
                Id = id,
                Description = description,
                LibompPath = libompPath,
                Env = profileEnv
            });
        }

        private void LoadMacRuntimeProfilesIfNeeded()
        {
            if (!OperatingSystem.IsMacOS())
                return;

            lock (_macRuntimeLock)
            {
                if (_macRuntimeProfilesLoaded)
                    return;

                _macRuntimeProfiles.Clear();
                string? selectedId = null;

                try
                {
                    var pythonHome = GetPythonHomeDirectory();
                    if (!string.IsNullOrWhiteSpace(pythonHome))
                    {
                        string configPath = Path.Combine(pythonHome, "runtime_mac.json");
                        if (File.Exists(configPath))
                        {
                            var json = File.ReadAllText(configPath);
                            var config = JsonSerializer.Deserialize<MacRuntimeConfig>(json, JsonOptions);
                            if (config?.Profiles != null)
                            {
                                selectedId = config.SelectedProfileId;
                                foreach (var p in config.Profiles)
                                {
                                    if (p == null || !p.Valid || string.IsNullOrWhiteSpace(p.Id))
                                        continue;

                                    TryAddMacProfile(
                                        _macRuntimeProfiles,
                                        p.Id!,
                                        p.Description ?? p.Id!,
                                        p.LibompPath,
                                        p.Env);
                                }
                            }
                            Log($"Loaded mac runtime profile config: {configPath}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"Failed to parse runtime_mac.json: {ex.Message}");
                }

                if (_macRuntimeProfiles.Count == 0)
                {
                    var pythonHome = GetPythonHomeDirectory();
                    if (!string.IsNullOrWhiteSpace(pythonHome))
                    {
                        foreach (var pyVersion in new[] { "3.10", "3.11", "3.9" })
                        {
                            var bundledLibomp = Path.Combine(pythonHome, "lib", $"python{pyVersion}", "site-packages", "torch", "lib", "libomp.dylib");
                            TryAddMacProfile(_macRuntimeProfiles, "bundled-torch-libomp", "Bundled torch libomp", bundledLibomp);
                        }
                    }

                    TryAddMacProfile(_macRuntimeProfiles, "homebrew-arm64-libomp", "Homebrew arm64 libomp", "/opt/homebrew/opt/libomp/lib/libomp.dylib");
                    TryAddMacProfile(_macRuntimeProfiles, "homebrew-x64-libomp", "Homebrew x64 libomp", "/usr/local/opt/libomp/lib/libomp.dylib");
                    TryAddMacProfile(_macRuntimeProfiles, "system-default", "No libomp override", null);
                }

                _activeMacRuntimeProfileIndex = 0;
                if (!string.IsNullOrWhiteSpace(selectedId))
                {
                    int idx = _macRuntimeProfiles.FindIndex(p => p.Id.Equals(selectedId, StringComparison.OrdinalIgnoreCase));
                    if (idx >= 0)
                        _activeMacRuntimeProfileIndex = idx;
                }

                _macRuntimeProfilesLoaded = true;
                if (_macRuntimeProfiles.Count > 0)
                {
                    Log($"macOS runtime profiles loaded: {_macRuntimeProfiles.Count}, active={_macRuntimeProfiles[_activeMacRuntimeProfileIndex].Id}");
                }
            }
        }

        private List<MacRuntimeProfile?> GetMacProfilesForAttempt(bool profileSensitiveCommand)
        {
            if (!OperatingSystem.IsMacOS())
                return new List<MacRuntimeProfile?> { null };

            LoadMacRuntimeProfilesIfNeeded();

            lock (_macRuntimeLock)
            {
                if (_macRuntimeProfiles.Count == 0)
                    return new List<MacRuntimeProfile?> { null };

                var profiles = new List<MacRuntimeProfile?>();
                int active = Math.Clamp(_activeMacRuntimeProfileIndex, 0, _macRuntimeProfiles.Count - 1);
                profiles.Add(_macRuntimeProfiles[active]);

                if (profileSensitiveCommand)
                {
                    for (int i = 0; i < _macRuntimeProfiles.Count; i++)
                    {
                        if (i == active)
                            continue;
                        profiles.Add(_macRuntimeProfiles[i]);
                    }
                }

                return profiles;
            }
        }

        private void SetActiveMacProfile(MacRuntimeProfile? profile)
        {
            if (!OperatingSystem.IsMacOS() || profile == null)
                return;

            lock (_macRuntimeLock)
            {
                int idx = _macRuntimeProfiles.FindIndex(p => p.Id.Equals(profile.Id, StringComparison.OrdinalIgnoreCase));
                if (idx >= 0)
                    _activeMacRuntimeProfileIndex = idx;
            }
        }

        private static bool IsProfileSensitiveCommand(string arguments)
        {
            return arguments.Contains("--command load", StringComparison.Ordinal)
                || arguments.Contains("--command ping", StringComparison.Ordinal)
                || arguments.Contains("--command healthcheck", StringComparison.Ordinal);
        }

        private static bool IsMacRuntimeBootstrapError(int exitCode, string stderr)
        {
            if (IsOpenMpShmError(exitCode, stderr))
                return true;

            if (IsTorchvisionNmsDuplicateRegistrationError(stderr))
                return true;

            if (string.IsNullOrWhiteSpace(stderr))
                return false;

            return stderr.Contains("libomp", StringComparison.OrdinalIgnoreCase)
                || stderr.Contains("Library not loaded", StringComparison.OrdinalIgnoreCase)
                || stderr.Contains("dlopen(", StringComparison.OrdinalIgnoreCase)
                || stderr.Contains("image not found", StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsTorchvisionNmsDuplicateRegistrationError(string stderr)
        {
            if (string.IsNullOrWhiteSpace(stderr))
                return false;

            return stderr.Contains("torchvision::nms", StringComparison.OrdinalIgnoreCase)
                && (stderr.Contains("register an operator", StringComparison.OrdinalIgnoreCase)
                    || stderr.Contains("duplicate registration", StringComparison.OrdinalIgnoreCase)
                    || stderr.Contains("already registered", StringComparison.OrdinalIgnoreCase));
        }

        private static void SetEnvironmentIfMissing(ProcessStartInfo psi, string key, string value)
        {
            if (!psi.Environment.ContainsKey(key))
                psi.Environment[key] = value;
        }

        private static string PrependPathEntry(string? existing, string prefix)
        {
            if (string.IsNullOrWhiteSpace(existing))
                return prefix;

            var segments = existing.Split(':', StringSplitOptions.RemoveEmptyEntries);
            if (segments.Any(s => s.Equals(prefix, StringComparison.Ordinal)))
                return existing;

            return $"{prefix}:{existing}";
        }

        private static void ApplyMacRuntimeProfile(ProcessStartInfo psi, MacRuntimeProfile? profile)
        {
            foreach (var kv in BuildDefaultMacRuntimeEnv())
            {
                SetEnvironmentIfMissing(psi, kv.Key, kv.Value);
            }

            if (profile == null)
                return;

            foreach (var kv in profile.Env)
            {
                psi.Environment[kv.Key] = kv.Value;
            }

            if (!string.IsNullOrWhiteSpace(profile.LibompPath) && File.Exists(profile.LibompPath))
            {
                var libompDir = Path.GetDirectoryName(profile.LibompPath);
                if (!string.IsNullOrWhiteSpace(libompDir))
                {
                    var dyld = PrependPathEntry(psi.Environment.TryGetValue("DYLD_LIBRARY_PATH", out var currentDyld) ? currentDyld : null, libompDir);
                    var dyldFallback = PrependPathEntry(psi.Environment.TryGetValue("DYLD_FALLBACK_LIBRARY_PATH", out var currentFallback) ? currentFallback : null, libompDir);
                    psi.Environment["DYLD_LIBRARY_PATH"] = dyld;
                    psi.Environment["DYLD_FALLBACK_LIBRARY_PATH"] = dyldFallback;
                }
            }

            psi.Environment["DEEP3D_MAC_RUNTIME_PROFILE"] = profile.Id;
        }

        private static void ApplySafeOpenMpSettings(ProcessStartInfo psi)
        {
            psi.Environment["KMP_USE_SHM"] = "0";
            psi.Environment["KMP_DUPLICATE_LIB_OK"] = "TRUE";
            psi.Environment["KMP_AFFINITY"] = "disabled";
            psi.Environment["OMP_NUM_THREADS"] = "1";
            psi.Environment["MKL_NUM_THREADS"] = "1";
            psi.Environment["OPENBLAS_NUM_THREADS"] = "1";
            psi.Environment["NUMEXPR_NUM_THREADS"] = "1";
            psi.Environment["VECLIB_MAXIMUM_THREADS"] = "1";
            psi.Environment["OMP_WAIT_POLICY"] = "PASSIVE";
            psi.Environment["KMP_INIT_AT_FORK"] = "FALSE";
        }

        private (int exitCode, string stdout, string stderr) RunPythonCommandInternal(
            string arguments,
            int timeoutMs = 300000,
            CancellationToken cancellationToken = default,
            bool safeOpenMpMode = false,
            MacRuntimeProfile? macProfile = null)
        {
            Log($"Running: {_pythonPath} {_scriptPath} {arguments}");
            if (safeOpenMpMode)
            {
                Log("Applying safe OpenMP environment for this run");
            }
            if (OperatingSystem.IsMacOS() && macProfile != null)
            {
                Log($"Using mac runtime profile: {macProfile.Id}");
            }

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

            if (_modelName.Equals("triposf", StringComparison.OrdinalIgnoreCase))
            {
                psi.Environment["DEEP3D_TRIPOSF_SKIP_MC"] = "1";
            }
            if (OperatingSystem.IsMacOS())
            {
                ApplyMacRuntimeProfile(psi, macProfile);
            }
            if (safeOpenMpMode)
            {
                ApplySafeOpenMpSettings(psi);
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

        private static void ParseConfidenceValues(JsonElement item, MeshData mesh)
        {
            foreach (var key in ConfidenceJsonKeys)
            {
                if (!item.TryGetProperty(key, out var confidenceProp) || confidenceProp.ValueKind != JsonValueKind.Array)
                    continue;

                foreach (var c in confidenceProp.EnumerateArray())
                {
                    switch (c.ValueKind)
                    {
                        case JsonValueKind.Number:
                            mesh.Confidence.Add((float)c.GetDouble());
                            break;
                        case JsonValueKind.Array:
                        {
                            foreach (var sub in c.EnumerateArray())
                            {
                                if (sub.ValueKind == JsonValueKind.Number)
                                {
                                    mesh.Confidence.Add((float)sub.GetDouble());
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }

                return;
            }
        }

        public bool Load(string weightsPath, string device = "auto")
        {
            // Store weights path and device for use in Infer (since each subprocess is a new process)
            _weightsPath = weightsPath;
            _device = device;

            if (_isLoaded)
            {
                Log("Model already loaded");
                return true;
            }

            if (OperatingSystem.IsMacOS())
            {
                TryRunMacHealthcheck();
            }

            // Skip launching a separate subprocess to load the model.
            // Each Infer() call spawns a new Python process that loads the model
            // via _ensure_loaded() anyway, so a separate Load subprocess is wasted
            // work — the model is discarded when that process exits.
            OnProgress?.Invoke("load", 0.5f, $"Configuring {_modelName}...");

            // Validate weights path if it's a local path (not a HuggingFace model ID)
            if (!string.IsNullOrEmpty(weightsPath) &&
                !weightsPath.Contains('/') &&
                !File.Exists(weightsPath) && !Directory.Exists(weightsPath))
            {
                Log($"Warning: Weights path not found: {weightsPath}");
            }

            _isLoaded = true;
            OnProgress?.Invoke("load", 1.0f, $"{_modelName} configured");
            Log($"Model configured (will load on first inference)");
            return true;
        }

        private void TryRunMacHealthcheck()
        {
            if (!OperatingSystem.IsMacOS() || _macHealthcheckAttempted)
                return;

            _macHealthcheckAttempted = true;

            try
            {
                string deviceArg = string.IsNullOrWhiteSpace(_device) ? "auto" : _device;
                string modelArg = string.IsNullOrWhiteSpace(_modelName) ? "dust3r" : _modelName;
                string args = $"--command healthcheck --model {modelArg} --device {deviceArg}";
                var (exitCode, stdout, stderr) = RunPythonCommand(args, 120000);
                if (exitCode == 0)
                {
                    Log("macOS runtime healthcheck succeeded.");
                }
                else
                {
                    Log($"macOS runtime healthcheck failed. exit={exitCode}");
                    if (!string.IsNullOrWhiteSpace(stderr))
                        Log(stderr);
                }
            }
            catch (Exception ex)
            {
                Log($"macOS runtime healthcheck exception: {ex.Message}");
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
                TemporaryFileManager.RegisterFile(inputPath);
                TemporaryFileManager.RegisterFile(outputPath);

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
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson, JsonOptions);

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

                                ParseConfidenceValues(item, mesh);

                                if (item.TryGetProperty("faces", out var facesProp))
                                {
                                    foreach (var f in facesProp.EnumerateArray())
                                    {
                                        mesh.Indices.Add(f.GetInt32());
                                    }
                                }

                                if (item.TryGetProperty("pose", out var poseProp))
                                {
                                    var rows = poseProp.EnumerateArray().ToArray();
                                    if (rows.Length == 4)
                                    {
                                        float[] m = new float[16];
                                        for(int r=0; r<4; r++)
                                        {
                                            var cols = rows[r].EnumerateArray().ToArray();
                                            if (cols.Length >= 4)
                                            {
                                                for(int c=0; c<4; c++)
                                                {
                                                    m[r*4 + c] = (float)cols[c].GetDouble();
                                                }
                                            }
                                        }
                                        // OpenTK Matrix4 constructor is row-major: M11, M12, M13, M14...
                                        mesh.Pose = new Matrix4(
                                            m[0], m[1], m[2], m[3],
                                            m[4], m[5], m[6], m[7],
                                            m[8], m[9], m[10], m[11],
                                            m[12], m[13], m[14], m[15]
                                        );
                                    }
                                }

                                // Parse multi-view data (Wonder3D normal maps + color)
                                if (item.TryGetProperty("views", out var viewsProp))
                                {
                                    var viewsList = new List<MultiViewData>();
                                    foreach (var viewElem in viewsProp.EnumerateArray())
                                    {
                                        var view = new MultiViewData();
                                        if (viewElem.TryGetProperty("width", out var wProp))
                                            view.Width = wProp.GetInt32();
                                        if (viewElem.TryGetProperty("height", out var hProp))
                                            view.Height = hProp.GetInt32();

                                        if (viewElem.TryGetProperty("color", out var colorArr))
                                            view.Colors = colorArr.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                                        if (viewElem.TryGetProperty("normal", out var normalArr))
                                            view.Normals = normalArr.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();

                                        if (viewElem.TryGetProperty("rotation", out var rotArr))
                                        {
                                            var r = rotArr.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();
                                            if (r.Length >= 9)
                                            {
                                                view.Rotation = new Matrix3(
                                                    r[0], r[1], r[2],
                                                    r[3], r[4], r[5],
                                                    r[6], r[7], r[8]);
                                            }
                                        }

                                        viewsList.Add(view);
                                    }

                                    if (viewsList.Count > 0)
                                    {
                                        mesh.MultiViews = viewsList;
                                        Log($"Parsed {viewsList.Count} multi-view images (normals: {viewsList.Any(v => v.Normals.Length > 0)})");
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
                TemporaryFileManager.RegisterFile(outputPath);

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
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson, JsonOptions);

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

                                ParseConfidenceValues(item, mesh);

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
                TemporaryFileManager.RegisterFile(inputPath);
                TemporaryFileManager.RegisterFile(outputPath);

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
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson, JsonOptions);

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
                TemporaryFileManager.RegisterFile(outputPath);

                string weightsArg = !string.IsNullOrEmpty(_weightsPath) ? $"--weights \"{_weightsPath}\"" : "";
                string deviceArg = $"--device {_device}";
                string args = $"--command infer --model {_modelName} --mesh-input \"{meshPath}\" --output \"{outputPath}\" {weightsArg} {deviceArg} --max-joints {maxJoints} --max-bones {maxBonesPerVertex}";

                var (exitCode, stdout, stderr) = RunPythonCommand(args, 600000, cancellationToken);

                OnProgress?.Invoke("inference", 0.8f, "Processing UniRig results...");

                if (exitCode == 0 && File.Exists(outputPath))
                {
                    string outputJson = File.ReadAllText(outputPath);
                    var outputData = JsonSerializer.Deserialize<JsonElement>(outputJson, JsonOptions);

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
