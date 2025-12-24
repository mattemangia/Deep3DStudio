using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using Python.Runtime;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.Python
{
    public class PythonService : IDisposable
    {
        private static PythonService? _instance;
        private static readonly object _lock = new object();
        private IntPtr _threadState;
        private bool _isInitialized = false;

        public bool IsInitialized => _isInitialized;
        public string InitializationError { get; private set; } = "";

        // Event to capture Python stdout/stderr
        public event Action<string>? OnLogOutput;

        public static PythonService Instance
        {
            get
            {
                lock (_lock)
                {
                    if (_instance == null)
                    {
                        _instance = new PythonService();
                    }
                    return _instance;
                }
            }
        }

        private PythonService()
        {
            // Constructor logic
        }

        public void Initialize()
        {
            if (_isInitialized) return;

            try
            {
                // Ensure environment is ready
                EnsurePythonEnvironment();

                string pythonHome = GetPythonHome();
                string pythonDll = GetPythonDllPath(pythonHome);

                if (!File.Exists(pythonDll))
                {
                    string msg = $"Python DLL not found at {pythonDll}";
                    InitializationError = msg;
                    Log($"Warning: {msg}");
                    Log($"Python features will be disabled. Please run setup_deployment.py to install Python environment.");
                    // Don't throw - allow app to continue without Python
                    return;
                }

                // CRITICAL: Clear any existing Python environment variables FIRST
                // This prevents system Python installations (like FEFLOW, Anaconda) from interfering
                ClearSystemPythonEnvironment();

                Runtime.PythonDLL = pythonDll;

                // Construct paths for the embedded environment (platform-specific)
                string libDir;
                string sitePackages;
                string dllsDir;

                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Windows: pythonHome/Lib and pythonHome/Lib/site-packages
                    libDir = Path.Combine(pythonHome, "Lib");
                    sitePackages = Path.Combine(libDir, "site-packages");
                    dllsDir = Path.Combine(pythonHome, "DLLs");
                }
                else
                {
                    // Linux/macOS: pythonHome/lib/python3.10 and pythonHome/lib/python3.10/site-packages
                    string libBase = Path.Combine(pythonHome, "lib");

                    // Find the python3.x directory dynamically
                    string? pythonLibDir = null;
                    if (Directory.Exists(libBase))
                    {
                        foreach (string dir in Directory.GetDirectories(libBase))
                        {
                            string dirName = Path.GetFileName(dir);
                            if (dirName.StartsWith("python3"))
                            {
                                pythonLibDir = dir;
                                break;
                            }
                        }
                    }

                    if (pythonLibDir != null)
                    {
                        libDir = pythonLibDir;
                        sitePackages = Path.Combine(pythonLibDir, "site-packages");
                        dllsDir = Path.Combine(pythonLibDir, "lib-dynload");
                    }
                    else
                    {
                        // Fallback to expected structure
                        libDir = Path.Combine(libBase, "python3.10");
                        sitePackages = Path.Combine(libDir, "site-packages");
                        dllsDir = Path.Combine(libDir, "lib-dynload");
                    }
                }

                // Build the PYTHONPATH - order matters!
                var pathComponents = new List<string>();

                // Standard library zip (if exists) - highest priority for standard library
                string stdlibZip = Path.Combine(pythonHome, "python310.zip");
                if (File.Exists(stdlibZip))
                {
                    pathComponents.Add(stdlibZip);
                }

                // Lib directory (standard library)
                if (Directory.Exists(libDir))
                {
                    pathComponents.Add(libDir);
                }

                // DLLs/lib-dynload directory (binary extensions)
                if (Directory.Exists(dllsDir))
                {
                    pathComponents.Add(dllsDir);
                }

                // site-packages (third-party packages)
                if (Directory.Exists(sitePackages))
                {
                    pathComponents.Add(sitePackages);
                }

                // Base Python home
                pathComponents.Add(pythonHome);

                string pythonPath = string.Join(Path.PathSeparator.ToString(), pathComponents);

                Log($"Configuring Python Environment (ISOLATED MODE):");
                Log($"  Platform: {(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Windows" : RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "Linux" : "macOS")}");
                Log($"  PYTHONHOME: {pythonHome}");
                Log($"  Python DLL: {pythonDll}");
                Log($"  Lib Dir: {libDir} (exists: {Directory.Exists(libDir)})");
                Log($"  Site-packages: {sitePackages} (exists: {Directory.Exists(sitePackages)})");
                Log($"  DLLs/lib-dynload: {dllsDir} (exists: {Directory.Exists(dllsDir)})");
                Log($"  PYTHONPATH: {pythonPath}");

                // Set Python.NET properties BEFORE initialization (this is the key fix)
                PythonEngine.PythonHome = pythonHome;
                PythonEngine.PythonPath = pythonPath;

                // Also set environment variables as backup
                Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome, EnvironmentVariableTarget.Process);
                Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath, EnvironmentVariableTarget.Process);

                // Isolation flags - prevent loading from user/system locations
                Environment.SetEnvironmentVariable("PYTHONNOUSERSITE", "1", EnvironmentVariableTarget.Process);
                Environment.SetEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", EnvironmentVariableTarget.Process);

                PythonEngine.Initialize();
                _threadState = PythonEngine.BeginAllowThreads();
                _isInitialized = true;

                // Post-initialization: Verify and clean sys.path
                CleanSysPath(pythonHome);

                SetupStdioRedirection();

                Log($"Python initialized successfully. Home: {pythonHome}");
            }
            catch (Exception ex)
            {
                InitializationError = $"Failed to initialize Python: {ex.Message}";
                Log($"Warning: {InitializationError}");
                Log($"Python features will be disabled. The application will continue without AI functionality.");
                // Don't re-throw - allow app to start without Python
            }
        }

        /// <summary>
        /// Clears any system Python environment variables that could interfere with our embedded Python.
        /// This is critical to prevent other Python installations from being picked up.
        /// </summary>
        private void ClearSystemPythonEnvironment()
        {
            // List of environment variables that could cause Python to look in wrong places
            string[] pythonEnvVars = {
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONSTARTUP",
                "PYTHONUSERBASE",
                "PYTHONEXECUTABLE",
                "PYTHONWARNINGS",
                "PYTHONHASHSEED",
                "PYTHONIOENCODING",
                "PYTHONLEGACYWINDOWSFSENCODING",
                "PYTHONLEGACYWINDOWSSTDIO",
                "PYTHONCOERCECLOCALE",
                "PYTHONDEVMODE",
                "PYTHONUTF8",
                "PYTHONFAULTHANDLER",
                "PYTHONTRACEMALLOC",
                "PYTHONPROFILEIMPORTTIME",
                "PYTHONMALLOC",
                "PYTHONMALLOCSTATS"
            };

            foreach (var varName in pythonEnvVars)
            {
                string? existingValue = Environment.GetEnvironmentVariable(varName);
                if (!string.IsNullOrEmpty(existingValue))
                {
                    Log($"  Clearing system {varName}: {existingValue}");
                    Environment.SetEnvironmentVariable(varName, null, EnvironmentVariableTarget.Process);
                }
            }

            // Also clear VIRTUAL_ENV if set
            string? virtualEnv = Environment.GetEnvironmentVariable("VIRTUAL_ENV");
            if (!string.IsNullOrEmpty(virtualEnv))
            {
                Log($"  Clearing VIRTUAL_ENV: {virtualEnv}");
                Environment.SetEnvironmentVariable("VIRTUAL_ENV", null, EnvironmentVariableTarget.Process);
            }
        }

        /// <summary>
        /// After Python initialization, clean sys.path to remove any paths outside our embedded environment.
        /// </summary>
        private void CleanSysPath(string pythonHome)
        {
            using (Py.GIL())
            {
                try
                {
                    dynamic sys = Py.Import("sys");

                    // Normalize our pythonHome for comparison
                    string normalizedHome = Path.GetFullPath(pythonHome).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

                    // Get current sys.path as a list
                    var currentPath = new List<string>();
                    foreach (var item in sys.path)
                    {
                        currentPath.Add(item.ToString());
                    }

                    Log($"  Checking sys.path ({currentPath.Count} entries)...");

                    // Filter to only keep paths under our pythonHome
                    var cleanedPath = new List<string>();
                    var removedPaths = new List<string>();

                    foreach (string path in currentPath)
                    {
                        if (string.IsNullOrEmpty(path))
                        {
                            continue;
                        }

                        string normalizedPath;
                        try
                        {
                            normalizedPath = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                        }
                        catch
                        {
                            // Invalid path, skip it
                            removedPaths.Add(path);
                            continue;
                        }

                        // Keep the path if it's under our pythonHome or is a zip file in pythonHome
                        if (normalizedPath.StartsWith(normalizedHome, StringComparison.OrdinalIgnoreCase))
                        {
                            cleanedPath.Add(path);
                        }
                        else
                        {
                            removedPaths.Add(path);
                        }
                    }

                    // Log removed paths
                    foreach (string removed in removedPaths)
                    {
                        Log($"  Removed external path from sys.path: {removed}");
                    }

                    // Rebuild sys.path with only our paths
                    sys.path.clear();
                    foreach (string path in cleanedPath)
                    {
                        sys.path.append(path);
                    }

                    Log($"  sys.path cleaned: {cleanedPath.Count} paths retained, {removedPaths.Count} external paths removed");
                }
                catch (Exception ex)
                {
                    Log($"  Warning: Failed to clean sys.path: {ex.Message}");
                }
            }
        }

        private void EnsurePythonEnvironment()
        {
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string targetDir = Path.Combine(appData, "Deep3DStudio", "python");
            string sourceZip = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_env.zip");

            if (!File.Exists(sourceZip))
            {
                // Fallback to local python folder if zip is missing (dev mode)
                string localPython = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python");
                if (Directory.Exists(localPython))
                {
                    Log("Using local 'python' directory.");
                    return;
                }
                Log("Warning: python_env.zip not found and no local python dir.");
                return;
            }

            // Simple version check or existence check
            // For now, if target dir exists, assume it's good?
            // Better: use a hash or version file.
            // Minimal: Check if target exists. If so, do nothing?
            // Issue: Updates won't apply.
            // Aggressive: Always delete and unzip? Slow.
            // Compromise: Check if zip is newer than target folder creation time?

            bool shouldExtract = false;
            if (!Directory.Exists(targetDir))
            {
                shouldExtract = true;
            }
            else
            {
                DateTime zipTime = File.GetLastWriteTime(sourceZip);
                DateTime dirTime = Directory.GetCreationTime(targetDir);
                if (zipTime > dirTime)
                {
                    shouldExtract = true;
                    Log("Update detected. Re-extracting python environment...");
                }
            }

            if (shouldExtract)
            {
                try
                {
                    if (Directory.Exists(targetDir))
                    {
                        Directory.Delete(targetDir, true);
                    }

                    Log($"Extracting {sourceZip} to {targetDir}...");
                    ZipFile.ExtractToDirectory(sourceZip, targetDir);
                }
                catch (Exception ex)
                {
                    Log($"Failed to extract python env: {ex.Message}");
                }
            }
        }

        private string GetPythonHome()
        {
            // 1. Search in AppData (Extracted location)
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string targetDir = Path.Combine(appData, "Deep3DStudio", "python");

            string? foundHome = FindPythonHomeRecursive(targetDir);
            if (foundHome != null) return foundHome;

            // 2. Search in Local Directory (Dev/Debug mode)
            string localPython = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python");
            foundHome = FindPythonHomeRecursive(localPython);
            if (foundHome != null) return foundHome;

            // 3. Fallback to basic paths if search failed
            if (Directory.Exists(targetDir)) return targetDir;
            return localPython;
        }

        private string? FindPythonHomeRecursive(string rootDir)
        {
            if (!Directory.Exists(rootDir)) return null;

            // Define the target DLL name based on platform
            string dllName = "python310.dll";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) dllName = "libpython3.10.so";
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX)) dllName = "libpython3.10.dylib";

            // DFS search for the DLL
            // We use a stack to avoid recursion depth issues, though unlikely here
            var stack = new Stack<string>();
            stack.Push(rootDir);

            int safetyCounter = 0;
            while (stack.Count > 0 && safetyCounter++ < 1000)
            {
                string currentDir = stack.Pop();

                // Check if DLL exists here
                string dllPath = Path.Combine(currentDir, dllName);
                if (File.Exists(dllPath))
                {
                    // Special case for Unix: libpython is often in 'lib' subdir, but PYTHONHOME is the parent
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                    {
                        if (Path.GetFileName(currentDir) == "lib")
                        {
                            return Directory.GetParent(currentDir)?.FullName ?? currentDir;
                        }
                    }
                    return currentDir;
                }

                try
                {
                    foreach (string dir in Directory.GetDirectories(currentDir))
                    {
                        stack.Push(dir);
                    }
                }
                catch { /* Ignore access errors */ }
            }

            return null;
        }

        private string GetPythonDllPath(string pythonHome)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return Path.Combine(pythonHome, "python310.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                // If pythonHome was set to the root (parent of lib), adjust path
                string libPath = Path.Combine(pythonHome, "lib", "libpython3.10.so");
                if (File.Exists(libPath)) return libPath;
                // Fallback if home points directly to lib (unlikely with FindPythonHomeRecursive logic but possible)
                return Path.Combine(pythonHome, "libpython3.10.so");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                string libPath = Path.Combine(pythonHome, "lib", "libpython3.10.dylib");
                if (File.Exists(libPath)) return libPath;
                return Path.Combine(pythonHome, "libpython3.10.dylib");
            }

            throw new PlatformNotSupportedException("Unsupported platform for embedded Python.");
        }

        private void SetupStdioRedirection()
        {
            using (Py.GIL())
            {
                string redirectScript = @"
import sys
class LoggerWriter:
    def write(self, message):
        import Deep3DStudio.Python
        # We will use a callback mechanism or simple print hook if possible.
        # But calling back into C# from here requires loading the CLR assembly.
        pass
    def flush(self):
        pass

# Simple redirection via sys.stdout assignment is tricky without full interop setup.
# Easier method: Use Python.Runtime.PyObject to wrap a C# object.
";
                // Alternative: We define a C# class that we expose to Python
                // However, simpler is just to capture output if we run scripts via RunString
                // But for global capturing:

                dynamic sys = Py.Import("sys");
                sys.stdout = new OutputRedirector(this);
                sys.stderr = new OutputRedirector(this);
            }
        }

        public void RunScript(string script)
        {
            EnsureInitialized();
            if (!_isInitialized) throw new InvalidOperationException($"Python not initialized: {InitializationError}");

            using (Py.GIL())
            {
                PythonEngine.Exec(script);
            }
        }

        public void ExecuteWithGIL(Action<PyModule> action)
        {
            EnsureInitialized();
            if (!_isInitialized) throw new InvalidOperationException($"Python not initialized: {InitializationError}");

            using (Py.GIL())
            using (var scope = Py.CreateScope())
            {
                action(scope);
            }
        }

        // Helper to run code in the main shared scope if needed, or import modules
        public dynamic Import(string moduleName)
        {
            EnsureInitialized();
            if (!_isInitialized) throw new InvalidOperationException($"Python not initialized: {InitializationError}");

            using (Py.GIL())
            {
                return Py.Import(moduleName);
            }
        }

        public void EnsureInitialized()
        {
            if (!_isInitialized) Initialize();
        }

        private void Log(string message)
        {
            Console.WriteLine("[PythonService] " + message);
            OnLogOutput?.Invoke(message);
        }

        public void Dispose()
        {
            if (_isInitialized)
            {
                PythonEngine.Shutdown();
                _isInitialized = false;
            }
        }
    }

    // Redirector class that Python can call
    public class OutputRedirector
    {
        private PythonService _service;
        public OutputRedirector(PythonService service)
        {
            _service = service;
        }

        public void write(string message)
        {
            if (!string.IsNullOrEmpty(message))
            {
                // Invoke log event.
                // Note: This runs on the thread Python is executing on.
                // Log method handles it.
                 // We might want to filter newlines if they are just flushing
                 if (message != "\n")
                    Console.WriteLine("[Py] " + message); // Direct to console for now, service event later
            }
        }

        public void flush() { }
    }
}
