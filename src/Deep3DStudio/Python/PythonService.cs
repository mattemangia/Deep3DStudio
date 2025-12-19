using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using Python.Runtime;
using Deep3DStudio.Configuration; // Assuming for logging or settings

namespace Deep3DStudio.Python
{
    public class PythonService : IDisposable
    {
        private static PythonService? _instance;
        private static readonly object _lock = new object();
        private IntPtr _threadState;
        private bool _isInitialized = false;

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
                    Log($"Error: Python DLL not found at {pythonDll}");
                    // Fallback or throw? For now log.
                }

                Runtime.PythonDLL = pythonDll;

                // Configure environment to ensure isolated execution and correct path resolution
                // We must set PYTHONHOME and PYTHONPATH to ensure the embedded environment is found
                // and system-wide packages (like FEFLOW or Anaconda) are ignored.

                // Construct a robust PYTHONPATH including Lib (standard library) and DLLs (extensions)
                // This is critical if the zip distribution is a full install (containing Lib folder) rather than an embeddable zip.
                string libDir = Path.Combine(pythonHome, "Lib");
                string sitePackages = Path.Combine(libDir, "site-packages");
                string dllsDir = Path.Combine(pythonHome, "DLLs");

                // Base path
                string pythonPath = pythonHome;

                // Add Lib if it exists (fixes 'encodings' not found)
                if (Directory.Exists(libDir))
                {
                    pythonPath += Path.PathSeparator + libDir;
                }

                // Add site-packages
                if (Directory.Exists(sitePackages))
                {
                    pythonPath += Path.PathSeparator + sitePackages;
                }

                // Add DLLs
                if (Directory.Exists(dllsDir))
                {
                    pythonPath += Path.PathSeparator + dllsDir;
                }

                // Also check for a zip file (standard embeddable)
                string zipPath = Path.ChangeExtension(pythonDll, ".zip");
                if (File.Exists(zipPath))
                {
                    pythonPath += Path.PathSeparator + zipPath;
                }

                Log($"Configuring Python Environment:");
                Log($"  PYTHONHOME: {pythonHome}");
                Log($"  PYTHONPATH: {pythonPath}");

                Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome, EnvironmentVariableTarget.Process);
                Environment.SetEnvironmentVariable("PYTHONPATH", pythonPath, EnvironmentVariableTarget.Process);
                Environment.SetEnvironmentVariable("PYTHONNOUSERSITE", "1", EnvironmentVariableTarget.Process);

                PythonEngine.Initialize();
                _threadState = PythonEngine.BeginAllowThreads();
                _isInitialized = true;

                SetupStdioRedirection();

                Log($"Python initialized. Home: {pythonHome}, DLL: {pythonDll}");
            }
            catch (Exception ex)
            {
                Log($"Failed to initialize Python: {ex.Message}");
                throw;
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
            using (Py.GIL())
            {
                PythonEngine.Exec(script);
            }
        }

        public void ExecuteWithGIL(Action<PyModule> action)
        {
            EnsureInitialized();
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
