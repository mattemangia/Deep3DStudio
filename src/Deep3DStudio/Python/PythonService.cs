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

                // Set PYTHONHOME before initialization if needed, though often setting PythonDLL is enough for Embeddable
                // However, for embeddable python, we often need to set environment variables or python paths manually
                // if the layout is custom.
                Environment.SetEnvironmentVariable("PYTHONHOME", pythonHome);
                Environment.SetEnvironmentVariable("PYTHONPATH", Path.Combine(pythonHome, "Lib", "site-packages") + ";" + pythonHome);

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
            // Prioritize the extracted location in AppData
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string targetDir = Path.Combine(appData, "Deep3DStudio", "python");

            // Fix for nested python folder in zip (e.g. zip contains 'python/' folder at root)
            string nestedDir = Path.Combine(targetDir, "python");
            if (Directory.Exists(nestedDir))
            {
                return nestedDir;
            }

            if (Directory.Exists(targetDir))
            {
                return targetDir;
            }

            // Fallback to local directory
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string localPython = Path.Combine(baseDir, "python");

            // Also check for nested in local
            string localNested = Path.Combine(localPython, "python");
            if (Directory.Exists(localNested)) return localNested;

            return localPython;
        }

        private string GetPythonDllPath(string pythonHome)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows embeddable usually has python3xx.dll in the root of the python folder
                return Path.Combine(pythonHome, "python310.dll");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                return Path.Combine(pythonHome, "lib", "libpython3.10.so");
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return Path.Combine(pythonHome, "lib", "libpython3.10.dylib");
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
