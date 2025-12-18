using System;
using System.IO;
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

        private string GetPythonHome()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            return Path.Combine(baseDir, "python");
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
