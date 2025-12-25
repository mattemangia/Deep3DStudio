using System;
using System.IO;
using System.Runtime.CompilerServices;

namespace Deep3DStudio
{
    /// <summary>
    /// Simple file-based logger that clears the log on each application start.
    /// Thread-safe for concurrent logging from multiple threads.
    /// </summary>
    public static class Logger
    {
        private static readonly object _lock = new object();
        private static string _logPath = null!;
        private static bool _initialized = false;

        /// <summary>
        /// Gets the path to the log file.
        /// </summary>
        public static string LogPath => _logPath;

        /// <summary>
        /// Initializes the logger, clearing any existing log file.
        /// Call this once at application startup.
        /// </summary>
        public static void Initialize()
        {
            lock (_lock)
            {
                if (_initialized) return;

                // Log file in the application directory
                string appDir = AppDomain.CurrentDomain.BaseDirectory;
                _logPath = Path.Combine(appDir, "deep3dstudio.log");

                // Clear the log file on each run
                try
                {
                    File.WriteAllText(_logPath, $"=== Deep3DStudio Log Started: {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===\n");
                    File.AppendAllText(_logPath, $"Platform: {Environment.OSVersion}\n");
                    File.AppendAllText(_logPath, $"Runtime: {Environment.Version}\n");
                    File.AppendAllText(_logPath, $"Working Directory: {Environment.CurrentDirectory}\n");
                    File.AppendAllText(_logPath, new string('=', 60) + "\n\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to initialize logger: {ex.Message}");
                }

                _initialized = true;
            }
        }

        /// <summary>
        /// Logs an informational message.
        /// </summary>
        public static void Info(string message,
            [CallerFilePath] string file = "",
            [CallerLineNumber] int line = 0,
            [CallerMemberName] string member = "")
        {
            Log("INFO", message, file, line, member);
        }

        /// <summary>
        /// Logs a warning message.
        /// </summary>
        public static void Warn(string message,
            [CallerFilePath] string file = "",
            [CallerLineNumber] int line = 0,
            [CallerMemberName] string member = "")
        {
            Log("WARN", message, file, line, member);
        }

        /// <summary>
        /// Logs an error message.
        /// </summary>
        public static void Error(string message,
            [CallerFilePath] string file = "",
            [CallerLineNumber] int line = 0,
            [CallerMemberName] string member = "")
        {
            Log("ERROR", message, file, line, member);
        }

        /// <summary>
        /// Logs an exception with full stack trace.
        /// </summary>
        public static void Exception(Exception ex, string context = "",
            [CallerFilePath] string file = "",
            [CallerLineNumber] int line = 0,
            [CallerMemberName] string member = "")
        {
            string message = string.IsNullOrEmpty(context)
                ? $"{ex.GetType().Name}: {ex.Message}"
                : $"{context} - {ex.GetType().Name}: {ex.Message}";

            Log("EXCEPTION", message, file, line, member);
            WriteRaw($"  Stack Trace:\n{ex.StackTrace}\n");

            if (ex.InnerException != null)
            {
                WriteRaw($"  Inner Exception: {ex.InnerException.GetType().Name}: {ex.InnerException.Message}\n");
                WriteRaw($"  Inner Stack Trace:\n{ex.InnerException.StackTrace}\n");
            }
        }

        /// <summary>
        /// Logs a debug message (for detailed tracing).
        /// </summary>
        public static void Debug(string message,
            [CallerFilePath] string file = "",
            [CallerLineNumber] int line = 0,
            [CallerMemberName] string member = "")
        {
            Log("DEBUG", message, file, line, member);
        }

        private static void Log(string level, string message, string file, int line, string member)
        {
            if (!_initialized)
            {
                Initialize();
            }

            string fileName = Path.GetFileName(file);
            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string threadId = Environment.CurrentManagedThreadId.ToString();
            string logLine = $"[{timestamp}] [{level}] [Thread:{threadId}] [{fileName}:{line} {member}] {message}\n";

            WriteRaw(logLine);
        }

        private static void WriteRaw(string text)
        {
            lock (_lock)
            {
                try
                {
                    File.AppendAllText(_logPath, text);
                }
                catch
                {
                    // Silently fail if we can't write to log
                }
            }

            // Also write to console for debugging
            Console.Write(text);
        }

        /// <summary>
        /// Flushes any buffered log entries (no-op for current implementation).
        /// </summary>
        public static void Flush()
        {
            // File.AppendAllText already flushes, but this is here for future use
        }
    }
}
