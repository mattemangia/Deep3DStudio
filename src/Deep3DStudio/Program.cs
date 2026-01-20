using System;
using Gtk;
using Deep3DStudio.Icons;
using Deep3DStudio.Configuration;
using Deep3DStudio.UI;
using Deep3DStudio.Python;
using Deep3DStudio.CLI;

namespace Deep3DStudio
{
    class Program
    {
        [STAThread]
        public static int Main(string[] args)
        {
            // Parse command-line options first
            var cliOptions = CommandLineOptions.Parse(args);

            // If CLI mode is detected, run in CLI mode without GUI
            if (cliOptions.IsCLIMode)
            {
                // Display logo in standard console
                Console.WriteLine(ConsoleLogo.GenerateAsciiLogo());
                Console.WriteLine("CLI mode detected");
                return RunCLI(cliOptions);
            }
            else
            {
                // Start TUI Status Monitor for GUI mode
                TuiStatusMonitor.Instance.Start();
                
                // Hook Python extraction progress to TUI
                PythonService.Instance.OnExtractionProgress += (msg, prog) => {
                    TuiStatusMonitor.Instance.UpdateProgress(msg, prog);
                };
                
                // Hook AI Model Manager progress to TUI
                var aiManager = Model.AIModels.AIModelManager.Instance;
                aiManager.ProgressUpdated += (msg, prog) => {
                     TuiStatusMonitor.Instance.UpdateProgress(msg, prog);
                };
                aiManager.ModelLoadProgress += (stage, prog, msg) => {
                     TuiStatusMonitor.Instance.UpdateProgress($"{stage}: {msg}", prog);
                };
            }

            // Try to force MESA to give a compatibility profile (legacy GL support)
            // This is critical for GL.Begin/GL.End calls in the viewport.
            Environment.SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3COMPAT");

            // CRITICAL: Initialize Python BEFORE GTK to avoid GC/ToggleRef conflicts
            // Python.NET and GtkSharp both hook into .NET's GC, and initializing Python
            // while GTK is running can cause ToggleRef corruption
            Console.WriteLine("Pre-initializing Python environment...");
            try
            {
                // Simple console logging during Python init (no GTK operations)
                PythonService.Instance.OnLogOutput += (msg) => {
                    if (msg != null && msg.Length > 0 && msg != "\n")
                        Console.WriteLine($"[Python] {msg.Trim()}");
                };
                PythonService.Instance.Initialize();
                Console.WriteLine("Python environment ready.");
                
                // Update TUI status
                if (!cliOptions.IsCLIMode)
                    TuiStatusMonitor.Instance.SetStatus("Ready");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Python initialization error: {ex.Message}");
                Console.WriteLine($"Application will continue without Python/AI features.");
            }

            // Force GC before GTK initialization to ensure clean slate
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            // NOW initialize GTK after Python is fully set up
            Application.Init();

            // Show Splash Screen
            var splash = new SplashScreen();
            splash.Show();
            splash.UpdateStatus("Loading Settings...");

            // Initialize settings from INI file (platform-specific location)
            var settings = IniSettings.Instance;
            Console.WriteLine($"Settings loaded from: {IniSettings.GetSettingsPath()}");

            splash.UpdateStatus("Initializing Application Icons...");
            // Set default application icon for all windows
            try
            {
                var iconSet = ApplicationIcon.CreateIconSet();
                Gtk.Window.DefaultIconList = iconSet;
                Console.WriteLine("Application icon set successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not set application icon: {ex.Message}");
            }

            splash.UpdateStatus("Python environment ready.");

            var app = new Application("org.Deep3DStudio.Deep3DStudio", GLib.ApplicationFlags.None);
            app.Register(GLib.Cancellable.Current);

            splash.UpdateStatus("Loading User Interface...");
            var win = new MainWindow();
            app.AddWindow(win);

            splash.Destroy();
            win.Show();
            Application.Run();

            // Prevent premature garbage collection of the Gtk.Application instance
            GC.KeepAlive(app);
            return 0;
        }

        /// <summary>
        /// Run in CLI mode for command-line inference testing.
        /// </summary>
        private static int RunCLI(CommandLineOptions options)
        {
            try
            {
                var runner = new CLIRunner(options);
                return runner.Run();
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Fatal error: {ex.Message}");
                if (options.Verbose)
                {
                    Console.Error.WriteLine(ex.StackTrace);
                }
                return 1;
            }
        }
    }
}
