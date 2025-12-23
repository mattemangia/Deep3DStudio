using System;
using Gtk;
using Deep3DStudio.Icons;
using Deep3DStudio.Configuration;
using Deep3DStudio.UI;
using Deep3DStudio.Python;

namespace Deep3DStudio
{
    class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            // Try to force MESA to give a compatibility profile (legacy GL support)
            // This is critical for GL.Begin/GL.End calls in the viewport.
            Environment.SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3COMPAT");

            // macOS-specific: Ensure proper GTK backend and rendering
            // This fixes the grey screen issue on macOS after installing gtk3+
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX))
            {
                // Force GTK to use Quartz backend on macOS for proper window rendering
                Environment.SetEnvironmentVariable("GDK_BACKEND", "quartz");

                // Disable client-side decorations which can cause rendering issues on macOS
                Environment.SetEnvironmentVariable("GTK_CSD", "0");
            }

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

            // Initialize Python Environment (Heavy Task)
            // This is now non-blocking - app will start even if Python fails
            splash.UpdateStatus("Initializing Python Environment (this may take a moment)...");

            // Hook up logging to splash screen
            Action<string> logHandler = (msg) => {
                if (msg != null && msg.Length > 0 && msg != "\n")
                {
                    Console.WriteLine($"[Python] {msg.Trim()}");
                    splash.UpdateStatus(msg.Trim());
                }
            };
            PythonService.Instance.OnLogOutput += logHandler;

            try
            {
                // Force initialization now to show progress on splash
                // This will extract the zip if needed
                // Note: Initialize() will not throw even if Python is missing - it will log warnings
                PythonService.Instance.Initialize();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Python initialization error: {ex.Message}");
                Console.WriteLine($"Application will continue without Python/AI features.");
                splash.UpdateStatus("Continuing without Python (AI features disabled)");
            }
            finally
            {
                PythonService.Instance.OnLogOutput -= logHandler;
            }

            var app = new Application("org.Deep3DStudio.Deep3DStudio", GLib.ApplicationFlags.None);
            app.Register(GLib.Cancellable.Current);

            splash.UpdateStatus("Loading User Interface...");
            var win = new MainWindow();
            app.AddWindow(win);

            // Process any pending events before destroying splash
            while (Application.EventsPending())
                Application.RunIteration();

            splash.Destroy();

            // ShowAll is critical for proper rendering, especially on macOS
            // This ensures all widgets are properly initialized and visible
            win.ShowAll();

            // macOS-specific: Ensure the window is brought to front and activated
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX))
            {
                // Present the window to bring it to front
                win.Present();

                // Activate to ensure it receives focus
                win.Activate();

                // Process events to ensure rendering
                while (Application.EventsPending())
                    Application.RunIteration();

                // Small delay for window manager
                System.Threading.Thread.Sleep(100);

                // Process events one more time
                while (Application.EventsPending())
                    Application.RunIteration();
            }
            else
            {
                // Non-macOS: Just process events
                while (Application.EventsPending())
                    Application.RunIteration();
            }

            Application.Run();

            // Prevent premature garbage collection of the Gtk.Application instance
            // while the native main loop is running.
            GC.KeepAlive(app);
        }
    }
}
