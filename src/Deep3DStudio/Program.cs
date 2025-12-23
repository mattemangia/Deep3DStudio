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

            // macOS-specific GTK configuration
            if (System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX))
            {
                // Don't force a backend - let GTK auto-detect (Quartz or X11)
                // Note: Removed GTK_CSD and GTK_THEME settings as they can cause
                // rendering issues on macOS with certain GTK versions
            }

            try
            {
                Application.Init();
                Console.WriteLine("GTK Application initialized successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FATAL: GTK init failed: {ex.Message}");
                Console.WriteLine($"Stack: {ex.StackTrace}");
                return;
            }

            // Check GTK display
            var display = Gdk.Display.Default;
            if (display == null)
            {
                Console.WriteLine("FATAL: No GDK display available");
                Console.WriteLine("On macOS, you may need to:");
                Console.WriteLine("  1. Install XQuartz: brew install --cask xquartz");
                Console.WriteLine("  2. Log out and log back in");
                Console.WriteLine("  3. Set DISPLAY environment variable");
                return;
            }
            Console.WriteLine($"GDK Display: {display.Name}");

            // Show Splash Screen
            Console.WriteLine("Creating splash screen...");
            var splash = new SplashScreen();
            Console.WriteLine("Presenting splash screen...");
            splash.Present();

            // Ensure the splash window is fully realized and visible
            int maxWait = 100; // Max iterations to wait
            while (!splash.IsRealized && maxWait-- > 0)
            {
                Application.RunIteration(false);
            }
            Console.WriteLine($"Splash screen realized: {splash.IsRealized}");

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

            splash.UpdateStatus("Loading User Interface...");
            var win = new MainWindow();

            // Process pending events
            while (Application.EventsPending())
                Application.RunIteration(false);

            // Destroy splash and show main window
            splash.Destroy();

            // Process the splash destruction
            while (Application.EventsPending())
                Application.RunIteration(false);

            // Show the main window
            win.ShowAll();
            win.Present();

            // Process events after showing
            while (Application.EventsPending())
                Application.RunIteration(false);

            Console.WriteLine("Starting GTK main loop...");
            Application.Run();
        }
    }
}
