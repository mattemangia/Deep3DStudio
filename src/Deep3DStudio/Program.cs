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

            // Redirect Console Output to Log Window
            try
            {
                Console.SetOut(new LogWriter(Console.Out, LogWindow.Instance));
                // Optional: Redirect error too if desired, but might want to keep native error stream
                // Console.SetError(new LogWriter(Console.Error, LogWindow.Instance));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to redirect console: {ex.Message}");
            }

            // Initialize Python Environment (Heavy Task)
            splash.UpdateStatus("Initializing Python Environment (this may take a moment)...");

            // Hook up logging to splash screen
            Action<string> logHandler = (msg) => {
                if (msg != null && msg.Length > 0 && msg != "\n")
                    splash.UpdateStatus(msg.Trim());
            };
            PythonService.Instance.OnLogOutput += logHandler;

            try
            {
                // Force initialization now to show progress on splash
                // This will extract the zip if needed
                PythonService.Instance.Initialize();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Python init warning: {ex.Message}");
                splash.UpdateStatus("Python init warning (check console)");
                // Continue, as some features might work or user can fix in settings
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

            splash.Destroy();
            win.Show();
            Application.Run();

            // Prevent premature garbage collection of the Gtk.Application instance
            // while the native main loop is running.
            GC.KeepAlive(app);
        }
    }
}
