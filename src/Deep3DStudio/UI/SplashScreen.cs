using System;
using Gtk;
using Gdk;

namespace Deep3DStudio.UI
{
    public class SplashScreen : Gtk.Window
    {
        private Label _statusLabel;
        private Spinner _spinner;
        private static bool _isMacOS = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
            System.Runtime.InteropServices.OSPlatform.OSX);

        public SplashScreen() : base(Gtk.WindowType.Toplevel)
        {
            // Set window properties - use Normal hint for better macOS compatibility
            this.TypeHint = _isMacOS ? WindowTypeHint.Normal : WindowTypeHint.Splashscreen;
            this.WindowPosition = WindowPosition.Center;
            this.Decorated = false;
            this.BorderWidth = 0;
            this.SetDefaultSize(350, 400);
            this.Resizable = false;
            this.KeepAbove = true;
            this.AppPaintable = true;

            // Apply dark background immediately
            var darkColor = new Gdk.Color(26, 26, 26);
            this.ModifyBg(StateType.Normal, darkColor);

            // Frame for border
            var frame = new Frame();
            frame.ShadowType = ShadowType.Out;
            frame.ModifyBg(StateType.Normal, darkColor);
            this.Add(frame);

            // Main container with EventBox for background color
            var eventBox = new EventBox();
            eventBox.ModifyBg(StateType.Normal, darkColor);
            frame.Add(eventBox);

            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 20;
            vbox.ModifyBg(StateType.Normal, darkColor);
            eventBox.Add(vbox);

            // Logo
            try
            {
                var assembly = System.Reflection.Assembly.GetExecutingAssembly();
                using (var stream = assembly.GetManifestResourceStream("Deep3DStudio.logo.png"))
                {
                    if (stream != null)
                    {
                        var pixbuf = new Gdk.Pixbuf(stream);
                        // Scale
                        if (pixbuf.Width > 200)
                            pixbuf = pixbuf.ScaleSimple(200, 200, Gdk.InterpType.Bilinear);

                        var image = new Image(pixbuf);
                        vbox.PackStart(image, true, true, 10);
                    }
                }
            }
            catch { /* Ignore if missing */ }

            // Title
            var titleLabel = new Label();
            titleLabel.Markup = "<span size='xx-large' weight='bold' foreground='#FFFFFF'>Deep3D Studio</span>";
            vbox.PackStart(titleLabel, false, false, 5);

            var subtitleLabel = new Label("AI-Powered 3D Reconstruction");
            subtitleLabel.ModifyFg(StateType.Normal, new Gdk.Color(200, 200, 200));
            vbox.PackStart(subtitleLabel, false, false, 0);

            // Spinner container to center it
            var spinnerBox = new Box(Orientation.Horizontal, 0);
            _spinner = new Spinner();
            _spinner.SetSizeRequest(32, 32);
            _spinner.Active = true;
            _spinner.Start();

            spinnerBox.PackStart(new Label(""), true, true, 0); // Spacer
            spinnerBox.PackStart(_spinner, false, false, 0);
            spinnerBox.PackStart(new Label(""), true, true, 0); // Spacer
            vbox.PackStart(spinnerBox, false, false, 15);

            // Status
            _statusLabel = new Label("Initializing...");
            _statusLabel.ModifyFg(StateType.Normal, new Gdk.Color(180, 180, 180));
            vbox.PackStart(_statusLabel, false, false, 5);

            Console.WriteLine("SplashScreen: Constructor completed");
        }

        /// <summary>
        /// Shows the splash screen and ensures it is properly displayed on all platforms
        /// </summary>
        public void Present()
        {
            Console.WriteLine("SplashScreen: Presenting window...");

            // Show all widgets first
            this.ShowAll();

            // Ensure the window is mapped
            if (!this.IsMapped)
            {
                this.Map();
            }

            // Force window to front on macOS
            if (_isMacOS)
            {
                this.KeepAbove = true;
                this.SetUrgencyHint(true);
            }

            // Process pending events to ensure window is displayed
            ProcessPendingEvents();

            // Force a redraw
            this.QueueDraw();
            ProcessPendingEvents();

            Console.WriteLine($"SplashScreen: Window visible={this.IsVisible}, mapped={this.IsMapped}");
        }

        private void ProcessPendingEvents()
        {
            // Process all pending GTK events
            while (Application.EventsPending())
            {
                Application.RunIteration(false);
            }
        }

        public void UpdateStatus(string message)
        {
            if (string.IsNullOrEmpty(message)) return;
            _statusLabel.Text = message;

            // Force UI update
            while (Application.EventsPending())
                Application.RunIteration();
        }
    }
}
