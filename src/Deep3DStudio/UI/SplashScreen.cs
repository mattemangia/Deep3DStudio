using System;
using Gtk;
using Gdk;

namespace Deep3DStudio.UI
{
    public class SplashScreen : Gtk.Window
    {
        private Label _statusLabel;
        private Spinner _spinner;

        public SplashScreen() : base(Gtk.WindowType.Toplevel)
        {
            this.TypeHint = WindowTypeHint.Splashscreen;
            this.WindowPosition = WindowPosition.Center;
            this.Decorated = false;
            this.BorderWidth = 0; // Use Frame for border

            // Frame for border
            var frame = new Frame();
            frame.ShadowType = ShadowType.Out;
            this.Add(frame);

            // Main container
            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 20;
            frame.Add(vbox);

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

            // Apply dark theme background
            // On macOS, skip CSS and use direct color modification for better compatibility
            bool isMacOS = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(
                System.Runtime.InteropServices.OSPlatform.OSX);

            if (isMacOS)
            {
                // macOS: Use ModifyBg directly, skip CSS
                Console.WriteLine("SplashScreen: Using ModifyBg for macOS");
                var black = new Gdk.Color(26, 26, 26);
                this.ModifyBg(StateType.Normal, black);
                frame.ModifyBg(StateType.Normal, black);
                vbox.ModifyBg(StateType.Normal, black);
            }
            else
            {
                // Other platforms: Try CSS first
                var cssProvider = new CssProvider();
                try
                {
                    cssProvider.LoadFromData(@"
                        window {
                            background-color: #1a1a1a;
                        }
                        frame {
                            background-color: #1a1a1a;
                        }
                        box {
                            background-color: #1a1a1a;
                        }
                        spinner {
                            color: #FFFFFF;
                        }
                    ");
                    this.StyleContext.AddProvider(cssProvider, StyleProviderPriority.Application);
                    frame.StyleContext.AddProvider(cssProvider, StyleProviderPriority.Application);
                    vbox.StyleContext.AddProvider(cssProvider, StyleProviderPriority.Application);
                    _spinner.StyleContext.AddProvider(cssProvider, StyleProviderPriority.Application);
                }
                catch
                {
                    // Fallback to ModifyBg if CSS fails
                    var black = new Gdk.Color(26, 26, 26);
                    this.ModifyBg(StateType.Normal, black);
                    frame.ModifyBg(StateType.Normal, black);
                    vbox.ModifyBg(StateType.Normal, black);
                }
            }

            Console.WriteLine("SplashScreen: Calling ShowAll()");
            this.ShowAll();
            Console.WriteLine("SplashScreen: ShowAll() completed");
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
