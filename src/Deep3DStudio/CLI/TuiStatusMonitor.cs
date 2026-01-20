using Terminal.Gui;
using System;
using System.Threading;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;

namespace Deep3DStudio.CLI
{
    public class TuiStatusMonitor
    {
        private static TuiStatusMonitor? _instance;
        public static TuiStatusMonitor Instance => _instance ??= new TuiStatusMonitor();

        private Thread? _tuiThread;
        private Window? _window;
        private TextView? _logView;
        private ProgressBar? _progressBar;
        private Label? _statusLabel;
        private bool _isRunning;
        private TuiWriter? _writer;
        private ManualResetEvent _initEvent = new ManualResetEvent(false);

        // P/Invoke to allocate console on Windows
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        static extern bool AllocConsole();

        [DllImport("kernel32.dll")]
        static extern IntPtr GetConsoleWindow();

        [DllImport("user32.dll")]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        private const int SW_SHOW = 5;

        public void Start()
        {
            if (_isRunning) return;
            _isRunning = true;
            _initEvent.Reset();

            // Ensure we have a console window on Windows
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                IntPtr handle = GetConsoleWindow();
                if (handle == IntPtr.Zero)
                {
                    AllocConsole();
                }
                else
                {
                    ShowWindow(handle, SW_SHOW);
                }
            }

            // Start TUI in a separate thread to not block GTK/ImGui
            _tuiThread = new Thread(TuiLoop);
            _tuiThread.IsBackground = true;
            
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _tuiThread.SetApartmentState(ApartmentState.STA); // Better for UI threads on Windows
            }
            
            _tuiThread.Start();

            // Wait for TUI to be ready (up to 5 seconds)
            _initEvent.WaitOne(5000);

            // Redirect Console Output
            _writer = new TuiWriter(this);
            Console.SetOut(_writer);
            Console.SetError(_writer);
            
            // Explicit test message
            Console.WriteLine("TUI Console attached successfully (via Console.WriteLine).");
            Log("TUI Direct Log Test.\n");
        }

        private void TuiLoop()
        {
            try
            {
                Application.Init();

                // High Contrast Dark Theme (White on Black)
                var darkScheme = new ColorScheme()
                {
                    Normal = new Terminal.Gui.Attribute(Color.White, Color.Black),
                    Focus = new Terminal.Gui.Attribute(Color.White, Color.Black),
                    HotNormal = new Terminal.Gui.Attribute(Color.Cyan, Color.Black),
                    HotFocus = new Terminal.Gui.Attribute(Color.Cyan, Color.Black)
                };

                _window = new Window("Deep3DStudio Console")
                {
                    X = 0,
                    Y = 0,
                    Width = Dim.Fill(),
                    Height = Dim.Fill(),
                    ColorScheme = darkScheme
                };

                _statusLabel = new Label("Status: Initializing...")
                {
                    X = 1,
                    Y = 1,
                    Width = Dim.Fill() - 2,
                    ColorScheme = darkScheme
                };

                _progressBar = new ProgressBar()
                {
                    X = 1,
                    Y = 2,
                    Width = Dim.Fill() - 2,
                    Fraction = 0f,
                    ColorScheme = darkScheme
                };

                _logView = new TextView()
                {
                    X = 1,
                    Y = 4,
                    Width = Dim.Fill() - 2,
                    Height = Dim.Fill(),
                    ReadOnly = true,
                    ColorScheme = darkScheme,
                    Text = "Deep3DStudio Log Started...\n"
                };

                _window.Add(_statusLabel, _progressBar, new Label(1, 3, "Logs:") { ColorScheme = darkScheme }, _logView);
                Application.Top.Add(_window);

                // Signal that UI is built
                _initEvent.Set();

                Application.Run();
            }
            catch (Exception ex)
            {
                _isRunning = false;
                _initEvent.Set();
                File.WriteAllText("tui_error.log", $"TUI Init Failed: {ex}\n");
            }
        }
        
        // ... Stop ...

        public void SetStatus(string status)
        {
            if (!_isRunning) return;
            Application.MainLoop.Invoke(() =>
            {
                if (_statusLabel != null) 
                {
                    _statusLabel.Text = $"Status: {status}";
                    _statusLabel.SetNeedsDisplay();
                    Application.MainLoop.Driver.Wakeup(); // Force wake up to redraw immediately
                }
            });
        }

        public void Log(string message)
        {
            // Debug logging to verify data reception
            try { File.AppendAllText("tui_debug.log", message); } catch { }

            if (!_isRunning || _logView == null) return;

            Application.MainLoop.Invoke(() =>
            {
                if (_logView != null)
                {
                    // Append text. 
                    var currentText = _logView.Text.ToString();
                    
                    if (currentText.Length > 20000) 
                        currentText = currentText.Substring(currentText.Length - 10000);
                        
                    _logView.Text = currentText + message;
                    
                    if (_logView.Lines > 0)
                    {
                        // Scroll to bottom: Set the top row such that the last line is at the bottom of the view
                        int height = _logView.Bounds.Height;
                        int topRow = Math.Max(0, _logView.Lines - height);
                        _logView.ScrollTo(topRow);
                    }
                        
                    _logView.SetNeedsDisplay();
                    Application.MainLoop.Driver.Wakeup(); // Force wake up to redraw immediately
                }
            });
        }

        public void UpdateProgress(string task, float progress)
        {
            if (!_isRunning) return;

            Application.MainLoop.Invoke(() =>
            {
                if (_statusLabel != null) {
                     _statusLabel.Text = $"Status: {task}";
                     _statusLabel.SetNeedsDisplay();
                }
                if (_progressBar != null) {
                    _progressBar.Fraction = Math.Clamp(progress, 0f, 1f);
                    _progressBar.SetNeedsDisplay();
                }
                Application.MainLoop.Driver.Wakeup(); // Force wake up to redraw immediately
            });
        }

        private class TuiWriter : TextWriter
        {
            private readonly TuiStatusMonitor _monitor;
            public override Encoding Encoding => Encoding.UTF8;

            public TuiWriter(TuiStatusMonitor monitor)
            {
                _monitor = monitor;
            }

            public override void Write(char value)
            {
                _monitor.Log(value.ToString());
            }

            public override void Write(string? value)
            {
                if (value != null) _monitor.Log(value);
            }

            public override void WriteLine(string? value)
            {
                if (value != null) _monitor.Log(value + "\n");
            }

            public override void Write(char[] buffer, int index, int count)
            {
                if (buffer != null)
                {
                    _monitor.Log(new string(buffer, index, count));
                }
            }

            public override void Flush() { }
        }
    }
}
