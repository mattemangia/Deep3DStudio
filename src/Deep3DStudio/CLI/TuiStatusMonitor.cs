using Terminal.Gui;
using System;
using System.Threading;
using System.IO;
using System.Text;

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

        public void Start()
        {
            if (_isRunning) return;
            _isRunning = true;

            // Start TUI in a separate thread to not block GTK/ImGui
            _tuiThread = new Thread(TuiLoop);
            _tuiThread.IsBackground = true;
            _tuiThread.Start();

            // Give the TUI a moment to initialize
            Thread.Sleep(500);

            // Redirect Console Output
            _writer = new TuiWriter(this);
            Console.SetOut(_writer);
            Console.SetError(_writer);
        }

        private void TuiLoop()
        {
            try
            {
                Application.Init();

                _window = new Window("Deep3DStudio Console")
                {
                    X = 0,
                    Y = 0,
                    Width = Dim.Fill(),
                    Height = Dim.Fill()
                };

                _statusLabel = new Label("Status: Initializing...")
                {
                    X = 1,
                    Y = 1,
                    Width = Dim.Fill() - 2
                };

                _progressBar = new ProgressBar()
                {
                    X = 1,
                    Y = 2,
                    Width = Dim.Fill() - 2,
                    Fraction = 0f
                };

                _logView = new TextView()
                {
                    X = 1,
                    Y = 4,
                    Width = Dim.Fill() - 2,
                    Height = Dim.Fill() - 1, // Leave space for border
                    ReadOnly = true,
                    ColorScheme = Colors.Base,
                    Text = "Deep3DStudio Log Started...\n"
                };

                _window.Add(_statusLabel, _progressBar, new Label(1, 3, "Logs:"), _logView);
                Application.Top.Add(_window);

                Application.Run();
            }
            catch (Exception)
            {
                // If TUI fails to init (e.g. no console), just fail silently
                // and let the app continue
                _isRunning = false;
            }
        }

        public void Stop()
        {
            if (!_isRunning) return;
            _isRunning = false;
            
            // Restore console? Maybe not needed if app is exiting
            // But we should stop the TUI loop
            Application.RequestStop();
        }

        public void Log(string message)
        {
            if (!_isRunning || _logView == null) return;

            Application.MainLoop.Invoke(() =>
            {
                if (_logView != null)
                {
                    // Append text. 
                    // Note: TextView optimization might be needed for huge logs, 
                    // but for now simple appending is fine.
                    var currentText = _logView.Text.ToString();
                    
                    // Simple buffer limiting
                    if (currentText.Length > 20000) 
                        currentText = currentText.Substring(currentText.Length - 10000);
                        
                    _logView.Text = currentText + message;
                    
                    // Auto-scroll attempt
                    _logView.ScrollTo(_logView.Lines - 1);
                }
            });
        }

        public void UpdateProgress(string task, float progress)
        {
            if (!_isRunning) return;

            Application.MainLoop.Invoke(() =>
            {
                if (_statusLabel != null) _statusLabel.Text = $"Status: {task}";
                if (_progressBar != null) _progressBar.Fraction = Math.Clamp(progress, 0f, 1f);
            });
        }

        // Custom TextWriter to redirect Console.Out/Error
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
        }
    }
}
