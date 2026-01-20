using System;
using System.Text;
using System.Threading;
using Gtk;

namespace Deep3DStudio.UI
{
    public enum OperationType
    {
        ImportExport,
        Processing,
        Extraction
    }

    public enum ProgressState
    {
        Idle,
        Running,
        Success,
        Error,
        Cancelled
    }

    /// <summary>
    /// Singleton progress dialog for GTK version.
    /// Thread-safe and can be updated from background threads via Application.Invoke.
    /// </summary>
    public class ProgressDialog
    {
        private static ProgressDialog? _instance;
        public static ProgressDialog Instance => _instance ??= new ProgressDialog();

        // UI elements
        private Window? _window;
        private ProgressBar? _progressBar;
        private Label? _statusLabel;
        private TextView? _logView;
        private Expander? _detailsExpander;
        private Button? _actionButton;

        // State
        private readonly object _lock = new object();
        private bool _isVisible;
        private string _title = "Processing";
        private float _progress;
        private string _statusText = "";
        private OperationType _opType;
        private ProgressState _state = ProgressState.Idle;
        private StringBuilder _logBuffer = new StringBuilder();
        private string _errorMessage = "";
        private Window? _parentWindow;

        public bool IsVisible { get { lock (_lock) return _isVisible; } }
        public ProgressState State { get { lock (_lock) return _state; } }
        public CancellationTokenSource? CancellationTokenSource { get; private set; }

        private ProgressDialog() { }

        /// <summary>
        /// Sets the parent window for dialog positioning.
        /// </summary>
        public void SetParent(Window parent)
        {
            _parentWindow = parent;
        }

        /// <summary>
        /// Starts a new progress operation and shows the dialog.
        /// Must be called from the main thread or via Application.Invoke.
        /// </summary>
        public void Start(string title, OperationType type)
        {
            lock (_lock)
            {
                _title = title;
                _opType = type;
                _state = ProgressState.Running;
                _progress = 0.0f;
                _statusText = "Starting...";
                _logBuffer.Clear();
                _errorMessage = "";
            }

            CancellationTokenSource = new CancellationTokenSource();

            // Create/show window on main thread
            Application.Invoke((s, e) => CreateAndShowWindow());
        }

        /// <summary>
        /// Updates the progress and status text.
        /// Thread-safe - can be called from any thread.
        /// </summary>
        public void Update(float progress, string status)
        {
            lock (_lock)
            {
                _progress = Math.Clamp(progress, 0f, 1f);
                _statusText = status;
                _logBuffer.AppendLine(status);
            }

            // Update UI on main thread
            Application.Invoke((s, e) => UpdateUI());
        }

        /// <summary>
        /// Logs a message without changing progress.
        /// Thread-safe - can be called from any thread.
        /// </summary>
        public void Log(string message)
        {
            lock (_lock)
            {
                _logBuffer.AppendLine(message);
            }

            Application.Invoke((s, e) => UpdateLogView());
        }

        /// <summary>
        /// Marks the operation as complete.
        /// Thread-safe - can be called from any thread.
        /// </summary>
        public void Complete()
        {
            lock (_lock)
            {
                _state = ProgressState.Success;
                _progress = 1.0f;
                _statusText = "Operation completed successfully";
            }

            Application.Invoke((s, e) =>
            {
                UpdateUI();

                // Auto-close for import/export operations
                if (_opType == OperationType.ImportExport)
                {
                    Close();
                }
            });
        }

        /// <summary>
        /// Marks the operation as failed.
        /// Thread-safe - can be called from any thread.
        /// </summary>
        public void Fail(Exception ex)
        {
            lock (_lock)
            {
                _state = ProgressState.Error;
                _errorMessage = ex.Message;
                _statusText = "Error occurred";
                _logBuffer.AppendLine($"ERROR: {ex.Message}");
                if (ex.StackTrace != null)
                    _logBuffer.AppendLine(ex.StackTrace);
            }

            Application.Invoke((s, e) =>
            {
                UpdateUI();
                // Auto-expand details on error
                if (_detailsExpander != null)
                    _detailsExpander.Expanded = true;
            });
        }

        /// <summary>
        /// Closes the dialog.
        /// Thread-safe - can be called from any thread.
        /// </summary>
        public void Close()
        {
            Application.Invoke((s, e) =>
            {
                lock (_lock)
                {
                    _isVisible = false;
                    _state = ProgressState.Idle;
                }

                CancellationTokenSource?.Dispose();
                CancellationTokenSource = null;

                if (_window != null)
                {
                    _window.Destroy();
                    _window = null;
                }
            });
        }

        private void CreateAndShowWindow()
        {
            // Close existing window if any
            if (_window != null)
            {
                _window.Destroy();
                _window = null;
            }

            string title;
            lock (_lock)
            {
                title = _title;
                _isVisible = true;
            }

            _window = new Window(WindowType.Toplevel);
            _window.Title = title;
            _window.SetDefaultSize(500, 200);
            _window.BorderWidth = 15;
            _window.Resizable = true;
            _window.WindowPosition = WindowPosition.Center;

            if (_parentWindow != null)
            {
                _window.TransientFor = _parentWindow;
                _window.Modal = false; // Allow user to interact with main window
            }

            // Prevent closing via window manager X button during operation
            _window.DeleteEvent += (s, e) =>
            {
                lock (_lock)
                {
                    if (_state == ProgressState.Running)
                    {
                        // Don't close during operation, just hide
                        e.RetVal = true;
                    }
                    else
                    {
                        Close();
                        e.RetVal = false;
                    }
                }
            };

            var vbox = new Box(Orientation.Vertical, 10);

            // Progress bar
            _progressBar = new ProgressBar();
            _progressBar.ShowText = true;
            vbox.PackStart(_progressBar, false, false, 0);

            // Status label
            _statusLabel = new Label();
            _statusLabel.LineWrap = true;
            _statusLabel.Halign = Align.Start;
            vbox.PackStart(_statusLabel, false, false, 0);

            // Details expander with log view
            _detailsExpander = new Expander("Details");
            _detailsExpander.Expanded = false;

            var scrolledWindow = new ScrolledWindow();
            scrolledWindow.SetSizeRequest(-1, 200);
            scrolledWindow.ShadowType = ShadowType.In;

            _logView = new TextView();
            _logView.Editable = false;
            _logView.CursorVisible = false;
            _logView.WrapMode = WrapMode.WordChar;
            // Use monospace font for log
            var fontDesc = Pango.FontDescription.FromString("Monospace 9");
            _logView.OverrideFont(fontDesc);

            scrolledWindow.Add(_logView);
            _detailsExpander.Add(scrolledWindow);
            vbox.PackStart(_detailsExpander, true, true, 0);

            // Button box
            var buttonBox = new Box(Orientation.Horizontal, 5);
            buttonBox.Halign = Align.End;

            _actionButton = new Button();
            _actionButton.Clicked += OnActionButtonClicked;
            buttonBox.PackEnd(_actionButton, false, false, 0);

            vbox.PackStart(buttonBox, false, false, 0);

            _window.Add(vbox);
            _window.ShowAll();

            UpdateUI();
        }

        private void UpdateUI()
        {
            if (_window == null || _progressBar == null || _statusLabel == null || _actionButton == null)
                return;

            ProgressState state;
            float progress;
            string statusText;
            string errorMessage;
            string title;

            lock (_lock)
            {
                state = _state;
                progress = _progress;
                statusText = _statusText;
                errorMessage = _errorMessage;
                title = _title;
            }

            _window.Title = title;
            _progressBar.Fraction = progress;
            _progressBar.Text = $"{statusText} ({(int)(progress * 100)}%)";

            if (state == ProgressState.Error)
            {
                _statusLabel.Markup = $"<span foreground='red'><b>Error:</b> {GLib.Markup.EscapeText(errorMessage)}</span>";
                _actionButton.Label = "OK";
                _actionButton.Sensitive = true;
            }
            else if (state == ProgressState.Success)
            {
                _statusLabel.Text = statusText;
                _actionButton.Label = "OK";
                _actionButton.Sensitive = true;
            }
            else if (state == ProgressState.Cancelled)
            {
                _statusLabel.Text = "Operation cancelled";
                _actionButton.Label = "OK";
                _actionButton.Sensitive = true;
            }
            else // Running
            {
                _statusLabel.Text = statusText;
                _actionButton.Label = "Cancel";
                _actionButton.Sensitive = true;
            }

            UpdateLogView();
        }

        private void UpdateLogView()
        {
            if (_logView == null) return;

            string logText;
            lock (_lock)
            {
                logText = _logBuffer.ToString();
            }

            _logView.Buffer.Text = logText;

            // Auto-scroll to end
            var endIter = _logView.Buffer.EndIter;
            _logView.Buffer.PlaceCursor(endIter);
            _logView.ScrollToIter(endIter, 0.0, true, 0.0, 1.0);
        }

        private void OnActionButtonClicked(object? sender, EventArgs e)
        {
            ProgressState state;
            lock (_lock)
            {
                state = _state;
            }

            if (state == ProgressState.Running)
            {
                // Cancel
                lock (_lock)
                {
                    _state = ProgressState.Cancelled;
                    _statusText = "Cancelling...";
                }
                CancellationTokenSource?.Cancel();
                UpdateUI();
            }
            else
            {
                // Close
                Close();
            }
        }
    }
}
