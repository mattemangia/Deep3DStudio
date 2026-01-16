using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public partial class MainWindow : Gtk.Window
    {
        private ThreeDView _viewport;
        private Label _statusLabel;
        private Model.Dust3rInference _inference;
        private List<string> _imagePaths = new List<string>();
        private ImageBrowserPanel _imageBrowser;
        private SceneResult? _lastSceneResult;

        // Scene Management
        private SceneGraph _sceneGraph;
        private SceneTreeView _sceneTreeView;

        // Rigging
        private RiggingPanel? _riggingPanel;
        private SkeletonObject? _activeSkeletonObject;
        private Box? _rightPanel;

        private bool _isDirty = false;

        // UI References for updates
        private ComboBoxText _workflowCombo = null!;
        private ToggleToolButton _pointsToggle = null!;
        private ToggleToolButton _wireToggle = null!;
        private ToggleToolButton _textureToggle = null!;
        private ToggleToolButton _meshToggle = null!;
        private ToggleToolButton _camerasToggle = null!;
        private ToggleToolButton _rgbColorToggle = null!;
        private ToggleToolButton _depthColorToggle = null!;
        private ToggleToolButton _autoWorkflowToggle = null!;

        // Auto Workflow Toggle - when enabled, Play button runs the full selected workflow
        // When disabled, user can manually trigger each step (e.g., Dust3R -> then LGM -> then UniRig)
        private bool _autoWorkflowEnabled = true;

        // Panel containers for show/hide
        private Widget _leftPanel = null!;
        private Widget _verticalToolbar = null!;
        private Paned _mainHPaned = null!;

        // Menu check items for panel visibility
        private CheckMenuItem? _showSceneTreeMenuItem;
        private CheckMenuItem? _showVerticalToolbarMenuItem;
        private CheckMenuItem? _showGridMenuItem;
        private CheckMenuItem? _showAxesMenuItem;
        private CheckMenuItem? _showCamerasMenuItem;
        private CheckMenuItem? _showInfoMenuItem;

        // Gizmo menu items
        private RadioMenuItem? _translateMenuItem;
        private RadioMenuItem? _rotateMenuItem;
        private RadioMenuItem? _scaleMenuItem;

        public MainWindow() : base(Gtk.WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio";

            // Restore window size from settings
            var settings = IniSettings.Instance;
            this.SetDefaultSize(settings.LastWindowWidth, settings.LastWindowHeight);

            // Set window icon explicitly (in addition to default icon list)
            try
            {
                this.Icon = ApplicationIcon.Create(64);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not set window icon: {ex.Message}");
            }

            // Handle window close - save settings and quit
            this.DeleteEvent += OnWindowDelete;

            // Initialize Scene Graph
            _sceneGraph = new SceneGraph();
            _sceneGraph.SceneChanged += (s, e) => { _isDirty = true; UpdateTitle(); };

            // Create viewport early (needed by menu bar)
            _viewport = new ThreeDView();
            _viewport.SetSceneGraph(_sceneGraph);
            _viewport.ObjectPicked += OnViewportObjectPicked;
            // Ensure viewport has minimum size
            _viewport.SetSizeRequest(400, 300);

            var mainVBox = new Box(Orientation.Vertical, 0);
            mainVBox.Visible = true;
            this.Add(mainVBox);

            Console.WriteLine("MainWindow: Creating UI components...");

            // 1. Menu Bar
            var menuBar = CreateMenuBar();
            menuBar.Visible = true;
            menuBar.SetSizeRequest(-1, 25); // Minimum height
            mainVBox.PackStart(menuBar, false, false, 0);
            Console.WriteLine("MainWindow: Menu bar created");

            // 2. Toolbar (Top)
            var toolbar = CreateToolbar();
            toolbar.Visible = true;
            toolbar.SetSizeRequest(-1, 35); // Minimum height
            mainVBox.PackStart(toolbar, false, false, 0);
            Console.WriteLine("MainWindow: Toolbar created");

            // 3. Main Content Area
            var contentBox = new Box(Orientation.Horizontal, 0);
            mainVBox.PackStart(contentBox, true, true, 0);

            // Vertical Toolbar (Left of viewport)
            _verticalToolbar = CreateVerticalToolbar();
            contentBox.PackStart(_verticalToolbar, false, false, 0);

            // Main Paned (Tree + Viewport)
            _mainHPaned = new Paned(Orientation.Horizontal);
            contentBox.PackStart(_mainHPaned, true, true, 0);

            // Left: Scene Tree View + Image Browser
            _leftPanel = CreateSceneTreePanel();
            _mainHPaned.Pack1(_leftPanel, false, false);

            // 3D Viewport
            _mainHPaned.Pack2(_viewport, true, false);

            // Status Bar
            _statusLabel = new Label("Ready");

            // Initialize inference engine
            _inference = new Model.Dust3rInference();

            // Hook up AI model loading progress to status bar
            AIModels.AIModelManager.Instance.ModelLoadProgress += (stage, progress, message) => {
                Application.Invoke((s, e) => {
                    _statusLabel.Text = $"[{(int)(progress * 100)}%] {message}";
                });
            };

            _mainHPaned.Position = settings.LastPanelWidth;

            _statusLabel.Halign = Align.Start;
            _statusLabel.Visible = true;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.Visible = true;
            statusBox.SetSizeRequest(-1, 25); // Minimum height
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);
            Console.WriteLine("MainWindow: Status bar created");

            // Load Initial Settings
            ApplyViewSettings();

            // Enable Drag and Drop
            Gtk.Drag.DestSet(this, DestDefaults.All, new TargetEntry[] { new TargetEntry("text/uri-list", 0, 0) }, Gdk.DragAction.Copy);
            this.DragDataReceived += OnDragDataReceived;

            Console.WriteLine("MainWindow: Calling ShowAll()");
            this.ShowAll();
            Console.WriteLine("MainWindow: ShowAll() completed");
        }

        /// <summary>
        /// Handles window close event - saves settings and quits application.
        /// </summary>
        private void OnWindowDelete(object o, DeleteEventArgs args)
        {
            if (!CheckSaveChanges())
            {
                args.RetVal = true; // Cancel close
                return;
            }
            SaveWindowState();
            Application.Quit();
        }

        private void UpdateTitle()
        {
            string title = "Deep3D Studio";
            if (_isDirty) title += " *";
            this.Title = title;
        }

        /// <summary>
        /// Saves the current window state to settings.
        /// </summary>
        private void SaveWindowState()
        {
            try
            {
                var settings = IniSettings.Instance;

                // Get window size
                this.GetSize(out int width, out int height);
                settings.LastWindowWidth = width;
                settings.LastWindowHeight = height;

                // Get panel position
                if (_mainHPaned != null)
                {
                    settings.LastPanelWidth = _mainHPaned.Position;
                }

                // Save viewport visibility settings
                settings.ShowGrid = _viewport?.ShowGrid ?? true;
                settings.ShowAxes = _viewport?.ShowAxes ?? true;
                settings.ShowCameras = _viewport?.ShowCameras ?? true;
                settings.ShowInfoOverlay = _viewport?.ShowInfoText ?? true;

                // Save to INI file
                settings.Save();
                Console.WriteLine("Window state saved to settings.ini");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving window state: {ex.Message}");
            }
        }
    }
}
