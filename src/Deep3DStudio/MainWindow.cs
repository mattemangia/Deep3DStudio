using System;
using Gtk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

namespace Deep3DStudio
{
    public class MainWindow : Window
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

        // UI References for updates
        private ComboBoxText _workflowCombo;
        private ToggleToolButton _pointsToggle;
        private ToggleToolButton _wireToggle;
        private ToggleToolButton _meshToggle;
        private ToggleToolButton _rgbColorToggle;
        private ToggleToolButton _depthColorToggle;

        // Panel containers for show/hide
        private Widget _leftPanel;
        private Widget _rightPanel;
        private Widget _verticalToolbar;
        private Paned _mainHPaned;
        private Paned _rightPaned;

        // Menu check items for panel visibility
        private CheckMenuItem? _showSceneTreeMenuItem;
        private CheckMenuItem? _showToolsPanelMenuItem;
        private CheckMenuItem? _showVerticalToolbarMenuItem;
        private CheckMenuItem? _showGridMenuItem;
        private CheckMenuItem? _showAxesMenuItem;
        private CheckMenuItem? _showCamerasMenuItem;
        private CheckMenuItem? _showInfoMenuItem;

        // Gizmo menu items
        private RadioMenuItem? _translateMenuItem;
        private RadioMenuItem? _rotateMenuItem;
        private RadioMenuItem? _scaleMenuItem;

        public MainWindow() : base(WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio";
            this.SetDefaultSize(1400, 900);
            this.DeleteEvent += (o, args) => Application.Quit();

            // Initialize Scene Graph
            _sceneGraph = new SceneGraph();

            // Create viewport early (needed by menu bar)
            _viewport = new ThreeDView();
            _viewport.SetSceneGraph(_sceneGraph);
            _viewport.ObjectPicked += OnViewportObjectPicked;

            var mainVBox = new Box(Orientation.Vertical, 0);
            this.Add(mainVBox);

            // 1. Menu Bar
            var menuBar = CreateMenuBar();
            mainVBox.PackStart(menuBar, false, false, 0);

            // 2. Toolbar (Top)
            var toolbar = CreateToolbar();
            mainVBox.PackStart(toolbar, false, false, 0);

            // 3. Main Content Area
            var contentBox = new Box(Orientation.Horizontal, 0);
            mainVBox.PackStart(contentBox, true, true, 0);

            // Vertical Toolbar (Left of viewport)
            _verticalToolbar = CreateVerticalToolbar();
            contentBox.PackStart(_verticalToolbar, false, false, 0);

            // Main Paned (Tree + Viewport + Side Panel)
            _mainHPaned = new Paned(Orientation.Horizontal);
            contentBox.PackStart(_mainHPaned, true, true, 0);

            // Left: Scene Tree View
            _leftPanel = CreateSceneTreePanel();
            _mainHPaned.Pack1(_leftPanel, false, false);

            // Center + Right: Viewport + Side Panel
            _rightPaned = new Paned(Orientation.Horizontal);
            _mainHPaned.Pack2(_rightPaned, true, false);

            // 3D Viewport (already created above, just add to layout)
            _rightPaned.Pack1(_viewport, true, false);

            // 4. Status Bar (initialize before CreateSidePanel)
            _statusLabel = new Label("Ready");

            // Side Panel (Right)
            _rightPanel = CreateSidePanel();
            _rightPaned.Pack2(_rightPanel, false, false);

            _rightPaned.Position = 850;
            _mainHPaned.Position = 250;

            _statusLabel.Halign = Align.Start;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);

            // Load Initial Settings
            ApplyViewSettings();

            this.ShowAll();
        }

        #region Menu Bar

        private Widget CreateMenuBar()
        {
            var menuBar = new MenuBar();

            // File Menu
            var fileMenu = new Menu();
            var fileMenuItem = new MenuItem("_File");
            fileMenuItem.Submenu = fileMenu;

            var openImagesItem = new MenuItem("_Open Images...");
            openImagesItem.Activated += OnAddImages;
            fileMenu.Append(openImagesItem);

            var importMeshItem = new MenuItem("_Import Mesh...");
            importMeshItem.Activated += OnImportMesh;
            fileMenu.Append(importMeshItem);

            fileMenu.Append(new SeparatorMenuItem());

            var exportMeshItem = new MenuItem("_Export Mesh...");
            exportMeshItem.Activated += OnExportMesh;
            fileMenu.Append(exportMeshItem);

            fileMenu.Append(new SeparatorMenuItem());

            var settingsItem = new MenuItem("_Settings...");
            settingsItem.Activated += OnOpenSettings;
            fileMenu.Append(settingsItem);

            fileMenu.Append(new SeparatorMenuItem());

            var exitItem = new MenuItem("E_xit");
            exitItem.Activated += (s, e) => Application.Quit();
            fileMenu.Append(exitItem);

            menuBar.Append(fileMenuItem);

            // Edit Menu
            var editMenu = new Menu();
            var editMenuItem = new MenuItem("_Edit");
            editMenuItem.Submenu = editMenu;

            var selectAllItem = new MenuItem("Select _All");
            selectAllItem.Activated += (s, e) => _sceneGraph.SelectAll();
            editMenu.Append(selectAllItem);

            var deselectAllItem = new MenuItem("_Deselect All");
            deselectAllItem.Activated += (s, e) => _sceneGraph.ClearSelection();
            editMenu.Append(deselectAllItem);

            editMenu.Append(new SeparatorMenuItem());

            var deleteItem = new MenuItem("_Delete");
            deleteItem.Activated += OnDeleteSelected;
            editMenu.Append(deleteItem);

            var duplicateItem = new MenuItem("D_uplicate");
            duplicateItem.Activated += OnDuplicateSelected;
            editMenu.Append(duplicateItem);

            editMenu.Append(new SeparatorMenuItem());

            // Transform submenu
            var transformMenu = new Menu();
            var transformMenuItem = new MenuItem("_Transform");
            transformMenuItem.Submenu = transformMenu;

            _translateMenuItem = new RadioMenuItem("_Move (W)");
            _translateMenuItem.Active = true;
            _translateMenuItem.Toggled += (s, e) => {
                if (_translateMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Translate);
            };
            transformMenu.Append(_translateMenuItem);

            _rotateMenuItem = new RadioMenuItem(_translateMenuItem, "_Rotate (E)");
            _rotateMenuItem.Toggled += (s, e) => {
                if (_rotateMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Rotate);
            };
            transformMenu.Append(_rotateMenuItem);

            _scaleMenuItem = new RadioMenuItem(_translateMenuItem, "_Scale (R)");
            _scaleMenuItem.Toggled += (s, e) => {
                if (_scaleMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Scale);
            };
            transformMenu.Append(_scaleMenuItem);

            transformMenu.Append(new SeparatorMenuItem());

            var resetTransformItem = new MenuItem("Reset Transform");
            resetTransformItem.Activated += OnResetTransform;
            transformMenu.Append(resetTransformItem);

            editMenu.Append(transformMenuItem);

            // Mesh Operations submenu
            var meshOpsMenu = new Menu();
            var meshOpsMenuItem = new MenuItem("_Mesh Operations");
            meshOpsMenuItem.Submenu = meshOpsMenu;

            var decimateItem = new MenuItem("_Decimate (50%)");
            decimateItem.Activated += OnDecimateClicked;
            meshOpsMenu.Append(decimateItem);

            var smoothItem = new MenuItem("_Smooth");
            smoothItem.Activated += OnSmoothClicked;
            meshOpsMenu.Append(smoothItem);

            var optimizeItem = new MenuItem("_Optimize");
            optimizeItem.Activated += OnOptimizeClicked;
            meshOpsMenu.Append(optimizeItem);

            var splitItem = new MenuItem("Split by _Connectivity");
            splitItem.Activated += OnSplitClicked;
            meshOpsMenu.Append(splitItem);

            var flipNormalsItem = new MenuItem("_Flip Normals");
            flipNormalsItem.Activated += OnFlipNormals;
            meshOpsMenu.Append(flipNormalsItem);

            meshOpsMenu.Append(new SeparatorMenuItem());

            var mergeItem = new MenuItem("_Merge Selected");
            mergeItem.Activated += OnMergeClicked;
            meshOpsMenu.Append(mergeItem);

            var alignItem = new MenuItem("_Align (ICP)");
            alignItem.Activated += OnAlignClicked;
            meshOpsMenu.Append(alignItem);

            editMenu.Append(meshOpsMenuItem);

            menuBar.Append(editMenuItem);

            // View Menu
            var viewMenu = new Menu();
            var viewMenuItem = new MenuItem("_View");
            viewMenuItem.Submenu = viewMenu;

            // Display mode
            var meshModeItem = new RadioMenuItem("Show _Mesh");
            meshModeItem.Active = true;
            meshModeItem.Toggled += (s, e) => {
                if (meshModeItem.Active)
                {
                    AppSettings.Instance.ShowPointCloud = false;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(meshModeItem);

            var pointsModeItem = new RadioMenuItem(meshModeItem, "Show _Points");
            pointsModeItem.Toggled += (s, e) => {
                if (pointsModeItem.Active)
                {
                    AppSettings.Instance.ShowPointCloud = true;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(pointsModeItem);

            var wireframeItem = new CheckMenuItem("_Wireframe");
            wireframeItem.Active = AppSettings.Instance.ShowWireframe;
            wireframeItem.Toggled += (s, e) => {
                AppSettings.Instance.ShowWireframe = wireframeItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(wireframeItem);

            viewMenu.Append(new SeparatorMenuItem());

            // Color mode
            var rgbColorItem = new RadioMenuItem("_RGB Colors");
            rgbColorItem.Active = AppSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            rgbColorItem.Toggled += (s, e) => {
                if (rgbColorItem.Active)
                {
                    AppSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(rgbColorItem);

            var depthColorItem = new RadioMenuItem(rgbColorItem, "_Depth Colors");
            depthColorItem.Active = AppSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            depthColorItem.Toggled += (s, e) => {
                if (depthColorItem.Active)
                {
                    AppSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(depthColorItem);

            viewMenu.Append(new SeparatorMenuItem());

            // Viewport elements
            _showGridMenuItem = new CheckMenuItem("Show _Grid");
            _showGridMenuItem.Active = _viewport.ShowGrid;
            _showGridMenuItem.Toggled += (s, e) => {
                _viewport.ShowGrid = _showGridMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showGridMenuItem);

            _showAxesMenuItem = new CheckMenuItem("Show _Axes");
            _showAxesMenuItem.Active = _viewport.ShowAxes;
            _showAxesMenuItem.Toggled += (s, e) => {
                _viewport.ShowAxes = _showAxesMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showAxesMenuItem);

            _showCamerasMenuItem = new CheckMenuItem("Show _Cameras");
            _showCamerasMenuItem.Active = _viewport.ShowCameras;
            _showCamerasMenuItem.Toggled += (s, e) => {
                _viewport.ShowCameras = _showCamerasMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showCamerasMenuItem);

            _showInfoMenuItem = new CheckMenuItem("Show _Info Overlay");
            _showInfoMenuItem.Active = _viewport.ShowInfoText;
            _showInfoMenuItem.Toggled += (s, e) => {
                _viewport.ShowInfoText = _showInfoMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showInfoMenuItem);

            viewMenu.Append(new SeparatorMenuItem());

            var focusItem = new MenuItem("_Focus on Selection (F)");
            focusItem.Activated += (s, e) => _viewport.FocusOnSelection();
            viewMenu.Append(focusItem);

            var resetViewItem = new MenuItem("_Reset View");
            resetViewItem.Activated += (s, e) => _viewport.FocusOnSelection();
            viewMenu.Append(resetViewItem);

            menuBar.Append(viewMenuItem);

            // Window Menu
            var windowMenu = new Menu();
            var windowMenuItem = new MenuItem("_Window");
            windowMenuItem.Submenu = windowMenu;

            _showSceneTreeMenuItem = new CheckMenuItem("_Scene Tree");
            _showSceneTreeMenuItem.Active = true;
            _showSceneTreeMenuItem.Toggled += OnToggleSceneTree;
            windowMenu.Append(_showSceneTreeMenuItem);

            _showToolsPanelMenuItem = new CheckMenuItem("_Tools Panel");
            _showToolsPanelMenuItem.Active = true;
            _showToolsPanelMenuItem.Toggled += OnToggleToolsPanel;
            windowMenu.Append(_showToolsPanelMenuItem);

            _showVerticalToolbarMenuItem = new CheckMenuItem("_Vertical Toolbar");
            _showVerticalToolbarMenuItem.Active = true;
            _showVerticalToolbarMenuItem.Toggled += OnToggleVerticalToolbar;
            windowMenu.Append(_showVerticalToolbarMenuItem);

            windowMenu.Append(new SeparatorMenuItem());

            var fullViewportItem = new MenuItem("_Full Viewport Mode");
            fullViewportItem.Activated += OnFullViewportMode;
            windowMenu.Append(fullViewportItem);

            var restorePanelsItem = new MenuItem("_Restore All Panels");
            restorePanelsItem.Activated += OnRestoreAllPanels;
            windowMenu.Append(restorePanelsItem);

            menuBar.Append(windowMenuItem);

            // Help Menu
            var helpMenu = new Menu();
            var helpMenuItem = new MenuItem("_Help");
            helpMenuItem.Submenu = helpMenu;

            var aboutItem = new MenuItem("_About");
            aboutItem.Activated += OnShowAbout;
            helpMenu.Append(aboutItem);

            menuBar.Append(helpMenuItem);

            return menuBar;
        }

        #endregion

        #region Vertical Toolbar

        private Widget CreateVerticalToolbar()
        {
            var vbox = new Box(Orientation.Vertical, 2);
            vbox.MarginStart = 2;
            vbox.MarginEnd = 2;
            vbox.MarginTop = 5;

            int btnSize = 36;

            // Transform tools section
            var moveBtn = CreateIconButton("move", "Move (W)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Translate));
            vbox.PackStart(moveBtn, false, false, 1);

            var rotateBtn = CreateIconButton("rotate", "Rotate (E)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Rotate));
            vbox.PackStart(rotateBtn, false, false, 1);

            var scaleBtn = CreateIconButton("scale", "Scale (R)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Scale));
            vbox.PackStart(scaleBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // View tools
            var focusBtn = CreateIconButton("focus", "Focus (F)", btnSize, () => _viewport.FocusOnSelection());
            vbox.PackStart(focusBtn, false, false, 1);

            var cropBtn = CreateIconButton("crop", "Crop Box", btnSize, () => _viewport.ToggleCropBox(true));
            vbox.PackStart(cropBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Mesh operations
            var decimateBtn = CreateIconButton("decimate", "Decimate", btnSize, () => OnDecimateClicked(null, EventArgs.Empty));
            vbox.PackStart(decimateBtn, false, false, 1);

            var smoothBtn = CreateIconButton("smooth", "Smooth", btnSize, () => OnSmoothClicked(null, EventArgs.Empty));
            vbox.PackStart(smoothBtn, false, false, 1);

            var optimizeBtn = CreateIconButton("optimize", "Optimize", btnSize, () => OnOptimizeClicked(null, EventArgs.Empty));
            vbox.PackStart(optimizeBtn, false, false, 1);

            var splitBtn = CreateIconButton("split", "Split", btnSize, () => OnSplitClicked(null, EventArgs.Empty));
            vbox.PackStart(splitBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Merge/Align
            var mergeBtn = CreateIconButton("merge", "Merge", btnSize, () => OnMergeClicked(null, EventArgs.Empty));
            vbox.PackStart(mergeBtn, false, false, 1);

            var alignBtn = CreateIconButton("align", "Align (ICP)", btnSize, () => OnAlignClicked(null, EventArgs.Empty));
            vbox.PackStart(alignBtn, false, false, 1);

            return vbox;
        }

        private Button CreateIconButton(string iconType, string tooltip, int size, Action onClick)
        {
            var btn = new Button();
            btn.TooltipText = tooltip;
            btn.SetSizeRequest(size, size);
            btn.Relief = ReliefStyle.None;

            // Create custom icon
            var icon = CreateCustomIcon(iconType, size - 8);
            btn.Add(icon);

            btn.Clicked += (s, e) => onClick();

            return btn;
        }

        private Widget CreateCustomIcon(string iconType, int size)
        {
            // Create a DrawingArea with custom drawing for each icon type
            var drawingArea = new DrawingArea();
            drawingArea.SetSizeRequest(size, size);
            drawingArea.Drawn += (o, args) =>
            {
                var cr = args.Cr;
                DrawIconContent(cr, iconType, size);
            };
            return drawingArea;
        }

        private void DrawIconContent(Cairo.Context cr, string iconType, int size)
        {
            double s = size;
            double cx = s / 2;
            double cy = s / 2;

            cr.SetSourceRGB(0.8, 0.8, 0.8); // Light gray for icons

            switch (iconType)
            {
                case "move":
                    // Cross with arrows
                    cr.LineWidth = 2;
                    // Horizontal line
                    cr.MoveTo(2, cy);
                    cr.LineTo(s - 2, cy);
                    cr.Stroke();
                    // Vertical line
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s - 2);
                    cr.Stroke();
                    // Arrow heads
                    cr.MoveTo(s - 2, cy);
                    cr.LineTo(s - 6, cy - 3);
                    cr.LineTo(s - 6, cy + 3);
                    cr.ClosePath();
                    cr.Fill();
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx - 3, 6);
                    cr.LineTo(cx + 3, 6);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "rotate":
                    // Circular arrow
                    cr.LineWidth = 2;
                    cr.Arc(cx, cy, s / 3, 0.5, 5.5);
                    cr.Stroke();
                    // Arrow head
                    double angle = 5.5;
                    double ax = cx + Math.Cos(angle) * s / 3;
                    double ay = cy + Math.Sin(angle) * s / 3;
                    cr.MoveTo(ax, ay);
                    cr.LineTo(ax + 4, ay - 2);
                    cr.LineTo(ax + 2, ay + 4);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "scale":
                    // Box with diagonal arrow
                    cr.LineWidth = 1.5;
                    cr.Rectangle(4, 4, s / 2, s / 2);
                    cr.Stroke();
                    cr.MoveTo(s / 2, s / 2);
                    cr.LineTo(s - 4, s - 4);
                    cr.Stroke();
                    // Arrow head
                    cr.MoveTo(s - 4, s - 4);
                    cr.LineTo(s - 8, s - 4);
                    cr.LineTo(s - 4, s - 8);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "focus":
                    // Target/crosshair
                    cr.LineWidth = 1.5;
                    cr.Arc(cx, cy, s / 3, 0, 2 * Math.PI);
                    cr.Stroke();
                    cr.Arc(cx, cy, s / 6, 0, 2 * Math.PI);
                    cr.Fill();
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s / 3);
                    cr.Stroke();
                    cr.MoveTo(cx, s - 2);
                    cr.LineTo(cx, s - s / 3);
                    cr.Stroke();
                    cr.MoveTo(2, cy);
                    cr.LineTo(s / 3, cy);
                    cr.Stroke();
                    cr.MoveTo(s - 2, cy);
                    cr.LineTo(s - s / 3, cy);
                    cr.Stroke();
                    break;

                case "crop":
                    // Crop corners
                    cr.LineWidth = 2;
                    // Top-left
                    cr.MoveTo(4, s / 3);
                    cr.LineTo(4, 4);
                    cr.LineTo(s / 3, 4);
                    cr.Stroke();
                    // Top-right
                    cr.MoveTo(s - s / 3, 4);
                    cr.LineTo(s - 4, 4);
                    cr.LineTo(s - 4, s / 3);
                    cr.Stroke();
                    // Bottom-left
                    cr.MoveTo(4, s - s / 3);
                    cr.LineTo(4, s - 4);
                    cr.LineTo(s / 3, s - 4);
                    cr.Stroke();
                    // Bottom-right
                    cr.MoveTo(s - s / 3, s - 4);
                    cr.LineTo(s - 4, s - 4);
                    cr.LineTo(s - 4, s - s / 3);
                    cr.Stroke();
                    break;

                case "decimate":
                    // Triangle with down arrow (simplify)
                    cr.LineWidth = 1.5;
                    cr.MoveTo(cx, 3);
                    cr.LineTo(s - 4, s - 6);
                    cr.LineTo(4, s - 6);
                    cr.ClosePath();
                    cr.Stroke();
                    // Down arrow
                    cr.SetSourceRGB(0.9, 0.5, 0.3);
                    cr.MoveTo(cx - 4, s - 10);
                    cr.LineTo(cx + 4, s - 10);
                    cr.LineTo(cx, s - 3);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "smooth":
                    // Wavy line becoming smooth
                    cr.LineWidth = 2;
                    cr.MoveTo(3, cy);
                    cr.CurveTo(s / 4, cy - 6, s / 2, cy + 6, s - 3, cy);
                    cr.Stroke();
                    break;

                case "optimize":
                    // Checkmark in circle
                    cr.LineWidth = 1.5;
                    cr.Arc(cx, cy, s / 3, 0, 2 * Math.PI);
                    cr.Stroke();
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 2;
                    cr.MoveTo(cx - 5, cy);
                    cr.LineTo(cx - 1, cy + 4);
                    cr.LineTo(cx + 5, cy - 4);
                    cr.Stroke();
                    break;

                case "split":
                    // Rectangle splitting into two
                    cr.LineWidth = 1.5;
                    cr.Rectangle(3, 4, s / 2 - 3, s - 8);
                    cr.Stroke();
                    cr.Rectangle(s / 2 + 1, 4, s / 2 - 4, s - 8);
                    cr.Stroke();
                    // Scissors/cut line
                    cr.SetSourceRGB(0.9, 0.4, 0.4);
                    cr.SetDash(new double[] { 2, 2 }, 0);
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s - 2);
                    cr.Stroke();
                    cr.SetDash(new double[] { }, 0);
                    break;

                case "merge":
                    // Two shapes merging
                    cr.LineWidth = 1.5;
                    cr.SetSourceRGB(0.6, 0.8, 1.0);
                    cr.Rectangle(3, 6, s / 2 - 2, s - 12);
                    cr.Fill();
                    cr.SetSourceRGB(1.0, 0.8, 0.6);
                    cr.Rectangle(s / 2 - 2, 6, s / 2 - 2, s - 12);
                    cr.Fill();
                    cr.SetSourceRGB(0.8, 0.8, 0.8);
                    cr.Rectangle(3, 6, s - 6, s - 12);
                    cr.Stroke();
                    // Arrow
                    cr.SetSourceRGB(0.3, 0.7, 0.3);
                    cr.MoveTo(s / 2 - 6, cy);
                    cr.LineTo(s / 2 + 6, cy);
                    cr.Stroke();
                    break;

                case "align":
                    // Two shapes with alignment arrows
                    cr.LineWidth = 1.5;
                    cr.Rectangle(4, 4, 8, 10);
                    cr.Stroke();
                    cr.Rectangle(s - 12, s - 14, 8, 10);
                    cr.Stroke();
                    // Alignment arrows
                    cr.SetSourceRGB(0.3, 0.6, 1.0);
                    cr.MoveTo(12, 9);
                    cr.LineTo(s - 12, s - 9);
                    cr.Stroke();
                    cr.MoveTo(s - 12, s - 9);
                    cr.LineTo(s - 16, s - 6);
                    cr.Stroke();
                    cr.MoveTo(s - 12, s - 9);
                    cr.LineTo(s - 9, s - 13);
                    cr.Stroke();
                    break;

                default:
                    // Default: simple square
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    break;
            }
        }

        #endregion

        #region Panel Management

        private Widget CreateSceneTreePanel()
        {
            var panel = new Box(Orientation.Vertical, 0);
            panel.SetSizeRequest(250, -1);

            _sceneTreeView = new SceneTreeView();
            _sceneTreeView.SetSceneGraph(_sceneGraph);
            _sceneTreeView.ObjectSelected += OnSceneObjectSelected;
            _sceneTreeView.ObjectDoubleClicked += OnSceneObjectDoubleClicked;
            _sceneTreeView.ObjectActionRequested += OnSceneObjectAction;
            panel.PackStart(_sceneTreeView, true, true, 0);

            return panel;
        }

        private void OnToggleSceneTree(object? sender, EventArgs e)
        {
            if (_showSceneTreeMenuItem != null)
            {
                _leftPanel.Visible = _showSceneTreeMenuItem.Active;
            }
        }

        private void OnToggleToolsPanel(object? sender, EventArgs e)
        {
            if (_showToolsPanelMenuItem != null)
            {
                _rightPanel.Visible = _showToolsPanelMenuItem.Active;
            }
        }

        private void OnToggleVerticalToolbar(object? sender, EventArgs e)
        {
            if (_showVerticalToolbarMenuItem != null)
            {
                _verticalToolbar.Visible = _showVerticalToolbarMenuItem.Active;
            }
        }

        private void OnFullViewportMode(object? sender, EventArgs e)
        {
            _leftPanel.Visible = false;
            _rightPanel.Visible = false;
            _verticalToolbar.Visible = false;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = false;
            if (_showToolsPanelMenuItem != null) _showToolsPanelMenuItem.Active = false;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = false;
        }

        private void OnRestoreAllPanels(object? sender, EventArgs e)
        {
            _leftPanel.Visible = true;
            _rightPanel.Visible = true;
            _verticalToolbar.Visible = true;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = true;
            if (_showToolsPanelMenuItem != null) _showToolsPanelMenuItem.Active = true;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = true;
        }

        #endregion

        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 24;

            // Open Files
            var openBtn = new ToolButton(AppIconFactory.GenerateIcon("open", iconSize), "Open Images");
            openBtn.TooltipText = "Load input images for reconstruction";
            openBtn.Clicked += OnAddImages;
            toolbar.Insert(openBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Settings
            var settingsBtn = new ToolButton(AppIconFactory.GenerateIcon("settings", iconSize), "Settings");
            settingsBtn.TooltipText = "Configure Processing, Meshing, and GPU";
            settingsBtn.Clicked += OnOpenSettings;
            toolbar.Insert(settingsBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // View Toggles
            _meshToggle = new ToggleToolButton();
            _meshToggle.IconWidget = AppIconFactory.GenerateIcon("mesh", iconSize);
            _meshToggle.Label = "Mesh";
            _meshToggle.TooltipText = "Show Solid Mesh";
            _meshToggle.Active = true;
            _meshToggle.Toggled += (s, e) => {
                 AppSettings.Instance.ShowPointCloud = !_meshToggle.Active;
                 _viewport.QueueDraw();
            };
            toolbar.Insert(_meshToggle, -1);

            _pointsToggle = new ToggleToolButton();
            _pointsToggle.IconWidget = AppIconFactory.GenerateIcon("pointcloud", iconSize);
            _pointsToggle.Label = "Points";
            _pointsToggle.TooltipText = "Show Point Cloud";
            _pointsToggle.Active = AppSettings.Instance.ShowPointCloud;
            _pointsToggle.Toggled += (s, e) => {
                AppSettings.Instance.ShowPointCloud = _pointsToggle.Active;
                _meshToggle.Active = !AppSettings.Instance.ShowPointCloud;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_pointsToggle, -1);

            _wireToggle = new ToggleToolButton();
            _wireToggle.IconWidget = AppIconFactory.GenerateIcon("wireframe", iconSize);
            _wireToggle.Label = "Wireframe";
            _wireToggle.TooltipText = "Toggle Wireframe Overlay";
            _wireToggle.Active = AppSettings.Instance.ShowWireframe;
            _wireToggle.Toggled += (s, e) => {
                AppSettings.Instance.ShowWireframe = _wireToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_wireToggle, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Point Cloud Color Mode Toggles
            _rgbColorToggle = new ToggleToolButton();
            _rgbColorToggle.IconWidget = AppIconFactory.GenerateIcon("rgb", iconSize);
            _rgbColorToggle.Label = "RGB";
            _rgbColorToggle.TooltipText = "Show original RGB colors";
            _rgbColorToggle.Active = AppSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            _rgbColorToggle.Toggled += (s, e) => {
                if (_rgbColorToggle.Active)
                {
                    AppSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
                    _depthColorToggle.Active = false;
                    _viewport.QueueDraw();
                }
                else if (!_depthColorToggle.Active)
                {
                    _rgbColorToggle.Active = true;
                }
            };
            toolbar.Insert(_rgbColorToggle, -1);

            _depthColorToggle = new ToggleToolButton();
            _depthColorToggle.IconWidget = AppIconFactory.GenerateIcon("depthmap", iconSize);
            _depthColorToggle.Label = "Depth";
            _depthColorToggle.TooltipText = "Show distance map with colormap";
            _depthColorToggle.Active = AppSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            _depthColorToggle.Toggled += (s, e) => {
                if (_depthColorToggle.Active)
                {
                    AppSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
                    _rgbColorToggle.Active = false;
                    _viewport.QueueDraw();
                }
                else if (!_rgbColorToggle.Active)
                {
                    _depthColorToggle.Active = true;
                }
            };
            toolbar.Insert(_depthColorToggle, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Workflow
            var wfItem = new ToolItem();
            var wfBox = new Box(Orientation.Horizontal, 5);
            wfBox.PackStart(new Label("Workflow: "), false, false, 0);
            _workflowCombo = new ComboBoxText();
            _workflowCombo.AppendText("Dust3r (Fast)");
            _workflowCombo.AppendText("NeRF (Refined)");
            _workflowCombo.Active = 0;
            wfBox.PackStart(_workflowCombo, false, false, 0);
            wfItem.Add(wfBox);
            toolbar.Insert(wfItem, -1);

            // Run
            var runBtn = new ToolButton(AppIconFactory.GenerateIcon("run", iconSize), "Run");
            runBtn.TooltipText = "Start Reconstruction Process";
            runBtn.Clicked += OnRunInference;
            toolbar.Insert(runBtn, -1);

            return toolbar;
        }

        private Widget CreateSidePanel()
        {
            var vbox = new Box(Orientation.Vertical, 5);
            vbox.Margin = 10;
            vbox.SetSizeRequest(280, -1);

            // Mesh Operations Section
            var meshOpsLabel = new Label("Mesh Operations");
            meshOpsLabel.Attributes = new Pango.AttrList();
            meshOpsLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(meshOpsLabel, false, false, 5);

            // Decimate button
            var decimateBtn = new Button("Decimate (50%)");
            decimateBtn.TooltipText = "Reduce mesh vertex count";
            decimateBtn.Clicked += OnDecimateClicked;
            vbox.PackStart(decimateBtn, false, false, 2);

            // Smooth button
            var smoothBtn = new Button("Smooth");
            smoothBtn.TooltipText = "Apply Laplacian smoothing";
            smoothBtn.Clicked += OnSmoothClicked;
            vbox.PackStart(smoothBtn, false, false, 2);

            // Optimize button
            var optimizeBtn = new Button("Optimize");
            optimizeBtn.TooltipText = "Remove duplicate vertices";
            optimizeBtn.Clicked += OnOptimizeClicked;
            vbox.PackStart(optimizeBtn, false, false, 2);

            // Split button
            var splitBtn = new Button("Split by Connectivity");
            splitBtn.TooltipText = "Split mesh into connected parts";
            splitBtn.Clicked += OnSplitClicked;
            vbox.PackStart(splitBtn, false, false, 2);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Merge/Align Section
            var alignLabel = new Label("Merge & Align");
            alignLabel.Attributes = new Pango.AttrList();
            alignLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(alignLabel, false, false, 5);

            var mergeBtn = new Button("Merge Selected");
            mergeBtn.TooltipText = "Merge selected meshes into one";
            mergeBtn.Clicked += OnMergeClicked;
            vbox.PackStart(mergeBtn, false, false, 2);

            var alignBtn = new Button("Align (ICP)");
            alignBtn.TooltipText = "Align selected to first using ICP";
            alignBtn.Clicked += OnAlignClicked;
            vbox.PackStart(alignBtn, false, false, 2);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Tools with 3D handles
            var toolsLabel = new Label("Crop Tools");
            toolsLabel.Attributes = new Pango.AttrList();
            toolsLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(toolsLabel, false, false, 5);

            var cropBtn = new Button("Toggle Crop Box");
            cropBtn.Clicked += (s, e) => {
                 _viewport.ToggleCropBox(true);
            };
            vbox.PackStart(cropBtn, false, false, 2);

            var applyCropBtn = new Button("Apply Crop");
            applyCropBtn.Clicked += (s, e) => {
                _viewport.ApplyCrop();
            };
            vbox.PackStart(applyCropBtn, false, false, 2);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 10);

            // Image Browser Panel with thumbnails
            _imageBrowser = new ImageBrowserPanel();
            _imageBrowser.ImageDoubleClicked += OnImageDoubleClicked;
            vbox.PackStart(_imageBrowser, true, true, 0);

            // Clear button
            var clearBtn = new Button("Clear Images");
            clearBtn.Clicked += (s, e) => {
                _imagePaths.Clear();
                _imageBrowser.Clear();
                _lastSceneResult = null;
            };
            vbox.PackStart(clearBtn, false, false, 5);

            _inference = new Model.Dust3rInference();
            if (_inference.IsLoaded) _statusLabel.Text = "Model Loaded";
            else _statusLabel.Text = "Model Not Found - Check dust3r.onnx";

            return vbox;
        }

        #region Menu Actions

        private void OnImportMesh(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Import Mesh", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Mesh Files";
            filter.AddPattern("*.obj");
            filter.AddPattern("*.ply");
            filter.AddPattern("*.stl");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                _statusLabel.Text = $"Import from {fc.Filename} - Not yet implemented";
            }
            fc.Destroy();
        }

        private void OnExportMesh(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh to export.");
                return;
            }

            var fc = new FileChooserDialog("Export Mesh", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "OBJ Files";
            filter.AddPattern("*.obj");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                _statusLabel.Text = $"Export to {fc.Filename} - Not yet implemented";
            }
            fc.Destroy();
        }

        private void OnDeleteSelected(object? sender, EventArgs e)
        {
            foreach (var obj in _sceneGraph.SelectedObjects.ToList())
            {
                _sceneGraph.RemoveObject(obj);
            }
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnDuplicateSelected(object? sender, EventArgs e)
        {
            var toDuplicate = _sceneGraph.SelectedObjects.ToList();
            foreach (var obj in toDuplicate)
            {
                var clone = obj.Clone();
                clone.Position += new OpenTK.Mathematics.Vector3(0.5f, 0, 0);
                _sceneGraph.AddObject(clone, obj.Parent);
            }
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnResetTransform(object? sender, EventArgs e)
        {
            foreach (var obj in _sceneGraph.SelectedObjects)
            {
                obj.Position = OpenTK.Mathematics.Vector3.Zero;
                obj.Rotation = OpenTK.Mathematics.Vector3.Zero;
                obj.Scale = OpenTK.Mathematics.Vector3.One;
            }
            _viewport.QueueDraw();
        }

        private void OnFlipNormals(object? sender, EventArgs e)
        {
            foreach (var meshObj in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                meshObj.MeshData = MeshOperations.FlipNormals(meshObj.MeshData);
            }
            _viewport.QueueDraw();
            _statusLabel.Text = "Flipped normals";
        }

        private void OnShowAbout(object? sender, EventArgs e)
        {
            var dialog = new MessageDialog(this, DialogFlags.Modal, MessageType.Info, ButtonsType.Ok,
                "Deep3D Studio\n\nA 3D reconstruction tool using Dust3r and NeRF.\n\nVersion 1.0");
            dialog.Run();
            dialog.Destroy();
        }

        #endregion

        #region Mesh Operation Handlers

        private void OnDecimateClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            foreach (var meshObj in selectedMeshes)
            {
                meshObj.MeshData = MeshOperations.Decimate(meshObj.MeshData, 0.5f);
                meshObj.UpdateBounds();
            }

            _viewport.QueueDraw();
            _statusLabel.Text = $"Decimated {selectedMeshes.Count} mesh(es)";
        }

        private void OnSmoothClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            foreach (var meshObj in selectedMeshes)
            {
                meshObj.MeshData = MeshOperations.SmoothTaubin(meshObj.MeshData, 2);
                meshObj.UpdateBounds();
            }

            _viewport.QueueDraw();
            _statusLabel.Text = $"Smoothed {selectedMeshes.Count} mesh(es)";
        }

        private void OnOptimizeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            int totalRemoved = 0;
            foreach (var meshObj in selectedMeshes)
            {
                int before = meshObj.VertexCount;
                meshObj.MeshData = MeshOperations.Optimize(meshObj.MeshData);
                meshObj.UpdateBounds();
                totalRemoved += before - meshObj.VertexCount;
            }

            _viewport.QueueDraw();
            _statusLabel.Text = $"Optimized: removed {totalRemoved} duplicate vertices";
        }

        private void OnSplitClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            int partsCreated = 0;
            foreach (var meshObj in selectedMeshes)
            {
                var parts = MeshOperations.SplitByConnectivity(meshObj.MeshData);
                if (parts.Count > 1)
                {
                    _sceneGraph.RemoveObject(meshObj);

                    for (int i = 0; i < parts.Count; i++)
                    {
                        var partObj = new MeshObject($"{meshObj.Name}_Part{i + 1}", parts[i]);
                        _sceneGraph.AddObject(partObj);
                        partsCreated++;
                    }
                }
            }

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Split into {partsCreated} parts";
        }

        private void OnMergeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count < 2)
            {
                ShowMessage("Please select at least 2 meshes to merge.");
                return;
            }

            var meshDataList = selectedMeshes.Select(m => m.MeshData).ToList();
            var merged = MeshOperations.MergeWithWelding(meshDataList);

            foreach (var m in selectedMeshes)
                _sceneGraph.RemoveObject(m);

            var mergedObj = new MeshObject("Merged Mesh", merged);
            _sceneGraph.AddObject(mergedObj);
            _sceneGraph.Select(mergedObj);

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Merged {selectedMeshes.Count} meshes";
        }

        private void OnAlignClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count < 2)
            {
                ShowMessage("Please select at least 2 meshes to align.");
                return;
            }

            var target = selectedMeshes[0];

            for (int i = 1; i < selectedMeshes.Count; i++)
            {
                var source = selectedMeshes[i];
                var transform = MeshOperations.AlignICP(source.MeshData, target.MeshData);
                source.MeshData.ApplyTransform(transform);
                source.UpdateBounds();
            }

            _viewport.QueueDraw();
            _statusLabel.Text = $"Aligned {selectedMeshes.Count - 1} mesh(es) to target";
        }

        #endregion

        #region Scene Event Handlers

        private void OnSceneObjectSelected(object? sender, SceneObject obj)
        {
            _statusLabel.Text = $"Selected: {obj.Name}";

            if (obj is MeshObject mesh)
            {
                _statusLabel.Text += $" ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)";
            }
            else if (obj is CameraObject cam)
            {
                _statusLabel.Text += $" ({cam.ImageWidth}x{cam.ImageHeight})";
            }
        }

        private void OnSceneObjectDoubleClicked(object? sender, SceneObject obj)
        {
            _sceneGraph.Select(obj);
            _viewport.FocusOnSelection();

            if (obj is CameraObject cam && !string.IsNullOrEmpty(cam.ImagePath))
            {
                // Could show image preview here
            }
        }

        private void OnSceneObjectAction(object? sender, (SceneObject obj, string action) args)
        {
            switch (args.action)
            {
                case "refresh_viewport":
                    _viewport.QueueDraw();
                    break;

                case "focus":
                    if (args.obj != null)
                    {
                        _sceneGraph.Select(args.obj);
                        _viewport.FocusOnSelection();
                    }
                    break;

                case "decimate":
                    OnDecimateClicked(null, EventArgs.Empty);
                    break;

                case "optimize":
                    OnOptimizeClicked(null, EventArgs.Empty);
                    break;

                case "smooth":
                    OnSmoothClicked(null, EventArgs.Empty);
                    break;

                case "split_connectivity":
                    OnSplitClicked(null, EventArgs.Empty);
                    break;

                case "merge_meshes":
                    OnMergeClicked(null, EventArgs.Empty);
                    break;

                case "align_meshes":
                    OnAlignClicked(null, EventArgs.Empty);
                    break;

                case "flip_normals":
                    OnFlipNormals(null, EventArgs.Empty);
                    break;

                case "view_from_camera":
                    if (args.obj is CameraObject cam)
                    {
                        _statusLabel.Text = $"View from {cam.Name}";
                    }
                    break;

                case "show_camera_image":
                    if (args.obj is CameraObject camImg && !string.IsNullOrEmpty(camImg.ImagePath))
                    {
                        var entry = new ImageEntry { FilePath = camImg.ImagePath };
                        var previewDialog = new ImagePreviewDialog(this, entry);
                        previewDialog.Run();
                        previewDialog.Destroy();
                    }
                    break;

                case "add_group":
                    var group = new GroupObject("New Group");
                    _sceneGraph.AddObject(group);
                    _sceneTreeView.RefreshTree();
                    break;
            }
        }

        private void OnViewportObjectPicked(object? sender, SceneObject? obj)
        {
            if (obj != null)
            {
                _sceneTreeView.SelectObject(obj);
            }
        }

        #endregion

        #region Other Event Handlers

        private void OnImageDoubleClicked(object? sender, ImageEntry entry)
        {
            var previewDialog = new ImagePreviewDialog(this, entry);
            previewDialog.Run();
            previewDialog.Destroy();
        }

        private void OnOpenSettings(object? sender, EventArgs e)
        {
            var dlg = new SettingsDialog(this);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                dlg.SaveSettings();
                ApplyViewSettings();
            }
            dlg.Destroy();
        }

        private void ApplyViewSettings()
        {
            var s = AppSettings.Instance;
            if (_pointsToggle != null) _pointsToggle.Active = s.ShowPointCloud;
            if (_wireToggle != null) _wireToggle.Active = s.ShowWireframe;
            if (_meshToggle != null) _meshToggle.Active = !s.ShowPointCloud;
            if (_rgbColorToggle != null) _rgbColorToggle.Active = s.PointCloudColor == PointCloudColorMode.RGB;
            if (_depthColorToggle != null) _depthColorToggle.Active = s.PointCloudColor == PointCloudColorMode.DistanceMap;
            _viewport.QueueDraw();
        }

        private void OnAddImages(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Choose Images", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);
            fc.SelectMultiple = true;

            var filter = new FileFilter();
            filter.Name = "Image Files";
            filter.AddPattern("*.jpg");
            filter.AddPattern("*.jpeg");
            filter.AddPattern("*.png");
            filter.AddPattern("*.bmp");
            filter.AddPattern("*.tiff");
            filter.AddPattern("*.tif");
            fc.AddFilter(filter);

            var allFilter = new FileFilter();
            allFilter.Name = "All Files";
            allFilter.AddPattern("*");
            fc.AddFilter(allFilter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                foreach (var f in fc.Filenames)
                {
                    _imagePaths.Add(f);
                    _imageBrowser.AddImage(f);
                }
                _statusLabel.Text = $"{_imageBrowser.ImageCount} images loaded";
            }
            fc.Destroy();
        }

        private async void OnRunInference(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Please add at least 2 images.");
                return;
            }

            string workflow = _workflowCombo.ActiveText;
            _statusLabel.Text = $"Running {workflow} on {AppSettings.Instance.Device}...";

            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                _statusLabel.Text = "Estimating Geometry (Dust3r)...";
                var result = await Task.Run(() => _inference.ReconstructScene(_imagePaths));

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Dust3r Inference failed.";
                    return;
                }

                _lastSceneResult = result;
                PopulateDepthData(result);

                _sceneGraph.Clear();

                if (workflow == "Dust3r (Fast)")
                {
                    _statusLabel.Text = $"Meshing using {AppSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                         var (grid, min, size) = VoxelizePoints(result.Meshes);
                         IMesher mesher = GetMesher(AppSettings.Instance.MeshingAlgo);
                         return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    var meshObj = new MeshObject("Reconstructed Mesh", meshedResult);
                    _sceneGraph.AddObject(meshObj);

                    AddCamerasToScene(result);

                    _statusLabel.Text = "Dust3r & Meshing Complete.";
                }
                else
                {
                    _statusLabel.Text = "Initializing NeRF Voxel Grid...";
                    var nerf = new VoxelGridNeRF();

                    await Task.Run(() =>
                    {
                        nerf.InitializeFromMesh(result.Meshes);
                        nerf.Train(result.Poses, iterations: 50);
                    });

                    _statusLabel.Text = $"Extracting NeRF Mesh ({AppSettings.Instance.MeshingAlgo})...";

                    var nerfMesh = await Task.Run(() => {
                         return nerf.GetMesh(GetMesher(AppSettings.Instance.MeshingAlgo));
                    });

                    var meshObj = new MeshObject("NeRF Mesh", nerfMesh);
                    _sceneGraph.AddObject(meshObj);

                    AddCamerasToScene(result);

                    _statusLabel.Text = "NeRF Complete.";
                }

                _sceneTreeView.RefreshTree();

                var (meshes, pcs, cams, verts, tris) = _sceneGraph.GetStatistics();
                _statusLabel.Text += $" | {meshes} meshes, {cams} cameras, {verts:N0} vertices";
            }
            catch (Exception ex)
            {
                 _statusLabel.Text = "Error: " + ex.Message;
                 Console.WriteLine(ex);
            }
        }

        private void AddCamerasToScene(SceneResult result)
        {
            var camerasGroup = new GroupObject("Cameras");
            _sceneGraph.AddObject(camerasGroup);

            for (int i = 0; i < result.Poses.Count; i++)
            {
                var pose = result.Poses[i];
                var camObj = new CameraObject($"Camera {i + 1}", pose);
                _sceneGraph.AddObject(camObj, camerasGroup);
            }
        }

        #endregion

        #region Helper Methods

        private void ShowMessage(string message)
        {
            var md = new MessageDialog(this, DialogFlags.Modal, MessageType.Info, ButtonsType.Ok, message);
            md.Run();
            md.Destroy();
        }

        private void PopulateDepthData(SceneResult result)
        {
            if (result.Poses.Count == 0) return;

            for (int i = 0; i < result.Poses.Count && i < result.Meshes.Count; i++)
            {
                var pose = result.Poses[i];
                var mesh = result.Meshes[i];

                var depthMap = ExtractDepthMap(mesh, pose.Width, pose.Height, pose.WorldToCamera);
                _imageBrowser.SetDepthData(i, depthMap);
            }

            _imageBrowser.QueueDraw();
        }

        private IMesher GetMesher(MeshingAlgorithm algo)
        {
            switch (algo)
            {
                case MeshingAlgorithm.GreedyMeshing: return new GreedyMesher();
                case MeshingAlgorithm.SurfaceNets: return new SurfaceNetsMesher();
                case MeshingAlgorithm.Blocky: return new BlockMesher();
                case MeshingAlgorithm.MarchingCubes: default: return new MarchingCubesMesher();
            }
        }

        private float[,] ExtractDepthMap(MeshData mesh, int width, int height, OpenTK.Mathematics.Matrix4 worldToCamera)
        {
            float[,] depthMap = new float[width, height];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    depthMap[x, y] = float.MaxValue;

            if (mesh.PixelToVertexIndex != null && mesh.PixelToVertexIndex.Length == width * height)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pIdx = y * width + x;
                        int vertIdx = mesh.PixelToVertexIndex[pIdx];
                        if (vertIdx >= 0 && vertIdx < mesh.Vertices.Count)
                        {
                            var v = mesh.Vertices[vertIdx];
                            var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                            depthMap[x, y] = -vCam.Z;
                        }
                    }
                }
            }
            else
            {
                foreach (var v in mesh.Vertices)
                {
                    var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                    int px = (int)((v.X + 1) * 0.5f * width);
                    int py = (int)((1 - (v.Y + 1) * 0.5f) * height);

                    if (px >= 0 && px < width && py >= 0 && py < height)
                    {
                        float depth = -vCam.Z;
                        if (depth < depthMap[px, py] && depth > 0)
                            depthMap[px, py] = depth;
                    }
                }
            }

            return depthMap;
        }

        private (float[,,], OpenTK.Mathematics.Vector3, float) VoxelizePoints(List<MeshData> meshes)
        {
            var min = new OpenTK.Mathematics.Vector3(float.MaxValue);
            var max = new OpenTK.Mathematics.Vector3(float.MinValue);
            foreach(var m in meshes) {
                foreach(var v in m.Vertices) {
                    min = OpenTK.Mathematics.Vector3.ComponentMin(min, v);
                    max = OpenTK.Mathematics.Vector3.ComponentMax(max, v);
                }
            }

            float voxelSize = 0.02f;
            int w = (int)((max.X - min.X) / voxelSize) + 5;
            int h = (int)((max.Y - min.Y) / voxelSize) + 5;
            int d = (int)((max.Z - min.Z) / voxelSize) + 5;

            if (w > 200) { voxelSize *= (w/200f); w=200; h=(int)((max.Y-min.Y)/voxelSize)+5; d=(int)((max.Z-min.Z)/voxelSize)+5; }

            float[,,] grid = new float[w,h,d];

            foreach(var m in meshes) {
                foreach(var v in m.Vertices) {
                    int x = (int)((v.X - min.X) / voxelSize);
                    int y = (int)((v.Y - min.Y) / voxelSize);
                    int z = (int)((v.Z - min.Z) / voxelSize);
                    if (x>=0 && x<w && y>=0 && y<h && z>=0 && z<d) {
                        grid[x,y,z] = 1.0f;
                    }
                }
            }

            float[,,] smooth = new float[w,h,d];
            for(int x=1; x<w-1; x++)
            for(int y=1; y<h-1; y++)
            for(int z=1; z<d-1; z++)
            {
                 if(grid[x,y,z] > 0) {
                     smooth[x,y,z] = 1;
                     smooth[x+1,y,z] = 1; smooth[x-1,y,z] = 1;
                     smooth[x,y+1,z] = 1; smooth[x,y-1,z] = 1;
                     smooth[x,y,z+1] = 1; smooth[x,y,z-1] = 1;
                 }
            }

            return (smooth, min, voxelSize);
        }

        #endregion
    }
}
