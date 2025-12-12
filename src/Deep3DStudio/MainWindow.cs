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
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public class MainWindow : Gtk.Window
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
        private ComboBoxText _workflowCombo = null!;
        private ToggleToolButton _pointsToggle = null!;
        private ToggleToolButton _wireToggle = null!;
        private ToggleToolButton _textureToggle = null!;
        private ToggleToolButton _meshToggle = null!;
        private ToggleToolButton _rgbColorToggle = null!;
        private ToggleToolButton _depthColorToggle = null!;

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
            if (_inference.IsLoaded) _statusLabel.Text = "Model Loaded";
            else _statusLabel.Text = "Model Not Found - Check dust3r.onnx";

            _mainHPaned.Position = settings.LastPanelWidth;

            _statusLabel.Halign = Align.Start;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);

            // Load Initial Settings
            ApplyViewSettings();

            // Enable Drag and Drop
            Gtk.Drag.DestSet(this, DestDefaults.All, new TargetEntry[] { new TargetEntry("text/uri-list", 0, 0) }, Gdk.DragAction.Copy);
            this.DragDataReceived += OnDragDataReceived;

            this.ShowAll();
        }

        /// <summary>
        /// Handles window close event - saves settings and quits application.
        /// </summary>
        private void OnWindowDelete(object o, DeleteEventArgs args)
        {
            SaveWindowState();
            Application.Quit();
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

        #region Drag and Drop

        private void OnDragDataReceived(object o, DragDataReceivedArgs args)
        {
            if (args.SelectionData.Length > 0 && args.SelectionData.Format == 8)
            {
                var uris = System.Text.Encoding.UTF8.GetString(args.SelectionData.Data).Split('\n');
                var imageFiles = new List<string>();
                int importedCount = 0;

                foreach (var uri in uris)
                {
                    var cleanUri = uri.Trim();
                    if (string.IsNullOrEmpty(cleanUri)) continue;

                    if (cleanUri.StartsWith("file://"))
                    {
                        // On Linux/Unix, file:///path/to/file.
                        // On Windows, file:///C:/path/to/file
                        string path = new Uri(cleanUri).LocalPath;

                        // LocalPath is already unescaped by the Uri class

                        if (System.IO.Directory.Exists(path)) continue; // Skip directories for now

                        string ext = System.IO.Path.GetExtension(path).ToLower();

                        if (ext == ".obj" || ext == ".stl" || (ext == ".ply" && IsMeshPly(path)))
                        {
                            try {
                                var meshData = MeshImporter.Load(path);
                                var meshObj = new MeshObject(System.IO.Path.GetFileNameWithoutExtension(path), meshData);
                                _sceneGraph.AddObject(meshObj);
                                importedCount++;
                            } catch (Exception ex) {
                                Console.WriteLine($"Error importing mesh {path}: {ex.Message}");
                            }
                        }
                        else if (ext == ".xyz" || ext == ".ply") // PLY defaults to point cloud if not clearly mesh or failed mesh check
                        {
                            try {
                                var pcObj = PointCloudImporter.Load(path);
                                _sceneGraph.AddObject(pcObj);
                                importedCount++;
                            } catch (Exception ex) {
                                Console.WriteLine($"Error importing point cloud {path}: {ex.Message}");
                            }
                        }
                        else if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".bmp")
                        {
                            imageFiles.Add(path);
                        }
                    }
                }

                if (imageFiles.Count > 0)
                {
                    foreach (var img in imageFiles)
                    {
                        if (!_imagePaths.Contains(img))
                        {
                            _imagePaths.Add(img);
                            _imageBrowser.AddImage(img);
                        }
                    }
                    _statusLabel.Text = $"{_imageBrowser.ImageCount} images loaded";
                }

                if (importedCount > 0)
                {
                    _statusLabel.Text = $"Imported {importedCount} 3D objects";
                    _viewport.QueueDraw();
                    _sceneTreeView.RefreshTree();
                }
            }
            Gtk.Drag.Finish(args.Context, true, false, args.Time);
        }

        private bool IsMeshPly(string path)
        {
            // Simple heuristic check if PLY contains "element face"
            try {
                using (var reader = new System.IO.StreamReader(path)) {
                    for(int i=0; i<20; i++) { // Check first 20 lines
                        var line = reader.ReadLine();
                        if (line == null) break;
                        if (line.Contains("element face") && !line.Contains("element face 0")) return true;
                        if (line == "end_header") break;
                    }
                }
            } catch {}
            return false;
        }

        #endregion

        #region Menu Bar

        private Widget CreateMenuBar()
        {
            var menuBar = new MenuBar();
            var accelGroup = new AccelGroup();
            this.AddAccelGroup(accelGroup);

            // File Menu
            var fileMenu = new Menu();
            var fileMenuItem = new MenuItem("_File");
            fileMenuItem.Submenu = fileMenu;

            var openImagesItem = new MenuItem("_Open Pictures...");
            openImagesItem.Activated += OnAddImages;
            fileMenu.Append(openImagesItem);

            var importMeshItem = new MenuItem("_Import Mesh...");
            importMeshItem.Activated += OnImportMesh;
            fileMenu.Append(importMeshItem);

            var importPointsItem = new MenuItem("Import _Point Cloud...");
            importPointsItem.Activated += OnImportPointCloud;
            fileMenu.Append(importPointsItem);

            fileMenu.Append(new SeparatorMenuItem());

            var exportMeshItem = new MenuItem("_Export Mesh...");
            exportMeshItem.Activated += OnExportMesh;
            fileMenu.Append(exportMeshItem);

            var exportPointsItem = new MenuItem("Export _Point Cloud...");
            exportPointsItem.Activated += OnExportPointCloud;
            fileMenu.Append(exportPointsItem);

            var exportDepthItem = new MenuItem("Export _Depth Maps...");
            exportDepthItem.Activated += OnExportDepthMaps;
            fileMenu.Append(exportDepthItem);

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

            var settingsItemEdit = new MenuItem("_Settings");
            settingsItemEdit.Activated += OnOpenSettings;
            settingsItemEdit.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.comma, Gdk.ModifierType.ControlMask, AccelFlags.Visible);
            editMenu.Append(settingsItemEdit);

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

            meshOpsMenu.Append(new SeparatorMenuItem());

            var scaleItem = new MenuItem("Set _Real Size...");
            scaleItem.Activated += OnSetRealSizeClicked;
            meshOpsMenu.Append(scaleItem);

            meshOpsMenu.Append(new SeparatorMenuItem());

            var cleanupItem = new MenuItem("_Cleanup Mesh...");
            cleanupItem.Activated += OnMeshCleanupClicked;
            meshOpsMenu.Append(cleanupItem);

            var bakeItem = new MenuItem("_Bake Textures...");
            bakeItem.Activated += OnBakeTexturesClicked;
            meshOpsMenu.Append(bakeItem);

            editMenu.Append(meshOpsMenuItem);

            // Triangle Editing submenu (Pen Tool operations)
            var triangleOpsMenu = new Menu();
            var triangleOpsMenuItem = new MenuItem("_Triangle Editing (Pen Tool)");
            triangleOpsMenuItem.Submenu = triangleOpsMenu;

            var deleteTrianglesItem = new MenuItem("_Delete Selected Triangles");
            deleteTrianglesItem.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.Delete, Gdk.ModifierType.None, AccelFlags.Visible);
            deleteTrianglesItem.Activated += OnDeleteSelectedTriangles;
            triangleOpsMenu.Append(deleteTrianglesItem);

            var flipTrianglesItem = new MenuItem("_Flip Selected Triangles");
            flipTrianglesItem.Activated += OnFlipSelectedTriangles;
            triangleOpsMenu.Append(flipTrianglesItem);

            var subdivideTrianglesItem = new MenuItem("_Subdivide Selected Triangles");
            subdivideTrianglesItem.Activated += OnSubdivideSelectedTriangles;
            triangleOpsMenu.Append(subdivideTrianglesItem);

            triangleOpsMenu.Append(new SeparatorMenuItem());

            var selectAllTrianglesItem = new MenuItem("Select _All Triangles");
            selectAllTrianglesItem.Activated += OnSelectAllTriangles;
            triangleOpsMenu.Append(selectAllTrianglesItem);

            var invertTriangleSelectionItem = new MenuItem("_Invert Selection");
            invertTriangleSelectionItem.Activated += OnInvertTriangleSelection;
            triangleOpsMenu.Append(invertTriangleSelectionItem);

            var growSelectionItem = new MenuItem("_Grow Selection");
            growSelectionItem.Activated += OnGrowTriangleSelection;
            triangleOpsMenu.Append(growSelectionItem);

            var clearTriangleSelectionItem = new MenuItem("_Clear Selection");
            clearTriangleSelectionItem.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.Escape, Gdk.ModifierType.None, AccelFlags.Visible);
            clearTriangleSelectionItem.Activated += OnClearTriangleSelection;
            triangleOpsMenu.Append(clearTriangleSelectionItem);

            triangleOpsMenu.Append(new SeparatorMenuItem());

            var weldVerticesItem = new MenuItem("_Weld Vertices");
            weldVerticesItem.Activated += OnWeldSelectedVertices;
            triangleOpsMenu.Append(weldVerticesItem);

            editMenu.Append(triangleOpsMenuItem);

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
                    IniSettings.Instance.ShowPointCloud = false;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(meshModeItem);

            var pointsModeItem = new RadioMenuItem(meshModeItem, "Show _Points");
            pointsModeItem.Toggled += (s, e) => {
                if (pointsModeItem.Active)
                {
                    IniSettings.Instance.ShowPointCloud = true;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(pointsModeItem);

            var wireframeItem = new CheckMenuItem("_Wireframe");
            wireframeItem.Active = IniSettings.Instance.ShowWireframe;
            wireframeItem.Toggled += (s, e) => {
                IniSettings.Instance.ShowWireframe = wireframeItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(wireframeItem);

            viewMenu.Append(new SeparatorMenuItem());

            // Color mode
            var rgbColorItem = new RadioMenuItem("_RGB Colors");
            rgbColorItem.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            rgbColorItem.Toggled += (s, e) => {
                if (rgbColorItem.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(rgbColorItem);

            var depthColorItem = new RadioMenuItem(rgbColorItem, "_Depth Colors");
            depthColorItem.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            depthColorItem.Toggled += (s, e) => {
                if (depthColorItem.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
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

            // Pen tool for triangle editing
            var penBtn = CreateIconButton("pen", "Pen Tool (P) - Edit Triangles", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Pen));
            vbox.PackStart(penBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // View tools
            var focusBtn = CreateIconButton("focus", "Focus (F)", btnSize, () => _viewport.FocusOnSelection());
            vbox.PackStart(focusBtn, false, false, 1);

            var cropBtn = CreateIconButton("crop", "Toggle Crop Box", btnSize, () => _viewport.ToggleCropBox(true));
            vbox.PackStart(cropBtn, false, false, 1);

            var applyCropBtn = CreateIconButton("apply_crop", "Apply Crop", btnSize, () => _viewport.ApplyCrop());
            vbox.PackStart(applyCropBtn, false, false, 1);

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

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            var cleanupBtn = CreateIconButton("cleanup", "Cleanup Mesh", btnSize, () => OnMeshCleanupClicked(null, EventArgs.Empty));
            vbox.PackStart(cleanupBtn, false, false, 1);

            var bakeBtn = CreateIconButton("bake", "Bake Textures", btnSize, () => OnBakeTexturesClicked(null, EventArgs.Empty));
            vbox.PackStart(bakeBtn, false, false, 1);

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

                case "apply_crop":
                    // Scissors/cut icon
                    cr.LineWidth = 2;
                    // Crop box outline
                    cr.SetSourceRGB(0.6, 0.6, 0.6);
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    // Checkmark inside
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 2.5;
                    cr.MoveTo(cx - 5, cy);
                    cr.LineTo(cx - 1, cy + 4);
                    cr.LineTo(cx + 5, cy - 4);
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

                case "cleanup":
                    // Broom icon
                    cr.LineWidth = 2;
                    // Handle
                    cr.SetSourceRGB(0.6, 0.4, 0.2);
                    cr.MoveTo(s - 4, 4);
                    cr.LineTo(s / 2, s / 2);
                    cr.Stroke();
                    // Bristles
                    cr.SetSourceRGB(0.9, 0.8, 0.4);
                    cr.MoveTo(s / 2, s / 2);
                    cr.LineTo(4, s - 8);
                    cr.LineTo(8, s - 4);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "bake":
                    // Texture/image icon
                    cr.LineWidth = 1.5;
                    cr.SetSourceRGB(0.8, 0.8, 0.8);
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    // Mountains/Sun
                    cr.MoveTo(4, s - 8);
                    cr.LineTo(s / 3, s / 2);
                    cr.LineTo(s / 2, s - 6);
                    cr.LineTo(2 * s / 3, s / 3);
                    cr.LineTo(s - 4, s - 8);
                    cr.Stroke();
                    cr.Arc(s - 8, 8, 2, 0, 2 * Math.PI);
                    cr.Fill();
                    break;

                case "pen":
                    // Pen/pencil icon for triangle editing
                    cr.LineWidth = 2;
                    cr.SetSourceRGB(1.0, 0.6, 0.2); // Orange color
                    // Pen body (diagonal)
                    cr.MoveTo(s - 4, 4);
                    cr.LineTo(6, s - 6);
                    cr.Stroke();
                    // Pen tip
                    cr.SetSourceRGB(0.4, 0.4, 0.4);
                    cr.MoveTo(6, s - 6);
                    cr.LineTo(4, s - 4);
                    cr.LineTo(8, s - 8);
                    cr.ClosePath();
                    cr.Fill();
                    // Triangle indicator
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 1.5;
                    cr.MoveTo(s - 8, s / 2);
                    cr.LineTo(s - 4, s - 4);
                    cr.LineTo(s / 2, s - 4);
                    cr.ClosePath();
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

            // Scene Tree (top)
            _sceneTreeView = new SceneTreeView();
            _sceneTreeView.SetSceneGraph(_sceneGraph);
            _sceneTreeView.ObjectSelected += OnSceneObjectSelected;
            _sceneTreeView.ObjectDoubleClicked += OnSceneObjectDoubleClicked;
            _sceneTreeView.ObjectActionRequested += OnSceneObjectAction;
            panel.PackStart(_sceneTreeView, true, true, 0);

            panel.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Image Browser (bottom)
            var imagesLabel = new Label("Input Images");
            imagesLabel.Attributes = new Pango.AttrList();
            imagesLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            panel.PackStart(imagesLabel, false, false, 2);

            _imageBrowser = new ImageBrowserPanel();
            _imageBrowser.ImageDoubleClicked += OnImageDoubleClicked;
            _imageBrowser.SetSizeRequest(-1, 150);
            panel.PackStart(_imageBrowser, false, false, 0);

            // Clear button
            var clearBtn = new Button("Clear Images");
            clearBtn.Clicked += (s, e) => {
                _imagePaths.Clear();
                _imageBrowser.Clear();
                _lastSceneResult = null;
            };
            panel.PackStart(clearBtn, false, false, 2);

            return panel;
        }

        private void OnToggleSceneTree(object? sender, EventArgs e)
        {
            if (_showSceneTreeMenuItem != null)
            {
                _leftPanel.Visible = _showSceneTreeMenuItem.Active;
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
            _verticalToolbar.Visible = false;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = false;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = false;
        }

        private void OnRestoreAllPanels(object? sender, EventArgs e)
        {
            _leftPanel.Visible = true;
            _verticalToolbar.Visible = true;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = true;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = true;
        }

        #endregion

        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 24;

            // Select Tool
            var selectBtn = new ToolButton(AppIconFactory.GenerateIcon("select", iconSize), "Select");
            selectBtn.TooltipText = "Select Objects (Q)";
            selectBtn.Clicked += (s, e) => _viewport.SetGizmoMode(GizmoMode.Select);
            toolbar.Insert(selectBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

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
                 IniSettings.Instance.ShowPointCloud = !_meshToggle.Active;
                 _viewport.QueueDraw();
            };
            toolbar.Insert(_meshToggle, -1);

            _pointsToggle = new ToggleToolButton();
            _pointsToggle.IconWidget = AppIconFactory.GenerateIcon("pointcloud", iconSize);
            _pointsToggle.Label = "Points";
            _pointsToggle.TooltipText = "Show Point Cloud";
            _pointsToggle.Active = IniSettings.Instance.ShowPointCloud;
            _pointsToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowPointCloud = _pointsToggle.Active;
                _meshToggle.Active = !IniSettings.Instance.ShowPointCloud;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_pointsToggle, -1);

            _wireToggle = new ToggleToolButton();
            _wireToggle.IconWidget = AppIconFactory.GenerateIcon("wireframe", iconSize);
            _wireToggle.Label = "Wireframe";
            _wireToggle.TooltipText = "Toggle Wireframe Overlay";
            _wireToggle.Active = IniSettings.Instance.ShowWireframe;
            _wireToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowWireframe = _wireToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_wireToggle, -1);

            _textureToggle = new ToggleToolButton();
            _textureToggle.IconWidget = AppIconFactory.GenerateIcon("texture", iconSize);
            _textureToggle.Label = "Texture";
            _textureToggle.TooltipText = "Toggle Texture Display";
            _textureToggle.Active = IniSettings.Instance.ShowTexture;
            _textureToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowTexture = _textureToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_textureToggle, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Point Cloud Color Mode Toggles
            _rgbColorToggle = new ToggleToolButton();
            _rgbColorToggle.IconWidget = AppIconFactory.GenerateIcon("rgb", iconSize);
            _rgbColorToggle.Label = "RGB";
            _rgbColorToggle.TooltipText = "Show original RGB colors";
            _rgbColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            _rgbColorToggle.Toggled += (s, e) => {
                if (_rgbColorToggle.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
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
            _depthColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            _depthColorToggle.Toggled += (s, e) => {
                if (_depthColorToggle.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
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
            _workflowCombo.AppendText("Interior Scan");
            _workflowCombo.Active = 0;
            wfBox.PackStart(_workflowCombo, false, false, 0);
            wfItem.Add(wfBox);
            toolbar.Insert(wfItem, -1);

            // Run
            var runPointsBtn = new ToolButton(AppIconFactory.GenerateIcon("pointcloud", iconSize), "Gen Points");
            runPointsBtn.TooltipText = "Generate Point Cloud Only";
            runPointsBtn.Clicked += OnGeneratePointCloud;
            toolbar.Insert(runPointsBtn, -1);

            var runMeshBtn = new ToolButton(AppIconFactory.GenerateIcon("mesh", iconSize), "Gen Mesh");
            runMeshBtn.TooltipText = "Generate Mesh from existing Point Cloud";
            runMeshBtn.Clicked += OnGenerateMesh;
            toolbar.Insert(runMeshBtn, -1);

            var runBtn = new ToolButton(AppIconFactory.GenerateIcon("run", iconSize), "Run All");
            runBtn.TooltipText = "Start Full Reconstruction Process";
            runBtn.Clicked += OnRunInference;
            toolbar.Insert(runBtn, -1);

            return toolbar;
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
                try
                {
                    string path = fc.Filename;
                    var meshData = MeshImporter.Load(path);
                    var meshObj = new MeshObject(System.IO.Path.GetFileNameWithoutExtension(path), meshData);
                    _sceneGraph.AddObject(meshObj);
                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Imported mesh from {path}";
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error importing mesh: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private void OnImportPointCloud(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Import Point Cloud", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Point Cloud Files";
            filter.AddPattern("*.ply");
            filter.AddPattern("*.xyz");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string path = fc.Filename;
                    var pcObj = PointCloudImporter.Load(path);
                    _sceneGraph.AddObject(pcObj);
                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Imported point cloud from {path}";
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error importing point cloud: {ex.Message}");
                }
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

        private void OnExportPointCloud(object? sender, EventArgs e)
        {
            var selectedObjects = _sceneGraph.SelectedObjects;
            if (selectedObjects.Count == 0)
            {
                ShowMessage("Please select objects (Mesh or Point Cloud) to export.");
                return;
            }

            var fc = new FileChooserDialog("Export Point Cloud", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            var plyFilter = new FileFilter { Name = "PLY Files" };
            plyFilter.AddPattern("*.ply");
            fc.AddFilter(plyFilter);

            var xyzFilter = new FileFilter { Name = "XYZ Files" };
            xyzFilter.AddPattern("*.xyz");
            fc.AddFilter(xyzFilter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                string filename = fc.Filename;
                if (!filename.EndsWith(".ply", StringComparison.OrdinalIgnoreCase) && !filename.EndsWith(".xyz", StringComparison.OrdinalIgnoreCase))
                {
                    // Default to PLY if no extension or unknown
                    filename += ".ply";
                }

                string ext = System.IO.Path.GetExtension(filename).ToLower();
                var format = ext == ".xyz" ? PointCloudExporter.ExportFormat.XYZ : PointCloudExporter.ExportFormat.PLY;

                // Ask for RGB
                bool includeColors = true;
                var colorMsg = new MessageDialog(this, DialogFlags.Modal, MessageType.Question, ButtonsType.YesNo, "Include RGB Colors?");
                if (colorMsg.Run() == (int)ResponseType.No) includeColors = false;
                colorMsg.Destroy();

                int count = 0;
                foreach (var obj in selectedObjects)
                {
                    // If multiple objects, append index or name to filename to avoid overwrite?
                    // For now, let's just export the first one or handle appropriately.
                    // Simplification: Export first valid object or merge them?
                    // Let's iterate and export individually with suffix if count > 1

                    string currentPath = filename;
                    if (selectedObjects.Count > 1)
                    {
                        string dir = System.IO.Path.GetDirectoryName(filename) ?? "";
                        string name = System.IO.Path.GetFileNameWithoutExtension(filename);
                        currentPath = System.IO.Path.Combine(dir, $"{name}_{obj.Name}{ext}");
                    }

                    if (obj is MeshObject meshObj)
                    {
                        PointCloudExporter.Export(currentPath, meshObj.MeshData, format, includeColors);
                        count++;
                    }
                    else if (obj is PointCloudObject pcObj)
                    {
                        PointCloudExporter.Export(currentPath, pcObj, format, includeColors);
                        count++;
                    }
                }

                _statusLabel.Text = $"Exported {count} point cloud(s).";
            }
            fc.Destroy();
        }

        private void OnExportDepthMaps(object? sender, EventArgs e)
        {
             var images = _imageBrowser.GetImages();
             var imagesWithDepth = images.Where(i => i.DepthMap != null).ToList();

             if (imagesWithDepth.Count == 0)
             {
                 ShowMessage("No depth maps available. Run reconstruction first.");
                 return;
             }

             var fc = new FileChooserDialog("Select Output Folder for Depth Maps", this, FileChooserAction.SelectFolder,
                 "Cancel", ResponseType.Cancel, "Select", ResponseType.Accept);

             if (fc.Run() == (int)ResponseType.Accept)
             {
                 string outputDir = fc.Filename;
                 int exported = 0;

                 _statusLabel.Text = "Exporting depth maps...";
                 while (Application.EventsPending()) Application.RunIteration();

                 foreach (var img in imagesWithDepth)
                 {
                     if (img.DepthMap == null) continue;

                     string baseName = System.IO.Path.GetFileNameWithoutExtension(img.FileName);
                     string outPath = System.IO.Path.Combine(outputDir, $"{baseName}_depth.png");

                     try
                     {
                         var depthMap = img.DepthMap;
                         int width = depthMap.GetLength(0);
                         int height = depthMap.GetLength(1);

                         // Find min/max for normalization
                        float minDepth = float.MaxValue;
                        float maxDepth = float.MinValue;
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float d = depthMap[x, y];
                                if (d > 0 && d < float.MaxValue)
                                {
                                    if (d < minDepth) minDepth = d;
                                    if (d > maxDepth) maxDepth = d;
                                }
                            }
                        }

                        float range = maxDepth - minDepth;
                        if (range < 0.0001f) range = 1.0f;

                        using var bitmap = new SkiaSharp.SKBitmap(width, height, SkiaSharp.SKColorType.Rgba8888, SkiaSharp.SKAlphaType.Premul);
                        for(int y=0; y<height; y++)
                        {
                            for(int x=0; x<width; x++)
                            {
                                float d = depthMap[x, y];
                                float t = (d - minDepth) / range;
                                t = Math.Clamp(t, 0f, 1f);

                                var (r, g, b) = ImageUtils.TurboColormap(t);
                                bitmap.SetPixel(x, y, new SkiaSharp.SKColor((byte)(r*255), (byte)(g*255), (byte)(b*255), 255));
                            }
                        }

                        using var image = SkiaSharp.SKImage.FromBitmap(bitmap);
                        using var data = image.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
                        using var stream = File.OpenWrite(outPath);
                        data.SaveTo(stream);

                        exported++;
                     }
                     catch (Exception ex)
                     {
                         Console.WriteLine($"Failed to export depth map for {img.FileName}: {ex.Message}");
                     }
                 }

                 _statusLabel.Text = $"Exported {exported} depth maps to {outputDir}";
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

        #region Triangle Editing Handlers

        private void OnDeleteSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.DeleteSelectedTriangles();
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Deleted {stats.triangleCount} triangles from {stats.meshCount} mesh(es)";
        }

        private void OnFlipSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.FlipSelectedTriangles();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Flipped {stats.triangleCount} triangles";
        }

        private void OnSubdivideSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.SubdivideSelectedTriangles();
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Subdivided {stats.triangleCount} triangles (each into 4)";
        }

        private void OnSelectAllTriangles(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                _statusLabel.Text = "No mesh selected. Select a mesh first.";
                return;
            }

            var tool = _viewport.MeshEditingTool;
            foreach (var mesh in selectedMeshes)
            {
                tool.SelectAll(mesh);
            }
            _viewport.QueueDraw();

            var stats = tool.GetSelectionStats();
            _statusLabel.Text = $"Selected all {stats.triangleCount} triangles";
        }

        private void OnInvertTriangleSelection(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                _statusLabel.Text = "No mesh selected. Select a mesh first.";
                return;
            }

            var tool = _viewport.MeshEditingTool;
            foreach (var mesh in selectedMeshes)
            {
                tool.InvertSelection(mesh);
            }
            _viewport.QueueDraw();

            var stats = tool.GetSelectionStats();
            _statusLabel.Text = $"Inverted selection: {stats.triangleCount} triangles now selected";
        }

        private void OnGrowTriangleSelection(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            int beforeCount = tool.SelectedTriangles.Count;
            tool.GrowSelection();
            int afterCount = tool.SelectedTriangles.Count;
            _viewport.QueueDraw();
            _statusLabel.Text = $"Selection grown: {beforeCount} -> {afterCount} triangles";
        }

        private void OnClearTriangleSelection(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            tool.ClearSelection();
            _viewport.QueueDraw();
            _statusLabel.Text = "Triangle selection cleared";
        }

        private void OnWeldSelectedVertices(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            tool.WeldSelectedVertices(0.001f);
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = "Welded duplicate vertices in selected area";
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
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();

            if (selectedMeshes.Count >= 2)
            {
                var meshDataList = selectedMeshes.Select(m => m.MeshData).ToList();
                var merged = MeshOperations.MergeWithWelding(meshDataList);

                foreach (var m in selectedMeshes)
                    _sceneGraph.RemoveObject(m);

                var mergedObj = new MeshObject("Merged Mesh", merged);
                _sceneGraph.AddObject(mergedObj);
                _sceneGraph.Select(mergedObj);

                _statusLabel.Text = $"Merged {selectedMeshes.Count} meshes";
            }
            else if (selectedPointClouds.Count >= 2)
            {
                var merged = MeshOperations.MergePointClouds(selectedPointClouds);

                foreach (var pc in selectedPointClouds)
                    _sceneGraph.RemoveObject(pc);

                _sceneGraph.AddObject(merged);
                _sceneGraph.Select(merged);

                _statusLabel.Text = $"Merged {selectedPointClouds.Count} point clouds";
            }
            else
            {
                ShowMessage("Please select at least 2 meshes or 2 point clouds to merge.");
                return;
            }

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnAlignClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();

            if (selectedMeshes.Count >= 2)
            {
                var target = selectedMeshes[0];
                for (int i = 1; i < selectedMeshes.Count; i++)
                {
                    var source = selectedMeshes[i];
                    var transform = MeshOperations.AlignICP(source.MeshData, target.MeshData);
                    source.MeshData.ApplyTransform(transform);
                    source.UpdateBounds();
                }
                _statusLabel.Text = $"Aligned {selectedMeshes.Count - 1} mesh(es) to target";
            }
            else if (selectedPointClouds.Count >= 2)
            {
                var target = selectedPointClouds[0];
                var targetPoints = target.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, target.GetWorldTransform())).ToList();

                for (int i = 1; i < selectedPointClouds.Count; i++)
                {
                    var source = selectedPointClouds[i];
                    var sourcePoints = source.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, source.GetWorldTransform())).ToList();

                    var transform = MeshOperations.AlignICP(sourcePoints, targetPoints);

                    // Apply transform to the object's existing transform
                    // Note: This applies it in local space, assuming the alignment was computed in world space
                    // We need to be careful with coordinate spaces.
                    // AlignICP returns a transform that maps Source -> Target in the space they were passed in (World Space)
                    // NewWorld = OldWorld * Transform
                    // Parent * NewLocal = Parent * OldLocal * Transform
                    // NewLocal = OldLocal * Transform

                    source.ApplyTransform(transform);
                    source.UpdateBounds();
                }
                _statusLabel.Text = $"Aligned {selectedPointClouds.Count - 1} point cloud(s) to target";
            }
            else
            {
                ShowMessage("Please select at least 2 meshes or 2 point clouds to align.");
                return;
            }

            _viewport.QueueDraw();
        }

        private void OnSetRealSizeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();

            // If no selection, apply to all meshes in the scene
            if (selectedMeshes.Count == 0)
            {
                var allMeshes = _sceneGraph.GetObjectsOfType<MeshObject>().ToList();
                if (allMeshes.Count == 0)
                {
                    ShowMessage("No meshes found to scale.");
                    return;
                }
                selectedMeshes = allMeshes;
            }

            // Calculate bounding box of selection
            var min = new OpenTK.Mathematics.Vector3(float.MaxValue);
            var max = new OpenTK.Mathematics.Vector3(float.MinValue);
            bool hasBounds = false;

            foreach (var obj in selectedMeshes)
            {
                var (bMin, bMax) = obj.GetWorldBounds();
                min = OpenTK.Mathematics.Vector3.ComponentMin(min, bMin);
                max = OpenTK.Mathematics.Vector3.ComponentMax(max, bMax);
                hasBounds = true;
            }

            if (!hasBounds) return;

            float sizeX = max.X - min.X;
            float sizeY = max.Y - min.Y;
            float sizeZ = max.Z - min.Z;

            var dlg = new ScaleCalibrationDialog(this, sizeX, sizeY, sizeZ);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                float factor = dlg.RealScaleFactor;
                if (Math.Abs(factor - 1.0f) > 0.0001f)
                {
                    // Apply scale to selected objects.
                    // We apply the transform directly to the mesh vertices ("baking" the scale)
                    // to ensure that exported models retain the correct physical dimensions
                    // regardless of the target software's handling of hierarchy transforms.

                    foreach (var meshObj in selectedMeshes)
                    {
                        var matrix = OpenTK.Mathematics.Matrix4.CreateScale(factor);
                        meshObj.MeshData.ApplyTransform(matrix);

                        // Update position to maintain relative distances between objects
                        meshObj.Position *= factor;
                        meshObj.UpdateBounds();
                    }

                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Scaled {selectedMeshes.Count} objects by {factor:F4}";
                }
            }
            dlg.Destroy();
        }

        private void OnMeshCleanupClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            // If multiple selected, ask to process first or all? For now process all.
            int processed = 0;
            foreach (var meshObj in selectedMeshes)
            {
                var dlg = new MeshCleanupDialog(this, meshObj.VertexCount, meshObj.TriangleCount);
                if (dlg.Run() == (int)ResponseType.Ok)
                {
                    _statusLabel.Text = $"Cleaning mesh {meshObj.Name}...";
                    while (Application.EventsPending()) Application.RunIteration();

                    meshObj.MeshData = MeshCleaningTools.CleanupMesh(meshObj.MeshData, dlg.Options);
                    meshObj.UpdateBounds();
                    processed++;
                }
                dlg.Destroy();
            }

            if (processed > 0)
            {
                _viewport.QueueDraw();
                _statusLabel.Text = $"Cleaned {processed} mesh(es)";
                _sceneTreeView.RefreshTree();
            }
        }

        private async void OnBakeTexturesClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh to bake textures onto.");
                return;
            }

            var meshObj = selectedMeshes[0]; // Process first one for now
            var cameras = _sceneGraph.GetObjectsOfType<CameraObject>().ToList();

            if (cameras.Count == 0)
            {
                ShowMessage("No cameras found in scene. Cannot bake textures from images.");
                return;
            }

            var dlg = new TextureBakingDialog(this, cameras);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                // Ask for output file if we are exporting
                string exportPath = "";
                if (dlg.ExportOptions.Format != TexturedMeshFormat.OBJ &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.GLTF &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.GLB &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.FBX_ASCII &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.PLY)
                {
                     // Should not happen if dialog returns valid format
                }

                var fc = new FileChooserDialog("Export Textured Mesh", this, FileChooserAction.Save,
                    "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

                string ext = dlg.ExportOptions.Format switch
                {
                    TexturedMeshFormat.OBJ => ".obj",
                    TexturedMeshFormat.GLTF => ".gltf",
                    TexturedMeshFormat.GLB => ".glb",
                    TexturedMeshFormat.FBX_ASCII => ".fbx",
                    TexturedMeshFormat.PLY => ".ply",
                    _ => ".obj"
                };

                fc.CurrentName = meshObj.Name + ext;

                if (fc.Run() == (int)ResponseType.Accept)
                {
                    exportPath = fc.Filename;
                }
                fc.Destroy();

                if (string.IsNullOrEmpty(exportPath)) return;

                _statusLabel.Text = "Baking textures... This may take a while.";
                while (Application.EventsPending()) Application.RunIteration();

                try
                {
                    var baker = new TextureBaker();
                    // Copy settings
                    baker.TextureSize = dlg.BakerSettings.TextureSize;
                    baker.IslandMargin = dlg.BakerSettings.IslandMargin;
                    baker.BlendMode = dlg.BakerSettings.BlendMode;
                    baker.MinViewAngleCosine = dlg.BakerSettings.MinViewAngleCosine;
                    baker.BlendSeams = dlg.BakerSettings.BlendSeams;
                    baker.DilationPasses = dlg.BakerSettings.DilationPasses;

                    var uvData = await Task.Run(() => baker.GenerateUVs(meshObj.MeshData, dlg.UVMethod));

                    BakedTextureResult? baked = null;

                    if (dlg.BakeFromCameras)
                    {
                         baked = await Task.Run(() => baker.BakeTextures(meshObj.MeshData, uvData, dlg.SelectedCameras));
                    }
                    else
                    {
                        var tex = await Task.Run(() => baker.BakeVertexColorsToTexture(meshObj.MeshData, uvData));
                        baked = new BakedTextureResult
                        {
                            DiffuseMap = tex,
                            TextureSize = baker.TextureSize,
                            WeightMap = new float[baker.TextureSize, baker.TextureSize]
                        };
                    }

                    _statusLabel.Text = "Exporting textured mesh...";
                    while (Application.EventsPending()) Application.RunIteration();

                    await Task.Run(() => TexturedMeshExporter.Export(exportPath, meshObj.MeshData, uvData, baked, dlg.ExportOptions));

                    baked?.Dispose();

                    _statusLabel.Text = $"Exported textured mesh to {exportPath}";
                    ShowMessage($"Baking and Export Complete!\nSaved to: {exportPath}");
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error during baking: {ex.Message}");
                    Console.WriteLine(ex);
                    _statusLabel.Text = "Baking failed.";
                }
            }
            dlg.Destroy();
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

                case "cleanup_mesh":
                    OnMeshCleanupClicked(null, EventArgs.Empty);
                    break;

                case "bake_textures":
                    OnBakeTexturesClicked(null, EventArgs.Empty);
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
            var s = IniSettings.Instance;
            if (_pointsToggle != null) _pointsToggle.Active = s.ShowPointCloud;
            if (_wireToggle != null) _wireToggle.Active = s.ShowWireframe;
            if (_textureToggle != null) _textureToggle.Active = s.ShowTexture;
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

        private async void OnGeneratePointCloud(object? sender, EventArgs e)
        {
             await RunPointCloudGeneration();
        }

        private async void OnGenerateMesh(object? sender, EventArgs e)
        {
             await RunMeshing();
        }

        private async void OnRunInference(object? sender, EventArgs e)
        {
             bool success = await RunPointCloudGeneration();
             if (success)
             {
                 await RunMeshing();
             }
        }

        private async Task<bool> RunPointCloudGeneration()
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Please add at least 2 images.");
                return false;
            }

            var settings = IniSettings.Instance;
            _statusLabel.Text = $"Estimating Geometry on {settings.Device}...";

            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                SceneResult result = new SceneResult();

                // Determine which reconstruction method to use
                bool useDust3r = settings.ReconstructionMethod == ReconstructionMethod.Dust3r;
                if (useDust3r && !_inference.IsLoaded)
                {
                    Console.WriteLine("Dust3r model not found, falling back to Feature Matching SfM.");
                    useDust3r = false;
                }

                if (useDust3r)
                {
                    _statusLabel.Text = "Estimating Geometry (Dust3r)...";
                    result = await Task.Run(() => _inference.ReconstructScene(_imagePaths));
                }
                else
                {
                    _statusLabel.Text = "Estimating Geometry (Feature Matching SfM)...";
                    var sfm = new Deep3DStudio.Model.SfM.SfMInference();
                    result = await Task.Run(() => sfm.ReconstructScene(_imagePaths));
                }

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Reconstruction failed. No points generated.";
                    return false;
                }

                _lastSceneResult = result;
                PopulateDepthData(result);

                // Update Scene with Point Cloud
                _sceneGraph.Clear();

                // Add Point Clouds (from result.Meshes acting as points)
                int totalPoints = 0;
                for(int i=0; i<result.Meshes.Count; i++)
                {
                    var mesh = result.Meshes[i];
                    Console.WriteLine($"PointCloud {i}: {mesh.Vertices.Count} points, {mesh.Colors.Count} colors");
                    totalPoints += mesh.Vertices.Count;

                    var pcObj = new PointCloudObject($"PointCloud_{i}", mesh);
                    _sceneGraph.AddObject(pcObj);
                }

                AddCamerasToScene(result);

                _sceneTreeView.RefreshTree();

                // Auto-focus on the generated point cloud
                _viewport.FocusOnSelection();
                _viewport.QueueDraw();
                _statusLabel.Text = $"Point Cloud Complete: {totalPoints:N0} points, {result.Poses.Count} cameras.";

                return true;
            }
            catch (Exception ex)
            {
                 _statusLabel.Text = "Error: " + ex.Message;
                 Console.WriteLine(ex);
                 return false;
            }
        }

        private async Task RunMeshing()
        {
            if (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0)
            {
                // Try to build result from current scene if possible?
                // For now, require point cloud generation first
                ShowMessage("No point cloud data available. Please generate point cloud first or import data.");
                return;
            }

            string workflow = _workflowCombo.ActiveText;
            _statusLabel.Text = $"Meshing ({workflow})...";
            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                // Remove existing meshes to avoid clutter, or keep them?
                // Let's remove old reconstruction meshes but keep imported ones?
                // For simplicity, we are working on _lastSceneResult data.

                if (workflow == "Dust3r (Fast)")
                {
                    _statusLabel.Text = $"Meshing using {IniSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                         var (grid, min, size) = VoxelizePoints(_lastSceneResult.Meshes);
                         IMesher mesher = GetMesher(IniSettings.Instance.MeshingAlgo);
                         return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    var meshObj = new MeshObject("Reconstructed Mesh", meshedResult);
                    _sceneGraph.AddObject(meshObj);
                    _statusLabel.Text = "Meshing Complete.";
                }
                else if (workflow == "Interior Scan")
                {
                    _statusLabel.Text = $"Meshing Interior (High Res) using {IniSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                         var (grid, min, size) = VoxelizePoints(_lastSceneResult.Meshes, 500);
                         IMesher mesher = GetMesher(IniSettings.Instance.MeshingAlgo);
                         return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    var meshObj = new MeshObject("Interior Mesh", meshedResult);
                    _sceneGraph.AddObject(meshObj);
                    _statusLabel.Text = "Interior Meshing Complete.";
                }
                else
                {
                    _statusLabel.Text = "Initializing NeRF Voxel Grid...";
                    var nerf = new VoxelGridNeRF();

                    var nerfSettings = IniSettings.Instance;
                    await Task.Run(() =>
                    {
                        nerf.InitializeFromMesh(_lastSceneResult.Meshes);
                        nerf.Train(_lastSceneResult.Poses, iterations: nerfSettings.NeRFIterations);
                    });

                    _statusLabel.Text = $"Extracting NeRF Mesh ({IniSettings.Instance.MeshingAlgo})...";

                    var nerfMesh = await Task.Run(() => {
                         return nerf.GetMesh(GetMesher(IniSettings.Instance.MeshingAlgo));
                    });

                    var meshObj = new MeshObject("NeRF Mesh", nerfMesh);
                    _sceneGraph.AddObject(meshObj);
                    _statusLabel.Text = "NeRF Meshing Complete.";
                }

                _sceneTreeView.RefreshTree();
                _viewport.QueueDraw();

                var (meshes, pcs, cams, verts, tris) = _sceneGraph.GetStatistics();
                _statusLabel.Text += $" | {meshes} meshes, {verts:N0} vertices";
            }
            catch (Exception ex)
            {
                 _statusLabel.Text = "Error during meshing: " + ex.Message;
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
            if (result.Poses.Count == 0 || result.Meshes.Count == 0) return;

            // SfM produces one combined mesh with all points, so use the same mesh for all cameras
            // Dust3r might produce per-camera meshes, so handle both cases
            var combinedMesh = result.Meshes[0];
            if (result.Meshes.Count > 1)
            {
                // Combine all meshes if there are multiple
                combinedMesh = new MeshData();
                foreach (var m in result.Meshes)
                {
                    combinedMesh.Vertices.AddRange(m.Vertices);
                    combinedMesh.Colors.AddRange(m.Colors);
                }
            }

            for (int i = 0; i < result.Poses.Count; i++)
            {
                var pose = result.Poses[i];
                var depthMap = ExtractDepthMap(combinedMesh, pose.Width, pose.Height, pose.WorldToCamera);
                _imageBrowser.SetDepthData(i, depthMap);
            }

            _imageBrowser.QueueDraw();
        }

        private IMesher GetMesher(MeshingAlgorithm algo)
        {
            switch (algo)
            {
                case MeshingAlgorithm.Poisson: return new PoissonMesher();
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

        private (float[,,], OpenTK.Mathematics.Vector3, float) VoxelizePoints(List<MeshData> meshes, int maxRes = 200)
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

            if (w > maxRes) { voxelSize *= (w/(float)maxRes); w=maxRes; h=(int)((max.Y-min.Y)/voxelSize)+5; d=(int)((max.Z-min.Z)/voxelSize)+5; }

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
