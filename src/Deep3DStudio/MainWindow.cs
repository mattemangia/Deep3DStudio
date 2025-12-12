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

        // Gizmo toolbar buttons
        private RadioToolButton? _translateBtn;
        private RadioToolButton? _rotateBtn;
        private RadioToolButton? _scaleBtn;

        public MainWindow() : base(WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio - Enhanced";
            this.SetDefaultSize(1400, 900);
            this.DeleteEvent += (o, args) => Application.Quit();

            // Initialize Scene Graph
            _sceneGraph = new SceneGraph();

            var mainVBox = new Box(Orientation.Vertical, 0);
            this.Add(mainVBox);

            // 1. Toolbar (Top)
            var toolbar = CreateToolbar();
            mainVBox.PackStart(toolbar, false, false, 0);

            // 2. Main Content (Tree + Viewport + Side Panel)
            var hPaned = new Paned(Orientation.Horizontal);
            mainVBox.PackStart(hPaned, true, true, 0);

            // Left: Scene Tree View
            var leftPanel = new Box(Orientation.Vertical, 0);
            leftPanel.SetSizeRequest(250, -1);

            _sceneTreeView = new SceneTreeView();
            _sceneTreeView.SetSceneGraph(_sceneGraph);
            _sceneTreeView.ObjectSelected += OnSceneObjectSelected;
            _sceneTreeView.ObjectDoubleClicked += OnSceneObjectDoubleClicked;
            _sceneTreeView.ObjectActionRequested += OnSceneObjectAction;
            leftPanel.PackStart(_sceneTreeView, true, true, 0);

            hPaned.Pack1(leftPanel, false, false);

            // Center + Right: Viewport + Side Panel
            var rightPaned = new Paned(Orientation.Horizontal);
            hPaned.Pack2(rightPaned, true, false);

            // 3D Viewport
            _viewport = new ThreeDView();
            _viewport.SetSceneGraph(_sceneGraph);
            _viewport.ObjectPicked += OnViewportObjectPicked;
            rightPaned.Pack1(_viewport, true, false);

            // 3. Status Bar (initialize before CreateSidePanel)
            _statusLabel = new Label("Ready");

            // Side Panel
            var sidePanel = CreateSidePanel();
            rightPaned.Pack2(sidePanel, false, false);
            rightPaned.Position = 900;
            hPaned.Position = 250;

            _statusLabel.Halign = Align.Start;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);

            // Load Initial Settings
            ApplyViewSettings();

            this.ShowAll();
        }

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

            // Transform Tool Buttons (Radio Group)
            _translateBtn = new RadioToolButton(new GLib.SList(IntPtr.Zero));
            _translateBtn.IconWidget = new Image(Stock.Fullscreen, IconSize.SmallToolbar);
            _translateBtn.Label = "Move";
            _translateBtn.TooltipText = "Move Tool (W)";
            _translateBtn.Active = true;
            _translateBtn.Toggled += (s, e) => {
                if (_translateBtn.Active)
                    _viewport.SetGizmoMode(GizmoMode.Translate);
            };
            toolbar.Insert(_translateBtn, -1);

            _rotateBtn = new RadioToolButton(_translateBtn.Group);
            _rotateBtn.IconWidget = new Image(Stock.Refresh, IconSize.SmallToolbar);
            _rotateBtn.Label = "Rotate";
            _rotateBtn.TooltipText = "Rotate Tool (E)";
            _rotateBtn.Toggled += (s, e) => {
                if (_rotateBtn.Active)
                    _viewport.SetGizmoMode(GizmoMode.Rotate);
            };
            toolbar.Insert(_rotateBtn, -1);

            _scaleBtn = new RadioToolButton(_translateBtn.Group);
            _scaleBtn.IconWidget = new Image(Stock.ZoomFit, IconSize.SmallToolbar);
            _scaleBtn.Label = "Scale";
            _scaleBtn.TooltipText = "Scale Tool (R)";
            _scaleBtn.Toggled += (s, e) => {
                if (_scaleBtn.Active)
                    _viewport.SetGizmoMode(GizmoMode.Scale);
            };
            toolbar.Insert(_scaleBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Focus button
            var focusBtn = new ToolButton(Stock.Home, "Focus");
            focusBtn.TooltipText = "Focus on Selection (F)";
            focusBtn.Clicked += (s, e) => _viewport.FocusOnSelection();
            toolbar.Insert(focusBtn, -1);

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
                    // Remove original
                    _sceneGraph.RemoveObject(meshObj);

                    // Add parts
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

            // Remove originals
            foreach (var m in selectedMeshes)
                _sceneGraph.RemoveObject(m);

            // Add merged
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
            // Focus on double-clicked object
            _sceneGraph.Select(obj);
            _viewport.FocusOnSelection();

            // If it's a camera, optionally show the image
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
                    if (args.obj is MeshObject meshObj)
                    {
                        meshObj.MeshData = MeshOperations.FlipNormals(meshObj.MeshData);
                        _viewport.QueueDraw();
                        _statusLabel.Text = "Flipped normals";
                    }
                    break;

                case "view_from_camera":
                    if (args.obj is CameraObject cam)
                    {
                        // TODO: Implement view from camera
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

        private void OnOpenSettings(object sender, EventArgs e)
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

        private void OnAddImages(object sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Choose Images", this, FileChooserAction.Open, "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);
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

        private async void OnRunInference(object sender, EventArgs e)
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
                // 1. Dust3r (Common)
                _statusLabel.Text = "Estimating Geometry (Dust3r)...";
                var result = await Task.Run(() => _inference.ReconstructScene(_imagePaths));

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Dust3r Inference failed.";
                    return;
                }

                // Store result and populate depth data
                _lastSceneResult = result;
                PopulateDepthData(result);

                // Clear old scene objects
                _sceneGraph.Clear();

                if (workflow == "Dust3r (Fast)")
                {
                    _statusLabel.Text = $"Meshing using {AppSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                         var (grid, min, size) = VoxelizePoints(result.Meshes);
                         IMesher mesher = GetMesher(AppSettings.Instance.MeshingAlgo);
                         return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    // Add mesh to scene graph
                    var meshObj = new MeshObject("Reconstructed Mesh", meshedResult);
                    _sceneGraph.AddObject(meshObj);

                    // Add cameras to scene graph
                    AddCamerasToScene(result);

                    _statusLabel.Text = "Dust3r & Meshing Complete.";
                }
                else // NeRF
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

                    // Add mesh to scene graph
                    var meshObj = new MeshObject("NeRF Mesh", nerfMesh);
                    _sceneGraph.AddObject(meshObj);

                    // Add cameras to scene graph
                    AddCamerasToScene(result);

                    _statusLabel.Text = "NeRF Complete.";
                }

                // Refresh tree view
                _sceneTreeView.RefreshTree();

                // Update statistics
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
