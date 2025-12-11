using System;
using Gtk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Deep3DStudio
{
    public class MainWindow : Window
    {
        private ThreeDView _viewport;
        private Label _statusLabel;
        private Model.Dust3rInference _inference;
        private List<string> _imagePaths = new List<string>();
        private ImageBrowserPanel _imageBrowser;
        private SceneResult? _lastSceneResult; // Store last reconstruction for depth visualization

        // UI References for updates
        private ComboBoxText _workflowCombo;
        private ToggleToolButton _pointsToggle;
        private ToggleToolButton _wireToggle;
        private ToggleToolButton _meshToggle;
        private ToggleToolButton _rgbColorToggle;
        private ToggleToolButton _depthColorToggle;

        public MainWindow() : base(WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio - Dust3r & NeRF";
            this.SetDefaultSize(1200, 800);
            this.DeleteEvent += (o, args) => Application.Quit();

            var mainVBox = new Box(Orientation.Vertical, 0);
            this.Add(mainVBox);

            // 1. Toolbar (Top)
            var toolbar = CreateToolbar();
            mainVBox.PackStart(toolbar, false, false, 0);

            // 2. Main Content (Viewport + Side Panel)
            var hPaned = new Paned(Orientation.Horizontal);
            mainVBox.PackStart(hPaned, true, true, 0);

            // 3D Viewport
            _viewport = new ThreeDView();
            hPaned.Pack1(_viewport, true, false);

            // 3. Status Bar (initialize before CreateSidePanel which uses _statusLabel)
            _statusLabel = new Label("Ready");

            // Side Panel
            var sidePanel = CreateSidePanel();
            hPaned.Pack2(sidePanel, false, false);
            hPaned.Position = 900;
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
            // ToolButton(Widget icon, string label)
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
            // ToggleToolButton constructors are weird in GtkSharp?
            // Usually: new ToggleToolButton() -> then set IconWidget and Label.

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
                    _rgbColorToggle.Active = true; // Keep one active
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
                    _depthColorToggle.Active = true; // Keep one active
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

            var label = new Label("Mesh Tools");
            label.Attributes = new Pango.AttrList();
            label.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(label, false, false, 5);

            // Tools with 3D handles
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

            // Add image file filter
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
                var md = new MessageDialog(this, DialogFlags.Modal, MessageType.Info, ButtonsType.Ok, "Please add at least 2 images.");
                md.Run();
                md.Destroy();
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

                // Store result and populate depth data for image browser
                _lastSceneResult = result;
                PopulateDepthData(result);

                if (workflow == "Dust3r (Fast)")
                {
                    _statusLabel.Text = $"Meshing using {AppSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                         // 1. Voxelize
                         var (grid, min, size) = VoxelizePoints(result.Meshes);
                         // 2. Mesh
                         IMesher mesher = GetMesher(AppSettings.Instance.MeshingAlgo);
                         return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    _viewport.SetMeshes(new List<MeshData> { meshedResult });
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

                    _viewport.SetMeshes(new List<MeshData> { nerfMesh });
                    _statusLabel.Text = "NeRF Complete.";
                }
            }
            catch (Exception ex)
            {
                 _statusLabel.Text = "Error: " + ex.Message;
                 Console.WriteLine(ex);
            }
        }

        /// <summary>
        /// Populates depth data for all images in the browser from the reconstruction result.
        /// </summary>
        private void PopulateDepthData(SceneResult result)
        {
            if (result.Poses.Count == 0) return;

            for (int i = 0; i < result.Poses.Count && i < result.Meshes.Count; i++)
            {
                var pose = result.Poses[i];
                var mesh = result.Meshes[i];

                // Extract depth map for this view
                var depthMap = ExtractDepthMap(mesh, pose.Width, pose.Height, pose.WorldToCamera);

                // Set on image browser (by index matching order added)
                _imageBrowser.SetDepthData(i, depthMap);
            }

            // Refresh display to show depth is now available
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

        /// <summary>
        /// Extracts depth map from a mesh for a given image size.
        /// Uses the PixelToVertexIndex mapping if available.
        /// </summary>
        private float[,] ExtractDepthMap(MeshData mesh, int width, int height, OpenTK.Mathematics.Matrix4 worldToCamera)
        {
            float[,] depthMap = new float[width, height];

            // Initialize with max value (no depth)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    depthMap[x, y] = float.MaxValue;

            // If we have pixel-to-vertex mapping, use it directly
            // PixelToVertexIndex is a 1D array with index = y * width + x
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
                            // Transform to camera space
                            var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);
                            depthMap[x, y] = -vCam.Z; // Depth is positive distance along camera Z
                        }
                    }
                }
            }
            else
            {
                // Fallback: project all vertices and rasterize
                foreach (var v in mesh.Vertices)
                {
                    var vCam = OpenTK.Mathematics.Vector3.TransformPosition(v, worldToCamera);

                    // Simple orthographic projection for depth visualization
                    // Scale to image coordinates (assumes normalized coordinates)
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
    }
}
