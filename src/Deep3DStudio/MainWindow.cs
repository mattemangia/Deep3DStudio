using System;
using Gtk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
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
        private ListStore _imgListStore;

        // UI References for updates
        private ComboBoxText _workflowCombo;
        private ToggleToolButton _pointsToggle;
        private ToggleToolButton _wireToggle;
        private ToggleToolButton _meshToggle;

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

            // Side Panel
            var sidePanel = CreateSidePanel();
            hPaned.Pack2(sidePanel, false, false);
            hPaned.Position = 900;

            // 3. Status Bar
            _statusLabel = new Label("Ready");
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

            var imgLabel = new Label("Input Images");
            imgLabel.Attributes = new Pango.AttrList();
            imgLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(imgLabel, false, false, 5);

            var scrolled = new ScrolledWindow();
            scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);

            _imgListStore = new ListStore(typeof(string));
            var treeView = new TreeView(_imgListStore);
            treeView.AppendColumn("Filename", new CellRendererText(), "text", 0);

            scrolled.Add(treeView);
            vbox.PackStart(scrolled, true, true, 0);

            _inference = new Model.Dust3rInference();
            if (_inference.IsLoaded) _statusLabel.Text = "Model Loaded";
            else _statusLabel.Text = "Model Not Found - Check dust3r.onnx";

            return vbox;
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
            _viewport.QueueDraw();
        }

        private void OnAddImages(object sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Choose Images", this, FileChooserAction.Open, "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);
            fc.SelectMultiple = true;
            if (fc.Run() == (int)ResponseType.Accept)
            {
                foreach (var f in fc.Filenames)
                {
                    _imagePaths.Add(f);
                    _imgListStore.AppendValues(System.IO.Path.GetFileName(f));
                }
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
