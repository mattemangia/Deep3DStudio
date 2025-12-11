using System;
using Gtk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using System.Collections.Generic;

namespace Deep3DStudio
{
    public class MainWindow : Window
    {
        private ThreeDView _viewport;
        private Label _statusLabel;
        private Model.Dust3rInference _inference;
        private List<string> _imagePaths = new List<string>();
        private ListStore _imgListStore;

        // Workflow Selection
        private ComboBoxText _workflowCombo;

        public MainWindow() : base(WindowType.Toplevel)
        {
            this.Title = "Deep3D Studio - Dust3r & NeRF";
            this.SetDefaultSize(1200, 800);
            this.DeleteEvent += (o, args) => Application.Quit();

            var mainVBox = new Box(Orientation.Vertical, 0);
            this.Add(mainVBox);

            // Toolbar
            var toolbar = CreateToolbar();
            mainVBox.PackStart(toolbar, false, false, 0);

            // Viewport and Side Panel (using Paned for resizing)
            var hPaned = new Paned(Orientation.Horizontal);
            mainVBox.PackStart(hPaned, true, true, 0);

            // 3D Viewport
            _viewport = new ThreeDView();
            hPaned.Pack1(_viewport, true, false);

            // Side Panel for Mesh Tools and Image List
            var sidePanel = CreateSidePanel();
            hPaned.Pack2(sidePanel, false, false);
            hPaned.Position = 900;

            // Status Bar
            _statusLabel = new Label("Ready");
            _statusLabel.Halign = Align.Start;
            var statusBox = new Box(Orientation.Horizontal, 5);
            statusBox.PackStart(_statusLabel, true, true, 5);
            mainVBox.PackStart(statusBox, false, false, 2);

            this.ShowAll();
        }

        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;

            // Add Image Button
            var addImgBtn = new ToolButton(IconGenerator.GenerateAddIcon(24), "Add Images");
            addImgBtn.TooltipText = "Add multiple images for processing";
            addImgBtn.Clicked += OnAddImages;
            toolbar.Insert(addImgBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Workflow Selection
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

            // Run Button
            var runBtn = new ToolButton(IconGenerator.GenerateRunIcon(24), "Run Reconstruction");
            runBtn.TooltipText = "Run selected reconstruction workflow";
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
            var cropBtn = new Button("Show Crop Box");
            cropBtn.Clicked += (s, e) => {
                 _viewport.ToggleCropBox(true);
            };
            vbox.PackStart(cropBtn, false, false, 2);

            var applyCropBtn = new Button("Apply Crop");
            applyCropBtn.Clicked += (s, e) => {
                _viewport.ApplyCrop();
            };
            vbox.PackStart(applyCropBtn, false, false, 2);

            var editBtn = new Button("Edit Vertices");
            vbox.PackStart(editBtn, false, false, 2);

            var optimizeBtn = new Button("Optimize Mesh");
            vbox.PackStart(optimizeBtn, false, false, 2);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 10);

            var imgLabel = new Label("Input Images");
            imgLabel.Attributes = new Pango.AttrList();
            imgLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            vbox.PackStart(imgLabel, false, false, 5);

            // Placeholder for image list
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
            _statusLabel.Text = $"Running {workflow} Workflow...";
            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                // Step 1: Run Dust3r (Required for both workflows: Poses or full mesh)
                _statusLabel.Text = "Estimating Geometry and Poses (Dust3r)...";
                var result = await System.Threading.Tasks.Task.Run(() => _inference.ReconstructScene(_imagePaths));

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Dust3r Inference failed.";
                    return;
                }

                if (workflow == "Dust3r (Fast)")
                {
                    _viewport.SetMeshes(result.Meshes);
                    _statusLabel.Text = $"Dust3r Complete. Meshes: {result.Meshes.Count}";
                }
                else if (workflow == "NeRF (Refined)")
                {
                    _statusLabel.Text = "Initializing NeRF Voxel Grid...";
                    var nerf = new VoxelGridNeRF();

                    await System.Threading.Tasks.Task.Run(() =>
                    {
                        // Initialize from Dust3r coarse geometry
                        nerf.InitializeFromMesh(result.Meshes);

                        // Train NeRF
                        // Update status from thread? (Careful with UI thread)
                        // Just run for now.
                        nerf.Train(result.Poses, iterations: 10);
                    });

                    _statusLabel.Text = "Extracting NeRF Mesh...";
                    var nerfMesh = await System.Threading.Tasks.Task.Run(() => nerf.GetMesh());

                    if (nerfMesh.Vertices.Count > 0)
                    {
                        _viewport.SetMeshes(new List<MeshData> { nerfMesh });
                        _statusLabel.Text = $"NeRF Complete. Vertices: {nerfMesh.Vertices.Count}";
                    }
                    else
                    {
                        _statusLabel.Text = "NeRF extraction failed (empty mesh).";
                    }
                }
            }
            catch (Exception ex)
            {
                 _statusLabel.Text = "Error: " + ex.Message;
                 Console.WriteLine(ex);
            }
        }
    }
}
