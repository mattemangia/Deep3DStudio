using System;
using System.Collections.Generic;
using System.Linq;
using Gtk;
using Deep3DStudio.Scene;
using Deep3DStudio.Texturing;
using Deep3DStudio.IO;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// Dialog for configuring texture baking operations
    /// </summary>
    public class TextureBakingDialog : Dialog
    {
        // UV Generation
        private ComboBoxText _uvMethodCombo;
        private SpinButton _textureSizeSpin;
        private SpinButton _islandMarginSpin;

        // Projection Settings
        private ComboBoxText _blendModeCombo;
        private SpinButton _minViewAngleSpin;
        private CheckButton _blendSeamsCheck;
        private SpinButton _dilationPassesSpin;

        // Export Settings
        private ComboBoxText _exportFormatCombo;
        private ComboBoxText _textureFormatCombo;
        private SpinButton _jpegQualitySpin;
        private CheckButton _exportNormalsCheck;
        private CheckButton _swapYZCheck;

        // Camera list
        private TreeView _cameraTreeView;
        private ListStore _cameraStore;
        private List<CameraObject> _availableCameras;

        // Source options
        private RadioButton _bakeFromCamerasRadio;
        private RadioButton _bakeFromVertexColorsRadio;

        // Results
        public UVUnwrapMethod UVMethod { get; private set; } = UVUnwrapMethod.SmartProject;
        public int TextureSize { get; private set; } = 2048;
        public TextureBlendMode BlendMode { get; private set; } = TextureBlendMode.ViewAngleWeighted;
        public MeshExportOptions ExportOptions { get; private set; } = new MeshExportOptions();
        public bool BakeFromCameras { get; private set; } = true;
        public List<CameraObject> SelectedCameras { get; private set; } = new List<CameraObject>();
        public TextureBakerSettings BakerSettings { get; private set; } = new TextureBakerSettings();

        public TextureBakingDialog(Window parent, List<CameraObject> availableCameras)
            : base("Texture Baking", parent, DialogFlags.Modal)
        {
            _availableCameras = availableCameras;
            SetDefaultSize(700, 650);
            BuildUI(availableCameras);
        }

        private void BuildUI(List<CameraObject> cameras)
        {
            var contentArea = ContentArea;
            contentArea.Margin = 10;
            contentArea.Spacing = 10;

            // Main notebook for organized tabs
            var notebook = new Notebook();

            // Tab 1: Source & UV Settings
            var sourceTab = CreateSourceTab(cameras);
            notebook.AppendPage(sourceTab, new Label("Source & UV"));

            // Tab 2: Baking Settings
            var bakingTab = CreateBakingTab();
            notebook.AppendPage(bakingTab, new Label("Baking"));

            // Tab 3: Export Settings
            var exportTab = CreateExportTab();
            notebook.AppendPage(exportTab, new Label("Export"));

            contentArea.PackStart(notebook, true, true, 0);

            // Dialog buttons
            AddButton("Cancel", ResponseType.Cancel);
            AddButton("Bake Textures", ResponseType.Ok);

            ShowAll();
        }

        private Widget CreateSourceTab(List<CameraObject> cameras)
        {
            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 10;

            // Source selection
            var sourceFrame = new Frame("Texture Source");
            var sourceBox = new Box(Orientation.Vertical, 5);
            sourceBox.Margin = 10;

            _bakeFromCamerasRadio = new RadioButton("Project from camera images");
            _bakeFromVertexColorsRadio = new RadioButton(_bakeFromCamerasRadio, "Bake vertex colors to texture");
            _bakeFromCamerasRadio.Active = true;

            _bakeFromCamerasRadio.Toggled += OnSourceChanged;

            sourceBox.PackStart(_bakeFromCamerasRadio, false, false, 0);
            sourceBox.PackStart(_bakeFromVertexColorsRadio, false, false, 0);
            sourceFrame.Add(sourceBox);
            vbox.PackStart(sourceFrame, false, false, 0);

            // Camera selection
            var cameraFrame = new Frame("Camera Selection");
            var cameraBox = new Box(Orientation.Vertical, 5);
            cameraBox.Margin = 10;

            var cameraLabel = new Label($"Available cameras with images: {cameras.Count(c => !string.IsNullOrEmpty(c.ImagePath))}");
            cameraLabel.Halign = Align.Start;
            cameraBox.PackStart(cameraLabel, false, false, 0);

            // Camera list with checkboxes
            var scrolled = new ScrolledWindow();
            scrolled.SetSizeRequest(-1, 150);
            scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);

            _cameraStore = new ListStore(typeof(bool), typeof(string), typeof(string), typeof(int));
            _cameraTreeView = new TreeView(_cameraStore);

            var toggleRenderer = new CellRendererToggle();
            toggleRenderer.Toggled += OnCameraToggled;
            var toggleColumn = new TreeViewColumn("Use", toggleRenderer, "active", 0);
            _cameraTreeView.AppendColumn(toggleColumn);

            _cameraTreeView.AppendColumn("Name", new CellRendererText(), "text", 1);
            _cameraTreeView.AppendColumn("Image", new CellRendererText(), "text", 2);

            foreach (var cam in cameras)
            {
                bool hasImage = !string.IsNullOrEmpty(cam.ImagePath);
                string imageName = hasImage ? System.IO.Path.GetFileName(cam.ImagePath) : "(no image)";
                _cameraStore.AppendValues(hasImage, cam.Name, imageName, cam.Id);
            }

            scrolled.Add(_cameraTreeView);
            cameraBox.PackStart(scrolled, true, true, 0);

            // Select all/none buttons
            var selectionBox = new Box(Orientation.Horizontal, 5);
            var selectAllBtn = new Button("Select All");
            selectAllBtn.Clicked += (s, e) => SetAllCameras(true);
            var selectNoneBtn = new Button("Select None");
            selectNoneBtn.Clicked += (s, e) => SetAllCameras(false);
            selectionBox.PackStart(selectAllBtn, false, false, 0);
            selectionBox.PackStart(selectNoneBtn, false, false, 0);
            cameraBox.PackStart(selectionBox, false, false, 0);

            cameraFrame.Add(cameraBox);
            vbox.PackStart(cameraFrame, true, true, 0);

            // UV Generation Settings
            var uvFrame = new Frame("UV Generation");
            var uvGrid = new Grid();
            uvGrid.Margin = 10;
            uvGrid.ColumnSpacing = 10;
            uvGrid.RowSpacing = 8;

            int row = 0;

            uvGrid.Attach(new Label("UV Method:") { Halign = Align.Start }, 0, row, 1, 1);
            _uvMethodCombo = new ComboBoxText();
            _uvMethodCombo.AppendText("Smart UV Project");
            _uvMethodCombo.AppendText("Lightmap Pack");
            _uvMethodCombo.AppendText("Box Projection");
            _uvMethodCombo.AppendText("Cylindrical Projection");
            _uvMethodCombo.AppendText("Spherical Projection");
            _uvMethodCombo.Active = 0;
            uvGrid.Attach(_uvMethodCombo, 1, row++, 1, 1);

            uvGrid.Attach(new Label("Texture Size:") { Halign = Align.Start }, 0, row, 1, 1);
            _textureSizeSpin = new SpinButton(256, 8192, 256);
            _textureSizeSpin.Value = 2048;
            uvGrid.Attach(_textureSizeSpin, 1, row++, 1, 1);

            uvGrid.Attach(new Label("Island Margin (px):") { Halign = Align.Start }, 0, row, 1, 1);
            _islandMarginSpin = new SpinButton(0, 32, 1);
            _islandMarginSpin.Value = 4;
            uvGrid.Attach(_islandMarginSpin, 1, row++, 1, 1);

            uvFrame.Add(uvGrid);
            vbox.PackStart(uvFrame, false, false, 0);

            return vbox;
        }

        private Widget CreateBakingTab()
        {
            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 10;

            // Projection Settings
            var projFrame = new Frame("Projection Settings");
            var projGrid = new Grid();
            projGrid.Margin = 10;
            projGrid.ColumnSpacing = 10;
            projGrid.RowSpacing = 8;

            int row = 0;

            projGrid.Attach(new Label("Blend Mode:") { Halign = Align.Start }, 0, row, 1, 1);
            _blendModeCombo = new ComboBoxText();
            _blendModeCombo.AppendText("Replace (Last camera wins)");
            _blendModeCombo.AppendText("Average (Equal weight)");
            _blendModeCombo.AppendText("View Angle Weighted (Recommended)");
            _blendModeCombo.AppendText("Distance Weighted");
            _blendModeCombo.Active = 2;
            projGrid.Attach(_blendModeCombo, 1, row++, 1, 1);

            projGrid.Attach(new Label("Min View Angle:") { Halign = Align.Start }, 0, row, 1, 1);
            _minViewAngleSpin = new SpinButton(0, 0.9, 0.05);
            _minViewAngleSpin.Value = 0.1;
            _minViewAngleSpin.Digits = 2;
            _minViewAngleSpin.TooltipText = "Cosine of minimum angle between face normal and view direction";
            projGrid.Attach(_minViewAngleSpin, 1, row++, 1, 1);

            projFrame.Add(projGrid);
            vbox.PackStart(projFrame, false, false, 0);

            // Post-Processing
            var postFrame = new Frame("Post-Processing");
            var postBox = new Box(Orientation.Vertical, 5);
            postBox.Margin = 10;

            _blendSeamsCheck = new CheckButton("Blend texture seams");
            _blendSeamsCheck.Active = true;
            _blendSeamsCheck.TooltipText = "Apply blur at UV seams to reduce visible boundaries";
            postBox.PackStart(_blendSeamsCheck, false, false, 0);

            var dilationBox = new Box(Orientation.Horizontal, 10);
            dilationBox.PackStart(new Label("Dilation passes:"), false, false, 0);
            _dilationPassesSpin = new SpinButton(0, 16, 1);
            _dilationPassesSpin.Value = 4;
            _dilationPassesSpin.TooltipText = "Fill transparent pixels by expanding neighboring colors";
            dilationBox.PackStart(_dilationPassesSpin, false, false, 0);
            postBox.PackStart(dilationBox, false, false, 0);

            postFrame.Add(postBox);
            vbox.PackStart(postFrame, false, false, 0);

            // Quality Tips
            var tipsFrame = new Frame("Quality Tips");
            var tipsLabel = new Label(
                "For best results:\n" +
                "- Use view angle weighted blending\n" +
                "- Ensure cameras have sufficient coverage\n" +
                "- Higher texture resolution = more detail but larger files\n" +
                "- Smart UV Project works best for organic shapes");
            tipsLabel.Halign = Align.Start;
            tipsLabel.Margin = 10;
            tipsLabel.Wrap = true;
            tipsFrame.Add(tipsLabel);
            vbox.PackStart(tipsFrame, false, false, 0);

            return vbox;
        }

        private Widget CreateExportTab()
        {
            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 10;

            // Mesh Export Format
            var formatFrame = new Frame("Export Format");
            var formatGrid = new Grid();
            formatGrid.Margin = 10;
            formatGrid.ColumnSpacing = 10;
            formatGrid.RowSpacing = 8;

            int row = 0;

            formatGrid.Attach(new Label("Mesh Format:") { Halign = Align.Start }, 0, row, 1, 1);
            _exportFormatCombo = new ComboBoxText();
            _exportFormatCombo.AppendText("OBJ (Wavefront)");
            _exportFormatCombo.AppendText("GLTF 2.0");
            _exportFormatCombo.AppendText("GLB (Binary GLTF)");
            _exportFormatCombo.AppendText("FBX (ASCII)");
            _exportFormatCombo.AppendText("PLY");
            _exportFormatCombo.Active = 0;
            formatGrid.Attach(_exportFormatCombo, 1, row++, 1, 1);

            formatGrid.Attach(new Label("Texture Format:") { Halign = Align.Start }, 0, row, 1, 1);
            _textureFormatCombo = new ComboBoxText();
            _textureFormatCombo.AppendText("PNG (Lossless)");
            _textureFormatCombo.AppendText("JPEG (Compressed)");
            _textureFormatCombo.AppendText("BMP");
            _textureFormatCombo.Active = 0;
            _textureFormatCombo.Changed += OnTextureFormatChanged;
            formatGrid.Attach(_textureFormatCombo, 1, row++, 1, 1);

            formatGrid.Attach(new Label("JPEG Quality:") { Halign = Align.Start }, 0, row, 1, 1);
            _jpegQualitySpin = new SpinButton(50, 100, 5);
            _jpegQualitySpin.Value = 90;
            _jpegQualitySpin.Sensitive = false;
            formatGrid.Attach(_jpegQualitySpin, 1, row++, 1, 1);

            formatFrame.Add(formatGrid);
            vbox.PackStart(formatFrame, false, false, 0);

            // Export Options
            var optionsFrame = new Frame("Export Options");
            var optionsBox = new Box(Orientation.Vertical, 5);
            optionsBox.Margin = 10;

            _exportNormalsCheck = new CheckButton("Export normals");
            _exportNormalsCheck.Active = true;
            optionsBox.PackStart(_exportNormalsCheck, false, false, 0);

            _swapYZCheck = new CheckButton("Swap Y/Z axes (for Z-up applications)");
            _swapYZCheck.Active = false;
            optionsBox.PackStart(_swapYZCheck, false, false, 0);

            optionsFrame.Add(optionsBox);
            vbox.PackStart(optionsFrame, false, false, 0);

            // Format Descriptions
            var descFrame = new Frame("Format Information");
            var descLabel = new Label(
                "<b>OBJ:</b> Universal format, separate MTL and texture files\n" +
                "<b>GLTF:</b> Modern web-ready format, good for game engines\n" +
                "<b>GLB:</b> Binary GLTF, single file with embedded textures\n" +
                "<b>FBX:</b> Industry standard for 3D software exchange\n" +
                "<b>PLY:</b> Point cloud format with UV support");
            descLabel.UseMarkup = true;
            descLabel.Halign = Align.Start;
            descLabel.Margin = 10;
            descLabel.Wrap = true;
            descFrame.Add(descLabel);
            vbox.PackStart(descFrame, false, false, 0);

            return vbox;
        }

        private void OnSourceChanged(object? sender, EventArgs e)
        {
            _cameraTreeView.Sensitive = _bakeFromCamerasRadio.Active;
            _blendModeCombo.Sensitive = _bakeFromCamerasRadio.Active;
            _minViewAngleSpin.Sensitive = _bakeFromCamerasRadio.Active;
        }

        private void OnCameraToggled(object o, ToggledArgs args)
        {
            if (_cameraStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                bool current = (bool)_cameraStore.GetValue(iter, 0);
                _cameraStore.SetValue(iter, 0, !current);
            }
        }

        private void SetAllCameras(bool selected)
        {
            if (_cameraStore.GetIterFirst(out TreeIter iter))
            {
                do
                {
                    _cameraStore.SetValue(iter, 0, selected);
                } while (_cameraStore.IterNext(ref iter));
            }
        }

        private void OnTextureFormatChanged(object? sender, EventArgs e)
        {
            _jpegQualitySpin.Sensitive = _textureFormatCombo.Active == 1;
        }

        protected override void OnResponse(ResponseType response_id)
        {
            if (response_id == ResponseType.Ok)
            {
                // UV Method
                UVMethod = _uvMethodCombo.Active switch
                {
                    0 => UVUnwrapMethod.SmartProject,
                    1 => UVUnwrapMethod.LightmapPack,
                    2 => UVUnwrapMethod.BoxProject,
                    3 => UVUnwrapMethod.CylindricalProject,
                    4 => UVUnwrapMethod.SphericalProject,
                    _ => UVUnwrapMethod.SmartProject
                };

                TextureSize = (int)_textureSizeSpin.Value;
                BakeFromCameras = _bakeFromCamerasRadio.Active;

                // Blend mode
                BlendMode = _blendModeCombo.Active switch
                {
                    0 => TextureBlendMode.Replace,
                    1 => TextureBlendMode.Average,
                    2 => TextureBlendMode.ViewAngleWeighted,
                    3 => TextureBlendMode.DistanceWeighted,
                    _ => TextureBlendMode.ViewAngleWeighted
                };

                // Baker settings
                BakerSettings = new TextureBakerSettings
                {
                    TextureSize = TextureSize,
                    IslandMargin = (int)_islandMarginSpin.Value,
                    BlendMode = BlendMode,
                    MinViewAngleCosine = (float)_minViewAngleSpin.Value,
                    BlendSeams = _blendSeamsCheck.Active,
                    DilationPasses = (int)_dilationPassesSpin.Value
                };

                // Export options
                ExportOptions = new MeshExportOptions
                {
                    Format = _exportFormatCombo.Active switch
                    {
                        0 => TexturedMeshFormat.OBJ,
                        1 => TexturedMeshFormat.GLTF,
                        2 => TexturedMeshFormat.GLB,
                        3 => TexturedMeshFormat.FBX_ASCII,
                        4 => TexturedMeshFormat.PLY,
                        _ => TexturedMeshFormat.OBJ
                    },
                    TextureFormat = _textureFormatCombo.Active switch
                    {
                        0 => TextureFormat.PNG,
                        1 => TextureFormat.JPEG,
                        2 => TextureFormat.BMP,
                        _ => TextureFormat.PNG
                    },
                    JpegQuality = (int)_jpegQualitySpin.Value,
                    ExportNormals = _exportNormalsCheck.Active,
                    SwapYZ = _swapYZCheck.Active,
                    ExportUVs = true,
                    ExportTextures = true
                };

                // Collect selected cameras
                SelectedCameras.Clear();

                if (_cameraStore.GetIterFirst(out TreeIter iter))
                {
                    do
                    {
                        bool selected = (bool)_cameraStore.GetValue(iter, 0);
                        if (selected)
                        {
                            int id = (int)_cameraStore.GetValue(iter, 3);
                            var cam = _availableCameras.FirstOrDefault(c => c.Id == id);
                            if (cam != null)
                                SelectedCameras.Add(cam);
                        }
                    } while (_cameraStore.IterNext(ref iter));
                }
            }

            base.OnResponse(response_id);
        }
    }

    /// <summary>
    /// Settings for texture baker
    /// </summary>
    public class TextureBakerSettings
    {
        public int TextureSize { get; set; } = 2048;
        public int IslandMargin { get; set; } = 4;
        public TextureBlendMode BlendMode { get; set; } = TextureBlendMode.ViewAngleWeighted;
        public float MinViewAngleCosine { get; set; } = 0.1f;
        public bool BlendSeams { get; set; } = true;
        public int DilationPasses { get; set; } = 4;
    }
}
