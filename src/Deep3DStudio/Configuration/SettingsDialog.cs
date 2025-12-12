using System;
using Gtk;
using Gdk;

namespace Deep3DStudio.Configuration
{
    public class SettingsDialog : Dialog
    {
        private ComboBoxText _deviceCombo;
        private ComboBoxText _meshingCombo;
        private ComboBoxText _coordCombo;
        private ComboBoxText _bboxCombo;
        private ComboBoxText _reconCombo;
        private ColorButton _bgColorButton;
        private ColorButton _gridColorButton;

        // NeRF settings
        private SpinButton _nerfIterationsSpin;
        private SpinButton _voxelSizeSpin;
        private SpinButton _learningRateSpin;

        public SettingsDialog(Gtk.Window parent) : base("Settings", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(450, 550);

            var vbox = this.ContentArea;
            vbox.Margin = 20;
            vbox.Spacing = 10;

            // Create notebook for tabbed settings
            var notebook = new Notebook();
            vbox.PackStart(notebook, true, true, 0);

            // ===== General Tab =====
            var generalBox = new Box(Orientation.Vertical, 8);
            generalBox.Margin = 10;

            // Device Selection
            generalBox.PackStart(new Label("Compute Device:") { Halign = Align.Start }, false, false, 0);
            _deviceCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(ComputeDevice)))
                _deviceCombo.AppendText(name);
            _deviceCombo.Active = (int)IniSettings.Instance.Device;
            generalBox.PackStart(_deviceCombo, false, false, 0);

            // Meshing Algorithm
            generalBox.PackStart(new Label("Meshing Algorithm:") { Halign = Align.Start }, false, false, 0);
            _meshingCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshingAlgorithm)))
                _meshingCombo.AppendText(name);
            _meshingCombo.Active = (int)IniSettings.Instance.MeshingAlgo;
            generalBox.PackStart(_meshingCombo, false, false, 0);

            // Reconstruction Method
            generalBox.PackStart(new Label("Reconstruction Method:") { Halign = Align.Start }, false, false, 0);
            _reconCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(ReconstructionMethod)))
                _reconCombo.AppendText(name);
            _reconCombo.Active = (int)IniSettings.Instance.ReconstructionMethod;
            generalBox.PackStart(_reconCombo, false, false, 0);

            // Coordinate System
            generalBox.PackStart(new Label("Coordinate System:") { Halign = Align.Start }, false, false, 0);
            _coordCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(CoordinateSystem)))
                _coordCombo.AppendText(name);
            _coordCombo.Active = (int)IniSettings.Instance.CoordSystem;
            generalBox.PackStart(_coordCombo, false, false, 0);

            // Bounding Box Style
            generalBox.PackStart(new Label("Bounding Box Style:") { Halign = Align.Start }, false, false, 0);
            _bboxCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(BoundingBoxMode)))
                _bboxCombo.AppendText(name);
            _bboxCombo.Active = (int)IniSettings.Instance.BoundingBoxStyle;
            generalBox.PackStart(_bboxCombo, false, false, 0);

            notebook.AppendPage(generalBox, new Label("General"));

            // ===== NeRF Tab =====
            var nerfBox = new Box(Orientation.Vertical, 8);
            nerfBox.Margin = 10;

            nerfBox.PackStart(new Label("<b>NeRF Training Parameters</b>") { UseMarkup = true, Halign = Align.Start }, false, false, 5);

            // Iterations
            var iterBox = new Box(Orientation.Horizontal, 10);
            iterBox.PackStart(new Label("Training Iterations:") { Halign = Align.Start }, false, false, 0);
            _nerfIterationsSpin = new SpinButton(1, 500, 5);
            _nerfIterationsSpin.Value = IniSettings.Instance.NeRFIterations;
            _nerfIterationsSpin.TooltipText = "Number of training iterations for NeRF refinement (default: 50)";
            iterBox.PackEnd(_nerfIterationsSpin, false, false, 0);
            nerfBox.PackStart(iterBox, false, false, 0);

            // Voxel Grid Size
            var voxelBox = new Box(Orientation.Horizontal, 10);
            voxelBox.PackStart(new Label("Voxel Grid Size:") { Halign = Align.Start }, false, false, 0);
            _voxelSizeSpin = new SpinButton(32, 512, 16);
            _voxelSizeSpin.Value = IniSettings.Instance.VoxelGridSize;
            _voxelSizeSpin.TooltipText = "Resolution of the voxel grid (default: 128). Higher = more detail but slower.";
            voxelBox.PackEnd(_voxelSizeSpin, false, false, 0);
            nerfBox.PackStart(voxelBox, false, false, 0);

            // Learning Rate
            var lrBox = new Box(Orientation.Horizontal, 10);
            lrBox.PackStart(new Label("Learning Rate:") { Halign = Align.Start }, false, false, 0);
            _learningRateSpin = new SpinButton(0.001, 1.0, 0.01);
            _learningRateSpin.Digits = 3;
            _learningRateSpin.Value = IniSettings.Instance.NeRFLearningRate;
            _learningRateSpin.TooltipText = "SGD learning rate for NeRF optimization (default: 0.1)";
            lrBox.PackEnd(_learningRateSpin, false, false, 0);
            nerfBox.PackStart(lrBox, false, false, 0);

            // Info label
            var infoLabel = new Label("<small>Note: Higher iterations and grid size require more time and memory.</small>")
            {
                UseMarkup = true,
                Halign = Align.Start,
                Wrap = true
            };
            nerfBox.PackStart(infoLabel, false, false, 10);

            notebook.AppendPage(nerfBox, new Label("NeRF"));

            // ===== Viewport Tab =====
            var viewportBox = new Box(Orientation.Vertical, 8);
            viewportBox.Margin = 10;

            viewportBox.PackStart(new Label("<b>Viewport Colors</b>") { UseMarkup = true, Halign = Align.Start }, false, false, 5);

            // Background Color
            var bgBox = new Box(Orientation.Horizontal, 10);
            bgBox.PackStart(new Label("Background Color:") { Halign = Align.Start }, false, false, 0);
            _bgColorButton = new ColorButton();
            _bgColorButton.Rgba = new RGBA
            {
                Red = IniSettings.Instance.ViewportBgR,
                Green = IniSettings.Instance.ViewportBgG,
                Blue = IniSettings.Instance.ViewportBgB,
                Alpha = 1.0
            };
            bgBox.PackEnd(_bgColorButton, false, false, 0);
            viewportBox.PackStart(bgBox, false, false, 0);

            // Grid Color
            var gridBox = new Box(Orientation.Horizontal, 10);
            gridBox.PackStart(new Label("Grid Color:") { Halign = Align.Start }, false, false, 0);
            _gridColorButton = new ColorButton();
            _gridColorButton.Rgba = new RGBA
            {
                Red = IniSettings.Instance.GridColorR,
                Green = IniSettings.Instance.GridColorG,
                Blue = IniSettings.Instance.GridColorB,
                Alpha = 1.0
            };
            gridBox.PackEnd(_gridColorButton, false, false, 0);
            viewportBox.PackStart(gridBox, false, false, 0);

            notebook.AppendPage(viewportBox, new Label("Viewport"));

            // ===== Info Tab =====
            var infoBox = new Box(Orientation.Vertical, 8);
            infoBox.Margin = 10;

            infoBox.PackStart(new Label("<b>Settings Location</b>") { UseMarkup = true, Halign = Align.Start }, false, false, 5);

            var pathLabel = new Label($"<small>{IniSettings.GetSettingsPath()}</small>")
            {
                UseMarkup = true,
                Halign = Align.Start,
                Selectable = true,
                Wrap = true
            };
            infoBox.PackStart(pathLabel, false, false, 0);

            infoBox.PackStart(new Separator(Orientation.Horizontal), false, false, 10);

            // Reset to defaults button
            var resetBtn = new Button("Reset to Defaults");
            resetBtn.Clicked += (s, e) => {
                var confirmDialog = new MessageDialog(this, DialogFlags.Modal, MessageType.Question, ButtonsType.YesNo,
                    "Reset all settings to their default values?");
                if (confirmDialog.Run() == (int)ResponseType.Yes)
                {
                    IniSettings.Instance.ResetToDefaults();
                    RefreshUIFromSettings();
                }
                confirmDialog.Destroy();
            };
            infoBox.PackStart(resetBtn, false, false, 5);

            notebook.AppendPage(infoBox, new Label("Info"));

            // Buttons
            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Save", ResponseType.Ok);

            this.ShowAll();
        }

        /// <summary>
        /// Refreshes the UI controls from current settings values.
        /// </summary>
        private void RefreshUIFromSettings()
        {
            _deviceCombo.Active = (int)IniSettings.Instance.Device;
            _meshingCombo.Active = (int)IniSettings.Instance.MeshingAlgo;
            _reconCombo.Active = (int)IniSettings.Instance.ReconstructionMethod;
            _coordCombo.Active = (int)IniSettings.Instance.CoordSystem;
            _bboxCombo.Active = (int)IniSettings.Instance.BoundingBoxStyle;

            _nerfIterationsSpin.Value = IniSettings.Instance.NeRFIterations;
            _voxelSizeSpin.Value = IniSettings.Instance.VoxelGridSize;
            _learningRateSpin.Value = IniSettings.Instance.NeRFLearningRate;

            _bgColorButton.Rgba = new RGBA
            {
                Red = IniSettings.Instance.ViewportBgR,
                Green = IniSettings.Instance.ViewportBgG,
                Blue = IniSettings.Instance.ViewportBgB,
                Alpha = 1.0
            };

            _gridColorButton.Rgba = new RGBA
            {
                Red = IniSettings.Instance.GridColorR,
                Green = IniSettings.Instance.GridColorG,
                Blue = IniSettings.Instance.GridColorB,
                Alpha = 1.0
            };
        }

        /// <summary>
        /// Saves the dialog values to the IniSettings and persists to disk.
        /// </summary>
        public void SaveSettings()
        {
            var settings = IniSettings.Instance;

            // General settings
            settings.Device = (ComputeDevice)_deviceCombo.Active;
            settings.MeshingAlgo = (MeshingAlgorithm)_meshingCombo.Active;
            settings.ReconstructionMethod = (ReconstructionMethod)_reconCombo.Active;
            settings.CoordSystem = (CoordinateSystem)_coordCombo.Active;
            settings.BoundingBoxStyle = (BoundingBoxMode)_bboxCombo.Active;

            // NeRF settings
            settings.NeRFIterations = (int)_nerfIterationsSpin.Value;
            settings.VoxelGridSize = (int)_voxelSizeSpin.Value;
            settings.NeRFLearningRate = (float)_learningRateSpin.Value;

            // Viewport colors
            var bgColor = _bgColorButton.Rgba;
            settings.ViewportBgR = (float)bgColor.Red;
            settings.ViewportBgG = (float)bgColor.Green;
            settings.ViewportBgB = (float)bgColor.Blue;

            var gridColor = _gridColorButton.Rgba;
            settings.GridColorR = (float)gridColor.Red;
            settings.GridColorG = (float)gridColor.Green;
            settings.GridColorB = (float)gridColor.Blue;

            // Save to INI file
            settings.Save();
        }
    }
}
