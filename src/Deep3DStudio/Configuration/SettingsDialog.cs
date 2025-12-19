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
        private ComboBoxText _meshRefineCombo;
        private ColorButton _bgColorButton;
        private ColorButton _gridColorButton;

        public SettingsDialog(Gtk.Window parent) : base("Settings", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(450, 400);

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
            generalBox.PackStart(new Label("Default Meshing Algorithm:") { Halign = Align.Start }, false, false, 0);
            _meshingCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshingAlgorithm)))
                _meshingCombo.AppendText(name);
            _meshingCombo.Active = (int)IniSettings.Instance.MeshingAlgo;
            generalBox.PackStart(_meshingCombo, false, false, 0);

            // Mesh Refinement
            generalBox.PackStart(new Label("Default Refinement Method:") { Halign = Align.Start }, false, false, 0);
            _meshRefineCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshRefinementMethod)))
                _meshRefineCombo.AppendText(name);
            _meshRefineCombo.Active = (int)IniSettings.Instance.MeshRefinement;
            generalBox.PackStart(_meshRefineCombo, false, false, 0);

            // Reconstruction Method
            generalBox.PackStart(new Label("Default Reconstruction:") { Halign = Align.Start }, false, false, 0);
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
            _meshRefineCombo.Active = (int)IniSettings.Instance.MeshRefinement;

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
            settings.MeshRefinement = (MeshRefinementMethod)_meshRefineCombo.Active;

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
