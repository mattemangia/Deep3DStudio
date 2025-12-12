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
        private ColorButton _bgColorButton;
        private ColorButton _gridColorButton;

        public SettingsDialog(Gtk.Window parent) : base("Settings", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(400, 400);

            var vbox = this.ContentArea;
            vbox.Margin = 20;
            vbox.Spacing = 10;

            // Device Selection
            vbox.PackStart(new Label("Compute Device:") { Halign = Align.Start }, false, false, 0);
            _deviceCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(ComputeDevice)))
                _deviceCombo.AppendText(name);
            _deviceCombo.Active = (int)AppSettings.Instance.Device;
            vbox.PackStart(_deviceCombo, false, false, 0);

            // Meshing Algorithm
            vbox.PackStart(new Label("Meshing Algorithm:") { Halign = Align.Start }, false, false, 0);
            _meshingCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshingAlgorithm)))
                _meshingCombo.AppendText(name);
            _meshingCombo.Active = (int)AppSettings.Instance.MeshingAlgo;
            vbox.PackStart(_meshingCombo, false, false, 0);

            // Coordinate System
            vbox.PackStart(new Label("Coordinate System:") { Halign = Align.Start }, false, false, 0);
            _coordCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(CoordinateSystem)))
                _coordCombo.AppendText(name);
            _coordCombo.Active = (int)AppSettings.Instance.CoordSystem;
            vbox.PackStart(_coordCombo, false, false, 0);

            // Separator
            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 10);

            // Viewport Colors Section
            vbox.PackStart(new Label("<b>Viewport Colors</b>") { UseMarkup = true, Halign = Align.Start }, false, false, 0);

            // Background Color
            var bgBox = new Box(Orientation.Horizontal, 10);
            bgBox.PackStart(new Label("Background Color:") { Halign = Align.Start }, false, false, 0);
            _bgColorButton = new ColorButton();
            _bgColorButton.Rgba = new RGBA
            {
                Red = AppSettings.Instance.ViewportBgR,
                Green = AppSettings.Instance.ViewportBgG,
                Blue = AppSettings.Instance.ViewportBgB,
                Alpha = 1.0
            };
            bgBox.PackEnd(_bgColorButton, false, false, 0);
            vbox.PackStart(bgBox, false, false, 0);

            // Grid Color
            var gridBox = new Box(Orientation.Horizontal, 10);
            gridBox.PackStart(new Label("Grid Color:") { Halign = Align.Start }, false, false, 0);
            _gridColorButton = new ColorButton();
            _gridColorButton.Rgba = new RGBA
            {
                Red = AppSettings.Instance.GridColorR,
                Green = AppSettings.Instance.GridColorG,
                Blue = AppSettings.Instance.GridColorB,
                Alpha = 1.0
            };
            gridBox.PackEnd(_gridColorButton, false, false, 0);
            vbox.PackStart(gridBox, false, false, 0);

            // Buttons
            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Save", ResponseType.Ok);

            this.ShowAll();
        }

        public void SaveSettings()
        {
            AppSettings.Instance.Device = (ComputeDevice)_deviceCombo.Active;
            AppSettings.Instance.MeshingAlgo = (MeshingAlgorithm)_meshingCombo.Active;
            AppSettings.Instance.CoordSystem = (CoordinateSystem)_coordCombo.Active;

            // Save colors
            var bgColor = _bgColorButton.Rgba;
            AppSettings.Instance.ViewportBgR = (float)bgColor.Red;
            AppSettings.Instance.ViewportBgG = (float)bgColor.Green;
            AppSettings.Instance.ViewportBgB = (float)bgColor.Blue;

            var gridColor = _gridColorButton.Rgba;
            AppSettings.Instance.GridColorR = (float)gridColor.Red;
            AppSettings.Instance.GridColorG = (float)gridColor.Green;
            AppSettings.Instance.GridColorB = (float)gridColor.Blue;

            AppSettings.Instance.Save();
        }
    }
}
