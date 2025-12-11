using System;
using Gtk;

namespace Deep3DStudio.Configuration
{
    public class SettingsDialog : Dialog
    {
        private ComboBoxText _deviceCombo;
        private ComboBoxText _meshingCombo;
        private ComboBoxText _coordCombo;

        public SettingsDialog(Window parent) : base("Settings", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(400, 300);

            var vbox = this.ContentArea;
            vbox.Margin = 20;
            vbox.Spacing = 10;

            // Device Selection
            vbox.PackStart(new Label("Compute Device:"), false, false, 0);
            _deviceCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(ComputeDevice)))
                _deviceCombo.AppendText(name);
            _deviceCombo.Active = (int)AppSettings.Instance.Device;
            vbox.PackStart(_deviceCombo, false, false, 0);

            // Meshing Algorithm
            vbox.PackStart(new Label("Meshing Algorithm:"), false, false, 0);
            _meshingCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshingAlgorithm)))
                _meshingCombo.AppendText(name);
            _meshingCombo.Active = (int)AppSettings.Instance.MeshingAlgo;
            vbox.PackStart(_meshingCombo, false, false, 0);

            // Coordinate System
            vbox.PackStart(new Label("Coordinate System:"), false, false, 0);
            _coordCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(CoordinateSystem)))
                _coordCombo.AppendText(name);
            _coordCombo.Active = (int)AppSettings.Instance.CoordSystem;
            vbox.PackStart(_coordCombo, false, false, 0);

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
            AppSettings.Instance.Save();
        }
    }
}
