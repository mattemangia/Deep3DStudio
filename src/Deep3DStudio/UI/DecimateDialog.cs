using System;
using Gtk;

namespace Deep3DStudio.UI
{
    public class DecimateDialog : Dialog
    {
        private RadioButton _rbPercentage;
        private RadioButton _rbUniform;
        private Scale _scalePercentage;
        private SpinButton _spinVoxelSize;

        public float Ratio => (float)_scalePercentage.Value / 100.0f;
        public float VoxelSize => (float)_spinVoxelSize.Value;
        public bool IsUniform => _rbUniform.Active;

        public DecimateDialog(Window parent) : base("Decimate Mesh", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(300, 200);
            this.Resizable = false;

            var vbox = this.ContentArea;
            vbox.Spacing = 10;
            vbox.BorderWidth = 10;

            // Percentage Option
            _rbPercentage = new RadioButton("Target Vertex Count");
            vbox.PackStart(_rbPercentage, false, false, 0);

            var hboxPerc = new HBox();
            hboxPerc.PackStart(new Label("Percentage (%):"), false, false, 5);
            _scalePercentage = new Scale(Orientation.Horizontal, 1, 99, 1);
            _scalePercentage.Value = 50;
            _scalePercentage.DrawValue = true;
            hboxPerc.PackStart(_scalePercentage, true, true, 5);
            vbox.PackStart(hboxPerc, false, false, 0);

            // Uniform Option
            _rbUniform = new RadioButton(_rbPercentage, "Voxel Grid (Uniform)");
            vbox.PackStart(_rbUniform, false, false, 0);

            var hboxVox = new HBox();
            hboxVox.PackStart(new Label("Voxel Size:"), false, false, 5);
            _spinVoxelSize = new SpinButton(0.001, 10.0, 0.001);
            _spinVoxelSize.Value = 0.01;
            _spinVoxelSize.Digits = 3;
            hboxVox.PackStart(_spinVoxelSize, true, true, 5);
            vbox.PackStart(hboxVox, false, false, 0);

            // Toggle sensitivity
            _rbPercentage.Toggled += (s, e) => UpdateSensitivity();
            UpdateSensitivity();

            // Buttons
            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Decimate", ResponseType.Ok);

            this.ShowAll();
        }

        private void UpdateSensitivity()
        {
            _scalePercentage.Sensitive = _rbPercentage.Active;
            _spinVoxelSize.Sensitive = _rbUniform.Active;
        }
    }
}
