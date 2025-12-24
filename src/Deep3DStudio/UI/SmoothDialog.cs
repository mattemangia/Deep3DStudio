using System;
using Gtk;

namespace Deep3DStudio.UI
{
    public class SmoothDialog : Dialog
    {
        private RadioButton _rbLaplacian;
        private RadioButton _rbTaubin;
        private SpinButton _spinIterations;
        private SpinButton _spinLambda;
        private SpinButton _spinMu;

        public bool IsTaubin => _rbTaubin.Active;
        public int Iterations => (int)_spinIterations.Value;
        public float Lambda => (float)_spinLambda.Value;
        public float Mu => (float)_spinMu.Value;

        public SmoothDialog(Window parent) : base("Smooth Mesh", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(300, 250);
            this.Resizable = false;

            var vbox = this.ContentArea;
            vbox.Spacing = 10;
            vbox.BorderWidth = 10;

            // Method
            var frameMethod = new Frame("Smoothing Method");
            var vboxMethod = new Box(Orientation.Vertical, 5);
            vboxMethod.BorderWidth = 5;

            _rbLaplacian = new RadioButton("Laplacian (Standard)");
            _rbTaubin = new RadioButton(_rbLaplacian, "Taubin (Volume Preserving)");

            vboxMethod.PackStart(_rbLaplacian, false, false, 0);
            vboxMethod.PackStart(_rbTaubin, false, false, 0);
            frameMethod.Add(vboxMethod);
            vbox.PackStart(frameMethod, false, false, 0);

            // Parameters
            var table = new Table(3, 2, false);
            table.RowSpacing = 5;
            table.ColumnSpacing = 10;

            table.Attach(new Label("Iterations:"), 0, 1, 0, 1, AttachOptions.Shrink, AttachOptions.Shrink, 0, 0);
            _spinIterations = new SpinButton(1, 100, 1);
            _spinIterations.Value = 2;
            table.Attach(_spinIterations, 1, 2, 0, 1, AttachOptions.Expand | AttachOptions.Fill, AttachOptions.Shrink, 0, 0);

            table.Attach(new Label("Strength (Lambda):"), 0, 1, 1, 2, AttachOptions.Shrink, AttachOptions.Shrink, 0, 0);
            _spinLambda = new SpinButton(0.01, 1.0, 0.01);
            _spinLambda.Value = 0.5;
            table.Attach(_spinLambda, 1, 2, 1, 2, AttachOptions.Expand | AttachOptions.Fill, AttachOptions.Shrink, 0, 0);

            table.Attach(new Label("Shrinkage (Mu):"), 0, 1, 2, 3, AttachOptions.Shrink, AttachOptions.Shrink, 0, 0);
            _spinMu = new SpinButton(-1.0, -0.01, 0.01);
            _spinMu.Value = -0.53;
            table.Attach(_spinMu, 1, 2, 2, 3, AttachOptions.Expand | AttachOptions.Fill, AttachOptions.Shrink, 0, 0);

            vbox.PackStart(table, false, false, 0);

            // Toggle sensitivity
            _rbLaplacian.Toggled += (s, e) => UpdateSensitivity();
            UpdateSensitivity();

            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Smooth", ResponseType.Ok);

            this.ShowAll();
        }

        private void UpdateSensitivity()
        {
            _spinMu.Sensitive = _rbTaubin.Active;
        }
    }
}
