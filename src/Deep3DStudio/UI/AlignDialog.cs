using System;
using Gtk;

namespace Deep3DStudio.UI
{
    public class AlignDialog : Dialog
    {
        private SpinButton _spinIterations;
        private SpinButton _spinThreshold;

        public int Iterations => (int)_spinIterations.Value;
        public float Threshold => (float)_spinThreshold.Value;

        public AlignDialog(Window parent) : base("Align Meshes (ICP)", parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(300, 150);
            this.Resizable = false;

            var vbox = this.ContentArea;
            vbox.Spacing = 10;
            vbox.BorderWidth = 10;

            var table = new Table(2, 2, false);
            table.RowSpacing = 5;
            table.ColumnSpacing = 10;

            table.Attach(new Label("Max Iterations:"), 0, 1, 0, 1, AttachOptions.Shrink, AttachOptions.Shrink, 0, 0);
            _spinIterations = new SpinButton(1, 500, 1);
            _spinIterations.Value = 50;
            table.Attach(_spinIterations, 1, 2, 0, 1, AttachOptions.Expand | AttachOptions.Fill, AttachOptions.Shrink, 0, 0);

            table.Attach(new Label("Convergence Threshold:"), 0, 1, 1, 2, AttachOptions.Shrink, AttachOptions.Shrink, 0, 0);
            _spinThreshold = new SpinButton(0.000001, 0.1, 0.00001);
            _spinThreshold.Digits = 6;
            _spinThreshold.Value = 0.0001;
            table.Attach(_spinThreshold, 1, 2, 1, 2, AttachOptions.Expand | AttachOptions.Fill, AttachOptions.Shrink, 0, 0);

            vbox.PackStart(table, false, false, 0);

            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Align", ResponseType.Ok);

            this.ShowAll();
        }
    }
}
