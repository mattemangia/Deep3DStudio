using System;
using Gtk;

namespace Deep3DStudio.UI
{
    public class ScaleCalibrationDialog : Dialog
    {
        private Entry _currentX, _currentY, _currentZ;
        private Entry _targetValue;
        private ComboBoxText _axisCombo;
        private float _currentDimX, _currentDimY, _currentDimZ;

        public float RealScaleFactor { get; private set; } = 1.0f;

        public ScaleCalibrationDialog(Window parent, float sizeX, float sizeY, float sizeZ)
            : base("Set Real Size", parent, DialogFlags.Modal)
        {
            _currentDimX = sizeX;
            _currentDimY = sizeY;
            _currentDimZ = sizeZ;

            this.SetDefaultSize(400, 300);

            var vbox = this.ContentArea;
            vbox.Margin = 10;
            vbox.Spacing = 10;

            // Info Label
            var infoLabel = new Label("Enter the known real-world size for one of the dimensions.\nThe entire model will be scaled uniformly to match.");
            infoLabel.Wrap = true;
            vbox.PackStart(infoLabel, false, false, 5);

            // Current Dimensions Grid
            var grid = new Grid();
            grid.RowSpacing = 5;
            grid.ColumnSpacing = 10;
            grid.MarginTop = 10;

            grid.Attach(new Label("Dimension"), 0, 0, 1, 1);
            grid.Attach(new Label("Current Size (units)"), 1, 0, 1, 1);

            grid.Attach(new Label("Width (X):"), 0, 1, 1, 1);
            _currentX = new Entry(sizeX.ToString("F4")) { IsEditable = false };
            grid.Attach(_currentX, 1, 1, 1, 1);

            grid.Attach(new Label("Height (Y):"), 0, 2, 1, 1);
            _currentY = new Entry(sizeY.ToString("F4")) { IsEditable = false };
            grid.Attach(_currentY, 1, 2, 1, 1);

            grid.Attach(new Label("Depth (Z):"), 0, 3, 1, 1);
            _currentZ = new Entry(sizeZ.ToString("F4")) { IsEditable = false };
            grid.Attach(_currentZ, 1, 3, 1, 1);

            vbox.PackStart(grid, false, false, 5);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Calibration Input
            var inputBox = new Box(Orientation.Horizontal, 5);
            inputBox.PackStart(new Label("Set known size for: "), false, false, 0);

            _axisCombo = new ComboBoxText();
            _axisCombo.AppendText("X (Width)");
            _axisCombo.AppendText("Y (Height)");
            _axisCombo.AppendText("Z (Depth)");

            // Auto-select largest dimension
            if (sizeY >= sizeX && sizeY >= sizeZ) _axisCombo.Active = 1;
            else if (sizeZ >= sizeX && sizeZ >= sizeY) _axisCombo.Active = 2;
            else _axisCombo.Active = 0;

            inputBox.PackStart(_axisCombo, false, false, 0);
            vbox.PackStart(inputBox, false, false, 0);

            var targetBox = new Box(Orientation.Horizontal, 5);
            targetBox.PackStart(new Label("Target Value: "), false, false, 0);
            _targetValue = new Entry();
            _targetValue.PlaceholderText = "e.g. 150.0";
            targetBox.PackStart(_targetValue, true, true, 0);
            targetBox.PackStart(new Label(" (e.g., mm, cm, m)"), false, false, 0);
            vbox.PackStart(targetBox, false, false, 0);

            // Buttons
            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("Apply Scale", ResponseType.Ok);

            this.ShowAll();
        }

        protected override void OnResponse(ResponseType response_id)
        {
            if (response_id == ResponseType.Ok)
            {
                if (float.TryParse(_targetValue.Text, out float target) && target > 0)
                {
                    float currentRef = 1.0f;
                    switch (_axisCombo.Active)
                    {
                        case 0: currentRef = _currentDimX; break;
                        case 1: currentRef = _currentDimY; break;
                        case 2: currentRef = _currentDimZ; break;
                    }

                    if (currentRef > 0.0001f)
                    {
                        RealScaleFactor = target / currentRef;
                    }
                }
            }
        }
    }
}
