using Gtk;
using Vector3 = OpenTK.Mathematics.Vector3;

namespace Deep3DStudio.UI
{
    public class Vector3InputDialog : Dialog
    {
        private readonly SpinButton _xInput;
        private readonly SpinButton _yInput;
        private readonly SpinButton _zInput;

        public Vector3 Value => new Vector3(
            (float)_xInput.Value,
            (float)_yInput.Value,
            (float)_zInput.Value);

        public Vector3InputDialog(
            Window parent,
            string title,
            string xLabel,
            string yLabel,
            string zLabel,
            Vector3 defaultValue,
            float minValue,
            float maxValue,
            float step,
            int digits = 4)
            : base(title, parent, DialogFlags.Modal)
        {
            SetDefaultSize(360, 170);
            Resizable = false;

            var area = ContentArea;
            area.BorderWidth = 12;
            area.Spacing = 8;

            var grid = new Grid
            {
                RowSpacing = 8,
                ColumnSpacing = 8
            };

            _xInput = CreateSpin(defaultValue.X, minValue, maxValue, step, digits);
            _yInput = CreateSpin(defaultValue.Y, minValue, maxValue, step, digits);
            _zInput = CreateSpin(defaultValue.Z, minValue, maxValue, step, digits);

            grid.Attach(new Label(xLabel) { Halign = Align.Start }, 0, 0, 1, 1);
            grid.Attach(_xInput, 1, 0, 1, 1);
            grid.Attach(new Label(yLabel) { Halign = Align.Start }, 0, 1, 1, 1);
            grid.Attach(_yInput, 1, 1, 1, 1);
            grid.Attach(new Label(zLabel) { Halign = Align.Start }, 0, 2, 1, 1);
            grid.Attach(_zInput, 1, 2, 1, 1);

            area.PackStart(grid, true, true, 0);

            AddButton("Cancel", ResponseType.Cancel);
            AddButton("Apply", ResponseType.Ok);
            DefaultResponse = ResponseType.Ok;

            ShowAll();
        }

        private static SpinButton CreateSpin(float value, float min, float max, float step, int digits)
        {
            var spin = new SpinButton(min, max, step)
            {
                Digits = (uint)digits,
                Value = value
            };
            return spin;
        }
    }
}
