using System;
using Gtk;

namespace Deep3DStudio.UI
{
    public class NumericInputDialog : Dialog
    {
        private SpinButton _spinValue;

        public float Value => (float)_spinValue.Value;

        public NumericInputDialog(Window parent, string title, string label, float defaultValue, float min, float max, float step, int digits = 4)
            : base(title, parent, DialogFlags.Modal)
        {
            this.SetDefaultSize(300, 100);
            this.Resizable = false;

            var vbox = this.ContentArea;
            vbox.Spacing = 10;
            vbox.BorderWidth = 10;

            var hbox = new HBox();
            hbox.PackStart(new Label(label), false, false, 5);

            _spinValue = new SpinButton(min, max, step);
            _spinValue.Digits = (uint)digits;
            _spinValue.Value = defaultValue;
            hbox.PackStart(_spinValue, true, true, 5);

            vbox.PackStart(hbox, false, false, 0);

            this.AddButton("Cancel", ResponseType.Cancel);
            this.AddButton("OK", ResponseType.Ok);

            this.ShowAll();
        }
    }
}
