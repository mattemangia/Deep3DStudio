using System;
using Gtk;
using System.IO;
using System.Text;

namespace Deep3DStudio.UI
{
    public class LogWindow : Window
    {
        private TextView _textView;
        private TextBuffer _buffer;
        private ScrolledWindow _scrolledWindow;
        private static LogWindow _instance;

        public static LogWindow Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new LogWindow();
                }
                return _instance;
            }
        }

        private LogWindow() : base("Application Log")
        {
            SetDefaultSize(800, 400);
            SetPosition(WindowPosition.Center);

            // Handle close to hide instead of destroy
            this.DeleteEvent += (o, args) => {
                this.Hide();
                args.RetVal = true;
            };

            var vbox = new Box(Orientation.Vertical, 0);
            this.Add(vbox);

            _scrolledWindow = new ScrolledWindow();
            _scrolledWindow.ShadowType = ShadowType.In;
            vbox.PackStart(_scrolledWindow, true, true, 0);

            _textView = new TextView();
            _textView.Editable = false;
            _textView.WrapMode = WrapMode.Word;
            _textView.Monospace = true;
            _buffer = _textView.Buffer;
            _scrolledWindow.Add(_textView);

            var btnBox = new ButtonBox(Orientation.Horizontal);
            btnBox.Layout = ButtonBoxStyle.End;
            vbox.PackStart(btnBox, false, false, 5);

            var clearBtn = new Button("Clear");
            clearBtn.Clicked += (s, e) => _buffer.Text = "";
            btnBox.PackStart(clearBtn, false, false, 0);

            var closeBtn = new Button("Close");
            closeBtn.Clicked += (s, e) => this.Hide();
            btnBox.PackStart(closeBtn, false, false, 0);

            this.ShowAll();
            this.Hide(); // Start hidden
        }

        public void AppendLog(string text)
        {
            Application.Invoke((s, e) =>
            {
                if (_buffer != null)
                {
                    TextIter end = _buffer.EndIter;
                    _buffer.Insert(ref end, text);
                    _textView.ScrollToIter(_buffer.EndIter, 0, false, 0, 0);
                }
            });
        }
    }

    public class LogWriter : TextWriter
    {
        private TextWriter _originalOut;
        private LogWindow _window;

        public LogWriter(TextWriter original, LogWindow window)
        {
            _originalOut = original;
            _window = window;
        }

        public override Encoding Encoding => _originalOut.Encoding;

        public override void Write(char value)
        {
            _originalOut.Write(value);
            _window.AppendLog(value.ToString());
        }

        public override void Write(string value)
        {
            _originalOut.Write(value);
            _window.AppendLog(value);
        }

        public override void WriteLine(string value)
        {
            _originalOut.WriteLine(value);
            _window.AppendLog(value + Environment.NewLine);
        }
    }
}
