using System;
using Gtk;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using System.Runtime.InteropServices;

namespace Deep3DStudio.Viewport
{
    public class ThreeDView : GLArea
    {
        private bool _loaded;
        private float _zoom = -5.0f;
        private float _rotationX = 0f;
        private float _rotationY = 0f;
        private float _panX = 0f;
        private float _panY = 0f;
        private Point _lastMousePos;
        private bool _isDragging;
        private bool _isPanning;

        // Tool State
        private bool _showCropBox = false;
        private float _cropSize = 2.0f;
        private int _selectedHandle = -1; // -1 none, 0-7 corners
        private Vector3[] _cropCorners;

        public ThreeDView()
        {
            this.HasFocus = true;
            this.AddEvents((int)Gdk.EventMask.ButtonPressMask |
                           (int)Gdk.EventMask.ButtonReleaseMask |
                           (int)Gdk.EventMask.PointerMotionMask |
                           (int)Gdk.EventMask.ScrollMask);

            this.Realized += OnRealized;
            this.Render += OnRender;
            this.Unrealized += OnUnrealized;

            this.ButtonPressEvent += OnButtonPress;
            this.ButtonReleaseEvent += OnButtonRelease;
            this.MotionNotifyEvent += OnMotionNotify;
            this.ScrollEvent += OnScroll;

            UpdateCropCorners();
        }

        public void ToggleCropBox(bool show)
        {
            _showCropBox = show;
            this.QueueDraw();
        }

        private void UpdateCropCorners()
        {
            float s = _cropSize;
            _cropCorners = new Vector3[8];
            int idx = 0;
            float[] v = { -s, s };
            foreach(var x in v)
                foreach(var y in v)
                    foreach(var z in v)
                        _cropCorners[idx++] = new Vector3(x, y, z);
        }

        private void OnRealized(object sender, EventArgs e)
        {
            this.MakeCurrent();
            try {
                GL.LoadBindings(new GdkBindingsContext());
                _loaded = true;
            } catch (Exception ex) {
                Console.WriteLine("Failed to load bindings: " + ex.Message);
                _loaded = false;
                return;
            }

            if (_loaded)
            {
                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.ClearColor(0.15f, 0.15f, 0.15f, 1.0f); // Dark background
            }
        }

        private void OnUnrealized(object sender, EventArgs e)
        {
            _loaded = false;
        }

        private void OnRender(object sender, RenderArgs args)
        {
            if (!_loaded) return;

            this.MakeCurrent();
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;
            if (h == 0) h = 1;
            GL.Viewport(0, 0, w, h);

            Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45f), (float)w / h, 0.1f, 100f);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref projection);

            Matrix4 view = Matrix4.CreateTranslation(_panX, _panY, _zoom) *
                           Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX)) *
                           Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref view);

            DrawGrid();
            DrawAxes();

            if (_showCropBox)
            {
                DrawCropBox();
            }
        }

        private void DrawGrid()
        {
            GL.Begin(PrimitiveType.Lines);
            GL.Color4(0.5f, 0.5f, 0.5f, 0.3f);

            int size = 10;
            float step = 1.0f;

            for (float i = -size; i <= size; i += step)
            {
                GL.Vertex3(i, 0, -size);
                GL.Vertex3(i, 0, size);
                GL.Vertex3(-size, 0, i);
                GL.Vertex3(size, 0, i);
            }
            GL.End();
        }

        private void DrawAxes()
        {
            GL.LineWidth(2.0f);
            GL.Begin(PrimitiveType.Lines);

            // X Axis - Red
            GL.Color3(1.0f, 0.0f, 0.0f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(1, 0, 0);

            // Y Axis - Green
            GL.Color3(0.0f, 1.0f, 0.0f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(0, 1, 0);

            // Z Axis - Blue
            GL.Color3(0.0f, 0.0f, 1.0f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(0, 0, 1);

            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawCropBox()
        {
            float s = _cropSize;

            // Draw Box Lines
            GL.Color4(1.0f, 1.0f, 0.0f, 0.8f);
            GL.LineWidth(1.0f);

            float[] v = { -s, s };
            foreach(var x in v)
                foreach(var y in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(x, y, -s);
                    GL.Vertex3(x, y, s);
                    GL.End();
                }
            foreach(var x in v)
                foreach(var z in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(x, -s, z);
                    GL.Vertex3(x, s, z);
                    GL.End();
                }
            foreach(var y in v)
                foreach(var z in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(-s, y, z);
                    GL.Vertex3(s, y, z);
                    GL.End();
                }

            // Draw Handles
            GL.PointSize(10.0f);
            GL.Begin(PrimitiveType.Points);
            for(int i=0; i<8; i++) {
                if (i == _selectedHandle) GL.Color4(1.0f, 0.0f, 0.0f, 1.0f); // Selected: Red
                else GL.Color4(1.0f, 0.5f, 0.0f, 1.0f); // Normal: Orange
                GL.Vertex3(_cropCorners[i]);
            }
            GL.End();
        }

        private Vector2 Project(Vector3 pos, Matrix4 view, Matrix4 projection, int width, int height)
        {
            Vector4 vec = new Vector4(pos, 1.0f);
            vec = vec * view;
            vec = vec * projection;

            if (vec.W == 0) return new Vector2(-1, -1);

            vec /= vec.W;

            float x = (vec.X + 1.0f) * 0.5f * width;
            float y = (1.0f - vec.Y) * 0.5f * height; // Flip Y for window coords

            return new Vector2(x, y);
        }

        private void CheckHandleSelection(int mouseX, int mouseY)
        {
            if (!_showCropBox) {
                _selectedHandle = -1;
                return;
            }

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;
            if (h == 0) h = 1;

            Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45f), (float)w / h, 0.1f, 100f);
            Matrix4 view = Matrix4.CreateTranslation(_panX, _panY, _zoom) *
                           Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX)) *
                           Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));

            float minDist = 15.0f; // Pixel distance threshold
            int bestIdx = -1;

            for(int i=0; i<8; i++) {
                Vector2 screenPos = Project(_cropCorners[i], view, projection, w, h);
                float d = (screenPos - new Vector2(mouseX, mouseY)).Length;
                if (d < minDist) {
                    minDist = d;
                    bestIdx = i;
                }
            }

            if (_selectedHandle != bestIdx) {
                _selectedHandle = bestIdx;
                this.QueueDraw();
            }
        }

        private void OnButtonPress(object o, ButtonPressEventArgs args)
        {
            if (args.Event.Button == 1) // Left click
            {
                if (_selectedHandle != -1)
                {
                     // Handle dragging logic could go here
                     // For now just confirming selection visually is implemented
                     _isDragging = true; // allow dragging to resize?
                     // Let's make dragging resize the box if handle selected
                }
                else
                {
                    _isDragging = true;
                }
                _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
            }
            else if (args.Event.Button == 2 || (args.Event.Button == 1 && (args.Event.State & Gdk.ModifierType.ShiftMask) != 0))
            {
                _isPanning = true;
                _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
            }
        }

        private void OnButtonRelease(object o, ButtonReleaseEventArgs args)
        {
            if (args.Event.Button == 1)
            {
                _isDragging = false;
                _isPanning = false;
            }
            else if (args.Event.Button == 2)
            {
                _isPanning = false;
            }
        }

        private void OnMotionNotify(object o, MotionNotifyEventArgs args)
        {
            int x = (int)args.Event.X;
            int y = (int)args.Event.Y;

            if (!_isDragging)
            {
                CheckHandleSelection(x, y);
            }

            if (_isDragging && !_isPanning)
            {
                int deltaX = x - _lastMousePos.X;
                int deltaY = y - _lastMousePos.Y;

                if (_selectedHandle != -1)
                {
                    // Simple resize logic: dragging right/up increases size
                    _cropSize += deltaX * 0.01f;
                    if (_cropSize < 0.1f) _cropSize = 0.1f;
                    UpdateCropCorners();
                    this.QueueDraw();
                }
                else
                {
                    _rotationY += deltaX * 0.5f;
                    _rotationX += deltaY * 0.5f;
                }
                _lastMousePos = new Point(x, y);
                this.QueueDraw();
            }
            else if (_isPanning)
            {
                int deltaX = x - _lastMousePos.X;
                int deltaY = y - _lastMousePos.Y;

                float panSpeed = 0.01f * Math.Abs(_zoom / 5.0f);

                _panX += deltaX * panSpeed;
                _panY -= deltaY * panSpeed;

                _lastMousePos = new Point(x, y);
                this.QueueDraw();
            }
        }

        private void OnScroll(object o, ScrollEventArgs args)
        {
            if (args.Event.Direction == Gdk.ScrollDirection.Up)
            {
                _zoom += 0.5f;
            }
            else if (args.Event.Direction == Gdk.ScrollDirection.Down)
            {
                _zoom -= 0.5f;
            }
            this.QueueDraw();
        }

        struct Point {
             public int X, Y;
             public Point(int x, int y) { X = x; Y = y; }
        }

        private class GdkBindingsContext : OpenTK.IBindingsContext
        {
            [DllImport("libepoxy.so.0", EntryPoint = "epoxy_glXGetProcAddress", CallingConvention = CallingConvention.Cdecl)]
            private static extern IntPtr GetProcAddressLinux(string procName);

            [DllImport("opengl32.dll", EntryPoint = "wglGetProcAddress", CallingConvention = CallingConvention.StdCall)]
            private static extern IntPtr wglGetProcAddress(string procName);

            [DllImport("kernel32.dll", EntryPoint = "GetProcAddress", CharSet = CharSet.Ansi)]
            private static extern IntPtr GetProcAddressWinKernel(IntPtr hModule, string procName);

            [DllImport("kernel32.dll", EntryPoint = "GetModuleHandle", CharSet = CharSet.Ansi)]
            private static extern IntPtr GetModuleHandle(string lpModuleName);

            // Mac Support
            private const string LibDL = "libdl.dylib";
            [DllImport(LibDL)]
            private static extern IntPtr dlopen(string fileName, int flags);
            [DllImport(LibDL)]
            private static extern IntPtr dlsym(IntPtr handle, string symbol);

            private static IntPtr _macGlHandle = IntPtr.Zero;

            public IntPtr GetProcAddress(string procName)
            {
                try {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
                        // On Windows, wglGetProcAddress returns null for standard OpenGL 1.1 functions
                        // We must first try wglGetProcAddress, if that fails, use GetProcAddress from opengl32.dll module

                        IntPtr ptr = wglGetProcAddress(procName);
                        if (ptr == IntPtr.Zero)
                        {
                            IntPtr glModule = GetModuleHandle("opengl32.dll");
                            ptr = GetProcAddressWinKernel(glModule, procName);
                        }
                        return ptr;
                    }
                    else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                    {
                        if (_macGlHandle == IntPtr.Zero)
                        {
                            _macGlHandle = dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", 1);
                        }
                        if (_macGlHandle != IntPtr.Zero)
                        {
                            var ptr = dlsym(_macGlHandle, procName);
                            if (ptr == IntPtr.Zero) ptr = dlsym(_macGlHandle, "_" + procName);
                            return ptr;
                        }
                        return IntPtr.Zero;
                    }
                    else
                    {
                         return GetProcAddressLinux(procName);
                    }
                }
                catch
                {
                    return IntPtr.Zero;
                }
            }
        }
    }
}
