using System;
using System.Diagnostics;
using Gtk;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using Deep3DStudio.Configuration;
using Deep3DStudio.Scene;
using Deep3DStudio.Model;

namespace Deep3DStudio.Viewport
{
    /// <summary>
    /// Gizmo mode for object manipulation
    /// </summary>
    public enum GizmoMode
    {
        None,
        Translate,
        Rotate,
        Scale,
        Select,
        Pen // Triangle editing mode
    }

    /// <summary>
    /// Enhanced 3D viewport with scene graph support, camera visualization, and transform gizmos
    /// </summary>
    public class ThreeDView : GLArea
    {
        private bool _loaded;
        private float _zoom = -5.0f;
        private float _rotationX = 0f;
        private float _rotationY = 0f;
        private Vector3 _cameraTarget = Vector3.Zero;
        private Point _lastMousePos;
        private bool _isDragging;
        private bool _isPanning;

        // Scene Graph
        private SceneGraph? _sceneGraph;
        private List<MeshData> _meshes = new List<MeshData>(); // Legacy support

        // Tool State
        private bool _showCropBox = false;
        private float _cropSize = 2.0f;
        private int _selectedHandle = -1;
        private Vector3[] _cropCorners = new Vector3[8];

        // Gizmo State
        private GizmoMode _gizmoMode = GizmoMode.Select; // Default to Select
        private int _activeGizmoAxis = -1; // -1=none, 0=X, 1=Y, 2=Z
        private bool _isDraggingGizmo = false;
        private Vector3 _gizmoStartPos;
        private Vector3 _gizmoDragStart;
        private float _gizmoSize = 1.0f;

        // Viewport Info
        private Stopwatch _frameTimer = new Stopwatch();
        private int _frameCount = 0;
        private float _fps = 0;
        private DateTime _lastFpsUpdate = DateTime.Now;

        // Modern GL State (Fallback if Legacy fails)
        private Shader? _shader;
        private int _gridVao, _gridVbo;
        private int _axesVao, _axesVbo;
        private bool _useModernGL = false;

        // Point cloud modern GL buffers (key = object Id)
        private Dictionary<int, (int vao, int vbo, int count)> _pointCloudBuffers = new Dictionary<int, (int, int, int)>();

        // Display Options
        public bool ShowGrid { get; set; } = true;
        public bool ShowAxes { get; set; } = true;
        public bool ShowGizmo { get; set; } = true;
        public bool ShowCameras { get; set; } = true;
        public bool ShowInfoText { get; set; } = true;
        public float CameraFrustumScale { get; set; } = 0.3f;

        // Selection
        public event EventHandler<SceneObject?>? ObjectPicked;
        public event EventHandler? SelectionChanged;

        // Matrices for picking and rendering
        private Matrix4 _viewMatrix;
        private Matrix4 _projectionMatrix;
        private Matrix4 _finalViewMatrix; // coordTransform * _viewMatrix

        // Mesh Editing Tool
        private MeshEditingTool _meshEditingTool = new MeshEditingTool();
        public MeshEditingTool MeshEditingTool => _meshEditingTool;

        // Triangle editing events
        public event EventHandler? TriangleSelectionChanged;

        // Color Palette for Selection
        private static readonly Vector3[] ColorPalette = new Vector3[]
        {
            new Vector3(1.0f, 0.0f, 0.0f),
            new Vector3(0.0f, 1.0f, 0.0f),
            new Vector3(0.0f, 0.0f, 1.0f),
            new Vector3(1.0f, 1.0f, 0.0f),
            new Vector3(1.0f, 0.0f, 1.0f),
            new Vector3(0.0f, 1.0f, 1.0f),
            new Vector3(1.0f, 0.5f, 0.0f),
            new Vector3(0.5f, 0.0f, 1.0f)
        };

        public ThreeDView()
        {
            this.HasDepthBuffer = true;
            this.HasStencilBuffer = false;
            // Requesting version 3.3 (Compatibility Profile) to satisfy GDK requirements
            // while maintaining support for fixed-function pipeline commands.
            this.SetRequiredVersion(3, 3);

            this.HasFocus = true;
            this.CanFocus = true;
            this.AddEvents((int)Gdk.EventMask.ButtonPressMask |
                           (int)Gdk.EventMask.ButtonReleaseMask |
                           (int)Gdk.EventMask.PointerMotionMask |
                           (int)Gdk.EventMask.ScrollMask |
                           (int)Gdk.EventMask.KeyPressMask);

            this.Realized += OnRealized;
            this.Render += OnRender;
            this.Unrealized += OnUnrealized;

            this.ButtonPressEvent += OnButtonPress;
            this.ButtonReleaseEvent += OnButtonRelease;
            this.MotionNotifyEvent += OnMotionNotify;
            this.ScrollEvent += OnScroll;
            this.KeyPressEvent += OnKeyPress;

            UpdateCropCorners();
            _frameTimer.Start();
        }

        #region Public Methods

        public void SetSceneGraph(SceneGraph sceneGraph)
        {
            _sceneGraph = sceneGraph;
            _sceneGraph.SelectionChanged += (s, e) => this.QueueDraw();
            _sceneGraph.SceneChanged += (s, e) => this.QueueDraw();
            AutoCenter();
            this.QueueDraw();
        }

        public void SetGizmoMode(GizmoMode mode)
        {
            _gizmoMode = mode;
            this.QueueDraw();
        }

        public GizmoMode GetGizmoMode() => _gizmoMode;

        public void ToggleCropBox(bool show)
        {
            _showCropBox = show;
            this.QueueDraw();
        }

        /// <summary>
        /// Legacy method for setting meshes directly
        /// </summary>
        public void SetMeshes(List<MeshData> meshes)
        {
            _meshes = meshes;
            AutoCenter();
            this.QueueDraw();
        }

        /// <summary>
        /// Focuses the view on selected objects or entire scene
        /// </summary>
        public void FocusOnSelection()
        {
            Vector3 min, max;

            if (_sceneGraph != null && _sceneGraph.SelectedObjects.Count > 0)
            {
                min = new Vector3(float.MaxValue);
                max = new Vector3(float.MinValue);

                foreach (var obj in _sceneGraph.SelectedObjects)
                {
                    var (objMin, objMax) = obj.GetWorldBounds();
                    min = Vector3.ComponentMin(min, objMin);
                    max = Vector3.ComponentMax(max, objMax);
                }
            }
            else if (_sceneGraph != null)
            {
                (min, max) = _sceneGraph.GetSceneBounds();
            }
            else
            {
                return;
            }

            // Validate bounds
            if (float.IsInfinity(min.X) || float.IsInfinity(max.X) ||
                float.IsNaN(min.X) || float.IsNaN(max.X))
            {
                Console.WriteLine("FocusOnSelection: Invalid bounds, using defaults");
                _cameraTarget = Vector3.Zero;
                _zoom = -5.0f;
                this.QueueDraw();
                return;
            }

            var center = (min + max) * 0.5f;
            var size = (max - min).Length;

            // Ensure minimum zoom distance
            if (size < 0.1f) size = 1.0f;

            _cameraTarget = center;
            _zoom = -size * 1.5f;

            Console.WriteLine($"FocusOnSelection: center({center.X:F2},{center.Y:F2},{center.Z:F2}), zoom={_zoom:F2}");

            this.QueueDraw();
        }

        public void ApplyCrop()
        {
            if (_meshes == null) return;

            Vector3 min = new Vector3(-_cropSize, -_cropSize, -_cropSize);
            Vector3 max = new Vector3(_cropSize, _cropSize, _cropSize);

            foreach (var mesh in _meshes)
            {
                GeometryUtils.CropMesh(mesh, min, max);
            }
            this.QueueDraw();
        }

        #endregion

        #region Private Methods

        private void AutoCenter()
        {
            Vector3 center = Vector3.Zero;
            int count = 0;

            if (_sceneGraph != null)
            {
                foreach (var obj in _sceneGraph.GetVisibleObjects())
                {
                    if (obj is MeshObject mesh && mesh.MeshData != null)
                    {
                        center += mesh.GetCentroid();
                        count++;
                    }
                }
            }

            if (count == 0 && _meshes.Count > 0)
            {
                foreach (var m in _meshes)
                {
                    foreach (var p in m.Vertices)
                    {
                        center += p;
                        count++;
                    }
                }
            }

            if (count > 0)
            {
                center /= count;
                _cameraTarget = center;
                _zoom = -5.0f;
            }
        }

        private void UpdateCropCorners()
        {
            float s = _cropSize;
            _cropCorners = new Vector3[8];
            int idx = 0;
            float[] v = { -s, s };
            foreach (var x in v)
                foreach (var y in v)
                    foreach (var z in v)
                        _cropCorners[idx++] = new Vector3(x, y, z);
        }

        private void UpdateFPS()
        {
            _frameCount++;
            var now = DateTime.Now;
            var elapsed = (now - _lastFpsUpdate).TotalSeconds;
            if (elapsed >= 1.0)
            {
                _fps = (float)(_frameCount / elapsed);
                _frameCount = 0;
                _lastFpsUpdate = now;
            }
        }

        #endregion

        #region GL Events

        private void OnRealized(object? sender, EventArgs e)
        {
            this.MakeCurrent();
            try
            {
                GL.LoadBindings(new GdkBindingsContext());
                _loaded = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to load bindings: " + ex.Message);
                _loaded = false;
                return;
            }

            if (_loaded)
            {
                // Check if we got a Core Profile (no GL.Begin support)
                string version = GL.GetString(StringName.Version);
                Console.WriteLine($"GL Version: {version}");

                // Heuristic: If version >= 3.2 and we suspect Core profile (or just to be safe), init modern GL
                // Note: On some drivers, Compatibility profile is available even in 4.x
                // We will try to init modern GL resources for grid/axes just in case.
                InitModernGL();

                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.PointSize(5.0f);
                GL.LineWidth(1.0f);

                // Ensure initial frame is drawn immediately
                this.QueueDraw();
            }
        }

        private void InitModernGL()
        {
            try
            {
                string vs = @"
                    #version 330 core
                    layout (location = 0) in vec3 aPos;
                    layout (location = 1) in vec3 aColor;
                    uniform mat4 model;
                    uniform mat4 view;
                    uniform mat4 projection;
                    uniform float pointSize;
                    out vec3 vertexColor;
                    void main() {
                        gl_Position = projection * view * model * vec4(aPos, 1.0);
                        vertexColor = aColor;
                        gl_PointSize = pointSize > 0.0 ? pointSize : 8.0;
                    }";

                string fs = @"
                    #version 330 core
                    in vec3 vertexColor;
                    out vec4 FragColor;
                    void main() {
                        FragColor = vec4(vertexColor, 1.0);
                    }";

                _shader = new Shader(vs, fs);

                // Initialize Grid Buffers
                List<float> gridVerts = new List<float>();
                int size = 10;
                float step = 1.0f;
                var s = IniSettings.Instance;

                for (float i = -size; i <= size; i += step)
                {
                    // Z-lines
                    gridVerts.Add(i); gridVerts.Add(0); gridVerts.Add(-size);
                    gridVerts.Add(s.GridColorR); gridVerts.Add(s.GridColorG); gridVerts.Add(s.GridColorB);

                    gridVerts.Add(i); gridVerts.Add(0); gridVerts.Add(size);
                    gridVerts.Add(s.GridColorR); gridVerts.Add(s.GridColorG); gridVerts.Add(s.GridColorB);

                    // X-lines
                    gridVerts.Add(-size); gridVerts.Add(0); gridVerts.Add(i);
                    gridVerts.Add(s.GridColorR); gridVerts.Add(s.GridColorG); gridVerts.Add(s.GridColorB);

                    gridVerts.Add(size); gridVerts.Add(0); gridVerts.Add(i);
                    gridVerts.Add(s.GridColorR); gridVerts.Add(s.GridColorG); gridVerts.Add(s.GridColorB);
                }

                _gridVao = GL.GenVertexArray();
                _gridVbo = GL.GenBuffer();

                GL.BindVertexArray(_gridVao);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _gridVbo);
                GL.BufferData(BufferTarget.ArrayBuffer, gridVerts.Count * sizeof(float), gridVerts.ToArray(), BufferUsageHint.StaticDraw);

                // Position
                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
                GL.EnableVertexAttribArray(0);

                // Color
                GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
                GL.EnableVertexAttribArray(1);

                // Axes Buffers
                float[] axesVerts = {
                    // X (Red)
                    0,0,0, 1,0,0,
                    1.5f,0,0, 1,0,0,
                    // Y (Green)
                    0,0,0, 0,1,0,
                    0,1.5f,0, 0,1,0,
                    // Z (Blue)
                    0,0,0, 0,0,1,
                    0,0,1.5f, 0,0,1
                };

                _axesVao = GL.GenVertexArray();
                _axesVbo = GL.GenBuffer();

                GL.BindVertexArray(_axesVao);
                GL.BindBuffer(BufferTarget.ArrayBuffer, _axesVbo);
                GL.BufferData(BufferTarget.ArrayBuffer, axesVerts.Length * sizeof(float), axesVerts, BufferUsageHint.StaticDraw);

                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
                GL.EnableVertexAttribArray(0);

                GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
                GL.EnableVertexAttribArray(1);

                _useModernGL = true;
                Console.WriteLine("Modern GL initialized successfully.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Modern GL Init failed: {ex.Message}. Falling back to Legacy.");
                _useModernGL = false;
            }
        }

        private void OnUnrealized(object? sender, EventArgs e)
        {
            _loaded = false;
        }

        private void OnRender(object? sender, RenderArgs args)
        {
            if (!_loaded) return;

            UpdateFPS();
            this.MakeCurrent();

            // Apply background color from settings (each frame so changes are reflected)
            var settings = IniSettings.Instance;
            GL.ClearColor(settings.ViewportBgR, settings.ViewportBgG, settings.ViewportBgB, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;
            if (h == 0) h = 1;
            GL.Viewport(0, 0, w, h);

            // Setup matrices
            _projectionMatrix = Matrix4.CreatePerspectiveFieldOfView(
                MathHelper.DegreesToRadians(45f), (float)w / h, 0.1f, 1000f);
            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref _projectionMatrix);

            // Apply Coordinate System Transformation
            Matrix4 coordTransform = Matrix4.Identity;

            if (settings.CoordSystem == CoordinateSystem.RightHanded_Z_Up)
            {
                coordTransform = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(-90));
            }

            // Improved camera logic:
            // 1. Translate world so target is at origin
            // 2. Rotate world (Camera Orbit) - Yaw (Y) then Pitch (X)
            // 3. Translate world so target is at distance (Zoom)
            var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
            var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
            var rotation = ry * rx;

            _viewMatrix = Matrix4.CreateTranslation(-_cameraTarget) *
                          rotation *
                          Matrix4.CreateTranslation(0, 0, _zoom);

            _finalViewMatrix = coordTransform * _viewMatrix;
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref _finalViewMatrix);

            // Draw scene elements
            if (_useModernGL && _shader != null)
            {
                _shader.Use();
                _shader.SetMatrix4("projection", _projectionMatrix);
                _shader.SetMatrix4("view", _finalViewMatrix);
                _shader.SetMatrix4("model", Matrix4.Identity);

                if (ShowGrid)
                {
                    GL.BindVertexArray(_gridVao);
                    GL.DrawArrays(PrimitiveType.Lines, 0, 84); // Approx count
                }
                if (ShowAxes)
                {
                    GL.LineWidth(2.5f);
                    GL.BindVertexArray(_axesVao);
                    GL.DrawArrays(PrimitiveType.Lines, 0, 6);
                    GL.LineWidth(1.0f);
                }

                // Note: For now, meshes and other objects still use legacy GL.
                // If the context is strictly Core, they won't render.
                // But Grid/Axes verify the context is working.
                GL.BindVertexArray(0);
                GL.UseProgram(0);
            }
            else
            {
                if (ShowGrid) DrawGrid();
                if (ShowAxes) DrawAxesEnhanced();
            }

            // Draw scene graph objects
            if (_sceneGraph != null)
            {
                DrawSceneGraph();
            }
            else if (_meshes != null)
            {
                // Legacy mesh rendering
                DrawLegacyMeshes();
            }

            // Draw cameras
            if (ShowCameras && _sceneGraph != null)
            {
                DrawCameras();
            }

            // Draw gizmo for selected objects
            // Hide transform gizmos when in Select or Pen Mode
            if (ShowGizmo && _sceneGraph != null && _sceneGraph.SelectedObjects.Count > 0 &&
                _gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.Pen)
            {
                DrawGizmo();
            }

            // Draw selected triangles highlight (Pen mode)
            if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
            {
                DrawSelectedTriangles();
            }

            // Draw crop box
            if (_showCropBox)
            {
                DrawCropBox();
            }

            // Draw info overlay (2D)
            if (ShowInfoText)
            {
                DrawInfoOverlay(w, h);
            }
        }

        #endregion

        #region Drawing Methods

        private void DrawGrid()
        {
            GL.Begin(PrimitiveType.Lines);

            int size = 10;
            float step = 1.0f;
            var s = IniSettings.Instance;

            // Major grid lines (use grid color from settings)
            GL.Color4(s.GridColorR, s.GridColorG, s.GridColorB, 0.5f);
            for (float i = -size; i <= size; i += step * 5)
            {
                GL.Vertex3(i, 0, -size);
                GL.Vertex3(i, 0, size);
                GL.Vertex3(-size, 0, i);
                GL.Vertex3(size, 0, i);
            }

            // Minor grid lines (dimmer version of grid color)
            GL.Color4(s.GridColorR * 0.7f, s.GridColorG * 0.7f, s.GridColorB * 0.7f, 0.3f);
            for (float i = -size; i <= size; i += step)
            {
                if (Math.Abs(i % (step * 5)) < 0.001f) continue;
                GL.Vertex3(i, 0, -size);
                GL.Vertex3(i, 0, size);
                GL.Vertex3(-size, 0, i);
                GL.Vertex3(size, 0, i);
            }

            GL.End();
        }

        private void DrawAxesEnhanced()
        {
            float axisLength = 1.5f;
            float arrowSize = 0.1f;

            GL.LineWidth(2.5f);
            GL.Begin(PrimitiveType.Lines);

            // X Axis - Red
            GL.Color3(0.9f, 0.2f, 0.2f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(axisLength, 0, 0);

            // Y Axis - Green
            GL.Color3(0.2f, 0.9f, 0.2f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(0, axisLength, 0);

            // Z Axis - Blue
            GL.Color3(0.2f, 0.4f, 0.9f);
            GL.Vertex3(0, 0, 0);
            GL.Vertex3(0, 0, axisLength);

            GL.End();

            // Draw arrow heads
            DrawArrowHead(new Vector3(axisLength, 0, 0), new Vector3(1, 0, 0), arrowSize, new Vector3(0.9f, 0.2f, 0.2f));
            DrawArrowHead(new Vector3(0, axisLength, 0), new Vector3(0, 1, 0), arrowSize, new Vector3(0.2f, 0.9f, 0.2f));
            DrawArrowHead(new Vector3(0, 0, axisLength), new Vector3(0, 0, 1), arrowSize, new Vector3(0.2f, 0.4f, 0.9f));

            GL.LineWidth(1.0f);
        }

        private void DrawArrowHead(Vector3 tip, Vector3 direction, float size, Vector3 color)
        {
            direction = direction.Normalized();

            // Find perpendicular vectors
            Vector3 up = Math.Abs(direction.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX;
            Vector3 right = Vector3.Cross(direction, up).Normalized();
            up = Vector3.Cross(right, direction).Normalized();

            Vector3 base1 = tip - direction * size + right * size * 0.3f;
            Vector3 base2 = tip - direction * size - right * size * 0.3f;
            Vector3 base3 = tip - direction * size + up * size * 0.3f;
            Vector3 base4 = tip - direction * size - up * size * 0.3f;

            GL.Color3(color.X, color.Y, color.Z);
            GL.Begin(PrimitiveType.Triangles);

            GL.Vertex3(tip); GL.Vertex3(base1); GL.Vertex3(base3);
            GL.Vertex3(tip); GL.Vertex3(base3); GL.Vertex3(base2);
            GL.Vertex3(tip); GL.Vertex3(base2); GL.Vertex3(base4);
            GL.Vertex3(tip); GL.Vertex3(base4); GL.Vertex3(base1);

            GL.End();
        }

        private void DrawSceneGraph()
        {
            if (_sceneGraph == null) return;

            var settings = IniSettings.Instance;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                GL.PushMatrix();

                var transform = obj.GetWorldTransform();
                GL.MultMatrix(ref transform);

                if (obj is MeshObject meshObj)
                {
                    bool isSelected = obj.Selected;

                    if (settings.ShowWireframe || meshObj.ShowWireframe)
                    {
                        GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    }
                    else
                    {
                        GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                    }

                    if (settings.ShowPointCloud || meshObj.ShowAsPointCloud)
                    {
                        DrawPointCloud(meshObj.MeshData, isSelected);
                    }
                    else
                    {
                        DrawMesh(meshObj.MeshData, isSelected);
                    }

                    // Draw selection outline
                    if (isSelected)
                    {
                        DrawSelectionOutline(obj);
                    }

                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                }
                else if (obj is PointCloudObject pcObj)
                {
                    DrawPointCloudObject(pcObj);

                    // Always show bounding box for point clouds if enabled, or if selected
                    if (obj.Selected || IniSettings.Instance.ShowPointCloudBounds)
                    {
                        DrawPointCloudBoundingBox(pcObj);
                    }
                }

                GL.PopMatrix();
            }
        }

        private void DrawLegacyMeshes()
        {
            var settings = IniSettings.Instance;

            if (settings.ShowWireframe)
            {
                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
            }
            else
            {
                GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            }

            foreach (var mesh in _meshes)
            {
                if (settings.ShowPointCloud)
                {
                    DrawPointCloud(mesh, false);
                }
                else
                {
                    DrawMesh(mesh, false);
                }
            }

            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
        }

        private void DrawPointCloud(MeshData mesh, bool isSelected)
        {
            if (mesh.Vertices.Count == 0) return;

            var settings = IniSettings.Instance;
            GL.PointSize(isSelected ? 6.0f : 4.0f);

            if (settings.PointCloudColor == PointCloudColorMode.DistanceMap)
            {
                DrawPointCloudDepthMap(mesh);
            }
            else
            {
                bool hasColors = mesh.Colors.Count >= mesh.Vertices.Count;

                GL.Begin(PrimitiveType.Points);
                for (int i = 0; i < mesh.Vertices.Count; i++)
                {
                    Vector3 c = hasColors ? mesh.Colors[i] : new Vector3(1, 1, 1);
                    if (isSelected)
                    {
                        GL.Color3(Math.Min(1f, c.X + 0.3f), Math.Min(1f, c.Y + 0.3f), c.Z);
                    }
                    else
                    {
                        GL.Color3(c.X, c.Y, c.Z);
                    }
                    GL.Vertex3(mesh.Vertices[i]);
                }
                GL.End();
            }
        }

        private void DrawPointCloudDepthMap(MeshData mesh)
        {
            if (mesh.Vertices.Count == 0) return;

            float minDist = float.MaxValue;
            float maxDist = float.MinValue;

            foreach (var v in mesh.Vertices)
            {
                float dist = v.Length;
                if (dist < minDist) minDist = dist;
                if (dist > maxDist) maxDist = dist;
            }

            float range = maxDist - minDist;
            if (range < 0.0001f) range = 1.0f;

            GL.Begin(PrimitiveType.Points);
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                float dist = v.Length;
                float t = (dist - minDist) / range;
                Vector3 color = TurboColormap(t);
                GL.Color3(color.X, color.Y, color.Z);
                GL.Vertex3(v);
            }
            GL.End();
        }

        private void DrawPointCloudObject(PointCloudObject pc)
        {
            if (pc.Points.Count == 0) return;

            GL.PointSize(pc.PointSize);

            // Use modern GL path if available (for Core profile compatibility)
            if (_useModernGL && _shader != null)
            {
                DrawPointCloudModern(pc);
                return;
            }

            // Legacy fixed-function path (requires Compatibility profile)
            GL.Begin(PrimitiveType.Points);

            bool hasColors = pc.Colors.Count >= pc.Points.Count;

            for (int i = 0; i < pc.Points.Count; i++)
            {
                if (hasColors)
                {
                    var c = pc.Colors[i];
                    GL.Color3(c.X, c.Y, c.Z);
                }
                else
                {
                    // Default white color if no colors available
                    GL.Color3(1.0f, 1.0f, 1.0f);
                }
                GL.Vertex3(pc.Points[i]);
            }

            GL.End();
        }

        private void DrawPointCloudModern(PointCloudObject pc)
        {
            // Create or update VAO/VBO for this point cloud
            if (!_pointCloudBuffers.TryGetValue(pc.Id, out var buffers) || buffers.count != pc.Points.Count)
            {
                // Delete old buffers if they exist
                if (buffers.vao != 0)
                {
                    GL.DeleteVertexArray(buffers.vao);
                    GL.DeleteBuffer(buffers.vbo);
                }

                // Create interleaved buffer: position (vec3) + color (vec3)
                var data = new float[pc.Points.Count * 6];
                bool hasColors = pc.Colors.Count >= pc.Points.Count;

                for (int i = 0; i < pc.Points.Count; i++)
                {
                    var p = pc.Points[i];
                    data[i * 6 + 0] = p.X;
                    data[i * 6 + 1] = p.Y;
                    data[i * 6 + 2] = p.Z;

                    if (hasColors)
                    {
                        var c = pc.Colors[i];
                        data[i * 6 + 3] = c.X;
                        data[i * 6 + 4] = c.Y;
                        data[i * 6 + 5] = c.Z;
                    }
                    else
                    {
                        data[i * 6 + 3] = 1.0f;
                        data[i * 6 + 4] = 1.0f;
                        data[i * 6 + 5] = 1.0f;
                    }
                }

                int vao = GL.GenVertexArray();
                int vbo = GL.GenBuffer();

                GL.BindVertexArray(vao);
                GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
                GL.BufferData(BufferTarget.ArrayBuffer, data.Length * sizeof(float), data, BufferUsageHint.StaticDraw);

                // Position attribute (location = 0)
                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
                GL.EnableVertexAttribArray(0);

                // Color attribute (location = 1)
                GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
                GL.EnableVertexAttribArray(1);

                GL.BindVertexArray(0);

                _pointCloudBuffers[pc.Id] = (vao, vbo, pc.Points.Count);
                buffers = (vao, vbo, pc.Points.Count);

                // Log sample data for debugging
                Console.WriteLine($"Created modern GL buffers for point cloud {pc.Id}: {pc.Points.Count} points");
                if (pc.Points.Count > 0)
                {
                    Console.WriteLine($"  Sample point 0: pos=({data[0]:F3},{data[1]:F3},{data[2]:F3}) color=({data[3]:F2},{data[4]:F2},{data[5]:F2})");
                    if (pc.Points.Count > 100)
                    {
                        int midIdx = 100 * 6;
                        Console.WriteLine($"  Sample point 100: pos=({data[midIdx]:F3},{data[midIdx+1]:F3},{data[midIdx+2]:F3}) color=({data[midIdx+3]:F2},{data[midIdx+4]:F2},{data[midIdx+5]:F2})");
                    }
                }
            }

            // Use shader and draw
            _shader!.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", pc.GetWorldTransform());
            _shader.SetFloat("pointSize", pc.PointSize);

            // Enable point size from shader
            GL.Enable(EnableCap.ProgramPointSize);

            GL.BindVertexArray(buffers.vao);
            GL.DrawArrays(PrimitiveType.Points, 0, buffers.count);
            GL.BindVertexArray(0);
            GL.UseProgram(0);

            GL.Disable(EnableCap.ProgramPointSize);
        }

        /// <summary>
        /// Draws a bounding box around the point cloud with a distinct color
        /// </summary>
        private void DrawPointCloudBoundingBox(PointCloudObject pc)
        {
            if (pc.Points.Count == 0) return;

            var min = pc.BoundsMin;
            var max = pc.BoundsMax;

            // Validate bounds
            if (float.IsInfinity(min.X) || float.IsInfinity(max.X) ||
                float.IsNaN(min.X) || float.IsNaN(max.X))
            {
                return;
            }

            GL.LineWidth(2.0f);

            // Get color for bounding box
            Vector3 color;
            if (pc.Selected)
            {
                color = ColorPalette[pc.Id % ColorPalette.Length];
            }
            else
            {
                color = new Vector3(0.0f, 0.8f, 0.9f); // Cyan
            }

            // Use modern GL if available
            if (_useModernGL && _shader != null)
            {
                DrawBoundingBoxModern(min, max, color, pc.GetWorldTransform());
                GL.LineWidth(1.0f);
                return;
            }

            // Legacy path
            GL.Color4(color.X, color.Y, color.Z, 0.6f);

            GL.Begin(PrimitiveType.Lines);

            // Bottom face
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, max.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, min.Z);

            // Top face
            GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
            GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, max.Z);
            GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
            GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, min.Z);

            // Vertical edges
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X, max.Y, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X, max.Y, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);

            GL.End();
            GL.LineWidth(1.0f);
        }

        private int _boundingBoxVao = 0;
        private int _boundingBoxVbo = 0;

        private void DrawBoundingBoxModern(Vector3 min, Vector3 max, Vector3 color, Matrix4 modelTransform)
        {
            // Create line data for bounding box (24 vertices for 12 edges)
            var vertices = new float[]
            {
                // Bottom face
                min.X, min.Y, min.Z, color.X, color.Y, color.Z,
                max.X, min.Y, min.Z, color.X, color.Y, color.Z,
                max.X, min.Y, min.Z, color.X, color.Y, color.Z,
                max.X, min.Y, max.Z, color.X, color.Y, color.Z,
                max.X, min.Y, max.Z, color.X, color.Y, color.Z,
                min.X, min.Y, max.Z, color.X, color.Y, color.Z,
                min.X, min.Y, max.Z, color.X, color.Y, color.Z,
                min.X, min.Y, min.Z, color.X, color.Y, color.Z,
                // Top face
                min.X, max.Y, min.Z, color.X, color.Y, color.Z,
                max.X, max.Y, min.Z, color.X, color.Y, color.Z,
                max.X, max.Y, min.Z, color.X, color.Y, color.Z,
                max.X, max.Y, max.Z, color.X, color.Y, color.Z,
                max.X, max.Y, max.Z, color.X, color.Y, color.Z,
                min.X, max.Y, max.Z, color.X, color.Y, color.Z,
                min.X, max.Y, max.Z, color.X, color.Y, color.Z,
                min.X, max.Y, min.Z, color.X, color.Y, color.Z,
                // Vertical edges
                min.X, min.Y, min.Z, color.X, color.Y, color.Z,
                min.X, max.Y, min.Z, color.X, color.Y, color.Z,
                max.X, min.Y, min.Z, color.X, color.Y, color.Z,
                max.X, max.Y, min.Z, color.X, color.Y, color.Z,
                max.X, min.Y, max.Z, color.X, color.Y, color.Z,
                max.X, max.Y, max.Z, color.X, color.Y, color.Z,
                min.X, min.Y, max.Z, color.X, color.Y, color.Z,
                min.X, max.Y, max.Z, color.X, color.Y, color.Z,
            };

            // Create or recreate VAO/VBO
            if (_boundingBoxVao == 0)
            {
                _boundingBoxVao = GL.GenVertexArray();
                _boundingBoxVbo = GL.GenBuffer();
            }

            GL.BindVertexArray(_boundingBoxVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _boundingBoxVbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.DynamicDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            _shader!.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", modelTransform);

            GL.DrawArrays(PrimitiveType.Lines, 0, 24);

            GL.BindVertexArray(0);
            GL.UseProgram(0);
        }

        private static Vector3 TurboColormap(float t)
        {
            var (r, g, b) = ImageUtils.TurboColormap(t);
            return new Vector3(r, g, b);
        }

        private void DrawMesh(MeshData mesh, bool isSelected)
        {
            if (mesh.Vertices.Count == 0 || mesh.Indices.Count == 0) return;

            bool useTexture = IniSettings.Instance.ShowTexture && mesh.Texture != null;

            if (useTexture)
            {
                if (mesh.TextureId == -1)
                {
                    UploadTexture(mesh);
                }

                if (mesh.TextureId != -1)
                {
                    GL.Enable(EnableCap.Texture2D);
                    GL.BindTexture(TextureTarget.Texture2D, mesh.TextureId);
                    GL.Color3(1.0f, 1.0f, 1.0f); // White to show texture colors
                }
                else
                {
                    useTexture = false;
                }
            }

            bool hasColors = mesh.Colors.Count >= mesh.Vertices.Count;

            GL.Begin(PrimitiveType.Triangles);
            for (int i = 0; i < mesh.Indices.Count; i++)
            {
                int idx = mesh.Indices[i];
                if (idx < mesh.Vertices.Count)
                {
                    if (useTexture && idx < mesh.UVs.Count)
                    {
                        var uv = mesh.UVs[idx];
                        GL.TexCoord2(uv.X, uv.Y);
                    }
                    else
                    {
                        Vector3 c = hasColors && idx < mesh.Colors.Count ? mesh.Colors[idx] : new Vector3(0.7f, 0.7f, 0.7f);
                        if (isSelected)
                        {
                            GL.Color3(Math.Min(1f, c.X + 0.2f), Math.Min(1f, c.Y + 0.2f), c.Z);
                        }
                        else
                        {
                            GL.Color3(c.X, c.Y, c.Z);
                        }
                    }
                    GL.Vertex3(mesh.Vertices[idx]);
                }
            }
            GL.End();

            if (useTexture)
            {
                GL.BindTexture(TextureTarget.Texture2D, 0);
                GL.Disable(EnableCap.Texture2D);
            }
        }

        private void UploadTexture(MeshData mesh)
        {
            if (mesh.Texture == null) return;

            int id = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, id);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.Repeat);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.Repeat);

            // SkiaSharp uses RGBA or BGRA usually.
            // We assume Rgba8888 for now as per ImageDecoder/TextureBaker usage
            var info = mesh.Texture.Info;
            PixelFormat pixelFormat = PixelFormat.Bgra; // Skia usually defaults to BGRA on desktop
            if (info.ColorType == SkiaSharp.SKColorType.Rgba8888) pixelFormat = PixelFormat.Rgba;

            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, info.Width, info.Height, 0,
                pixelFormat, PixelType.UnsignedByte, mesh.Texture.GetPixels());

            mesh.TextureId = id;
        }

        private void DrawSelectionOutline(SceneObject obj)
        {
            var (min, max) = (obj.BoundsMin, obj.BoundsMax);

            GL.LineWidth(2.0f);

            // Assign color based on Object ID
            var color = ColorPalette[obj.Id % ColorPalette.Length];
            GL.Color4(color.X, color.Y, color.Z, 0.8f);

            var mode = IniSettings.Instance.BoundingBoxStyle;

            GL.Begin(PrimitiveType.Lines);

            if (mode == BoundingBoxMode.Full)
            {
                // Bottom face
                GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, min.Z);
                GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, max.Z);
                GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, max.Z);
                GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, min.Z);

                // Top face
                GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
                GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, max.Z);
                GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
                GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, min.Z);

                // Vertical edges
                GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X, max.Y, min.Z);
                GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
                GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X, max.Y, max.Z);
                GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
            }
            else // Corners
            {
                float cornerSize = Math.Min(Math.Min(max.X - min.X, max.Y - min.Y), max.Z - min.Z) * 0.2f;
                DrawCornerBox(min, max, cornerSize);
            }

            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawCornerBox(Vector3 min, Vector3 max, float s)
        {
            // Bottom-Left-Front
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X + s, min.Y, min.Z);
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X, min.Y + s, min.Z);
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X, min.Y, min.Z + s);

            // Bottom-Right-Front
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X - s, min.Y, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y + s, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, min.Z + s);

            // Bottom-Left-Back
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X + s, min.Y, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y + s, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, max.Z - s);

            // Bottom-Right-Back
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X - s, min.Y, max.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X, min.Y + s, max.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X, min.Y, max.Z - s);

            // Top-Left-Front
            GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(min.X + s, max.Y, min.Z);
            GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(min.X, max.Y - s, min.Z);
            GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(min.X, max.Y, min.Z + s);

            // Top-Right-Front
            GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X - s, max.Y, min.Z);
            GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y - s, min.Z);
            GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z + s);

            // Top-Left-Back
            GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X + s, max.Y, max.Z);
            GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y - s, max.Z);
            GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z - s);

            // Top-Right-Back
            GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(max.X - s, max.Y, max.Z);
            GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(max.X, max.Y - s, max.Z);
            GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(max.X, max.Y, max.Z - s);
        }

        private void DrawCameras()
        {
            if (_sceneGraph == null) return;

            foreach (var cam in _sceneGraph.GetObjectsOfType<CameraObject>())
            {
                if (!cam.Visible || !cam.ShowFrustum) continue;

                DrawCameraFrustum(cam);
            }
        }

        private void DrawCameraFrustum(CameraObject cam)
        {
            Vector3 pos = cam.Position;
            if (cam.Pose != null)
            {
                pos = cam.Pose.CameraToWorld.ExtractTranslation();
            }

            var corners = cam.GetFrustumCorners(CameraFrustumScale);
            var color = cam.Selected ? new Vector3(1f, 1f, 0f) : cam.FrustumColor;

            GL.LineWidth(cam.Selected ? 2.5f : 1.5f);
            GL.Color3(color.X, color.Y, color.Z);

            GL.Begin(PrimitiveType.Lines);

            // Lines from camera origin to frustum corners
            for (int i = 0; i < 4; i++)
            {
                GL.Vertex3(pos);
                GL.Vertex3(corners[i]);
            }

            // Frustum rectangle
            GL.Vertex3(corners[0]); GL.Vertex3(corners[1]);
            GL.Vertex3(corners[1]); GL.Vertex3(corners[2]);
            GL.Vertex3(corners[2]); GL.Vertex3(corners[3]);
            GL.Vertex3(corners[3]); GL.Vertex3(corners[0]);

            // Cross on image plane
            GL.Vertex3(corners[0]); GL.Vertex3(corners[2]);
            GL.Vertex3(corners[1]); GL.Vertex3(corners[3]);

            GL.End();

            // Draw camera body
            GL.Color3(color.X * 0.8f, color.Y * 0.8f, color.Z * 0.8f);
            float camSize = CameraFrustumScale * 0.15f;

            GL.Begin(PrimitiveType.Triangles);

            // Simple pyramid shape for camera body
            Vector3 up = cam.GetUpDirection() * camSize;
            Vector3 right = Vector3.Cross(cam.GetViewDirection(), up).Normalized() * camSize;

            Vector3 c1 = pos + up + right;
            Vector3 c2 = pos + up - right;
            Vector3 c3 = pos - up - right;
            Vector3 c4 = pos - up + right;
            Vector3 tip = pos - cam.GetViewDirection() * camSize * 1.5f;

            // Front face
            GL.Vertex3(c1); GL.Vertex3(c2); GL.Vertex3(c3);
            GL.Vertex3(c1); GL.Vertex3(c3); GL.Vertex3(c4);

            // Side faces
            GL.Vertex3(tip); GL.Vertex3(c1); GL.Vertex3(c2);
            GL.Vertex3(tip); GL.Vertex3(c2); GL.Vertex3(c3);
            GL.Vertex3(tip); GL.Vertex3(c3); GL.Vertex3(c4);
            GL.Vertex3(tip); GL.Vertex3(c4); GL.Vertex3(c1);

            GL.End();

            GL.LineWidth(1.0f);
        }

        /// <summary>
        /// Draw highlight overlay for selected triangles in Pen mode
        /// </summary>
        private void DrawSelectedTriangles()
        {
            var vertices = _meshEditingTool.GetSelectedTriangleVertices();
            if (vertices.Count == 0)
                return;

            // Disable depth test for overlay effect, or enable for proper occlusion
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            // Draw filled triangles with semi-transparent highlight
            GL.Begin(PrimitiveType.Triangles);
            GL.Color4(1.0f, 0.5f, 0.0f, 0.4f); // Orange highlight

            for (int i = 0; i < vertices.Count; i++)
            {
                GL.Vertex3(vertices[i]);
            }

            GL.End();

            // Draw wireframe edges for clarity
            GL.LineWidth(2.0f);
            GL.Begin(PrimitiveType.Lines);
            GL.Color4(1.0f, 0.3f, 0.0f, 1.0f); // Darker orange

            for (int i = 0; i < vertices.Count; i += 3)
            {
                // Edge 0-1
                GL.Vertex3(vertices[i]);
                GL.Vertex3(vertices[i + 1]);
                // Edge 1-2
                GL.Vertex3(vertices[i + 1]);
                GL.Vertex3(vertices[i + 2]);
                // Edge 2-0
                GL.Vertex3(vertices[i + 2]);
                GL.Vertex3(vertices[i]);
            }

            GL.End();
            GL.LineWidth(1.0f);

            GL.Disable(EnableCap.Blend);
        }

        private void DrawGizmo()
        {
            if (_sceneGraph == null || _sceneGraph.SelectedObjects.Count == 0) return;

            // Calculate gizmo center (centroid of selected objects)
            Vector3 center = Vector3.Zero;
            foreach (var obj in _sceneGraph.SelectedObjects)
            {
                center += obj.Position;
            }
            center /= _sceneGraph.SelectedObjects.Count;

            // Calculate gizmo size based on distance to camera
            float distToCamera = Math.Abs(_zoom);
            _gizmoSize = distToCamera * 0.15f;

            GL.Disable(EnableCap.DepthTest);

            switch (_gizmoMode)
            {
                case GizmoMode.Translate:
                    DrawTranslateGizmo(center);
                    break;
                case GizmoMode.Rotate:
                    DrawRotateGizmo(center);
                    break;
                case GizmoMode.Scale:
                    DrawScaleGizmo(center);
                    break;
            }

            GL.Enable(EnableCap.DepthTest);
        }

        private void DrawTranslateGizmo(Vector3 center)
        {
            float len = _gizmoSize;
            float arrowSize = len * 0.15f;

            GL.LineWidth(3.0f);
            GL.Begin(PrimitiveType.Lines);

            // X axis (red)
            GL.Color3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, _activeGizmoAxis == 0 ? 1.0f : 0.2f, 0.2f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(len, 0, 0));

            // Y axis (green)
            GL.Color3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, _activeGizmoAxis == 1 ? 1.0f : 0.2f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(0, len, 0));

            // Z axis (blue)
            GL.Color3(0.2f, _activeGizmoAxis == 2 ? 1.0f : 0.4f, _activeGizmoAxis == 2 ? 1.0f : 0.9f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(0, 0, len));

            GL.End();

            // Arrow heads
            DrawArrowHead(center + new Vector3(len, 0, 0), Vector3.UnitX, arrowSize,
                new Vector3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, 0.2f, 0.2f));
            DrawArrowHead(center + new Vector3(0, len, 0), Vector3.UnitY, arrowSize,
                new Vector3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, 0.2f));
            DrawArrowHead(center + new Vector3(0, 0, len), Vector3.UnitZ, arrowSize,
                new Vector3(0.2f, 0.4f, _activeGizmoAxis == 2 ? 1.0f : 0.9f));

            GL.LineWidth(1.0f);
        }

        private void DrawRotateGizmo(Vector3 center)
        {
            float radius = _gizmoSize;
            int segments = 48;

            GL.LineWidth(3.0f);

            // X rotation circle (YZ plane) - red
            GL.Color3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, _activeGizmoAxis == 0 ? 1.0f : 0.2f, 0.2f);
            GL.Begin(PrimitiveType.LineLoop);
            for (int i = 0; i < segments; i++)
            {
                float angle = (float)i / segments * MathF.PI * 2;
                GL.Vertex3(center.X, center.Y + MathF.Cos(angle) * radius, center.Z + MathF.Sin(angle) * radius);
            }
            GL.End();

            // Y rotation circle (XZ plane) - green
            GL.Color3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, _activeGizmoAxis == 1 ? 1.0f : 0.2f);
            GL.Begin(PrimitiveType.LineLoop);
            for (int i = 0; i < segments; i++)
            {
                float angle = (float)i / segments * MathF.PI * 2;
                GL.Vertex3(center.X + MathF.Cos(angle) * radius, center.Y, center.Z + MathF.Sin(angle) * radius);
            }
            GL.End();

            // Z rotation circle (XY plane) - blue
            GL.Color3(0.2f, _activeGizmoAxis == 2 ? 1.0f : 0.4f, _activeGizmoAxis == 2 ? 1.0f : 0.9f);
            GL.Begin(PrimitiveType.LineLoop);
            for (int i = 0; i < segments; i++)
            {
                float angle = (float)i / segments * MathF.PI * 2;
                GL.Vertex3(center.X + MathF.Cos(angle) * radius, center.Y + MathF.Sin(angle) * radius, center.Z);
            }
            GL.End();

            GL.LineWidth(1.0f);
        }

        private void DrawScaleGizmo(Vector3 center)
        {
            float len = _gizmoSize;
            float boxSize = len * 0.1f;

            GL.LineWidth(3.0f);
            GL.Begin(PrimitiveType.Lines);

            // X axis
            GL.Color3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, _activeGizmoAxis == 0 ? 1.0f : 0.2f, 0.2f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(len, 0, 0));

            // Y axis
            GL.Color3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, _activeGizmoAxis == 1 ? 1.0f : 0.2f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(0, len, 0));

            // Z axis
            GL.Color3(0.2f, _activeGizmoAxis == 2 ? 1.0f : 0.4f, _activeGizmoAxis == 2 ? 1.0f : 0.9f);
            GL.Vertex3(center);
            GL.Vertex3(center + new Vector3(0, 0, len));

            GL.End();

            // Draw boxes at ends
            DrawBox(center + new Vector3(len, 0, 0), boxSize, new Vector3(0.8f, 0.2f, 0.2f));
            DrawBox(center + new Vector3(0, len, 0), boxSize, new Vector3(0.2f, 0.8f, 0.2f));
            DrawBox(center + new Vector3(0, 0, len), boxSize, new Vector3(0.2f, 0.4f, 0.9f));

            GL.LineWidth(1.0f);
        }

        private void DrawBox(Vector3 center, float size, Vector3 color)
        {
            float h = size * 0.5f;
            GL.Color3(color.X, color.Y, color.Z);

            GL.Begin(PrimitiveType.Quads);

            // Front
            GL.Vertex3(center.X - h, center.Y - h, center.Z + h);
            GL.Vertex3(center.X + h, center.Y - h, center.Z + h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z + h);
            GL.Vertex3(center.X - h, center.Y + h, center.Z + h);

            // Back
            GL.Vertex3(center.X - h, center.Y - h, center.Z - h);
            GL.Vertex3(center.X - h, center.Y + h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y - h, center.Z - h);

            // Top
            GL.Vertex3(center.X - h, center.Y + h, center.Z - h);
            GL.Vertex3(center.X - h, center.Y + h, center.Z + h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z + h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z - h);

            // Bottom
            GL.Vertex3(center.X - h, center.Y - h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y - h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y - h, center.Z + h);
            GL.Vertex3(center.X - h, center.Y - h, center.Z + h);

            // Right
            GL.Vertex3(center.X + h, center.Y - h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z - h);
            GL.Vertex3(center.X + h, center.Y + h, center.Z + h);
            GL.Vertex3(center.X + h, center.Y - h, center.Z + h);

            // Left
            GL.Vertex3(center.X - h, center.Y - h, center.Z - h);
            GL.Vertex3(center.X - h, center.Y - h, center.Z + h);
            GL.Vertex3(center.X - h, center.Y + h, center.Z + h);
            GL.Vertex3(center.X - h, center.Y + h, center.Z - h);

            GL.End();
        }

        private void DrawCropBox()
        {
            float s = _cropSize;

            GL.Color4(1.0f, 1.0f, 0.0f, 0.8f);
            GL.LineWidth(1.0f);

            float[] v = { -s, s };
            foreach (var x in v)
                foreach (var y in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(x, y, -s);
                    GL.Vertex3(x, y, s);
                    GL.End();
                }
            foreach (var x in v)
                foreach (var z in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(x, -s, z);
                    GL.Vertex3(x, s, z);
                    GL.End();
                }
            foreach (var y in v)
                foreach (var z in v)
                {
                    GL.Begin(PrimitiveType.Lines);
                    GL.Vertex3(-s, y, z);
                    GL.Vertex3(s, y, z);
                    GL.End();
                }

            // Draw Handles
            GL.PointSize(10.0f);
            GL.Begin(PrimitiveType.Points);
            for (int i = 0; i < 8; i++)
            {
                if (i == _selectedHandle) GL.Color4(1.0f, 0.0f, 0.0f, 1.0f);
                else GL.Color4(1.0f, 0.5f, 0.0f, 1.0f);
                GL.Vertex3(_cropCorners[i]);
            }
            GL.End();
        }

        private void DrawInfoOverlay(int width, int height)
        {
            // Setup 2D projection
            GL.MatrixMode(MatrixMode.Projection);
            GL.PushMatrix();
            GL.LoadIdentity();
            GL.Ortho(0, width, height, 0, -1, 1);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.PushMatrix();
            GL.LoadIdentity();

            GL.Disable(EnableCap.DepthTest);

            // Draw info background
            GL.Color4(0.0f, 0.0f, 0.0f, 0.5f);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(5, 5);
            GL.Vertex2(200, 5);
            GL.Vertex2(200, 80);
            GL.Vertex2(5, 80);
            GL.End();

            // Draw text using simple bitmap approach (placeholder - actual text rendering needs font support)
            // In GTK, we'd use Pango for text rendering, but for GL we need a different approach
            // For now, we'll draw colored rectangles to indicate status

            // FPS indicator bar
            float fpsRatio = Math.Min(1.0f, _fps / 60.0f);
            GL.Color3(1.0f - fpsRatio, fpsRatio, 0.0f);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(10, 15);
            GL.Vertex2(10 + fpsRatio * 100, 15);
            GL.Vertex2(10 + fpsRatio * 100, 25);
            GL.Vertex2(10, 25);
            GL.End();

            // Objects indicator
            int objCount = _sceneGraph?.GetVisibleObjects().Count() ?? _meshes.Count;
            float objRatio = Math.Min(1.0f, objCount / 20.0f);
            GL.Color3(0.3f, 0.6f, 1.0f);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(10, 35);
            GL.Vertex2(10 + objRatio * 100, 35);
            GL.Vertex2(10 + objRatio * 100, 45);
            GL.Vertex2(10, 45);
            GL.End();

            // Selection indicator
            int selCount = _sceneGraph?.SelectedObjects.Count ?? 0;
            if (selCount > 0)
            {
                GL.Color3(1.0f, 0.6f, 0.0f);
                GL.Begin(PrimitiveType.Quads);
                GL.Vertex2(10, 55);
                GL.Vertex2(10 + Math.Min(selCount * 20, 100), 55);
                GL.Vertex2(10 + Math.Min(selCount * 20, 100), 65);
                GL.Vertex2(10, 65);
                GL.End();
            }

            GL.Enable(EnableCap.DepthTest);

            // Restore matrices
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Projection);
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Modelview);
        }

        #endregion

        #region Picking & Interaction

        private Vector2 Project(Vector3 pos, Matrix4 view, Matrix4 projection, int width, int height)
        {
            Vector4 vec = new Vector4(pos, 1.0f);
            vec = vec * view;
            vec = vec * projection;

            if (vec.W == 0) return new Vector2(-1, -1);

            vec /= vec.W;

            float x = (vec.X + 1.0f) * 0.5f * width;
            float y = (1.0f - vec.Y) * 0.5f * height;

            return new Vector2(x, y);
        }

        private void CheckHandleSelection(int mouseX, int mouseY)
        {
            if (!_showCropBox)
            {
                _selectedHandle = -1;
                return;
            }

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;
            if (h == 0) h = 1;

            float minDist = 15.0f;
            int bestIdx = -1;

            for (int i = 0; i < 8; i++)
            {
                Vector2 screenPos = Project(_cropCorners[i], _viewMatrix, _projectionMatrix, w, h);
                float d = (screenPos - new Vector2(mouseX, mouseY)).Length;
                if (d < minDist)
                {
                    minDist = d;
                    bestIdx = i;
                }
            }

            if (_selectedHandle != bestIdx)
            {
                _selectedHandle = bestIdx;
                this.QueueDraw();
            }
        }

        private int CheckGizmoSelection(int mouseX, int mouseY)
        {
            if (_sceneGraph == null || _sceneGraph.SelectedObjects.Count == 0 || _gizmoMode == GizmoMode.Select)
                return -1;

            Vector3 center = Vector3.Zero;
            foreach (var obj in _sceneGraph.SelectedObjects)
                center += obj.Position;
            center /= _sceneGraph.SelectedObjects.Count;

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;

            Vector2 screenCenter = Project(center, _viewMatrix, _projectionMatrix, w, h);
            Vector2 screenX = Project(center + new Vector3(_gizmoSize, 0, 0), _viewMatrix, _projectionMatrix, w, h);
            Vector2 screenY = Project(center + new Vector3(0, _gizmoSize, 0), _viewMatrix, _projectionMatrix, w, h);
            Vector2 screenZ = Project(center + new Vector3(0, 0, _gizmoSize), _viewMatrix, _projectionMatrix, w, h);

            Vector2 mouse = new Vector2(mouseX, mouseY);
            float threshold = 15.0f;

            // Check distance to each axis line
            float distX = DistanceToLineSegment(mouse, screenCenter, screenX);
            float distY = DistanceToLineSegment(mouse, screenCenter, screenY);
            float distZ = DistanceToLineSegment(mouse, screenCenter, screenZ);

            if (distX < threshold && distX < distY && distX < distZ) return 0;
            if (distY < threshold && distY < distX && distY < distZ) return 1;
            if (distZ < threshold && distZ < distX && distZ < distY) return 2;

            return -1;
        }

        private float DistanceToLineSegment(Vector2 point, Vector2 lineStart, Vector2 lineEnd)
        {
            Vector2 line = lineEnd - lineStart;
            float len = line.Length;
            if (len < 0.001f) return (point - lineStart).Length;

            float t = Math.Max(0, Math.Min(1, Vector2.Dot(point - lineStart, line) / (len * len)));
            Vector2 projection = lineStart + t * line;
            return (point - projection).Length;
        }

        private SceneObject? PickObject(int mouseX, int mouseY)
        {
            if (_sceneGraph == null) return null;

            int w = this.Allocation.Width;
            int h = this.Allocation.Height;
            Vector2 mouse = new Vector2(mouseX, mouseY);

            SceneObject? closest = null;
            float minDist = float.MaxValue;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                var (boundsMin, boundsMax) = obj.GetWorldBounds();
                var center = (boundsMin + boundsMax) * 0.5f;
                var screenPos = Project(center, _viewMatrix, _projectionMatrix, w, h);
                float dist = (screenPos - mouse).Length;

                // Simple distance check to center
                // A better approach would be ray-AABB intersection
                if (dist < 50 && dist < minDist)
                {
                    minDist = dist;
                    closest = obj;
                }
            }

            return closest;
        }

        #endregion

        #region Input Events

        private void OnButtonPress(object o, ButtonPressEventArgs args)
        {
            this.GrabFocus();

            if (args.Event.Button == 1) // Left click
            {
                // Pen mode: triangle picking
                if (_gizmoMode == GizmoMode.Pen)
                {
                    HandlePenModeClick((int)args.Event.X, (int)args.Event.Y, args.Event.State);
                    _isDragging = true;
                    _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
                    return;
                }

                // Check gizmo first
                int gizmoAxis = CheckGizmoSelection((int)args.Event.X, (int)args.Event.Y);
                if (gizmoAxis >= 0 && _sceneGraph?.SelectedObjects.Count > 0)
                {
                    _activeGizmoAxis = gizmoAxis;
                    _isDraggingGizmo = true;
                    _gizmoDragStart = new Vector3((float)args.Event.X, (float)args.Event.Y, 0);

                    foreach (var obj in _sceneGraph.SelectedObjects)
                        _gizmoStartPos = obj.Position;

                    this.QueueDraw();
                    return;
                }

                // Check crop box handles
                if (_selectedHandle != -1)
                {
                    _isDragging = true;
                }
                else
                {
                    // Object picking - Always allow picking in Select mode or if no gizmo was hit
                    var picked = PickObject((int)args.Event.X, (int)args.Event.Y);

                    // Allow Shift for multiple selection
                    bool multipleSelection = (args.Event.State & Gdk.ModifierType.ShiftMask) != 0 ||
                                             (args.Event.State & Gdk.ModifierType.ControlMask) != 0;

                    if (picked != null && _sceneGraph != null)
                    {
                        if (multipleSelection)
                        {
                            // Toggle selection
                            if (picked.Selected)
                                _sceneGraph.Deselect(picked);
                            else
                                _sceneGraph.Select(picked, true);
                        }
                        else
                        {
                            _sceneGraph.Select(picked, false);
                        }

                        ObjectPicked?.Invoke(this, picked);
                    }
                    else if (_sceneGraph != null && !multipleSelection)
                    {
                        // Deselect all if clicked on empty space without shift
                        _sceneGraph.ClearSelection();
                    }

                    _isDragging = true;
                }
                _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
            }
            else if (args.Event.Button == 2 || (args.Event.Button == 1 && (args.Event.State & Gdk.ModifierType.ShiftMask) != 0))
            {
                // Pan with middle mouse or Shift+Left.
                // Prioritize panning over selection if Shift is held and drag initiates.
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
                _isDraggingGizmo = false;
                _activeGizmoAxis = -1;
                this.QueueDraw();
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

            if (!_isDragging && !_isPanning && !_isDraggingGizmo)
            {
                CheckHandleSelection(x, y);
                int gizmoAxis = CheckGizmoSelection(x, y);
                if (_activeGizmoAxis != gizmoAxis)
                {
                    _activeGizmoAxis = gizmoAxis;
                    this.QueueDraw();
                }
            }

            if (_isDraggingGizmo && _sceneGraph != null)
            {
                int deltaX = x - (int)_gizmoDragStart.X;
                int deltaY = y - (int)_gizmoDragStart.Y;

                float sensitivity = 0.01f * Math.Abs(_zoom / 5.0f);

                Vector3 delta = Vector3.Zero;
                switch (_activeGizmoAxis)
                {
                    case 0: delta.X = deltaX * sensitivity; break;
                    case 1: delta.Y = -deltaY * sensitivity; break;
                    case 2: delta.Z = deltaX * sensitivity; break;
                }

                foreach (var obj in _sceneGraph.SelectedObjects)
                {
                    switch (_gizmoMode)
                    {
                        case GizmoMode.Translate:
                            obj.Position = _gizmoStartPos + delta;
                            break;
                        case GizmoMode.Rotate:
                            obj.Rotation += delta * 50;
                            break;
                        case GizmoMode.Scale:
                            obj.Scale = Vector3.One + delta;
                            break;
                    }
                }

                this.QueueDraw();
            }
            else if (_isDragging && !_isPanning)
            {
                // Rotate view if not dragging a handle and not selecting.
                // In Select mode, left drag rotates the view (box selection is not implemented).

                int deltaX = x - _lastMousePos.X;
                int deltaY = y - _lastMousePos.Y;

                if (_selectedHandle != -1)
                {
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

                float panSpeed = 0.002f * Math.Abs(_zoom);

                // Calculate screen-aligned pan vectors
                var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
                var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
                var rotation = ry * rx;

                // Right is Column 0, Up is Column 1 of the View Matrix Rotation
                Vector3 right = new Vector3(rotation.M11, rotation.M21, rotation.M31);
                Vector3 up = new Vector3(rotation.M12, rotation.M22, rotation.M32);

                // Move target opposite to mouse movement to drag the scene
                _cameraTarget -= right * deltaX * panSpeed;
                _cameraTarget += up * deltaY * panSpeed;

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

        private void OnKeyPress(object o, KeyPressEventArgs args)
        {
            switch (args.Event.Key)
            {
                case Gdk.Key.q: // Q for Select
                case Gdk.Key.Q:
                    SetGizmoMode(GizmoMode.Select);
                    break;
                case Gdk.Key.w:
                case Gdk.Key.W:
                    SetGizmoMode(GizmoMode.Translate);
                    break;
                case Gdk.Key.e:
                case Gdk.Key.E:
                    SetGizmoMode(GizmoMode.Rotate);
                    break;
                case Gdk.Key.r:
                case Gdk.Key.R:
                    SetGizmoMode(GizmoMode.Scale);
                    break;
                case Gdk.Key.p:
                case Gdk.Key.P:
                    SetGizmoMode(GizmoMode.Pen);
                    break;
                case Gdk.Key.f:
                case Gdk.Key.F:
                    FocusOnSelection();
                    break;
                case Gdk.Key.Delete:
                    // In Pen mode, delete selected triangles
                    if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
                    {
                        _meshEditingTool.DeleteSelectedTriangles();
                        TriangleSelectionChanged?.Invoke(this, EventArgs.Empty);
                    }
                    else if (_sceneGraph != null)
                    {
                        foreach (var obj in _sceneGraph.SelectedObjects.ToList())
                        {
                            _sceneGraph.RemoveObject(obj);
                        }
                    }
                    break;
                case Gdk.Key.Escape:
                    // In Pen mode, clear triangle selection first
                    if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
                    {
                        _meshEditingTool.ClearSelection();
                        TriangleSelectionChanged?.Invoke(this, EventArgs.Empty);
                    }
                    else
                    {
                        _sceneGraph?.ClearSelection();
                    }
                    break;
            }
            this.QueueDraw();
        }

        /// <summary>
        /// Handle click events in Pen (triangle editing) mode
        /// </summary>
        private void HandlePenModeClick(int mouseX, int mouseY, Gdk.ModifierType modifiers)
        {
            if (_sceneGraph == null)
                return;

            // Get ray from mouse position
            var (rayOrigin, rayDir) = GetRayFromScreenPoint(mouseX, mouseY);

            // Get all mesh objects from scene
            var meshObjects = _sceneGraph.GetVisibleObjects()
                .OfType<MeshObject>()
                .ToList();

            // Perform triangle picking
            var (pickedMesh, triangleIndex, distance) = _meshEditingTool.PickTriangle(rayOrigin, rayDir, meshObjects);

            bool addToSelection = (modifiers & Gdk.ModifierType.ShiftMask) != 0 ||
                                  (modifiers & Gdk.ModifierType.ControlMask) != 0;

            if (pickedMesh != null && triangleIndex >= 0)
            {
                if (addToSelection)
                {
                    _meshEditingTool.ToggleTriangleSelection(pickedMesh, triangleIndex);
                }
                else
                {
                    _meshEditingTool.SelectTriangle(pickedMesh, triangleIndex, false);
                }
            }
            else if (!addToSelection)
            {
                _meshEditingTool.ClearSelection();
            }

            TriangleSelectionChanged?.Invoke(this, EventArgs.Empty);
            this.QueueDraw();
        }

        /// <summary>
        /// Generate a ray from camera through screen point for picking
        /// </summary>
        private (Vector3 origin, Vector3 direction) GetRayFromScreenPoint(int screenX, int screenY)
        {
            int width = this.Allocation.Width;
            int height = this.Allocation.Height;

            if (width <= 0 || height <= 0)
                return (Vector3.Zero, -Vector3.UnitZ);

            // Convert screen coordinates to normalized device coordinates (-1 to 1)
            float ndcX = (2.0f * screenX / width) - 1.0f;
            float ndcY = 1.0f - (2.0f * screenY / height); // Y is flipped

            // Create near and far points in NDC
            Vector4 nearPoint = new Vector4(ndcX, ndcY, -1.0f, 1.0f);
            Vector4 farPoint = new Vector4(ndcX, ndcY, 1.0f, 1.0f);

            // Unproject to world space
            Matrix4 invProjection = _projectionMatrix.Inverted();
            Matrix4 invView = _viewMatrix.Inverted();
            Matrix4 invVP = invProjection * invView;

            Vector4 nearWorld = nearPoint * invVP;
            Vector4 farWorld = farPoint * invVP;

            // Perspective divide
            if (Math.Abs(nearWorld.W) > 1e-6f)
            {
                nearWorld /= nearWorld.W;
            }
            if (Math.Abs(farWorld.W) > 1e-6f)
            {
                farWorld /= farWorld.W;
            }

            Vector3 rayOrigin = new Vector3(nearWorld.X, nearWorld.Y, nearWorld.Z);
            Vector3 rayDir = new Vector3(farWorld.X - nearWorld.X, farWorld.Y - nearWorld.Y, farWorld.Z - nearWorld.Z);

            if (rayDir.LengthSquared > 1e-6f)
            {
                rayDir.Normalize();
            }
            else
            {
                rayDir = -Vector3.UnitZ;
            }

            return (rayOrigin, rayDir);
        }

        #endregion

        struct Point
        {
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

            private const string LibDL = "libdl.dylib";
            [DllImport(LibDL)]
            private static extern IntPtr dlopen(string fileName, int flags);
            [DllImport(LibDL)]
            private static extern IntPtr dlsym(IntPtr handle, string symbol);

            private static IntPtr _macGlHandle = IntPtr.Zero;

            public IntPtr GetProcAddress(string procName)
            {
                try
                {
                    if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                    {
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
