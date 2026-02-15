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
        Pen,
        Rigging,
        SplittingPlane
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
        private Vector3 _gizmoDragStart;
        private float _gizmoSize = 1.0f;
        private readonly Dictionary<int, GizmoStartState> _gizmoStartStates = new Dictionary<int, GizmoStartState>();

        private readonly struct GizmoStartState
        {
            public readonly Vector3 Position;
            public readonly Vector3 Rotation;
            public readonly Vector3 Scale;

            public GizmoStartState(Vector3 position, Vector3 rotation, Vector3 scale)
            {
                Position = position;
                Rotation = rotation;
                Scale = scale;
            }
        }

        // Viewport Info
        private Stopwatch _frameTimer = new Stopwatch();
        private int _frameCount = 0;
        private float _fps = 0;
        private DateTime _lastFpsUpdate = DateTime.Now;

        // Modern GL State (Fallback if Legacy fails)
        private Shader? _shader;
        private int _gridVao, _gridVbo;
        private int _axesVao, _axesVbo;
        private int _cameraVao, _cameraVbo;
        private bool _useModernGL = false;
        private bool _legacySupported = true;
        private bool _legacyWarningLogged = false;

        // Point cloud modern GL buffers (key = object Id)
        private Dictionary<int, (int vao, int vbo, int count, PointCloudColorMode colorMode)> _pointCloudBuffers
            = new Dictionary<int, (int, int, int, PointCloudColorMode)>();

        // Mesh modern GL buffers (key = object Id)
        private Dictionary<int, (int vao, int vbo, int ebo, int indexCount)> _meshBuffers
            = new Dictionary<int, (int, int, int, int)>();

        private enum ColorLegendMode
        {
            None,
            Depth,
            Confidence,
            ConfidenceDepthFallback
        }

        private ColorLegendMode _colorLegendMode = ColorLegendMode.None;
        private float _colorLegendMin = 0.0f;
        private float _colorLegendMax = 1.0f;
        private readonly TextRenderer _overlayTextRenderer = new TextRenderer
        {
            FontFamily = "Monospace",
            FontSize = 11
        };
        private bool _hasHoveredSourceDistance;
        private float _hoveredSourceDistance;
        private bool _hoveredDistanceIsMeters;
        private string _hoveredSourceCameraName = string.Empty;

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

        // Splitting Plane Tool
        private SplittingPlaneTool? _splittingPlaneTool = null;
        public SplittingPlaneTool? GetSplittingPlaneTool() => _splittingPlaneTool;

        // Event for splitting plane confirmation
        public event EventHandler? SplittingPlaneConfirmed;

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
            // Requesting version 2.1 to favor a Compatibility Profile / Legacy context
            // while still allowing modern GL features if available.
            // On Windows, requesting 3.3 often forces a strict Core Profile.
            this.SetRequiredVersion(2, 1);

            // Disable automatic rendering to save CPU.
            // QueueRender() is called manually when scene or camera changes.
            this.AutoRender = false;

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
            this.Mapped += OnMapped;

            this.ButtonPressEvent += OnButtonPress;
            this.ButtonReleaseEvent += OnButtonRelease;
            this.MotionNotifyEvent += OnMotionNotify;
            this.ScrollEvent += OnScroll;
            this.KeyPressEvent += OnKeyPress;

            UpdateCropCorners();
            _frameTimer.Start();
        }

        private void OnMapped(object? sender, EventArgs e)
        {
            Console.WriteLine($"ThreeDView: Mapped - triggering initial render");
            // Trigger initial render when widget becomes visible
            this.QueueRender();
        }

        private bool EnsureLegacySupport(string feature)
        {
            if (_legacySupported) return true;

            if (!_legacyWarningLogged)
            {
                Console.WriteLine("Legacy OpenGL not available (core profile). Skipping legacy rendering paths to avoid driver crashes.");
                _legacyWarningLogged = true;
            }

            return false;
        }

        #region Public Methods

        public void SetSceneGraph(SceneGraph sceneGraph)
        {
            _sceneGraph = sceneGraph;
            _sceneGraph.SelectionChanged += (s, e) => this.QueueRender();
            _sceneGraph.SceneChanged += (s, e) => this.QueueRender();
            AutoCenter();
            this.QueueRender();
        }

        public void SetGizmoMode(GizmoMode mode)
        {
            if (_gizmoMode == GizmoMode.SplittingPlane && mode != GizmoMode.SplittingPlane)
                _splittingPlaneTool = null;

            if (mode == GizmoMode.SplittingPlane && _splittingPlaneTool == null)
                _splittingPlaneTool = new SplittingPlaneTool();

            _gizmoMode = mode;
            this.QueueRender();
        }

        public GizmoMode GetGizmoMode() => _gizmoMode;

        public bool TryGetHoveredSourceCameraDistance(out float value, out bool isMeters, out string sourceCameraName)
        {
            value = _hoveredSourceDistance;
            isMeters = _hoveredDistanceIsMeters;
            sourceCameraName = _hoveredSourceCameraName;
            return _hasHoveredSourceDistance;
        }

        public void ToggleCropBox(bool show)
        {
            _showCropBox = show;
            this.QueueRender();
        }

        /// <summary>
        /// Legacy method for setting meshes directly
        /// </summary>
        public void SetMeshes(List<MeshData> meshes)
        {
            _meshes = meshes;
            AutoCenter();
            this.QueueRender();
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
                this.QueueRender();
                return;
            }

            var center = (min + max) * 0.5f;
            var size = (max - min).Length;

            // Ensure minimum zoom distance
            if (size < 0.1f) size = 1.0f;

            _cameraTarget = center;
            _zoom = -size * 1.5f;

            Console.WriteLine($"FocusOnSelection: center({center.X:F2},{center.Y:F2},{center.Z:F2}), zoom={_zoom:F2}");

            this.QueueRender();
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
            this.QueueRender();
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

                try
                {
                    // OpenTK 4.8.2 does not expose ContextProfileMask on GetPName,
                    // but the enum value still exists under All, so cast accordingly.
                    int profileMask = GL.GetInteger((GetPName)All.ContextProfileMask);
                    var profile = (ContextProfileMask)profileMask;

                    _legacySupported = (profile & ContextProfileMask.ContextCompatibilityProfileBit) != 0 || profileMask == 0;

                    if (!_legacySupported)
                    {
                        Console.WriteLine("Warning: Core profile detected. Disabling legacy immediate-mode rendering.");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Unable to determine OpenGL profile. Assuming legacy support. Details: {ex.Message}");
                    _legacySupported = true;
                }

                InitModernGL();

                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

                // GL.PointSize and GL.LineWidth are legacy functions - only call if legacy is supported
                if (_legacySupported)
                {
                    GL.PointSize(5.0f);
                    GL.LineWidth(1.0f);
                }

                // Ensure initial frame is drawn immediately
                this.QueueRender();
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
                    uniform vec4 uniformColor;
                    uniform bool useUniformColor;
                    out vec4 FragColor;
                    void main() {
                        if (useUniformColor) {
                            FragColor = uniformColor;
                        } else {
                            FragColor = vec4(vertexColor, 1.0);
                        }
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

                _gridVertexCount = gridVerts.Count / 6; // 6 floats per vertex (pos + color)

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

                // Axes Buffers - Longer and clearer axes
                float[] axesVerts = {
                    // X (Red)
                    0,0,0, 1,0,0,
                    2.0f,0,0, 1,0,0,
                    // Y (Green)
                    0,0,0, 0,1,0,
                    0,2.0f,0, 0,1,0,
                    // Z (Blue)
                    0,0,0, 0,0,1,
                    0,0,2.0f, 0,0,1
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
            if (_cameraVao != 0) GL.DeleteVertexArray(_cameraVao);
            if (_cameraVbo != 0) GL.DeleteBuffer(_cameraVbo);
            if (_gridVao != 0) GL.DeleteVertexArray(_gridVao);
            if (_gridVbo != 0) GL.DeleteBuffer(_gridVbo);
            if (_axesVao != 0) GL.DeleteVertexArray(_axesVao);
            if (_axesVbo != 0) GL.DeleteBuffer(_axesVbo);

            foreach(var kvp in _pointCloudBuffers)
            {
                GL.DeleteVertexArray(kvp.Value.vao);
                GL.DeleteBuffer(kvp.Value.vbo);
            }
            _pointCloudBuffers.Clear();

            foreach (var kvp in _meshBuffers)
            {
                GL.DeleteVertexArray(kvp.Value.vao);
                GL.DeleteBuffer(kvp.Value.vbo);
                GL.DeleteBuffer(kvp.Value.ebo);
            }
            _meshBuffers.Clear();

            _overlayTextRenderer.Dispose();
        }

        // Track grid vertex count for modern GL
        private int _gridVertexCount = 0;
        private bool _renderDebugLogged = false;

        private void OnRender(object? sender, RenderArgs args)
        {
            if (!_loaded)
            {
                if (!_renderDebugLogged)
                {
                    Console.WriteLine("OnRender: Not loaded yet, skipping");
                }
                return;
            }

            if (!_renderDebugLogged)
            {
                Console.WriteLine($"OnRender: First render - useModernGL={_useModernGL}, legacySupported={_legacySupported}, gridVertexCount={_gridVertexCount}");
                _renderDebugLogged = true;
            }

            // If neither modern GL nor legacy is supported, we can't render
            if (!_useModernGL && !_legacySupported)
            {
                // At least clear to show we're running
                GL.ClearColor(0.2f, 0.2f, 0.2f, 1.0f);
                GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
                return;
            }

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

            // Ensure clean state for fixed-function and modern rendering
            GL.Disable(EnableCap.Texture2D);
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.CullFace);
            GL.Disable(EnableCap.ScissorTest);
            GL.DepthMask(true);
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.UseProgram(0);

            // Disable all client-side arrays to prevent interference with legacy/immediate mode
            GL.DisableClientState(ArrayCap.VertexArray);
            GL.DisableClientState(ArrayCap.ColorArray);
            GL.DisableClientState(ArrayCap.TextureCoordArray);
            GL.DisableClientState(ArrayCap.NormalArray);

            // Disable vertex attributes on VAO 0 to avoid interference with legacy rendering
            for (int i = 0; i < 16; i++) GL.DisableVertexAttribArray(i);

            // Only use legacy matrix stack if legacy is supported
            if (_legacySupported)
            {
                GL.MatrixMode(MatrixMode.Projection);
                GL.LoadMatrix(ref _projectionMatrix);
                GL.MatrixMode(MatrixMode.Modelview);
                GL.LoadMatrix(ref _finalViewMatrix);
            }

            // Draw scene elements using modern GL when available
            _colorLegendMode = ColorLegendMode.None;
            _colorLegendMin = 0.0f;
            _colorLegendMax = 1.0f;

            if (_useModernGL && _shader != null)
            {
                _shader.Use();
                _shader.SetMatrix4("projection", _projectionMatrix);
                _shader.SetMatrix4("view", _finalViewMatrix);
                _shader.SetMatrix4("model", Matrix4.Identity);
                
                // Ensure we start without uniform color override
                _shader.SetBool("useUniformColor", false);

                if (ShowGrid && _gridVao != 0)
                {
                    GL.BindVertexArray(_gridVao);
                    GL.DrawArrays(PrimitiveType.Lines, 0, _gridVertexCount);
                }
                if (ShowAxes && _axesVao != 0)
                {
                    GL.BindVertexArray(_axesVao);
                    GL.DrawArrays(PrimitiveType.Lines, 0, 6);
                }

                // Check for GL errors
                var err = GL.GetError();
                if (err != ErrorCode.NoError && !_renderDebugLogged)
                {
                    Console.WriteLine($"GL Error after modern rendering: {err}");
                }

                // Draw point clouds using modern GL
                if (_sceneGraph != null)
                {
                    DrawPointCloudsModernGL();
                    DrawMeshesModernGL();

                    // Draw transform handles in modern mode
                    if (ShowGizmo && _sceneGraph.SelectedObjects.Count > 0 &&
                        _gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.Pen)
                    {
                        DrawGizmoModern();
                    }

                    // Draw cameras in modern mode
                    if (ShowCameras)
                    {
                        DrawCamerasModern();
                    }

                    // Draw selected triangles highlight in modern mode
                    if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
                    {
                        DrawSelectedTrianglesModern();
                    }
                }

                // Draw orientation gizmo in the corner (always visible)
                if (ShowAxes)
                {
                    DrawOrientationGizmoModern(w, h);
                }

                GL.BindVertexArray(0);
                GL.UseProgram(0);
            }
            // Legacy rendering path
            if (_legacySupported)
            {
                // Ensure no VAO is bound for immediate mode rendering
                GL.BindVertexArray(0);
                
                // Clear any potential errors from modern GL setup
                while (GL.GetError() != ErrorCode.NoError) { }

                if (ShowGrid) DrawGrid();
                if (ShowAxes) DrawAxesEnhanced();
            }

            // Draw scene graph objects (only if legacy is supported - these use GL.Begin/End)
            if (_legacySupported)
            {
                // Ensure no VAO is bound
                GL.BindVertexArray(0);
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

                // Draw splitting plane
                if (_gizmoMode == GizmoMode.SplittingPlane && _splittingPlaneTool != null)
                {
                    _splittingPlaneTool.Render();
                }

                // Draw info overlay (2D)
                if (ShowInfoText)
                {
                    DrawInfoOverlay(w, h);
                }
            }
        }

        /// <summary>
        /// Draws point clouds using modern OpenGL (for Core profile compatibility)
        /// </summary>
        private void DrawPointCloudsModernGL()
        {
            if (_sceneGraph == null || _shader == null) return;

            var settings = IniSettings.Instance;
            if (!settings.ShowPointCloud) return;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                if (obj is PointCloudObject pcObj)
                {
                    bool bboxOnly = pcObj.RenderMode == ObjectRenderMode.BoundingBoxOnly;
                    
                    if (!bboxOnly)
                    {
                        // Apply object transform
                        var transform = obj.GetWorldTransform();
                        _shader.SetMatrix4("model", transform);
                        _shader.SetFloat("pointSize", pcObj.PointSize);

                        // Enable point size control from shader
                        GL.Enable(EnableCap.ProgramPointSize);

                        DrawPointCloudModern(pcObj);
                    }

                    // Draw bounding box if enabled or selected
                    if (bboxOnly || pcObj.Selected || IniSettings.Instance.ShowPointCloudBounds)
                    {
                        var color = pcObj.Selected ? ColorPalette[pcObj.Id % ColorPalette.Length] : new Vector3(0.0f, 0.8f, 0.9f);
                        DrawBoundingBoxModern(pcObj.BoundsMin, pcObj.BoundsMax, color, pcObj.GetWorldTransform());
                    }
                }
            }

            // Reset model matrix
            _shader.SetMatrix4("model", Matrix4.Identity);
        }

        /// <summary>
        /// Draws meshes using modern OpenGL (for Core profile compatibility)
        /// </summary>
        private void DrawMeshesModernGL()
        {
            if (_sceneGraph == null || _shader == null) return;

            var settings = IniSettings.Instance;
            if (!settings.ShowMesh) return;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                if (obj is MeshObject meshObj && meshObj.MeshData != null)
                {
                    bool bboxOnly = meshObj.RenderMode == ObjectRenderMode.BoundingBoxOnly;
                    bool isSelected = meshObj.Selected;

                    if (!bboxOnly && ShouldDrawMeshSurface(meshObj, settings))
                    {
                        DrawMeshModern(meshObj);
                    }

                    // Draw selection highlight in modern mode
                    if (isSelected || bboxOnly)
                    {
                        var color = ColorPalette[meshObj.Id % ColorPalette.Length];
                        DrawBoundingBoxModern(meshObj.BoundsMin, meshObj.BoundsMax, color, meshObj.GetWorldTransform());
                    }
                }
            }
        }

        private void DrawMeshModern(MeshObject mesh)
        {
            if (mesh.MeshData == null || mesh.MeshData.Vertices.Count == 0) return;

            // Simple vertex format for now: Pos(3) + Color(3) = 6 floats per vertex
            // We use the same shader as point clouds
            
            bool isSelected = mesh.Selected;

            // Rebuild buffer if count changed
            if (!_meshBuffers.TryGetValue(mesh.Id, out var buffers)
                || buffers.indexCount != mesh.MeshData.Indices.Count)
            {
                if (buffers.vao != 0)
                {
                    GL.DeleteVertexArray(buffers.vao);
                    GL.DeleteBuffer(buffers.vbo);
                    GL.DeleteBuffer(buffers.ebo);
                }

                float[] data = new float[mesh.MeshData.Vertices.Count * 6];
                bool hasColors = mesh.MeshData.Colors.Count >= mesh.MeshData.Vertices.Count;

                for (int i = 0; i < mesh.MeshData.Vertices.Count; i++)
                {
                    var p = mesh.MeshData.Vertices[i];
                    data[i * 6 + 0] = p.X;
                    data[i * 6 + 1] = p.Y;
                    data[i * 6 + 2] = p.Z;

                    if (hasColors)
                    {
                        var c = mesh.MeshData.Colors[i];
                        data[i * 6 + 3] = c.X;
                        data[i * 6 + 4] = c.Y;
                        data[i * 6 + 5] = c.Z;
                    }
                    else
                    {
                        data[i * 6 + 3] = 0.7f;
                        data[i * 6 + 4] = 0.7f;
                        data[i * 6 + 5] = 0.7f;
                    }
                }

                int vao = GL.GenVertexArray();
                int vbo = GL.GenBuffer();
                int ebo = GL.GenBuffer();

                GL.BindVertexArray(vao);

                GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
                GL.BufferData(BufferTarget.ArrayBuffer, data.Length * sizeof(float), data, BufferUsageHint.StaticDraw);

                GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
                GL.BufferData(BufferTarget.ElementArrayBuffer, mesh.MeshData.Indices.Count * sizeof(int), mesh.MeshData.Indices.ToArray(), BufferUsageHint.StaticDraw);

                // Attributes (Location 0 = Pos, Location 1 = Color)
                GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
                GL.EnableVertexAttribArray(0);

                GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
                GL.EnableVertexAttribArray(1);

                GL.BindVertexArray(0);
                _meshBuffers[mesh.Id] = (vao, vbo, ebo, mesh.MeshData.Indices.Count);
                buffers = (vao, vbo, ebo, mesh.MeshData.Indices.Count);
            }

            _shader!.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", mesh.GetWorldTransform());
            _shader.SetFloat("pointSize", 0.0f);

            // Handle selection tint
            if (isSelected)
            {
                _shader.SetVector4("uniformColor", new Vector4(1.0f, 0.9f, 0.7f, 1.0f));
                _shader.SetBool("useUniformColor", true);
            }
            else
            {
                _shader.SetBool("useUniformColor", false);
            }

            // Handle wireframe mode
            bool wireframe = ResolveWireframe(mesh, IniSettings.Instance);
            if (wireframe) GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
            
            GL.BindVertexArray(buffers.vao);
            GL.DrawElements(PrimitiveType.Triangles, buffers.indexCount, DrawElementsType.UnsignedInt, 0);
            GL.BindVertexArray(0);

            if (wireframe) GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            _shader.SetBool("useUniformColor", false);
        }

        #endregion

        #region Drawing Methods

        private void DrawGrid()
        {
            if (!EnsureLegacySupport("grid")) return;

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
            if (!EnsureLegacySupport("axes")) return;

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
            if (!EnsureLegacySupport("axis arrow")) return;

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
                    if (!_legacySupported)
                    {
                        EnsureLegacySupport("mesh rendering");
                        GL.PopMatrix();
                        continue;
                    }

                    bool isSelected = obj.Selected;
                    bool bboxOnly = meshObj.RenderMode == ObjectRenderMode.BoundingBoxOnly;

                    if (ResolveWireframe(meshObj, settings))
                    {
                        GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    }
                    else
                    {
                        GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                    }

                    if (!bboxOnly && ShouldDrawMeshSurface(meshObj, settings))
                    {
                        DrawMesh(meshObj.MeshData, isSelected, ResolveTexture(meshObj, settings));
                    }

                    if (!bboxOnly && ShouldDrawMeshAsPointCloud(meshObj, settings))
                    {
                        DrawPointCloud(meshObj.MeshData, isSelected);
                    }

                    // Draw selection outline
                    if (isSelected || bboxOnly)
                    {
                        DrawSelectionOutline(obj);
                    }

                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
                }
                else if (obj is PointCloudObject pcObj)
                {
                    if (pcObj.RenderMode == ObjectRenderMode.BoundingBoxOnly)
                    {
                        DrawPointCloudBoundingBox(pcObj);
                    }
                    else if (settings.ShowPointCloud)
                    {
                        DrawPointCloudObject(pcObj);

                        // Always show bounding box for point clouds if enabled, or if selected
                        if (obj.Selected || IniSettings.Instance.ShowPointCloudBounds)
                        {
                            DrawPointCloudBoundingBox(pcObj);
                        }
                    }
                }
                else if (obj is SkeletonObject skelObj)
                {
                    if (skelObj.RenderMode == ObjectRenderMode.BoundingBoxOnly)
                    {
                        DrawSelectionOutline(skelObj);
                    }
                    else
                    {
                        DrawSkeletonObject(skelObj);
                    }
                }

                GL.PopMatrix();
            }
        }

        private void DrawLegacyMeshes()
        {
            if (!EnsureLegacySupport("legacy meshes")) return;

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
                if (settings.ShowMesh)
                {
                    DrawMesh(mesh, false, settings.ShowTexture);
                }

                if (settings.ShowPointCloud)
                {
                    DrawPointCloud(mesh, false);
                }
            }

            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
        }

        private void DrawPointCloud(MeshData mesh, bool isSelected)
        {
            if (!_legacySupported)
            {
                EnsureLegacySupport("mesh point cloud");
                return;
            }

            if (mesh.Vertices.Count == 0) return;

            var settings = IniSettings.Instance;
            GL.PointSize(isSelected ? 6.0f : 4.0f);

            switch (settings.PointCloudColor)
            {
                case PointCloudColorMode.DistanceMap:
                    DrawMappedPointCloud(mesh.Vertices, mesh.Confidence, PointCloudColorMode.DistanceMap);
                    break;
                case PointCloudColorMode.Confidence:
                    DrawMappedPointCloud(mesh.Vertices, mesh.Confidence, PointCloudColorMode.Confidence);
                    break;
                default:
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
                    break;
            }
        }

        private void DrawMappedPointCloud(IReadOnlyList<Vector3> points, IReadOnlyList<float> confidence, PointCloudColorMode mode)
        {
            if (!_legacySupported)
            {
                EnsureLegacySupport("mapped point cloud");
                return;
            }

            if (points.Count == 0) return;

            bool useDepth = mode == PointCloudColorMode.DistanceMap || confidence.Count < points.Count;
            float minValue = float.MaxValue;
            float maxValue = float.MinValue;

            if (useDepth)
            {
                foreach (var p in points)
                {
                    float dist = p.Length;
                    if (dist < minValue) minValue = dist;
                    if (dist > maxValue) maxValue = dist;
                }
                UpdateColorLegend(
                    mode == PointCloudColorMode.Confidence ? ColorLegendMode.ConfidenceDepthFallback : ColorLegendMode.Depth,
                    minValue,
                    maxValue);
            }
            else
            {
                for (int i = 0; i < points.Count; i++)
                {
                    float c = confidence[i];
                    if (float.IsNaN(c) || float.IsInfinity(c))
                        continue;
                    if (c < minValue) minValue = c;
                    if (c > maxValue) maxValue = c;
                }
                if (minValue == float.MaxValue || maxValue == float.MinValue)
                {
                    useDepth = true;
                    foreach (var p in points)
                    {
                        float dist = p.Length;
                        if (dist < minValue) minValue = dist;
                        if (dist > maxValue) maxValue = dist;
                    }
                    UpdateColorLegend(ColorLegendMode.ConfidenceDepthFallback, minValue, maxValue);
                }
                else
                {
                    UpdateColorLegend(ColorLegendMode.Confidence, minValue, maxValue);
                }
            }

            float range = maxValue - minValue;
            if (range < 0.0001f) range = 1.0f;

            GL.Begin(PrimitiveType.Points);
            for (int i = 0; i < points.Count; i++)
            {
                float value = useDepth ? points[i].Length : confidence[i];
                float t = (value - minValue) / range;
                Vector3 color = TurboColormap(t);
                GL.Color3(color.X, color.Y, color.Z);
                GL.Vertex3(points[i]);
            }
            GL.End();
        }

        private void DrawPointCloudObject(PointCloudObject pc)
        {
            int visibleCount = pc.GetVisiblePointCount();
            if (visibleCount == 0) return;

            GL.PointSize(pc.PointSize);
            var colorMode = IniSettings.Instance.PointCloudColor;

            // Use modern GL path if available (for Core profile compatibility)
            if (_useModernGL && _shader != null)
            {
                DrawPointCloudModern(pc);
                return;
            }

            // Legacy fixed-function path (requires Compatibility profile)
            if (!EnsureLegacySupport("point cloud")) return;

            GL.Begin(PrimitiveType.Points);

            bool hasColors = pc.Colors.Count >= pc.Points.Count;
            bool useDepthMap = colorMode == PointCloudColorMode.DistanceMap;
            bool useConfidence = colorMode == PointCloudColorMode.Confidence;
            bool confidenceFallbackDepth = useConfidence && pc.Confidence.Count < pc.Points.Count;

            float minValue = 0.0f;
            float range = 1.0f;
            if (useDepthMap || confidenceFallbackDepth)
            {
                float minDist = float.MaxValue;
                float maxDist = float.MinValue;
                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count) continue;
                    float d = pc.Points[sourceIndex].Length;
                    if (d < minDist) minDist = d;
                    if (d > maxDist) maxDist = d;
                }
                minValue = minDist;
                range = maxDist - minDist;
                if (range < 0.0001f) range = 1.0f;
                UpdateColorLegend(
                    confidenceFallbackDepth ? ColorLegendMode.ConfidenceDepthFallback : ColorLegendMode.Depth,
                    minDist,
                    maxDist);
            }
            else if (useConfidence)
            {
                float minConf = float.MaxValue;
                float maxConf = float.MinValue;
                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count || sourceIndex >= pc.Confidence.Count) continue;
                    float c = pc.Confidence[sourceIndex];
                    if (float.IsNaN(c) || float.IsInfinity(c)) continue;
                    if (c < minConf) minConf = c;
                    if (c > maxConf) maxConf = c;
                }
                if (minConf == float.MaxValue || maxConf == float.MinValue)
                {
                    minConf = 0.0f;
                    maxConf = 1.0f;
                }
                minValue = minConf;
                range = maxConf - minConf;
                if (range < 0.0001f) range = 1.0f;
                UpdateColorLegend(ColorLegendMode.Confidence, minConf, maxConf);
            }

            for (int i = 0; i < visibleCount; i++)
            {
                int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                if (sourceIndex < 0 || sourceIndex >= pc.Points.Count)
                    continue;

                if (useDepthMap || confidenceFallbackDepth || useConfidence)
                {
                    float value = useConfidence && !confidenceFallbackDepth && sourceIndex < pc.Confidence.Count
                        ? pc.Confidence[sourceIndex]
                        : pc.Points[sourceIndex].Length;
                    float t = (value - minValue) / range;
                    var col = TurboColormap(t);
                    GL.Color3(col.X, col.Y, col.Z);
                }
                else if (hasColors)
                {
                    var c = pc.Colors[sourceIndex];
                    GL.Color3(c.X, c.Y, c.Z);
                }
                else
                {
                    // Default white color if no colors available
                    GL.Color3(1.0f, 1.0f, 1.0f);
                }
                GL.Vertex3(pc.Points[sourceIndex]);
            }

            GL.End();
        }

        private void DrawPointCloudModern(PointCloudObject pc)
        {
            int visibleCount = pc.GetVisiblePointCount();
            if (visibleCount == 0)
                return;

            var colorMode = IniSettings.Instance.PointCloudColor;
            bool useDepthMap = colorMode == PointCloudColorMode.DistanceMap;
            bool useConfidence = colorMode == PointCloudColorMode.Confidence;
            bool confidenceFallbackDepth = useConfidence && pc.Confidence.Count < pc.Points.Count;

            float minValue = 0.0f;
            float range = 1.0f;
            if (useDepthMap || confidenceFallbackDepth)
            {
                float minDist = float.MaxValue;
                float maxDist = float.MinValue;
                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count) continue;
                    float d = pc.Points[sourceIndex].Length;
                    if (d < minDist) minDist = d;
                    if (d > maxDist) maxDist = d;
                }
                minValue = minDist;
                range = maxDist - minDist;
                if (range < 0.0001f) range = 1.0f;
                UpdateColorLegend(
                    confidenceFallbackDepth ? ColorLegendMode.ConfidenceDepthFallback : ColorLegendMode.Depth,
                    minDist,
                    maxDist);
            }
            else if (useConfidence)
            {
                float minConf = float.MaxValue;
                float maxConf = float.MinValue;
                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count || sourceIndex >= pc.Confidence.Count) continue;
                    float c = pc.Confidence[sourceIndex];
                    if (float.IsNaN(c) || float.IsInfinity(c)) continue;
                    if (c < minConf) minConf = c;
                    if (c > maxConf) maxConf = c;
                }
                if (minConf == float.MaxValue || maxConf == float.MinValue)
                {
                    minConf = 0.0f;
                    maxConf = 1.0f;
                }
                minValue = minConf;
                range = maxConf - minConf;
                if (range < 0.0001f) range = 1.0f;
                UpdateColorLegend(ColorLegendMode.Confidence, minConf, maxConf);
            }

            // Create or update VAO/VBO for this point cloud
            if (!_pointCloudBuffers.TryGetValue(pc.Id, out var buffers)
                || buffers.count != visibleCount
                || buffers.colorMode != colorMode)
            {
                // Delete old buffers if they exist
                if (buffers.vao != 0)
                {
                    GL.DeleteVertexArray(buffers.vao);
                    GL.DeleteBuffer(buffers.vbo);
                }

                // Create interleaved buffer: position (vec3) + color (vec3)
                var data = new float[visibleCount * 6];
                bool hasColors = pc.Colors.Count >= pc.Points.Count;

                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count)
                        continue;

                    var p = pc.Points[sourceIndex];
                    data[i * 6 + 0] = p.X;
                    data[i * 6 + 1] = p.Y;
                    data[i * 6 + 2] = p.Z;

                    if (useDepthMap || confidenceFallbackDepth || useConfidence)
                    {
                        float value = useConfidence && !confidenceFallbackDepth && sourceIndex < pc.Confidence.Count
                            ? pc.Confidence[sourceIndex]
                            : p.Length;
                        float t = (value - minValue) / range;
                        var c = TurboColormap(t);
                        data[i * 6 + 3] = c.X;
                        data[i * 6 + 4] = c.Y;
                        data[i * 6 + 5] = c.Z;
                    }
                    else if (hasColors)
                    {
                        var c = pc.Colors[sourceIndex];
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

                _pointCloudBuffers[pc.Id] = (vao, vbo, visibleCount, colorMode);
                buffers = (vao, vbo, visibleCount, colorMode);

                // Log sample data for debugging
                Console.WriteLine($"Created modern GL buffers for point cloud {pc.Id}: {visibleCount}/{pc.Points.Count} visible points");
                if (visibleCount > 0)
                {
                    Console.WriteLine($"  Sample point 0: pos=({data[0]:F3},{data[1]:F3},{data[2]:F3}) color=({data[3]:F2},{data[4]:F2},{data[5]:F2})");
                    if (visibleCount > 100)
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
            if (!EnsureLegacySupport("point cloud bounding box")) return;

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

        private void UpdateColorLegend(ColorLegendMode mode, float minValue, float maxValue)
        {
            _colorLegendMode = mode;
            _colorLegendMin = minValue;
            _colorLegendMax = maxValue;
        }

        private static bool ShouldDrawMeshSurface(MeshObject meshObj, IniSettings settings)
        {
            return meshObj.RenderMode switch
            {
                ObjectRenderMode.Shaded => true,
                ObjectRenderMode.Wireframe => true,
                ObjectRenderMode.NoTexture => true,
                ObjectRenderMode.Texture => true,
                ObjectRenderMode.BoundingBoxOnly => false,
                _ => settings.ShowMesh
            };
        }

        private static bool ShouldDrawMeshAsPointCloud(MeshObject meshObj, IniSettings settings)
        {
            if (meshObj.RenderMode != ObjectRenderMode.InheritGlobal)
                return false;

            return settings.ShowPointCloud || meshObj.ShowAsPointCloud;
        }

        private static bool ResolveWireframe(MeshObject meshObj, IniSettings settings)
        {
            return meshObj.RenderMode switch
            {
                ObjectRenderMode.Wireframe => true,
                ObjectRenderMode.Shaded => false,
                ObjectRenderMode.NoTexture => false,
                ObjectRenderMode.Texture => false,
                ObjectRenderMode.BoundingBoxOnly => false,
                _ => settings.ShowWireframe || meshObj.ShowWireframe
            };
        }

        private static bool ResolveTexture(MeshObject meshObj, IniSettings settings)
        {
            return meshObj.RenderMode switch
            {
                ObjectRenderMode.Texture => true,
                ObjectRenderMode.NoTexture => false,
                ObjectRenderMode.Wireframe => false,
                ObjectRenderMode.Shaded => false,
                ObjectRenderMode.BoundingBoxOnly => false,
                _ => settings.ShowTexture
            };
        }

        private void DrawMesh(MeshData mesh, bool isSelected, bool allowTexture)
        {
            if (!EnsureLegacySupport("mesh")) return;

            if (mesh.Vertices.Count == 0 || mesh.Indices.Count == 0) return;

            bool useTexture = allowTexture && mesh.Texture != null;

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
            if (!EnsureLegacySupport("selection outline")) return;

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
            if (!EnsureLegacySupport("corner box")) return;

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

            if (_useModernGL && _shader != null)
            {
                DrawCamerasModern();
                return;
            }

            if (!EnsureLegacySupport("camera frustums")) return;

            foreach (var cam in _sceneGraph.GetObjectsOfType<CameraObject>())
            {
                if (!cam.Visible || !cam.ShowFrustum) continue;

                DrawCameraFrustum(cam);
            }
        }

        private void DrawCamerasModern()
        {
            var cameras = _sceneGraph!.GetObjectsOfType<CameraObject>()
                .Where(c => c.Visible && c.ShowFrustum).ToList();

            if (cameras.Count == 0) return;

            List<float> vertices = new List<float>();

            foreach (var cam in cameras)
            {
                Vector3 pos = cam.Position;
                if (cam.Pose != null)
                {
                    pos = cam.Pose.CameraToWorld.ExtractTranslation();
                }

                var corners = cam.GetFrustumCorners(CameraFrustumScale);
                var color = cam.Selected ? new Vector3(1f, 1f, 0f) : cam.FrustumColor;

                void AddLine(Vector3 p1, Vector3 p2)
                {
                    vertices.Add(p1.X); vertices.Add(p1.Y); vertices.Add(p1.Z);
                    vertices.Add(color.X); vertices.Add(color.Y); vertices.Add(color.Z);
                    vertices.Add(p2.X); vertices.Add(p2.Y); vertices.Add(p2.Z);
                    vertices.Add(color.X); vertices.Add(color.Y); vertices.Add(color.Z);
                }

                for (int i = 0; i < 4; i++) AddLine(pos, corners[i]);

                AddLine(corners[0], corners[1]);
                AddLine(corners[1], corners[2]);
                AddLine(corners[2], corners[3]);
                AddLine(corners[3], corners[0]);
                AddLine(corners[0], corners[2]);
                AddLine(corners[1], corners[3]);
            }

            if (_cameraVao == 0)
            {
                _cameraVao = GL.GenVertexArray();
                _cameraVbo = GL.GenBuffer();
            }

            GL.BindVertexArray(_cameraVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _cameraVbo);
            GL.BufferData(BufferTarget.ArrayBuffer, vertices.Count * sizeof(float), vertices.ToArray(), BufferUsageHint.DynamicDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            _shader!.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", Matrix4.Identity);

            GL.DrawArrays(PrimitiveType.Lines, 0, vertices.Count / 6);

            GL.BindVertexArray(0);
            GL.UseProgram(0);
        }

        private void DrawCameraFrustum(CameraObject cam)
        {
            if (!EnsureLegacySupport("camera frustum")) return;

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

            // Draw camera body (Legacy GL only for now)
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
            if (!EnsureLegacySupport("triangle highlight")) return;

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
            if (!EnsureLegacySupport("gizmo")) return;

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
            if (!EnsureLegacySupport("crop box")) return;

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
            if (!EnsureLegacySupport("info overlay")) return;

            // Setup 2D projection
            GL.MatrixMode(MatrixMode.Projection);
            GL.PushMatrix();
            GL.LoadIdentity();
            GL.Ortho(0, width, height, 0, -1, 1);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.PushMatrix();
            GL.LoadIdentity();

            GL.Disable(EnableCap.DepthTest);

            bool showLegend = _colorLegendMode != ColorLegendMode.None;
            float panelHeight = showLegend ? 118.0f : 80.0f;
            if (_hasHoveredSourceDistance)
                panelHeight += 34.0f;

            // Draw info background
            GL.Color4(0.0f, 0.0f, 0.0f, 0.5f);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(5, 5);
            GL.Vertex2(200, 5);
            GL.Vertex2(200, panelHeight);
            GL.Vertex2(5, panelHeight);
            GL.End();

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

            if (showLegend)
            {
                float x0 = 14.0f;
                float x1 = 190.0f;
                float y0 = 86.0f;
                float y1 = 102.0f;
                int steps = 36;

                GL.Begin(PrimitiveType.Quads);
                for (int i = 0; i < steps; i++)
                {
                    float t0 = (float)i / steps;
                    float t1 = (float)(i + 1) / steps;
                    var c0 = TurboColormap(t0);
                    var c1 = TurboColormap(t1);
                    float sx0 = x0 + (x1 - x0) * t0;
                    float sx1 = x0 + (x1 - x0) * t1;

                    GL.Color3(c0.X, c0.Y, c0.Z);
                    GL.Vertex2(sx0, y0);
                    GL.Vertex2(sx0, y1);
                    GL.Color3(c1.X, c1.Y, c1.Z);
                    GL.Vertex2(sx1, y1);
                    GL.Vertex2(sx1, y0);
                }
                GL.End();

                // Border and ticks
                GL.Color3(0.2f, 0.2f, 0.2f);
                GL.LineWidth(1.0f);
                GL.Begin(PrimitiveType.LineLoop);
                GL.Vertex2(x0, y0);
                GL.Vertex2(x1, y0);
                GL.Vertex2(x1, y1);
                GL.Vertex2(x0, y1);
                GL.End();

                GL.Begin(PrimitiveType.Lines);
                GL.Vertex2(x0, y1 + 1);
                GL.Vertex2(x0, y1 + 8);
                GL.Vertex2((x0 + x1) * 0.5f, y1 + 1);
                GL.Vertex2((x0 + x1) * 0.5f, y1 + 6);
                GL.Vertex2(x1, y1 + 1);
                GL.Vertex2(x1, y1 + 8);
                GL.End();
            }

            if (_hasHoveredSourceDistance)
            {
                string unit = _hoveredDistanceIsMeters ? "m" : "px";
                string cameraText = $"SrcCam: {_hoveredSourceCameraName}";
                string distanceText = $"Dist: {_hoveredSourceDistance:F3} {unit}";
                float textY = showLegend ? 111.0f : 72.0f;
                _overlayTextRenderer.DrawText(cameraText, 12.0f, textY, (0.95f, 0.95f, 0.95f, 1.0f), (0f, 0f, 0f, 0.6f));
                _overlayTextRenderer.DrawText(distanceText, 12.0f, textY + 14.0f, (1.0f, 0.88f, 0.45f, 1.0f), (0f, 0f, 0f, 0.6f));
            }

            GL.Enable(EnableCap.DepthTest);

            // Restore matrices
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Projection);
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Modelview);
        }

        #region Skeleton Rendering

        /// <summary>
        /// Draw a skeleton object with joints and bones
        /// </summary>
        private void DrawSkeletonObject(SkeletonObject skelObj)
        {
            if (!EnsureLegacySupport("skeleton")) return;
            if (skelObj.Skeleton == null) return;

            var skeleton = skelObj.Skeleton;
            var objPos = skelObj.Position;

            // Draw bones first (behind joints)
            if (skelObj.ShowBones)
            {
                foreach (var bone in skeleton.Bones)
                {
                    if (!bone.IsVisible) continue;

                    var start = bone.StartJoint.GetWorldPosition() + objPos;
                    var end = bone.EndJoint.GetWorldPosition() + objPos;

                    // Determine color
                    Vector3 color;
                    if (bone.IsSelected)
                        color = skelObj.SelectedColor;
                    else
                        color = skelObj.BoneColor;

                    DrawBone(start, end, skelObj.BoneDisplayThickness, color);
                }
            }

            // Draw joints
            if (skelObj.ShowJoints)
            {
                foreach (var joint in skeleton.Joints)
                {
                    if (!joint.IsVisible) continue;

                    var pos = joint.GetWorldPosition() + objPos;

                    // Determine color
                    Vector3 color;
                    if (joint.IsSelected)
                        color = skelObj.SelectedColor;
                    else
                        color = joint.Color;

                    float size = joint.JointSize > 0 ? joint.JointSize : skelObj.JointDisplaySize;
                    DrawJoint(pos, size, color);
                }
            }
        }

        /// <summary>
        /// Draw a single joint as a sphere/octahedron
        /// </summary>
        private void DrawJoint(Vector3 position, float size, Vector3 color)
        {
            if (!EnsureLegacySupport("joint")) return;

            GL.Color3(color.X, color.Y, color.Z);

            // Draw as an octahedron for better visibility
            float h = size;

            GL.Begin(PrimitiveType.Triangles);

            // Top pyramid
            // Front
            GL.Vertex3(position.X, position.Y + h, position.Z);
            GL.Vertex3(position.X + h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z + h);

            // Right
            GL.Vertex3(position.X, position.Y + h, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z + h);
            GL.Vertex3(position.X - h, position.Y, position.Z);

            // Back
            GL.Vertex3(position.X, position.Y + h, position.Z);
            GL.Vertex3(position.X - h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z - h);

            // Left
            GL.Vertex3(position.X, position.Y + h, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z - h);
            GL.Vertex3(position.X + h, position.Y, position.Z);

            // Bottom pyramid
            // Front
            GL.Vertex3(position.X, position.Y - h, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z + h);
            GL.Vertex3(position.X + h, position.Y, position.Z);

            // Right
            GL.Vertex3(position.X, position.Y - h, position.Z);
            GL.Vertex3(position.X - h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z + h);

            // Back
            GL.Vertex3(position.X, position.Y - h, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z - h);
            GL.Vertex3(position.X - h, position.Y, position.Z);

            // Left
            GL.Vertex3(position.X, position.Y - h, position.Z);
            GL.Vertex3(position.X + h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z - h);

            GL.End();

            // Draw outline
            GL.LineWidth(1.5f);
            GL.Color3(color.X * 0.5f, color.Y * 0.5f, color.Z * 0.5f);
            GL.Begin(PrimitiveType.LineLoop);
            GL.Vertex3(position.X + h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z + h);
            GL.Vertex3(position.X - h, position.Y, position.Z);
            GL.Vertex3(position.X, position.Y, position.Z - h);
            GL.End();
            GL.LineWidth(1.0f);
        }

        /// <summary>
        /// Draw a bone connecting two joints
        /// </summary>
        private void DrawBone(Vector3 start, Vector3 end, float thickness, Vector3 color)
        {
            if (!EnsureLegacySupport("bone")) return;

            var direction = end - start;
            float length = direction.Length;
            if (length < 0.0001f) return;

            direction = direction.Normalized();

            // Find perpendicular vectors
            Vector3 up = Math.Abs(direction.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX;
            Vector3 right = Vector3.Cross(direction, up).Normalized();
            up = Vector3.Cross(right, direction).Normalized();

            float t = thickness;
            float taperFactor = 0.3f; // Taper toward end

            // Calculate corner points
            var p1 = start + right * t + up * t;
            var p2 = start + right * t - up * t;
            var p3 = start - right * t - up * t;
            var p4 = start - right * t + up * t;

            var p5 = end + right * t * taperFactor + up * t * taperFactor;
            var p6 = end + right * t * taperFactor - up * t * taperFactor;
            var p7 = end - right * t * taperFactor - up * t * taperFactor;
            var p8 = end - right * t * taperFactor + up * t * taperFactor;

            GL.Color3(color.X, color.Y, color.Z);

            GL.Begin(PrimitiveType.Quads);

            // Side 1
            GL.Vertex3(p1); GL.Vertex3(p2); GL.Vertex3(p6); GL.Vertex3(p5);
            // Side 2
            GL.Vertex3(p2); GL.Vertex3(p3); GL.Vertex3(p7); GL.Vertex3(p6);
            // Side 3
            GL.Vertex3(p3); GL.Vertex3(p4); GL.Vertex3(p8); GL.Vertex3(p7);
            // Side 4
            GL.Vertex3(p4); GL.Vertex3(p1); GL.Vertex3(p5); GL.Vertex3(p8);

            GL.End();

            // Draw end caps
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex3(p1); GL.Vertex3(p4); GL.Vertex3(p3); GL.Vertex3(p2);
            GL.Vertex3(p5); GL.Vertex3(p6); GL.Vertex3(p7); GL.Vertex3(p8);
            GL.End();

            // Draw outline for better visibility
            GL.LineWidth(1.5f);
            GL.Color3(color.X * 0.6f, color.Y * 0.6f, color.Z * 0.6f);

            GL.Begin(PrimitiveType.Lines);
            // Edges from start to end
            GL.Vertex3(p1); GL.Vertex3(p5);
            GL.Vertex3(p2); GL.Vertex3(p6);
            GL.Vertex3(p3); GL.Vertex3(p7);
            GL.Vertex3(p4); GL.Vertex3(p8);
            GL.End();

            GL.LineWidth(1.0f);
        }

        /// <summary>
        /// Draw a gizmo for manipulating joints in rigging mode
        /// </summary>
        public void DrawJointGizmo(Joint joint, Vector3 objectPosition)
        {
            if (!EnsureLegacySupport("joint gizmo")) return;

            var pos = joint.GetWorldPosition() + objectPosition;

            // Calculate gizmo size based on distance to camera
            float distToCamera = Math.Abs(_zoom);
            float gizmoSize = distToCamera * 0.1f;

            GL.Disable(EnableCap.DepthTest);
            GL.LineWidth(2.5f);

            // Draw coordinate axes
            GL.Begin(PrimitiveType.Lines);

            // X axis (red)
            GL.Color3(1.0f, 0.2f, 0.2f);
            GL.Vertex3(pos);
            GL.Vertex3(pos + new Vector3(gizmoSize, 0, 0));

            // Y axis (green)
            GL.Color3(0.2f, 1.0f, 0.2f);
            GL.Vertex3(pos);
            GL.Vertex3(pos + new Vector3(0, gizmoSize, 0));

            // Z axis (blue)
            GL.Color3(0.2f, 0.4f, 1.0f);
            GL.Vertex3(pos);
            GL.Vertex3(pos + new Vector3(0, 0, gizmoSize));

            GL.End();

            GL.LineWidth(1.0f);
            GL.Enable(EnableCap.DepthTest);
        }

        #endregion

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

        private void ClearHoveredSourceDistance()
        {
            bool hadValue = _hasHoveredSourceDistance;
            _hasHoveredSourceDistance = false;
            _hoveredSourceDistance = 0.0f;
            _hoveredDistanceIsMeters = false;
            _hoveredSourceCameraName = string.Empty;
            if (hadValue)
                this.QueueRender();
        }

        private void UpdateHoveredSourceDistanceFromMouse(int mouseX, int mouseY)
        {
            if (_sceneGraph == null)
            {
                ClearHoveredSourceDistance();
                return;
            }

            int width = this.Allocation.Width;
            int height = this.Allocation.Height;
            if (width <= 0 || height <= 0)
            {
                ClearHoveredSourceDistance();
                return;
            }

            if (!TryPickHoveredPoint(mouseX, mouseY, width, height, out var hoveredPointWorld))
            {
                ClearHoveredSourceDistance();
                return;
            }

            if (TryComputeSourceCameraDistance(hoveredPointWorld, out float distanceValue, out bool isMeters, out string sourceCamera))
            {
                _hasHoveredSourceDistance = true;
                _hoveredSourceDistance = distanceValue;
                _hoveredDistanceIsMeters = isMeters;
                _hoveredSourceCameraName = sourceCamera;
                this.QueueRender();
                return;
            }

            ClearHoveredSourceDistance();
        }

        private bool TryPickHoveredPoint(int mouseX, int mouseY, int width, int height, out Vector3 pointWorld)
        {
            pointWorld = Vector3.Zero;
            if (_sceneGraph == null)
                return false;

            Vector2 mouse = new Vector2(mouseX, mouseY);
            float bestDistPx = 12.0f;
            bool found = false;
            const int maxSamplesPerCloud = 20000;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                if (obj is not PointCloudObject pc || pc.Points.Count == 0)
                    continue;

                int visibleCount = pc.GetVisiblePointCount();
                if (visibleCount <= 0)
                    continue;

                int step = Math.Max(1, visibleCount / maxSamplesPerCloud);
                Matrix4 world = pc.GetWorldTransform();

                for (int i = 0; i < visibleCount; i += step)
                {
                    int sourceIdx = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIdx < 0 || sourceIdx >= pc.Points.Count)
                        continue;

                    Vector3 pWorld = Vector3.TransformPosition(pc.Points[sourceIdx], world);
                    if (!TryProjectToScreen(pWorld, width, height, out Vector2 screenPos, out _))
                        continue;

                    float distPx = (screenPos - mouse).Length;
                    if (distPx < bestDistPx)
                    {
                        bestDistPx = distPx;
                        pointWorld = pWorld;
                        found = true;
                    }
                }
            }

            return found;
        }

        private bool TryProjectToScreen(Vector3 pos, int width, int height, out Vector2 screenPos, out float ndcDepth)
        {
            screenPos = Vector2.Zero;
            ndcDepth = 0.0f;
            if (width <= 0 || height <= 0)
                return false;

            Vector4 vec = new Vector4(pos, 1.0f);
            vec = vec * _finalViewMatrix;
            vec = vec * _projectionMatrix;

            if (Math.Abs(vec.W) < 1e-6f || vec.W <= 0.0f)
                return false;

            vec /= vec.W;
            ndcDepth = vec.Z;

            if (ndcDepth < -1.0f || ndcDepth > 1.0f)
                return false;

            float x = (vec.X + 1.0f) * 0.5f * width;
            float y = (1.0f - vec.Y) * 0.5f * height;
            screenPos = new Vector2(x, y);
            return true;
        }

        private bool TryComputeSourceCameraDistance(Vector3 pointWorld, out float value, out bool isMeters, out string sourceCameraName)
        {
            value = 0.0f;
            isMeters = false;
            sourceCameraName = string.Empty;

            if (_sceneGraph == null)
                return false;

            CameraObject? sourceCamera = null;
            float bestDist2 = float.MaxValue;
            Vector3 sourceCameraPos = Vector3.Zero;

            foreach (var camera in _sceneGraph.GetObjectsOfType<CameraObject>())
            {
                if (!camera.Visible || camera.Pose == null)
                    continue;

                Vector3 camPos = camera.Pose.CameraToWorld.ExtractTranslation();
                float d2 = (camPos - pointWorld).LengthSquared;
                if (d2 < bestDist2)
                {
                    bestDist2 = d2;
                    sourceCamera = camera;
                    sourceCameraPos = camPos;
                }
            }

            if (sourceCamera?.Pose == null)
                return false;

            sourceCameraName = sourceCamera.Name;

            if (GeoReferenceRuntime.HasActiveGeoreference)
            {
                Vector3 pointGeo = GeoReferenceService.TransformModelToWorld(pointWorld);
                Vector3 camGeo = GeoReferenceService.TransformModelToWorld(sourceCameraPos);
                value = (pointGeo - camGeo).Length;
                isMeters = true;
                return float.IsFinite(value);
            }

            float fx = sourceCamera.Pose.GetEffectiveFocalLength();
            if (fx <= 1e-6f)
                return false;

            var pointCam = Vector3.TransformPosition(pointWorld, sourceCamera.Pose.WorldToCamera);
            float depth = Math.Abs(pointCam.Z);
            if (!float.IsFinite(depth) || depth <= 1e-6f)
            {
                var fallbackW2C = sourceCamera.Pose.CameraToWorld.Inverted();
                pointCam = Vector3.TransformPosition(pointWorld, fallbackW2C);
                depth = Math.Abs(pointCam.Z);
            }

            if (!float.IsFinite(depth) || depth <= 1e-6f)
                return false;

            value = fx / depth;
            isMeters = false;
            return float.IsFinite(value);
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
                this.QueueRender();
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

            if (args.Event.Button == 1)
            {
                if (_gizmoMode == GizmoMode.SplittingPlane && _splittingPlaneTool != null)
                {
                    var (rayOrigin, rayDir) = GetRayFromScreenPoint((int)args.Event.X, (int)args.Event.Y);
                    int handleAxis = _splittingPlaneTool.CheckHandleIntersection(rayOrigin, rayDir, _finalViewMatrix, _projectionMatrix);
                    
                    if (handleAxis >= 0)
                    {
                        _splittingPlaneTool.StartDrag(handleAxis, rayOrigin, rayDir);
                        _isDragging = true;
                        _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
                        return;
                    }
                }

                if (_gizmoMode == GizmoMode.Pen)
                {
                    HandlePenModeClick((int)args.Event.X, (int)args.Event.Y, args.Event.State);
                    _isDragging = true;
                    _lastMousePos = new Point((int)args.Event.X, (int)args.Event.Y);
                    return;
                }

                int gizmoAxis = CheckGizmoSelection((int)args.Event.X, (int)args.Event.Y);
                if (gizmoAxis >= 0 && _sceneGraph?.SelectedObjects.Count > 0)
                {
                    _activeGizmoAxis = gizmoAxis;
                    _isDraggingGizmo = true;
                    _gizmoDragStart = new Vector3((float)args.Event.X, (float)args.Event.Y, 0);
                    _gizmoStartStates.Clear();
                    foreach (var obj in _sceneGraph.SelectedObjects)
                    {
                        _gizmoStartStates[obj.Id] = new GizmoStartState(obj.Position, obj.Rotation, obj.Scale);
                    }

                    this.QueueRender();
                    return;
                }

                if (_selectedHandle != -1)
                {
                    _isDragging = true;
                }
                else
                {
                    var picked = PickObject((int)args.Event.X, (int)args.Event.Y);

                    bool multipleSelection = (args.Event.State & Gdk.ModifierType.ShiftMask) != 0 ||
                                             (args.Event.State & Gdk.ModifierType.ControlMask) != 0;

                    if (picked != null && _sceneGraph != null)
                    {
                        if (multipleSelection)
                        {
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
                        _sceneGraph.ClearSelection();
                    }

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
                if (_splittingPlaneTool != null)
                {
                    _splittingPlaneTool.EndDrag();
                }
                _isDragging = false;
                _isPanning = false;
                _isDraggingGizmo = false;
                _gizmoStartStates.Clear();
                _activeGizmoAxis = -1;
                this.QueueRender();
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
            UpdateHoveredSourceDistanceFromMouse(x, y);

            if (!_isDragging && !_isPanning && !_isDraggingGizmo)
            {
                if (_gizmoMode == GizmoMode.SplittingPlane && _splittingPlaneTool != null)
                {
                    var (rayOrigin, rayDir) = GetRayFromScreenPoint(x, y);
                    int handleAxis = _splittingPlaneTool.CheckHandleIntersection(rayOrigin, rayDir, _finalViewMatrix, _projectionMatrix);
                    _splittingPlaneTool.EndDrag();
                }
                
                CheckHandleSelection(x, y);
                int gizmoAxis = CheckGizmoSelection(x, y);
                if (_activeGizmoAxis != gizmoAxis)
                {
                    _activeGizmoAxis = gizmoAxis;
                    this.QueueRender();
                }
            }

            if (_isDragging && _gizmoMode == GizmoMode.SplittingPlane && _splittingPlaneTool != null)
            {
                var (rayOrigin, rayDir) = GetRayFromScreenPoint(x, y);
                _splittingPlaneTool.UpdateDrag(rayOrigin, rayDir);
                this.QueueRender();
            }
            else if (_isDraggingGizmo && _sceneGraph != null)
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
                    if (!_gizmoStartStates.TryGetValue(obj.Id, out var start))
                        continue;

                    switch (_gizmoMode)
                    {
                        case GizmoMode.Translate:
                            obj.Position = start.Position + delta;
                            break;
                        case GizmoMode.Rotate:
                            obj.Rotation = start.Rotation + delta * 50.0f;
                            break;
                        case GizmoMode.Scale:
                            obj.Scale = ClampScale(start.Scale + delta);
                            break;
                    }
                }

                this.QueueRender();
            }
            else if (_isDragging && !_isPanning)
            {
                int deltaX = x - _lastMousePos.X;
                int deltaY = y - _lastMousePos.Y;

                if (_selectedHandle != -1)
                {
                    _cropSize += deltaX * 0.01f;
                    if (_cropSize < 0.1f) _cropSize = 0.1f;
                    UpdateCropCorners();
                    this.QueueRender();
                }
                else
                {
                    _rotationY += deltaX * 0.5f;
                    _rotationX += deltaY * 0.5f;
                }
                _lastMousePos = new Point(x, y);
                this.QueueRender();
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
                this.QueueRender();
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
            this.QueueRender();
        }

        private void OnKeyPress(object o, KeyPressEventArgs args)
        {
            switch (args.Event.Key)
            {
                case Gdk.Key.q:
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
                case Gdk.Key.y:
                case Gdk.Key.Y:
                    SetGizmoMode(GizmoMode.SplittingPlane);
                    break;
                case Gdk.Key.f:
                case Gdk.Key.F:
                    FocusOnSelection();
                    break;
                case Gdk.Key.Return:
                case Gdk.Key.KP_Enter:
                    if (_gizmoMode == GizmoMode.SplittingPlane)
                    {
                        SplittingPlaneConfirmed?.Invoke(this, EventArgs.Empty);
                    }
                    break;
                case Gdk.Key.Delete:
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
                    if (_gizmoMode == GizmoMode.SplittingPlane)
                    {
                        SetGizmoMode(GizmoMode.Select);
                    }
                    else if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
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
            this.QueueRender();
        }

        private static Vector3 ClampScale(Vector3 scale)
        {
            const float minScale = 0.001f;
            return new Vector3(
                Math.Max(minScale, scale.X),
                Math.Max(minScale, scale.Y),
                Math.Max(minScale, scale.Z));
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
            this.QueueRender();
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

        private int _gizmoVao = 0;
        private int _gizmoVbo = 0;

        private int _selectionVao = 0;
        private int _selectionVbo = 0;

        private void DrawSelectedTrianglesModern()
        {
            var vertices = _meshEditingTool.GetSelectedTriangleVertices();
            if (vertices.Count == 0 || _shader == null) return;

            List<float> data = new List<float>();
            Vector3 color = new Vector3(1.0f, 0.5f, 0.0f); // Orange
            foreach (var v in vertices)
            {
                data.Add(v.X); data.Add(v.Y); data.Add(v.Z);
                data.Add(color.X); data.Add(color.Y); data.Add(color.Z);
            }

            if (_selectionVao == 0)
            {
                _selectionVao = GL.GenVertexArray();
                _selectionVbo = GL.GenBuffer();
            }

            GL.BindVertexArray(_selectionVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _selectionVbo);
            GL.BufferData(BufferTarget.ArrayBuffer, data.Count * sizeof(float), data.ToArray(), BufferUsageHint.DynamicDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            _shader.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", Matrix4.Identity);
            
            GL.Enable(EnableCap.Blend);
            GL.Disable(EnableCap.DepthTest);

            // Draw faces
            _shader.SetVector4("uniformColor", new Vector4(color.X, color.Y, color.Z, 0.4f));
            _shader.SetBool("useUniformColor", true);
            GL.DrawArrays(PrimitiveType.Triangles, 0, vertices.Count);

            // Draw edges
            GL.LineWidth(2.0f);
            _shader.SetVector4("uniformColor", new Vector4(color.X, color.Y, color.Z, 1.0f));
            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
            GL.DrawArrays(PrimitiveType.Triangles, 0, vertices.Count);
            GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);
            GL.LineWidth(1.0f);

            _shader.SetBool("useUniformColor", false);
            GL.Enable(EnableCap.DepthTest);
            GL.Disable(EnableCap.Blend);
            GL.BindVertexArray(0);
        }

        private void DrawGizmoModern()
        {
            if (_sceneGraph == null || _sceneGraph.SelectedObjects.Count == 0 || _shader == null) return;

            Vector3 center = Vector3.Zero;
            foreach (var obj in _sceneGraph.SelectedObjects) center += obj.Position;
            center /= _sceneGraph.SelectedObjects.Count;

            float distToCamera = Math.Abs(_zoom);
            _gizmoSize = distToCamera * 0.15f;
            float len = _gizmoSize;

            List<float> lines = new List<float>();
            void AddLine(Vector3 p1, Vector3 p2, Vector3 color)
            {
                lines.Add(p1.X); lines.Add(p1.Y); lines.Add(p1.Z);
                lines.Add(color.X); lines.Add(color.Y); lines.Add(color.Z);
                lines.Add(p2.X); lines.Add(p2.Y); lines.Add(p2.Z);
                lines.Add(color.X); lines.Add(color.Y); lines.Add(color.Z);
            }

            Vector3 red = _activeGizmoAxis == 0 ? new Vector3(1, 1, 0) : new Vector3(1, 0, 0);
            Vector3 green = _activeGizmoAxis == 1 ? new Vector3(1, 1, 0) : new Vector3(0, 1, 0);
            Vector3 blue = _activeGizmoAxis == 2 ? new Vector3(1, 1, 0) : new Vector3(0, 0, 1);

            if (_gizmoMode == GizmoMode.Translate || _gizmoMode == GizmoMode.Scale)
            {
                AddLine(center, center + new Vector3(len, 0, 0), red);
                AddLine(center, center + new Vector3(0, len, 0), green);
                AddLine(center, center + new Vector3(0, 0, len), blue);
            }
            else if (_gizmoMode == GizmoMode.Rotate)
            {
                float r = len * 0.8f;
                for (int i = 0; i < 4; i++)
                {
                    float a1 = i * MathF.PI / 2;
                    float a2 = (i + 1) * MathF.PI / 2;
                    AddLine(center + new Vector3(0, MathF.Cos(a1) * r, MathF.Sin(a1) * r), center + new Vector3(0, MathF.Cos(a2) * r, MathF.Sin(a2) * r), red);
                    AddLine(center + new Vector3(MathF.Cos(a1) * r, 0, MathF.Sin(a1) * r), center + new Vector3(MathF.Cos(a2) * r, 0, MathF.Sin(a2) * r), green);
                    AddLine(center + new Vector3(MathF.Cos(a1) * MathF.Sin(a1) * r, MathF.Sin(a1) * r, 0), center + new Vector3(MathF.Cos(a2) * r, MathF.Sin(a2) * r, 0), blue);
                }
            }

            if (_gizmoVao == 0)
            {
                _gizmoVao = GL.GenVertexArray();
                _gizmoVbo = GL.GenBuffer();
            }

            GL.BindVertexArray(_gizmoVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _gizmoVbo);
            GL.BufferData(BufferTarget.ArrayBuffer, lines.Count * sizeof(float), lines.ToArray(), BufferUsageHint.DynamicDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            _shader.Use();
            _shader.SetMatrix4("projection", _projectionMatrix);
            _shader.SetMatrix4("view", _finalViewMatrix);
            _shader.SetMatrix4("model", Matrix4.Identity);
            _shader.SetBool("useUniformColor", false);

            GL.Disable(EnableCap.DepthTest);
            GL.LineWidth(3.0f);
            GL.DrawArrays(PrimitiveType.Lines, 0, lines.Count / 6);
            GL.LineWidth(1.0f);
            GL.Enable(EnableCap.DepthTest);

            GL.BindVertexArray(0);
        }

        private int _orientVao = 0;
        private int _orientVbo = 0;

        private void DrawOrientationGizmoModern(int width, int height)
        {
            if (_shader == null) return;

            // Setup a small viewport in the corner or use orthographic overlay
            // We'll use orthographic projection for the gizmo
            float size = 60.0f; // size in pixels
            float margin = 20.0f;
            
            // Gizmo centered at (margin + size/2, margin + size/2) in screen space
            // But we'll just use a small view matrix and a fixed projection
            
            Matrix4 ortho = Matrix4.CreateOrthographicOffCenter(0, width, 0, height, -100, 100);
            
            // Get only the rotation part of the view matrix
            var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
            var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
            Matrix4 rotation = ry * rx;
            
            // For Z-up correction if needed
            if (IniSettings.Instance.CoordSystem == CoordinateSystem.RightHanded_Z_Up)
            {
                rotation = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(-90)) * rotation;
            }

            Vector3 center = new Vector3(margin + size, margin + size, 0);
            float axisLen = size * 0.8f;

            List<float> lines = new List<float>();
            void AddAxis(Vector3 dir, Vector3 color)
            {
                Vector3 end = center + Vector3.TransformVector(dir, rotation) * axisLen;
                lines.Add(center.X); lines.Add(center.Y); lines.Add(center.Z);
                lines.Add(color.X); lines.Add(color.Y); lines.Add(color.Z);
                lines.Add(end.X); lines.Add(end.Y); lines.Add(end.Z);
                lines.Add(color.X); lines.Add(color.Y); lines.Add(color.Z);
            }

            AddAxis(Vector3.UnitX, new Vector3(1, 0, 0)); // X - Red
            AddAxis(Vector3.UnitY, new Vector3(0, 1, 0)); // Y - Green
            AddAxis(Vector3.UnitZ, new Vector3(0, 0, 1)); // Z - Blue

            if (_orientVao == 0)
            {
                _orientVao = GL.GenVertexArray();
                _orientVbo = GL.GenBuffer();
            }

            GL.BindVertexArray(_orientVao);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _orientVbo);
            GL.BufferData(BufferTarget.ArrayBuffer, lines.Count * sizeof(float), lines.ToArray(), BufferUsageHint.DynamicDraw);

            GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
            GL.EnableVertexAttribArray(0);
            GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
            GL.EnableVertexAttribArray(1);

            _shader.Use();
            _shader.SetMatrix4("projection", ortho);
            _shader.SetMatrix4("view", Matrix4.Identity);
            _shader.SetMatrix4("model", Matrix4.Identity);
            _shader.SetBool("useUniformColor", false);

            GL.Disable(EnableCap.DepthTest);
            GL.LineWidth(2.5f);
            GL.DrawArrays(PrimitiveType.Lines, 0, 6);
            GL.LineWidth(1.0f);
            GL.Enable(EnableCap.DepthTest);

            GL.BindVertexArray(0);
        }

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
