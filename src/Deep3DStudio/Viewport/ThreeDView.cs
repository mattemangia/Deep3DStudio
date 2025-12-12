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
        Select // Added Select mode
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
        private float _panX = 0f;
        private float _panY = 0f;
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

        // Matrices for picking
        private Matrix4 _viewMatrix;
        private Matrix4 _projectionMatrix;

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
            // Requesting version 2.1 ensures we get a context compatible with
            // the fixed-function pipeline (GL.Begin/End) used in this codebase.
            this.SetRequiredVersion(2, 1);

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

            var center = (min + max) * 0.5f;
            var size = (max - min).Length;

            _panX = -center.X;
            _panY = -center.Y;
            _zoom = -size * 1.5f;

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
                _panX = -center.X;
                _panY = -center.Y;
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

        private void OnRealized(object sender, EventArgs e)
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
                GL.Enable(EnableCap.DepthTest);
                GL.Enable(EnableCap.Blend);
                GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                GL.PointSize(3.0f);
                GL.LineWidth(1.0f);

                // Ensure initial frame is drawn immediately
                this.QueueDraw();
            }
        }

        private void OnUnrealized(object sender, EventArgs e)
        {
            _loaded = false;
        }

        private void OnRender(object sender, RenderArgs args)
        {
            if (!_loaded) return;

            UpdateFPS();
            this.MakeCurrent();

            // Apply background color from settings (each frame so changes are reflected)
            var settings = AppSettings.Instance;
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

            _viewMatrix = Matrix4.CreateTranslation(_panX, _panY, _zoom) *
                          Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX)) *
                          Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));

            var finalView = coordTransform * _viewMatrix;
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref finalView);

            // Draw scene elements
            if (ShowGrid) DrawGrid();
            if (ShowAxes) DrawAxesEnhanced();

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
            // Hide transform gizmos when in Select Mode
            if (ShowGizmo && _sceneGraph != null && _sceneGraph.SelectedObjects.Count > 0 && _gizmoMode != GizmoMode.Select)
            {
                DrawGizmo();
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
            var s = AppSettings.Instance;

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

            var settings = AppSettings.Instance;

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

                    if (obj.Selected)
                    {
                        DrawSelectionOutline(obj);
                    }
                }

                GL.PopMatrix();
            }
        }

        private void DrawLegacyMeshes()
        {
            var settings = AppSettings.Instance;

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
            var settings = AppSettings.Instance;
            GL.PointSize(isSelected ? 5.0f : 3.0f);

            if (settings.PointCloudColor == PointCloudColorMode.DistanceMap)
            {
                DrawPointCloudDepthMap(mesh);
            }
            else
            {
                GL.Begin(PrimitiveType.Points);
                for (int i = 0; i < mesh.Vertices.Count; i++)
                {
                    var c = mesh.Colors[i];
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
            GL.PointSize(pc.PointSize);
            GL.Begin(PrimitiveType.Points);

            for (int i = 0; i < pc.Points.Count; i++)
            {
                var c = pc.Colors[i];
                GL.Color3(c.X, c.Y, c.Z);
                GL.Vertex3(pc.Points[i]);
            }

            GL.End();
        }

        private static Vector3 TurboColormap(float t)
        {
            t = Math.Max(0f, Math.Min(1f, t));
            float r, g, b;

            if (t < 0.25f)
            {
                float s = t / 0.25f;
                r = 0.0f; g = s; b = 1.0f;
            }
            else if (t < 0.5f)
            {
                float s = (t - 0.25f) / 0.25f;
                r = 0.0f; g = 1.0f; b = 1.0f - s;
            }
            else if (t < 0.75f)
            {
                float s = (t - 0.5f) / 0.25f;
                r = s; g = 1.0f; b = 0.0f;
            }
            else
            {
                float s = (t - 0.75f) / 0.25f;
                r = 1.0f; g = 1.0f - s; b = 0.0f;
            }

            return new Vector3(r, g, b);
        }

        private void DrawMesh(MeshData mesh, bool isSelected)
        {
            GL.Begin(PrimitiveType.Triangles);
            for (int i = 0; i < mesh.Indices.Count; i++)
            {
                int idx = mesh.Indices[i];
                if (idx < mesh.Vertices.Count)
                {
                    var c = mesh.Colors[idx];
                    if (isSelected)
                    {
                        // Add tint to selection if we want to visualize it on the mesh itself,
                        // but the bounding box is usually enough.
                        // Keeping existing behavior:
                        GL.Color3(Math.Min(1f, c.X + 0.2f), Math.Min(1f, c.Y + 0.2f), c.Z);
                    }
                    else
                    {
                        GL.Color3(c.X, c.Y, c.Z);
                    }
                    GL.Vertex3(mesh.Vertices[idx]);
                }
            }
            GL.End();
        }

        private void DrawSelectionOutline(SceneObject obj)
        {
            var (min, max) = (obj.BoundsMin, obj.BoundsMax);

            GL.LineWidth(2.0f);

            // Assign color based on Object ID
            var color = ColorPalette[obj.Id % ColorPalette.Length];
            GL.Color4(color.X, color.Y, color.Z, 0.8f);

            var mode = AppSettings.Instance.BoundingBoxStyle;

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
                // Pan with middle mouse or Shift+Left (Note: Shift+Left collides with multiple select, but panning usually requires drag)
                // We'll prioritize panning if Shift is held and mouse moves
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
                // Rotate view if not dragging a handle and not selecting
                // But in Select mode, left drag should probably still rotate view unless box selecting (which isn't implemented)

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
                case Gdk.Key.f:
                case Gdk.Key.F:
                    FocusOnSelection();
                    break;
                case Gdk.Key.Delete:
                    if (_sceneGraph != null)
                    {
                        foreach (var obj in _sceneGraph.SelectedObjects.ToList())
                        {
                            _sceneGraph.RemoveObject(obj);
                        }
                    }
                    break;
                case Gdk.Key.Escape:
                    _sceneGraph?.ClearSelection();
                    break;
            }
            this.QueueDraw();
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
