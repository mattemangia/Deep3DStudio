using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using Deep3DStudio.Scene;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;
using System.Linq;
using ErrorCode = OpenTK.Graphics.OpenGL.ErrorCode;

namespace Deep3DStudio.Viewport
{
    public enum GizmoMode
    {
        None,
        Translate,
        Rotate,
        Scale,
        Select,
        Pen,
        Rigging,
        LassoSelect,
        RectSelect,
        SplittingPlane
    }

    public class ThreeDView
    {
        private SceneGraph _sceneGraph;
        private float _zoom = -5.0f;
        private float _rotationX = 0f;
        private float _rotationY = 0f;
        private Vector3 _cameraTarget = Vector3.Zero;

        // Input state
        private bool _isDragging;
        private bool _isPanning;
        private Vector2 _lastMousePos;

        // Matrices
        private Matrix4 _viewMatrix;
        private Matrix4 _projectionMatrix;
        private Matrix4 _finalViewMatrix;

        // Gizmo State
        private GizmoMode _gizmoMode = GizmoMode.Select;
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

        // Mesh Editing Tool for Pen mode
        private MeshEditingTool _meshEditingTool = new MeshEditingTool();
        public MeshEditingTool MeshEditingTool => _meshEditingTool;

        // Splitting Plane Tool
        private SplittingPlaneTool? _splittingPlaneTool = null;
        public SplittingPlaneTool? GetSplittingPlaneTool() => _splittingPlaneTool;
        public event EventHandler? SplittingPlaneConfirmed;

        // Triangle selection events
        public event EventHandler? TriangleSelectionChanged;

        // Lasso/Rect selection state
        private List<Vector2> _lassoPoints = new List<Vector2>();
        private Vector2 _rectStart, _rectEnd;
        private bool _isDrawingSelection = false;
        private HashSet<(PointCloudObject pc, int pointIndex)> _selectedPoints = new HashSet<(PointCloudObject, int)>();
        public IReadOnlyCollection<(PointCloudObject pc, int pointIndex)> SelectedPoints => _selectedPoints;
        public event EventHandler? SelectionChanged;

        // Viewport dimensions for ray casting
        private int _viewportX, _viewportY, _viewportWidth, _viewportHeight;

        // Performance
        private int _frameCount = 0;
        private float _fps = 0;
        private DateTime _lastFpsUpdate = DateTime.Now;

        // Point cloud buffers (key = object Id)
        private Dictionary<int, (int vao, int vbo, int count, PointCloudColorMode colorMode)> _pointCloudBuffers
            = new Dictionary<int, (int, int, int, PointCloudColorMode)>();

        // Mesh buffers (key = object Id)
        private Dictionary<int, (int vao, int vbo, int ebo, int indexCount, bool hasTexture)> _meshBuffers
            = new Dictionary<int, (int, int, int, int, bool)>();

        private bool _hasPointCloudLegend = false;
        private PointCloudColorMode _legendMode = PointCloudColorMode.RGB;
        private float _legendMin = 0.0f;
        private float _legendMax = 1.0f;
        private bool _legendUsesDepthFallback = false;
        private bool _hasHoveredSourceDistance = false;
        private float _hoveredSourceDistance = 0.0f;
        private bool _hoveredDistanceIsMeters = false;
        private string _hoveredSourceCameraName = string.Empty;

        public float FPS => _fps;

        public bool TryGetPointCloudLegend(out PointCloudColorMode mode, out float minValue, out float maxValue, out bool depthFallback)
        {
            mode = _legendMode;
            minValue = _legendMin;
            maxValue = _legendMax;
            depthFallback = _legendUsesDepthFallback;
            return _hasPointCloudLegend;
        }

        public bool TryGetHoveredSourceCameraDistance(out float value, out bool isMeters, out string sourceCameraName)
        {
            value = _hoveredSourceDistance;
            isMeters = _hoveredDistanceIsMeters;
            sourceCameraName = _hoveredSourceCameraName;
            return _hasHoveredSourceDistance;
        }

        public ThreeDView(SceneGraph sceneGraph)
        {
            _sceneGraph = sceneGraph;
        }

        // Display Options
        public bool ShowGrid { get; set; } = true;
        public bool ShowAxes { get; set; } = true;
        public bool ShowGizmo { get; set; } = true;
        public bool ShowCameras { get; set; } = true;
        public bool ShowInfoText { get; set; } = true; // Kept for API compatibility, but rendering is moved to UI
        public float CameraFrustumScale { get; set; } = 0.3f;

        public GizmoMode CurrentGizmoMode
        {
            get => _gizmoMode;
            set
            {
                if (_gizmoMode == GizmoMode.SplittingPlane && value != GizmoMode.SplittingPlane)
                    _splittingPlaneTool = null;
                
                if (value == GizmoMode.SplittingPlane && _splittingPlaneTool == null)
                    _splittingPlaneTool = new SplittingPlaneTool();
                
                _gizmoMode = value;
            }
        }

        public void InitGL()
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.PointSize(5.0f);
            GL.LineWidth(1.0f);
        }

        public void Cleanup()
        {
            foreach (var kvp in _pointCloudBuffers)
            {
                if (kvp.Value.vao != 0) GL.DeleteVertexArray(kvp.Value.vao);
                if (kvp.Value.vbo != 0) GL.DeleteBuffer(kvp.Value.vbo);
            }
            _pointCloudBuffers.Clear();

            foreach (var kvp in _meshBuffers)
            {
                if (kvp.Value.vao != 0) GL.DeleteVertexArray(kvp.Value.vao);
                if (kvp.Value.vbo != 0) GL.DeleteBuffer(kvp.Value.vbo);
                if (kvp.Value.ebo != 0) GL.DeleteBuffer(kvp.Value.ebo);
            }
            _meshBuffers.Clear();
        }

        public void Render(int width, int height)
        {
            CheckError("Start Render");

            // Update FPS
            _frameCount++;
            var now = DateTime.Now;
            var elapsed = (now - _lastFpsUpdate).TotalSeconds;
            if (elapsed >= 1.0)
            {
                _fps = (float)(_frameCount / elapsed);
                _frameCount = 0;
                _lastFpsUpdate = now;
            }

            // Ensure fixed-function pipeline is active (disable any active shaders from ImGui)
            GL.UseProgram(0);

            // Explicitly unbind everything to ensure clean state for legacy drawing
            GL.BindVertexArray(0);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, 0);

            // Disable states that might have been left on by ImGui
            GL.Disable(EnableCap.Texture2D);
            GL.Disable(EnableCap.Lighting);
            GL.Disable(EnableCap.CullFace); 
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            // Disable vertex attributes that ImGui might have enabled on VAO 0
            for (int i = 0; i < 8; i++) GL.DisableVertexAttribArray(i);

            // Consume potential errors from state reset on strict contexts
            while (GL.GetError() != ErrorCode.NoError) { }

            if (height == 0) height = 1;

            // Setup Matrices
            _projectionMatrix = Matrix4.CreatePerspectiveFieldOfView(
                MathHelper.DegreesToRadians(45f), (float)width / height, 0.1f, 1000f);

            var s = IniSettings.Instance;
            Matrix4 coordTransform = Matrix4.Identity;
            if (s.CoordSystem == CoordinateSystem.RightHanded_Z_Up)
            {
                coordTransform = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(-90));
            }

            var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
            var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
            var rotation = ry * rx;

            _viewMatrix = Matrix4.CreateTranslation(-_cameraTarget) *
                          rotation *
                          Matrix4.CreateTranslation(0, 0, _zoom);

            _finalViewMatrix = coordTransform * _viewMatrix;

            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref _projectionMatrix);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref _finalViewMatrix);
            CheckError("Matrices");

            if (s.ShowGrid) DrawGrid();
            if (s.ShowAxes) DrawAxes();

            ResetPointCloudLegend();
            DrawScene();
            CheckError("DrawScene");

            // Draw gizmo - hide in Select, Pen, Rigging, Lasso, Rect modes
            if (s.ShowGizmo && _sceneGraph.SelectedObjects.Count > 0 &&
                _gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.None &&
                _gizmoMode != GizmoMode.Pen && _gizmoMode != GizmoMode.Rigging &&
                _gizmoMode != GizmoMode.LassoSelect && _gizmoMode != GizmoMode.RectSelect)
            {
                DrawGizmo();
            }

            // Draw selected triangles highlight (Pen mode or selection modes)
            if ((_gizmoMode == GizmoMode.Pen || _gizmoMode == GizmoMode.LassoSelect || _gizmoMode == GizmoMode.RectSelect)
                && _meshEditingTool.SelectedTriangles.Count > 0)
            {
                DrawSelectedTriangles();
            }

            // Draw selected points highlight
            if ((_gizmoMode == GizmoMode.LassoSelect || _gizmoMode == GizmoMode.RectSelect)
                && _selectedPoints.Count > 0)
            {
                DrawSelectedPoints();
            }

            // Draw splitting plane
            if (_gizmoMode == GizmoMode.SplittingPlane && _splittingPlaneTool != null)
            {
                _splittingPlaneTool.Render();
            }
        }

        // Error tracking to avoid spamming console
        private static DateTime _lastErrorLog = DateTime.MinValue;
        private static int _errorCount = 0;

        private void CheckError(string stage)
        {
            // Drain all errors from the queue
            ErrorCode err;
            while ((err = GL.GetError()) != ErrorCode.NoError)
            {
                // Skip InvalidFramebufferOperation and InvalidOperation caused by legacy/modern GL switching
                if (err == ErrorCode.InvalidFramebufferOperation || err == ErrorCode.InvalidOperation)
                    continue;

                // Rate limit error logging
                _errorCount++;
                if ((DateTime.Now - _lastErrorLog).TotalSeconds > 5)
                {
                    Console.WriteLine($"OpenGL Error at ThreeDView {stage}: {err} (count: {_errorCount})");
                    _lastErrorLog = DateTime.Now;
                    _errorCount = 0;
                }
            }
        }

        /// <summary>
        /// Clear OpenGL error queue without logging
        /// </summary>
        private static void DrainGLErrors()
        {
            while (GL.GetError() != ErrorCode.NoError) { }
        }

        public void UpdateInput(MouseState mouse, KeyboardState keyboard, float dt, int width, int height)
        {
             bool pointerInViewport =
                 mouse.X >= _viewportX && mouse.X < (_viewportX + _viewportWidth) &&
                 mouse.Y >= _viewportY && mouse.Y < (_viewportY + _viewportHeight);

             bool keepCurrentInteraction = _isDragging || _isDraggingGizmo || _isPanning;
             if (!pointerInViewport && !keepCurrentInteraction)
             {
                 ClearHoveredSourceDistance();
                 return;
             }

             if (pointerInViewport && _viewportWidth > 0 && _viewportHeight > 0)
             {
                 int localX = (int)mouse.X - _viewportX;
                 int localY = (int)mouse.Y - _viewportY;
                 UpdateHoveredSourceDistance(localX, localY, _viewportWidth, _viewportHeight);
             }

             // Gizmo Interaction or Camera Interaction
             if (mouse.IsButtonDown(MouseButton.Left))
             {
                 if (!_isDragging && !_isDraggingGizmo)
                 {
                     // Pen mode: triangle picking
                     if (_gizmoMode == GizmoMode.Pen)
                     {
                         HandlePenModeClick((int)mouse.X, (int)mouse.Y, keyboard);
                         _isDragging = true;
                         _lastMousePos = mouse.Position;
                         return;
                     }

                     // Lasso/Rect selection modes
                     if (_gizmoMode == GizmoMode.LassoSelect || _gizmoMode == GizmoMode.RectSelect)
                     {
                         int localX = (int)mouse.X - _viewportX;
                         int localY = (int)mouse.Y - _viewportY;
                         _isDrawingSelection = true;

                         if (_gizmoMode == GizmoMode.LassoSelect)
                         {
                             _lassoPoints.Clear();
                             _lassoPoints.Add(new Vector2(localX, localY));
                         }
                         else
                         {
                             _rectStart = new Vector2(localX, localY);
                             _rectEnd = _rectStart;
                         }

                         _isDragging = true;
                         _lastMousePos = mouse.Position;
                         return;
                     }

                     // Check Gizmo Intersection first (but not for Pen/Rigging modes)
                     if (_gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.None &&
                         _gizmoMode != GizmoMode.Pen && _gizmoMode != GizmoMode.Rigging &&
                         _sceneGraph.SelectedObjects.Count > 0)
                     {
                         int axis = CheckGizmoIntersection(mouse.X, mouse.Y, width, height);
                         if (axis != -1)
                         {
                             _activeGizmoAxis = axis;
                             _isDraggingGizmo = true;
                             _gizmoDragStart = new Vector3(mouse.X, mouse.Y, 0);
                             _gizmoStartStates.Clear();
                             foreach (var obj in _sceneGraph.SelectedObjects)
                             {
                                 _gizmoStartStates[obj.Id] = new GizmoStartState(obj.Position, obj.Rotation, obj.Scale);
                             }
                             _lastMousePos = mouse.Position;
                             return;
                         }
                     }

                     // If not gizmo, check picking
                     if (_gizmoMode == GizmoMode.Select)
                     {
                         var picked = PickObject((int)mouse.X, (int)mouse.Y, width, height);
                         bool multi = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.RightShift) ||
                                      keyboard.IsKeyDown(Keys.LeftControl) || keyboard.IsKeyDown(Keys.RightControl);

                         if (picked != null)
                         {
                             if (multi)
                             {
                                 if (picked.Selected) _sceneGraph.Deselect(picked);
                                 else _sceneGraph.Select(picked, true);
                             }
                             else
                             {
                                 _sceneGraph.Select(picked, false);
                             }
                         }
                         else if (!multi)
                         {
                             _sceneGraph.ClearSelection();
                         }
                     }

                     _isDragging = true;
                     _lastMousePos = mouse.Position;
                 }

                 if (_isDraggingGizmo)
                 {
                     HandleGizmoDrag(mouse.X, mouse.Y);
                 }
                 else if (_isDragging && _isDrawingSelection)
                 {
                     int localX = (int)mouse.X - _viewportX;
                     int localY = (int)mouse.Y - _viewportY;

                     if (_gizmoMode == GizmoMode.LassoSelect)
                     {
                         var newPoint = new Vector2(localX, localY);
                         if (_lassoPoints.Count == 0 || (newPoint - _lassoPoints[^1]).Length > 3f)
                             _lassoPoints.Add(newPoint);
                     }
                     else if (_gizmoMode == GizmoMode.RectSelect)
                     {
                         _rectEnd = new Vector2(localX, localY);
                     }
                 }
                 else if (_isDragging && _gizmoMode != GizmoMode.Pen)
                 {
                     var delta = mouse.Position - _lastMousePos;
                     _rotationY += delta.X * 0.5f;
                     _rotationX += delta.Y * 0.5f;
                 }

                 _lastMousePos = mouse.Position;
             }
             else
             {
                 if (_isDrawingSelection)
                 {
                     bool addToSelection = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.RightShift) ||
                                           keyboard.IsKeyDown(Keys.LeftControl) || keyboard.IsKeyDown(Keys.RightControl);
                     if (_gizmoMode == GizmoMode.LassoSelect && _lassoPoints.Count >= 3)
                     {
                         SelectGeometryInLasso(addToSelection);
                     }
                     else if (_gizmoMode == GizmoMode.RectSelect)
                     {
                         SelectGeometryInRect(addToSelection);
                     }
                     _isDrawingSelection = false;
                 }
                 _isDragging = false;
                 _isDraggingGizmo = false;
                 _gizmoStartStates.Clear();
                 _activeGizmoAxis = -1;
             }

             bool shiftDown = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.RightShift);
             if (mouse.IsButtonDown(MouseButton.Middle) || (mouse.IsButtonDown(MouseButton.Left) && shiftDown))
             {
                 if (!_isPanning)
                 {
                     _isPanning = true;
                     _lastMousePos = mouse.Position;
                 }

                 var delta = mouse.Position - _lastMousePos;
                 float panSpeed = 0.005f * Math.Abs(_zoom);

                  var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
                 var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
                 var rotation = ry * rx;

                 Vector3 right = new Vector3(rotation.M11, rotation.M21, rotation.M31);
                 Vector3 up = new Vector3(rotation.M12, rotation.M22, rotation.M32);

                 _cameraTarget -= right * delta.X * panSpeed;
                 _cameraTarget += up * delta.Y * panSpeed;

                 _lastMousePos = mouse.Position;
             }
             else
             {
                 _isPanning = false;
             }
        }

        /// <summary>
        /// Handle click in Pen mode for triangle selection
        /// </summary>
        private void HandlePenModeClick(int mouseX, int mouseY, KeyboardState keyboard)
        {
            if (_sceneGraph == null) return;

            // Adjust mouse position relative to viewport
            int localX = mouseX - _viewportX;
            int localY = mouseY - _viewportY;

            // Get ray from mouse position
            var (rayOrigin, rayDir) = GetRayFromScreenPoint(localX, localY);

            // Get all mesh objects from scene
            var meshObjects = _sceneGraph.GetVisibleObjects()
                .OfType<MeshObject>()
                .ToList();

            // Perform triangle picking
            var (pickedMesh, triangleIndex, distance) = _meshEditingTool.PickTriangle(rayOrigin, rayDir, meshObjects);

            bool addToSelection = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.RightShift) ||
                                  keyboard.IsKeyDown(Keys.LeftControl) || keyboard.IsKeyDown(Keys.RightControl);

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
        }

        /// <summary>
        /// Generate a ray from camera through screen point for picking
        /// </summary>
        private (Vector3 origin, Vector3 direction) GetRayFromScreenPoint(int screenX, int screenY)
        {
            int width = _viewportWidth;
            int height = _viewportHeight;

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
            Matrix4 invView = _finalViewMatrix.Inverted();
            Matrix4 invViewProj = invView * invProjection;

            // Transform near point
            Vector4 worldNear = nearPoint * invViewProj;
            if (Math.Abs(worldNear.W) > 0.0001f)
                worldNear /= worldNear.W;

            // Transform far point
            Vector4 worldFar = farPoint * invViewProj;
            if (Math.Abs(worldFar.W) > 0.0001f)
                worldFar /= worldFar.W;

            Vector3 origin = new Vector3(worldNear.X, worldNear.Y, worldNear.Z);
            Vector3 direction = new Vector3(worldFar.X - worldNear.X, worldFar.Y - worldNear.Y, worldFar.Z - worldNear.Z);

            return (origin, direction.Normalized());
        }

        /// <summary>
        /// Draw highlight overlay for selected triangles in Pen mode
        /// </summary>
        private void DrawSelectedTriangles()
        {
            var vertices = _meshEditingTool.GetSelectedTriangleVertices();
            if (vertices.Count == 0) return;

            GL.Disable(EnableCap.DepthTest);

            // Draw filled triangles with transparency
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            GL.Begin(PrimitiveType.Triangles);
            GL.Color4(1.0f, 0.3f, 0.3f, 0.4f); // Semi-transparent red

            for (int i = 0; i < vertices.Count; i++)
            {
                GL.Vertex3(vertices[i]);
            }
            GL.End();

            // Draw wireframe edges
            GL.LineWidth(2.0f);
            GL.Begin(PrimitiveType.Lines);
            GL.Color4(1.0f, 1.0f, 0.0f, 1.0f); // Solid yellow edges

            for (int i = 0; i < vertices.Count; i += 3)
            {
                if (i + 2 < vertices.Count)
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
            }
            GL.End();

            GL.LineWidth(1.0f);
            GL.Disable(EnableCap.Blend);
            GL.Enable(EnableCap.DepthTest);
        }

        /// <summary>
        /// Draw highlight for selected points in point clouds
        /// </summary>
        private void DrawSelectedPoints()
        {
            if (_selectedPoints.Count == 0) return;

            GL.Disable(EnableCap.DepthTest);
            GL.PointSize(8.0f);
            GL.Begin(PrimitiveType.Points);
            GL.Color4(0.0f, 1.0f, 0.3f, 1.0f); // Bright green

            foreach (var (pc, idx) in _selectedPoints)
            {
                if (idx >= 0 && idx < pc.Points.Count)
                {
                    var worldPos = Vector3.TransformPosition(pc.Points[idx], pc.GetWorldTransform());
                    GL.Vertex3(worldPos);
                }
            }

            GL.End();
            GL.PointSize(1.0f);
            GL.Enable(EnableCap.DepthTest);
        }

        /// <summary>
        /// Project world position to viewport screen coordinates
        /// </summary>
        private Vector2 ProjectToScreen(Vector3 worldPos)
        {
            Matrix4 viewProj = _finalViewMatrix * _projectionMatrix;
            Vector4 clip = new Vector4(worldPos, 1.0f) * viewProj;

            if (Math.Abs(clip.W) < 0.0001f)
                return new Vector2(-10000, -10000);

            float ndcX = clip.X / clip.W;
            float ndcY = clip.Y / clip.W;

            float screenX = (ndcX + 1.0f) * 0.5f * _viewportWidth;
            float screenY = (1.0f - ndcY) * 0.5f * _viewportHeight;

            return new Vector2(screenX, screenY);
        }

        /// <summary>
        /// Test if a 2D point is inside a polygon (lasso) using ray casting
        /// </summary>
        private static bool IsPointInLasso(Vector2 point, List<Vector2> lasso)
        {
            int count = lasso.Count;
            if (count < 3) return false;

            bool inside = false;
            for (int i = 0, j = count - 1; i < count; j = i++)
            {
                if ((lasso[i].Y > point.Y) != (lasso[j].Y > point.Y) &&
                    point.X < (lasso[j].X - lasso[i].X) * (point.Y - lasso[i].Y) / (lasso[j].Y - lasso[i].Y) + lasso[i].X)
                {
                    inside = !inside;
                }
            }
            return inside;
        }

        /// <summary>
        /// Test if a 2D point is inside a rectangle
        /// </summary>
        private static bool IsPointInRect(Vector2 point, Vector2 rectMin, Vector2 rectMax)
        {
            return point.X >= rectMin.X && point.X <= rectMax.X &&
                   point.Y >= rectMin.Y && point.Y <= rectMax.Y;
        }

        /// <summary>
        /// Select geometry (triangles + points) that falls within the lasso region
        /// </summary>
        private void SelectGeometryInLasso(bool addToSelection)
        {
            if (!addToSelection)
            {
                _meshEditingTool.ClearSelection();
                _selectedPoints.Clear();
            }

            SelectGeometryInRegion(pt => IsPointInLasso(pt, _lassoPoints));
        }

        /// <summary>
        /// Select geometry (triangles + points) that falls within the rectangle region
        /// </summary>
        private void SelectGeometryInRect(bool addToSelection)
        {
            if (!addToSelection)
            {
                _meshEditingTool.ClearSelection();
                _selectedPoints.Clear();
            }

            var rectMin = new Vector2(Math.Min(_rectStart.X, _rectEnd.X), Math.Min(_rectStart.Y, _rectEnd.Y));
            var rectMax = new Vector2(Math.Max(_rectStart.X, _rectEnd.X), Math.Max(_rectStart.Y, _rectEnd.Y));

            // Minimum size to avoid accidental clicks
            if (Math.Abs(rectMax.X - rectMin.X) < 5 && Math.Abs(rectMax.Y - rectMin.Y) < 5)
                return;

            SelectGeometryInRegion(pt => IsPointInRect(pt, rectMin, rectMax));
        }

        /// <summary>
        /// Core selection logic: tests projected geometry against a 2D region predicate
        /// </summary>
        private void SelectGeometryInRegion(Func<Vector2, bool> isInRegion)
        {
            if (_sceneGraph == null) return;

            var visibleObjects = _sceneGraph.GetVisibleObjects();

            // Select triangles from meshes
            foreach (var obj in visibleObjects.OfType<MeshObject>())
            {
                if (obj.MeshData == null) continue;
                var meshData = obj.MeshData;
                var worldTransform = obj.GetWorldTransform();
                int triCount = meshData.Indices.Count / 3;

                for (int t = 0; t < triCount; t++)
                {
                    int i0 = meshData.Indices[t * 3];
                    int i1 = meshData.Indices[t * 3 + 1];
                    int i2 = meshData.Indices[t * 3 + 2];

                    if (i0 >= meshData.Vertices.Count || i1 >= meshData.Vertices.Count || i2 >= meshData.Vertices.Count)
                        continue;

                    var s0 = ProjectToScreen(Vector3.TransformPosition(meshData.Vertices[i0], worldTransform));
                    var s1 = ProjectToScreen(Vector3.TransformPosition(meshData.Vertices[i1], worldTransform));
                    var s2 = ProjectToScreen(Vector3.TransformPosition(meshData.Vertices[i2], worldTransform));

                    // Select if all 3 vertices are in the region
                    if (isInRegion(s0) && isInRegion(s1) && isInRegion(s2))
                    {
                        _meshEditingTool.SelectTriangle(obj, t, true);
                    }
                }
            }

            // Select points from point clouds
            foreach (var pc in visibleObjects.OfType<PointCloudObject>())
            {
                var worldTransform = pc.GetWorldTransform();
                int visibleCount = pc.VisiblePointCount;

                for (int i = 0; i < visibleCount; i++)
                {
                    var screenPos = ProjectToScreen(Vector3.TransformPosition(pc.Points[i], worldTransform));
                    if (isInRegion(screenPos))
                    {
                        _selectedPoints.Add((pc, i));
                    }
                }
            }

            TriangleSelectionChanged?.Invoke(this, EventArgs.Empty);
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Get the current lasso points for rendering overlay
        /// </summary>
        public IReadOnlyList<Vector2> GetLassoPoints() => _lassoPoints;

        /// <summary>
        /// Get the current rectangle selection for rendering overlay
        /// </summary>
        public (Vector2 start, Vector2 end) GetRectSelection() => (_rectStart, _rectEnd);

        /// <summary>
        /// Whether a selection drag is in progress
        /// </summary>
        public bool IsDrawingSelection => _isDrawingSelection;

        /// <summary>
        /// Clear all point selections
        /// </summary>
        public void ClearPointSelection()
        {
            _selectedPoints.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Delete selected points from their point clouds
        /// </summary>
        public int DeleteSelectedPoints()
        {
            if (_selectedPoints.Count == 0) return 0;

            // Group by point cloud
            var byCloud = _selectedPoints.GroupBy(x => x.pc).ToList();
            int totalRemoved = 0;

            foreach (var group in byCloud)
            {
                var pc = group.Key;
                var indicesToRemove = group.Select(x => x.pointIndex).OrderByDescending(i => i).Distinct().ToList();

                foreach (int idx in indicesToRemove)
                {
                    if (idx < pc.Points.Count)
                    {
                        pc.Points.RemoveAt(idx);
                        if (idx < pc.Colors.Count) pc.Colors.RemoveAt(idx);
                        if (idx < pc.Normals.Count) pc.Normals.RemoveAt(idx);
                        if (idx < pc.Confidence.Count) pc.Confidence.RemoveAt(idx);
                        totalRemoved++;
                    }
                }

                pc.UpdateBounds();
            }

            _selectedPoints.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
            return totalRemoved;
        }

        /// <summary>
        /// Copy selected points into a new PointCloudObject
        /// </summary>
        public PointCloudObject? CopySelectedPoints()
        {
            if (_selectedPoints.Count == 0) return null;

            var result = new PointCloudObject("Selection Copy");

            foreach (var (pc, idx) in _selectedPoints)
            {
                if (idx >= 0 && idx < pc.Points.Count)
                {
                    var worldTransform = pc.GetWorldTransform();
                    result.Points.Add(Vector3.TransformPosition(pc.Points[idx], worldTransform));
                    if (idx < pc.Colors.Count) result.Colors.Add(pc.Colors[idx]);
                    if (idx < pc.Normals.Count) result.Normals.Add(pc.Normals[idx]);
                    if (idx < pc.Confidence.Count) result.Confidence.Add(pc.Confidence[idx]);
                }
            }

            result.UpdateBounds();
            return result;
        }

        /// <summary>
        /// Grow point selection by adding nearby points within a radius
        /// </summary>
        public void GrowPointSelection(float radiusFactor = 2.0f)
        {
            if (_selectedPoints.Count == 0) return;

            var newSelections = new HashSet<(PointCloudObject, int)>();

            var byCloud = _selectedPoints.GroupBy(x => x.pc).ToList();
            foreach (var group in byCloud)
            {
                var pc = group.Key;
                var selectedIndices = new HashSet<int>(group.Select(x => x.pointIndex));
                var worldTransform = pc.GetWorldTransform();

                // Estimate average spacing from selected points
                float avgSpacing = 0;
                int count = 0;
                var selList = selectedIndices.Take(100).ToList();
                foreach (int idx in selList)
                {
                    if (idx >= pc.Points.Count) continue;
                    var p = pc.Points[idx];
                    float minDist = float.MaxValue;
                    for (int j = Math.Max(0, idx - 50); j < Math.Min(pc.Points.Count, idx + 50); j++)
                    {
                        if (j == idx) continue;
                        float d = (pc.Points[j] - p).LengthSquared;
                        if (d < minDist) minDist = d;
                    }
                    if (minDist < float.MaxValue) { avgSpacing += MathF.Sqrt(minDist); count++; }
                }
                float radius = count > 0 ? (avgSpacing / count) * radiusFactor : 0.01f;
                float radiusSq = radius * radius;

                // Find neighbors
                int visibleCount = pc.VisiblePointCount;
                foreach (int selIdx in selectedIndices)
                {
                    if (selIdx >= pc.Points.Count) continue;
                    var selPoint = pc.Points[selIdx];

                    for (int i = 0; i < visibleCount; i++)
                    {
                        if (selectedIndices.Contains(i)) continue;
                        if ((pc.Points[i] - selPoint).LengthSquared <= radiusSq)
                        {
                            newSelections.Add((pc, i));
                        }
                    }
                }
            }

            foreach (var sel in newSelections)
                _selectedPoints.Add(sel);

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Shrink point selection by removing boundary points
        /// </summary>
        public void ShrinkPointSelection(float radiusFactor = 2.0f)
        {
            if (_selectedPoints.Count == 0) return;

            var toRemove = new HashSet<(PointCloudObject, int)>();
            var byCloud = _selectedPoints.GroupBy(x => x.pc).ToList();

            foreach (var group in byCloud)
            {
                var pc = group.Key;
                var selectedIndices = new HashSet<int>(group.Select(x => x.pointIndex));

                // Estimate average spacing
                float avgSpacing = 0;
                int count = 0;
                var selList = selectedIndices.Take(100).ToList();
                foreach (int idx in selList)
                {
                    if (idx >= pc.Points.Count) continue;
                    var p = pc.Points[idx];
                    float minDist = float.MaxValue;
                    for (int j = Math.Max(0, idx - 50); j < Math.Min(pc.Points.Count, idx + 50); j++)
                    {
                        if (j == idx) continue;
                        float d = (pc.Points[j] - p).LengthSquared;
                        if (d < minDist) minDist = d;
                    }
                    if (minDist < float.MaxValue) { avgSpacing += MathF.Sqrt(minDist); count++; }
                }
                float radius = count > 0 ? (avgSpacing / count) * radiusFactor : 0.01f;
                float radiusSq = radius * radius;

                // Remove boundary points (those with unselected neighbors within radius)
                foreach (int selIdx in selectedIndices)
                {
                    if (selIdx >= pc.Points.Count) continue;
                    var selPoint = pc.Points[selIdx];
                    int visibleCount = pc.VisiblePointCount;

                    for (int i = 0; i < visibleCount; i++)
                    {
                        if (selectedIndices.Contains(i)) continue;
                        if ((pc.Points[i] - selPoint).LengthSquared <= radiusSq)
                        {
                            toRemove.Add((pc, selIdx));
                            break;
                        }
                    }
                }
            }

            foreach (var sel in toRemove)
                _selectedPoints.Remove(sel);

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        public void OnMouseWheel(float offset)
        {
            _zoom += offset;
            // Clamp zoom
            _zoom = Math.Clamp(_zoom, -100.0f, -0.5f);
        }

        /// <summary>
        /// Render with specified viewport region
        /// </summary>
        public void Render(
            int vpX, int vpY, int vpW, int vpH,
            int fbVpX, int fbVpY, int fbVpW, int fbVpH,
            int framebufferWidth, int framebufferHeight)
        {
            // Store logical viewport dimensions for ray casting / picking.
            _viewportX = vpX;
            _viewportY = vpY;
            _viewportWidth = Math.Max(1, vpW);
            _viewportHeight = Math.Max(1, vpH);

            int safeFbVpX = Math.Max(0, fbVpX);
            int safeFbVpY = Math.Max(0, fbVpY);
            int safeFbVpW = Math.Max(1, fbVpW);
            int safeFbVpH = Math.Max(1, fbVpH);
            int safeFbWindowWidth = Math.Max(1, framebufferWidth);
            int safeFbWindowHeight = Math.Max(1, framebufferHeight);

            // Set viewport for the 3D view area
            GL.Viewport(safeFbVpX, safeFbWindowHeight - safeFbVpY - safeFbVpH, safeFbVpW, safeFbVpH);
            GL.Scissor(safeFbVpX, safeFbWindowHeight - safeFbVpY - safeFbVpH, safeFbVpW, safeFbVpH);
            GL.Enable(EnableCap.ScissorTest);

            // Clear just this region using background color from settings
            var settings = IniSettings.Instance;
            GL.ClearColor(settings.ViewportBgR, settings.ViewportBgG, settings.ViewportBgB, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            Render(safeFbVpW, safeFbVpH);

            GL.Disable(EnableCap.ScissorTest);

            // Reset viewport to full window for ImGui
            GL.Viewport(0, 0, safeFbWindowWidth, safeFbWindowHeight);
        }

        /// <summary>
        /// Focus camera on the currently selected objects
        /// </summary>
        public void FocusOnSelection()
        {
            if (_sceneGraph == null || _sceneGraph.SelectedObjects.Count == 0) return;

            Vector3 center = Vector3.Zero;
            Vector3 boundsMin = new Vector3(float.MaxValue);
            Vector3 boundsMax = new Vector3(float.MinValue);

            foreach (var obj in _sceneGraph.SelectedObjects)
            {
                var (min, max) = obj.GetWorldBounds();
                boundsMin = Vector3.ComponentMin(boundsMin, min);
                boundsMax = Vector3.ComponentMax(boundsMax, max);
            }

            center = (boundsMin + boundsMax) * 0.5f;
            float size = (boundsMax - boundsMin).Length;

            _cameraTarget = center;
            _zoom = Math.Max(-size * 2.5f, -50f);
        }

        /// <summary>
        /// Focus camera on a specific object
        /// </summary>
        public void FocusOnObject(SceneObject obj)
        {
            if (obj == null) return;

            var (boundsMin, boundsMax) = obj.GetWorldBounds();
            var center = (boundsMin + boundsMax) * 0.5f;
            float size = (boundsMax - boundsMin).Length;

            _cameraTarget = center;
            _zoom = Math.Max(-size * 2.5f, -50f);
        }

        /// <summary>
        /// Reset camera to default view
        /// </summary>
        public void ResetCamera()
        {
            _zoom = -5.0f;
            _rotationX = 25f;
            _rotationY = -45f;
            _cameraTarget = Vector3.Zero;
        }

        private void DrawGrid()
        {
            GL.Begin(PrimitiveType.Lines);
            var s = IniSettings.Instance;
            GL.Color4(s.GridColorR, s.GridColorG, s.GridColorB, 0.5f);
            int size = 10;
            for (int i = -size; i <= size; i++)
            {
                GL.Vertex3(i, 0, -size); GL.Vertex3(i, 0, size);
                GL.Vertex3(-size, 0, i); GL.Vertex3(size, 0, i);
            }
            GL.End();
        }

        private void DrawAxes()
        {
            GL.Begin(PrimitiveType.Lines);
            GL.LineWidth(2.0f);
            GL.Color3(1.0f, 0.0f, 0.0f); GL.Vertex3(0, 0, 0); GL.Vertex3(1, 0, 0);
            GL.Color3(0.0f, 1.0f, 0.0f); GL.Vertex3(0, 0, 0); GL.Vertex3(0, 1, 0);
            GL.Color3(0.0f, 0.0f, 1.0f); GL.Vertex3(0, 0, 0); GL.Vertex3(0, 0, 1);
            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawScene()
        {
            if (_sceneGraph == null) return;
            var s = IniSettings.Instance;

            // Draw Cameras
            if (s.ShowCameras)
            {
                foreach(var cam in _sceneGraph.GetObjectsOfType<CameraObject>())
                {
                    if (cam.Visible && cam.ShowFrustum) DrawCameraFrustum(cam);
                }
            }

            foreach(var obj in _sceneGraph.GetVisibleObjects())
            {
                if (obj is CameraObject) continue; // Already drawn

                if (obj is SkeletonObject skelObj)
                {
                    if (skelObj.RenderMode == ObjectRenderMode.BoundingBoxOnly)
                    {
                        GL.PushMatrix();
                        var skelTransform = skelObj.GetWorldTransform();
                        GL.MultMatrix(ref skelTransform);
                        DrawBoundingBox(skelObj, skelObj.Selected);
                        GL.PopMatrix();
                    }
                    else
                    {
                        DrawSkeleton(skelObj);
                    }
                    continue;
                }

                GL.PushMatrix();
                var t = obj.GetWorldTransform();
                GL.MultMatrix(ref t);

                if (obj is PointCloudObject pc)
                {
                    bool bboxOnly = pc.RenderMode == ObjectRenderMode.BoundingBoxOnly;
                    if (!bboxOnly && s.ShowPointCloud)
                    {
                        DrawPointCloud(pc);
                    }
                    if (bboxOnly || obj.Selected)
                    {
                        DrawBoundingBox(obj, obj.Selected);
                    }
                }

                if (obj is MeshObject mesh)
                {
                    bool isSelected = obj.Selected;
                    bool bboxOnly = mesh.RenderMode == ObjectRenderMode.BoundingBoxOnly;

                    if (!bboxOnly && ShouldDrawMeshAsPointCloud(mesh, s) && mesh.MeshData != null && mesh.MeshData.Vertices.Count > 0)
                    {
                        DrawPointCloud(mesh.MeshData.Vertices, mesh.MeshData.Colors, mesh.MeshData.Confidence, mesh.PointSize);
                    }

                    if (ResolveWireframe(mesh, s)) GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    else GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (!bboxOnly && ShouldDrawMeshSurface(mesh, s) && mesh.MeshData != null)
                    {
                        DrawMeshVBO(mesh, isSelected, ResolveTexture(mesh, s));
                    }
                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (bboxOnly || isSelected)
                    {
                        DrawBoundingBox(obj, isSelected);
                    }
                }

                GL.PopMatrix();
            }
        }

        private void DrawMeshVBO(MeshObject mesh, bool isSelected, bool allowTexture)
        {
            if (mesh.MeshData == null || mesh.MeshData.Vertices.Count == 0) return;

            bool useTexture = allowTexture && mesh.MeshData.Texture != null;
            if (useTexture && mesh.MeshData.TextureId == -1) UploadTexture(mesh.MeshData);

            // Rebuild buffer if count changed or texture mode toggled (UVs needed vs Colors needed)
            if (!_meshBuffers.TryGetValue(mesh.Id, out var buffers)
                || buffers.indexCount != mesh.MeshData.Indices.Count
                || buffers.hasTexture != useTexture)
            {
                if (buffers.vao != 0) GL.DeleteVertexArray(buffers.vao);
                if (buffers.vbo != 0) GL.DeleteBuffer(buffers.vbo);
                if (buffers.ebo != 0) GL.DeleteBuffer(buffers.ebo);

                // Interleaved: Pos(3) + Color(3) + UV(2) = 8 floats per vertex
                float[] data = new float[mesh.MeshData.Vertices.Count * 8];
                bool hasColors = mesh.MeshData.Colors.Count >= mesh.MeshData.Vertices.Count;
                bool hasUVs = mesh.MeshData.UVs.Count >= mesh.MeshData.Vertices.Count;

                for (int i = 0; i < mesh.MeshData.Vertices.Count; i++)
                {
                    var p = mesh.MeshData.Vertices[i];
                    data[i * 8 + 0] = p.X;
                    data[i * 8 + 1] = p.Y;
                    data[i * 8 + 2] = p.Z;

                    if (hasColors)
                    {
                        var c = mesh.MeshData.Colors[i];
                        data[i * 8 + 3] = c.X;
                        data[i * 8 + 4] = c.Y;
                        data[i * 8 + 5] = c.Z;
                    }
                    else
                    {
                        data[i * 8 + 3] = 0.7f;
                        data[i * 8 + 4] = 0.7f;
                        data[i * 8 + 5] = 0.7f;
                    }

                    if (hasUVs)
                    {
                        var uv = mesh.MeshData.UVs[i];
                        data[i * 8 + 6] = uv.X;
                        data[i * 8 + 7] = uv.Y;
                    }
                    else
                    {
                        data[i * 8 + 6] = 0;
                        data[i * 8 + 7] = 0;
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

                // Attributes
                GL.EnableClientState(ArrayCap.VertexArray);
                GL.VertexPointer(3, VertexPointerType.Float, 8 * sizeof(float), 0);

                GL.EnableClientState(ArrayCap.ColorArray);
                GL.ColorPointer(3, ColorPointerType.Float, 8 * sizeof(float), 3 * sizeof(float));

                GL.EnableClientState(ArrayCap.TextureCoordArray);
                GL.TexCoordPointer(2, TexCoordPointerType.Float, 8 * sizeof(float), 6 * sizeof(float));

                GL.BindVertexArray(0);
                _meshBuffers[mesh.Id] = (vao, vbo, ebo, mesh.MeshData.Indices.Count, useTexture);
                buffers = (vao, vbo, ebo, mesh.MeshData.Indices.Count, useTexture);
            }

            if (useTexture && mesh.MeshData.TextureId != -1)
            {
                GL.Enable(EnableCap.Texture2D);
                GL.BindTexture(TextureTarget.Texture2D, mesh.MeshData.TextureId);
                GL.Color3(1.0f, 1.0f, 1.0f);
            }

            // Apply selection tint if selected (using fixed-function color tinting)
            if (isSelected && !useTexture)
            {
                GL.Color3(1.0f, 0.9f, 0.7f); // Warm tint for selection
            }

            GL.BindVertexArray(buffers.vao);

            // Re-setup pointers every frame because some drivers (and 3.3 Compat profiles) 
            // don't reliably capture legacy client-side pointers in VAOs.
            GL.BindBuffer(BufferTarget.ArrayBuffer, buffers.vbo);
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.VertexPointer(3, VertexPointerType.Float, 8 * sizeof(float), 0);

            GL.EnableClientState(ArrayCap.ColorArray);
            GL.ColorPointer(3, ColorPointerType.Float, 8 * sizeof(float), 3 * sizeof(float));

            if (buffers.hasTexture)
            {
                GL.EnableClientState(ArrayCap.TextureCoordArray);
                GL.TexCoordPointer(2, TexCoordPointerType.Float, 8 * sizeof(float), 6 * sizeof(float));
            }
            else
            {
                GL.DisableClientState(ArrayCap.TextureCoordArray);
            }

            GL.DrawElements(PrimitiveType.Triangles, buffers.indexCount, DrawElementsType.UnsignedInt, 0);
            GL.BindVertexArray(0);

            if (useTexture)
            {
                GL.BindTexture(TextureTarget.Texture2D, 0);
                GL.Disable(EnableCap.Texture2D);
            }
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

        private void ResetPointCloudLegend()
        {
            _hasPointCloudLegend = false;
            _legendMode = PointCloudColorMode.RGB;
            _legendMin = 0.0f;
            _legendMax = 1.0f;
            _legendUsesDepthFallback = false;
        }

        private void UpdatePointCloudLegend(PointCloudColorMode mode, float minValue, float maxValue, bool depthFallback)
        {
            _hasPointCloudLegend = true;
            _legendMode = mode;
            _legendMin = minValue;
            _legendMax = maxValue;
            _legendUsesDepthFallback = depthFallback;
        }

        private static bool TryComputeValueRange(
            IReadOnlyList<Vector3> points,
            IReadOnlyList<float> confidence,
            PointCloudColorMode mode,
            Func<int, int> indexAt,
            int count,
            out float minValue,
            out float maxValue,
            out bool depthFallback)
        {
            minValue = float.MaxValue;
            maxValue = float.MinValue;
            depthFallback = false;

            bool useDepth = mode == PointCloudColorMode.DistanceMap;
            bool useConfidence = mode == PointCloudColorMode.Confidence && confidence.Count >= points.Count;
            if (mode == PointCloudColorMode.Confidence && !useConfidence)
            {
                useDepth = true;
                depthFallback = true;
            }

            for (int i = 0; i < count; i++)
            {
                int source = indexAt(i);
                if (source < 0 || source >= points.Count)
                    continue;

                float value = useDepth ? points[source].Length : confidence[source];
                if (float.IsNaN(value) || float.IsInfinity(value))
                    continue;

                if (value < minValue) minValue = value;
                if (value > maxValue) maxValue = value;
            }

            if (minValue == float.MaxValue || maxValue == float.MinValue)
            {
                minValue = 0.0f;
                maxValue = 1.0f;
                return false;
            }

            return true;
        }

        private static Vector3 GetPointColor(
            int sourceIndex,
            IReadOnlyList<Vector3> points,
            IReadOnlyList<Vector3> colors,
            IReadOnlyList<float> confidence,
            PointCloudColorMode mode,
            float minValue,
            float range,
            bool depthFallback)
        {
            if (mode == PointCloudColorMode.DistanceMap || depthFallback)
            {
                float t = (points[sourceIndex].Length - minValue) / range;
                var (r, g, b) = ImageUtils.TurboColormap(t);
                return new Vector3(r, g, b);
            }

            if (mode == PointCloudColorMode.Confidence)
            {
                float value = sourceIndex < confidence.Count ? confidence[sourceIndex] : 0.0f;
                float t = (value - minValue) / range;
                var (r, g, b) = ImageUtils.TurboColormap(t);
                return new Vector3(r, g, b);
            }

            if (sourceIndex < colors.Count)
                return colors[sourceIndex];
            return new Vector3(1.0f, 1.0f, 1.0f);
        }

        private void DrawPointCloud(
            IReadOnlyList<Vector3> points,
            IReadOnlyList<Vector3> colors,
            IReadOnlyList<float> confidence,
            float pointSize)
        {
            if (points.Count == 0) return;

            var colorMode = IniSettings.Instance.PointCloudColor;
            bool useLegend = colorMode == PointCloudColorMode.DistanceMap || colorMode == PointCloudColorMode.Confidence;
            float minValue = 0.0f;
            float range = 1.0f;
            bool depthFallback = false;

            if (useLegend && TryComputeValueRange(points, confidence, colorMode, i => i, points.Count, out float min, out float max, out depthFallback))
            {
                minValue = min;
                range = max - min;
                if (range < 0.0001f) range = 1.0f;
                UpdatePointCloudLegend(colorMode, min, max, depthFallback);
            }

            // Simple ID for non-scene-object clouds (use negative hash to avoid collision)
            int cloudId = -Math.Abs(points.GetHashCode());

            if (!_pointCloudBuffers.TryGetValue(cloudId, out var buffers)
                || buffers.count != points.Count
                || buffers.colorMode != colorMode)
            {
                if (buffers.vao != 0) GL.DeleteVertexArray(buffers.vao);
                if (buffers.vbo != 0) GL.DeleteBuffer(buffers.vbo);

                float[] data = new float[points.Count * 6];
                for (int i = 0; i < points.Count; i++)
                {
                    var c = GetPointColor(i, points, colors, confidence, colorMode, minValue, range, depthFallback);
                    data[i * 6 + 0] = points[i].X;
                    data[i * 6 + 1] = points[i].Y;
                    data[i * 6 + 2] = points[i].Z;
                    data[i * 6 + 3] = c.X;
                    data[i * 6 + 4] = c.Y;
                    data[i * 6 + 5] = c.Z;
                }

                int vao = GL.GenVertexArray();
                int vbo = GL.GenBuffer();
                GL.BindVertexArray(vao);
                GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
                GL.BufferData(BufferTarget.ArrayBuffer, data.Length * sizeof(float), data, BufferUsageHint.StaticDraw);

                GL.EnableClientState(ArrayCap.VertexArray);
                GL.VertexPointer(3, VertexPointerType.Float, 6 * sizeof(float), 0);
                GL.EnableClientState(ArrayCap.ColorArray);
                GL.ColorPointer(3, ColorPointerType.Float, 6 * sizeof(float), 3 * sizeof(float));

                GL.BindVertexArray(0);
                _pointCloudBuffers[cloudId] = (vao, vbo, points.Count, colorMode);
                buffers = (vao, vbo, points.Count, colorMode);
            }

            GL.PointSize(pointSize);
            GL.BindVertexArray(buffers.vao);

            // Re-setup pointers every frame because some drivers don't reliably capture legacy pointers in VAOs
            GL.BindBuffer(BufferTarget.ArrayBuffer, buffers.vbo);
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.VertexPointer(3, VertexPointerType.Float, 6 * sizeof(float), 0);
            GL.EnableClientState(ArrayCap.ColorArray);
            GL.ColorPointer(3, ColorPointerType.Float, 6 * sizeof(float), 3 * sizeof(float));
            GL.DisableClientState(ArrayCap.TextureCoordArray);

            GL.DrawArrays(PrimitiveType.Points, 0, buffers.count);
            GL.BindVertexArray(0);
        }

        private void DrawPointCloud(PointCloudObject pointCloud)
        {
            int totalPoints = pointCloud.Points.Count;
            int visibleCount = pointCloud.GetVisiblePointCount();
            if (totalPoints == 0 || visibleCount == 0) return;

            var colorMode = IniSettings.Instance.PointCloudColor;
            bool useLegend = colorMode == PointCloudColorMode.DistanceMap || colorMode == PointCloudColorMode.Confidence;
            float minValue = 0.0f;
            float range = 1.0f;
            bool depthFallback = false;

            if (useLegend && TryComputeValueRange(
                    pointCloud.Points,
                    pointCloud.Confidence,
                    colorMode,
                    i => pointCloud.GetSourcePointIndex(i, visibleCount),
                    visibleCount,
                    out float min,
                    out float max,
                    out depthFallback))
            {
                minValue = min;
                range = max - min;
                if (range < 0.0001f) range = 1.0f;
                UpdatePointCloudLegend(colorMode, min, max, depthFallback);
            }

            if (!_pointCloudBuffers.TryGetValue(pointCloud.Id, out var buffers)
                || buffers.count != visibleCount
                || buffers.colorMode != colorMode)
            {
                if (buffers.vao != 0) GL.DeleteVertexArray(buffers.vao);
                if (buffers.vbo != 0) GL.DeleteBuffer(buffers.vbo);

                float[] data = new float[visibleCount * 6];
                for (int i = 0; i < visibleCount; i++)
                {
                    int sourceIndex = pointCloud.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= totalPoints)
                        continue;

                    var c = GetPointColor(
                        sourceIndex,
                        pointCloud.Points,
                        pointCloud.Colors,
                        pointCloud.Confidence,
                        colorMode,
                        minValue,
                        range,
                        depthFallback);

                    data[i * 6 + 0] = pointCloud.Points[sourceIndex].X;
                    data[i * 6 + 1] = pointCloud.Points[sourceIndex].Y;
                    data[i * 6 + 2] = pointCloud.Points[sourceIndex].Z;
                    data[i * 6 + 3] = c.X;
                    data[i * 6 + 4] = c.Y;
                    data[i * 6 + 5] = c.Z;
                }

                int vao = GL.GenVertexArray();
                int vbo = GL.GenBuffer();
                GL.BindVertexArray(vao);
                GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
                GL.BufferData(BufferTarget.ArrayBuffer, data.Length * sizeof(float), data, BufferUsageHint.StaticDraw);

                GL.EnableClientState(ArrayCap.VertexArray);
                GL.VertexPointer(3, VertexPointerType.Float, 6 * sizeof(float), 0);
                GL.EnableClientState(ArrayCap.ColorArray);
                GL.ColorPointer(3, ColorPointerType.Float, 6 * sizeof(float), 3 * sizeof(float));

                GL.BindVertexArray(0);
                _pointCloudBuffers[pointCloud.Id] = (vao, vbo, visibleCount, colorMode);
                buffers = (vao, vbo, visibleCount, colorMode);
            }

            GL.PointSize(pointCloud.PointSize);
            GL.BindVertexArray(buffers.vao);

            // Re-setup pointers every frame because some drivers don't reliably capture legacy pointers in VAOs
            GL.BindBuffer(BufferTarget.ArrayBuffer, buffers.vbo);
            GL.EnableClientState(ArrayCap.VertexArray);
            GL.VertexPointer(3, VertexPointerType.Float, 6 * sizeof(float), 0);
            GL.EnableClientState(ArrayCap.ColorArray);
            GL.ColorPointer(3, ColorPointerType.Float, 6 * sizeof(float), 3 * sizeof(float));
            GL.DisableClientState(ArrayCap.TextureCoordArray);

            GL.DrawArrays(PrimitiveType.Points, 0, buffers.count);
            GL.BindVertexArray(0);
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

            // Access pixels
            var info = mesh.Texture.Info;
            // Depending on Skia setup, might be BGRA or RGBA.
            // On cross platform, ColorType matters.
            PixelFormat format = PixelFormat.Bgra;
            if (info.ColorType == SkiaSharp.SKColorType.Rgba8888) format = PixelFormat.Rgba;

            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, info.Width, info.Height, 0,
                format, PixelType.UnsignedByte, mesh.Texture.GetPixels());

            mesh.TextureId = id;
        }

        private void DrawSkeleton(SkeletonObject skel)
        {
            GL.PushMatrix();
            // Apply Skeleton's world transform so local positions work,
            // BUT Rigging.cs has GetWorldPosition() which walks the hierarchy.
            // If we use GetWorldPosition(), we should NOT apply the Skeleton's transform here
            // IF Position is already part of that world calculation.
            // Looking at Rigging.cs: GetWorldPosition() uses Parent.GetWorldTransform().
            // And SkeletonObject simply holds the data.
            // The joints are usually relative to the skeleton root.
            // So we apply the SkeletonObject transform.

            var t = skel.GetWorldTransform();
            GL.MultMatrix(ref t);

            GL.Disable(EnableCap.DepthTest); // See through mesh

            // Draw Bones
            if (skel.ShowBones)
            {
                GL.LineWidth(3.0f);
                GL.Begin(PrimitiveType.Lines);
                foreach (var bone in skel.Skeleton.Bones)
                {
                    if (!bone.IsVisible) continue;

                    var start = bone.StartJoint.GetWorldPosition();
                    var end = bone.EndJoint.GetWorldPosition();

                    Vector3 color = bone.IsSelected ? skel.SelectedColor : skel.BoneColor;
                    GL.Color3(color);

                    GL.Vertex3(start);
                    GL.Vertex3(end);
                }
                GL.End();
            }

            // Draw Joints
            if (skel.ShowJoints)
            {
                GL.PointSize(8.0f);
                GL.Begin(PrimitiveType.Points);
                foreach (var joint in skel.Skeleton.Joints)
                {
                    if (!joint.IsVisible) continue;

                    var pos = joint.GetWorldPosition();
                    Vector3 color = joint.IsSelected ? skel.SelectedColor : joint.Color;
                    GL.Color3(color);

                    GL.Vertex3(pos);
                }
                GL.End();
            }

            GL.Enable(EnableCap.DepthTest);
            GL.PopMatrix();
        }

        private void DrawBoundingBox(SceneObject obj, bool highlighted)
        {
            var min = obj.BoundsMin;
            var max = obj.BoundsMax;
            if (highlighted)
            {
                GL.Color3(1f, 1f, 0f);
                GL.LineWidth(2.5f);
            }
            else
            {
                GL.Color3(0.45f, 0.7f, 1f);
                GL.LineWidth(1.0f);
            }
            GL.Begin(PrimitiveType.Lines);
            // Bottom
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, max.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, min.Z);
            // Top
            GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
            GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, max.Z);
            GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
            GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, min.Z);
            // Sides
            GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(min.X, max.Y, min.Z);
            GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
            GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(max.X, max.Y, max.Z);
            GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawCameraFrustum(CameraObject cam)
        {
            float scale = cam.FrustumScale > 0 ? cam.FrustumScale : Math.Max(0.001f, CameraFrustumScale);
            var corners = GetFrustumCornersWorld(cam, scale);
            var camPos = GetCameraPositionWorld(cam);
            var color = cam.Selected ? new Vector3(1f, 1f, 0f) : cam.FrustumColor;

            GL.Color3(color.X, color.Y, color.Z);
            GL.LineWidth(cam.Selected ? 2.5f : 1.5f);

            GL.Begin(PrimitiveType.Lines);
            // Camera center to image plane corners
            GL.Vertex3(camPos); GL.Vertex3(corners[0]);
            GL.Vertex3(camPos); GL.Vertex3(corners[1]);
            GL.Vertex3(camPos); GL.Vertex3(corners[2]);
            GL.Vertex3(camPos); GL.Vertex3(corners[3]);

            // Image plane rect
            GL.Vertex3(corners[0]); GL.Vertex3(corners[1]);
            GL.Vertex3(corners[1]); GL.Vertex3(corners[2]);
            GL.Vertex3(corners[2]); GL.Vertex3(corners[3]);
            GL.Vertex3(corners[3]); GL.Vertex3(corners[0]);

            // Plane cross to improve readability at distance
            GL.Vertex3(corners[0]); GL.Vertex3(corners[2]);
            GL.Vertex3(corners[1]); GL.Vertex3(corners[3]);
            GL.End();

            GL.LineWidth(1.0f);
        }

        private static Vector3[] GetFrustumCornersWorld(CameraObject cam, float distance)
        {
            float halfFovRad = MathHelper.DegreesToRadians(cam.FieldOfView * 0.5f);
            float halfHeight = distance * (float)Math.Tan(halfFovRad);
            float halfWidth = halfHeight * cam.AspectRatio;

            var corners = new[]
            {
                new Vector3(-halfWidth, -halfHeight, -distance),
                new Vector3(halfWidth, -halfHeight, -distance),
                new Vector3(halfWidth, halfHeight, -distance),
                new Vector3(-halfWidth, halfHeight, -distance)
            };

            Matrix4 transform = cam.Pose?.CameraToWorld ?? cam.GetWorldTransform();
            for (int i = 0; i < corners.Length; i++)
            {
                corners[i] = Vector3.TransformPosition(corners[i], transform);
            }

            return corners;
        }

        private static Vector3 GetCameraPositionWorld(CameraObject cam)
        {
            if (cam.Pose != null)
            {
                return cam.Pose.CameraToWorld.ExtractTranslation();
            }

            return Vector3.TransformPosition(Vector3.Zero, cam.GetWorldTransform());
        }

        // --- Interaction Logic ---

        private SceneObject? PickObject(int screenX, int screenY, int width, int height)
        {
            if (_sceneGraph == null) return null;

            SceneObject? closest = null;
            float minDist = float.MaxValue;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                // Specialized picking for Skeleton
                if (obj is SkeletonObject skel)
                {
                    // Check joints
                    foreach (var joint in skel.Skeleton.Joints)
                    {
                        if (!joint.IsVisible) continue;

                        var worldPos = Vector3.TransformPosition(joint.GetWorldPosition(), skel.GetWorldTransform());

                        var screenPosJ = Project(worldPos, width, height);
                        if (screenPosJ.Z < 0) continue;

                        float distJ = (new Vector2(screenPosJ.X, screenPosJ.Y) - new Vector2(screenX, screenY)).Length;
                        if (distJ < 20 && distJ < minDist)
                        {
                            minDist = distJ;
                            skel.Skeleton.SelectJoint(joint, false);
                            closest = skel;
                        }
                    }
                    if (closest == skel) continue; // Found a joint, skip bounding box check
                }

                var (boundsMin, boundsMax) = obj.GetWorldBounds();
                var center = (boundsMin + boundsMax) * 0.5f;
                var screenPos = Project(center, width, height);

                if (screenPos.Z < 0) continue; // Behind camera

                float dist = (new Vector2(screenPos.X, screenPos.Y) - new Vector2(screenX, screenY)).Length;

                // Simple proximity check
                if (dist < 50 && dist < minDist)
                {
                    minDist = dist;
                    closest = obj;
                }
            }
            return closest;
        }

        private Vector3 Project(Vector3 worldPos, int width, int height)
        {
            Vector4 vec = new Vector4(worldPos, 1.0f);
            vec = vec * _finalViewMatrix * _projectionMatrix;
            if (Math.Abs(vec.W) < 0.0001f) return Vector3.Zero;

            vec /= vec.W;

            float x = (vec.X + 1.0f) * 0.5f * width;
            float y = (1.0f - vec.Y) * 0.5f * height; // Flip Y for window coords

            return new Vector3(x, y, vec.Z);
        }

        private void ClearHoveredSourceDistance()
        {
            _hasHoveredSourceDistance = false;
            _hoveredSourceDistance = 0.0f;
            _hoveredDistanceIsMeters = false;
            _hoveredSourceCameraName = string.Empty;
        }

        private void UpdateHoveredSourceDistance(int localMouseX, int localMouseY, int width, int height)
        {
            if (_sceneGraph == null)
            {
                ClearHoveredSourceDistance();
                return;
            }

            if (!TryPickHoveredPoint(localMouseX, localMouseY, width, height, out var hoveredPointWorld))
            {
                ClearHoveredSourceDistance();
                return;
            }

            if (TryComputeSourceCameraDistance(hoveredPointWorld, out float value, out bool isMeters, out string sourceCameraName))
            {
                _hasHoveredSourceDistance = true;
                _hoveredSourceDistance = value;
                _hoveredDistanceIsMeters = isMeters;
                _hoveredSourceCameraName = sourceCameraName;
                return;
            }

            ClearHoveredSourceDistance();
        }

        private bool TryPickHoveredPoint(int localMouseX, int localMouseY, int width, int height, out Vector3 pointWorld)
        {
            pointWorld = Vector3.Zero;
            if (_sceneGraph == null)
                return false;

            Vector2 mouse = new Vector2(localMouseX, localMouseY);
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
                    int sourceIndex = pc.GetSourcePointIndex(i, visibleCount);
                    if (sourceIndex < 0 || sourceIndex >= pc.Points.Count)
                        continue;

                    Vector3 pWorld = Vector3.TransformPosition(pc.Points[sourceIndex], world);
                    Vector3 screenPos = Project(pWorld, width, height);
                    if (screenPos.Z < -1.0f || screenPos.Z > 1.0f)
                        continue;

                    float distPx = (new Vector2(screenPos.X, screenPos.Y) - mouse).Length;
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

        private int CheckGizmoIntersection(float mouseX, float mouseY, int width, int height)
        {
            if (_sceneGraph.SelectedObjects.Count == 0) return -1;

            Vector3 center = Vector3.Zero;
            foreach (var obj in _sceneGraph.SelectedObjects) center += obj.Position;
            center /= _sceneGraph.SelectedObjects.Count;

            Vector3 screenCenter = Project(center, width, height);
            Vector3 screenX = Project(center + new Vector3(_gizmoSize, 0, 0), width, height);
            Vector3 screenY = Project(center + new Vector3(0, _gizmoSize, 0), width, height);
            Vector3 screenZ = Project(center + new Vector3(0, 0, _gizmoSize), width, height);

            Vector2 m = new Vector2(mouseX, mouseY);
            float threshold = 20.0f;

            float dx = DistanceToLineSegment(m, new Vector2(screenCenter.X, screenCenter.Y), new Vector2(screenX.X, screenX.Y));
            float dy = DistanceToLineSegment(m, new Vector2(screenCenter.X, screenCenter.Y), new Vector2(screenY.X, screenY.Y));
            float dz = DistanceToLineSegment(m, new Vector2(screenCenter.X, screenCenter.Y), new Vector2(screenZ.X, screenZ.Y));

            if (dx < threshold && dx < dy && dx < dz) return 0;
            if (dy < threshold && dy < dx && dy < dz) return 1;
            if (dz < threshold && dz < dx && dz < dy) return 2;

            return -1;
        }

        private float DistanceToLineSegment(Vector2 p, Vector2 v, Vector2 w)
        {
            float l2 = (v - w).LengthSquared;
            if (l2 == 0) return (p - v).Length;
            float t = Math.Max(0, Math.Min(1, Vector2.Dot(p - v, w - v) / l2));
            Vector2 projection = v + t * (w - v);
            return (p - projection).Length;
        }

        private void HandleGizmoDrag(float mouseX, float mouseY)
        {
            float delta = 0;
            float sensitivity = 0.01f * Math.Abs(_zoom / 5.0f);

            // Simplified drag logic: mostly horizontal mouse movement maps to axis
            delta = (mouseX - _gizmoDragStart.X) * sensitivity;
            // Invert for Y if vertical drag preferred
            if (_activeGizmoAxis == 1) delta = - (mouseY - _gizmoDragStart.Y) * sensitivity;

            Vector3 change = Vector3.Zero;
            if (_activeGizmoAxis == 0) change.X = delta;
            if (_activeGizmoAxis == 1) change.Y = delta;
            if (_activeGizmoAxis == 2) change.Z = delta;

            foreach(var obj in _sceneGraph.SelectedObjects)
            {
                if (!_gizmoStartStates.TryGetValue(obj.Id, out var start))
                    continue;

                if (_gizmoMode == GizmoMode.Translate) obj.Position = start.Position + change;
                if (_gizmoMode == GizmoMode.Scale) obj.Scale = ClampScale(start.Scale + change);
                if (_gizmoMode == GizmoMode.Rotate) obj.Rotation = start.Rotation + change * 50.0f;
            }
        }

        private static Vector3 ClampScale(Vector3 scale)
        {
            const float minScale = 0.001f;
            return new Vector3(
                Math.Max(minScale, scale.X),
                Math.Max(minScale, scale.Y),
                Math.Max(minScale, scale.Z));
        }

        private void DrawGizmo()
        {
            if (_sceneGraph.SelectedObjects.Count == 0) return;
            Vector3 center = Vector3.Zero;
            foreach (var obj in _sceneGraph.SelectedObjects) center += obj.Position;
            center /= _sceneGraph.SelectedObjects.Count;

            _gizmoSize = Math.Abs(_zoom) * 0.15f;

            GL.Disable(EnableCap.DepthTest);
            if (_gizmoMode == GizmoMode.Translate) DrawTranslateGizmo(center);
            else if (_gizmoMode == GizmoMode.Rotate) DrawRotateGizmo(center);
            else if (_gizmoMode == GizmoMode.Scale) DrawScaleGizmo(center);
            GL.Enable(EnableCap.DepthTest);
        }

        private void DrawTranslateGizmo(Vector3 center)
        {
            float len = _gizmoSize;
            GL.LineWidth(3.0f);
            GL.Begin(PrimitiveType.Lines);
            // X
            GL.Color3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, 0.2f, 0.2f);
            GL.Vertex3(center); GL.Vertex3(center + new Vector3(len, 0, 0));
            // Y
            GL.Color3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, 0.2f);
            GL.Vertex3(center); GL.Vertex3(center + new Vector3(0, len, 0));
            // Z
            GL.Color3(0.2f, 0.2f, _activeGizmoAxis == 2 ? 1.0f : 0.8f);
            GL.Vertex3(center); GL.Vertex3(center + new Vector3(0, 0, len));
            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawRotateGizmo(Vector3 center)
        {
            // Simplified circles
            float r = _gizmoSize;
            int segs = 32;
            GL.LineWidth(2.0f);

            GL.Color3(_activeGizmoAxis == 0 ? 1.0f : 0.8f, 0.2f, 0.2f);
            GL.Begin(PrimitiveType.LineLoop);
            for(int i=0; i<segs; i++) { float a = i*MathF.PI*2/segs; GL.Vertex3(center.X, center.Y+MathF.Cos(a)*r, center.Z+MathF.Sin(a)*r); }
            GL.End();

            GL.Color3(0.2f, _activeGizmoAxis == 1 ? 1.0f : 0.8f, 0.2f);
            GL.Begin(PrimitiveType.LineLoop);
            for(int i=0; i<segs; i++) { float a = i*MathF.PI*2/segs; GL.Vertex3(center.X+MathF.Cos(a)*r, center.Y, center.Z+MathF.Sin(a)*r); }
            GL.End();

            GL.Color3(0.2f, 0.2f, _activeGizmoAxis == 2 ? 1.0f : 0.8f);
            GL.Begin(PrimitiveType.LineLoop);
            for(int i=0; i<segs; i++) { float a = i*MathF.PI*2/segs; GL.Vertex3(center.X+MathF.Cos(a)*r, center.Y+MathF.Sin(a)*r, center.Z); }
            GL.End();
            GL.LineWidth(1.0f);
        }

        private void DrawScaleGizmo(Vector3 center)
        {
            DrawTranslateGizmo(center); // Re-use for visual simplicity in this step, distinguishing via cubes is logic-heavy drawing
        }
    }
}
