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
        Pen,     // Triangle editing mode
        Rigging  // Skeleton rigging mode
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
        private Vector3 _gizmoStartPos;
        private Vector3 _gizmoDragStart;
        private float _gizmoSize = 1.0f;

        // Mesh Editing Tool for Pen mode
        private MeshEditingTool _meshEditingTool = new MeshEditingTool();
        public MeshEditingTool MeshEditingTool => _meshEditingTool;

        // Triangle selection events
        public event EventHandler? TriangleSelectionChanged;

        // Viewport dimensions for ray casting
        private int _viewportX, _viewportY, _viewportWidth, _viewportHeight;

        // Performance
        private int _frameCount = 0;
        private float _fps = 0;
        private DateTime _lastFpsUpdate = DateTime.Now;

        public float FPS => _fps;

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
            set => _gizmoMode = value;
        }

        public void InitGL()
        {
            GL.Enable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.PointSize(5.0f);
            GL.LineWidth(1.0f);
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

            DrawScene();
            CheckError("DrawScene");

            // Draw gizmo - hide in Select, Pen, and Rigging modes
            if (s.ShowGizmo && _sceneGraph.SelectedObjects.Count > 0 &&
                _gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.None &&
                _gizmoMode != GizmoMode.Pen && _gizmoMode != GizmoMode.Rigging)
            {
                DrawGizmo();
            }

            // Draw selected triangles highlight (Pen mode)
            if (_gizmoMode == GizmoMode.Pen && _meshEditingTool.SelectedTriangles.Count > 0)
            {
                DrawSelectedTriangles();
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
                             // Store initial state
                             _gizmoStartPos = _sceneGraph.SelectedObjects[0].Position;
                             _lastMousePos = mouse.Position;
                             return;
                         }
                     }

                     // If not gizmo, check picking
                     if (_gizmoMode == GizmoMode.Select)
                     {
                         var picked = PickObject((int)mouse.X, (int)mouse.Y, width, height);
                         bool multi = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.LeftControl);

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
                 _isDragging = false;
                 _isDraggingGizmo = false;
                 _activeGizmoAxis = -1;
             }

             if (mouse.IsButtonDown(MouseButton.Middle) || (mouse.IsButtonDown(MouseButton.Left) && keyboard.IsKeyDown(Keys.LeftShift)))
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

            bool addToSelection = keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.LeftControl);

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

        public void OnMouseWheel(float offset)
        {
            _zoom += offset;
            // Clamp zoom
            _zoom = Math.Clamp(_zoom, -100.0f, -0.5f);
        }

        /// <summary>
        /// Render with specified viewport region
        /// </summary>
        public void Render(int vpX, int vpY, int vpW, int vpH, int windowWidth, int windowHeight)
        {
            // Store viewport dimensions for ray casting
            _viewportX = vpX;
            _viewportY = vpY;
            _viewportWidth = vpW;
            _viewportHeight = vpH;

            // Set viewport for the 3D view area
            GL.Viewport(vpX, windowHeight - vpY - vpH, vpW, vpH);
            GL.Scissor(vpX, windowHeight - vpY - vpH, vpW, vpH);
            GL.Enable(EnableCap.ScissorTest);

            // Clear just this region using background color from settings
            var settings = IniSettings.Instance;
            GL.ClearColor(settings.ViewportBgR, settings.ViewportBgG, settings.ViewportBgB, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            Render(vpW, vpH);

            GL.Disable(EnableCap.ScissorTest);

            // Reset viewport to full window for ImGui
            GL.Viewport(0, 0, windowWidth, windowHeight);
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
                    if (cam.Visible) DrawCameraFrustum(cam);
                }
            }

            foreach(var obj in _sceneGraph.GetVisibleObjects())
            {
                if (obj is CameraObject) continue; // Already drawn

                if (obj is SkeletonObject skel && skel.Visible)
                {
                    DrawSkeleton(skel);
                    // Don't push matrix for skeleton as DrawSkeleton handles it per joint/bone logic or needs global context
                    // Actually DrawSkeleton should probably handle the root transform.
                    // Let's defer to the standard logic if possible, or handle it specially.
                    // SkeletonObject is a SceneObject, so it has a transform.
                    // The joints are usually relative to the skeleton root.
                    // So we apply the SkeletonObject transform.
                    continue;
                }

                GL.PushMatrix();
                var t = obj.GetWorldTransform();
                GL.MultMatrix(ref t);

                if (obj is PointCloudObject pc && s.ShowPointCloud)
                {
                    DrawPointCloud(pc.Points, pc.Colors, pc.PointSize);
                    if (obj.Selected)
                    {
                        DrawBoundingBox(obj);
                    }
                }

                if (obj is MeshObject mesh)
                {
                    bool isSelected = obj.Selected;
                    if ((s.ShowPointCloud || mesh.ShowAsPointCloud) && mesh.MeshData != null && mesh.MeshData.Vertices.Count > 0)
                    {
                        DrawPointCloud(mesh.MeshData.Vertices, mesh.MeshData.Colors, mesh.PointSize);
                    }

                    if (s.ShowWireframe || mesh.ShowWireframe) GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    else GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (s.ShowMesh && mesh.MeshData != null)
                    {
                         bool useTexture = s.ShowTexture && mesh.MeshData.Texture != null;
                         if (useTexture)
                         {
                             if (mesh.MeshData.TextureId == -1) UploadTexture(mesh.MeshData);
                             if (mesh.MeshData.TextureId != -1)
                             {
                                 GL.Enable(EnableCap.Texture2D);
                                 GL.BindTexture(TextureTarget.Texture2D, mesh.MeshData.TextureId);
                                 GL.Color3(1.0f, 1.0f, 1.0f);
                             }
                             else useTexture = false;
                         }

                         GL.Begin(PrimitiveType.Triangles);
                         bool hasColors = mesh.MeshData.Colors.Count >= mesh.MeshData.Vertices.Count;
                         bool hasUVs = useTexture && mesh.MeshData.UVs.Count >= mesh.MeshData.Vertices.Count;

                         for(int i=0; i < mesh.MeshData.Indices.Count; i++)
                         {
                             int idx = mesh.MeshData.Indices[i];
                             if (idx < mesh.MeshData.Vertices.Count)
                             {
                                 if (useTexture && hasUVs)
                                 {
                                     var uv = mesh.MeshData.UVs[idx];
                                     GL.TexCoord2(uv.X, uv.Y);
                                 }
                                 else if (hasColors)
                                 {
                                     var c = mesh.MeshData.Colors[idx];
                                     if(isSelected) GL.Color3(Math.Min(1, c.X + 0.2f), Math.Min(1, c.Y+0.2f), c.Z);
                                     else GL.Color3(c.X, c.Y, c.Z);
                                 }
                                 else
                                 {
                                     GL.Color3(isSelected ? 0.9f : 0.7f, 0.7f, 0.7f);
                                 }
                                 GL.Vertex3(mesh.MeshData.Vertices[idx]);
                             }
                         }
                         GL.End();

                         if (useTexture)
                         {
                             GL.BindTexture(TextureTarget.Texture2D, 0);
                             GL.Disable(EnableCap.Texture2D);
                         }
                    }
                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (isSelected)
                    {
                        DrawBoundingBox(obj);
                    }
                }

                GL.PopMatrix();
            }
        }

        private static void DrawPointCloud(IReadOnlyList<Vector3> points, IReadOnlyList<Vector3> colors, float pointSize)
        {
            if (points.Count == 0) return;

            GL.PointSize(pointSize);
            GL.Begin(PrimitiveType.Points);
            bool hasColors = colors.Count >= points.Count;
            for (int i = 0; i < points.Count; i++)
            {
                if (hasColors)
                {
                    var c = colors[i];
                    GL.Color3(c.X, c.Y, c.Z);
                }
                else
                {
                    GL.Color3(1.0f, 1.0f, 1.0f);
                }
                GL.Vertex3(points[i]);
            }
            GL.End();
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

        private void DrawBoundingBox(SceneObject obj)
        {
            var min = obj.BoundsMin;
            var max = obj.BoundsMax;
            GL.Color3(1, 1, 0);
            GL.LineWidth(2.0f);
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
            GL.PushMatrix();
            if (cam.Pose != null)
            {
                var m = cam.Pose.CameraToWorld;
                var mat = new Matrix4(
                    m.M11, m.M12, m.M13, m.M14,
                    m.M21, m.M22, m.M23, m.M24,
                    m.M31, m.M32, m.M33, m.M34,
                    m.M41, m.M42, m.M43, m.M44
                );
                GL.MultMatrix(ref mat);
            }
            else
            {
                var t = cam.GetWorldTransform();
                GL.MultMatrix(ref t);
            }

            float scale = 0.3f;
            var corners = cam.GetFrustumCorners(scale);
            var color = cam.Selected ? new Vector3(1f, 1f, 0f) : cam.FrustumColor;

            GL.Color3(color.X, color.Y, color.Z);
            GL.LineWidth(2.0f);

            GL.Begin(PrimitiveType.Lines);
            // 0 (Cam Center) to 4 corners
            GL.Vertex3(0,0,0); GL.Vertex3(corners[0]);
            GL.Vertex3(0,0,0); GL.Vertex3(corners[1]);
            GL.Vertex3(0,0,0); GL.Vertex3(corners[2]);
            GL.Vertex3(0,0,0); GL.Vertex3(corners[3]);

            // Image plane rect
            GL.Vertex3(corners[0]); GL.Vertex3(corners[1]);
            GL.Vertex3(corners[1]); GL.Vertex3(corners[2]);
            GL.Vertex3(corners[2]); GL.Vertex3(corners[3]);
            GL.Vertex3(corners[3]); GL.Vertex3(corners[0]);
            GL.End();

            GL.LineWidth(1.0f);
            GL.PopMatrix();
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
                if (_gizmoMode == GizmoMode.Translate) obj.Position = _gizmoStartPos + change;
                if (_gizmoMode == GizmoMode.Scale) obj.Scale = Vector3.One + change;
                if (_gizmoMode == GizmoMode.Rotate) obj.Rotation = _gizmoStartPos + change * 50.0f;
            }
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
