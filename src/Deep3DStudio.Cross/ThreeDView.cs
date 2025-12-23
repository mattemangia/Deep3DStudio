using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.GraphicsLibraryFramework;
using Deep3DStudio.Scene;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model;
using System.Linq;

namespace Deep3DStudio.Viewport
{
    public enum GizmoMode
    {
        None,
        Translate,
        Rotate,
        Scale,
        Select
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

        public ThreeDView(SceneGraph sceneGraph)
        {
            _sceneGraph = sceneGraph;
        }

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

            if (s.ShowGrid) DrawGrid();
            if (s.ShowAxes) DrawAxes();

            DrawScene();

            if (s.ShowGizmo && _sceneGraph.SelectedObjects.Count > 0 && _gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.None)
            {
                DrawGizmo();
            }
        }

        public void UpdateInput(MouseState mouse, KeyboardState keyboard, float dt, int width, int height)
        {
             // Gizmo Interaction or Camera Interaction
             if (mouse.IsButtonDown(MouseButton.Left))
             {
                 if (!_isDragging && !_isDraggingGizmo)
                 {
                     // Check Gizmo Intersection first
                     if (_gizmoMode != GizmoMode.Select && _gizmoMode != GizmoMode.None && _sceneGraph.SelectedObjects.Count > 0)
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
                 else if (_isDragging)
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

        public void OnMouseWheel(float offset)
        {
            _zoom += offset;
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

            foreach(var obj in _sceneGraph.GetVisibleObjects())
            {
                GL.PushMatrix();
                var t = obj.GetWorldTransform();
                GL.MultMatrix(ref t);

                if (obj is PointCloudObject pc && s.ShowPointCloud)
                {
                    GL.PointSize(pc.PointSize);
                    GL.Begin(PrimitiveType.Points);
                    bool hasColors = pc.Colors.Count >= pc.Points.Count;
                    for(int i=0; i<pc.Points.Count; i++)
                    {
                        if(hasColors) { var c = pc.Colors[i]; GL.Color3(c.X, c.Y, c.Z); }
                        else GL.Color3(1.0f, 1.0f, 1.0f);
                        GL.Vertex3(pc.Points[i]);
                    }
                    GL.End();
                }

                if (obj is MeshObject mesh)
                {
                    bool isSelected = obj.Selected;

                    if (s.ShowWireframe || mesh.ShowWireframe) GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Line);
                    else GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (s.ShowMesh && mesh.MeshData != null)
                    {
                         GL.Begin(PrimitiveType.Triangles);
                         bool hasColors = mesh.MeshData.Colors.Count >= mesh.MeshData.Vertices.Count;
                         for(int i=0; i < mesh.MeshData.Indices.Count; i++)
                         {
                             int idx = mesh.MeshData.Indices[i];
                             if (idx < mesh.MeshData.Vertices.Count)
                             {
                                 if (hasColors)
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
                    }
                    GL.PolygonMode(MaterialFace.FrontAndBack, PolygonMode.Fill);

                    if (isSelected)
                    {
                        // Draw box
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
                }

                GL.PopMatrix();
            }
        }

        // --- Interaction Logic ---

        private SceneObject? PickObject(int screenX, int screenY, int width, int height)
        {
            if (_sceneGraph == null) return null;

            SceneObject? closest = null;
            float minDist = float.MaxValue;

            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
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
