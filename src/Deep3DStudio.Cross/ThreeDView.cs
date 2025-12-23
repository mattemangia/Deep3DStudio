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

        // Legacy compatibility
        private List<MeshData> _meshes = new List<MeshData>();

        public ThreeDView(SceneGraph sceneGraph)
        {
            _sceneGraph = sceneGraph;
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

            var rx = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(_rotationX));
            var ry = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationY));
            var rotation = ry * rx;

            _viewMatrix = Matrix4.CreateTranslation(-_cameraTarget) *
                          rotation *
                          Matrix4.CreateTranslation(0, 0, _zoom);

            GL.MatrixMode(MatrixMode.Projection);
            GL.LoadMatrix(ref _projectionMatrix);
            GL.MatrixMode(MatrixMode.Modelview);
            GL.LoadMatrix(ref _viewMatrix);

            DrawGrid();
            DrawAxes();
            DrawScene();
        }

        public void UpdateInput(MouseState mouse, KeyboardState keyboard, float dt, int width, int height)
        {
             if (mouse.IsButtonDown(MouseButton.Left))
             {
                 if (!_isDragging)
                 {
                     _isDragging = true;
                     _lastMousePos = mouse.Position;
                 }

                 var delta = mouse.Position - _lastMousePos;
                 _rotationY += delta.X * 0.5f;
                 _rotationX += delta.Y * 0.5f;

                 _lastMousePos = mouse.Position;
             }
             else
             {
                 _isDragging = false;
             }

             if (mouse.IsButtonDown(MouseButton.Middle))
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
            GL.Color4(0.3f, 0.3f, 0.3f, 0.5f);
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

            foreach(var obj in _sceneGraph.GetVisibleObjects())
            {
                GL.PushMatrix();
                var t = obj.GetWorldTransform();
                GL.MultMatrix(ref t);

                if (obj is PointCloudObject pc)
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
                    // Simple lighting override
                    GL.Color3(0.8f, 0.8f, 0.8f);

                    if (mesh.MeshData != null && mesh.MeshData.Vertices.Count > 0)
                    {
                         GL.Begin(PrimitiveType.Triangles);
                         foreach(var idx in mesh.MeshData.Indices)
                         {
                             if (idx < mesh.MeshData.Vertices.Count)
                             {
                                 // Basic normal lighting would be good here, but flat for now
                                 GL.Vertex3(mesh.MeshData.Vertices[idx]);
                             }
                         }
                         GL.End();
                    }

                    if (isSelected)
                    {
                        // Draw wireframe or box
                        var min = obj.BoundsMin;
                        var max = obj.BoundsMax;
                        GL.Color3(1, 1, 0);
                        GL.LineWidth(2.0f);
                        GL.Begin(PrimitiveType.Lines);
                        GL.Vertex3(min.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, min.Z);
                        GL.Vertex3(max.X, min.Y, min.Z); GL.Vertex3(max.X, min.Y, max.Z);
                        GL.Vertex3(max.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, max.Z);
                        GL.Vertex3(min.X, min.Y, max.Z); GL.Vertex3(min.X, min.Y, min.Z);

                        GL.Vertex3(min.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, min.Z);
                        GL.Vertex3(max.X, max.Y, min.Z); GL.Vertex3(max.X, max.Y, max.Z);
                        GL.Vertex3(max.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, max.Z);
                        GL.Vertex3(min.X, max.Y, max.Z); GL.Vertex3(min.X, max.Y, min.Z);

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
    }
}
