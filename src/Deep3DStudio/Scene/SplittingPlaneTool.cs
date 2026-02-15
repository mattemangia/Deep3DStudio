using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using OpenTK.Graphics.OpenGL;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    public class SplittingPlaneTool
    {
        public Vector3 Position { get; set; } = Vector3.Zero;
        public Quaternion Rotation { get; set; } = Quaternion.Identity;
        public float PlaneSize { get; set; } = 2.0f;
        
        private int _activeHandleAxis = -1;
        private bool _isDragging = false;
        private Vector3 _dragStart;
        
        private Vector3[] _planeVertices = new Vector3[4];
        private Vector3[] _handlePositions = new Vector3[3];
        
        private static readonly Vector3[] AxisColors = new Vector3[]
        {
            new Vector3(0.9f, 0.2f, 0.2f),
            new Vector3(0.2f, 0.9f, 0.2f),
            new Vector3(0.2f, 0.4f, 0.9f)
        };
        
        private static readonly Vector3[] AxisDirections = new Vector3[]
        {
            Vector3.UnitX,
            Vector3.UnitY,
            Vector3.UnitZ
        };
        
        public SplittingPlaneTool()
        {
            UpdateGeometry();
        }
        
        public void Reset()
        {
            Position = Vector3.Zero;
            Rotation = Quaternion.Identity;
            PlaneSize = 2.0f;
            _activeHandleAxis = -1;
            _isDragging = false;
            UpdateGeometry();
        }
        
        public (Vector3 point, Vector3 normal) GetPlaneEquation()
        {
            var normal = Vector3.TransformVector(Vector3.UnitY, GetRotationMatrix());
            return (Position, normal.Normalized());
        }
        
        public Matrix4 GetTransformMatrix()
        {
            var rotationMatrix = GetRotationMatrix();
            return Matrix4.CreateScale(PlaneSize) * rotationMatrix * Matrix4.CreateTranslation(Position);
        }
        
        public Matrix4 GetRotationMatrix()
        {
            return Matrix4.CreateFromQuaternion(Rotation);
        }
        
        private void UpdateGeometry()
        {
            float hs = 0.5f;
            _planeVertices[0] = new Vector3(-hs, 0, -hs);
            _planeVertices[1] = new Vector3(hs, 0, -hs);
            _planeVertices[2] = new Vector3(hs, 0, hs);
            _planeVertices[3] = new Vector3(-hs, 0, hs);
            
            float handleLength = 0.6f;
            _handlePositions[0] = new Vector3(handleLength, 0, 0);
            _handlePositions[1] = new Vector3(0, handleLength, 0);
            _handlePositions[2] = new Vector3(0, 0, handleLength);
        }
        
        public void Rotate(float deltaX, float deltaY)
        {
            var yaw = Matrix3.CreateRotationY(MathHelper.DegreesToRadians(deltaX));
            var pitch = Matrix3.CreateRotationX(MathHelper.DegreesToRadians(deltaY));
            
            var currentRotation = new Matrix3(GetRotationMatrix());
            var newRotation = pitch * currentRotation * yaw;
            
            Rotation = Quaternion.FromMatrix(newRotation);
        }
        
        public void RotateAroundAxis(int axis, float angleDegrees)
        {
            var axisVector = axis switch
            {
                0 => Vector3.UnitX,
                1 => Vector3.UnitY,
                2 => Vector3.UnitZ,
                _ => Vector3.UnitY
            };
            
            var additionalRotation = Quaternion.FromAxisAngle(axisVector, MathHelper.DegreesToRadians(angleDegrees));
            Rotation = additionalRotation * Rotation;
        }
        
        public int CheckHandleIntersection(Vector3 rayOrigin, Vector3 rayDirection, Matrix4 viewMatrix, Matrix4 projectionMatrix)
        {
            var transform = GetTransformMatrix();
            
            float bestT = float.MaxValue;
            int bestAxis = -1;
            float handleRadius = 0.1f * PlaneSize;
            float handleLength = 0.6f * PlaneSize;
            
            for (int axis = 0; axis < 3; axis++)
            {
                var localAxis = AxisDirections[axis];
                var worldAxisStart = Vector3.TransformPosition(Vector3.Zero, transform);
                var worldAxisEnd = Vector3.TransformPosition(localAxis * handleLength, transform);
                
                float t = RayCylinderIntersection(rayOrigin, rayDirection, worldAxisStart, worldAxisEnd, handleRadius);
                
                if (t >= 0 && t < bestT)
                {
                    bestT = t;
                    bestAxis = axis;
                }
            }
            
            return bestAxis;
        }
        
        private float RayCylinderIntersection(Vector3 rayOrigin, Vector3 rayDir, Vector3 cylinderStart, Vector3 cylinderEnd, float radius)
        {
            var axis = cylinderEnd - cylinderStart;
            var axisLength = axis.Length;
            axis /= axisLength;
            
            var oc = rayOrigin - cylinderStart;
            
            var dPerp = rayDir - axis * Vector3.Dot(rayDir, axis);
            var ocPerp = oc - axis * Vector3.Dot(oc, axis);
            
            float a = dPerp.LengthSquared;
            float b = 2 * Vector3.Dot(dPerp, ocPerp);
            float c = ocPerp.LengthSquared - radius * radius;
            
            float discriminant = b * b - 4 * a * c;
            
            if (discriminant < 0) return -1;
            
            float sqrtD = MathF.Sqrt(discriminant);
            float t1 = (-b - sqrtD) / (2 * a);
            float t2 = (-b + sqrtD) / (2 * a);
            
            float t = t1 >= 0 ? t1 : (t2 >= 0 ? t2 : -1);
            
            if (t < 0) return -1;
            
            var hitPoint = rayOrigin + rayDir * t;
            float alongAxis = Vector3.Dot(hitPoint - cylinderStart, axis);
            
            if (alongAxis < 0 || alongAxis > axisLength) return -1;
            
            return t;
        }
        
        public void StartDrag(int axis, Vector3 rayOrigin, Vector3 rayDirection)
        {
            _activeHandleAxis = axis;
            _isDragging = true;
            
            var planeEq = GetPlaneEquation();
            float t = RayPlaneIntersection(rayOrigin, rayDirection, Position, planeEq.normal);
            _dragStart = t >= 0 ? rayOrigin + rayDirection * t : Position;
        }
        
        public void UpdateDrag(Vector3 rayOrigin, Vector3 rayDirection)
        {
            if (!_isDragging || _activeHandleAxis < 0) return;
            
            var planeEq = GetPlaneEquation();
            
            var axisDir = _activeHandleAxis switch
            {
                0 => Vector3.UnitX,
                1 => Vector3.UnitY,
                2 => Vector3.UnitZ,
                _ => Vector3.UnitY
            };
            
            var worldAxis = Vector3.TransformVector(axisDir, GetRotationMatrix());
            
            var dragPlaneNormal = Vector3.Cross(worldAxis, planeEq.normal);
            if (dragPlaneNormal.LengthSquared < 0.001f)
                dragPlaneNormal = Vector3.Cross(worldAxis, Vector3.UnitX);
            dragPlaneNormal = dragPlaneNormal.Normalized();
            
            float t = RayPlaneIntersection(rayOrigin, rayDirection, _dragStart, dragPlaneNormal);
            if (t < 0) return;
            
            var currentPoint = rayOrigin + rayDirection * t;
            var delta = currentPoint - _dragStart;
            
            float projectedDelta = Vector3.Dot(delta, worldAxis);
            Position += worldAxis * projectedDelta;
            
            _dragStart = currentPoint;
        }
        
        public void EndDrag()
        {
            _isDragging = false;
            _activeHandleAxis = -1;
        }
        
        private float RayPlaneIntersection(Vector3 rayOrigin, Vector3 rayDir, Vector3 planePoint, Vector3 planeNormal)
        {
            float denom = Vector3.Dot(planeNormal, rayDir);
            if (Math.Abs(denom) < 0.0001f) return -1;
            
            float t = Vector3.Dot(planePoint - rayOrigin, planeNormal) / denom;
            return t >= 0 ? t : -1;
        }
        
        public static (MeshData above, MeshData below) SplitMeshByPlane(
            MeshData mesh,
            Vector3 planePoint,
            Vector3 planeNormal,
            IProgress<float>? progress = null)
        {
            planeNormal = planeNormal.Normalized();
            
            var aboveVertices = new List<int>();
            var belowVertices = new List<int>();
            
            int totalVertices = mesh.Vertices.Count;
            
            for (int i = 0; i < totalVertices; i++)
            {
                var v = mesh.Vertices[i];
                float dist = Vector3.Dot(v - planePoint, planeNormal);
                
                if (dist >= 0)
                    aboveVertices.Add(i);
                else
                    belowVertices.Add(i);
                
                progress?.Report((float)i / totalVertices * 0.5f);
            }
            
            var aboveMesh = ExtractSubMesh(mesh, new HashSet<int>(aboveVertices));
            progress?.Report(0.75f);
            
            var belowMesh = ExtractSubMesh(mesh, new HashSet<int>(belowVertices));
            progress?.Report(1.0f);
            
            return (aboveMesh, belowMesh);
        }
        
        private static MeshData ExtractSubMesh(MeshData mesh, HashSet<int> vertexIndices)
        {
            var keepVertex = new bool[mesh.Vertices.Count];
            foreach (int idx in vertexIndices)
                keepVertex[idx] = true;
            
            var newVertices = new List<Vector3>();
            var newColors = new List<Vector3>();
            var newUVs = new List<Vector2>();
            var oldToNew = new int[mesh.Vertices.Count];
            
            for (int i = 0; i < mesh.Vertices.Count; i++)
                oldToNew[i] = -1;
            
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                if (keepVertex[i])
                {
                    oldToNew[i] = newVertices.Count;
                    newVertices.Add(mesh.Vertices[i]);
                    
                    if (i < mesh.Colors.Count)
                        newColors.Add(mesh.Colors[i]);
                    else
                        newColors.Add(new Vector3(0.8f));
                    
                    if (i < mesh.UVs.Count)
                        newUVs.Add(mesh.UVs[i]);
                }
            }
            
            var newIndices = new List<int>();
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];
                
                int ni0 = oldToNew[i0];
                int ni1 = oldToNew[i1];
                int ni2 = oldToNew[i2];
                
                if (ni0 >= 0 && ni1 >= 0 && ni2 >= 0)
                {
                    newIndices.Add(ni0);
                    newIndices.Add(ni1);
                    newIndices.Add(ni2);
                }
            }
            
            return new MeshData
            {
                Vertices = newVertices,
                Colors = newColors,
                UVs = newUVs,
                Indices = newIndices
            };
        }
        
        public void Render()
        {
            var transform = GetTransformMatrix();
            
            GL.PushMatrix();
            
            unsafe
            {
                float* matrix = stackalloc float[16];
                var transposed = transform;
                matrix[0] = transposed.M11; matrix[1] = transposed.M12; matrix[2] = transposed.M13; matrix[3] = transposed.M14;
                matrix[4] = transposed.M21; matrix[5] = transposed.M22; matrix[6] = transposed.M23; matrix[7] = transposed.M24;
                matrix[8] = transposed.M31; matrix[9] = transposed.M32; matrix[10] = transposed.M33; matrix[11] = transposed.M34;
                matrix[12] = transposed.M41; matrix[13] = transposed.M42; matrix[14] = transposed.M43; matrix[15] = transposed.M44;
                GL.MultMatrix(matrix);
            }
            
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.Disable(EnableCap.Lighting);
            GL.Enable(EnableCap.LineSmooth);
            
            GL.Color4(0.9f, 0.7f, 0.2f, 0.35f);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex3(-0.5f, 0, -0.5f);
            GL.Vertex3(0.5f, 0, -0.5f);
            GL.Vertex3(0.5f, 0, 0.5f);
            GL.Vertex3(-0.5f, 0, 0.5f);
            GL.End();
            
            GL.LineWidth(2.5f);
            GL.Color4(1.0f, 0.8f, 0.3f, 1.0f);
            GL.Begin(PrimitiveType.LineLoop);
            GL.Vertex3(-0.5f, 0, -0.5f);
            GL.Vertex3(0.5f, 0, -0.5f);
            GL.Vertex3(0.5f, 0, 0.5f);
            GL.Vertex3(-0.5f, 0, 0.5f);
            GL.End();
            
            GL.LineWidth(1.0f);
            GL.Color4(1.0f, 1.0f, 1.0f, 0.3f);
            GL.Begin(PrimitiveType.Lines);
            for (int i = -2; i <= 2; i++)
            {
                if (i == 0) continue;
                float t = i * 0.2f;
                GL.Vertex3(-0.5f, 0, t);
                GL.Vertex3(0.5f, 0, t);
                GL.Vertex3(t, 0, -0.5f);
                GL.Vertex3(t, 0, 0.5f);
            }
            GL.End();
            
            float handleLength = 0.6f;
            float handleRadius = 0.04f;
            
            for (int axis = 0; axis < 3; axis++)
            {
                var color = AxisColors[axis];
                var dir = AxisDirections[axis];
                bool isActive = _activeHandleAxis == axis;
                
                if (isActive)
                    color = new Vector3(1.0f, 1.0f, 0.0f);
                
                GL.Color4(color.X, color.Y, color.Z, 1.0f);
                GL.LineWidth(isActive ? 4.0f : 2.5f);
                GL.Begin(PrimitiveType.Lines);
                GL.Vertex3(0, 0, 0);
                GL.Vertex3(dir * handleLength);
                GL.End();
                
                GL.PointSize(isActive ? 12.0f : 8.0f);
                GL.Begin(PrimitiveType.Points);
                GL.Vertex3(dir * handleLength);
                GL.End();
                
                DrawArrowHead(dir * handleLength, dir, 0.08f, color);
            }
            
            GL.LineWidth(1.0f);
            GL.PointSize(1.0f);
            GL.Disable(EnableCap.LineSmooth);
            GL.Disable(EnableCap.Blend);
            
            GL.PopMatrix();
        }
        
        private void DrawArrowHead(Vector3 tip, Vector3 direction, float size, Vector3 color)
        {
            direction = direction.Normalized();
            
            Vector3 up = Math.Abs(direction.Y) < 0.9f ? Vector3.UnitY : Vector3.UnitX;
            Vector3 right = Vector3.Cross(direction, up).Normalized();
            up = Vector3.Cross(right, direction).Normalized();
            
            Vector3 base1 = tip - direction * size + right * size * 0.3f;
            Vector3 base2 = tip - direction * size - right * size * 0.3f;
            Vector3 base3 = tip - direction * size + up * size * 0.3f;
            
            GL.Color3(color.X, color.Y, color.Z);
            GL.Begin(PrimitiveType.Triangles);
            GL.Vertex3(tip); GL.Vertex3(base1); GL.Vertex3(base3);
            GL.Vertex3(tip); GL.Vertex3(base3); GL.Vertex3(base2);
            GL.Vertex3(tip); GL.Vertex3(base2); GL.Vertex3(base1);
            GL.End();
        }
    }
}
