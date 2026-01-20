using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    /// <summary>
    /// Types of scene objects that can be displayed in the viewport
    /// </summary>
    public enum SceneObjectType
    {
        Root,
        Mesh,
        PointCloud,
        Camera,
        Group,
        Light,
        Annotation,
        Skeleton
    }

    /// <summary>
    /// Base class for all objects in the scene graph
    /// </summary>
    public abstract class SceneObject
    {
        private static int _nextId = 1;

        public int Id { get; }
        public string Name { get; set; }
        public SceneObjectType ObjectType { get; protected set; }
        public bool Visible { get; set; } = true;
        public bool Selected { get; set; } = false;
        public bool Locked { get; set; } = false;

        // Transform properties
        public Vector3 Position { get; set; } = Vector3.Zero;
        public Vector3 Rotation { get; set; } = Vector3.Zero; // Euler angles in degrees
        public Vector3 Scale { get; set; } = Vector3.One;

        // Hierarchy
        public SceneObject? Parent { get; private set; }
        protected List<SceneObject> _children = new List<SceneObject>();
        public IReadOnlyList<SceneObject> Children => _children;

        // Bounding box (local space)
        public Vector3 BoundsMin { get; protected set; } = Vector3.Zero;
        public Vector3 BoundsMax { get; protected set; } = Vector3.Zero;

        // Color for selection/highlight
        public Vector3 OverrideColor { get; set; } = new Vector3(1, 1, 1);
        public bool UseOverrideColor { get; set; } = false;

        public SceneObject(string name)
        {
            Id = _nextId++;
            Name = name;
        }

        /// <summary>
        /// Gets the local transformation matrix
        /// </summary>
        public Matrix4 GetLocalTransform()
        {
            var translation = Matrix4.CreateTranslation(Position);
            var rotX = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(Rotation.X));
            var rotY = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(Rotation.Y));
            var rotZ = Matrix4.CreateRotationZ(MathHelper.DegreesToRadians(Rotation.Z));
            var scale = Matrix4.CreateScale(Scale);

            return scale * rotX * rotY * rotZ * translation;
        }

        /// <summary>
        /// Gets the world transformation matrix (including all parent transforms)
        /// </summary>
        public Matrix4 GetWorldTransform()
        {
            var local = GetLocalTransform();
            if (Parent != null)
            {
                return local * Parent.GetWorldTransform();
            }
            return local;
        }

        /// <summary>
        /// Sets the world position (adjusting local position based on parent)
        /// </summary>
        public void SetWorldPosition(Vector3 worldPos)
        {
            if (Parent != null)
            {
                var parentInverse = Parent.GetWorldTransform().Inverted();
                var localPos = Vector3.TransformPosition(worldPos, parentInverse);
                Position = localPos;
            }
            else
            {
                Position = worldPos;
            }
        }

        /// <summary>
        /// Gets the world-space bounding box
        /// </summary>
        public (Vector3 min, Vector3 max) GetWorldBounds()
        {
            var transform = GetWorldTransform();
            var corners = new Vector3[8];
            corners[0] = new Vector3(BoundsMin.X, BoundsMin.Y, BoundsMin.Z);
            corners[1] = new Vector3(BoundsMax.X, BoundsMin.Y, BoundsMin.Z);
            corners[2] = new Vector3(BoundsMin.X, BoundsMax.Y, BoundsMin.Z);
            corners[3] = new Vector3(BoundsMax.X, BoundsMax.Y, BoundsMin.Z);
            corners[4] = new Vector3(BoundsMin.X, BoundsMin.Y, BoundsMax.Z);
            corners[5] = new Vector3(BoundsMax.X, BoundsMin.Y, BoundsMax.Z);
            corners[6] = new Vector3(BoundsMin.X, BoundsMax.Y, BoundsMax.Z);
            corners[7] = new Vector3(BoundsMax.X, BoundsMax.Y, BoundsMax.Z);

            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);

            foreach (var corner in corners)
            {
                var worldCorner = Vector3.TransformPosition(corner, transform);
                min = Vector3.ComponentMin(min, worldCorner);
                max = Vector3.ComponentMax(max, worldCorner);
            }

            return (min, max);
        }

        public void AddChild(SceneObject child)
        {
            if (child.Parent != null)
                child.Parent.RemoveChild(child);

            child.Parent = this;
            _children.Add(child);
        }

        public void RemoveChild(SceneObject child)
        {
            if (_children.Remove(child))
            {
                child.Parent = null;
            }
        }

        /// <summary>
        /// Applies a transform to this object's position/rotation/scale
        /// </summary>
        public virtual void ApplyTransform(Matrix4 transform)
        {
            var currentTransform = GetLocalTransform();
            var newTransform = currentTransform * transform;

            // Extract translation
            Position = newTransform.ExtractTranslation();

            // Extract scale (approximate)
            var scaleX = new Vector3(newTransform.M11, newTransform.M21, newTransform.M31).Length;
            var scaleY = new Vector3(newTransform.M12, newTransform.M22, newTransform.M32).Length;
            var scaleZ = new Vector3(newTransform.M13, newTransform.M23, newTransform.M33).Length;
            Scale = new Vector3(scaleX, scaleY, scaleZ);
        }

        /// <summary>
        /// Called when the object needs to recalculate its bounds
        /// </summary>
        public abstract void UpdateBounds();

        /// <summary>
        /// Creates a deep copy of this object
        /// </summary>
        public abstract SceneObject Clone();
    }

    /// <summary>
    /// Mesh object that contains geometry data
    /// </summary>
    public class MeshObject : SceneObject
    {
        public MeshData MeshData { get; set; }
        public bool ShowAsPointCloud { get; set; } = false;
        public float PointSize { get; set; } = 8.0f;
        public bool ShowWireframe { get; set; } = false;

        // Statistics
        public int VertexCount => MeshData?.Vertices.Count ?? 0;
        public int TriangleCount => (MeshData?.Indices.Count ?? 0) / 3;

        public MeshObject(string name, MeshData mesh) : base(name)
        {
            ObjectType = SceneObjectType.Mesh;
            MeshData = mesh;
            UpdateBounds();
            Console.WriteLine($"MeshObject '{name}' created: {VertexCount} vertices, {TriangleCount} triangles, bounds: ({BoundsMin.X:F2},{BoundsMin.Y:F2},{BoundsMin.Z:F2}) to ({BoundsMax.X:F2},{BoundsMax.Y:F2},{BoundsMax.Z:F2})");
        }

        public override void UpdateBounds()
        {
            if (MeshData == null || MeshData.Vertices.Count == 0)
            {
                BoundsMin = BoundsMax = Vector3.Zero;
                return;
            }

            BoundsMin = new Vector3(float.MaxValue);
            BoundsMax = new Vector3(float.MinValue);

            foreach (var v in MeshData.Vertices)
            {
                BoundsMin = Vector3.ComponentMin(BoundsMin, v);
                BoundsMax = Vector3.ComponentMax(BoundsMax, v);
            }
        }

        public Vector3 GetCentroid()
        {
            if (MeshData == null || MeshData.Vertices.Count == 0)
                return Vector3.Zero;

            var sum = Vector3.Zero;
            foreach (var v in MeshData.Vertices)
                sum += v;
            return sum / MeshData.Vertices.Count;
        }

        public override SceneObject Clone()
        {
            var clonedMesh = new MeshData
            {
                Vertices = new List<Vector3>(MeshData.Vertices),
                Colors = new List<Vector3>(MeshData.Colors),
                Indices = new List<int>(MeshData.Indices),
                PixelToVertexIndex = MeshData.PixelToVertexIndex
            };

            return new MeshObject(Name + " (Copy)", clonedMesh)
            {
                Position = Position,
                Rotation = Rotation,
                Scale = Scale,
                Visible = Visible,
                ShowAsPointCloud = ShowAsPointCloud,
                PointSize = PointSize,
                ShowWireframe = ShowWireframe
            };
        }
    }

    /// <summary>
    /// Point cloud object (similar to mesh but rendered as points)
    /// </summary>
    public class PointCloudObject : SceneObject
    {
        public List<Vector3> Points { get; set; } = new List<Vector3>();
        public List<Vector3> Colors { get; set; } = new List<Vector3>();
        public float PointSize { get; set; } = 8.0f;

        public int PointCount => Points.Count;

        public PointCloudObject(string name) : base(name)
        {
            ObjectType = SceneObjectType.PointCloud;
        }

        public PointCloudObject(string name, MeshData mesh) : base(name)
        {
            ObjectType = SceneObjectType.PointCloud;
            Points = new List<Vector3>(mesh.Vertices);
            Colors = new List<Vector3>(mesh.Colors);
            UpdateBounds();
            Console.WriteLine($"PointCloudObject '{name}' created: {Points.Count} points, bounds: ({BoundsMin.X:F2},{BoundsMin.Y:F2},{BoundsMin.Z:F2}) to ({BoundsMax.X:F2},{BoundsMax.Y:F2},{BoundsMax.Z:F2})");
        }

        public override void UpdateBounds()
        {
            if (Points.Count == 0)
            {
                BoundsMin = BoundsMax = Vector3.Zero;
                return;
            }

            BoundsMin = new Vector3(float.MaxValue);
            BoundsMax = new Vector3(float.MinValue);

            foreach (var p in Points)
            {
                BoundsMin = Vector3.ComponentMin(BoundsMin, p);
                BoundsMax = Vector3.ComponentMax(BoundsMax, p);
            }
        }

        public override SceneObject Clone()
        {
            return new PointCloudObject(Name + " (Copy)")
            {
                Points = new List<Vector3>(Points),
                Colors = new List<Vector3>(Colors),
                PointSize = PointSize,
                Position = Position,
                Rotation = Rotation,
                Scale = Scale,
                Visible = Visible
            };
        }
    }

    /// <summary>
    /// Camera object representing a viewpoint
    /// </summary>
    public class CameraObject : SceneObject
    {
        public CameraPose? Pose { get; set; }
        public float NearPlane { get; set; } = 0.1f;
        public float FarPlane { get; set; } = 100f;
        public float FieldOfView { get; set; } = 60f; // degrees
        public float AspectRatio { get; set; } = 1.333f;

        // Image associated with this camera
        public string ImagePath { get; set; } = string.Empty;
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }

        // Visualization settings
        public float FrustumScale { get; set; } = 0.3f; // Size of camera frustum visualization
        public Vector3 FrustumColor { get; set; } = new Vector3(1f, 0.8f, 0f); // Yellow-ish
        public bool ShowFrustum { get; set; } = true;
        public bool ShowImagePlane { get; set; } = false;

        public CameraObject(string name) : base(name)
        {
            ObjectType = SceneObjectType.Camera;
            BoundsMin = new Vector3(-0.1f);
            BoundsMax = new Vector3(0.1f);
        }

        public CameraObject(string name, CameraPose pose) : base(name)
        {
            ObjectType = SceneObjectType.Camera;
            Pose = pose;
            ImagePath = pose.ImagePath;
            ImageWidth = pose.Width;
            ImageHeight = pose.Height;
            AspectRatio = (float)pose.Width / pose.Height;

            // Extract position from pose matrix
            var pos = pose.CameraToWorld.ExtractTranslation();
            Position = pos;

            BoundsMin = new Vector3(-0.1f);
            BoundsMax = new Vector3(0.1f);
        }

        /// <summary>
        /// Gets the camera's view direction in world space
        /// </summary>
        public Vector3 GetViewDirection()
        {
            if (Pose != null)
            {
                // Camera looks along -Z in camera space
                var dir = new Vector4(0, 0, -1, 0);
                var worldDir = dir * Pose.CameraToWorld;
                return new Vector3(worldDir.X, worldDir.Y, worldDir.Z).Normalized();
            }

            // Use rotation
            var rotX = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(Rotation.X));
            var rotY = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(Rotation.Y));
            var rotZ = Matrix4.CreateRotationZ(MathHelper.DegreesToRadians(Rotation.Z));
            var forward = new Vector4(0, 0, -1, 0) * (rotX * rotY * rotZ);
            return new Vector3(forward.X, forward.Y, forward.Z).Normalized();
        }

        /// <summary>
        /// Gets the camera's up vector in world space
        /// </summary>
        public Vector3 GetUpDirection()
        {
            if (Pose != null)
            {
                var up = new Vector4(0, 1, 0, 0);
                var worldUp = up * Pose.CameraToWorld;
                return new Vector3(worldUp.X, worldUp.Y, worldUp.Z).Normalized();
            }

            var rotX = Matrix4.CreateRotationX(MathHelper.DegreesToRadians(Rotation.X));
            var rotY = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(Rotation.Y));
            var rotZ = Matrix4.CreateRotationZ(MathHelper.DegreesToRadians(Rotation.Z));
            var upVec = new Vector4(0, 1, 0, 0) * (rotX * rotY * rotZ);
            return new Vector3(upVec.X, upVec.Y, upVec.Z).Normalized();
        }

        /// <summary>
        /// Gets the frustum corners for visualization
        /// </summary>
        public Vector3[] GetFrustumCorners(float distance)
        {
            var corners = new Vector3[4];
            float halfFovRad = MathHelper.DegreesToRadians(FieldOfView * 0.5f);
            float halfHeight = distance * (float)Math.Tan(halfFovRad);
            float halfWidth = halfHeight * AspectRatio;

            // Corners in camera space (looking along -Z)
            corners[0] = new Vector3(-halfWidth, -halfHeight, -distance); // Bottom-left
            corners[1] = new Vector3(halfWidth, -halfHeight, -distance);  // Bottom-right
            corners[2] = new Vector3(halfWidth, halfHeight, -distance);   // Top-right
            corners[3] = new Vector3(-halfWidth, halfHeight, -distance);  // Top-left

            // Transform to world space
            var transform = Pose?.CameraToWorld ?? GetLocalTransform();
            for (int i = 0; i < 4; i++)
            {
                corners[i] = Vector3.TransformPosition(corners[i], transform);
            }

            return corners;
        }

        public override void UpdateBounds()
        {
            // Camera bounds are fixed for now
            BoundsMin = new Vector3(-FrustumScale);
            BoundsMax = new Vector3(FrustumScale);
        }

        public override SceneObject Clone()
        {
            return new CameraObject(Name + " (Copy)")
            {
                Pose = Pose,
                Position = Position,
                Rotation = Rotation,
                Scale = Scale,
                Visible = Visible,
                NearPlane = NearPlane,
                FarPlane = FarPlane,
                FieldOfView = FieldOfView,
                AspectRatio = AspectRatio,
                ImagePath = ImagePath,
                ImageWidth = ImageWidth,
                ImageHeight = ImageHeight,
                FrustumScale = FrustumScale,
                FrustumColor = FrustumColor,
                ShowFrustum = ShowFrustum,
                ShowImagePlane = ShowImagePlane
            };
        }
    }

    /// <summary>
    /// Group object for organizing other objects
    /// </summary>
    public class GroupObject : SceneObject
    {
        public GroupObject(string name) : base(name)
        {
            ObjectType = SceneObjectType.Group;
        }

        public override void UpdateBounds()
        {
            if (_children.Count == 0)
            {
                BoundsMin = BoundsMax = Vector3.Zero;
                return;
            }

            BoundsMin = new Vector3(float.MaxValue);
            BoundsMax = new Vector3(float.MinValue);

            foreach (var child in _children)
            {
                var (min, max) = child.GetWorldBounds();
                BoundsMin = Vector3.ComponentMin(BoundsMin, min);
                BoundsMax = Vector3.ComponentMax(BoundsMax, max);
            }
        }

        public override SceneObject Clone()
        {
            var clone = new GroupObject(Name + " (Copy)")
            {
                Position = Position,
                Rotation = Rotation,
                Scale = Scale,
                Visible = Visible
            };

            foreach (var child in _children)
            {
                clone.AddChild(child.Clone());
            }

            return clone;
        }
    }

    /// <summary>
    /// Main scene graph manager
    /// </summary>
    public class SceneGraph
    {
        public GroupObject Root { get; }
        public List<SceneObject> SelectedObjects { get; } = new List<SceneObject>();

        public int ObjectCount => GetAllObjects().Count;
        public IEnumerable<SceneObject> AllObjects => GetObjectsRecursive(Root);

        public event EventHandler<SceneObject>? ObjectAdded;
        public event EventHandler<SceneObject>? ObjectRemoved;
        public event EventHandler<SceneObject>? ObjectSelected;
        public event EventHandler<SceneObject>? ObjectDeselected;
        public event EventHandler? SelectionChanged;
        public event EventHandler? SceneChanged;

        public SceneGraph()
        {
            Root = new GroupObject("Scene");
        }

        public IReadOnlyList<SceneObject> GetSelectedObjects()
        {
            return SelectedObjects;
        }

        public List<SceneObject> GetAllObjects()
        {
            return GetObjectsRecursive(Root).ToList();
        }

        public void AddObject(SceneObject obj, SceneObject? parent = null)
        {
            var targetParent = parent ?? Root;
            targetParent.AddChild(obj);
            ObjectAdded?.Invoke(this, obj);
            SceneChanged?.Invoke(this, EventArgs.Empty);
        }

        public void RemoveObject(SceneObject obj)
        {
            if (obj == Root) return;

            // Deselect if selected
            if (obj.Selected)
                Deselect(obj);

            obj.Parent?.RemoveChild(obj);
            ObjectRemoved?.Invoke(this, obj);
            SceneChanged?.Invoke(this, EventArgs.Empty);
        }

        public void Select(SceneObject obj, bool addToSelection = false)
        {
            if (!addToSelection)
            {
                foreach (var selected in SelectedObjects.ToList())
                {
                    selected.Selected = false;
                    ObjectDeselected?.Invoke(this, selected);
                }
                SelectedObjects.Clear();
            }

            if (!obj.Selected)
            {
                obj.Selected = true;
                SelectedObjects.Add(obj);
                ObjectSelected?.Invoke(this, obj);
            }

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        public void Deselect(SceneObject obj)
        {
            if (obj.Selected)
            {
                obj.Selected = false;
                SelectedObjects.Remove(obj);
                ObjectDeselected?.Invoke(this, obj);
                SelectionChanged?.Invoke(this, EventArgs.Empty);
            }
        }

        public void ClearSelection()
        {
            foreach (var obj in SelectedObjects.ToList())
            {
                obj.Selected = false;
                ObjectDeselected?.Invoke(this, obj);
            }
            SelectedObjects.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        public void SelectAll()
        {
            ClearSelection();
            SelectAllRecursive(Root);
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        private void SelectAllRecursive(SceneObject obj)
        {
            if (obj != Root)
            {
                obj.Selected = true;
                SelectedObjects.Add(obj);
            }

            foreach (var child in obj.Children)
            {
                SelectAllRecursive(child);
            }
        }

        /// <summary>
        /// Gets all objects of a specific type
        /// </summary>
        public IEnumerable<T> GetObjectsOfType<T>() where T : SceneObject
        {
            return GetObjectsRecursive(Root).OfType<T>();
        }

        /// <summary>
        /// Gets all visible objects
        /// </summary>
        public IEnumerable<SceneObject> GetVisibleObjects()
        {
            return GetObjectsRecursive(Root).Where(o => o.Visible && IsParentVisible(o));
        }

        private bool IsParentVisible(SceneObject obj)
        {
            var parent = obj.Parent;
            while (parent != null)
            {
                if (!parent.Visible) return false;
                parent = parent.Parent;
            }
            return true;
        }

        private IEnumerable<SceneObject> GetObjectsRecursive(SceneObject obj)
        {
            if (obj != Root)
                yield return obj;

            foreach (var child in obj.Children)
            {
                foreach (var descendant in GetObjectsRecursive(child))
                {
                    yield return descendant;
                }
            }
        }

        /// <summary>
        /// Finds an object by ID
        /// </summary>
        public SceneObject? FindById(int id)
        {
            return GetObjectsRecursive(Root).FirstOrDefault(o => o.Id == id);
        }

        /// <summary>
        /// Finds objects by name (case insensitive)
        /// </summary>
        public IEnumerable<SceneObject> FindByName(string name)
        {
            return GetObjectsRecursive(Root).Where(o =>
                o.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Gets the total bounds of all visible objects
        /// </summary>
        public (Vector3 min, Vector3 max) GetSceneBounds()
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            bool hasObjects = false;

            foreach (var obj in GetVisibleObjects())
            {
                var (objMin, objMax) = obj.GetWorldBounds();
                min = Vector3.ComponentMin(min, objMin);
                max = Vector3.ComponentMax(max, objMax);
                hasObjects = true;
            }

            if (!hasObjects)
            {
                return (new Vector3(-1), new Vector3(1));
            }

            return (min, max);
        }

        /// <summary>
        /// Clears all objects from the scene
        /// </summary>
        public void Clear()
        {
            ClearSelection();
            foreach (var child in Root.Children.ToList())
            {
                Root.RemoveChild(child);
                ObjectRemoved?.Invoke(this, child);
            }
            SceneChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Gets statistics about the scene
        /// </summary>
        public (int meshes, int pointClouds, int cameras, int totalVertices, int totalTriangles) GetStatistics()
        {
            int meshes = 0, pointClouds = 0, cameras = 0, totalVertices = 0, totalTriangles = 0;

            foreach (var obj in GetObjectsRecursive(Root))
            {
                switch (obj)
                {
                    case MeshObject mesh:
                        meshes++;
                        totalVertices += mesh.VertexCount;
                        totalTriangles += mesh.TriangleCount;
                        break;
                    case PointCloudObject pc:
                        pointClouds++;
                        totalVertices += pc.PointCount;
                        break;
                    case CameraObject:
                        cameras++;
                        break;
                }
            }

            return (meshes, pointClouds, cameras, totalVertices, totalTriangles);
        }
    }
}
