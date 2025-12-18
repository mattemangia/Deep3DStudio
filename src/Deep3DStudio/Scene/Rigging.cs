using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;

namespace Deep3DStudio.Scene
{
    /// <summary>
    /// Represents a single joint in a skeleton
    /// </summary>
    public class Joint
    {
        private static int _nextId = 1;

        public int Id { get; }
        public string Name { get; set; }
        public Vector3 Position { get; set; }
        public Quaternion Rotation { get; set; } = Quaternion.Identity;
        public Vector3 Scale { get; set; } = Vector3.One;

        // Hierarchy
        public Joint? Parent { get; private set; }
        private List<Joint> _children = new List<Joint>();
        public IReadOnlyList<Joint> Children => _children;
        public int ParentIndex { get; set; } = -1; // -1 for root

        // Visual properties
        public float JointSize { get; set; } = 0.02f;
        public Vector3 Color { get; set; } = new Vector3(1.0f, 0.8f, 0.0f); // Yellow
        public bool IsSelected { get; set; } = false;
        public bool IsVisible { get; set; } = true;
        public bool IsLocked { get; set; } = false;

        // Animation/pose
        public Vector3 BindPosition { get; set; }
        public Quaternion BindRotation { get; set; } = Quaternion.Identity;

        public Joint(string name, Vector3 position)
        {
            Id = _nextId++;
            Name = name;
            Position = position;
            BindPosition = position;
        }

        public void AddChild(Joint child)
        {
            if (child.Parent != null)
                child.Parent.RemoveChild(child);

            child.Parent = this;
            child.ParentIndex = this.Id;
            _children.Add(child);
        }

        public void RemoveChild(Joint child)
        {
            if (_children.Remove(child))
            {
                child.Parent = null;
                child.ParentIndex = -1;
            }
        }

        /// <summary>
        /// Gets the local transformation matrix
        /// </summary>
        public Matrix4 GetLocalTransform()
        {
            var translation = Matrix4.CreateTranslation(Position);
            var rotation = Matrix4.CreateFromQuaternion(Rotation);
            var scale = Matrix4.CreateScale(Scale);
            return scale * rotation * translation;
        }

        /// <summary>
        /// Gets the world transformation matrix
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
        /// Gets the world position
        /// </summary>
        public Vector3 GetWorldPosition()
        {
            if (Parent != null)
            {
                return Vector3.TransformPosition(Position, Parent.GetWorldTransform());
            }
            return Position;
        }

        /// <summary>
        /// Sets the world position (adjusts local position based on parent)
        /// </summary>
        public void SetWorldPosition(Vector3 worldPos)
        {
            if (Parent != null)
            {
                var parentInverse = Parent.GetWorldTransform().Inverted();
                Position = Vector3.TransformPosition(worldPos, parentInverse);
            }
            else
            {
                Position = worldPos;
            }
        }

        /// <summary>
        /// Clone this joint
        /// </summary>
        public Joint Clone()
        {
            return new Joint(Name + "_copy", Position)
            {
                Rotation = Rotation,
                Scale = Scale,
                JointSize = JointSize,
                Color = Color,
                BindPosition = BindPosition,
                BindRotation = BindRotation
            };
        }
    }

    /// <summary>
    /// Represents a bone connecting two joints
    /// </summary>
    public class Bone
    {
        public Joint StartJoint { get; }
        public Joint EndJoint { get; }
        public string Name => $"{StartJoint.Name} -> {EndJoint.Name}";

        // Visual properties
        public float Thickness { get; set; } = 0.01f;
        public Vector3 Color { get; set; } = new Vector3(0.8f, 0.8f, 0.8f);
        public bool IsSelected { get; set; } = false;
        public bool IsVisible { get; set; } = true;

        public Bone(Joint start, Joint end)
        {
            StartJoint = start;
            EndJoint = end;
        }

        /// <summary>
        /// Gets the length of the bone
        /// </summary>
        public float Length => (EndJoint.GetWorldPosition() - StartJoint.GetWorldPosition()).Length;

        /// <summary>
        /// Gets the direction vector of the bone (normalized)
        /// </summary>
        public Vector3 Direction
        {
            get
            {
                var dir = EndJoint.GetWorldPosition() - StartJoint.GetWorldPosition();
                return dir.Length > 0.0001f ? dir.Normalized() : Vector3.UnitY;
            }
        }

        /// <summary>
        /// Gets the midpoint of the bone
        /// </summary>
        public Vector3 Midpoint => (StartJoint.GetWorldPosition() + EndJoint.GetWorldPosition()) * 0.5f;
    }

    /// <summary>
    /// Complete skeleton data with joints, bones, and skinning weights
    /// </summary>
    public class SkeletonData
    {
        public string Name { get; set; } = "Skeleton";
        public List<Joint> Joints { get; } = new List<Joint>();
        public List<Bone> Bones { get; } = new List<Bone>();

        // Skinning data
        public float[,]? SkinningWeights { get; set; } // [vertexIndex, jointIndex]
        public int MaxBonesPerVertex { get; set; } = 4;

        // Events
        public event EventHandler? SkeletonChanged;
        public event EventHandler<Joint>? JointAdded;
        public event EventHandler<Joint>? JointRemoved;
        public event EventHandler<Joint>? JointSelected;

        /// <summary>
        /// Get the root joint (first joint with no parent)
        /// </summary>
        public Joint? RootJoint => Joints.FirstOrDefault(j => j.Parent == null);

        /// <summary>
        /// Add a new joint to the skeleton
        /// </summary>
        public Joint AddJoint(string name, Vector3 position, Joint? parent = null)
        {
            var joint = new Joint(name, position);

            if (parent != null)
            {
                parent.AddChild(joint);
                // Auto-create bone between parent and child
                var bone = new Bone(parent, joint);
                Bones.Add(bone);
            }

            Joints.Add(joint);
            JointAdded?.Invoke(this, joint);
            SkeletonChanged?.Invoke(this, EventArgs.Empty);

            return joint;
        }

        /// <summary>
        /// Remove a joint and its children
        /// </summary>
        public void RemoveJoint(Joint joint)
        {
            // Remove children recursively
            foreach (var child in joint.Children.ToList())
            {
                RemoveJoint(child);
            }

            // Remove bones connected to this joint
            Bones.RemoveAll(b => b.StartJoint == joint || b.EndJoint == joint);

            // Remove from parent
            joint.Parent?.RemoveChild(joint);

            Joints.Remove(joint);
            JointRemoved?.Invoke(this, joint);
            SkeletonChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Add a bone between two joints
        /// </summary>
        public Bone AddBone(Joint start, Joint end)
        {
            var bone = new Bone(start, end);
            Bones.Add(bone);
            SkeletonChanged?.Invoke(this, EventArgs.Empty);
            return bone;
        }

        /// <summary>
        /// Remove a bone
        /// </summary>
        public void RemoveBone(Bone bone)
        {
            Bones.Remove(bone);
            SkeletonChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Find joint by ID
        /// </summary>
        public Joint? FindJointById(int id)
        {
            return Joints.FirstOrDefault(j => j.Id == id);
        }

        /// <summary>
        /// Find joint by name
        /// </summary>
        public Joint? FindJointByName(string name)
        {
            return Joints.FirstOrDefault(j => j.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        }

        /// <summary>
        /// Get all joints in hierarchical order (root first, depth-first)
        /// </summary>
        public IEnumerable<Joint> GetJointsHierarchical()
        {
            var root = RootJoint;
            if (root == null) yield break;

            var stack = new Stack<Joint>();
            stack.Push(root);

            while (stack.Count > 0)
            {
                var current = stack.Pop();
                yield return current;

                // Add children in reverse order so first child is processed first
                for (int i = current.Children.Count - 1; i >= 0; i--)
                {
                    stack.Push(current.Children[i]);
                }
            }
        }

        /// <summary>
        /// Create skeleton from UniRig result
        /// </summary>
        public static SkeletonData FromRigResult(RigResult rigResult)
        {
            var skeleton = new SkeletonData();

            if (rigResult.JointPositions == null || rigResult.ParentIndices == null)
                return skeleton;

            int jointCount = rigResult.JointPositions.Length;
            var jointNames = rigResult.JointNames ?? GenerateDefaultJointNames(jointCount);

            // Create joints
            var joints = new Joint[jointCount];
            for (int i = 0; i < jointCount; i++)
            {
                joints[i] = new Joint(jointNames[i], rigResult.JointPositions[i]);
                skeleton.Joints.Add(joints[i]);
            }

            // Set up hierarchy and create bones
            for (int i = 0; i < jointCount; i++)
            {
                int parentIdx = rigResult.ParentIndices[i];
                if (parentIdx >= 0 && parentIdx < jointCount)
                {
                    joints[parentIdx].AddChild(joints[i]);
                    skeleton.Bones.Add(new Bone(joints[parentIdx], joints[i]));
                }
            }

            // Copy skinning weights
            skeleton.SkinningWeights = rigResult.SkinningWeights;

            return skeleton;
        }

        /// <summary>
        /// Convert to RigResult for export
        /// </summary>
        public RigResult ToRigResult()
        {
            var orderedJoints = GetJointsHierarchical().ToList();

            // Build index mapping
            var jointToIndex = new Dictionary<Joint, int>();
            for (int i = 0; i < orderedJoints.Count; i++)
            {
                jointToIndex[orderedJoints[i]] = i;
            }

            var result = new RigResult
            {
                JointPositions = orderedJoints.Select(j => j.GetWorldPosition()).ToArray(),
                JointNames = orderedJoints.Select(j => j.Name).ToArray(),
                ParentIndices = orderedJoints.Select(j =>
                    j.Parent != null && jointToIndex.ContainsKey(j.Parent)
                        ? jointToIndex[j.Parent]
                        : -1
                ).ToArray(),
                SkinningWeights = SkinningWeights,
                Success = true
            };

            return result;
        }

        /// <summary>
        /// Generate default joint names
        /// </summary>
        private static string[] GenerateDefaultJointNames(int count)
        {
            var names = new string[count];
            for (int i = 0; i < count; i++)
            {
                names[i] = $"Joint_{i}";
            }
            return names;
        }

        /// <summary>
        /// Create a humanoid skeleton template
        /// </summary>
        public static SkeletonData CreateHumanoidTemplate(Vector3 rootPosition, float scale = 1.0f)
        {
            var skeleton = new SkeletonData { Name = "Humanoid" };

            // Root/Hips
            var hips = skeleton.AddJoint("Hips", rootPosition);

            // Spine
            var spine = skeleton.AddJoint("Spine", rootPosition + new Vector3(0, 0.1f, 0) * scale, hips);
            var spine1 = skeleton.AddJoint("Spine1", rootPosition + new Vector3(0, 0.2f, 0) * scale, spine);
            var spine2 = skeleton.AddJoint("Spine2", rootPosition + new Vector3(0, 0.3f, 0) * scale, spine1);
            var neck = skeleton.AddJoint("Neck", rootPosition + new Vector3(0, 0.4f, 0) * scale, spine2);
            var head = skeleton.AddJoint("Head", rootPosition + new Vector3(0, 0.5f, 0) * scale, neck);

            // Left Arm
            var leftShoulder = skeleton.AddJoint("LeftShoulder", rootPosition + new Vector3(-0.1f, 0.35f, 0) * scale, spine2);
            var leftArm = skeleton.AddJoint("LeftArm", rootPosition + new Vector3(-0.15f, 0.3f, 0) * scale, leftShoulder);
            var leftForeArm = skeleton.AddJoint("LeftForeArm", rootPosition + new Vector3(-0.25f, 0.3f, 0) * scale, leftArm);
            var leftHand = skeleton.AddJoint("LeftHand", rootPosition + new Vector3(-0.35f, 0.3f, 0) * scale, leftForeArm);

            // Right Arm
            var rightShoulder = skeleton.AddJoint("RightShoulder", rootPosition + new Vector3(0.1f, 0.35f, 0) * scale, spine2);
            var rightArm = skeleton.AddJoint("RightArm", rootPosition + new Vector3(0.15f, 0.3f, 0) * scale, rightShoulder);
            var rightForeArm = skeleton.AddJoint("RightForeArm", rootPosition + new Vector3(0.25f, 0.3f, 0) * scale, rightArm);
            var rightHand = skeleton.AddJoint("RightHand", rootPosition + new Vector3(0.35f, 0.3f, 0) * scale, rightForeArm);

            // Left Leg
            var leftUpLeg = skeleton.AddJoint("LeftUpLeg", rootPosition + new Vector3(-0.1f, -0.05f, 0) * scale, hips);
            var leftLeg = skeleton.AddJoint("LeftLeg", rootPosition + new Vector3(-0.1f, -0.25f, 0) * scale, leftUpLeg);
            var leftFoot = skeleton.AddJoint("LeftFoot", rootPosition + new Vector3(-0.1f, -0.45f, 0) * scale, leftLeg);
            var leftToe = skeleton.AddJoint("LeftToeBase", rootPosition + new Vector3(-0.1f, -0.48f, 0.05f) * scale, leftFoot);

            // Right Leg
            var rightUpLeg = skeleton.AddJoint("RightUpLeg", rootPosition + new Vector3(0.1f, -0.05f, 0) * scale, hips);
            var rightLeg = skeleton.AddJoint("RightLeg", rootPosition + new Vector3(0.1f, -0.25f, 0) * scale, rightUpLeg);
            var rightFoot = skeleton.AddJoint("RightFoot", rootPosition + new Vector3(0.1f, -0.45f, 0) * scale, rightLeg);
            var rightToe = skeleton.AddJoint("RightToeBase", rootPosition + new Vector3(0.1f, -0.48f, 0.05f) * scale, rightFoot);

            return skeleton;
        }

        /// <summary>
        /// Clear all selections
        /// </summary>
        public void ClearSelection()
        {
            foreach (var joint in Joints)
                joint.IsSelected = false;
            foreach (var bone in Bones)
                bone.IsSelected = false;
        }

        /// <summary>
        /// Select a joint
        /// </summary>
        public void SelectJoint(Joint joint, bool addToSelection = false)
        {
            if (!addToSelection)
                ClearSelection();

            joint.IsSelected = true;
            JointSelected?.Invoke(this, joint);
        }

        /// <summary>
        /// Get selected joints
        /// </summary>
        public IEnumerable<Joint> GetSelectedJoints()
        {
            return Joints.Where(j => j.IsSelected);
        }

        /// <summary>
        /// Clone the skeleton
        /// </summary>
        public SkeletonData Clone()
        {
            var clone = new SkeletonData { Name = Name + "_copy" };

            // Clone joints maintaining hierarchy
            var jointMap = new Dictionary<Joint, Joint>();

            foreach (var joint in GetJointsHierarchical())
            {
                Joint? parent = joint.Parent != null && jointMap.ContainsKey(joint.Parent)
                    ? jointMap[joint.Parent]
                    : null;

                var newJoint = clone.AddJoint(joint.Name, joint.Position, parent);
                newJoint.Rotation = joint.Rotation;
                newJoint.Scale = joint.Scale;
                newJoint.JointSize = joint.JointSize;
                newJoint.Color = joint.Color;
                newJoint.BindPosition = joint.BindPosition;
                newJoint.BindRotation = joint.BindRotation;

                jointMap[joint] = newJoint;
            }

            // Clone skinning weights
            if (SkinningWeights != null)
            {
                clone.SkinningWeights = (float[,])SkinningWeights.Clone();
            }

            return clone;
        }
    }

    /// <summary>
    /// Scene object that contains a skeleton for visualization and editing
    /// </summary>
    public class SkeletonObject : SceneObject
    {
        public SkeletonData Skeleton { get; set; }

        // Associated mesh for skinning
        public MeshObject? TargetMesh { get; set; }

        // Visualization options
        public bool ShowJoints { get; set; } = true;
        public bool ShowBones { get; set; } = true;
        public bool ShowLabels { get; set; } = false;
        public float JointDisplaySize { get; set; } = 0.02f;
        public float BoneDisplayThickness { get; set; } = 0.01f;

        // Colors
        public Vector3 JointColor { get; set; } = new Vector3(1.0f, 0.8f, 0.0f);
        public Vector3 BoneColor { get; set; } = new Vector3(0.8f, 0.8f, 0.8f);
        public Vector3 SelectedColor { get; set; } = new Vector3(0.0f, 1.0f, 0.5f);

        public SkeletonObject(string name, SkeletonData skeleton) : base(name)
        {
            ObjectType = SceneObjectType.Skeleton;
            Skeleton = skeleton;
            UpdateBounds();

            skeleton.SkeletonChanged += (s, e) => UpdateBounds();
        }

        public override void UpdateBounds()
        {
            if (Skeleton == null || Skeleton.Joints.Count == 0)
            {
                BoundsMin = BoundsMax = Position;
                return;
            }

            BoundsMin = new Vector3(float.MaxValue);
            BoundsMax = new Vector3(float.MinValue);

            foreach (var joint in Skeleton.Joints)
            {
                var worldPos = joint.GetWorldPosition() + Position;
                BoundsMin = Vector3.ComponentMin(BoundsMin, worldPos);
                BoundsMax = Vector3.ComponentMax(BoundsMax, worldPos);
            }

            // Add some padding for joint display
            var padding = new Vector3(JointDisplaySize);
            BoundsMin -= padding;
            BoundsMax += padding;
        }

        public override SceneObject Clone()
        {
            return new SkeletonObject(Name + " (Copy)", Skeleton.Clone())
            {
                Position = Position,
                Rotation = Rotation,
                Scale = Scale,
                Visible = Visible,
                ShowJoints = ShowJoints,
                ShowBones = ShowBones,
                ShowLabels = ShowLabels,
                JointDisplaySize = JointDisplaySize,
                BoneDisplayThickness = BoneDisplayThickness,
                JointColor = JointColor,
                BoneColor = BoneColor,
                SelectedColor = SelectedColor
            };
        }

        /// <summary>
        /// Get joint at world position (for picking)
        /// </summary>
        public Joint? GetJointAtPosition(Vector3 worldPos, float threshold = 0.05f)
        {
            Joint? closest = null;
            float closestDist = threshold;

            foreach (var joint in Skeleton.Joints)
            {
                var jointWorldPos = joint.GetWorldPosition() + Position;
                float dist = (jointWorldPos - worldPos).Length;
                if (dist < closestDist)
                {
                    closestDist = dist;
                    closest = joint;
                }
            }

            return closest;
        }

        /// <summary>
        /// Get bone at world position (for picking)
        /// </summary>
        public Bone? GetBoneAtPosition(Vector3 worldPos, float threshold = 0.03f)
        {
            Bone? closest = null;
            float closestDist = threshold;

            foreach (var bone in Skeleton.Bones)
            {
                var start = bone.StartJoint.GetWorldPosition() + Position;
                var end = bone.EndJoint.GetWorldPosition() + Position;

                // Point-to-line distance
                var line = end - start;
                float t = Vector3.Dot(worldPos - start, line) / line.LengthSquared;
                t = Math.Clamp(t, 0, 1);
                var closestPoint = start + line * t;

                float dist = (worldPos - closestPoint).Length;
                if (dist < closestDist)
                {
                    closestDist = dist;
                    closest = bone;
                }
            }

            return closest;
        }
    }
}
