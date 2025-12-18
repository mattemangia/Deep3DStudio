using System;
using System.Collections.Generic;
using System.Linq;
using Gtk;
using OpenTK.Mathematics;
using Deep3DStudio.Scene;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// Panel for managing skeleton rigging with tree view
    /// </summary>
    public class RiggingPanel : Box
    {
        private TreeView _treeView;
        private TreeStore _treeStore;
        private SkeletonObject? _skeletonObject;

        // Property editors
        private Entry _nameEntry;
        private SpinButton _posXSpin;
        private SpinButton _posYSpin;
        private SpinButton _posZSpin;
        private SpinButton _jointSizeSpin;
        private ColorButton _jointColorBtn;

        // Column indices
        private const int COL_ICON = 0;
        private const int COL_NAME = 1;
        private const int COL_VISIBLE = 2;
        private const int COL_LOCKED = 3;
        private const int COL_JOINT_ID = 4;
        private const int COL_TYPE = 5; // "Joint" or "Bone"

        // Events
        public event EventHandler<Joint>? JointSelected;
        public event EventHandler<Joint>? JointTransformChanged;
        public event EventHandler? SkeletonChanged;
        public event System.Action? RequestAddJoint;
        public event System.Action? RequestAddBone;
        public event System.Action? RequestDeleteSelected;
        public event System.Action? RequestImportUniRig;
        public event System.Action? RequestExportRig;
        public event System.Action? RequestCreateHumanoid;
        public event System.Action? RefreshViewport;

        private bool _isUpdating = false;

        public RiggingPanel() : base(Orientation.Vertical, 5)
        {
            Margin = 5;
            BuildUI();
        }

        private void BuildUI()
        {
            // Header
            var headerBox = new Box(Orientation.Horizontal, 5);
            var headerLabel = new Label("Rigging Tool");
            headerLabel.Attributes = new Pango.AttrList();
            headerLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            headerBox.PackStart(headerLabel, false, false, 5);
            PackStart(headerBox, false, false, 0);

            // Toolbar
            var toolbar = CreateToolbar();
            PackStart(toolbar, false, false, 0);

            // TreeView in scrolled window
            var scrolledWindow = new ScrolledWindow();
            scrolledWindow.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            scrolledWindow.ShadowType = ShadowType.In;
            scrolledWindow.SetSizeRequest(-1, 200);

            // Store: Icon, Name, Visible, Locked, JointId, Type
            _treeStore = new TreeStore(
                typeof(Gdk.Pixbuf),  // Icon
                typeof(string),      // Name
                typeof(bool),        // Visible
                typeof(bool),        // Locked
                typeof(int),         // Joint ID
                typeof(string)       // Type
            );

            _treeView = new TreeView(_treeStore);
            _treeView.HeadersVisible = true;
            _treeView.EnableSearch = true;
            _treeView.SearchColumn = COL_NAME;
            _treeView.Selection.Mode = SelectionMode.Multiple;

            // Icon + Name column
            var nameColumn = new TreeViewColumn();
            nameColumn.Title = "Joint/Bone";
            nameColumn.Resizable = true;
            nameColumn.Expand = true;

            var iconRenderer = new CellRendererPixbuf();
            nameColumn.PackStart(iconRenderer, false);
            nameColumn.AddAttribute(iconRenderer, "pixbuf", COL_ICON);

            var nameRenderer = new CellRendererText();
            nameRenderer.Editable = true;
            nameRenderer.Edited += OnNameEdited;
            nameColumn.PackStart(nameRenderer, true);
            nameColumn.AddAttribute(nameRenderer, "text", COL_NAME);

            _treeView.AppendColumn(nameColumn);

            // Visible column
            var visibleColumn = new TreeViewColumn();
            visibleColumn.Title = "V";
            visibleColumn.MinWidth = 30;

            var visibleRenderer = new CellRendererToggle();
            visibleRenderer.Toggled += OnVisibleToggled;
            visibleColumn.PackStart(visibleRenderer, false);
            visibleColumn.AddAttribute(visibleRenderer, "active", COL_VISIBLE);

            _treeView.AppendColumn(visibleColumn);

            // Locked column
            var lockedColumn = new TreeViewColumn();
            lockedColumn.Title = "L";
            lockedColumn.MinWidth = 30;

            var lockedRenderer = new CellRendererToggle();
            lockedRenderer.Toggled += OnLockedToggled;
            lockedColumn.PackStart(lockedRenderer, false);
            lockedColumn.AddAttribute(lockedRenderer, "active", COL_LOCKED);

            _treeView.AppendColumn(lockedColumn);

            // Events
            _treeView.Selection.Changed += OnSelectionChanged;
            _treeView.RowActivated += OnRowActivated;
            _treeView.ButtonPressEvent += OnButtonPress;

            scrolledWindow.Add(_treeView);
            PackStart(scrolledWindow, true, true, 0);

            // Properties panel
            var propertiesFrame = new Frame("Joint Properties");
            var propertiesBox = CreatePropertiesPanel();
            propertiesFrame.Add(propertiesBox);
            PackStart(propertiesFrame, false, false, 5);

            // Preset buttons
            var presetFrame = new Frame("Presets");
            var presetBox = CreatePresetPanel();
            presetFrame.Add(presetBox);
            PackStart(presetFrame, false, false, 5);
        }

        private Widget CreateToolbar()
        {
            var toolbar = new Box(Orientation.Horizontal, 2);
            toolbar.Margin = 2;

            var addJointBtn = new Button();
            addJointBtn.TooltipText = "Add Joint";
            addJointBtn.Add(new Image(Stock.Add, IconSize.SmallToolbar));
            addJointBtn.Clicked += (s, e) => RequestAddJoint?.Invoke();
            toolbar.PackStart(addJointBtn, false, false, 0);

            var addBoneBtn = new Button();
            addBoneBtn.TooltipText = "Connect Selected (Add Bone)";
            addBoneBtn.Add(new Image(Stock.Connect, IconSize.SmallToolbar));
            addBoneBtn.Clicked += (s, e) => RequestAddBone?.Invoke();
            toolbar.PackStart(addBoneBtn, false, false, 0);

            var deleteBtn = new Button();
            deleteBtn.TooltipText = "Delete Selected";
            deleteBtn.Add(new Image(Stock.Delete, IconSize.SmallToolbar));
            deleteBtn.Clicked += (s, e) => RequestDeleteSelected?.Invoke();
            toolbar.PackStart(deleteBtn, false, false, 0);

            toolbar.PackStart(new Separator(Orientation.Vertical), false, false, 5);

            var importBtn = new Button();
            importBtn.TooltipText = "Import from UniRig";
            importBtn.Add(new Image(Stock.Open, IconSize.SmallToolbar));
            importBtn.Clicked += (s, e) => RequestImportUniRig?.Invoke();
            toolbar.PackStart(importBtn, false, false, 0);

            var exportBtn = new Button();
            exportBtn.TooltipText = "Export Rigged Mesh";
            exportBtn.Add(new Image(Stock.Save, IconSize.SmallToolbar));
            exportBtn.Clicked += (s, e) => RequestExportRig?.Invoke();
            toolbar.PackStart(exportBtn, false, false, 0);

            toolbar.PackStart(new Separator(Orientation.Vertical), false, false, 5);

            var collapseBtn = new Button();
            collapseBtn.TooltipText = "Collapse All";
            collapseBtn.Add(new Image(Stock.GotoTop, IconSize.SmallToolbar));
            collapseBtn.Clicked += (s, e) => _treeView.CollapseAll();
            toolbar.PackStart(collapseBtn, false, false, 0);

            var expandBtn = new Button();
            expandBtn.TooltipText = "Expand All";
            expandBtn.Add(new Image(Stock.GotoBottom, IconSize.SmallToolbar));
            expandBtn.Clicked += (s, e) => _treeView.ExpandAll();
            toolbar.PackStart(expandBtn, false, false, 0);

            return toolbar;
        }

        private Box CreatePropertiesPanel()
        {
            var box = new Box(Orientation.Vertical, 5);
            box.Margin = 5;

            // Name
            var nameBox = new Box(Orientation.Horizontal, 5);
            nameBox.PackStart(new Label("Name:"), false, false, 0);
            _nameEntry = new Entry();
            _nameEntry.Changed += OnPropertyChanged;
            nameBox.PackStart(_nameEntry, true, true, 0);
            box.PackStart(nameBox, false, false, 0);

            // Position
            var posLabel = new Label("Position:");
            posLabel.Halign = Align.Start;
            box.PackStart(posLabel, false, false, 0);

            var posBox = new Box(Orientation.Horizontal, 3);

            posBox.PackStart(new Label("X:"), false, false, 0);
            _posXSpin = new SpinButton(-100, 100, 0.01);
            _posXSpin.Digits = 4;
            _posXSpin.ValueChanged += OnPropertyChanged;
            posBox.PackStart(_posXSpin, true, true, 0);

            posBox.PackStart(new Label("Y:"), false, false, 0);
            _posYSpin = new SpinButton(-100, 100, 0.01);
            _posYSpin.Digits = 4;
            _posYSpin.ValueChanged += OnPropertyChanged;
            posBox.PackStart(_posYSpin, true, true, 0);

            posBox.PackStart(new Label("Z:"), false, false, 0);
            _posZSpin = new SpinButton(-100, 100, 0.01);
            _posZSpin.Digits = 4;
            _posZSpin.ValueChanged += OnPropertyChanged;
            posBox.PackStart(_posZSpin, true, true, 0);

            box.PackStart(posBox, false, false, 0);

            // Joint size and color
            var visualBox = new Box(Orientation.Horizontal, 5);

            visualBox.PackStart(new Label("Size:"), false, false, 0);
            _jointSizeSpin = new SpinButton(0.001, 1.0, 0.005);
            _jointSizeSpin.Digits = 3;
            _jointSizeSpin.Value = 0.02;
            _jointSizeSpin.ValueChanged += OnPropertyChanged;
            visualBox.PackStart(_jointSizeSpin, false, false, 0);

            visualBox.PackStart(new Label("Color:"), false, false, 0);
            _jointColorBtn = new ColorButton();
            _jointColorBtn.Color = new Gdk.Color(255, 200, 0);
            _jointColorBtn.ColorSet += OnPropertyChanged;
            visualBox.PackStart(_jointColorBtn, false, false, 0);

            box.PackStart(visualBox, false, false, 0);

            return box;
        }

        private Box CreatePresetPanel()
        {
            var box = new Box(Orientation.Vertical, 5);
            box.Margin = 5;

            var humanoidBtn = new Button("Create Humanoid Template");
            humanoidBtn.Clicked += (s, e) => RequestCreateHumanoid?.Invoke();
            box.PackStart(humanoidBtn, false, false, 0);

            var mirrorBtn = new Button("Mirror Skeleton (X)");
            mirrorBtn.Clicked += OnMirrorSkeleton;
            box.PackStart(mirrorBtn, false, false, 0);

            var autoWeightsBtn = new Button("Auto-Calculate Weights");
            autoWeightsBtn.Clicked += OnAutoWeights;
            autoWeightsBtn.TooltipText = "Calculate skinning weights from mesh (requires selected mesh)";
            box.PackStart(autoWeightsBtn, false, false, 0);

            return box;
        }

        /// <summary>
        /// Set the skeleton object to edit
        /// </summary>
        public void SetSkeleton(SkeletonObject? skeletonObj)
        {
            if (_skeletonObject != null)
            {
                _skeletonObject.Skeleton.SkeletonChanged -= OnSkeletonDataChanged;
            }

            _skeletonObject = skeletonObj;

            if (_skeletonObject != null)
            {
                _skeletonObject.Skeleton.SkeletonChanged += OnSkeletonDataChanged;
            }

            RefreshTree();
        }

        /// <summary>
        /// Get the current skeleton object
        /// </summary>
        public SkeletonObject? GetSkeleton() => _skeletonObject;

        /// <summary>
        /// Refresh the tree view from skeleton data
        /// </summary>
        public void RefreshTree()
        {
            _treeStore.Clear();

            if (_skeletonObject?.Skeleton == null) return;

            // Add joints hierarchically
            var root = _skeletonObject.Skeleton.RootJoint;
            if (root != null)
            {
                AddJointToTree(root, TreeIter.Zero, true);
            }

            _treeView.ExpandAll();
        }

        private void AddJointToTree(Joint joint, TreeIter parentIter, bool isRoot = false)
        {
            TreeIter iter;

            if (isRoot)
            {
                iter = _treeStore.AppendValues(
                    GetJointIcon(),
                    joint.Name,
                    joint.IsVisible,
                    joint.IsLocked,
                    joint.Id,
                    "Joint"
                );
            }
            else
            {
                iter = _treeStore.AppendValues(
                    parentIter,
                    GetJointIcon(),
                    joint.Name,
                    joint.IsVisible,
                    joint.IsLocked,
                    joint.Id,
                    "Joint"
                );
            }

            // Add children
            foreach (var child in joint.Children)
            {
                AddJointToTree(child, iter);
            }
        }

        private Gdk.Pixbuf? GetJointIcon()
        {
            try
            {
                int size = 16;
                var pixbuf = new Gdk.Pixbuf(Gdk.Colorspace.Rgb, true, 8, size, size);
                uint pixel = ((uint)255 << 24) | ((uint)200 << 16) | ((uint)0 << 8) | 255; // Yellow
                pixbuf.Fill(pixel);
                return pixbuf;
            }
            catch
            {
                return null;
            }
        }

        private Joint? GetJointFromIter(TreeIter iter)
        {
            if (_skeletonObject?.Skeleton == null) return null;

            int id = (int)_treeStore.GetValue(iter, COL_JOINT_ID);
            return _skeletonObject.Skeleton.FindJointById(id);
        }

        /// <summary>
        /// Get currently selected joints
        /// </summary>
        public IEnumerable<Joint> GetSelectedJoints()
        {
            var selectedPaths = _treeView.Selection.GetSelectedRows();
            foreach (var path in selectedPaths)
            {
                if (_treeStore.GetIter(out TreeIter iter, path))
                {
                    var joint = GetJointFromIter(iter);
                    if (joint != null)
                        yield return joint;
                }
            }
        }

        /// <summary>
        /// Select a joint in the tree
        /// </summary>
        public void SelectJoint(Joint joint)
        {
            SelectJointRecursive(TreeIter.Zero, joint, true);
        }

        private bool SelectJointRecursive(TreeIter parentIter, Joint target, bool isRoot)
        {
            TreeIter iter;
            bool hasChildren;

            if (isRoot)
            {
                hasChildren = _treeStore.GetIterFirst(out iter);
            }
            else
            {
                hasChildren = _treeStore.IterChildren(out iter, parentIter);
            }

            while (hasChildren)
            {
                int id = (int)_treeStore.GetValue(iter, COL_JOINT_ID);
                if (id == target.Id)
                {
                    var path = _treeStore.GetPath(iter);
                    _treeView.ExpandToPath(path);
                    _treeView.Selection.SelectPath(path);
                    _treeView.ScrollToCell(path, null, true, 0.5f, 0);
                    return true;
                }

                if (SelectJointRecursive(iter, target, false))
                    return true;

                hasChildren = _treeStore.IterNext(ref iter);
            }

            return false;
        }

        private void UpdatePropertiesPanel(Joint? joint)
        {
            _isUpdating = true;

            if (joint == null)
            {
                _nameEntry.Text = "";
                _posXSpin.Value = 0;
                _posYSpin.Value = 0;
                _posZSpin.Value = 0;
                _jointSizeSpin.Value = 0.02;
            }
            else
            {
                _nameEntry.Text = joint.Name;
                _posXSpin.Value = joint.Position.X;
                _posYSpin.Value = joint.Position.Y;
                _posZSpin.Value = joint.Position.Z;
                _jointSizeSpin.Value = joint.JointSize;

                var c = joint.Color;
                _jointColorBtn.Color = new Gdk.Color(
                    (byte)(c.X * 255),
                    (byte)(c.Y * 255),
                    (byte)(c.Z * 255)
                );
            }

            _isUpdating = false;
        }

        #region Event Handlers

        private void OnNameEdited(object o, EditedArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var joint = GetJointFromIter(iter);
                if (joint != null)
                {
                    joint.Name = args.NewText;
                    _treeStore.SetValue(iter, COL_NAME, args.NewText);
                    SkeletonChanged?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        private void OnVisibleToggled(object o, ToggledArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var joint = GetJointFromIter(iter);
                if (joint != null)
                {
                    joint.IsVisible = !joint.IsVisible;
                    _treeStore.SetValue(iter, COL_VISIBLE, joint.IsVisible);
                    RefreshViewport?.Invoke();
                }
            }
        }

        private void OnLockedToggled(object o, ToggledArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var joint = GetJointFromIter(iter);
                if (joint != null)
                {
                    joint.IsLocked = !joint.IsLocked;
                    _treeStore.SetValue(iter, COL_LOCKED, joint.IsLocked);
                }
            }
        }

        private void OnSelectionChanged(object? sender, EventArgs e)
        {
            if (_skeletonObject?.Skeleton == null) return;

            _skeletonObject.Skeleton.ClearSelection();

            var selectedJoints = GetSelectedJoints().ToList();

            foreach (var joint in selectedJoints)
            {
                joint.IsSelected = true;
            }

            if (selectedJoints.Count == 1)
            {
                UpdatePropertiesPanel(selectedJoints[0]);
                JointSelected?.Invoke(this, selectedJoints[0]);
            }
            else
            {
                UpdatePropertiesPanel(null);
            }

            RefreshViewport?.Invoke();
        }

        private void OnRowActivated(object o, RowActivatedArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, args.Path))
            {
                var joint = GetJointFromIter(iter);
                if (joint != null)
                {
                    JointSelected?.Invoke(this, joint);
                }
            }
        }

        [GLib.ConnectBefore]
        private void OnButtonPress(object o, ButtonPressEventArgs args)
        {
            // Right-click context menu
            if (args.Event.Button == 3)
            {
                var path = new TreePath();
                if (_treeView.GetPathAtPos((int)args.Event.X, (int)args.Event.Y, out path))
                {
                    _treeView.Selection.SelectPath(path);
                }

                ShowContextMenu(args.Event);
                args.RetVal = true;
            }
        }

        private void ShowContextMenu(Gdk.EventButton evt)
        {
            var menu = new Menu();

            var addJointItem = new MenuItem("Add Child Joint");
            addJointItem.Activated += (s, e) => RequestAddJoint?.Invoke();
            menu.Append(addJointItem);

            var connectItem = new MenuItem("Connect to Selected");
            connectItem.Activated += (s, e) => RequestAddBone?.Invoke();
            menu.Append(connectItem);

            menu.Append(new SeparatorMenuItem());

            var duplicateItem = new MenuItem("Duplicate");
            duplicateItem.Activated += OnDuplicateJoint;
            menu.Append(duplicateItem);

            var deleteItem = new MenuItem("Delete");
            deleteItem.Activated += (s, e) => RequestDeleteSelected?.Invoke();
            menu.Append(deleteItem);

            menu.Append(new SeparatorMenuItem());

            var selectChildrenItem = new MenuItem("Select Children");
            selectChildrenItem.Activated += OnSelectChildren;
            menu.Append(selectChildrenItem);

            var selectBranchItem = new MenuItem("Select Branch");
            selectBranchItem.Activated += OnSelectBranch;
            menu.Append(selectBranchItem);

            menu.ShowAll();
            menu.Popup();
        }

        private void OnPropertyChanged(object? sender, EventArgs e)
        {
            if (_isUpdating) return;

            var selectedJoints = GetSelectedJoints().ToList();
            if (selectedJoints.Count != 1) return;

            var joint = selectedJoints[0];

            joint.Name = _nameEntry.Text;
            joint.Position = new Vector3(
                (float)_posXSpin.Value,
                (float)_posYSpin.Value,
                (float)_posZSpin.Value
            );
            joint.JointSize = (float)_jointSizeSpin.Value;

            var c = _jointColorBtn.Color;
            joint.Color = new Vector3(c.Red / 65535f, c.Green / 65535f, c.Blue / 65535f);

            JointTransformChanged?.Invoke(this, joint);
            RefreshViewport?.Invoke();
        }

        private void OnSkeletonDataChanged(object? sender, EventArgs e)
        {
            RefreshTree();
        }

        private void OnDuplicateJoint(object? sender, EventArgs e)
        {
            if (_skeletonObject?.Skeleton == null) return;

            var selectedJoints = GetSelectedJoints().ToList();
            foreach (var joint in selectedJoints)
            {
                var clone = joint.Clone();
                clone.Position += new Vector3(0.05f, 0, 0);
                _skeletonObject.Skeleton.Joints.Add(clone);

                if (joint.Parent != null)
                {
                    joint.Parent.AddChild(clone);
                }
            }

            RefreshTree();
            SkeletonChanged?.Invoke(this, EventArgs.Empty);
            RefreshViewport?.Invoke();
        }

        private void OnSelectChildren(object? sender, EventArgs e)
        {
            var selectedJoints = GetSelectedJoints().ToList();

            foreach (var joint in selectedJoints)
            {
                foreach (var child in joint.Children)
                {
                    SelectJoint(child);
                }
            }
        }

        private void OnSelectBranch(object? sender, EventArgs e)
        {
            var selectedJoints = GetSelectedJoints().ToList();

            foreach (var joint in selectedJoints)
            {
                SelectJointAndDescendants(joint);
            }
        }

        private void SelectJointAndDescendants(Joint joint)
        {
            SelectJoint(joint);
            foreach (var child in joint.Children)
            {
                SelectJointAndDescendants(child);
            }
        }

        private void OnMirrorSkeleton(object? sender, EventArgs e)
        {
            if (_skeletonObject?.Skeleton == null) return;

            var skeleton = _skeletonObject.Skeleton;
            var newJoints = new List<Joint>();

            // Mirror joints across X axis
            foreach (var joint in skeleton.Joints.ToList())
            {
                // Check if this is a "Left" joint that needs mirroring
                if (joint.Name.Contains("Left"))
                {
                    var mirroredName = joint.Name.Replace("Left", "Right");

                    // Check if mirror doesn't already exist
                    if (skeleton.FindJointByName(mirroredName) == null)
                    {
                        var mirrorPos = new Vector3(-joint.Position.X, joint.Position.Y, joint.Position.Z);
                        var mirrorJoint = skeleton.AddJoint(mirroredName, mirrorPos);

                        // Try to find mirrored parent
                        if (joint.Parent != null)
                        {
                            var mirrorParentName = joint.Parent.Name.Replace("Left", "Right");
                            var mirrorParent = skeleton.FindJointByName(mirrorParentName);
                            if (mirrorParent != null)
                            {
                                mirrorParent.AddChild(mirrorJoint);
                            }
                        }
                    }
                }
            }

            RefreshTree();
            SkeletonChanged?.Invoke(this, EventArgs.Empty);
            RefreshViewport?.Invoke();
        }

        private void OnAutoWeights(object? sender, EventArgs e)
        {
            if (_skeletonObject?.Skeleton == null || _skeletonObject.TargetMesh == null)
            {
                Console.WriteLine("Auto-weights requires both a skeleton and a target mesh");
                return;
            }

            var mesh = _skeletonObject.TargetMesh.MeshData;
            var skeleton = _skeletonObject.Skeleton;

            if (mesh == null || skeleton.Joints.Count == 0) return;

            int vertexCount = mesh.Vertices.Count;
            int jointCount = skeleton.Joints.Count;

            // Initialize weights
            var weights = new float[vertexCount, jointCount];

            // Simple distance-based weight calculation
            var joints = skeleton.GetJointsHierarchical().ToList();
            var jointIndexMap = new Dictionary<Joint, int>();
            for (int j = 0; j < joints.Count; j++)
            {
                jointIndexMap[joints[j]] = j;
            }

            for (int v = 0; v < vertexCount; v++)
            {
                var vertexPos = mesh.Vertices[v];
                var distances = new List<(int jointIdx, float dist)>();

                for (int j = 0; j < joints.Count; j++)
                {
                    float dist = (vertexPos - joints[j].GetWorldPosition()).Length;
                    distances.Add((j, dist));
                }

                // Sort by distance and take closest N joints
                distances.Sort((a, b) => a.dist.CompareTo(b.dist));

                int maxBones = skeleton.MaxBonesPerVertex;
                float totalWeight = 0;

                for (int i = 0; i < Math.Min(maxBones, distances.Count); i++)
                {
                    float dist = distances[i].dist;
                    float weight = 1.0f / (dist + 0.001f); // Inverse distance
                    weights[v, distances[i].jointIdx] = weight;
                    totalWeight += weight;
                }

                // Normalize weights
                if (totalWeight > 0)
                {
                    for (int j = 0; j < jointCount; j++)
                    {
                        weights[v, j] /= totalWeight;
                    }
                }
            }

            skeleton.SkinningWeights = weights;
            Console.WriteLine($"Calculated skinning weights for {vertexCount} vertices and {jointCount} joints");

            SkeletonChanged?.Invoke(this, EventArgs.Empty);
        }

        #endregion
    }
}
