using System;
using System.Collections.Generic;
using System.Linq;
using Gtk;
using Deep3DStudio.Scene;
using Deep3DStudio.Icons;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// TreeView panel for displaying and managing scene objects
    /// </summary>
    public class SceneTreeView : Box
    {
        private TreeView _treeView;
        private TreeStore _treeStore;
        private SceneGraph? _sceneGraph;

        // Column indices
        private const int COL_ICON = 0;
        private const int COL_NAME = 1;
        private const int COL_VISIBLE = 2;
        private const int COL_LOCKED = 3;
        private const int COL_OBJECT_ID = 4;
        private const int COL_TYPE_NAME = 5;

        // Events
        public event EventHandler<SceneObject>? ObjectSelected;
        public event EventHandler<SceneObject>? ObjectDoubleClicked;
        public event EventHandler<(SceneObject obj, string action)>? ObjectActionRequested;

        public SceneTreeView() : base(Orientation.Vertical, 5)
        {
            BuildUI();
        }

        private void BuildUI()
        {
            // Header
            var headerBox = new Box(Orientation.Horizontal, 5);
            var headerLabel = new Label("Scene Objects");
            headerLabel.Attributes = new Pango.AttrList();
            headerLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            headerBox.PackStart(headerLabel, false, false, 5);
            PackStart(headerBox, false, false, 0);

            // Toolbar
            var toolbar = CreateToolbar();
            PackStart(toolbar, false, false, 0);

            // TreeView
            var scrolledWindow = new ScrolledWindow();
            scrolledWindow.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            scrolledWindow.ShadowType = ShadowType.In;

            // Store: Icon, Name, Visible, Locked, ObjectId, TypeName
            _treeStore = new TreeStore(
                typeof(Gdk.Pixbuf),  // Icon
                typeof(string),      // Name
                typeof(bool),        // Visible
                typeof(bool),        // Locked
                typeof(int),         // Object ID
                typeof(string)       // Type name
            );

            _treeView = new TreeView(_treeStore);
            _treeView.HeadersVisible = true;
            _treeView.EnableSearch = true;
            _treeView.SearchColumn = COL_NAME;
            _treeView.Selection.Mode = SelectionMode.Multiple;

            // Icon + Name column
            var nameColumn = new TreeViewColumn();
            nameColumn.Title = "Name";
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

            // Type column
            var typeColumn = new TreeViewColumn();
            typeColumn.Title = "Type";
            typeColumn.Resizable = true;

            var typeRenderer = new CellRendererText();
            typeColumn.PackStart(typeRenderer, true);
            typeColumn.AddAttribute(typeRenderer, "text", COL_TYPE_NAME);

            _treeView.AppendColumn(typeColumn);

            // Events
            _treeView.Selection.Changed += OnSelectionChanged;
            _treeView.RowActivated += OnRowActivated;
            _treeView.ButtonPressEvent += OnButtonPress;

            scrolledWindow.Add(_treeView);
            PackStart(scrolledWindow, true, true, 0);

            // Info panel at bottom
            var infoBox = CreateInfoPanel();
            PackStart(infoBox, false, false, 5);
        }

        private Widget CreateToolbar()
        {
            var toolbar = new Box(Orientation.Horizontal, 2);
            toolbar.Margin = 2;

            var addGroupBtn = new Button();
            addGroupBtn.TooltipText = "Add Group";
            addGroupBtn.Add(new Image(Stock.Directory, IconSize.SmallToolbar));
            addGroupBtn.Clicked += (s, e) => ObjectActionRequested?.Invoke(this, (null!, "add_group"));
            toolbar.PackStart(addGroupBtn, false, false, 0);

            var deleteBtn = new Button();
            deleteBtn.TooltipText = "Delete Selected";
            deleteBtn.Add(new Image(Stock.Delete, IconSize.SmallToolbar));
            deleteBtn.Clicked += OnDeleteClicked;
            toolbar.PackStart(deleteBtn, false, false, 0);

            var duplicateBtn = new Button();
            duplicateBtn.TooltipText = "Duplicate Selected";
            duplicateBtn.Add(new Image(Stock.Copy, IconSize.SmallToolbar));
            duplicateBtn.Clicked += OnDuplicateClicked;
            toolbar.PackStart(duplicateBtn, false, false, 0);

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

        private Box CreateInfoPanel()
        {
            var box = new Box(Orientation.Vertical, 2);
            box.Margin = 5;

            var separator = new Separator(Orientation.Horizontal);
            box.PackStart(separator, false, false, 0);

            var infoLabel = new Label("Select an object to see details");
            infoLabel.Halign = Align.Start;
            infoLabel.Name = "info_label";
            box.PackStart(infoLabel, false, false, 0);

            return box;
        }

        /// <summary>
        /// Binds the tree view to a scene graph
        /// </summary>
        public void SetSceneGraph(SceneGraph sceneGraph)
        {
            // Unbind old
            if (_sceneGraph != null)
            {
                _sceneGraph.ObjectAdded -= OnSceneObjectAdded;
                _sceneGraph.ObjectRemoved -= OnSceneObjectRemoved;
                _sceneGraph.SelectionChanged -= OnSceneSelectionChanged;
                _sceneGraph.SceneChanged -= OnSceneChanged;
            }

            _sceneGraph = sceneGraph;

            // Bind new
            if (_sceneGraph != null)
            {
                _sceneGraph.ObjectAdded += OnSceneObjectAdded;
                _sceneGraph.ObjectRemoved += OnSceneObjectRemoved;
                _sceneGraph.SelectionChanged += OnSceneSelectionChanged;
                _sceneGraph.SceneChanged += OnSceneChanged;
            }

            RefreshTree();
        }

        /// <summary>
        /// Refreshes the entire tree from the scene graph
        /// </summary>
        public void RefreshTree()
        {
            _treeStore.Clear();

            if (_sceneGraph == null) return;

            AddObjectToTree(_sceneGraph.Root, TreeIter.Zero, true);
            _treeView.ExpandAll();
        }

        private void AddObjectToTree(SceneObject obj, TreeIter parentIter, bool isRoot = false)
        {
            if (isRoot)
            {
                // Add children of root directly
                foreach (var child in obj.Children)
                {
                    var iter = _treeStore.AppendValues(
                        GetIconForType(child.ObjectType),
                        child.Name,
                        child.Visible,
                        child.Locked,
                        child.Id,
                        GetTypeName(child.ObjectType)
                    );

                    foreach (var grandChild in child.Children)
                    {
                        AddObjectToTree(grandChild, iter);
                    }
                }
            }
            else
            {
                var iter = _treeStore.AppendValues(
                    parentIter,
                    GetIconForType(obj.ObjectType),
                    obj.Name,
                    obj.Visible,
                    obj.Locked,
                    obj.Id,
                    GetTypeName(obj.ObjectType)
                );

                foreach (var child in obj.Children)
                {
                    AddObjectToTree(child, iter);
                }
            }
        }

        private Gdk.Pixbuf? GetIconForType(SceneObjectType type)
        {
            // Return a simple colored rectangle based on type
            // In a real implementation, you'd load proper icons
            try
            {
                int size = 16;
                var color = type switch
                {
                    SceneObjectType.Mesh => new Gdk.Color(100, 149, 237),        // Cornflower blue
                    SceneObjectType.PointCloud => new Gdk.Color(50, 205, 50),    // Lime green
                    SceneObjectType.Camera => new Gdk.Color(255, 165, 0),        // Orange
                    SceneObjectType.Group => new Gdk.Color(169, 169, 169),       // Dark gray
                    SceneObjectType.Light => new Gdk.Color(255, 255, 0),         // Yellow
                    _ => new Gdk.Color(128, 128, 128)                            // Gray
                };

                // Create a simple colored pixbuf
                var pixbuf = new Gdk.Pixbuf(Gdk.Colorspace.Rgb, true, 8, size, size);
                uint pixel = ((uint)color.Red << 24) | ((uint)color.Green << 16) | ((uint)color.Blue << 8) | 255;
                pixbuf.Fill(pixel);

                return pixbuf;
            }
            catch
            {
                return null;
            }
        }

        private string GetTypeName(SceneObjectType type)
        {
            return type switch
            {
                SceneObjectType.Mesh => "Mesh",
                SceneObjectType.PointCloud => "Points",
                SceneObjectType.Camera => "Camera",
                SceneObjectType.Group => "Group",
                SceneObjectType.Light => "Light",
                SceneObjectType.Annotation => "Note",
                _ => "Object"
            };
        }

        private SceneObject? GetObjectFromIter(TreeIter iter)
        {
            if (_sceneGraph == null) return null;

            int id = (int)_treeStore.GetValue(iter, COL_OBJECT_ID);
            return _sceneGraph.FindById(id);
        }

        #region Event Handlers

        private void OnNameEdited(object o, EditedArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var obj = GetObjectFromIter(iter);
                if (obj != null)
                {
                    obj.Name = args.NewText;
                    _treeStore.SetValue(iter, COL_NAME, args.NewText);
                }
            }
        }

        private void OnVisibleToggled(object o, ToggledArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var obj = GetObjectFromIter(iter);
                if (obj != null)
                {
                    obj.Visible = !obj.Visible;
                    _treeStore.SetValue(iter, COL_VISIBLE, obj.Visible);
                    ObjectActionRequested?.Invoke(this, (obj, "visibility_changed"));
                }
            }
        }

        private void OnLockedToggled(object o, ToggledArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, new TreePath(args.Path)))
            {
                var obj = GetObjectFromIter(iter);
                if (obj != null)
                {
                    obj.Locked = !obj.Locked;
                    _treeStore.SetValue(iter, COL_LOCKED, obj.Locked);
                }
            }
        }

        private void OnSelectionChanged(object? sender, EventArgs e)
        {
            if (_sceneGraph == null) return;

            var selectedPaths = _treeView.Selection.GetSelectedRows();
            _sceneGraph.ClearSelection();

            foreach (var path in selectedPaths)
            {
                if (_treeStore.GetIter(out TreeIter iter, path))
                {
                    var obj = GetObjectFromIter(iter);
                    if (obj != null)
                    {
                        _sceneGraph.Select(obj, true);
                        ObjectSelected?.Invoke(this, obj);
                    }
                }
            }

            UpdateInfoPanel();
        }

        private void OnRowActivated(object o, RowActivatedArgs args)
        {
            if (_treeStore.GetIter(out TreeIter iter, args.Path))
            {
                var obj = GetObjectFromIter(iter);
                if (obj != null)
                {
                    ObjectDoubleClicked?.Invoke(this, obj);
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

            var selectedObjects = GetSelectedObjects().ToList();

            // Transform submenu
            var transformMenu = new Menu();
            var transformItem = new MenuItem("Transform");
            transformItem.Submenu = transformMenu;

            var moveItem = new MenuItem("Move...");
            moveItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (selectedObjects.FirstOrDefault()!, "move"));
            transformMenu.Append(moveItem);

            var rotateItem = new MenuItem("Rotate...");
            rotateItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (selectedObjects.FirstOrDefault()!, "rotate"));
            transformMenu.Append(rotateItem);

            var scaleItem = new MenuItem("Scale...");
            scaleItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (selectedObjects.FirstOrDefault()!, "scale"));
            transformMenu.Append(scaleItem);

            transformMenu.Append(new SeparatorMenuItem());

            var resetTransformItem = new MenuItem("Reset Transform");
            resetTransformItem.Activated += (s, e) => {
                foreach (var obj in selectedObjects)
                {
                    obj.Position = OpenTK.Mathematics.Vector3.Zero;
                    obj.Rotation = OpenTK.Mathematics.Vector3.Zero;
                    obj.Scale = OpenTK.Mathematics.Vector3.One;
                }
                ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
            };
            transformMenu.Append(resetTransformItem);

            menu.Append(transformItem);

            // Mesh operations (if mesh selected)
            var meshObjects = selectedObjects.OfType<MeshObject>().ToList();
            if (meshObjects.Count > 0)
            {
                var meshMenu = new Menu();
                var meshItem = new MenuItem("Mesh Operations");
                meshItem.Submenu = meshMenu;

                var decimateItem = new MenuItem("Decimate...");
                decimateItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "decimate"));
                meshMenu.Append(decimateItem);

                var optimizeItem = new MenuItem("Optimize");
                optimizeItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "optimize"));
                meshMenu.Append(optimizeItem);

                var smoothItem = new MenuItem("Smooth");
                smoothItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "smooth"));
                meshMenu.Append(smoothItem);

                meshMenu.Append(new SeparatorMenuItem());

                var splitItem = new MenuItem("Split by Connectivity");
                splitItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "split_connectivity"));
                meshMenu.Append(splitItem);

                var flipNormalsItem = new MenuItem("Flip Normals");
                flipNormalsItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "flip_normals"));
                meshMenu.Append(flipNormalsItem);

                if (meshObjects.Count >= 2)
                {
                    meshMenu.Append(new SeparatorMenuItem());

                    var mergeItem = new MenuItem("Merge Selected");
                    mergeItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "merge_meshes"));
                    meshMenu.Append(mergeItem);

                    var alignItem = new MenuItem("Align to First");
                    alignItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (meshObjects.First(), "align_meshes"));
                    meshMenu.Append(alignItem);
                }

                menu.Append(meshItem);
            }

            // Point cloud operations
            var pointCloudObjects = selectedObjects.OfType<PointCloudObject>().ToList();
            if (pointCloudObjects.Count > 0)
            {
                var pcMenu = new Menu();
                var pcItem = new MenuItem("Point Cloud Operations");
                pcItem.Submenu = pcMenu;

                var downsampleItem = new MenuItem("Downsample...");
                downsampleItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (pointCloudObjects.First(), "downsample"));
                pcMenu.Append(downsampleItem);

                if (pointCloudObjects.Count >= 2)
                {
                    pcMenu.Append(new SeparatorMenuItem());

                    var mergePCItem = new MenuItem("Merge Selected");
                    mergePCItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (pointCloudObjects.First(), "merge_pointclouds"));
                    pcMenu.Append(mergePCItem);

                    var alignPCItem = new MenuItem("Align to First (ICP)");
                    alignPCItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (pointCloudObjects.First(), "align_pointclouds"));
                    pcMenu.Append(alignPCItem);
                }

                menu.Append(pcItem);
            }

            // Camera operations
            var cameraObjects = selectedObjects.OfType<CameraObject>().ToList();
            if (cameraObjects.Count > 0)
            {
                var camMenu = new Menu();
                var camItem = new MenuItem("Camera Operations");
                camItem.Submenu = camMenu;

                var viewFromItem = new MenuItem("View from Camera");
                viewFromItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (cameraObjects.First(), "view_from_camera"));
                camMenu.Append(viewFromItem);

                var showImageItem = new MenuItem("Show Image");
                showImageItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (cameraObjects.First(), "show_camera_image"));
                camMenu.Append(showImageItem);

                camMenu.Append(new SeparatorMenuItem());

                var toggleFrustumItem = new MenuItem("Toggle Frustum");
                toggleFrustumItem.Activated += (s, e) => {
                    foreach (var cam in cameraObjects)
                        cam.ShowFrustum = !cam.ShowFrustum;
                    ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
                };
                camMenu.Append(toggleFrustumItem);

                menu.Append(camItem);
            }

            menu.Append(new SeparatorMenuItem());

            // Common operations
            var duplicateItem = new MenuItem("Duplicate");
            duplicateItem.Activated += (s, e) => OnDuplicateClicked(s, e);
            menu.Append(duplicateItem);

            var deleteItem = new MenuItem("Delete");
            deleteItem.Activated += (s, e) => OnDeleteClicked(s, e);
            menu.Append(deleteItem);

            menu.Append(new SeparatorMenuItem());

            // Visibility shortcuts
            var hideItem = new MenuItem("Hide");
            hideItem.Activated += (s, e) => {
                foreach (var obj in selectedObjects)
                    obj.Visible = false;
                RefreshTree();
                ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
            };
            menu.Append(hideItem);

            var showAllItem = new MenuItem("Show All");
            showAllItem.Activated += (s, e) => {
                if (_sceneGraph != null)
                {
                    foreach (var obj in _sceneGraph.GetObjectsOfType<SceneObject>())
                        obj.Visible = true;
                    RefreshTree();
                    ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
                }
            };
            menu.Append(showAllItem);

            menu.Append(new SeparatorMenuItem());

            // Focus
            var focusItem = new MenuItem("Focus on Selected");
            focusItem.Activated += (s, e) => ObjectActionRequested?.Invoke(this, (selectedObjects.FirstOrDefault()!, "focus"));
            menu.Append(focusItem);

            menu.ShowAll();
            menu.Popup();
        }

        private void OnDeleteClicked(object? sender, EventArgs e)
        {
            if (_sceneGraph == null) return;

            var toDelete = GetSelectedObjects().ToList();
            foreach (var obj in toDelete)
            {
                _sceneGraph.RemoveObject(obj);
            }
            RefreshTree();
            ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
        }

        private void OnDuplicateClicked(object? sender, EventArgs e)
        {
            if (_sceneGraph == null) return;

            var toDuplicate = GetSelectedObjects().ToList();
            foreach (var obj in toDuplicate)
            {
                var clone = obj.Clone();
                clone.Position += new OpenTK.Mathematics.Vector3(0.5f, 0, 0);
                _sceneGraph.AddObject(clone, obj.Parent);
            }
            RefreshTree();
            ObjectActionRequested?.Invoke(this, (null!, "refresh_viewport"));
        }

        private void OnSceneObjectAdded(object? sender, SceneObject obj)
        {
            RefreshTree();
        }

        private void OnSceneObjectRemoved(object? sender, SceneObject obj)
        {
            RefreshTree();
        }

        private void OnSceneSelectionChanged(object? sender, EventArgs e)
        {
            // Sync selection from scene to tree
            // This is handled by scene graph's selection events
        }

        private void OnSceneChanged(object? sender, EventArgs e)
        {
            RefreshTree();
        }

        private void UpdateInfoPanel()
        {
            // Update the info panel based on selection
            // Find the info label and update it
            foreach (var child in Children)
            {
                if (child is Box box)
                {
                    foreach (var subChild in box.Children)
                    {
                        if (subChild is Label label && label.Name == "info_label")
                        {
                            var selected = GetSelectedObjects().ToList();
                            if (selected.Count == 0)
                            {
                                label.Text = "Select an object to see details";
                            }
                            else if (selected.Count == 1)
                            {
                                var obj = selected[0];
                                string info = $"{obj.Name} ({GetTypeName(obj.ObjectType)})";

                                if (obj is MeshObject mesh)
                                {
                                    info += $"\n{mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles";
                                }
                                else if (obj is PointCloudObject pc)
                                {
                                    info += $"\n{pc.PointCount:N0} points";
                                }
                                else if (obj is CameraObject cam)
                                {
                                    info += $"\n{cam.ImageWidth}x{cam.ImageHeight}, FOV: {cam.FieldOfView:F0}";
                                }

                                label.Text = info;
                            }
                            else
                            {
                                label.Text = $"{selected.Count} objects selected";
                            }
                            return;
                        }
                    }
                }
            }
        }

        #endregion

        /// <summary>
        /// Gets currently selected objects
        /// </summary>
        public IEnumerable<SceneObject> GetSelectedObjects()
        {
            var selectedPaths = _treeView.Selection.GetSelectedRows();
            foreach (var path in selectedPaths)
            {
                if (_treeStore.GetIter(out TreeIter iter, path))
                {
                    var obj = GetObjectFromIter(iter);
                    if (obj != null)
                        yield return obj;
                }
            }
        }

        /// <summary>
        /// Selects an object in the tree view
        /// </summary>
        public void SelectObject(SceneObject obj)
        {
            SelectObjectRecursive(TreeIter.Zero, obj, true);
        }

        private bool SelectObjectRecursive(TreeIter parentIter, SceneObject target, bool isRoot)
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
                int id = (int)_treeStore.GetValue(iter, COL_OBJECT_ID);
                if (id == target.Id)
                {
                    var path = _treeStore.GetPath(iter);
                    _treeView.ExpandToPath(path);
                    _treeView.Selection.SelectPath(path);
                    _treeView.ScrollToCell(path, null, true, 0.5f, 0);
                    return true;
                }

                if (SelectObjectRecursive(iter, target, false))
                    return true;

                hasChildren = _treeStore.IterNext(ref iter);
            }

            return false;
        }
    }
}
