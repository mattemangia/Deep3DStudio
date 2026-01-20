using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnAutoRig(object? sender, EventArgs e)
        {
            var selectedMeshObj = GetSelectedMeshObject();
            if (selectedMeshObj == null)
            {
                ShowMessage("No mesh selected", "Please select a mesh to rig.");
                return;
            }

            var rigSetting = IniSettings.Instance.RiggingModel;
            if (rigSetting != RiggingMethod.UniRig)
            {
                ShowMessage("Rigging disabled", "Select UniRig as the rigging model in AI Model Settings to enable rigging.");
                return;
            }

            UI.ProgressDialog.Instance.Start("Auto Rigging with UniRig...", UI.OperationType.Processing);

            Task.Run(async () =>
            {
                try
                {
                    var mesh = selectedMeshObj.MeshData;
                    UI.ProgressDialog.Instance.Update(0.1f, $"Rigging {selectedMeshObj.Name}...");

                    var rigResult = await AIModels.AIModelManager.Instance.RigMeshAsync(
                        mesh.Vertices.ToArray(),
                        mesh.Indices.ToArray(),
                        msg => Application.Invoke((s, e2) => UI.ProgressDialog.Instance.Log(msg)));

                    SkeletonData skeleton;
                    if (rigResult != null && rigResult.Success && rigResult.JointPositions?.Length > 0)
                    {
                        skeleton = SkeletonData.FromRigResult(rigResult);
                    }
                    else
                    {
                        var (min, max) = selectedMeshObj.GetWorldBounds();
                        var center = (min + max) * 0.5f;
                        var size = max - min;
                        float height = Math.Max(size.Y, 0.1f);
                        float scale = height;
                        var rootPosition = new OpenTK.Mathematics.Vector3(center.X, min.Y + height * 0.5f, center.Z);
                        skeleton = SkeletonData.CreateHumanoidTemplate(rootPosition, scale);
                    }

                    var skelObj = new SkeletonObject($"Rig_{selectedMeshObj.Name}", skeleton)
                    {
                        TargetMesh = selectedMeshObj,
                        Position = OpenTK.Mathematics.Vector3.Zero
                    };

                    Application.Invoke((s, e2) =>
                    {
                        _sceneGraph.AddObject(skelObj);
                        _activeSkeletonObject = skelObj;
                        _riggingPanel?.SetSkeleton(skelObj);
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"UniRig rigging complete. {skeleton.Joints.Count} joints.";
                        UI.ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    Application.Invoke((s, e2) =>
                    {
                        UI.ProgressDialog.Instance.Fail(ex);
                        _statusLabel.Text = "UniRig rigging failed.";
                    });
                }
            });
        }

        private void OnShowRiggingPanel(object? sender, EventArgs e)
        {
            if (_rightPanel == null)
            {
                CreateRiggingPanel();
            }

            if (_rightPanel != null)
            {
                _rightPanel.Visible = true;
                _viewport.SetGizmoMode(GizmoMode.Rigging);
            }
        }

        private void CreateRiggingPanel()
        {
            _riggingPanel = new RiggingPanel();

            // Hook up events
            _riggingPanel.RequestAddJoint += OnAddJoint;
            _riggingPanel.RequestAddBone += OnAddBone;
            _riggingPanel.RequestDeleteSelected += OnDeleteSelectedJoints;
            _riggingPanel.RequestImportUniRig += () => OnAutoRig(null, EventArgs.Empty);
            _riggingPanel.RequestExportRig += () => OnExportRiggedMesh(null, EventArgs.Empty);
            _riggingPanel.RequestCreateHumanoid += () => OnCreateHumanoidSkeleton(null, EventArgs.Empty);
            _riggingPanel.RefreshViewport += () => _viewport.QueueDraw();
            _riggingPanel.JointSelected += OnRiggingJointSelected;
            _riggingPanel.JointTransformChanged += OnRiggingJointTransformed;

            // Create right panel container
            _rightPanel = new Box(Orientation.Vertical, 0);
            _rightPanel.SetSizeRequest(280, -1);
            _rightPanel.PackStart(_riggingPanel, true, true, 0);

            // Add close button
            var closeBtn = new Button("Close Rigging Panel");
            closeBtn.Clicked += (s, e) => {
                if (_rightPanel != null) _rightPanel.Visible = false;
                _viewport.SetGizmoMode(GizmoMode.Select);
            };
            _rightPanel.PackStart(closeBtn, false, false, 5);

            // Add to main layout - need to find the paned container
            if (_mainHPaned != null)
            {
                // Get current child and wrap with another paned
                var currentChild = _mainHPaned.Child2;
                _mainHPaned.Remove(currentChild);

                var innerPaned = new Paned(Orientation.Horizontal);
                innerPaned.Pack1(currentChild, true, true);
                innerPaned.Pack2(_rightPanel, false, false);

                _mainHPaned.Pack2(innerPaned, true, true);
                _mainHPaned.ShowAll();
            }

            // Set active skeleton if one exists
            if (_activeSkeletonObject != null)
            {
                _riggingPanel.SetSkeleton(_activeSkeletonObject);
            }
        }

        private void OnCreateNewSkeleton(object? sender, EventArgs e)
        {
            // Get position from selected mesh centroid or origin
            var selectedMesh = GetSelectedMeshObject();
            var position = selectedMesh?.GetCentroid() ?? OpenTK.Mathematics.Vector3.Zero;

            var skeleton = new SkeletonData { Name = "New Skeleton" };
            skeleton.AddJoint("Root", position);

            _activeSkeletonObject = new SkeletonObject("Skeleton", skeleton);

            // Associate with selected mesh if any
            if (selectedMesh != null)
            {
                _activeSkeletonObject.TargetMesh = selectedMesh;
            }

            _sceneGraph.AddObject(_activeSkeletonObject);

            // Show rigging panel
            OnShowRiggingPanel(null, EventArgs.Empty);
            _riggingPanel?.SetSkeleton(_activeSkeletonObject);

            _statusLabel.Text = "Created new skeleton. Use rigging panel to add joints.";
        }

        private void OnCreateHumanoidSkeleton(object? sender, EventArgs e)
        {
            var selectedMesh = GetSelectedMeshObject();
            var position = selectedMesh?.GetCentroid() ?? OpenTK.Mathematics.Vector3.Zero;

            // Calculate scale based on mesh bounds
            float scale = 1.0f;
            if (selectedMesh != null)
            {
                var bounds = selectedMesh.GetWorldBounds();
                float height = bounds.max.Y - bounds.min.Y;
                scale = height > 0.1f ? height : 1.0f;
            }

            var skeleton = SkeletonData.CreateHumanoidTemplate(position, scale);
            _activeSkeletonObject = new SkeletonObject("Humanoid Skeleton", skeleton);

            if (selectedMesh != null)
            {
                _activeSkeletonObject.TargetMesh = selectedMesh;
            }

            _sceneGraph.AddObject(_activeSkeletonObject);

            OnShowRiggingPanel(null, EventArgs.Empty);
            _riggingPanel?.SetSkeleton(_activeSkeletonObject);

            _statusLabel.Text = $"Created humanoid skeleton with {skeleton.Joints.Count} joints.";
        }

        private void OnAddJoint()
        {
            if (_activeSkeletonObject?.Skeleton == null)
            {
                ShowMessage("No skeleton", "Create or select a skeleton first.");
                return;
            }

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();

            // Get parent joint (first selected, or null for root)
            Joint? parent = selectedJoints.FirstOrDefault();

            // Calculate position
            OpenTK.Mathematics.Vector3 position;
            if (parent != null)
            {
                position = parent.Position + new OpenTK.Mathematics.Vector3(0, 0.1f, 0);
            }
            else if (skeleton.RootJoint != null)
            {
                parent = skeleton.RootJoint;
                position = parent.Position + new OpenTK.Mathematics.Vector3(0, 0.1f, 0);
            }
            else
            {
                position = _activeSkeletonObject.Position;
            }

            string name = $"Joint_{skeleton.Joints.Count}";
            var newJoint = skeleton.AddJoint(name, position, parent);

            _riggingPanel?.RefreshTree();
            _riggingPanel?.SelectJoint(newJoint);
            _viewport.QueueDraw();

            _statusLabel.Text = $"Added joint '{name}'";
        }

        private void OnAddBone()
        {
            if (_activeSkeletonObject?.Skeleton == null) return;

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();

            if (selectedJoints.Count < 2)
            {
                ShowMessage("Select joints", "Select at least 2 joints to create a bone between them.");
                return;
            }

            // Create bones between consecutive selected joints
            for (int i = 0; i < selectedJoints.Count - 1; i++)
            {
                skeleton.AddBone(selectedJoints[i], selectedJoints[i + 1]);
            }

            _riggingPanel?.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Created {selectedJoints.Count - 1} bone(s)";
        }

        private void OnDeleteSelectedJoints()
        {
            if (_activeSkeletonObject?.Skeleton == null) return;

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();

            foreach (var joint in selectedJoints)
            {
                skeleton.RemoveJoint(joint);
            }

            _riggingPanel?.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Deleted {selectedJoints.Count} joint(s)";
        }

        private void OnRiggingJointSelected(object? sender, Joint joint)
        {
            // Focus viewport on joint
            _viewport.QueueDraw();
        }

        private void OnRiggingJointTransformed(object? sender, Joint joint)
        {
            _viewport.QueueDraw();
        }

        private void OnExportRiggedMesh(object? sender, EventArgs e)
        {
            var selectedMesh = GetSelectedMeshObject();
            if (selectedMesh == null)
            {
                ShowMessage("No mesh selected", "Please select a mesh to export.");
                return;
            }

            if (_activeSkeletonObject?.Skeleton == null)
            {
                ShowMessage("No skeleton", "Create or load a skeleton first using the rigging tool.");
                return;
            }

            var dialog = new FileChooserDialog(
                "Export Rigged Mesh",
                this,
                FileChooserAction.Save,
                "Cancel", ResponseType.Cancel,
                "Export", ResponseType.Accept);

            dialog.DoOverwriteConfirmation = true;

            // Add file filters
            var fbxFilter = new FileFilter();
            fbxFilter.Name = "FBX ASCII (*.fbx)";
            fbxFilter.AddPattern("*.fbx");
            dialog.AddFilter(fbxFilter);

            var glbFilter = new FileFilter();
            glbFilter.Name = "glTF Binary (*.glb)";
            glbFilter.AddPattern("*.glb");
            dialog.AddFilter(glbFilter);

            var gltfFilter = new FileFilter();
            gltfFilter.Name = "glTF (*.gltf)";
            gltfFilter.AddPattern("*.gltf");
            dialog.AddFilter(gltfFilter);

            dialog.SetFilename(selectedMesh.Name + ".fbx");

            if (dialog.Run() == (int)ResponseType.Accept)
            {
                var filePath = dialog.Filename;
                var options = new RiggedMeshExportOptions
                {
                    ExportSkeleton = true,
                    ExportSkinningWeights = _activeSkeletonObject.Skeleton.SkinningWeights != null,
                    ExportNormals = true,
                    ExportUVs = true
                };

                // Determine format from extension
                var ext = System.IO.Path.GetExtension(filePath).ToLowerInvariant();
                options.Format = ext switch
                {
                    ".glb" => TexturedMeshFormat.GLB,
                    ".gltf" => TexturedMeshFormat.GLTF,
                    _ => TexturedMeshFormat.FBX_ASCII
                };

                try
                {
                    RiggedMeshExporter.Export(
                        filePath,
                        selectedMesh.MeshData,
                        _activeSkeletonObject.Skeleton,
                        null, null, options);

                    _statusLabel.Text = $"Exported rigged mesh to {filePath}";
                }
                catch (Exception ex)
                {
                    ShowMessage("Export failed", ex.Message);
                }
            }

            dialog.Destroy();
        }

        private Scene.MeshObject? GetSelectedMeshObject()
        {
            var selected = _sceneGraph.GetSelectedObjects();
            foreach (var obj in selected)
            {
                if (obj is Scene.MeshObject meshObj)
                {
                    return meshObj;
                }
            }
            return null;
        }
    }
}
