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
        private enum ToolbarMergeMode
        {
            Cancel,
            MergeOnly,
            AlignThenMerge
        }

        private void OnDecimateClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            float ratio = Math.Clamp(_meshDecimateRatio, 0.01f, 0.99f);
            float voxelSize = Math.Max(0.0001f, _meshDecimateVoxelSize);
            bool isUniform = _meshDecimateMethod == 1;

            _statusLabel.Text = "Decimating...";
            while (Application.EventsPending()) Application.RunIteration();

            Task.Run(() => {
                var results = new List<(Scene.MeshObject obj, MeshData newData)>();
                foreach (var meshObj in selectedMeshes)
                {
                    var newData = isUniform
                        ? MeshOperations.DecimateUniform(meshObj.MeshData, voxelSize)
                        : MeshOperations.Decimate(meshObj.MeshData, ratio);
                    results.Add((meshObj, newData));
                }

                Application.Invoke((s, args) => {
                    foreach (var res in results)
                    {
                        res.obj.MeshData = res.newData;
                        res.obj.UpdateBounds();
                    }
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Decimated {selectedMeshes.Count} mesh(es)";
                    _isDirty = true;
                    UpdateTitle();
                });
            });
        }

        private void ConfigureDecimateOptions()
        {
            var dlg = new DecimateDialog(this)
            {
                Ratio = _meshDecimateRatio,
                VoxelSize = _meshDecimateVoxelSize,
                IsUniform = _meshDecimateMethod == 1
            };
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _meshDecimateRatio = dlg.Ratio;
                _meshDecimateVoxelSize = dlg.VoxelSize;
                _meshDecimateMethod = dlg.IsUniform ? 1 : 0;
            }
            dlg.Destroy();
        }

        private void OnSmoothClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            bool isTaubin = _meshSmoothMethod == 1;
            int iterations = Math.Clamp(_meshSmoothIterations, 1, 100);
            float lambda = Math.Clamp(_meshSmoothLambda, 0.01f, 1.0f);
            float mu = Math.Clamp(_meshSmoothMu, -1.0f, -0.01f);

            _statusLabel.Text = "Smoothing...";
            while (Application.EventsPending()) Application.RunIteration();

            Task.Run(() => {
                var results = new List<(Scene.MeshObject obj, MeshData newData)>();
                foreach (var meshObj in selectedMeshes)
                {
                    var newData = isTaubin
                        ? MeshOperations.SmoothTaubin(meshObj.MeshData, iterations, lambda, mu)
                        : MeshOperations.Smooth(meshObj.MeshData, iterations, lambda);
                    results.Add((meshObj, newData));
                }

                Application.Invoke((s, args) => {
                    foreach (var res in results)
                    {
                        res.obj.MeshData = res.newData;
                        res.obj.UpdateBounds();
                    }
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Smoothed {selectedMeshes.Count} mesh(es)";
                    _isDirty = true;
                    UpdateTitle();
                });
            });
        }

        private void ConfigureSmoothOptions()
        {
            var dlg = new SmoothDialog(this)
            {
                IsTaubin = _meshSmoothMethod == 1,
                Iterations = _meshSmoothIterations,
                Lambda = _meshSmoothLambda,
                Mu = _meshSmoothMu
            };
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _meshSmoothMethod = dlg.IsTaubin ? 1 : 0;
                _meshSmoothIterations = dlg.Iterations;
                _meshSmoothLambda = dlg.Lambda;
                _meshSmoothMu = dlg.Mu;
            }
            dlg.Destroy();
        }

        private void OnOptimizeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            float epsilon = Math.Clamp(_meshOptimizeEpsilon, 0.000001f, 1.0f);
            _statusLabel.Text = "Optimizing...";
            while (Application.EventsPending()) Application.RunIteration();

            Task.Run(() => {
                var results = new List<(Scene.MeshObject obj, MeshData newData, int removed)>();
                foreach (var meshObj in selectedMeshes)
                {
                    int before = meshObj.VertexCount;
                    var newData = MeshOperations.Optimize(meshObj.MeshData, epsilon);
                    int removed = before - newData.Vertices.Count;
                    results.Add((meshObj, newData, removed));
                }

                Application.Invoke((s, args) => {
                    int totalRemoved = 0;
                    foreach (var res in results)
                    {
                        res.obj.MeshData = res.newData;
                        res.obj.UpdateBounds();
                        totalRemoved += res.removed;
                    }
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Optimized: removed {totalRemoved} duplicate vertices";
                    _isDirty = true;
                    UpdateTitle();
                });
            });
        }

        private void ConfigureOptimizeOptions()
        {
            var dlg = new NumericInputDialog(this, "Optimize Mesh", "Weld Distance / Epsilon:", _meshOptimizeEpsilon, 0.000001f, 1.0f, 0.0001f, 6);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _meshOptimizeEpsilon = dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnSplitClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            int partsCreated = 0;
            foreach (var meshObj in selectedMeshes)
            {
                var parts = MeshOperations.SplitByConnectivity(meshObj.MeshData);
                if (parts.Count > 1)
                {
                    _sceneGraph.RemoveObject(meshObj);

                    for (int i = 0; i < parts.Count; i++)
                    {
                        var partObj = new MeshObject($"{meshObj.Name}_Part{i + 1}", parts[i]);
                        _sceneGraph.AddObject(partObj);
                        partsCreated++;
                    }
                }
            }

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Split into {partsCreated} parts";
        }

        private void OnMergeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();

            if (selectedMeshes.Count > 0 && selectedPointClouds.Count > 0)
            {
                ShowMessage("Mixed Selection Not Supported", "Select only meshes or only point clouds before merging.");
                return;
            }

            if (selectedMeshes.Count >= 2)
            {
                var dlg = new NumericInputDialog(this, "Merge Meshes", "Weld Distance:", 0.001f, 0.0f, 1.0f, 0.001f, 6);
                if (dlg.Run() == (int)ResponseType.Ok)
                {
                    float weldDist = dlg.Value;
                    _statusLabel.Text = "Merging...";
                    while (Application.EventsPending()) Application.RunIteration();

                    Task.Run(() => {
                        var meshDataList = selectedMeshes.Select(m => m.MeshData).ToList();
                        var merged = MeshOperations.MergeWithWelding(meshDataList, weldDist);

                        Application.Invoke((s, args) => {
                            foreach (var m in selectedMeshes)
                                _sceneGraph.RemoveObject(m);

                            var mergedObj = new MeshObject("Merged Mesh", merged);
                            _sceneGraph.AddObject(mergedObj);
                            _sceneGraph.Select(mergedObj);
                            _sceneTreeView.RefreshTree();
                            _viewport.QueueDraw();
                            _statusLabel.Text = $"Merged {selectedMeshes.Count} meshes";
                        });
                    });
                }
                dlg.Destroy();
            }
            else if (selectedPointClouds.Count >= 2)
            {
                var merged = MeshOperations.MergePointClouds(selectedPointClouds);

                foreach (var pc in selectedPointClouds)
                    _sceneGraph.RemoveObject(pc);

                _sceneGraph.AddObject(merged);
                _sceneGraph.Select(merged);

                _statusLabel.Text = $"Merged {selectedPointClouds.Count} point clouds";
                _sceneTreeView.RefreshTree();
                _viewport.QueueDraw();
            }
            else
            {
                ShowMessage("Please select at least 2 meshes or 2 point clouds to merge.");
                return;
            }
        }

        private void OnMergeMeshesToolbarClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count < 2)
            {
                ShowMessage("Need Mesh Selection", "Select at least 2 meshes to merge.");
                return;
            }

            var mode = PromptToolbarMergeMode("Merge Meshes");
            if (mode == ToolbarMergeMode.Cancel)
                return;

            int alignIterations = 50;
            float alignThreshold = 0.0001f;
            if (mode == ToolbarMergeMode.AlignThenMerge && !TryPromptAlignParameters(out alignIterations, out alignThreshold))
                return;

            if (!TryPromptMeshWeldDistance(out float weldDistance))
                return;

            _statusLabel.Text = mode == ToolbarMergeMode.AlignThenMerge ? "Aligning and merging meshes..." : "Merging meshes...";
            while (Application.EventsPending()) Application.RunIteration();

            Task.Run(() =>
            {
                try
                {
                    var meshDataList = selectedMeshes.Select(m => m.MeshData.Clone()).ToList();
                    if (mode == ToolbarMergeMode.AlignThenMerge)
                    {
                        var target = meshDataList[0];
                        for (int i = 1; i < meshDataList.Count; i++)
                        {
                            var transform = MeshOperations.AlignICP(meshDataList[i], target, alignIterations, alignThreshold);
                            meshDataList[i].ApplyTransform(transform);
                        }
                    }

                    var merged = MeshOperations.MergeWithWelding(meshDataList, weldDistance);
                    Application.Invoke((s, args) =>
                    {
                        foreach (var mesh in selectedMeshes)
                            _sceneGraph.RemoveObject(mesh);

                        var mergedObj = new MeshObject("Merged Mesh", merged);
                        _sceneGraph.AddObject(mergedObj);
                        _sceneGraph.Select(mergedObj);
                        _sceneTreeView.RefreshTree();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Merged {selectedMeshes.Count} mesh(es)";
                    });
                }
                catch (Exception ex)
                {
                    Application.Invoke((s, args) => ShowMessage("Mesh Merge Failed", ex.Message));
                }
            });
        }

        private void OnMergePointCloudsToolbarClicked(object? sender, EventArgs e)
        {
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (selectedPointClouds.Count < 2)
            {
                ShowMessage("Need Point Clouds", "Select at least 2 point clouds to merge.");
                return;
            }

            var mode = PromptToolbarMergeMode("Merge Point Clouds");
            if (mode == ToolbarMergeMode.Cancel)
                return;

            int alignIterations = 50;
            float alignThreshold = 0.0001f;
            if (mode == ToolbarMergeMode.AlignThenMerge && !TryPromptAlignParameters(out alignIterations, out alignThreshold))
                return;

            _statusLabel.Text = mode == ToolbarMergeMode.AlignThenMerge ? "Aligning and merging point clouds..." : "Merging point clouds...";
            while (Application.EventsPending()) Application.RunIteration();

            Task.Run(() =>
            {
                try
                {
                    if (mode == ToolbarMergeMode.AlignThenMerge)
                    {
                        var target = selectedPointClouds[0];
                        var targetPoints = target.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, target.GetWorldTransform())).ToList();
                        for (int i = 1; i < selectedPointClouds.Count; i++)
                        {
                            var source = selectedPointClouds[i];
                            var sourcePoints = source.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, source.GetWorldTransform())).ToList();
                            var transform = MeshOperations.AlignICP(sourcePoints, targetPoints, alignIterations, alignThreshold);
                            source.ApplyTransform(transform);
                            source.UpdateBounds();
                        }
                    }

                    var merged = MeshOperations.MergePointClouds(selectedPointClouds);
                    Application.Invoke((s, args) =>
                    {
                        foreach (var pc in selectedPointClouds)
                            _sceneGraph.RemoveObject(pc);

                        _sceneGraph.AddObject(merged);
                        _sceneGraph.Select(merged);
                        _sceneTreeView.RefreshTree();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Merged {selectedPointClouds.Count} point cloud(s)";
                    });
                }
                catch (Exception ex)
                {
                    Application.Invoke((s, args) => ShowMessage("Point Cloud Merge Failed", ex.Message));
                }
            });
        }

        private ToolbarMergeMode PromptToolbarMergeMode(string title)
        {
            var dialog = new MessageDialog(
                this,
                DialogFlags.Modal,
                MessageType.Question,
                ButtonsType.None,
                "Choose operation mode:");

            dialog.Title = title;
            dialog.SecondaryText = "You can merge directly, or align first and then merge.";
            dialog.AddButton("Cancel", ResponseType.Cancel);
            dialog.AddButton("Merge", ResponseType.Yes);
            dialog.AddButton("Align + Merge", ResponseType.Apply);

            var response = (ResponseType)dialog.Run();
            dialog.Destroy();
            return response switch
            {
                ResponseType.Yes => ToolbarMergeMode.MergeOnly,
                ResponseType.Apply => ToolbarMergeMode.AlignThenMerge,
                _ => ToolbarMergeMode.Cancel
            };
        }

        private bool TryPromptAlignParameters(out int iterations, out float threshold)
        {
            iterations = 50;
            threshold = 0.0001f;

            var alignDialog = new AlignDialog(this);
            try
            {
                if (alignDialog.Run() != (int)ResponseType.Ok)
                    return false;

                iterations = alignDialog.Iterations;
                threshold = alignDialog.Threshold;
                return true;
            }
            finally
            {
                alignDialog.Destroy();
            }
        }

        private bool TryPromptMeshWeldDistance(out float weldDistance)
        {
            weldDistance = 0.001f;
            var weldDialog = new NumericInputDialog(this, "Merge Meshes", "Weld Distance:", 0.001f, 0.0f, 1.0f, 0.001f, 6);
            try
            {
                if (weldDialog.Run() != (int)ResponseType.Ok)
                    return false;

                weldDistance = weldDialog.Value;
                return true;
            }
            finally
            {
                weldDialog.Destroy();
            }
        }

        private void OnAlignClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();

            if (selectedMeshes.Count > 0 && selectedPointClouds.Count > 0)
            {
                ShowMessage("Mixed Selection Not Supported", "Select only meshes or only point clouds before alignment.");
                return;
            }

            if (selectedMeshes.Count >= 2)
            {
                var dlg = new AlignDialog(this);
                if (dlg.Run() == (int)ResponseType.Ok)
                {
                    int iter = dlg.Iterations;
                    float threshold = dlg.Threshold;

                    _statusLabel.Text = "Aligning...";
                    while (Application.EventsPending()) Application.RunIteration();

                    Task.Run(() => {
                        var target = selectedMeshes[0];
                        var transforms = new List<(Scene.MeshObject obj, OpenTK.Mathematics.Matrix4 transform)>();

                        for (int i = 1; i < selectedMeshes.Count; i++)
                        {
                            var transform = MeshOperations.AlignICP(selectedMeshes[i].MeshData, target.MeshData, iter, threshold);
                            transforms.Add((selectedMeshes[i], transform));
                        }

                        Application.Invoke((s, args) => {
                            foreach (var t in transforms)
                            {
                                t.obj.MeshData.ApplyTransform(t.transform);
                                t.obj.UpdateBounds();
                            }
                            _viewport.QueueDraw();
                            _statusLabel.Text = $"Aligned {selectedMeshes.Count - 1} mesh(es) to target";
                        });
                    });
                }
                dlg.Destroy();
            }
            else if (selectedPointClouds.Count >= 2)
            {
                var dlg = new AlignDialog(this);
                if (dlg.Run() == (int)ResponseType.Ok)
                {
                    int iter = dlg.Iterations;
                    float threshold = dlg.Threshold;

                    _statusLabel.Text = "Aligning...";
                    while (Application.EventsPending()) Application.RunIteration();

                    Task.Run(() => {
                        var target = selectedPointClouds[0];
                        var targetPoints = target.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, target.GetWorldTransform())).ToList();

                        for (int i = 1; i < selectedPointClouds.Count; i++)
                        {
                            var source = selectedPointClouds[i];
                            var sourcePoints = source.Points.Select(p => OpenTK.Mathematics.Vector3.TransformPosition(p, source.GetWorldTransform())).ToList();

                            var transform = MeshOperations.AlignICP(sourcePoints, targetPoints, iter, threshold);
                            source.ApplyTransform(transform);
                            source.UpdateBounds();
                        }

                        Application.Invoke((s, args) => {
                            _viewport.QueueDraw();
                            _statusLabel.Text = $"Aligned {selectedPointClouds.Count - 1} point cloud(s) to target";
                        });
                    });
                }
                dlg.Destroy();
            }
            else
            {
                ShowMessage("Please select at least 2 meshes or 2 point clouds to align.");
                return;
            }
        }

        private void OnSetRealSizeClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();

            // If no selection, apply to all meshes in the scene
            if (selectedMeshes.Count == 0)
            {
                var allMeshes = _sceneGraph.GetObjectsOfType<MeshObject>().ToList();
                if (allMeshes.Count == 0)
                {
                    ShowMessage("No meshes found to scale.");
                    return;
                }
                selectedMeshes = allMeshes;
            }

            // Calculate bounding box of selection
            var min = new OpenTK.Mathematics.Vector3(float.MaxValue);
            var max = new OpenTK.Mathematics.Vector3(float.MinValue);
            bool hasBounds = false;

            foreach (var obj in selectedMeshes)
            {
                var (bMin, bMax) = obj.GetWorldBounds();
                min = OpenTK.Mathematics.Vector3.ComponentMin(min, bMin);
                max = OpenTK.Mathematics.Vector3.ComponentMax(max, bMax);
                hasBounds = true;
            }

            if (!hasBounds) return;

            float sizeX = max.X - min.X;
            float sizeY = max.Y - min.Y;
            float sizeZ = max.Z - min.Z;

            var dlg = new ScaleCalibrationDialog(this, sizeX, sizeY, sizeZ);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                float factor = dlg.RealScaleFactor;
                if (Math.Abs(factor - 1.0f) > 0.0001f)
                {
                    // Apply scale to selected objects.
                    // We apply the transform directly to the mesh vertices ("baking" the scale)
                    // to ensure that exported models retain the correct physical dimensions
                    // regardless of the target software's handling of hierarchy transforms.

                    foreach (var meshObj in selectedMeshes)
                    {
                        var matrix = OpenTK.Mathematics.Matrix4.CreateScale(factor);
                        meshObj.MeshData.ApplyTransform(matrix);

                        // Update position to maintain relative distances between objects
                        meshObj.Position *= factor;
                        meshObj.UpdateBounds();
                    }

                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Scaled {selectedMeshes.Count} objects by {factor:F4}";
                }
            }
            dlg.Destroy();
        }

        private void OnFillHolesClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            _statusLabel.Text = "Filling holes...";
            while (Application.EventsPending()) Application.RunIteration();

            int filled = 0;
            foreach (var meshObj in selectedMeshes)
            {
                int oldTris = meshObj.TriangleCount;
                meshObj.MeshData = MeshCleaningTools.FillHoles(meshObj.MeshData);
                meshObj.UpdateBounds();
                filled += meshObj.TriangleCount - oldTris;
            }

            _viewport.QueueDraw();
            _statusLabel.Text = $"Filled holes in {selectedMeshes.Count} mesh(es): +{filled} triangles";
            _sceneTreeView.RefreshTree();
            _isDirty = true;
        }

        private void OnMeshCleanupClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            // If multiple selected, ask to process first or all? For now process all.
            int processed = 0;
            foreach (var meshObj in selectedMeshes)
            {
                var dlg = new MeshCleanupDialog(this, meshObj.VertexCount, meshObj.TriangleCount);
                if (dlg.Run() == (int)ResponseType.Ok)
                {
                    _statusLabel.Text = $"Cleaning mesh {meshObj.Name}...";
                    while (Application.EventsPending()) Application.RunIteration();

                    meshObj.MeshData = MeshCleaningTools.CleanupMesh(meshObj.MeshData, dlg.Options);
                    meshObj.UpdateBounds();
                    processed++;
                }
                dlg.Destroy();
            }

            if (processed > 0)
            {
                _viewport.QueueDraw();
                _statusLabel.Text = $"Cleaned {processed} mesh(es)";
                _sceneTreeView.RefreshTree();
            }
        }

        private async void OnBakeTexturesClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh to bake textures onto.");
                return;
            }

            var meshObj = selectedMeshes[0]; // Process first one for now
            var cameras = _sceneGraph.GetObjectsOfType<CameraObject>().ToList();

            if (cameras.Count == 0)
            {
                ShowMessage("No cameras found in scene. Cannot bake textures from images.");
                return;
            }

            var dlg = new TextureBakingDialog(this, cameras);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                // Ask for output file if we are exporting
                string exportPath = "";
                if (dlg.ExportOptions.Format != TexturedMeshFormat.OBJ &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.GLTF &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.GLB &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.FBX_ASCII &&
                    dlg.ExportOptions.Format != TexturedMeshFormat.PLY)
                {
                    // Should not happen if dialog returns valid format
                }

                var fc = new FileChooserDialog("Export Textured Mesh", this, FileChooserAction.Save,
                    "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

                string ext = dlg.ExportOptions.Format switch
                {
                    TexturedMeshFormat.OBJ => ".obj",
                    TexturedMeshFormat.GLTF => ".gltf",
                    TexturedMeshFormat.GLB => ".glb",
                    TexturedMeshFormat.FBX_ASCII => ".fbx",
                    TexturedMeshFormat.PLY => ".ply",
                    _ => ".obj"
                };

                fc.CurrentName = meshObj.Name + ext;

                if (fc.Run() == (int)ResponseType.Accept)
                {
                    exportPath = fc.Filename;
                }
                fc.Destroy();

                if (string.IsNullOrEmpty(exportPath)) return;

                _statusLabel.Text = "Baking textures... This may take a while.";
                while (Application.EventsPending()) Application.RunIteration();

                try
                {
                    var baker = new TextureBaker();
                    // Copy settings
                    baker.TextureSize = dlg.BakerSettings.TextureSize;
                    baker.IslandMargin = dlg.BakerSettings.IslandMargin;
                    baker.BlendMode = dlg.BakerSettings.BlendMode;
                    baker.MinViewAngleCosine = dlg.BakerSettings.MinViewAngleCosine;
                    baker.BlendSeams = dlg.BakerSettings.BlendSeams;
                    baker.DilationPasses = dlg.BakerSettings.DilationPasses;

                    var uvData = await Task.Run(() => baker.GenerateUVs(meshObj.MeshData, dlg.UVMethod));

                    BakedTextureResult? baked = null;

                    if (dlg.BakeFromCameras)
                    {
                        baked = await Task.Run(() => baker.BakeTextures(meshObj.MeshData, uvData, dlg.SelectedCameras));
                    }
                    else
                    {
                        var tex = await Task.Run(() => baker.BakeVertexColorsToTexture(meshObj.MeshData, uvData));
                        baked = new BakedTextureResult
                        {
                            DiffuseMap = tex,
                            TextureSize = baker.TextureSize,
                            WeightMap = new float[baker.TextureSize, baker.TextureSize]
                        };
                    }

                    _statusLabel.Text = "Exporting textured mesh...";
                    while (Application.EventsPending()) Application.RunIteration();

                    await Task.Run(() => TexturedMeshExporter.Export(exportPath, meshObj.MeshData, uvData, baked, dlg.ExportOptions));

                    baked?.Dispose();

                    _statusLabel.Text = $"Exported textured mesh to {exportPath}";
                    ShowMessage($"Baking and Export Complete!\nSaved to: {exportPath}");
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error during baking: {ex.Message}");
                    Console.WriteLine(ex);
                    _statusLabel.Text = "Baking failed.";
                }
            }
            dlg.Destroy();
        }

        private void OnCreatePrimitive(MeshPrimitiveType type)
        {
            var mesh = CreatePrimitiveWithCurrentPreset(type);
            var name = type switch
            {
                MeshPrimitiveType.UVSphere => "UV Sphere",
                _ => type.ToString()
            };

            var obj = new MeshObject(name, mesh);
            _sceneGraph.AddObject(obj);
            _sceneGraph.Select(obj);
            _sceneTreeView.RefreshTree();
            _viewport.FocusOnSelection();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Created {name} primitive";
            _isDirty = true;
            UpdateTitle();
        }

        private MeshData CreatePrimitiveWithCurrentPreset(MeshPrimitiveType type)
        {
            return type switch
            {
                MeshPrimitiveType.Plane => MeshPrimitiveFactory.CreatePlane(_primSize, _primSize),
                MeshPrimitiveType.Cube => MeshPrimitiveFactory.CreateCube(_primSize),
                MeshPrimitiveType.UVSphere => MeshPrimitiveFactory.CreateUVSphere(_primRadius, _primSegments, _primRings),
                MeshPrimitiveType.Cylinder => MeshPrimitiveFactory.CreateCylinder(_primRadius, _primHeight, _primSegments, _primCapEnds),
                MeshPrimitiveType.Cone => MeshPrimitiveFactory.CreateCone(_primRadius, _primHeight, _primSegments, _primCapEnds),
                MeshPrimitiveType.Torus => MeshPrimitiveFactory.CreateTorus(Math.Max(_primRadius, 0.05f), Math.Max(_primRadius * 0.35f, 0.02f), _primSegments, _primMinorSegments),
                MeshPrimitiveType.Circle => MeshPrimitiveFactory.CreateCircle(Math.Max(_primRadius, 0.01f), _primSegments),
                MeshPrimitiveType.Polygon => MeshPrimitiveFactory.CreatePolygon(_primPolygonSides, Math.Max(_primRadius, 0.01f)),
                MeshPrimitiveType.Grid => MeshPrimitiveFactory.CreateGrid(_primGridCells, _primCellSize),
                _ => MeshPrimitiveFactory.CreatePrimitive(type)
            };
        }

        private void ConfigurePrimitiveOptions(MeshPrimitiveType type)
        {
            using var dlg = new Dialog($"Primitive Options - {type}", this, DialogFlags.Modal);
            dlg.SetDefaultSize(360, 260);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Save", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var sizeSpin = new SpinButton(0.01, 1000, 0.01) { Digits = 3, Value = _primSize };
            var radiusSpin = new SpinButton(0.001, 1000, 0.01) { Digits = 3, Value = _primRadius };
            var heightSpin = new SpinButton(0.001, 1000, 0.01) { Digits = 3, Value = _primHeight };
            var segSpin = new SpinButton(3, 256, 1) { Value = _primSegments };
            var ringSpin = new SpinButton(3, 256, 1) { Value = _primRings };
            var minorSegSpin = new SpinButton(3, 256, 1) { Value = _primMinorSegments };
            var sideSpin = new SpinButton(3, 64, 1) { Value = _primPolygonSides };
            var gridCellSpin = new SpinButton(1, 512, 1) { Value = _primGridCells };
            var gridCellSizeSpin = new SpinButton(0.001, 100, 0.01) { Digits = 3, Value = _primCellSize };
            var capCheck = new CheckButton("Cap Ends / Base") { Active = _primCapEnds };

            void AddRow(string label, Widget widget)
            {
                var row = new Box(Orientation.Horizontal, 8);
                row.PackStart(new Label(label) { Halign = Align.Start, WidthRequest = 160 }, false, false, 0);
                row.PackStart(widget, true, true, 0);
                area.PackStart(row, false, false, 0);
            }

            switch (type)
            {
                case MeshPrimitiveType.Plane:
                case MeshPrimitiveType.Cube:
                    AddRow("Size", sizeSpin);
                    break;
                case MeshPrimitiveType.UVSphere:
                    AddRow("Radius", radiusSpin);
                    AddRow("Segments", segSpin);
                    AddRow("Rings", ringSpin);
                    break;
                case MeshPrimitiveType.Cylinder:
                case MeshPrimitiveType.Cone:
                    AddRow("Radius", radiusSpin);
                    AddRow("Height", heightSpin);
                    AddRow("Segments", segSpin);
                    AddRow("Cap", capCheck);
                    break;
                case MeshPrimitiveType.Torus:
                    AddRow("Major Radius", radiusSpin);
                    AddRow("Major Segments", segSpin);
                    AddRow("Minor Segments", minorSegSpin);
                    break;
                case MeshPrimitiveType.Circle:
                    AddRow("Radius", radiusSpin);
                    AddRow("Segments", segSpin);
                    break;
                case MeshPrimitiveType.Polygon:
                    AddRow("Radius", radiusSpin);
                    AddRow("Sides", sideSpin);
                    break;
                case MeshPrimitiveType.Grid:
                    AddRow("Cells Per Side", gridCellSpin);
                    AddRow("Cell Size", gridCellSizeSpin);
                    break;
            }

            dlg.ShowAll();
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _primSize = (float)sizeSpin.Value;
                _primRadius = (float)radiusSpin.Value;
                _primHeight = (float)heightSpin.Value;
                _primSegments = (int)segSpin.Value;
                _primRings = (int)ringSpin.Value;
                _primMinorSegments = (int)minorSegSpin.Value;
                _primPolygonSides = (int)sideSpin.Value;
                _primGridCells = (int)gridCellSpin.Value;
                _primCellSize = (float)gridCellSizeSpin.Value;
                _primCapEnds = capCheck.Active;
            }
        }

        private List<PointCloudObject> GetSelectedPointClouds()
        {
            return _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
        }

        private void OnPointCloudVisibilitySliderChanged(object? sender, EventArgs e)
        {
            if (_updatingPointCloudVisibilityControls || _pcVisibilityScale == null)
                return;

            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
                return;

            float fraction = Math.Clamp((float)(_pcVisibilityScale.Value / 100.0), 0.0f, 1.0f);
            foreach (var pc in selected)
                pc.VisibleFraction = fraction;

            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();

            int visible = selected.Sum(pc => pc.VisiblePointCount);
            int total = selected.Sum(pc => pc.PointCount);
            _statusLabel.Text = $"Visible points: {visible:N0}/{total:N0} ({fraction * 100f:F1}%)";
            _isDirty = true;
            UpdateTitle();
        }

        private void UpdatePointCloudVisibilityControls()
        {
            if (_pcVisibilityScale == null || _pcVisibilityLabel == null)
                return;

            var selected = GetSelectedPointClouds();
            _updatingPointCloudVisibilityControls = true;
            try
            {
                if (selected.Count == 0)
                {
                    _pcVisibilityScale.Sensitive = false;
                    _pcVisibilityLabel.Text = "--";
                    return;
                }

                _pcVisibilityScale.Sensitive = true;
                float avgFraction = selected.Average(pc => pc.VisibleFraction);
                _pcVisibilityScale.Value = Math.Round(avgFraction * 100.0f, 1);

                int visible = selected.Sum(pc => pc.VisiblePointCount);
                int total = selected.Sum(pc => pc.PointCount);
                _pcVisibilityLabel.Text = $"{avgFraction * 100f:F1}% ({visible:N0}/{total:N0})";
            }
            finally
            {
                _updatingPointCloudVisibilityControls = false;
            }
        }

        private void OnPointCloudVoxelDownsampleClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            float voxelSize = _pcVoxelSize;
            ProgressDialog.Instance.Start("Voxel Downsampling...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.VoxelDownsampleAsync(
                            selected[i],
                            voxelSize,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Voxel downsample complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Voxel downsampling cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudVoxel()
        {
            var dlg = new NumericInputDialog(this, "Voxel Downsample", "Voxel Size:", _pcVoxelSize, 0.0001f, 10f, 0.001f, 4);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _pcVoxelSize = dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnPointCloudRemoveOutliersClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            int kNeighbors = _pcOutlierK;
            float stdRatio = _pcOutlierStdRatio;
            ProgressDialog.Instance.Start("Removing Outliers...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveStatisticalOutliersAsync(
                            selected[i],
                            kNeighbors,
                            stdRatio,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Outlier filter complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Outlier removal cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudOutliers()
        {
            var kDlg = new NumericInputDialog(this, "Outlier Removal", "K Neighbors:", _pcOutlierK, 2, 200, 1, 0);
            if (kDlg.Run() != (int)ResponseType.Ok)
            {
                kDlg.Destroy();
                return;
            }
            _pcOutlierK = (int)kDlg.Value;
            kDlg.Destroy();

            var stdDlg = new NumericInputDialog(this, "Outlier Removal", "Std. Ratio:", _pcOutlierStdRatio, 0.1f, 10f, 0.1f, 2);
            if (stdDlg.Run() == (int)ResponseType.Ok)
            {
                _pcOutlierStdRatio = stdDlg.Value;
            }
            stdDlg.Destroy();
        }

        private void OnPointCloudRemoveDuplicatesClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            float threshold = _pcDuplicateThreshold;
            ProgressDialog.Instance.Start("Removing Duplicates...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveDuplicatesAsync(
                            selected[i],
                            threshold,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Duplicate removal complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Duplicate removal cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudDuplicates()
        {
            var dlg = new NumericInputDialog(this, "Remove Duplicates", "Distance Threshold:", _pcDuplicateThreshold, 0.000001f, 1f, 0.0001f, 6);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _pcDuplicateThreshold = dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnPointCloudRemoveSkyBlueClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            var options = new PointCloudBlueFilterOptions
            {
                MinBlue = _pcSkyMinBlue,
                MaxRed = _pcSkyMaxRed,
                MaxGreen = _pcSkyMaxGreen,
                MinBlueDominance = _pcSkyBlueDominance
            };

            ProgressDialog.Instance.Start("Removing Sky/Blue Points...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveBlueDominantPointsAsync(
                            selected[i],
                            options,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Sky/blue filter complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Sky/blue filter cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudSkyBlue()
        {
            var dlg = new Dialog("Sky / Blue Filter", this, DialogFlags.Modal);
            dlg.SetDefaultSize(340, 220);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Apply", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var minBlue = new SpinButton(0.0, 1.0, 0.01) { Digits = 3, Value = _pcSkyMinBlue };
            var maxRed = new SpinButton(0.0, 1.0, 0.01) { Digits = 3, Value = _pcSkyMaxRed };
            var maxGreen = new SpinButton(0.0, 1.0, 0.01) { Digits = 3, Value = _pcSkyMaxGreen };
            var dominance = new SpinButton(0.0, 1.0, 0.01) { Digits = 3, Value = _pcSkyBlueDominance };

            area.PackStart(new Label("Min Blue"), false, false, 0);
            area.PackStart(minBlue, false, false, 0);
            area.PackStart(new Label("Max Red"), false, false, 0);
            area.PackStart(maxRed, false, false, 0);
            area.PackStart(new Label("Max Green"), false, false, 0);
            area.PackStart(maxGreen, false, false, 0);
            area.PackStart(new Label("Min Blue Dominance (B - max(R,G))"), false, false, 0);
            area.PackStart(dominance, false, false, 0);

            dlg.ShowAll();
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _pcSkyMinBlue = (float)minBlue.Value;
                _pcSkyMaxRed = (float)maxRed.Value;
                _pcSkyMaxGreen = (float)maxGreen.Value;
                _pcSkyBlueDominance = (float)dominance.Value;
            }

            dlg.Destroy();
        }

        private void OnPointCloudEstimateNormalsClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            int kNeighbors = _pcNormalK;
            ProgressDialog.Instance.Start("Estimating Normals...", OperationType.Processing);

            Task.Run(() => {
                try {
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        PointCloudOperations.EstimateNormalsAsync(
                            selected[i],
                            kNeighbors,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Wait();
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Estimated normals for {selected.Count} point cloud(s).";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Normal estimation cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudNormals()
        {
            var dlg = new NumericInputDialog(this, "Estimate Normals", "K Neighbors:", _pcNormalK, 3, 200, 1, 0);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _pcNormalK = (int)dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnPointCloudPassThroughClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            int axis = _pcPassAxis;
            float minValue = _pcPassMin;
            float maxValue = _pcPassMax;
            ProgressDialog.Instance.Start("Pass-Through Filter...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.PassThroughAxisAsync(
                            selected[i],
                            axis,
                            minValue,
                            maxValue,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Pass-through complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Pass-through filter cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudPassThrough()
        {
            var dlg = new Dialog("Pass-Through Filter", this, DialogFlags.Modal);
            dlg.SetDefaultSize(320, 180);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Apply", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var axisCombo = new ComboBoxText();
            axisCombo.AppendText("X");
            axisCombo.AppendText("Y");
            axisCombo.AppendText("Z");
            axisCombo.Active = Math.Clamp(_pcPassAxis, 0, 2);

            var minSpin = new SpinButton(-100000, 100000, 0.01) { Digits = 3, Value = _pcPassMin };
            var maxSpin = new SpinButton(-100000, 100000, 0.01) { Digits = 3, Value = _pcPassMax };

            area.PackStart(new Label("Axis"), false, false, 0);
            area.PackStart(axisCombo, false, false, 0);
            area.PackStart(new Label("Min Value"), false, false, 0);
            area.PackStart(minSpin, false, false, 0);
            area.PackStart(new Label("Max Value"), false, false, 0);
            area.PackStart(maxSpin, false, false, 0);
            dlg.ShowAll();

            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _pcPassAxis = axisCombo.Active;
                _pcPassMin = (float)minSpin.Value;
                _pcPassMax = (float)maxSpin.Value;
            }

            dlg.Destroy();
        }

        private void OnPointCloudRadiusCropClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            var center = _pcRadiusCenter;
            float radius = _pcRadius;
            ProgressDialog.Instance.Start("Radius Crop...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RadiusCropAsync(
                            selected[i],
                            center,
                            radius,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    Application.Invoke((s, args) => {
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        _statusLabel.Text = $"Radius crop complete. Removed {totalRemoved:N0} points.";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Radius crop cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudRadiusCrop()
        {
            var centerDlg = new Vector3InputDialog(
                this,
                "Radius Crop Center",
                "Center X", "Center Y", "Center Z",
                _pcRadiusCenter,
                -100000f, 100000f, 0.01f, 3);

            if (centerDlg.Run() != (int)ResponseType.Ok)
            {
                centerDlg.Destroy();
                return;
            }

            _pcRadiusCenter = centerDlg.Value;
            centerDlg.Destroy();

            var radiusDlg = new NumericInputDialog(this, "Radius Crop", "Radius:", _pcRadius, 0.001f, 100000f, 0.01f, 3);
            if (radiusDlg.Run() == (int)ResponseType.Ok)
            {
                _pcRadius = radiusDlg.Value;
            }
            radiusDlg.Destroy();
        }

        private void OnPointCloudDenseClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            float radius = _pcDenseRadius;
            int seeds = _pcDensePointsPerSeed;
            int before = selected.Sum(pc => pc.PointCount);

            ProgressDialog.Instance.Start("Densifying Point Cloud...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalAdded = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var added = PointCloudOperations.DensifyAsync(
                            selected[i],
                            radius,
                            seeds,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress,
                            cloudProgress).Result;

                        totalAdded += added;
                    }

                    Application.Invoke((s, args) => {
                        int after = selected.Sum(pc => pc.PointCount);
                        _sceneTreeView.RefreshTree();
                        UpdatePointCloudVisibilityControls();
                        _viewport.QueueDraw();
                        if (totalAdded > 0)
                        {
                            _statusLabel.Text = $"Dense cloud complete. Added {totalAdded:N0} points ({before:N0} -> {after:N0}).";
                        }
                        else
                        {
                            _statusLabel.Text = $"Dense cloud: no points added. Increase radius (current {radius:F4}) or reduce filtering.";
                        }
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    Application.Invoke((s, args) => {
                        _statusLabel.Text = "Densification cancelled.";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ConfigurePointCloudDense()
        {
            var radiusDlg = new NumericInputDialog(this, "Point Cloud to Dense Cloud", "Neighbor Radius:", _pcDenseRadius, 0.0001f, 10f, 0.001f, 4);
            if (radiusDlg.Run() != (int)ResponseType.Ok)
            {
                radiusDlg.Destroy();
                return;
            }
            _pcDenseRadius = radiusDlg.Value;
            radiusDlg.Destroy();

            var ppsDlg = new NumericInputDialog(this, "Point Cloud to Dense Cloud", "Points Per Seed:", _pcDensePointsPerSeed, 1, 8, 1, 0);
            if (ppsDlg.Run() == (int)ResponseType.Ok)
            {
                _pcDensePointsPerSeed = (int)ppsDlg.Value;
            }
            ppsDlg.Destroy();
        }

        private void OnSplittingPlaneClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }
            if (selectedMeshes.Count > 1)
            {
                ShowMessage("Please select only one mesh for splitting by plane.");
                return;
            }

            _viewport.SetGizmoMode(GizmoMode.SplittingPlane);
            _statusLabel.Text = "Splitting Plane mode - Adjust plane, press Enter to cut, Esc to cancel";
        }

        private void ApplySplittingPlane()
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
                return;

            var meshObj = selectedMeshes[0];
            var splittingTool = _viewport.GetSplittingPlaneTool();
            if (splittingTool == null)
                return;

            var (planePoint, planeNormal) = splittingTool.GetPlaneEquation();

            ProgressDialog.Instance.Start("Splitting mesh by plane...", OperationType.Processing);

            var originalName = meshObj.Name;
            var originalMesh = meshObj.MeshData;

            Task.Run(() => {
                try {
                    var progress = new Progress<float>(p => {
                        ProgressDialog.Instance.Update(p, $"Processing... {(int)(p * 100)}%");
                    });

                    var (above, below) = SplittingPlaneTool.SplitMeshByPlane(
                        originalMesh, planePoint, planeNormal, progress);

                    Application.Invoke((s, args) => {
                        if (above.Vertices.Count > 0 && below.Vertices.Count > 0)
                        {
                            _sceneGraph.RemoveObject(meshObj);

                            var aboveObj = new MeshObject($"{originalName}_Above", above);
                            var belowObj = new MeshObject($"{originalName}_Below", below);

                            _sceneGraph.AddObject(aboveObj);
                            _sceneGraph.AddObject(belowObj);

                            _sceneTreeView.RefreshTree();
                            _viewport.QueueDraw();
                            ProgressDialog.Instance.Complete();
                    _statusLabel.Text = $"Split into 2 meshes: {above.Indices.Count / 3} + {below.Indices.Count / 3} triangles";
                            _isDirty = true;
                            UpdateTitle();
                        }
                        else if (above.Vertices.Count > 0 || below.Vertices.Count > 0)
                        {
                            ProgressDialog.Instance.Fail(new Exception(
                                "Plane does not properly intersect mesh. All vertices are on one side of the plane. Adjust the plane position."));
                        }
                        else
                        {
                            ProgressDialog.Instance.Fail(new Exception("Mesh is empty after split."));
                        }
                    });
                }
                catch (Exception ex) {
                    Application.Invoke((s, args) => ProgressDialog.Instance.Fail(ex));
                }
            });

            _viewport.SetGizmoMode(GizmoMode.Select);
        }
    }
}
