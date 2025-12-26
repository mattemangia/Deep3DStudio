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
        private void OnDecimateClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh first.");
                return;
            }

            var dlg = new DecimateDialog(this);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                float ratio = dlg.Ratio;
                float voxelSize = dlg.VoxelSize;
                bool isUniform = dlg.IsUniform;

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
                    });
                });
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

            var dlg = new SmoothDialog(this);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                bool isTaubin = dlg.IsTaubin;
                int iterations = dlg.Iterations;
                float lambda = dlg.Lambda;
                float mu = dlg.Mu;

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
                    });
                });
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

            var dlg = new NumericInputDialog(this, "Optimize Mesh", "Weld Distance / Epsilon:", 0.0001f, 0.000001f, 1.0f, 0.0001f, 6);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                float epsilon = dlg.Value;
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
                    });
                });
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

        private void OnAlignClicked(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();

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
    }
}
