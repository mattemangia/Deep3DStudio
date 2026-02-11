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

            int removed = 0;
            foreach (var pc in selected)
                removed += PointCloudOperations.VoxelDownsample(pc, _pcVoxelSize);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Voxel downsample complete. Removed {removed:N0} points.";
            _isDirty = true;
            UpdateTitle();
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

            int removed = 0;
            foreach (var pc in selected)
                removed += PointCloudOperations.RemoveStatisticalOutliers(pc, _pcOutlierK, _pcOutlierStdRatio);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Outlier filter complete. Removed {removed:N0} points.";
            _isDirty = true;
            UpdateTitle();
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

            int removed = 0;
            foreach (var pc in selected)
                removed += PointCloudOperations.RemoveDuplicates(pc, _pcDuplicateThreshold);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Duplicate removal complete. Removed {removed:N0} points.";
            _isDirty = true;
            UpdateTitle();
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

        private void OnPointCloudEstimateNormalsClicked(object? sender, EventArgs e)
        {
            var selected = GetSelectedPointClouds();
            if (selected.Count == 0)
            {
                ShowMessage("Please select at least one point cloud first.");
                return;
            }

            foreach (var pc in selected)
                PointCloudOperations.EstimateNormals(pc, _pcNormalK);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Estimated normals for {selected.Count} point cloud(s).";
            _isDirty = true;
            UpdateTitle();
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

            int removed = 0;
            foreach (var pc in selected)
                removed += PointCloudOperations.PassThroughAxis(pc, _pcPassAxis, _pcPassMin, _pcPassMax);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Pass-through complete. Removed {removed:N0} points.";
            _isDirty = true;
            UpdateTitle();
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

            int removed = 0;
            foreach (var pc in selected)
                removed += PointCloudOperations.RadiusCrop(pc, _pcRadiusCenter, _pcRadius);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Radius crop complete. Removed {removed:N0} points.";
            _isDirty = true;
            UpdateTitle();
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

            int before = selected.Sum(pc => pc.PointCount);
            int added = 0;
            foreach (var pc in selected)
                added += PointCloudOperations.Densify(pc, _pcDenseRadius, _pcDensePointsPerSeed);
            int after = selected.Sum(pc => pc.PointCount);

            _sceneTreeView.RefreshTree();
            UpdatePointCloudVisibilityControls();
            _viewport.QueueDraw();
            if (added > 0)
            {
                _statusLabel.Text = $"Dense cloud complete. Added {added:N0} points ({before:N0} -> {after:N0}).";
            }
            else
            {
                _statusLabel.Text = $"Dense cloud: no points added. Increase radius (current {_pcDenseRadius:F4}) or reduce filtering.";
            }
            _isDirty = true;
            UpdateTitle();
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
    }
}
