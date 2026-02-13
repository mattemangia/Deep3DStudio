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
        private void OnImageDoubleClicked(object? sender, ImageEntry entry)
        {
            var previewDialog = new ImagePreviewDialog(this, entry);
            previewDialog.Run();
            previewDialog.Destroy();
        }

        private void OnOpenSettings(object? sender, EventArgs e)
        {
            var dlg = new SettingsDialog(this);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                dlg.SaveSettings();
                ApplyViewSettings();
            }
            dlg.Destroy();
        }

        private void ApplyViewSettings()
        {
            var s = IniSettings.Instance;
            if (_pointsToggle != null) _pointsToggle.Active = s.ShowPointCloud;
            if (_wireToggle != null) _wireToggle.Active = s.ShowWireframe;
            if (_textureToggle != null) _textureToggle.Active = s.ShowTexture;
            if (_meshToggle != null) _meshToggle.Active = s.ShowMesh;
            if (_camerasToggle != null) _camerasToggle.Active = s.ShowCameras;
            SyncPointCloudColorToggles();
            if (_topToolbar != null) _topToolbar.Visible = s.ShowTopToolbar;
            if (_verticalToolbar != null) _verticalToolbar.Visible = s.ShowVerticalToolbar;
            if (_meshEditorToolbar != null) _meshEditorToolbar.Visible = s.ShowMeshEditorToolbar;
            if (_pointCloudToolbar != null) _pointCloudToolbar.Visible = s.ShowPointCloudToolbar;
            if (_showTopToolbarMenuItem != null) _showTopToolbarMenuItem.Active = s.ShowTopToolbar;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = s.ShowVerticalToolbar;
            if (_showMeshEditorToolbarMenuItem != null) _showMeshEditorToolbarMenuItem.Active = s.ShowMeshEditorToolbar;
            if (_showPointCloudToolbarMenuItem != null) _showPointCloudToolbarMenuItem.Active = s.ShowPointCloudToolbar;
            _viewport.QueueDraw();
        }

        private void SetPointCloudColorMode(PointCloudColorMode mode)
        {
            IniSettings.Instance.PointCloudColor = mode;
            SyncPointCloudColorToggles();
            _viewport.QueueDraw();
        }

        private void SyncPointCloudColorToggles()
        {
            var mode = IniSettings.Instance.PointCloudColor;
            _updatingPointCloudColorToggles = true;
            try
            {
                if (_rgbColorToggle != null) _rgbColorToggle.Active = mode == PointCloudColorMode.RGB;
                if (_depthColorToggle != null) _depthColorToggle.Active = mode == PointCloudColorMode.DistanceMap;
                if (_confidenceColorToggle != null) _confidenceColorToggle.Active = mode == PointCloudColorMode.Confidence;
            }
            finally
            {
                _updatingPointCloudColorToggles = false;
            }
        }

        private void OnAddImages(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Choose Images", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);
            fc.SelectMultiple = true;

            var filter = new FileFilter();
            filter.Name = "Image Files";
            filter.AddPattern("*.jpg");
            filter.AddPattern("*.jpeg");
            filter.AddPattern("*.png");
            filter.AddPattern("*.bmp");
            filter.AddPattern("*.tiff");
            filter.AddPattern("*.tif");
            fc.AddFilter(filter);

            var allFilter = new FileFilter();
            allFilter.Name = "All Files";
            allFilter.AddPattern("*");
            fc.AddFilter(allFilter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                foreach (var f in fc.Filenames)
                {
                    _imagePaths.Add(f);
                    _imageBrowser.AddImage(f);
                }
                _statusLabel.Text = $"{_imageBrowser.ImageCount} images loaded";
            }
            fc.Destroy();
        }

        private async void OnGeneratePointCloud(object? sender, EventArgs e)
        {
            await RunPointCloudGeneration();
        }

        private async void OnGenerateMesh(object? sender, EventArgs e)
        {
            await RunMeshing();
        }

        private async void OnRunInference(object? sender, EventArgs e)
        {
            if (_autoWorkflowEnabled)
            {
                // Auto workflow mode: run full pipeline based on selected workflow
                bool success = await RunPointCloudGeneration();
                if (success)
                {
                    await RunMeshing();
                }
            }
            else
            {
                // Manual mode: run only Dust3R (point cloud generation) as the first step
                // User can then choose to run other steps manually
                await RunPointCloudGeneration();
            }
        }

        private async Task<bool> RunPointCloudGeneration()
        {
            var settings = IniSettings.Instance;

            // Determine effective method from Workflow Combo
            // First option "Multi-View (...)" uses the engine from Settings
            // Other options explicitly set their method
            ReconstructionMethod method = settings.ReconstructionMethod;
            string workflow = _workflowCombo.ActiveText;

            if (!string.IsNullOrEmpty(workflow))
            {
                if (workflow.StartsWith("Multi-View"))
                {
                    // Use the reconstruction method from Settings
                    method = settings.ReconstructionMethod;
                }
                else if (workflow.Contains("Feature Matching") || workflow.Contains("SfM"))
                {
                    method = ReconstructionMethod.FeatureMatching;
                }
                else if (workflow.Contains("TripoSR"))
                {
                    method = ReconstructionMethod.TripoSR;
                }
                else if (workflow.Contains("Wonder3D"))
                {
                    method = ReconstructionMethod.Wonder3D;
                }
            }

            // Special check for LGM workflow to allow single image pass-through (handled in Meshing phase)
            bool isLGM = !string.IsNullOrEmpty(workflow) && workflow.Contains("LGM");

            bool requiresMultiView = !isLGM && (method == ReconstructionMethod.Dust3r ||
                                     method == ReconstructionMethod.Mast3r ||
                                     method == ReconstructionMethod.Must3r ||
                                     method == ReconstructionMethod.FeatureMatching);
            int minImages = requiresMultiView ? 2 : 1;

            if (_imagePaths.Count < minImages)
            {
                ShowMessage($"Please add at least {minImages} image{(minImages > 1 ? "s" : "")} for {method}.");
                return false;
            }

            _statusLabel.Text = $"Estimating Geometry ({method}) on {settings.Device}...";

            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                SceneResult result = new SceneResult();

                // If LGM, we skip point generation here as it happens in RunMeshing (ImageToLGM pipeline)
                if (isLGM)
                {
                    _statusLabel.Text = "LGM Workflow selected. Point cloud generation step skipped (handled in Meshing).";
                    return true;
                }

                switch (method)
                {
                    case ReconstructionMethod.Dust3r:
                        if (!_inference.IsLoaded)
                        {
                            Console.WriteLine("Dust3r model not found, falling back to Feature Matching SfM.");
                            goto case ReconstructionMethod.FeatureMatching;
                        }
                        _statusLabel.Text = "Estimating Geometry (Dust3r)...";
                        result = await Task.Run(() => _inference.ReconstructScene(_imagePaths));
                        break;

                    case ReconstructionMethod.FeatureMatching:
                        result = await RunFeatureMatchingSfmFallback("Feature Matching");
                        break;

                    case ReconstructionMethod.TripoSR:
                        _statusLabel.Text = "Estimating Geometry (TripoSR)...";
                        var tripoResult = await AIModels.AIModelManager.Instance.GenerateFromSingleImageAsync(
                            _imagePaths[0],
                            ImageTo3DModel.TripoSR,
                            msg => Application.Invoke((s, e) => _statusLabel.Text = msg));
                        if (tripoResult != null)
                        {
                            result = tripoResult;
                        }
                        break;

                    case ReconstructionMethod.Wonder3D:
                        _statusLabel.Text = "Estimating Geometry (Wonder3D)...";
                        var wonderResult = await AIModels.AIModelManager.Instance.GenerateFromSingleImageAsync(
                            _imagePaths[0],
                            ImageTo3DModel.Wonder3D,
                            msg => Application.Invoke((s, e) => _statusLabel.Text = msg));
                        if (wonderResult != null)
                        {
                            result = wonderResult;
                        }
                        break;

                    case ReconstructionMethod.Mast3r:
                        _statusLabel.Text = "Estimating Geometry (MASt3R)...";
                        // Process pending events and force GC before starting inference
                        // This ensures GTK reference tracking and Python memory are in sync
                        while (Application.EventsPending()) Application.RunIteration();
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        using (var mast3r = new Deep3DStudio.Model.Mast3rInference())
                        {
                            // Don't use Application.Invoke during inference - it can cause GTK reference issues
                            // Just log to console instead
                            mast3r.LogCallback = msg => Console.WriteLine($"[MASt3R] {msg}");
                            result = await Task.Run(() => mast3r.ReconstructScene(_imagePaths, useRetrieval: true));
                        }
                        // Force GC and process pending GTK events after inference
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        while (Application.EventsPending()) Application.RunIteration();
                        if (result.Meshes.Count == 0 || !result.Meshes.Any(m => m.Vertices.Count > 0))
                        {
                            Console.WriteLine("[MASt3R] No geometry generated, falling back to Feature Matching SfM.");
                            result = await RunFeatureMatchingSfmFallback("MASt3R");
                        }
                        break;

                    case ReconstructionMethod.Must3r:
                        _statusLabel.Text = "Estimating Geometry (MUSt3R)...";
                        // Process pending events and force GC before starting inference
                        // This ensures GTK reference tracking and Python memory are in sync
                        while (Application.EventsPending()) Application.RunIteration();
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        using (var must3r = new Deep3DStudio.Model.Must3rInference())
                        {
                            // Don't use Application.Invoke during inference - it can cause GTK reference issues
                            // Just log to console instead
                            must3r.LogCallback = msg => Console.WriteLine($"[MUSt3R] {msg}");
                            result = await Task.Run(() => must3r.ReconstructScene(_imagePaths, useRetrieval: true));
                        }
                        // Force GC and process pending GTK events after inference
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        while (Application.EventsPending()) Application.RunIteration();
                        if (result.Meshes.Count == 0 || !result.Meshes.Any(m => m.Vertices.Count > 0))
                        {
                            Console.WriteLine("[MUSt3R] No geometry generated, falling back to Feature Matching SfM.");
                            result = await RunFeatureMatchingSfmFallback("MUSt3R");
                        }
                        break;
                }

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Reconstruction failed. No points generated.";
                    return false;
                }

                ApplyPointCloudResultToScene(result);
                _statusLabel.Text = $"Point Cloud Complete: {_sceneGraph.GetObjectsOfType<PointCloudObject>().Sum(pc => pc.PointCount):N0} points, {result.Poses.Count} cameras.";

                return true;
            }
            catch (Exception ex)
            {
                _statusLabel.Text = "Error: " + ex.Message;
                Console.WriteLine(ex);
                return false;
            }
        }

        private async Task<SceneResult> RunFeatureMatchingSfmFallback(string sourceName)
        {
            _statusLabel.Text = sourceName == "Feature Matching"
                ? "Estimating Geometry (Feature Matching SfM)..."
                : $"{sourceName} failed, trying Feature Matching (SfM)...";

            var sfm = new Deep3DStudio.Model.SfM.SfMInference();
            var result = await Task.Run(() => sfm.ReconstructScene(_imagePaths));

            if (result.Meshes.Count > 0)
            {
                var sparseMesh = result.Meshes[0];
                Console.WriteLine($"Sparse SfM cloud: {sparseMesh.Vertices.Count} points, {sparseMesh.Colors.Count} colors");

                _statusLabel.Text = "Densifying Point Cloud...";
                while (Application.EventsPending()) Application.RunIteration();

                var denseMesh = await Task.Run(() => GenerateDensePointCloud(result));

                // Replace sparse with dense if we got significantly more points
                if (denseMesh.Vertices.Count > sparseMesh.Vertices.Count * 1.5)
                {
                    Console.WriteLine($"Densification: Using dense cloud ({denseMesh.Vertices.Count} pts) over sparse ({sparseMesh.Vertices.Count} pts)");
                    result.Meshes.Clear();
                    result.Meshes.Add(denseMesh);
                }
                else
                {
                    Console.WriteLine($"Densification: Keeping sparse cloud ({sparseMesh.Vertices.Count} pts) - dense only has {denseMesh.Vertices.Count} pts");
                }
            }

            return result;
        }

        private void ApplyPointCloudResultToScene(SceneResult result)
        {
            _lastSceneResult = result;
            PopulateDepthData(result);

            _sceneGraph.Clear();

            IniSettings.Instance.ShowPointCloud = true;
            IniSettings.Instance.ShowCameras = true;
            if (_pointsToggle != null) _pointsToggle.Active = true;
            if (_camerasToggle != null) _camerasToggle.Active = true;
            _viewport.ShowCameras = true;

            PointCloudObject? firstPc = null;
            for (int i = 0; i < result.Meshes.Count; i++)
            {
                var mesh = result.Meshes[i];
                Console.WriteLine($"PointCloud {i}: {mesh.Vertices.Count} points, {mesh.Colors.Count} colors");

                var pcObj = new PointCloudObject($"PointCloud_{i}", mesh);
                _sceneGraph.AddObject(pcObj);
                if (firstPc == null) firstPc = pcObj;
            }

            if (firstPc != null)
            {
                _sceneGraph.Select(firstPc);
            }

            AddCamerasToScene(result);

            _sceneTreeView.RefreshTree();

            var (sceneMin, sceneMax) = _sceneGraph.GetSceneBounds();
            Console.WriteLine($"Scene bounds: min({sceneMin.X:F2},{sceneMin.Y:F2},{sceneMin.Z:F2}) max({sceneMax.X:F2},{sceneMax.Y:F2},{sceneMax.Z:F2})");
            Console.WriteLine($"Scene contains {_sceneGraph.GetObjectsOfType<PointCloudObject>().Count()} point clouds, {_sceneGraph.GetVisibleObjects().Count()} visible objects");

            _viewport.FocusOnSelection();
            _viewport.QueueDraw();
            TryAutoRefineGeoreferenceFromScene("point cloud reconstruction");
        }

        private async Task RunMeshing()
        {
            if (_meshingInProgress)
            {
                _statusLabel.Text = "Meshing already in progress.";
                return;
            }

            _meshingInProgress = true;
            try
            {
                string workflow = _workflowCombo.ActiveText;
                bool isLGM = !string.IsNullOrEmpty(workflow) && workflow.Contains("LGM");

                var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
                Console.WriteLine($"[Meshing] Selected point clouds from sceneGraph: {selectedPointClouds.Count}");

                if (!isLGM && selectedPointClouds.Count == 0 && _sceneTreeView != null)
                {
                    selectedPointClouds = _sceneTreeView.GetSelectedObjects().OfType<PointCloudObject>().ToList();
                    Console.WriteLine($"[Meshing] Selected point clouds from treeView: {selectedPointClouds.Count}");
                }

                // Fallback: use all point clouds in the scene if none are selected
                if (!isLGM && selectedPointClouds.Count == 0)
                {
                    selectedPointClouds = _sceneGraph.GetObjectsOfType<PointCloudObject>().ToList();
                    Console.WriteLine($"[Meshing] Fallback to all point clouds in scene: {selectedPointClouds.Count}");
                }

                if (!isLGM && selectedPointClouds.Count == 0)
                {
                    ShowMessage("No point cloud found. Please import a point cloud first.");
                    return;
                }

                if (!isLGM)
                {
                    int selectedTotalPoints = selectedPointClouds.Sum(pc => pc.PointCount);
                    int selectedVisiblePoints = selectedPointClouds.Sum(pc => pc.VisiblePointCount);
                    if (selectedVisiblePoints == 0 && selectedTotalPoints > 0)
                    {
                        ShowMessage("No Visible Points", "All selected point clouds currently expose 0 visible points. Increase the Visible slider before meshing.");
                        return;
                    }
                }

                // Log point cloud stats
                int totalPoints = selectedPointClouds.Sum(pc => pc.PointCount);
                int visiblePoints = selectedPointClouds.Sum(pc => pc.VisiblePointCount);
                Console.WriteLine($"[Meshing] Total points to mesh: {visiblePoints}/{totalPoints} visible");

                _statusLabel.Text = $"Meshing ({workflow})...";
                while (Application.EventsPending()) Application.RunIteration();

                try
                {
                    var cancellationToken = UI.ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var meshingAlgo = IniSettings.Instance.MeshingAlgo;

                    // Override if workflow implies a specific AI method
                    if (isLGM) meshingAlgo = MeshingAlgorithm.LGM;

                    if (meshingAlgo == MeshingAlgorithm.LGM)
                    {
                        ShowMessage("LGM is image-based. Use the Image -> LGM workflow instead of point cloud meshing.");
                        return;
                    }

                    if (meshingAlgo == MeshingAlgorithm.DeepMeshPrior ||
                        meshingAlgo == MeshingAlgorithm.TripoSF ||
                        meshingAlgo == MeshingAlgorithm.GaussianSDF)
                    {
                        // Use Poisson reconstruction for better topology before refinement
                        var baseMesh = await Task.Run(() => GenerateMeshFromPointClouds(selectedPointClouds, MeshingAlgorithm.Poisson));
                        if (baseMesh == null || baseMesh.Vertices.Count == 0)
                        {
                            _statusLabel.Text = "Base meshing failed.";
                            return;
                        }

                        var refinedMesh = await RefineMeshAsync(baseMesh, meshingAlgo, cancellationToken);
                        if (refinedMesh == null || refinedMesh.Vertices.Count == 0)
                        {
                            _statusLabel.Text = "AI refinement did not return a result.";
                            return;
                        }

                        var aiObj = new MeshObject("Refined Mesh", refinedMesh);
                        _sceneGraph.AddObject(aiObj);
                        _sceneGraph.Select(aiObj);
                        TryAutoRefineGeoreferenceFromScene("AI meshing refinement");
                        _viewport.FocusOnSelection();
                        _statusLabel.Text = "AI meshing complete.";
                        _sceneTreeView.RefreshTree();
                        _viewport.QueueDraw();
                        return;
                    }

                    int maxRes = (!string.IsNullOrEmpty(workflow) && workflow.Contains("Interior")) ? 500 : 200;
                    _statusLabel.Text = $"Meshing using {meshingAlgo}...";

                    var meshedResult = await Task.Run(() => GenerateMeshFromPointClouds(selectedPointClouds, meshingAlgo, maxRes));

                    Console.WriteLine($"Meshing result: {meshedResult.Vertices.Count} vertices, {meshedResult.Indices.Count} indices ({meshedResult.Indices.Count / 3} triangles)");

                    if (meshedResult.Vertices.Count > 0)
                    {
                        var meshObj = new MeshObject("Reconstructed Mesh", meshedResult);
                        _sceneGraph.AddObject(meshObj);
                        _sceneGraph.Select(meshObj);
                        TryAutoRefineGeoreferenceFromScene("meshing");
                        _viewport.FocusOnSelection();
                    }
                    _statusLabel.Text = "Meshing Complete.";

                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();

                    var (meshes, pcs, cams, verts, tris) = _sceneGraph.GetStatistics();
                    _statusLabel.Text += $" | {meshes} meshes, {verts:N0} vertices";
                }
                catch (OperationCanceledException)
                {
                    _statusLabel.Text = "Meshing cancelled.";
                }
                catch (Exception ex)
                {
                    _statusLabel.Text = "Error during meshing: " + ex.Message;
                    Console.WriteLine(ex);
                    if (UI.ProgressDialog.Instance.IsVisible)
                    {
                        UI.ProgressDialog.Instance.Fail(ex);
                    }
                }
            }
            finally
            {
                _meshingInProgress = false;
            }
        }

        private MeshData GenerateMeshFromPointClouds(List<PointCloudObject> pointClouds, MeshingAlgorithm algorithm, int maxRes = 200)
        {
            Console.WriteLine($"[Meshing] GenerateMeshFromPointClouds: {pointClouds.Count} point clouds, algorithm={algorithm}, maxRes={maxRes}");

            var meshes = pointClouds.Select(pc => ToMeshData(pc, visibleOnly: true)).ToList();
            int totalVerts = meshes.Sum(m => m.Vertices.Count);
            Console.WriteLine($"[Meshing] Total vertices from point clouds: {totalVerts}");
            if (totalVerts == 0)
            {
                Console.WriteLine("[Meshing] No visible points available for meshing.");
                return new MeshData();
            }

            var (grid, colorGrid, min, voxelSize) = VoxelizePoints(meshes, maxRes);
            int gridX = grid.GetLength(0);
            int gridY = grid.GetLength(1);
            int gridZ = grid.GetLength(2);
            Console.WriteLine($"[Meshing] Voxel grid: {gridX}x{gridY}x{gridZ}, origin=({min.X:F2},{min.Y:F2},{min.Z:F2}), voxelSize={voxelSize:F4}");

            // Count non-zero voxels
            int nonZeroVoxels = 0;
            for (int x = 0; x < gridX; x++)
                for (int y = 0; y < gridY; y++)
                    for (int z = 0; z < gridZ; z++)
                        if (grid[x, y, z] > 0.01f) nonZeroVoxels++;
            Console.WriteLine($"[Meshing] Non-zero voxels: {nonZeroVoxels}");

            IMesher mesher = GetMesher(algorithm);
            Console.WriteLine($"[Meshing] Running {mesher.GetType().Name}...");

            var result = mesher.GenerateMesh(grid, min, voxelSize, 0.5f);
            Console.WriteLine($"[Meshing] Result: {result.Vertices.Count} vertices, {result.Indices.Count} indices");

            ApplyColorsFromGrid(result, colorGrid, min, voxelSize);
            result = PostProcessMesh(result, voxelSize);

            return result;
        }

        private static void ApplyColorsFromGrid(MeshData mesh, OpenTK.Mathematics.Vector3[,,]? colorGrid, OpenTK.Mathematics.Vector3 origin, float voxelSize)
        {
            if (colorGrid == null || mesh.Vertices.Count == 0) return;
            int w = colorGrid.GetLength(0), h = colorGrid.GetLength(1), d = colorGrid.GetLength(2);
            mesh.Colors.Clear();
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                int gx = Math.Clamp((int)((v.X - origin.X) / voxelSize), 0, w - 1);
                int gy = Math.Clamp((int)((v.Y - origin.Y) / voxelSize), 0, h - 1);
                int gz = Math.Clamp((int)((v.Z - origin.Z) / voxelSize), 0, d - 1);
                mesh.Colors.Add(colorGrid[gx, gy, gz]);
            }
        }

        private static MeshData PostProcessMesh(MeshData mesh, float voxelSize)
        {
            if (mesh.Vertices.Count == 0 || mesh.Indices.Count == 0)
                return mesh;

            mesh = MeshCleaningTools.RemoveOversizedTriangles(mesh, voxelSize * 10.0f);
            mesh = MeshCleaningTools.RemoveDegenerateTriangles(mesh);
            mesh = MeshCleaningTools.RemoveSliverTriangles(mesh, 0.01f);
            mesh = MeshCleaningTools.RemoveSmallComponentsByRatio(mesh, 0.01f);
            return mesh;
        }

        private async Task<MeshData?> RefineMeshAsync(
            MeshData inputMesh,
            MeshingAlgorithm algorithm,
            System.Threading.CancellationToken cancellationToken = default)
        {
            switch (algorithm)
            {
                case MeshingAlgorithm.DeepMeshPrior:
                    var deepMeshPrior = new DeepMeshPriorMesher();
                    return await deepMeshPrior.RefineMeshAsync(inputMesh, (status, progress) =>
                        Application.Invoke((s, e) => _statusLabel.Text = status),
                        cancellationToken);

                case MeshingAlgorithm.TripoSF:
                    return await Task.Run(() =>
                    {
                        using var tripo = new AIModels.TripoSFInference();
                        cancellationToken.ThrowIfCancellationRequested();
                        return tripo.RefineMesh(inputMesh, cancellationToken);
                    }, cancellationToken);

                case MeshingAlgorithm.GaussianSDF:
                    var gaussian = new GaussianSDFRefiner();
                    return await gaussian.RefineMeshAsync(inputMesh, (status, progress) =>
                        Application.Invoke((s, e) => _statusLabel.Text = status),
                        cancellationToken);
            }

            return inputMesh;
        }

        private MeshData ToMeshData(PointCloudObject pointCloud, bool visibleOnly = false)
        {
            return PointCloudOperations.ToMeshData(pointCloud, visibleOnly);
        }

        private async Task<bool> RunAIMeshingAsync(MeshingAlgorithm algorithm, string? contextLabel = null)
        {
            var manager = AIModels.AIModelManager.Instance;
            AIModels.WorkflowPipeline pipeline;

            switch (algorithm)
            {
                case MeshingAlgorithm.DeepMeshPrior:
                    pipeline = new AIModels.WorkflowPipeline
                    {
                        Name = "DeepMeshPrior Optimization",
                        Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.DeepMeshPriorRefinement }
                    };
                    break;
                case MeshingAlgorithm.TripoSF:
                    pipeline = new AIModels.WorkflowPipeline
                    {
                        Name = "TripoSF Mesh Refinement",
                        Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.TripoSFRefinement }
                    };
                    break;
                case MeshingAlgorithm.LGM:
                    pipeline = AIModels.WorkflowPipeline.ImageToLGM;
                    break;
                case MeshingAlgorithm.GaussianSDF:
                    pipeline = new AIModels.WorkflowPipeline
                    {
                        Name = "GaussianSDF Mesh Refinement",
                        Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.GaussianSDFRefinement }
                    };
                    break;
                default:
                    return false;
            }

            string label = contextLabel ?? pipeline.Name;
            _statusLabel.Text = $"{label}...";

            // Process pending GTK events before starting to ensure clean state
            while (Application.EventsPending()) Application.RunIteration();

            var cancellationToken = UI.ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
            var result = await manager.ExecuteWorkflowAsync(
                pipeline,
                _imagePaths,
                _lastSceneResult,
                (message, _) => Application.Invoke((s, e) => _statusLabel.Text = message),
                cancellationToken
            );

            // Process pending GTK events after workflow to ensure queued Application.Invoke calls
            // are processed before continuing - prevents GTK reference tracking issues
            while (Application.EventsPending()) Application.RunIteration();

            if (result != null)
            {
                Application.Invoke((s, e) =>
                {
                    _lastSceneResult = result;
                    UpdateSceneFromResult(result);
                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                });
                return true;
            }

            return false;
        }

        private void AddCamerasToScene(SceneResult result)
        {
            var camerasGroup = new GroupObject("Cameras");
            _sceneGraph.AddObject(camerasGroup);

            for (int i = 0; i < result.Poses.Count; i++)
            {
                var pose = result.Poses[i];
                var camObj = new CameraObject($"Camera {i + 1}", pose);
                _sceneGraph.AddObject(camObj, camerasGroup);
            }
        }
    }
}
