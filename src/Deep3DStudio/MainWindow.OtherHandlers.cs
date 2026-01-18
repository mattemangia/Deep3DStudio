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
            if (_rgbColorToggle != null) _rgbColorToggle.Active = s.PointCloudColor == PointCloudColorMode.RGB;
            if (_depthColorToggle != null) _depthColorToggle.Active = s.PointCloudColor == PointCloudColorMode.DistanceMap;
            _viewport.QueueDraw();
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
                        _statusLabel.Text = "Estimating Geometry (Feature Matching SfM)...";
                        var sfm = new Deep3DStudio.Model.SfM.SfMInference();
                        result = await Task.Run(() => sfm.ReconstructScene(_imagePaths));

                        // Densify the sparse SfM point cloud
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
                        // Process pending events before starting inference to ensure GTK state is clean
                        while (Application.EventsPending()) Application.RunIteration();
                        using (var mast3r = new Deep3DStudio.Model.Mast3rInference())
                        {
                            mast3r.LogCallback = msg => Application.Invoke((s, e) => _statusLabel.Text = msg);
                            result = await Task.Run(() => mast3r.ReconstructScene(_imagePaths, useRetrieval: true));
                        }
                        // Process pending GTK events to ensure queued Application.Invoke calls are processed
                        // This prevents GTK reference tracking issues with Python interop
                        while (Application.EventsPending()) Application.RunIteration();
                        break;

                    case ReconstructionMethod.Must3r:
                        _statusLabel.Text = "Estimating Geometry (MUSt3R)...";
                        // Process pending events before starting inference to ensure GTK state is clean
                        while (Application.EventsPending()) Application.RunIteration();
                        using (var must3r = new Deep3DStudio.Model.Must3rInference())
                        {
                            must3r.LogCallback = msg => Application.Invoke((s, e) => _statusLabel.Text = msg);
                            result = await Task.Run(() => must3r.ReconstructScene(_imagePaths, useRetrieval: true));
                        }
                        // Process pending GTK events to ensure queued Application.Invoke calls are processed
                        // This prevents GTK reference tracking issues with Python interop
                        while (Application.EventsPending()) Application.RunIteration();
                        break;
                }

                if (result.Meshes.Count == 0)
                {
                    _statusLabel.Text = "Reconstruction failed. No points generated.";
                    return false;
                }

                _lastSceneResult = result;
                PopulateDepthData(result);

                // Update Scene with Point Cloud
                _sceneGraph.Clear();

                // Add Point Clouds (from result.Meshes acting as points)
                int totalPoints = 0;
                for (int i = 0; i < result.Meshes.Count; i++)
                {
                    var mesh = result.Meshes[i];
                    Console.WriteLine($"PointCloud {i}: {mesh.Vertices.Count} points, {mesh.Colors.Count} colors");
                    totalPoints += mesh.Vertices.Count;

                    var pcObj = new PointCloudObject($"PointCloud_{i}", mesh);
                    _sceneGraph.AddObject(pcObj);
                }

                AddCamerasToScene(result);

                _sceneTreeView.RefreshTree();

                // Log scene bounds for debugging
                var (sceneMin, sceneMax) = _sceneGraph.GetSceneBounds();
                Console.WriteLine($"Scene bounds: min({sceneMin.X:F2},{sceneMin.Y:F2},{sceneMin.Z:F2}) max({sceneMax.X:F2},{sceneMax.Y:F2},{sceneMax.Z:F2})");
                Console.WriteLine($"Scene contains {_sceneGraph.GetObjectsOfType<PointCloudObject>().Count()} point clouds, {_sceneGraph.GetVisibleObjects().Count()} visible objects");

                // Auto-focus on the generated point cloud
                _viewport.FocusOnSelection();
                _viewport.QueueDraw();
                _statusLabel.Text = $"Point Cloud Complete: {totalPoints:N0} points, {result.Poses.Count} cameras.";

                return true;
            }
            catch (Exception ex)
            {
                _statusLabel.Text = "Error: " + ex.Message;
                Console.WriteLine(ex);
                return false;
            }
        }

        private async Task RunMeshing()
        {
            string workflow = _workflowCombo.ActiveText;
            bool isLGM = !string.IsNullOrEmpty(workflow) && workflow.Contains("LGM");

            if (!isLGM && (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0))
            {
                // Try to build result from current scene if possible?
                // For now, require point cloud generation first
                ShowMessage("No point cloud data available. Please generate point cloud first or import data.");
                return;
            }

            _statusLabel.Text = $"Meshing ({workflow})...";
            while (Application.EventsPending()) Application.RunIteration();

            try
            {
                var meshingAlgo = IniSettings.Instance.MeshingAlgo;

                // Override if workflow implies a specific AI method
                if (isLGM) meshingAlgo = MeshingAlgorithm.LGM;

                if ((meshingAlgo == MeshingAlgorithm.DeepMeshPrior || meshingAlgo == MeshingAlgorithm.TripoSF) &&
                    (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0))
                {
                    ShowMessage("No point cloud data available. Please generate a point cloud first or import data.");
                    return;
                }

                if (meshingAlgo == MeshingAlgorithm.DeepMeshPrior ||
                    meshingAlgo == MeshingAlgorithm.TripoSF ||
                    meshingAlgo == MeshingAlgorithm.LGM)
                {
                    bool aiMeshSuccess = await RunAIMeshingAsync(meshingAlgo, "AI Meshing");
                    if (!aiMeshSuccess)
                    {
                        _statusLabel.Text = "AI meshing did not return a result.";
                    }
                    return;
                }

                // Remove reconstruction meshes generated by previous runs to avoid clutter while keeping imported assets.
                // For simplicity, operations below use the data stored in _lastSceneResult.

                if (workflow == "Dust3r (Fast)")
                {
                    _statusLabel.Text = $"Meshing using {IniSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                        var (grid, min, size) = VoxelizePoints(_lastSceneResult.Meshes);
                        IMesher mesher = GetMesher(IniSettings.Instance.MeshingAlgo);
                        return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    Console.WriteLine($"Meshing result: {meshedResult.Vertices.Count} vertices, {meshedResult.Indices.Count} indices ({meshedResult.Indices.Count / 3} triangles)");

                    if (meshedResult.Vertices.Count > 0)
                    {
                        var meshObj = new MeshObject("Reconstructed Mesh", meshedResult);
                        _sceneGraph.AddObject(meshObj);
                        _viewport.FocusOnSelection();
                    }
                    _statusLabel.Text = "Meshing Complete.";
                }
                else if (workflow == "Interior Scan")
                {
                    _statusLabel.Text = $"Meshing Interior (High Res) using {IniSettings.Instance.MeshingAlgo}...";

                    var meshedResult = await Task.Run(() => {
                        var (grid, min, size) = VoxelizePoints(_lastSceneResult.Meshes, 500);
                        IMesher mesher = GetMesher(IniSettings.Instance.MeshingAlgo);
                        return mesher.GenerateMesh(grid, min, size, 0.5f);
                    });

                    Console.WriteLine($"Interior Meshing result: {meshedResult.Vertices.Count} vertices, {meshedResult.Indices.Count} indices ({meshedResult.Indices.Count / 3} triangles)");

                    if (meshedResult.Vertices.Count > 0)
                    {
                        var meshObj = new MeshObject("Interior Mesh", meshedResult);
                        _sceneGraph.AddObject(meshObj);
                        _viewport.FocusOnSelection();
                    }
                    _statusLabel.Text = "Interior Meshing Complete.";
                }
                else
                {
                    _statusLabel.Text = "Initializing NeRF Voxel Grid...";
                    var nerf = new VoxelGridNeRF();

                    var nerfSettings = IniSettings.Instance;
                    await Task.Run(() =>
                    {
                        nerf.InitializeFromMesh(_lastSceneResult.Meshes);
                        nerf.Train(_lastSceneResult.Poses, iterations: nerfSettings.NeRFIterations);
                    });

                    _statusLabel.Text = $"Extracting NeRF Mesh ({IniSettings.Instance.MeshingAlgo})...";

                    var nerfMesh = await Task.Run(() => {
                        return nerf.GetMesh(GetMesher(IniSettings.Instance.MeshingAlgo));
                    });

                    Console.WriteLine($"NeRF Meshing result: {nerfMesh.Vertices.Count} vertices, {nerfMesh.Indices.Count} indices ({nerfMesh.Indices.Count / 3} triangles)");

                    if (nerfMesh.Vertices.Count > 0)
                    {
                        var meshObj = new MeshObject("NeRF Mesh", nerfMesh);
                        _sceneGraph.AddObject(meshObj);
                        _viewport.FocusOnSelection();
                    }
                    _statusLabel.Text = "NeRF Meshing Complete.";
                }

                _sceneTreeView.RefreshTree();
                _viewport.QueueDraw();

                var (meshes, pcs, cams, verts, tris) = _sceneGraph.GetStatistics();
                _statusLabel.Text += $" | {meshes} meshes, {verts:N0} vertices";
            }
            catch (Exception ex)
            {
                _statusLabel.Text = "Error during meshing: " + ex.Message;
                Console.WriteLine(ex);
            }
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

            var result = await manager.ExecuteWorkflowAsync(
                pipeline,
                _imagePaths,
                _lastSceneResult,
                (message, _) => Application.Invoke((s, e) => _statusLabel.Text = message)
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
