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
        private async void OnAIRefine(object? sender, EventArgs e)
        {
            var refineSetting = IniSettings.Instance.MeshRefinement;
            switch (refineSetting)
            {
                case MeshRefinementMethod.DeepMeshPrior:
                    await TryRefineFromSelectionAsync(MeshRefinementMethod.DeepMeshPrior);
                    break;
                case MeshRefinementMethod.TripoSF:
                    await TryRefineFromSelectionAsync(MeshRefinementMethod.TripoSF);
                    break;
                case MeshRefinementMethod.GaussianSDF:
                    await TryRefineFromSelectionAsync(MeshRefinementMethod.GaussianSDF);
                    break;
                default:
                    ShowMessage("No refinement method selected", "Please choose a mesh refinement model in Settings.");
                    break;
            }
        }

        private async Task RefineSelectedMeshesAsync(List<MeshObject> meshes, MeshRefinementMethod method)
        {
            string label = method switch
            {
                MeshRefinementMethod.DeepMeshPrior => "DeepMeshPrior Optimization",
                MeshRefinementMethod.TripoSF => "TripoSF Refinement",
                MeshRefinementMethod.GaussianSDF => "GaussianSDF Refinement",
                _ => "Mesh Refinement"
            };

            _statusLabel.Text = $"{label}...";
            try
            {
                foreach (var meshObj in meshes)
                {
                    MeshData? refined = null;
                    switch (method)
                    {
                        case MeshRefinementMethod.DeepMeshPrior:
                            var deep = new DeepMeshPriorMesher();
                            refined = await deep.RefineMeshAsync(meshObj.MeshData, (status, progress) =>
                                Application.Invoke((s, e) => _statusLabel.Text = status));
                            break;
                        case MeshRefinementMethod.TripoSF:
                            refined = await Task.Run(() =>
                            {
                                using var tripo = new AIModels.TripoSFInference();
                                return tripo.RefineMesh(meshObj.MeshData);
                            });
                            break;
                        case MeshRefinementMethod.GaussianSDF:
                            var gaussian = new GaussianSDFRefiner();
                            refined = await gaussian.RefineMeshAsync(meshObj.MeshData, (status, progress) =>
                                Application.Invoke((s, e) => _statusLabel.Text = status));
                            break;
                    }

                    if (refined != null && refined.Vertices.Count > 0)
                    {
                        meshObj.MeshData = refined;
                        meshObj.UpdateBounds();
                    }
                }

                _viewport.QueueDraw();
                _sceneTreeView.RefreshTree();
                _statusLabel.Text = $"{label} complete.";
            }
            catch (Exception ex)
            {
                ShowMessage("Refinement failed", ex.Message);
                _statusLabel.Text = $"{label} failed.";
            }
        }

        private void OnTripoSRGenerate(object? sender, EventArgs e)
        {
            if (_imagePaths.Count == 0)
            {
                ShowMessage("No Images", "Please load at least one image for TripoSR generation.");
                return;
            }

            _statusLabel.Text = "Running TripoSR single-image 3D generation...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.ImageToTripoSR);
        }

        private void OnLGMGenerate(object? sender, EventArgs e)
        {
            if (_imagePaths.Count == 0)
            {
                ShowMessage("No Images", "Please load at least one image for LGM generation.");
                return;
            }

            _statusLabel.Text = "Running LGM high-quality 3D generation...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.ImageToLGM);
        }

        private void OnWonder3DGenerate(object? sender, EventArgs e)
        {
            if (_imagePaths.Count == 0)
            {
                ShowMessage("No Images", "Please load at least one image for Wonder3D generation.");
                return;
            }

            _statusLabel.Text = "Running Wonder3D single-image multi-view 3D generation...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.ImageToWonder3D);
        }

        private void OnDeepMeshPriorRefine(object? sender, EventArgs e)
        {
            _ = RunRefineWithFallbackAsync(MeshRefinementMethod.DeepMeshPrior,
                "DeepMeshPrior Optimization",
                new AIModels.WorkflowPipeline
                {
                    Name = "DeepMeshPrior Optimization",
                    Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.DeepMeshPriorRefinement }
                });
        }

        private void OnTripoSFRefine(object? sender, EventArgs e)
        {
            _ = RunRefineWithFallbackAsync(MeshRefinementMethod.TripoSF,
                "TripoSF Refinement",
                new AIModels.WorkflowPipeline
                {
                    Name = "TripoSF Refinement",
                    Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.TripoSFRefinement }
                });
        }

        private void OnGaussianSDFRefine(object? sender, EventArgs e)
        {
            _ = RunRefineWithFallbackAsync(MeshRefinementMethod.GaussianSDF,
                "GaussianSDF Refinement",
                new AIModels.WorkflowPipeline
                {
                    Name = "GaussianSDF Refinement",
                    Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.GaussianSDFRefinement }
                });
        }

        private async Task RunRefineWithFallbackAsync(MeshRefinementMethod method, string label, AIModels.WorkflowPipeline fallbackPipeline)
        {
            if (_meshingInProgress)
            {
                _statusLabel.Text = "Meshing already in progress.";
                return;
            }

            bool refined = await TryRefineFromSelectionAsync(method);
            if (refined) return;

            if (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0)
            {
                ShowMessage("No Mesh", $"Please generate or select a mesh to refine with {label}.");
                return;
            }

            _statusLabel.Text = $"Running {label}...";
            RunAIWorkflowAsync(fallbackPipeline);
        }

        private async Task<bool> TryRefineFromSelectionAsync(MeshRefinementMethod method)
        {
            if (_meshingInProgress)
            {
                _statusLabel.Text = "Meshing already in progress.";
                return true;
            }

            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count > 0)
            {
                await RefineSelectedMeshesAsync(selectedMeshes, method);
                return true;
            }

            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (selectedPointClouds.Count == 0 && _sceneTreeView != null)
            {
                selectedPointClouds = _sceneTreeView.GetSelectedObjects().OfType<PointCloudObject>().ToList();
            }

            if (selectedPointClouds.Count == 0)
            {
                ShowMessage("No selection", "Please select a mesh or point cloud to refine.");
                return false;
            }

            MeshingAlgorithm baseAlgo = IniSettings.Instance.MeshingAlgo;
            if (baseAlgo == MeshingAlgorithm.DeepMeshPrior ||
                baseAlgo == MeshingAlgorithm.TripoSF ||
                baseAlgo == MeshingAlgorithm.GaussianSDF ||
                baseAlgo == MeshingAlgorithm.LGM)
            {
                if (baseAlgo == MeshingAlgorithm.LGM)
                {
                    _statusLabel.Text = "LGM is image-based. Using MarchingCubes for point cloud meshing.";
                    while (Application.EventsPending()) Application.RunIteration();
                }
                baseAlgo = MeshingAlgorithm.MarchingCubes;
            }

            _statusLabel.Text = $"Meshing point cloud ({baseAlgo}) for refinement...";
            while (Application.EventsPending()) Application.RunIteration();

            var baseMesh = await Task.Run(() => GenerateMeshFromPointClouds(selectedPointClouds, baseAlgo));
            if (baseMesh == null || baseMesh.Vertices.Count == 0 || baseMesh.Indices.Count == 0)
            {
                ShowMessage("Meshing failed", "Point cloud meshing produced no faces. Try a different meshing algorithm in Settings.");
                _statusLabel.Text = "Point cloud meshing failed.";
                return true;
            }

            MeshingAlgorithm refineAlgo = method switch
            {
                MeshRefinementMethod.DeepMeshPrior => MeshingAlgorithm.DeepMeshPrior,
                MeshRefinementMethod.TripoSF => MeshingAlgorithm.TripoSF,
                MeshRefinementMethod.GaussianSDF => MeshingAlgorithm.GaussianSDF,
                _ => MeshingAlgorithm.MarchingCubes
            };

            var cancellationToken = UI.ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
            var refinedMesh = await RefineMeshAsync(baseMesh, refineAlgo, cancellationToken);
            if (refinedMesh == null || refinedMesh.Vertices.Count == 0 || refinedMesh.Indices.Count == 0)
            {
                ShowMessage("Refinement failed", "AI refinement returned no geometry.");
                _statusLabel.Text = "Refinement failed.";
                return true;
            }

            var refinedObj = new MeshObject("Refined Mesh", refinedMesh);
            _sceneGraph.AddObject(refinedObj);
            _sceneGraph.Select(refinedObj);
            _sceneTreeView.RefreshTree();
            _viewport.FocusOnSelection();
            _viewport.QueueDraw();
            _statusLabel.Text = "Refinement complete.";
            return true;
        }

        private void OnMultiViewDeepMeshPriorWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", $"Please load at least 2 images for {GetCurrentEngineName()} reconstruction.");
                return;
            }

            _statusLabel.Text = $"Running {GetCurrentEngineName()} → DeepMeshPrior workflow...";
            // Build pipeline using settings-based reconstruction step
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = $"{GetCurrentEngineName()} → DeepMeshPrior",
                Steps = new List<AIModels.WorkflowStep>
                {
                    AIModels.WorkflowStep.LoadImages,
                    GetReconstructionStep(),
                    AIModels.WorkflowStep.DeepMeshPriorRefinement
                }
            };
            RunAIWorkflowAsync(pipeline);
        }

        private void OnMultiViewNeRFDeepMeshPriorWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", "Please load at least 2 images for this workflow.");
                return;
            }

            _statusLabel.Text = $"Running {GetCurrentEngineName()} → NeRF → DeepMeshPrior workflow...";
            // Build pipeline using settings-based reconstruction step
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = $"{GetCurrentEngineName()} → NeRF → DeepMeshPrior",
                Steps = new List<AIModels.WorkflowStep>
                {
                    AIModels.WorkflowStep.LoadImages,
                    GetReconstructionStep(),
                    AIModels.WorkflowStep.NeRFRefinement,
                    AIModels.WorkflowStep.DeepMeshPriorRefinement
                }
            };
            RunAIWorkflowAsync(pipeline);
        }

        private void OnFullPipelineWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", "Please load at least 2 images for the full pipeline.");
                return;
            }

            _statusLabel.Text = "Running full pipeline (reconstruction + meshing)...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.FullPipeline);
        }

        private void OnSfMToAIWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", "Please load at least 2 images for SfM reconstruction.");
                return;
            }

            _statusLabel.Text = "Running SfM → AI refinement workflow...";
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = "SfM to AI Refinement",
                Steps = new List<AIModels.WorkflowStep>
                {
                    AIModels.WorkflowStep.SfMReconstruction,
                    AIModels.WorkflowStep.TripoSFRefinement,
                    AIModels.WorkflowStep.DeepMeshPriorRefinement
                }
            };
            RunAIWorkflowAsync(pipeline);
        }

        private void OnPointCloudMergeWorkflow(object? sender, EventArgs e)
        {
            if (_lastSceneResult == null || _lastSceneResult.Meshes.Count < 2)
            {
                ShowMessage("Need More Point Clouds", "Please generate at least 2 point clouds to merge.");
                return;
            }

            _statusLabel.Text = "Running point cloud merge and refinement...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.PointCloudMergeRefine);
        }

        private void OnAIModelSettings(object? sender, EventArgs e)
        {
            var dialog = new AIModelSettingsDialog(this, IniSettings.Instance);
            if (dialog.Run() == (int)ResponseType.Ok)
            {
                dialog.ApplySettings();
                IniSettings.Instance.Save();
                _statusLabel.Text = "AI model settings saved.";
            }
            dialog.Destroy();
        }

        private async void RunAIWorkflowAsync(AIModels.WorkflowPipeline pipeline)
        {
            // Process pending GTK events before starting to ensure clean state
            while (Application.EventsPending()) Application.RunIteration();

            // Start the progress dialog
            UI.ProgressDialog.Instance.Start($"Running {pipeline.Name}...", UI.OperationType.Processing);
            var cancellationToken = UI.ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;

            try
            {
                var manager = AIModels.AIModelManager.Instance;
                var result = await manager.ExecuteWorkflowAsync(
                    pipeline,
                    _imagePaths,
                    _lastSceneResult,
                    (message, progress) => Application.Invoke((s, e) => {
                        _statusLabel.Text = message;
                        if (UI.ProgressDialog.Instance.IsVisible)
                        {
                            UI.ProgressDialog.Instance.Update(progress, message);
                        }
                    }),
                    cancellationToken
                );

                // Process pending GTK events after workflow to ensure queued Application.Invoke calls
                // are processed before continuing - prevents GTK reference tracking issues
                while (Application.EventsPending()) Application.RunIteration();

                if (result != null)
                {
                    bool hasGeometry = result.Meshes.Any(m => m.Vertices.Count > 0);
                    if (!hasGeometry)
                    {
                        Application.Invoke((s, e) =>
                        {
                            UI.ProgressDialog.Instance.Fail(new Exception("Workflow completed but no geometry was generated."));
                            _statusLabel.Text = $"Workflow '{pipeline.Name}' failed.";
                        });
                        return;
                    }
                    Application.Invoke((s, e) =>
                    {
                        _lastSceneResult = result;
                        UpdateSceneFromResult(result);
                        if (cancellationToken.IsCancellationRequested)
                        {
                            _statusLabel.Text = $"Workflow '{pipeline.Name}' cancelled.";
                        }
                        else
                        {
                            _statusLabel.Text = $"Workflow '{pipeline.Name}' completed successfully.";
                            UI.ProgressDialog.Instance.Complete();
                        }
                    });
                }
                else
                {
                    Application.Invoke((s, e) =>
                    {
                        UI.ProgressDialog.Instance.Fail(new Exception("Workflow completed but no result was generated."));
                    });
                }
            }
            catch (OperationCanceledException)
            {
                Application.Invoke((s, e) =>
                {
                    _statusLabel.Text = "Workflow cancelled.";
                });
            }
            catch (Exception ex)
            {
                Application.Invoke((s, e) =>
                {
                    UI.ProgressDialog.Instance.Fail(ex);
                    _statusLabel.Text = "Workflow failed.";
                });
            }

            // Final event processing to ensure all UI updates are applied
            while (Application.EventsPending()) Application.RunIteration();
        }

        private void UpdateSceneFromResult(SceneResult result)
        {
            // Add new meshes to scene
            foreach (var mesh in result.Meshes)
            {
                var meshObj = new Scene.MeshObject($"Mesh_{_sceneGraph.GetAllObjects().Count}", mesh);
                _sceneGraph.AddObject(meshObj);
            }

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _isDirty = true;
            UpdateTitle();
        }

        /// <summary>
        /// Run a single workflow step standalone (without running the full workflow).
        /// This allows users to manually control each step of the pipeline.
        /// </summary>
        private async void OnRunSingleStep(AIModels.WorkflowStep step)
        {
            if (step == AIModels.WorkflowStep.PoissonReconstruction)
            {
                await RunMeshing();
                return;
            }

            // Validate prerequisites for each step
            switch (step)
            {
                case AIModels.WorkflowStep.Dust3rReconstruction:
                case AIModels.WorkflowStep.Mast3rReconstruction:
                case AIModels.WorkflowStep.Must3rReconstruction:
                case AIModels.WorkflowStep.SfMReconstruction:
                    if (_imagePaths.Count < 2)
                    {
                        ShowMessage("Need More Images", "Please load at least 2 images for reconstruction.");
                        return;
                    }
                    break;

                case AIModels.WorkflowStep.TripoSRGeneration:
                case AIModels.WorkflowStep.LGMGeneration:
                case AIModels.WorkflowStep.Wonder3DGeneration:
                    if (_imagePaths.Count == 0)
                    {
                        ShowMessage("No Images", "Please load at least one image.");
                        return;
                    }
                    break;

                case AIModels.WorkflowStep.NeRFRefinement:
                case AIModels.WorkflowStep.DeepMeshPriorRefinement:
                case AIModels.WorkflowStep.TripoSFRefinement:
                case AIModels.WorkflowStep.GaussianSDFRefinement:
                case AIModels.WorkflowStep.MeshSmoothing:
                case AIModels.WorkflowStep.MeshDecimation:
                case AIModels.WorkflowStep.UniRigAutoRig:
                    if (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0)
                    {
                        ShowMessage("No Geometry", "Please generate or load geometry first.");
                        return;
                    }
                    break;
            }

            string stepName = GetStepDisplayName(step);
            _statusLabel.Text = $"Running {stepName}...";

            // Process pending GTK events before starting to ensure clean state
            while (Application.EventsPending()) Application.RunIteration();

            // Start the progress dialog
            UI.ProgressDialog.Instance.Start($"Running {stepName}...", UI.OperationType.Processing);
            var cancellationToken = UI.ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;

            try
            {
                // Create a single-step pipeline
                var pipeline = new AIModels.WorkflowPipeline
                {
                    Name = stepName,
                    Steps = new List<AIModels.WorkflowStep> { step }
                };

                var manager = AIModels.AIModelManager.Instance;
                var result = await manager.ExecuteWorkflowAsync(
                    pipeline,
                    _imagePaths,
                    _lastSceneResult,
                    (message, progress) => Application.Invoke((s, e) => {
                        _statusLabel.Text = message;
                        if (UI.ProgressDialog.Instance.IsVisible)
                        {
                            UI.ProgressDialog.Instance.Update(progress, message);
                        }
                    }),
                    cancellationToken
                );

                // Process pending GTK events after workflow to ensure queued Application.Invoke calls
                // are processed before continuing - prevents GTK reference tracking issues
                while (Application.EventsPending()) Application.RunIteration();

                if (result != null)
                {
                    bool hasGeometry = result.Meshes.Any(m => m.Vertices.Count > 0);
                    if (!hasGeometry)
                    {
                        Application.Invoke((s, e) =>
                        {
                            UI.ProgressDialog.Instance.Fail(new Exception($"{stepName} completed but no geometry was generated."));
                            _statusLabel.Text = $"{stepName} failed.";
                        });
                        return;
                    }
                    Application.Invoke((s, e) =>
                    {
                        _lastSceneResult = result;
                        if (step == AIModels.WorkflowStep.Dust3rReconstruction ||
                            step == AIModels.WorkflowStep.Mast3rReconstruction ||
                            step == AIModels.WorkflowStep.Must3rReconstruction ||
                            step == AIModels.WorkflowStep.SfMReconstruction)
                        {
                            ApplyPointCloudResultToScene(result);
                        }
                        else
                        {
                            UpdateSceneFromResult(result);
                        }
                        if (cancellationToken.IsCancellationRequested)
                        {
                            _statusLabel.Text = $"{stepName} cancelled.";
                        }
                        else
                        {
                            _statusLabel.Text = $"{stepName} completed successfully.";
                            UI.ProgressDialog.Instance.Complete();
                        }
                    });
                }
                else
                {
                    Application.Invoke((s, e) =>
                    {
                        UI.ProgressDialog.Instance.Fail(new Exception($"{stepName} completed but no result was generated."));
                    });
                }
            }
            catch (OperationCanceledException)
            {
                Application.Invoke((s, e) =>
                {
                    _statusLabel.Text = $"{stepName} cancelled.";
                });
            }
            catch (Exception ex)
            {
                Application.Invoke((s, e) =>
                {
                    UI.ProgressDialog.Instance.Fail(ex);
                    _statusLabel.Text = $"{stepName} failed.";
                });
            }

            // Final event processing to ensure all UI updates are applied
            while (Application.EventsPending()) Application.RunIteration();
        }

        /// <summary>
        /// Get a human-readable name for a workflow step
        /// </summary>
        private string GetStepDisplayName(AIModels.WorkflowStep step)
        {
            return step switch
            {
                AIModels.WorkflowStep.Dust3rReconstruction => "Dust3R Reconstruction",
                AIModels.WorkflowStep.SfMReconstruction => "Feature Matching (SfM)",
                AIModels.WorkflowStep.TripoSRGeneration => "TripoSR Generation",
                AIModels.WorkflowStep.LGMGeneration => "LGM Generation",
                AIModels.WorkflowStep.Wonder3DGeneration => "Wonder3D Generation",
                AIModels.WorkflowStep.NeRFRefinement => "NeRF Refinement",
                AIModels.WorkflowStep.DeepMeshPriorRefinement => "DeepMeshPrior Refinement",
                AIModels.WorkflowStep.TripoSFRefinement => "TripoSF Refinement",
                AIModels.WorkflowStep.GaussianSDFRefinement => "GaussianSDF Refinement",
                AIModels.WorkflowStep.PoissonReconstruction => "Poisson Reconstruction",
                AIModels.WorkflowStep.MarchingCubes => "Marching Cubes",
                AIModels.WorkflowStep.MeshSmoothing => "Mesh Smoothing",
                AIModels.WorkflowStep.MeshDecimation => "Mesh Decimation",
                AIModels.WorkflowStep.UniRigAutoRig => "UniRig Auto-Rig",
                AIModels.WorkflowStep.VoxelizePointCloud => "Voxelize Point Cloud",
                AIModels.WorkflowStep.MergePointClouds => "Merge Point Clouds",
                AIModels.WorkflowStep.AlignPointClouds => "Align Point Clouds",
                AIModels.WorkflowStep.FilterPointCloud => "Filter Point Cloud",
                _ => step.ToString()
            };
        }
    }
}
