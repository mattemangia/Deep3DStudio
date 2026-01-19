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
            if (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0)
            {
                ShowMessage("No geometry", "Please generate a point cloud or mesh first.");
                return;
            }

            var refineSetting = IniSettings.Instance.MeshRefinement;
            switch (refineSetting)
            {
                case MeshRefinementMethod.DeepMeshPrior:
                    await RunAIMeshingAsync(MeshingAlgorithm.DeepMeshPrior, "Mesh Optimization (DeepMeshPrior)");
                    break;
                case MeshRefinementMethod.TripoSF:
                    await RunAIMeshingAsync(MeshingAlgorithm.TripoSF, "Mesh Refinement (TripoSF)");
                    break;
                case MeshRefinementMethod.GaussianSDF:
                    await RunAIMeshingAsync(MeshingAlgorithm.GaussianSDF, "Mesh Refinement (GaussianSDF)");
                    break;
                default:
                    ShowMessage("No refinement method selected", "Please choose a mesh refinement model in Settings.");
                    break;
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

            _statusLabel.Text = "Running Wonder3D multi-view 3D generation...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.ImageToWonder3D);
        }

        private void OnDeepMeshPriorRefine(object? sender, EventArgs e)
        {
            if (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0)
            {
                ShowMessage("No Mesh", "Please generate a mesh first before optimizing.");
                return;
            }

            _statusLabel.Text = "Running DeepMeshPrior optimization...";
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = "DeepMeshPrior Optimization",
                Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.DeepMeshPriorRefinement }
            };
            RunAIWorkflowAsync(pipeline);
        }

        private void OnTripoSFRefine(object? sender, EventArgs e)
        {
            var selectedMesh = GetSelectedMesh();
            if (selectedMesh == null && (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0))
            {
                ShowMessage("No Mesh", "Please generate or select a mesh to refine with TripoSF.");
                return;
            }

            _statusLabel.Text = "Running TripoSF mesh refinement...";
            // Create a custom workflow for TripoSF refinement
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = "TripoSF Refinement",
                Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.TripoSFRefinement }
            };
            RunAIWorkflowAsync(pipeline);
        }

        private void OnGaussianSDFRefine(object? sender, EventArgs e)
        {
            var selectedMesh = GetSelectedMesh();
            if (selectedMesh == null && (_lastSceneResult == null || _lastSceneResult.Meshes.Count == 0))
            {
                ShowMessage("No Mesh", "Please generate or select a mesh to refine with GaussianSDF.");
                return;
            }

            _statusLabel.Text = "Running GaussianSDF mesh refinement...";
            var pipeline = new AIModels.WorkflowPipeline
            {
                Name = "GaussianSDF Refinement",
                Steps = new List<AIModels.WorkflowStep> { AIModels.WorkflowStep.GaussianSDFRefinement }
            };
            RunAIWorkflowAsync(pipeline);
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
                    })
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
                        _statusLabel.Text = $"Workflow '{pipeline.Name}' completed successfully.";
                        UI.ProgressDialog.Instance.Complete();
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
            // Validate prerequisites for each step
            switch (step)
            {
                case AIModels.WorkflowStep.Dust3rReconstruction:
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
                case AIModels.WorkflowStep.PoissonReconstruction:
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
                    })
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
                        _statusLabel.Text = $"{stepName} completed successfully.";
                        UI.ProgressDialog.Instance.Complete();
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
