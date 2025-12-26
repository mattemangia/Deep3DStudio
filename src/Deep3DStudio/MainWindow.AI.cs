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

        private void OnDust3rDeepMeshPriorWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", "Please load at least 2 images for Dust3r reconstruction.");
                return;
            }

            _statusLabel.Text = "Running Dust3r → DeepMeshPrior workflow...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.Dust3rToDeepMeshPrior);
        }

        private void OnDust3rNeRFDeepMeshPriorWorkflow(object? sender, EventArgs e)
        {
            if (_imagePaths.Count < 2)
            {
                ShowMessage("Need More Images", "Please load at least 2 images for this workflow.");
                return;
            }

            _statusLabel.Text = "Running Dust3r → NeRF → DeepMeshPrior workflow...";
            RunAIWorkflowAsync(AIModels.WorkflowPipeline.Dust3rToNeRFToMesh);
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
            try
            {
                var manager = AIModels.AIModelManager.Instance;
                var result = await manager.ExecuteWorkflowAsync(
                    pipeline,
                    _imagePaths,
                    _lastSceneResult,
                    (message, _) => Application.Invoke((s, e) => _statusLabel.Text = message)
                );

                if (result != null)
                {
                    Application.Invoke((s, e) =>
                    {
                        _lastSceneResult = result;
                        UpdateSceneFromResult(result);
                        _statusLabel.Text = $"Workflow '{pipeline.Name}' completed successfully.";
                    });
                }
            }
            catch (Exception ex)
            {
                Application.Invoke((s, e) =>
                {
                    ShowMessage("Workflow Error", $"Error running workflow: {ex.Message}");
                    _statusLabel.Text = "Workflow failed.";
                });
            }
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
    }
}
