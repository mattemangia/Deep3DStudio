using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Deep3DStudio.Configuration;
using Deep3DStudio.Model.SfM;
using Deep3DStudio.Python;
using OpenCvSharp;
using OpenTK.Mathematics;
using Python.Runtime;

namespace Deep3DStudio.Model.AIModels
{
    // Removed duplicate ImageTo3DModel enum as it exists in Configuration

    public enum WorkflowStep
    {
        LoadImages,
        LoadPointCloud,
        LoadMesh,
        Dust3rReconstruction,
        Mast3rReconstruction,   // MASt3R - Metric reconstruction with dense feature matching
        Must3rReconstruction,   // MUSt3R - Multi-view network for many images/video
        SfMReconstruction,
        TripoSRGeneration,
        LGMGeneration,
        Wonder3DGeneration,
        TripoSFRefinement,
        DeepMeshPriorRefinement,
        GaussianSDFRefinement,
        NeRFRefinement,
        MergePointClouds,
        AlignPointClouds,
        FilterPointCloud,
        VoxelizePointCloud,
        MarchingCubes,
        PoissonReconstruction,
        MeshSmoothing,
        MeshDecimation,
        UniRigAutoRig,
        ExportMesh,
        ExportPointCloud,
        ExportRiggedMesh
    }

    public class WorkflowPipeline
    {
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public List<WorkflowStep> Steps { get; set; } = new();

        public static WorkflowPipeline ImageToDust3rToMesh => new()
        {
            Name = "Images -> Dust3r -> Mesh",
            Description = "Multi-view reconstruction using Dust3r with mesh generation",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Dust3rReconstruction,
                WorkflowStep.VoxelizePointCloud,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline ImageToMast3rToMesh => new()
        {
            Name = "Images -> MASt3R -> Mesh",
            Description = "Multi-view metric reconstruction using MASt3R (2+ images, metric pointmaps)",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Mast3rReconstruction,
                WorkflowStep.VoxelizePointCloud,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline ImageToMust3rToMesh => new()
        {
            Name = "Images/Video -> MUSt3R -> Mesh",
            Description = "Multi-view reconstruction using MUSt3R (many images, video support, 8-11 FPS)",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Must3rReconstruction,
                WorkflowStep.VoxelizePointCloud,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline ImageToTripoSR => new()
        {
            Name = "Image -> TripoSR -> Mesh",
            Description = "Single image to 3D using TripoSR",
            Steps = new List<WorkflowStep> { WorkflowStep.LoadImages, WorkflowStep.TripoSRGeneration }
        };

         public static WorkflowPipeline ImageToLGM => new()
        {
            Name = "Image -> LGM -> Mesh",
            Description = "Single image to 3D using LGM (Large Multi-View Gaussian Model)",
            Steps = new List<WorkflowStep> { WorkflowStep.LoadImages, WorkflowStep.LGMGeneration }
        };

        public static WorkflowPipeline ImageToWonder3D => new()
        {
            Name = "Image -> Wonder3D -> Mesh",
            Description = "Single image to 3D using Wonder3D",
            Steps = new List<WorkflowStep> { WorkflowStep.LoadImages, WorkflowStep.Wonder3DGeneration }
        };

        public static WorkflowPipeline ImageToSfM => new()
        {
            Name = "Images -> Feature Matching SfM",
            Description = "Multi-view reconstruction using classical feature matching (no AI required)",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.SfMReconstruction
            }
        };

        public static WorkflowPipeline Dust3rToDeepMeshPrior => new()
        {
            Name = "Images -> Dust3r -> DeepMeshPrior",
            Description = "Multi-view with DeepMeshPrior refinement",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Dust3rReconstruction,
                WorkflowStep.VoxelizePointCloud,
                WorkflowStep.MarchingCubes,
                WorkflowStep.DeepMeshPriorRefinement
            }
        };

        public static WorkflowPipeline Dust3rToNeRFToMesh => new()
        {
            Name = "Images -> Dust3r -> NeRF -> Mesh",
            Description = "Multi-view with NeRF refinement",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Dust3rReconstruction,
                WorkflowStep.NeRFRefinement,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline PointCloudMergeRefine => new()
        {
            Name = "Merge Point Clouds -> Refine",
            Description = "Merge multiple point clouds with AI refinement",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadPointCloud,
                WorkflowStep.MergePointClouds,
                WorkflowStep.AlignPointClouds,
                WorkflowStep.TripoSFRefinement,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline FullPipeline => new()
        {
            Name = "Full Pipeline (Images -> Mesh)",
            Description = "Complete workflow from images to a meshed 3D model",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Dust3rReconstruction,
                WorkflowStep.NeRFRefinement,
                WorkflowStep.VoxelizePointCloud,
                WorkflowStep.MarchingCubes,
                WorkflowStep.DeepMeshPriorRefinement,
                WorkflowStep.MeshDecimation
            }
        };

        public static List<WorkflowPipeline> GetAllPipelines() => new()
        {
            ImageToDust3rToMesh,
            ImageToMast3rToMesh,
            ImageToMust3rToMesh,
            ImageToSfM,
            ImageToTripoSR,
            ImageToLGM,
            ImageToWonder3D,
            Dust3rToDeepMeshPrior,
            Dust3rToNeRFToMesh,
            PointCloudMergeRefine,
            FullPipeline
        };
    }

    public class AIModelManager : IDisposable
    {
        private static AIModelManager? _instance;
        public static AIModelManager Instance => _instance ??= new AIModelManager();

        private TripoSRInference? _tripoSR;
        private TripoSFInference? _tripoSF;
        private LGMInference? _lgm;
        private Wonder3DInference? _wonder3D;
        private UniRigInference? _uniRig;
        private Dust3rInference? _dust3r;

        private bool _disposed;

        /// <summary>
        /// Event fired when a model's status changes (loaded/unloaded)
        /// </summary>
        public event Action<string, bool>? ModelStatusChanged;

        /// <summary>
        /// Event fired during workflow execution with (stepName, progress 0-1)
        /// </summary>
        public event Action<string, float>? ProgressUpdated;

        /// <summary>
        /// Event fired during model loading with (stage, progress 0-1, message)
        /// </summary>
        public event Action<string, float, string>? ModelLoadProgress;

        private AIModelManager() { }

        private T? CreateInferenceWithProgress<T>(ref T? field, Func<T> factory) where T : BasePythonInference
        {
            if (field == null)
            {
                field = factory();
                // Wire up progress callback
                field.OnLoadProgress += (stage, progress, message) =>
                {
                    ModelLoadProgress?.Invoke(stage, progress, message);
                };
            }
            return field;
        }

        public TripoSRInference? TripoSR => CreateInferenceWithProgress(ref _tripoSR, () => new TripoSRInference());
        public TripoSFInference? TripoSF => CreateInferenceWithProgress(ref _tripoSF, () => new TripoSFInference());
        public LGMInference? LGM => CreateInferenceWithProgress(ref _lgm, () => new LGMInference());
        public Wonder3DInference? Wonder3D => CreateInferenceWithProgress(ref _wonder3D, () => new Wonder3DInference());
        public UniRigInference? UniRig => CreateInferenceWithProgress(ref _uniRig, () => new UniRigInference());
        public Dust3rInference? Dust3r => _dust3r ??= new Dust3rInference();  // Dust3r doesn't inherit from BasePythonInference

        public async Task<SceneResult?> GenerateFromSingleImageAsync(
            string imagePath,
            ImageTo3DModel model,
            Action<string>? statusCallback = null)
        {
            return await Task.Run(() =>
            {
                try
                {
                    statusCallback?.Invoke($"Loading image: {Path.GetFileName(imagePath)}...");
                    MeshData? mesh = null;

                    switch (model)
                    {
                        case ImageTo3DModel.TripoSR: mesh = TripoSR?.GenerateFromImage(imagePath); break;
                        case ImageTo3DModel.LGM: mesh = LGM?.GenerateFromImage(imagePath); break;
                        case ImageTo3DModel.Wonder3D: mesh = Wonder3D?.GenerateFromImage(imagePath); break;
                    }

                    if (mesh != null && mesh.Vertices.Count > 0)
                    {
                        var res = new SceneResult();
                        res.Meshes.Add(mesh);
                        return res;
                    }
                    return null;
                }
                catch (Exception ex)
                {
                    statusCallback?.Invoke($"Error: {ex.Message}");
                    return null;
                }
            });
        }

        public async Task<SceneResult?> ExecuteWorkflowAsync(
            WorkflowPipeline pipeline,
            List<string>? imagePaths = null,
            SceneResult? existingScene = null,
            Action<string, float>? progressCallback = null)
        {
            return await Task.Run(async () =>
            {
                SceneResult? currentResult = existingScene ?? new SceneResult();
                int totalSteps = pipeline.Steps.Count;

                for (int i = 0; i < totalSteps; i++)
                {
                    var step = pipeline.Steps[i];
                    float progress = (float)i / totalSteps;
                    progressCallback?.Invoke($"Step {step}", progress);

                    switch (step)
                    {
                        case WorkflowStep.LoadImages:
                            break;

                        case WorkflowStep.Dust3rReconstruction:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                progressCallback?.Invoke("Attempting Dust3r reconstruction...", progress);

                                // Try Dust3r first - it will attempt to initialize internally
                                SceneResult? dust3rResult = null;
                                try
                                {
                                    if (Dust3r != null)
                                    {
                                        Dust3r.LogCallback = (msg) => progressCallback?.Invoke(msg, progress);
                                    }
                                    dust3rResult = Dust3r?.ReconstructScene(imagePaths);
                                }
                                catch (Exception ex)
                                {
                                    progressCallback?.Invoke($"Dust3r failed: {ex.Message}", progress);
                                }

                                // Check if Dust3r succeeded (has actual mesh data)
                                if (dust3rResult != null && dust3rResult.Meshes.Count > 0 &&
                                    dust3rResult.Meshes.Any(m => m.Vertices.Count > 0))
                                {
                                    progressCallback?.Invoke("Dust3r reconstruction complete.", progress + 0.1f);
                                    currentResult = dust3rResult;
                                }
                                else
                                {
                                    // Fall back to SfM (Feature Matching)
                                    progressCallback?.Invoke("Dust3r not available or failed, trying Feature Matching SfM...", progress);

                                    // Clean up any corrupted state from Dust3r before running SfM
                                    // This is critical to prevent crashes when falling back from failed Python/native operations
                                    try
                                    {
                                        progressCallback?.Invoke("Cleaning up resources before SfM fallback...", progress);

                                        // Dispose and reset Dust3r instance if it failed
                                        if (_dust3r != null)
                                        {
                                            _dust3r.Dispose();
                                            _dust3r = null;
                                        }

                                        // Force Python cleanup if Python is initialized
                                        // This releases PyTorch tensors and CUDA memory that could corrupt other native libraries
                                        if (PythonService.Instance.IsInitialized)
                                        {
                                            try
                                            {
                                                progressCallback?.Invoke("Releasing Python/GPU resources...", progress);
                                                using (Py.GIL())
                                                {
                                                    // Run aggressive Python cleanup
                                                    string cleanupScript = @"
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except ImportError:
    pass
except Exception:
    pass
gc.collect()
";
                                                    PythonEngine.Exec(cleanupScript);
                                                }
                                            }
                                            catch (Exception pyEx)
                                            {
                                                progressCallback?.Invoke($"Warning: Python cleanup had issues: {pyEx.Message}", progress);
                                            }
                                        }

                                        // Force garbage collection to clean up any lingering native objects
                                        GC.Collect();
                                        GC.WaitForPendingFinalizers();
                                        GC.Collect();

                                        // Reset OpenCV state to ensure clean initialization for SfM
                                        // This prevents any corrupted state from affecting OpenCV operations
                                        try
                                        {
                                            progressCallback?.Invoke("Resetting OpenCV state...", progress);
                                            // Reset OpenCV optimization settings to default
                                            Cv2.SetUseOptimized(true);
                                            // Note: OpenCvSharp doesn't expose direct memory pool reset,
                                            // but creating fresh Mat objects in SfM will use clean allocations
                                        }
                                        catch (Exception cvEx)
                                        {
                                            progressCallback?.Invoke($"Warning: OpenCV reset had issues: {cvEx.Message}", progress);
                                        }

                                        // Small delay to ensure native resources are fully released
                                        System.Threading.Thread.Sleep(200);
                                    }
                                    catch (Exception cleanupEx)
                                    {
                                        progressCallback?.Invoke($"Warning: cleanup before SfM fallback had issues: {cleanupEx.Message}", progress);
                                    }

                                    try
                                    {
                                        // Verify image paths are still valid before passing to SfM
                                        // Dust3r only reads images (doesn't modify them), but verify to be safe
                                        var validImagePaths = new List<string>();
                                        foreach (var path in imagePaths)
                                        {
                                            if (File.Exists(path))
                                            {
                                                validImagePaths.Add(path);
                                                progressCallback?.Invoke($"[SfM] Image verified: {Path.GetFileName(path)}", progress);
                                            }
                                            else
                                            {
                                                progressCallback?.Invoke($"[SfM] Warning: Image not found: {path}", progress);
                                            }
                                        }

                                        if (validImagePaths.Count < 2)
                                        {
                                            progressCallback?.Invoke($"SfM requires at least 2 valid images. Found: {validImagePaths.Count}", progress);
                                        }
                                        else
                                        {
                                            // Create SfM in a completely fresh state
                                            using (var sfm = new SfMInference())
                                            {
                                                sfm.LogCallback = (msg) => progressCallback?.Invoke(msg, progress);
                                                currentResult = sfm.ReconstructScene(validImagePaths);
                                            }
                                        }

                                        if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                        {
                                            progressCallback?.Invoke($"SfM reconstruction complete. {currentResult.Meshes[0].Vertices.Count} points.", progress + 0.1f);
                                        }
                                        else
                                        {
                                            progressCallback?.Invoke("SfM reconstruction failed - no points generated.", progress);
                                        }
                                    }
                                    catch (TypeInitializationException ex)
                                    {
                                        progressCallback?.Invoke($"SfM unavailable (OpenCV native libraries not found): {ex.InnerException?.Message ?? ex.Message}", progress);
                                    }
                                    catch (Exception ex)
                                    {
                                        progressCallback?.Invoke($"SfM failed: {ex.Message}", progress);
                                    }
                                }
                            }
                            break;

                        case WorkflowStep.SfMReconstruction:
                            if (imagePaths != null && imagePaths.Count >= 2)
                            {
                                progressCallback?.Invoke($"Running Feature Matching SfM with {imagePaths.Count} images...", progress);
                                try
                                {
                                    using (var sfm = new SfMInference())
                                    {
                                        sfm.LogCallback = (msg) => progressCallback?.Invoke(msg, progress);
                                        currentResult = sfm.ReconstructScene(imagePaths);
                                    }

                                    if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    {
                                        progressCallback?.Invoke($"SfM complete. Generated {currentResult.Meshes[0].Vertices.Count} points from {currentResult.Poses.Count} cameras.", progress + 0.1f);
                                    }
                                    else
                                    {
                                        progressCallback?.Invoke("SfM reconstruction failed - insufficient feature matches or images.", progress);
                                    }
                                }
                                catch (TypeInitializationException ex)
                                {
                                    progressCallback?.Invoke($"SfM unavailable - OpenCV native libraries not found. Please install OpenCV or use Dust3r instead. Error: {ex.InnerException?.Message ?? ex.Message}", progress);
                                }
                                catch (Exception ex)
                                {
                                    progressCallback?.Invoke($"SfM failed: {ex.Message}", progress);
                                }
                            }
                            else if (imagePaths != null && imagePaths.Count < 2)
                            {
                                progressCallback?.Invoke("SfM requires at least 2 images.", progress);
                            }
                            break;

                        case WorkflowStep.TripoSRGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                progressCallback?.Invoke("Running TripoSR generation...", progress);
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.TripoSR,
                                    msg => progressCallback?.Invoke(msg, progress));

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    progressCallback?.Invoke($"TripoSR complete. {currentResult.Meshes[0].Vertices.Count} vertices.", progress + 0.1f);
                                else
                                    progressCallback?.Invoke("TripoSR failed - model not loaded or generation failed.", progress);
                            }
                            break;

                        case WorkflowStep.LGMGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                progressCallback?.Invoke("Running LGM generation...", progress);
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.LGM,
                                    msg => progressCallback?.Invoke(msg, progress));

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    progressCallback?.Invoke($"LGM complete. {currentResult.Meshes[0].Vertices.Count} vertices.", progress + 0.1f);
                                else
                                    progressCallback?.Invoke("LGM failed - model not loaded or generation failed.", progress);
                            }
                            break;

                        case WorkflowStep.Wonder3DGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                progressCallback?.Invoke("Running Wonder3D generation...", progress);
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.Wonder3D,
                                    msg => progressCallback?.Invoke(msg, progress));

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    progressCallback?.Invoke($"Wonder3D complete. {currentResult.Meshes[0].Vertices.Count} vertices.", progress + 0.1f);
                                else
                                    progressCallback?.Invoke("Wonder3D failed - model not loaded or generation failed.", progress);
                            }
                            break;

                        case WorkflowStep.TripoSFRefinement:
                            // TripoSF logic: If we have an original image, re-generate high quality
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                progressCallback?.Invoke("Running TripoSF refinement...", progress);
                                var refinedMesh = TripoSF?.GenerateFromImage(imagePaths[0]);
                                if (refinedMesh != null && refinedMesh.Vertices.Count > 0)
                                {
                                    if(currentResult.Meshes.Count == 0) currentResult.Meshes.Add(refinedMesh);
                                    else currentResult.Meshes[0] = refinedMesh;
                                    progressCallback?.Invoke($"TripoSF refinement complete. {refinedMesh.Vertices.Count} vertices.", progress + 0.1f);
                                }
                                else
                                {
                                    progressCallback?.Invoke("TripoSF refinement failed.", progress);
                                }
                            }
                            break;

                        case WorkflowStep.MarchingCubes:
                            progressCallback?.Invoke("Marching cubes step (placeholder).", progress);
                            break;

                        case WorkflowStep.DeepMeshPriorRefinement:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                progressCallback?.Invoke("Running DeepMeshPrior refinement...", progress);
                                var dmpOptimizer = new DeepMeshPrior.DeepMeshPriorOptimizer();
                                var settings = Configuration.IniSettings.Instance;

                                for (int meshIdx = 0; meshIdx < currentResult.Meshes.Count; meshIdx++)
                                {
                                    var inputMesh = currentResult.Meshes[meshIdx];
                                    if (inputMesh.Vertices.Count > 0)
                                    {
                                        var refinedMesh = await dmpOptimizer.OptimizeAsync(
                                            inputMesh,
                                            settings.DeepMeshPriorIterations,
                                            settings.DeepMeshPriorLearningRate,
                                            settings.DeepMeshPriorLaplacianWeight,
                                            (msg, p) => progressCallback?.Invoke(msg, progress + p * 0.1f));
                                        currentResult.Meshes[meshIdx] = refinedMesh;
                                    }
                                }
                                progressCallback?.Invoke("DeepMeshPrior refinement complete.", progress + 0.1f);
                            }
                            else
                            {
                                progressCallback?.Invoke("DeepMeshPrior skipped - no mesh available.", progress);
                            }
                            break;

                        case WorkflowStep.GaussianSDFRefinement:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                progressCallback?.Invoke("Running GaussianSDF refinement...", progress);
                                var gsdfRefiner = new Meshing.GaussianSDFRefiner();

                                for (int meshIdx = 0; meshIdx < currentResult.Meshes.Count; meshIdx++)
                                {
                                    var inputMesh = currentResult.Meshes[meshIdx];
                                    if (inputMesh.Vertices.Count > 0 && inputMesh.Indices.Count > 0)
                                    {
                                        var refinedMesh = await gsdfRefiner.RefineMeshAsync(
                                            inputMesh,
                                            (msg, p) => progressCallback?.Invoke(msg, progress + p * 0.1f));
                                        currentResult.Meshes[meshIdx] = refinedMesh;
                                    }
                                }
                                progressCallback?.Invoke("GaussianSDF refinement complete.", progress + 0.1f);
                            }
                            else
                            {
                                progressCallback?.Invoke("GaussianSDF skipped - no mesh available.", progress);
                            }
                            break;

                        case WorkflowStep.UniRigAutoRig:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                progressCallback?.Invoke("Running UniRig auto-rigging...", progress);
                                var mesh = currentResult.Meshes[0];
                                if (mesh.Vertices.Count > 0)
                                {
                                    var rigResult = await RigMeshAsync(
                                        mesh.Vertices.ToArray(),
                                        mesh.Indices.ToArray(),
                                        msg => progressCallback?.Invoke(msg, progress));

                                    if (rigResult != null && rigResult.Success)
                                    {
                                        currentResult.RigResult = rigResult;
                                        progressCallback?.Invoke($"UniRig complete. {rigResult.JointPositions?.Length ?? 0} joints.", progress + 0.1f);
                                    }
                                    else
                                    {
                                        progressCallback?.Invoke("UniRig failed or model not available.", progress);
                                    }
                                }
                            }
                            else
                            {
                                progressCallback?.Invoke("UniRig skipped - no mesh available.", progress);
                            }
                            break;

                        case WorkflowStep.NeRFRefinement:
                            if (currentResult != null && currentResult.Poses.Count > 0)
                            {
                                progressCallback?.Invoke("Running NeRF refinement...", progress);
                                var nerf = new VoxelGridNeRF();
                                nerf.InitializeFromMesh(currentResult.Meshes);
                                nerf.Train(currentResult.Poses, 5);
                                var mesh = nerf.GetMesh();
                                currentResult.Meshes.Clear();
                                currentResult.Meshes.Add(mesh);
                                progressCallback?.Invoke($"NeRF refinement complete. {mesh.Vertices.Count} vertices.", progress + 0.1f);
                            }
                            else
                            {
                                progressCallback?.Invoke("NeRF refinement skipped - no poses available.", progress);
                            }
                            break;
                    }
                }

                // Final status report
                if (currentResult != null && currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                {
                    int totalVerts = currentResult.Meshes.Sum(m => m.Vertices.Count);
                    progressCallback?.Invoke($"Workflow complete. Total: {totalVerts} vertices, {currentResult.Meshes.Count} mesh(es), {currentResult.Poses.Count} pose(s).", 1.0f);
                }
                else
                {
                    progressCallback?.Invoke("Workflow completed but no geometry was generated.", 1.0f);
                }

                return currentResult;
            });
        }

        public async Task<SceneResult?> RefineSfMResultAsync(SceneResult sfmResult, Action<string>? statusCallback = null)
        {
             return await Task.FromResult(sfmResult);
        }

        public async Task<RigResult?> RigMeshAsync(Vector3[] vertices, int[] triangles, Action<string>? statusCallback = null)
        {
             return await Task.Run(() =>
             {
                 var mesh = new MeshData();
                 // Reconstruct mesh from inputs
                 foreach(var v in vertices) mesh.Vertices.Add(v);
                 foreach(var t in triangles) mesh.Indices.Add(t);

                 return UniRig?.RigMesh(mesh);
             });
        }

        public void LoadAllModels() { }

        public void UnloadAllModels()
        {
            _tripoSR?.Dispose(); _tripoSR = null;
            _tripoSF?.Dispose(); _tripoSF = null;
            _lgm?.Dispose(); _lgm = null;
            _wonder3D?.Dispose(); _wonder3D = null;
            _uniRig?.Dispose(); _uniRig = null;
            _dust3r?.Dispose(); _dust3r = null;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                UnloadAllModels();
                _disposed = true;
            }
        }
    }
}
