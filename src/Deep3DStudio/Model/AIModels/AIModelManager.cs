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
            Description = "Single-image 3D using LGM (Gaussian model)",
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

        private T? CreateInferenceWithProgress<T>(ref T? field, Func<T> factory) where T : class, IInferenceWithProgress
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
            Action<string>? statusCallback = null,
            System.Threading.CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    statusCallback?.Invoke($"Loading image: {Path.GetFileName(imagePath)}...");
                    MeshData? mesh = null;

                    switch (model)
                    {
                        case ImageTo3DModel.TripoSR: mesh = TripoSR?.GenerateFromImage(imagePath, cancellationToken); break;
                        case ImageTo3DModel.LGM: mesh = LGM?.GenerateFromImage(imagePath, cancellationToken); break;
                        case ImageTo3DModel.Wonder3D: mesh = Wonder3D?.GenerateFromImage(imagePath, cancellationToken); break;
                    }

                    if (mesh != null && mesh.Vertices.Count > 0)
                    {
                        var res = new SceneResult();
                        res.Meshes.Add(mesh);
                        return res;
                    }
                    return null;
                }
                catch (OperationCanceledException)
                {
                    statusCallback?.Invoke("Cancelled.");
                    throw;
                }
                catch (Exception ex)
                {
                    statusCallback?.Invoke($"Error: {ex.Message}");
                    return null;
                }
            }, cancellationToken);
        }

        public async Task<SceneResult?> ExecuteWorkflowAsync(
            WorkflowPipeline pipeline,
            List<string>? imagePaths = null,
            SceneResult? existingScene = null,
            Action<string, float>? progressCallback = null,
            System.Threading.CancellationToken cancellationToken = default)
        {
            return await Task.Run(async () =>
            {
                SceneResult? currentResult = existingScene ?? new SceneResult();
                float[,,]? voxelGrid = null;
                Vector3 voxelOrigin = Vector3.Zero;
                float voxelSize = 0.02f;
                int totalSteps = pipeline.Steps.Count;

                for (int i = 0; i < totalSteps; i++)
                {
                    var step = pipeline.Steps[i];
                    float stepBase = totalSteps > 0 ? (float)i / totalSteps : 0f;
                    float stepSpan = totalSteps > 0 ? 1f / totalSteps : 1f;

                    void Report(string message, float stepProgress)
                    {
                        float clamped = Math.Clamp(stepProgress, 0f, 1f);
                        float overall = Math.Clamp(stepBase + clamped * stepSpan, 0f, 1f);
                        ProgressUpdated?.Invoke(message, overall);
                        progressCallback?.Invoke(message, overall);
                    }

                    if (cancellationToken.IsCancellationRequested)
                    {
                        Report("Cancelled by user.", 0f);
                        return currentResult;
                    }

                    Report($"Step {step}", 0f);

                    try
                    {
                        switch (step)
                        {
                        case WorkflowStep.LoadImages:
                            break;

                        case WorkflowStep.Dust3rReconstruction:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                Report("Attempting Dust3r reconstruction...", 0.05f);

                                // Try Dust3r first - it will attempt to initialize internally
                                SceneResult? dust3rResult = null;
                                Action<string, float, string>? dust3rProgress = null;
                                try
                                {
                                    if (Dust3r != null)
                                    {
                                        dust3rProgress = (stage, p, message) => Report($"Dust3r: {message}", p);
                                        Dust3r.OnProgress += dust3rProgress;
                                        Dust3r.LogCallback = msg => Report(msg, 0.1f);
                                    }
                                    dust3rResult = Dust3r?.ReconstructScene(imagePaths, cancellationToken);
                                }
                                catch (OperationCanceledException)
                                {
                                    Report("Dust3r cancelled.", 0.6f);
                                    return currentResult;
                                }
                                catch (Exception ex)
                                {
                                    Report($"Dust3r failed: {ex.Message}", 0.6f);
                                }
                                finally
                                {
                                    if (Dust3r != null && dust3rProgress != null)
                                    {
                                        Dust3r.OnProgress -= dust3rProgress;
                                    }
                                }

                                // Check if Dust3r succeeded (has actual mesh data)
                                if (dust3rResult != null && dust3rResult.Meshes.Count > 0 &&
                                    dust3rResult.Meshes.Any(m => m.Vertices.Count > 0))
                                {
                                    Report("Dust3r reconstruction complete.", 1.0f);
                                    currentResult = dust3rResult;
                                }
                                else
                                {
                                    // Fall back to SfM (Feature Matching)
                                    Report("Dust3r not available or failed, trying Feature Matching SfM...", 0.2f);

                                    // Clean up any corrupted state from Dust3r before running SfM
                                    // This is critical to prevent crashes when falling back from failed Python/native operations
                                    try
                                    {
                                        Report("Cleaning up resources before SfM fallback...", 0.25f);

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
                                                Report("Releasing Python/GPU resources...", 0.3f);
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
                                                Report($"Warning: Python cleanup had issues: {pyEx.Message}", 0.3f);
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
                                            Report("Resetting OpenCV state...", 0.4f);
                                            // Reset OpenCV optimization settings to default
                                            Cv2.SetUseOptimized(true);
                                            // Note: OpenCvSharp doesn't expose direct memory pool reset,
                                            // but creating fresh Mat objects in SfM will use clean allocations
                                        }
                                        catch (Exception cvEx)
                                        {
                                            Report($"Warning: OpenCV reset had issues: {cvEx.Message}", 0.4f);
                                        }

                                        // Small delay to ensure native resources are fully released
                                        System.Threading.Thread.Sleep(200);
                                    }
                                    catch (Exception cleanupEx)
                                    {
                                        Report($"Warning: cleanup before SfM fallback had issues: {cleanupEx.Message}", 0.45f);
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
                                                Report($"[SfM] Image verified: {Path.GetFileName(path)}", 0.5f);
                                            }
                                            else
                                            {
                                                Report($"[SfM] Warning: Image not found: {path}", 0.5f);
                                            }
                                        }

                                        if (validImagePaths.Count < 2)
                                        {
                                            Report($"SfM requires at least 2 valid images. Found: {validImagePaths.Count}", 0.6f);
                                        }
                                        else
                                        {
                                            // Create SfM in a completely fresh state
                                            using (var sfm = new SfMInference())
                                            {
                                                sfm.LogCallback = msg => Report(msg, 0.6f);
                                                currentResult = sfm.ReconstructScene(validImagePaths);
                                            }
                                        }

                                        if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                        {
                                            Report($"SfM reconstruction complete. {currentResult.Meshes[0].Vertices.Count} points.", 1.0f);
                                        }
                                        else
                                        {
                                            Report("SfM reconstruction failed - no points generated.", 0.9f);
                                        }
                                    }
                                    catch (TypeInitializationException ex)
                                    {
                                        Report($"SfM unavailable (OpenCV native libraries not found): {ex.InnerException?.Message ?? ex.Message}", 0.9f);
                                    }
                                    catch (Exception ex)
                                    {
                                        Report($"SfM failed: {ex.Message}", 0.9f);
                                    }
                                }
                            }
                            break;

                        case WorkflowStep.Mast3rReconstruction:
                            if (imagePaths != null && imagePaths.Count >= 2)
                            {
                                Report("Running MASt3R reconstruction...", 0.05f);
                                try
                                {
                                    using (var mast3r = new Mast3rInference())
                                    {
                                        Action<string, float, string> mast3rProgress = (stage, p, message) => Report($"MASt3R: {message}", p);
                                        mast3r.OnProgress += mast3rProgress;
                                        mast3r.LogCallback = msg => Report(msg, 0.1f);
                                        try
                                        {
                                            // Use retrieval for optimal pairing of unordered images
                                            currentResult = mast3r.ReconstructScene(imagePaths, useRetrieval: true, cancellationToken: cancellationToken);
                                        }
                                        finally
                                        {
                                            mast3r.OnProgress -= mast3rProgress;
                                        }
                                    }

                                    if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    {
                                        Report($"MASt3R reconstruction complete. {currentResult.Meshes[0].Vertices.Count} points.", 1.0f);
                                    }
                                    else
                                    {
                                        Report("MASt3R reconstruction failed - no points generated.", 0.9f);
                                    }
                                }
                                catch (OperationCanceledException)
                                {
                                    Report("MASt3R cancelled.", 0.6f);
                                    return currentResult;
                                }
                                catch (Exception ex)
                                {
                                    Report($"MASt3R failed: {ex.Message}", 0.9f);
                                }
                            }
                            else
                            {
                                Report("MASt3R requires at least 2 images.", 0.1f);
                            }
                            break;

                        case WorkflowStep.Must3rReconstruction:
                            if (imagePaths != null && imagePaths.Count >= 2)
                            {
                                Report("Running MUSt3R reconstruction...", 0.05f);
                                try
                                {
                                    using (var must3r = new Must3rInference())
                                    {
                                        Action<string, float, string> must3rProgress = (stage, p, message) => Report($"MUSt3R: {message}", p);
                                        must3r.OnProgress += must3rProgress;
                                        must3r.LogCallback = msg => Report(msg, 0.1f);
                                        try
                                        {
                                            // Use retrieval for optimal pairing of unordered images
                                            currentResult = must3r.ReconstructScene(imagePaths, useRetrieval: true, cancellationToken: cancellationToken);
                                        }
                                        finally
                                        {
                                            must3r.OnProgress -= must3rProgress;
                                        }
                                    }

                                    if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    {
                                        Report($"MUSt3R reconstruction complete. {currentResult.Meshes[0].Vertices.Count} points.", 1.0f);
                                    }
                                    else
                                    {
                                        Report("MUSt3R reconstruction failed - no points generated.", 0.9f);
                                    }
                                }
                                catch (OperationCanceledException)
                                {
                                    Report("MUSt3R cancelled.", 0.6f);
                                    return currentResult;
                                }
                                catch (Exception ex)
                                {
                                    Report($"MUSt3R failed: {ex.Message}", 0.9f);
                                }
                            }
                            else
                            {
                                Report("MUSt3R requires at least 2 images.", 0.1f);
                            }
                            break;

                        case WorkflowStep.SfMReconstruction:
                            if (imagePaths != null && imagePaths.Count >= 2)
                            {
                                Report($"Running Feature Matching SfM with {imagePaths.Count} images...", 0.05f);
                                try
                                {
                                    using (var sfm = new SfMInference())
                                    {
                                        sfm.LogCallback = msg => Report(msg, 0.6f);
                                        currentResult = sfm.ReconstructScene(imagePaths);
                                    }

                                    if (currentResult.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    {
                                        Report($"SfM complete. Generated {currentResult.Meshes[0].Vertices.Count} points from {currentResult.Poses.Count} cameras.", 1.0f);
                                    }
                                    else
                                    {
                                        Report("SfM reconstruction failed - insufficient feature matches or images.", 0.9f);
                                    }
                                }
                                catch (TypeInitializationException ex)
                                {
                                    Report($"SfM unavailable - OpenCV native libraries not found. Please install OpenCV or use Dust3r instead. Error: {ex.InnerException?.Message ?? ex.Message}", 0.9f);
                                }
                                catch (Exception ex)
                                {
                                    Report($"SfM failed: {ex.Message}", 0.9f);
                                }
                            }
                            else if (imagePaths != null && imagePaths.Count < 2)
                            {
                                Report("SfM requires at least 2 images.", 0.1f);
                            }
                            break;

                        case WorkflowStep.TripoSRGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                Report("Running TripoSR generation...", 0.05f);
                                var tripo = TripoSR;
                                Action<string, float, string>? tripoProgress = null;
                                if (tripo != null)
                                {
                                    tripoProgress = (stage, p, message) => Report($"TripoSR: {message}", p);
                                    tripo.OnProgress += tripoProgress;
                                }

                                try
                                {
                                    currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.TripoSR,
                                        msg => Report(msg, 0.1f), cancellationToken);
                                }
                                finally
                                {
                                    if (tripo != null && tripoProgress != null)
                                    {
                                        tripo.OnProgress -= tripoProgress;
                                    }
                                }

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    Report($"TripoSR complete. {currentResult.Meshes[0].Vertices.Count} vertices.", 1.0f);
                                else
                                    Report("TripoSR failed - model not loaded or generation failed.", 0.9f);
                            }
                            break;

                        case WorkflowStep.LGMGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                Report("Running LGM generation...", 0.05f);
                                var lgm = LGM;
                                Action<string, float, string>? lgmProgress = null;
                                if (lgm != null)
                                {
                                    lgmProgress = (stage, p, message) => Report($"LGM: {message}", p);
                                    lgm.OnProgress += lgmProgress;
                                }

                                try
                                {
                                    currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.LGM,
                                        msg => Report(msg, 0.1f), cancellationToken);
                                }
                                finally
                                {
                                    if (lgm != null && lgmProgress != null)
                                    {
                                        lgm.OnProgress -= lgmProgress;
                                    }
                                }

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    Report($"LGM complete. {currentResult.Meshes[0].Vertices.Count} vertices.", 1.0f);
                                else
                                    Report("LGM failed - model not loaded or generation failed.", 0.9f);
                            }
                            break;

                        case WorkflowStep.Wonder3DGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                Report("Running Wonder3D generation...", 0.05f);
                                var wonder3D = Wonder3D;
                                Action<string, float, string>? wonderProgress = null;
                                if (wonder3D != null)
                                {
                                    wonderProgress = (stage, p, message) => Report($"Wonder3D: {message}", p);
                                    wonder3D.OnProgress += wonderProgress;
                                }

                                try
                                {
                                    currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.Wonder3D,
                                        msg => Report(msg, 0.1f), cancellationToken);
                                }
                                finally
                                {
                                    if (wonder3D != null && wonderProgress != null)
                                    {
                                        wonder3D.OnProgress -= wonderProgress;
                                    }
                                }

                                if (currentResult?.Meshes.Count > 0 && currentResult.Meshes.Any(m => m.Vertices.Count > 0))
                                    Report($"Wonder3D complete. {currentResult.Meshes[0].Vertices.Count} vertices.", 1.0f);
                                else
                                    Report("Wonder3D failed - model not loaded or generation failed.", 0.9f);
                            }
                            break;

                        case WorkflowStep.TripoSFRefinement:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                Report("Running TripoSF refinement...", 0.05f);
                                var tripoSF = TripoSF;
                                Action<string, float, string>? tripoSfProgress = null;
                                if (tripoSF != null)
                                {
                                    tripoSfProgress = (stage, p, message) => Report($"TripoSF: {message}", p);
                                    tripoSF.OnProgress += tripoSfProgress;
                                }
                                try
                                {
                                    for (int meshIdx = 0; meshIdx < currentResult.Meshes.Count; meshIdx++)
                                    {
                                        var inputMesh = currentResult.Meshes[meshIdx];
                                        if (inputMesh.Vertices.Count == 0) continue;

                                        var refinedMesh = tripoSF?.RefineMesh(inputMesh, cancellationToken);
                                        if (refinedMesh != null && refinedMesh.Vertices.Count > 0)
                                        {
                                            currentResult.Meshes[meshIdx] = refinedMesh;
                                        }
                                    }
                                }
                                finally
                                {
                                    if (tripoSF != null && tripoSfProgress != null)
                                    {
                                        tripoSF.OnProgress -= tripoSfProgress;
                                    }
                                }
                                Report("TripoSF refinement complete.", 1.0f);
                            }
                            else
                            {
                                Report("TripoSF refinement skipped - no mesh available.", 0.1f);
                            }
                            break;

                        case WorkflowStep.VoxelizePointCloud:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                Report("Voxelizing point cloud...", 0.1f);
                                var voxelized = VoxelizePointClouds(currentResult.Meshes, 200);
                                if (voxelized.grid != null)
                                {
                                    voxelGrid = voxelized.grid;
                                    voxelOrigin = voxelized.origin;
                                    voxelSize = voxelized.voxelSize;
                                    Report("Voxelization complete.", 1.0f);
                                }
                                else
                                {
                                    Report("Voxelization skipped - no points found.", 0.1f);
                                }
                            }
                            else
                            {
                                Report("Voxelization skipped - no point cloud available.", 0.1f);
                            }
                            break;

                        case WorkflowStep.MarchingCubes:
                            if (voxelGrid == null)
                            {
                                Report("Marching cubes skipped - no voxel grid.", 0.1f);
                                break;
                            }
                            Report("Running marching cubes...", 0.1f);
                            var mesher = new Meshing.MarchingCubesMesher();
                            var marchingMesh = mesher.GenerateMesh(voxelGrid, voxelOrigin, voxelSize, 0.5f);
                            if (marchingMesh.Vertices.Count > 0 && marchingMesh.Indices.Count > 0)
                            {
                                currentResult.Meshes.Clear();
                                currentResult.Meshes.Add(marchingMesh);
                                Report($"Marching cubes complete. {marchingMesh.Vertices.Count} vertices.", 1.0f);
                            }
                            else
                            {
                                Report("Marching cubes produced no geometry.", 0.9f);
                            }
                            break;

                        case WorkflowStep.DeepMeshPriorRefinement:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                Report("Running DeepMeshPrior refinement...", 0.05f);
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
                                            (msg, p) => Report(msg, p),
                                            cancellationToken);
                                        currentResult.Meshes[meshIdx] = refinedMesh;
                                    }
                                }
                                Report("DeepMeshPrior refinement complete.", 1.0f);
                            }
                            else
                            {
                                Report("DeepMeshPrior skipped - no mesh available.", 0.1f);
                            }
                            break;

                        case WorkflowStep.GaussianSDFRefinement:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                Report("Running GaussianSDF refinement...", 0.05f);
                                var gsdfRefiner = new Meshing.GaussianSDFRefiner();

                                for (int meshIdx = 0; meshIdx < currentResult.Meshes.Count; meshIdx++)
                                {
                                    var inputMesh = currentResult.Meshes[meshIdx];
                                    if (inputMesh.Vertices.Count > 0 && inputMesh.Indices.Count > 0)
                                    {
                                        var refinedMesh = await gsdfRefiner.RefineMeshAsync(
                                            inputMesh,
                                            (msg, p) => Report(msg, p),
                                            cancellationToken);
                                        currentResult.Meshes[meshIdx] = refinedMesh;
                                    }
                                }
                                Report("GaussianSDF refinement complete.", 1.0f);
                            }
                            else
                            {
                                Report("GaussianSDF skipped - no mesh available.", 0.1f);
                            }
                            break;

                        case WorkflowStep.UniRigAutoRig:
                            if (currentResult != null && currentResult.Meshes.Count > 0)
                            {
                                Report("Running UniRig auto-rigging...", 0.05f);
                                var mesh = currentResult.Meshes[0];
                                if (mesh.Vertices.Count > 0)
                                {
                                    var rigResult = await RigMeshAsync(
                                        mesh.Vertices.ToArray(),
                                        mesh.Indices.ToArray(),
                                        msg => Report(msg, 0.2f));

                                    if (rigResult != null && rigResult.Success)
                                    {
                                        currentResult.RigResult = rigResult;
                                        Report($"UniRig complete. {rigResult.JointPositions?.Length ?? 0} joints.", 1.0f);
                                    }
                                    else
                                    {
                                        Report("UniRig failed or model not available.", 0.9f);
                                    }
                                }
                            }
                            else
                            {
                                Report("UniRig skipped - no mesh available.", 0.1f);
                            }
                            break;

                        case WorkflowStep.NeRFRefinement:
                            if (currentResult != null && currentResult.Poses.Count > 0)
                            {
                                Report("Running NeRF refinement...", 0.05f);
                                var nerf = new VoxelGridNeRF();
                                nerf.InitializeFromMesh(currentResult.Meshes);
                                var nerfIterations = Configuration.IniSettings.Instance.NeRFIterations;
                                bool cancelled = nerf.Train(currentResult.Poses, nerfIterations, cancellationToken);
                                if (cancelled)
                                {
                                    Report("NeRF cancelled. Returning partial mesh.", 0.6f);
                                }
                                var mesh = nerf.GetMesh();
                                currentResult.Meshes.Clear();
                                currentResult.Meshes.Add(mesh);
                                Report($"NeRF refinement complete. {mesh.Vertices.Count} vertices.", 1.0f);
                            }
                            else
                            {
                                Report("NeRF refinement skipped - no poses available.", 0.1f);
                            }
                            break;
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        Report("Cancelled by user.", 0f);
                        return currentResult;
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

        private static (float[,,]? grid, Vector3 origin, float voxelSize) VoxelizePointClouds(List<MeshData> meshes, int maxRes)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            bool hasPoint = false;

            foreach (var mesh in meshes)
            {
                foreach (var v in mesh.Vertices)
                {
                    hasPoint = true;
                    min = Vector3.ComponentMin(min, v);
                    max = Vector3.ComponentMax(max, v);
                }
            }

            var settings = IniSettings.Instance;
            float voxelSize = Math.Max(0.001f, settings.MergerVoxelSize);
            if (!hasPoint)
                return (null, Vector3.Zero, voxelSize);

            int w = (int)((max.X - min.X) / voxelSize) + 5;
            int h = (int)((max.Y - min.Y) / voxelSize) + 5;
            int d = (int)((max.Z - min.Z) / voxelSize) + 5;

            if (w > maxRes)
            {
                voxelSize *= (w / (float)maxRes);
                w = maxRes;
                h = (int)((max.Y - min.Y) / voxelSize) + 5;
                d = (int)((max.Z - min.Z) / voxelSize) + 5;
            }

            float[,,] grid = new float[w, h, d];

            foreach (var mesh in meshes)
            {
                foreach (var v in mesh.Vertices)
                {
                    int x = (int)((v.X - min.X) / voxelSize);
                    int y = (int)((v.Y - min.Y) / voxelSize);
                    int z = (int)((v.Z - min.Z) / voxelSize);
                    if (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d)
                    {
                        grid[x, y, z] = 1.0f;
                    }
                }
            }

            float[,,] smooth = new float[w, h, d];
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 1; y++)
                    for (int z = 1; z < d - 1; z++)
                    {
                        if (grid[x, y, z] > 0)
                        {
                            smooth[x, y, z] = 1;
                            smooth[x + 1, y, z] = 1; smooth[x - 1, y, z] = 1;
                            smooth[x, y + 1, z] = 1; smooth[x, y - 1, z] = 1;
                            smooth[x, y, z + 1] = 1; smooth[x, y, z - 1] = 1;
                        }
                    }

            return (smooth, min, voxelSize);
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
