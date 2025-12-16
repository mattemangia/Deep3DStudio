using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Deep3DStudio.Configuration;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// Workflow pipeline step types.
    /// </summary>
    public enum WorkflowStep
    {
        // Input steps
        LoadImages,
        LoadPointCloud,
        LoadMesh,

        // Reconstruction steps
        Dust3rReconstruction,
        SfMReconstruction,
        TripoSRGeneration,
        TripoSGGeneration,
        Wonder3DGeneration,

        // Refinement steps
        TripoSFRefinement,
        FlexiCubesExtraction,
        NeRFRefinement,

        // Point cloud operations
        MergePointClouds,
        AlignPointClouds,
        FilterPointCloud,
        VoxelizePointCloud,

        // Mesh operations
        MarchingCubes,
        PoissonReconstruction,
        MeshSmoothing,
        MeshDecimation,

        // Rigging
        UniRigAutoRig,

        // Export
        ExportMesh,
        ExportPointCloud,
        ExportRiggedMesh
    }

    /// <summary>
    /// A workflow pipeline definition.
    /// </summary>
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

        public static WorkflowPipeline ImageToTripoSR => new()
        {
            Name = "Image -> TripoSR -> Mesh",
            Description = "Single image to 3D using TripoSR (fast)",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.TripoSRGeneration,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline ImageToTripoSG => new()
        {
            Name = "Image -> TripoSG -> Mesh",
            Description = "Single image to 3D using TripoSG (high quality)",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.TripoSGGeneration,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline ImageToWonder3D => new()
        {
            Name = "Image -> Wonder3D -> Mesh",
            Description = "Single image to multi-view then mesh via Wonder3D",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Wonder3DGeneration,
                WorkflowStep.MarchingCubes
            }
        };

        public static WorkflowPipeline Dust3rToFlexiCubes => new()
        {
            Name = "Images -> Dust3r -> FlexiCubes",
            Description = "Multi-view with FlexiCubes mesh extraction",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadImages,
                WorkflowStep.Dust3rReconstruction,
                WorkflowStep.FlexiCubesExtraction
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

        public static WorkflowPipeline MeshToRig => new()
        {
            Name = "Mesh -> Auto Rig (UniRig)",
            Description = "Automatically rig a mesh with UniRig",
            Steps = new List<WorkflowStep>
            {
                WorkflowStep.LoadMesh,
                WorkflowStep.UniRigAutoRig,
                WorkflowStep.ExportRiggedMesh
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
                WorkflowStep.FlexiCubesExtraction,
                WorkflowStep.MeshSmoothing,
                WorkflowStep.MeshDecimation,
                WorkflowStep.MarchingCubes
            }
        };

        public static List<WorkflowPipeline> GetAllPipelines() => new()
        {
            ImageToDust3rToMesh,
            ImageToTripoSR,
            ImageToTripoSG,
            ImageToWonder3D,
            Dust3rToFlexiCubes,
            Dust3rToNeRFToMesh,
            PointCloudMergeRefine,
            MeshToRig,
            FullPipeline
        };
    }

    /// <summary>
    /// Manages AI model instances and provides a unified interface for the application.
    /// Supports complex workflows mixing SfM, AI models, and traditional algorithms.
    /// </summary>
    public class AIModelManager : IDisposable
    {
        private static AIModelManager? _instance;
        public static AIModelManager Instance => _instance ??= new AIModelManager();

        // Model instances (lazy loaded)
        private TripoSRInference? _tripoSR;
        private bool _disposed;

        /// <summary>
        /// Event raised when model loading status changes.
        /// </summary>
        public event Action<string, bool>? ModelStatusChanged;

        /// <summary>
        /// Event raised for progress updates during long operations.
        /// </summary>
        public event Action<string, float>? ProgressUpdated;

        private AIModelManager() { }

        #region Model Access

        public TripoSRInference? TripoSR
        {
            get
            {
                if (_tripoSR == null)
                {
                    _tripoSR = new TripoSRInference();
                    ModelStatusChanged?.Invoke("TripoSR", _tripoSR.IsLoaded);
                }
                return _tripoSR;
            }
        }

        /// <summary>
        /// Get status of all AI models.
        /// </summary>
        public Dictionary<string, bool> GetModelStatus()
        {
            return new Dictionary<string, bool>
            {
                { "Dust3r", File.Exists(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "dust3r.onnx")) },
                { "TripoSR", _tripoSR?.IsLoaded ?? false },
                { "TripoSG", false }, // Not yet implemented
                { "TripoSF", false },
                { "Wonder3D", false },
                { "UniRig", false },
                { "FlexiCubes", false }
            };
        }

        /// <summary>
        /// Check if any AI model is loaded.
        /// </summary>
        public bool HasAnyModelLoaded()
        {
            return (_tripoSR?.IsLoaded ?? false);
            // Add other models as they are implemented
        }

        #endregion

        #region High-Level Workflows

        /// <summary>
        /// Generate 3D from single image using selected model.
        /// </summary>
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

                    switch (model)
                    {
                        case ImageTo3DModel.TripoSR:
                            if (TripoSR?.IsLoaded != true)
                            {
                                statusCallback?.Invoke("TripoSR model not loaded");
                                return null;
                            }
                            statusCallback?.Invoke("Running TripoSR inference...");
                            var result = TripoSR.GenerateFromImage(imagePath);
                            if (result.Success)
                            {
                                statusCallback?.Invoke(result.StatusMessage ?? "Complete");
                                return ConvertToSceneResult(result);
                            }
                            statusCallback?.Invoke(result.StatusMessage ?? "Failed");
                            return null;

                        case ImageTo3DModel.TripoSG:
                        case ImageTo3DModel.Wonder3D:
                            statusCallback?.Invoke($"{model} not yet implemented");
                            return null;

                        default:
                            statusCallback?.Invoke("No model selected");
                            return null;
                    }
                }
                catch (Exception ex)
                {
                    statusCallback?.Invoke($"Error: {ex.Message}");
                    return null;
                }
            });
        }

        /// <summary>
        /// Refine SfM result using AI model.
        /// This is the hybrid workflow: SfM for poses + AI for geometry.
        /// </summary>
        public async Task<SceneResult?> RefineSfMResultAsync(
            SceneResult sfmResult,
            Action<string>? statusCallback = null)
        {
            return await Task.Run(() =>
            {
                try
                {
                    var settings = IniSettings.Instance;

                    // Get point cloud from SfM result
                    if (sfmResult.Meshes == null || sfmResult.Meshes.Count == 0)
                    {
                        statusCallback?.Invoke("No point cloud in SfM result");
                        return sfmResult;
                    }

                    statusCallback?.Invoke("Extracting point cloud from SfM result...");

                    // Get vertices from first mesh (point cloud)
                    var firstMesh = sfmResult.Meshes[0];
                    var points = firstMesh.Vertices;
                    var colors = firstMesh.Colors;

                    if (points == null || points.Count == 0)
                    {
                        statusCallback?.Invoke("No vertices in SfM point cloud");
                        return sfmResult;
                    }

                    statusCallback?.Invoke($"Processing {points.Count} points...");

                    // Apply mesh extraction based on settings
                    switch (settings.MeshExtraction)
                    {
                        case MeshExtractionMethod.FlexiCubes:
                            statusCallback?.Invoke("FlexiCubes refinement not yet implemented");
                            break;

                        case MeshExtractionMethod.TripoSF:
                            statusCallback?.Invoke("TripoSF refinement not yet implemented");
                            break;

                        default:
                            statusCallback?.Invoke("Using standard meshing");
                            break;
                    }

                    // For now, return the original SfM result
                    // In the future, this will apply AI refinement
                    return sfmResult;
                }
                catch (Exception ex)
                {
                    statusCallback?.Invoke($"Error during refinement: {ex.Message}");
                    return sfmResult;
                }
            });
        }

        /// <summary>
        /// Rig a mesh using UniRig.
        /// </summary>
        public async Task<RigResult?> RigMeshAsync(
            Vector3[] vertices,
            int[] triangles,
            Action<string>? statusCallback = null)
        {
            try
            {
                statusCallback?.Invoke("UniRig not yet implemented");
                return await Task.FromResult<RigResult?>(null);
            }
            catch (Exception ex)
            {
                statusCallback?.Invoke($"Error: {ex.Message}");
                return await Task.FromResult<RigResult?>(null);
            }
        }

        #endregion

        #region Workflow Execution

        /// <summary>
        /// Execute a predefined workflow pipeline.
        /// </summary>
        public async Task<SceneResult?> ExecuteWorkflowAsync(
            WorkflowPipeline pipeline,
            List<string>? imagePaths = null,
            SceneResult? existingScene = null,
            Action<string, float>? progressCallback = null)
        {
            return await Task.Run(async () =>
            {
                SceneResult? currentResult = existingScene;
                int totalSteps = pipeline.Steps.Count;

                for (int i = 0; i < totalSteps; i++)
                {
                    var step = pipeline.Steps[i];
                    float progress = (float)i / totalSteps;
                    progressCallback?.Invoke($"Step {i + 1}/{totalSteps}: {GetStepName(step)}", progress);

                    try
                    {
                        currentResult = await ExecuteStepAsync(step, imagePaths, currentResult, progressCallback);

                        if (currentResult == null && RequiresResult(step))
                        {
                            progressCallback?.Invoke($"Step {step} failed - aborting pipeline", progress);
                            return null;
                        }
                    }
                    catch (Exception ex)
                    {
                        progressCallback?.Invoke($"Error in step {step}: {ex.Message}", progress);
                        return null;
                    }
                }

                progressCallback?.Invoke("Pipeline complete", 1.0f);
                return currentResult;
            });
        }

        private async Task<SceneResult?> ExecuteStepAsync(
            WorkflowStep step,
            List<string>? imagePaths,
            SceneResult? currentResult,
            Action<string, float>? progressCallback)
        {
            switch (step)
            {
                case WorkflowStep.LoadImages:
                    // Images are passed in, nothing to do
                    return currentResult ?? new SceneResult();

                case WorkflowStep.LoadPointCloud:
                case WorkflowStep.LoadMesh:
                    // These should be handled by the caller
                    return currentResult;

                case WorkflowStep.Dust3rReconstruction:
                    // Delegate to Dust3r inference (existing code)
                    progressCallback?.Invoke("Running Dust3r reconstruction...", 0);
                    // This would call _dust3rInference.ReconstructScene(imagePaths)
                    return currentResult;

                case WorkflowStep.SfMReconstruction:
                    progressCallback?.Invoke("Running SfM reconstruction...", 0);
                    // This would call SfMInference.ReconstructScene(imagePaths)
                    return currentResult;

                case WorkflowStep.TripoSRGeneration:
                    if (imagePaths == null || imagePaths.Count == 0)
                        return null;
                    return await GenerateFromSingleImageAsync(
                        imagePaths[0],
                        ImageTo3DModel.TripoSR,
                        msg => progressCallback?.Invoke(msg, 0));

                case WorkflowStep.TripoSGGeneration:
                    if (imagePaths == null || imagePaths.Count == 0)
                        return null;
                    return await GenerateFromSingleImageAsync(
                        imagePaths[0],
                        ImageTo3DModel.TripoSG,
                        msg => progressCallback?.Invoke(msg, 0));

                case WorkflowStep.Wonder3DGeneration:
                    if (imagePaths == null || imagePaths.Count == 0)
                        return null;
                    return await GenerateFromSingleImageAsync(
                        imagePaths[0],
                        ImageTo3DModel.Wonder3D,
                        msg => progressCallback?.Invoke(msg, 0));

                case WorkflowStep.TripoSFRefinement:
                    var tripoSfPath = GetAbsoluteModelPath(IniSettings.Instance.TripoSFModelPath);
                    progressCallback?.Invoke($"TripoSF refinement (model: {tripoSfPath}) not yet implemented", 0);
                    return currentResult;

                case WorkflowStep.FlexiCubesExtraction:
                    var flexiPath = GetAbsoluteModelPath(IniSettings.Instance.FlexiCubesModelPath);
                    progressCallback?.Invoke($"FlexiCubes extraction (model: {flexiPath}) not yet implemented", 0);
                    return currentResult;

                case WorkflowStep.NeRFRefinement:
                    progressCallback?.Invoke("Running NeRF refinement...", 0);
                    // This would use existing VoxelGridNeRF
                    return currentResult;

                case WorkflowStep.MergePointClouds:
                    return MergePointClouds(currentResult);

                case WorkflowStep.AlignPointClouds:
                    return AlignPointClouds(currentResult);

                case WorkflowStep.FilterPointCloud:
                    return FilterPointCloud(currentResult);

                case WorkflowStep.VoxelizePointCloud:
                    return VoxelizePointCloud(currentResult);

                case WorkflowStep.MarchingCubes:
                    return ApplyMarchingCubes(currentResult);

                case WorkflowStep.PoissonReconstruction:
                    return ApplyPoisson(currentResult);

                case WorkflowStep.MeshSmoothing:
                    return SmoothMesh(currentResult);

                case WorkflowStep.MeshDecimation:
                    return DecimateMesh(currentResult);

                case WorkflowStep.UniRigAutoRig:
                    progressCallback?.Invoke("UniRig auto-rigging not yet implemented", 0);
                    return currentResult;

                default:
                    return currentResult;
            }
        }

        private string GetStepName(WorkflowStep step)
        {
            return step switch
            {
                WorkflowStep.LoadImages => "Loading images",
                WorkflowStep.Dust3rReconstruction => "Dust3r reconstruction",
                WorkflowStep.SfMReconstruction => "SfM reconstruction",
                WorkflowStep.TripoSRGeneration => "TripoSR 3D generation",
                WorkflowStep.TripoSGGeneration => "TripoSG 3D generation",
                WorkflowStep.Wonder3DGeneration => "Wonder3D multi-view generation",
                WorkflowStep.TripoSFRefinement => "TripoSF mesh refinement",
                WorkflowStep.FlexiCubesExtraction => "FlexiCubes mesh extraction",
                WorkflowStep.NeRFRefinement => "NeRF refinement",
                WorkflowStep.MergePointClouds => "Merging point clouds",
                WorkflowStep.AlignPointClouds => "Aligning point clouds",
                WorkflowStep.MarchingCubes => "Marching cubes mesh extraction",
                WorkflowStep.MeshSmoothing => "Mesh smoothing",
                WorkflowStep.MeshDecimation => "Mesh decimation",
                WorkflowStep.UniRigAutoRig => "UniRig auto-rigging",
                _ => step.ToString()
            };
        }

        private bool RequiresResult(WorkflowStep step)
        {
            return step switch
            {
                WorkflowStep.LoadImages => false,
                WorkflowStep.LoadPointCloud => false,
                WorkflowStep.LoadMesh => false,
                _ => true
            };
        }

        #endregion

        #region Point Cloud Operations

        private SceneResult? MergePointClouds(SceneResult? result)
        {
            if (result?.Meshes == null || result.Meshes.Count < 2)
                return result;

            // Merge all point clouds into one
            var allVertices = new List<Vector3>();
            var allColors = new List<Vector3>();

            foreach (var mesh in result.Meshes)
            {
                if (mesh.Vertices != null && mesh.Vertices.Count > 0)
                {
                    allVertices.AddRange(mesh.Vertices);
                    if (mesh.Colors != null && mesh.Colors.Count == mesh.Vertices.Count)
                        allColors.AddRange(mesh.Colors);
                    else
                        allColors.AddRange(Enumerable.Repeat(new Vector3(0.5f, 0.5f, 0.5f), mesh.Vertices.Count));
                }
            }

            var merged = new SceneResult();
            merged.Meshes.Add(new MeshData
            {
                Vertices = allVertices,
                Colors = allColors,
                Indices = new List<int>()
            });
            merged.Poses.AddRange(result.Poses);

            return merged;
        }

        private SceneResult? AlignPointClouds(SceneResult? result)
        {
            // Use ICP or similar alignment algorithm
            // For now, return as-is
            return result;
        }

        private SceneResult? FilterPointCloud(SceneResult? result)
        {
            if (result?.Meshes == null || result.Meshes.Count == 0)
                return result;

            // Statistical outlier removal or similar
            // For now, return as-is
            return result;
        }

        private SceneResult? VoxelizePointCloud(SceneResult? result)
        {
            // Convert to voxel grid for mesh extraction
            // For now, return as-is
            return result;
        }

        #endregion

        #region Mesh Operations

        private SceneResult? ApplyMarchingCubes(SceneResult? result)
        {
            // Apply marching cubes to voxel/SDF data
            // For now, return as-is
            return result;
        }

        private SceneResult? ApplyPoisson(SceneResult? result)
        {
            // Apply Poisson surface reconstruction
            // For now, return as-is
            return result;
        }

        private SceneResult? SmoothMesh(SceneResult? result)
        {
            // Apply Laplacian smoothing
            // For now, return as-is
            return result;
        }

        private SceneResult? DecimateMesh(SceneResult? result)
        {
            // Apply mesh decimation
            // For now, return as-is
            return result;
        }

        #endregion

        #region Utility Methods

        private SceneResult? ConvertToSceneResult(AIModelResult aiResult)
        {
            if (!aiResult.Success)
                return null;

            var sceneResult = new SceneResult();

            // If we have vertices and triangles, create a mesh
            if (aiResult.Vertices != null && aiResult.Vertices.Length > 0)
            {
                var mesh = new MeshData
                {
                    Vertices = aiResult.Vertices.ToList(),
                    Colors = aiResult.Colors?.ToList() ??
                             Enumerable.Repeat(new Vector3(0.5f, 0.5f, 0.5f), aiResult.Vertices.Length).ToList(),
                    Indices = aiResult.Triangles?.ToList() ?? new List<int>()
                };
                sceneResult.Meshes.Add(mesh);
            }

            // If we have triplane tokens, store them for later processing
            if (aiResult.TriplaneTokens != null)
            {
                // Store in metadata or custom field
                // For now, we can't directly use triplane tokens without a decoder
            }

            return sceneResult;
        }

        /// <summary>
        /// Load all models based on settings.
        /// </summary>
        public void LoadAllModels()
        {
            var settings = IniSettings.Instance;

            // Load TripoSR if path exists
            var triposrPath = GetAbsoluteModelPath(settings.TripoSRModelPath);
            if (Directory.Exists(triposrPath))
            {
                _ = TripoSR; // Trigger lazy loading
            }

            // Check for other exported ONNX models so we can surface availability in the UI
            ModelStatusChanged?.Invoke("Dust3r", File.Exists(GetAbsoluteModelPath("dust3r.onnx")));
            ModelStatusChanged?.Invoke("TripoSG", Directory.Exists(GetAbsoluteModelPath(settings.TripoSGModelPath)));
            ModelStatusChanged?.Invoke("TripoSF", Directory.Exists(GetAbsoluteModelPath(settings.TripoSFModelPath)));
            ModelStatusChanged?.Invoke("Wonder3D", Directory.Exists(GetAbsoluteModelPath(settings.Wonder3DModelPath)));
            ModelStatusChanged?.Invoke("FlexiCubes", Directory.Exists(GetAbsoluteModelPath(settings.FlexiCubesModelPath)));
            ModelStatusChanged?.Invoke("UniRig", Directory.Exists(GetAbsoluteModelPath(settings.UniRigModelPath)));
        }

        /// <summary>
        /// Unload all models to free memory.
        /// </summary>
        public void UnloadAllModels()
        {
            _tripoSR?.Dispose();
            _tripoSR = null;

            // Add other models as implemented

            ModelStatusChanged?.Invoke("All", false);
        }

        private string GetAbsoluteModelPath(string path)
        {
            if (Path.IsPathRooted(path))
                return path;
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, path);
        }

        #endregion

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
