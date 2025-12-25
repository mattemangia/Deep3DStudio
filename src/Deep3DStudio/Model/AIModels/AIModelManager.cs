using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Deep3DStudio.Configuration;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model.AIModels
{
    // Removed duplicate ImageTo3DModel enum as it exists in Configuration

    public enum WorkflowStep
    {
        LoadImages,
        LoadPointCloud,
        LoadMesh,
        Dust3rReconstruction,
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
                                currentResult = Dust3r?.ReconstructScene(imagePaths);
                            }
                            break;

                        case WorkflowStep.TripoSRGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.TripoSR);
                            break;

                        case WorkflowStep.LGMGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.LGM);
                            break;

                        case WorkflowStep.Wonder3DGeneration:
                            if (imagePaths != null && imagePaths.Count > 0)
                                currentResult = await GenerateFromSingleImageAsync(imagePaths[0], ImageTo3DModel.Wonder3D);
                            break;

                        case WorkflowStep.TripoSFRefinement:
                            // TripoSF logic: If we have an original image, re-generate high quality
                            if (imagePaths != null && imagePaths.Count > 0)
                            {
                                var refinedMesh = TripoSF?.GenerateFromImage(imagePaths[0]);
                                if (refinedMesh != null && refinedMesh.Vertices.Count > 0)
                                {
                                    if(currentResult.Meshes.Count == 0) currentResult.Meshes.Add(refinedMesh);
                                    else currentResult.Meshes[0] = refinedMesh;
                                }
                            }
                            break;

                        case WorkflowStep.MarchingCubes:
                             break;

                        case WorkflowStep.NeRFRefinement:
                            if (currentResult != null && currentResult.Poses.Count > 0)
                            {
                                var nerf = new VoxelGridNeRF();
                                nerf.InitializeFromMesh(currentResult.Meshes);
                                nerf.Train(currentResult.Poses, 5);
                                var mesh = nerf.GetMesh();
                                currentResult.Meshes.Clear();
                                currentResult.Meshes.Add(mesh);
                            }
                            break;
                    }
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
