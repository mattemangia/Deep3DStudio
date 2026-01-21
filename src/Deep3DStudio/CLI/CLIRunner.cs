using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.CLI
{
    public sealed class CLIRunner
    {
        private readonly CommandLineOptions _options;

        public CLIRunner(CommandLineOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
        }

        public int Run()
        {
            if (_options.ShowHelp)
            {
                PrintHelp();
                return 0;
            }

            if (string.IsNullOrWhiteSpace(_options.Command))
            {
                Console.Error.WriteLine("CLI mode enabled but no command was provided.");
                PrintHelp();
                return 1;
            }

            switch (_options.Command.Trim().ToLowerInvariant())
            {
                case "test-all":
                case "test-models":
                case "test":
                    return RunTestAllModels();
                case "test-problematic":
                    return RunTestProblematicModels();
                case "nerf":
                    return RunNeRFWorkflow();
                default:
                    Console.Error.WriteLine($"Unknown CLI command: {_options.Command}");
                    PrintHelp();
                    return 1;
            }
        }

        private int RunTestAllModels()
        {
            var images = ResolveInputImages();
            if (images.Count == 0)
            {
                Console.Error.WriteLine("No input images found. Provide --input <file|dir> or place images in Croco_Examples.");
                return 1;
            }

            Console.WriteLine($"Using {images.Count} image(s):");
            foreach (var img in images)
                Console.WriteLine($"  {img}");

            var manager = AIModelManager.Instance;
            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            manager.ModelLoadProgress += (stage, progress, message) =>
            {
                Console.WriteLine($"[ModelLoad] {stage} {progress:P0} {message}");
                TuiStatusMonitor.Instance.UpdateProgress($"{stage}: {message}", progress);
            };

            try
            {
                var pipelines = new List<WorkflowPipeline>
                {
                    WorkflowPipeline.ImageToDust3rToMesh,
                    WorkflowPipeline.ImageToMast3rToMesh,
                    WorkflowPipeline.ImageToMust3rToMesh,
                    WorkflowPipeline.ImageToSfM,
                    WorkflowPipeline.ImageToTripoSR,
                    WorkflowPipeline.ImageToLGM,
                    WorkflowPipeline.ImageToWonder3D
                };

                bool allOk = true;
                int exitCode = 1;
                MeshData? firstMesh = null;

                try
                {
                    foreach (var pipeline in pipelines)
                    {
                        Console.WriteLine($"=== Running {pipeline.Name} ===");
                        if (RequiresMultiView(pipeline) && images.Count < 2)
                        {
                            Console.WriteLine($"Skipping {pipeline.Name}: requires at least 2 images.");
                            allOk = false;
                            continue;
                        }

                        var result = manager.ExecuteWorkflowAsync(
                            pipeline,
                            images,
                            null,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{pipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{pipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        if (cancellationSource.IsCancellationRequested)
                        {
                            Console.WriteLine("Cancellation requested. Exiting early.");
                            exitCode = 1;
                            return exitCode;
                        }

                        bool expectsMesh = pipeline.Steps.Contains(WorkflowStep.MarchingCubes) ||
                                           pipeline.Steps.Contains(WorkflowStep.TripoSRGeneration) ||
                                           pipeline.Steps.Contains(WorkflowStep.Wonder3DGeneration);

                        var ok = result != null &&
                                 result.Meshes.Count > 0 &&
                                 result.Meshes.Any(m => m.Vertices.Count > 0 && (!expectsMesh || m.Indices.Count > 0));

                        Console.WriteLine(ok
                            ? $"OK: {pipeline.Name} produced geometry."
                            : $"FAIL: {pipeline.Name} produced no geometry.");

                        if (!ok)
                            allOk = false;

                        if (firstMesh == null && result != null)
                        {
                            firstMesh = result.Meshes.FirstOrDefault(m => m.Vertices.Count > 0 && m.Indices.Count > 0);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("Cancelled by user.");
                    exitCode = 1;
                    return exitCode;
                }

                if (firstMesh != null)
                {
                    Console.WriteLine("=== Running TripoSF refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refined = manager.TripoSF?.RefineMesh(firstMesh, cancellationSource.Token);
                        bool ok = refined != null && refined.Vertices.Count > 0;
                        Console.WriteLine(ok
                            ? $"OK: TripoSF refined mesh with {refined!.Vertices.Count} vertices."
                            : "FAIL: TripoSF refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("TripoSF refinement cancelled.");
                        exitCode = 1;
                        return exitCode;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: TripoSF refinement threw: {ex.Message}");
                        allOk = false;
                    }

                    Console.WriteLine("=== Running DeepMeshPrior refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refineScene = new SceneResult { Meshes = new List<MeshData> { firstMesh.Clone() } };
                        var dmpPipeline = new WorkflowPipeline
                        {
                            Name = "DeepMeshPrior Refinement",
                            Steps = new List<WorkflowStep> { WorkflowStep.DeepMeshPriorRefinement }
                        };
                        var dmpResult = manager.ExecuteWorkflowAsync(
                            dmpPipeline,
                            images,
                            refineScene,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{dmpPipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{dmpPipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        bool ok = dmpResult != null &&
                                  dmpResult.Meshes.Count > 0 &&
                                  dmpResult.Meshes.Any(m => m.Vertices.Count > 0);
                        Console.WriteLine(ok
                            ? $"OK: DeepMeshPrior refined mesh with {dmpResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                            : "FAIL: DeepMeshPrior refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("DeepMeshPrior refinement cancelled.");
                        exitCode = 1;
                        return exitCode;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: DeepMeshPrior refinement threw: {ex.Message}");
                        allOk = false;
                    }

                    Console.WriteLine("=== Running GaussianSDF refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refineScene = new SceneResult { Meshes = new List<MeshData> { firstMesh.Clone() } };
                        var gsdfPipeline = new WorkflowPipeline
                        {
                            Name = "GaussianSDF Refinement",
                            Steps = new List<WorkflowStep> { WorkflowStep.GaussianSDFRefinement }
                        };
                        var gsdfResult = manager.ExecuteWorkflowAsync(
                            gsdfPipeline,
                            images,
                            refineScene,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{gsdfPipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{gsdfPipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        bool ok = gsdfResult != null &&
                                  gsdfResult.Meshes.Count > 0 &&
                                  gsdfResult.Meshes.Any(m => m.Vertices.Count > 0);
                        Console.WriteLine(ok
                            ? $"OK: GaussianSDF refined mesh with {gsdfResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                            : "FAIL: GaussianSDF refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("GaussianSDF refinement cancelled.");
                        exitCode = 1;
                        return exitCode;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: GaussianSDF refinement threw: {ex.Message}");
                        allOk = false;
                    }

                    Console.WriteLine("=== Running UniRig auto-rig ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var rig = manager.UniRig?.RigMesh(firstMesh);
                        bool ok = rig != null && rig.Success;
                        Console.WriteLine(ok
                            ? $"OK: UniRig produced {rig!.JointPositions?.Length ?? 0} joints."
                            : $"FAIL: UniRig rigging failed ({rig?.StatusMessage ?? "no result"}).");
                        if (!ok)
                            allOk = false;
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("UniRig rigging cancelled.");
                        exitCode = 1;
                        return exitCode;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: UniRig rigging threw: {ex.Message}");
                        allOk = false;
                    }
                }
                else
                {
                    Console.WriteLine("Skipping TripoSF: no mesh with faces produced by earlier models.");
                    allOk = false;

                    var unirigExample = FindUniRigExampleMesh();
                    if (!string.IsNullOrEmpty(unirigExample))
                    {
                        Console.WriteLine("=== Running UniRig auto-rig (example mesh) ===");
                        try
                        {
                            cancellationSource.Token.ThrowIfCancellationRequested();
                            var rig = manager.UniRig?.RigMeshFromFile(unirigExample);
                            bool ok = rig != null && rig.Success;
                            Console.WriteLine(ok
                                ? $"OK: UniRig produced {rig!.JointPositions?.Length ?? 0} joints."
                                : $"FAIL: UniRig rigging failed ({rig?.StatusMessage ?? "no result"}).");
                            if (!ok)
                                allOk = false;
                        }
                        catch (OperationCanceledException)
                        {
                            Console.WriteLine("UniRig rigging cancelled.");
                            exitCode = 1;
                            return exitCode;
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"FAIL: UniRig rigging threw: {ex.Message}");
                            allOk = false;
                        }
                    }
                    else
                    {
                        Console.WriteLine("Skipping UniRig: no example mesh found.");
                        allOk = false;
                    }
                }

                Console.WriteLine("=== Running NeRF reconstruction (timeout 5m) ===");
                try
                {
                    var settings = IniSettings.Instance;
                    var nerfPipeline = new WorkflowPipeline
                    {
                        Name = "Images -> Reconstruction -> NeRF",
                        Steps = new List<WorkflowStep>
                        {
                            WorkflowStep.LoadImages,
                            settings.ReconstructionMethod switch
                            {
                                ReconstructionMethod.Mast3r => WorkflowStep.Mast3rReconstruction,
                                ReconstructionMethod.Must3r => WorkflowStep.Must3rReconstruction,
                                ReconstructionMethod.FeatureMatching => WorkflowStep.SfMReconstruction,
                                _ => WorkflowStep.Dust3rReconstruction
                            },
                            WorkflowStep.NeRFRefinement
                        }
                    };

                    using var nerfCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(cancellationSource.Token);
                    nerfCts.CancelAfter(TimeSpan.FromMinutes(5));

                    var nerfResult = manager.ExecuteWorkflowAsync(
                        nerfPipeline,
                        images,
                        null,
                        (msg, progress) =>
                        {
                            Console.WriteLine($"[{nerfPipeline.Name}] {progress:P0} {msg}");
                            TuiStatusMonitor.Instance.UpdateProgress($"{nerfPipeline.Name}: {msg}", progress);
                        },
                        nerfCts.Token
                    ).GetAwaiter().GetResult();

                    if (nerfCts.IsCancellationRequested && !cancellationSource.IsCancellationRequested)
                    {
                        Console.WriteLine("NeRF timeout reached. Returning partial result.");
                    }

                    bool ok = nerfResult != null &&
                              nerfResult.Meshes.Count > 0 &&
                              nerfResult.Meshes.Any(m => m.Vertices.Count > 0);
                    Console.WriteLine(ok
                        ? $"OK: NeRF produced mesh with {nerfResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                        : "FAIL: NeRF produced no geometry.");
                    if (!ok)
                        allOk = false;
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("NeRF reconstruction cancelled.");
                    exitCode = 1;
                    return exitCode;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"FAIL: NeRF reconstruction threw: {ex.Message}");
                    allOk = false;
                }

                exitCode = allOk ? 0 : 1;
                return exitCode;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private int RunTestProblematicModels()
        {
            var images = ResolveInputImages();
            if (images.Count == 0)
            {
                Console.Error.WriteLine("No input images found. Provide --input <file|dir> or place images in Croco_Examples.");
                return 1;
            }

            Console.WriteLine($"Using {images.Count} image(s):");
            foreach (var img in images)
                Console.WriteLine($"  {img}");

            var manager = AIModelManager.Instance;
            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            manager.ModelLoadProgress += (stage, progress, message) =>
            {
                Console.WriteLine($"[ModelLoad] {stage} {progress:P0} {message}");
                TuiStatusMonitor.Instance.UpdateProgress($"{stage}: {message}", progress);
            };

            try
            {
                var pipelines = new List<WorkflowPipeline>
                {
                    WorkflowPipeline.ImageToTripoSR,
                    WorkflowPipeline.ImageToLGM,
                    // WorkflowPipeline.ImageToWonder3D
                };

                bool allOk = true;
                int exitCode = 1;
                MeshData? firstMesh = null;

                try
                {
                    foreach (var pipeline in pipelines)
                    {
                        Console.WriteLine($"=== Running {pipeline.Name} ===");
                        var result = manager.ExecuteWorkflowAsync(
                            pipeline,
                            images,
                            null,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{pipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{pipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        if (cancellationSource.IsCancellationRequested)
                        {
                            Console.WriteLine("Cancellation requested. Exiting early.");
                            exitCode = 1;
                            return exitCode;
                        }

                        bool expectsMesh = !pipeline.Name.Contains("LGM");

                        var ok = result != null &&
                                 result.Meshes.Count > 0 &&
                                 result.Meshes.Any(m => m.Vertices.Count > 0 && (!expectsMesh || m.Indices.Count > 0));

                        Console.WriteLine(ok
                            ? $"OK: {pipeline.Name} produced geometry."
                            : $"FAIL: {pipeline.Name} produced no geometry.");

                        if (!ok)
                            allOk = false;

                        if (firstMesh == null && result != null)
                        {
                            firstMesh = result.Meshes.FirstOrDefault(m => m.Vertices.Count > 0 && m.Indices.Count > 0);
                        }
                    }
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("Cancelled by user.");
                    exitCode = 1;
                    return exitCode;
                }

                if (firstMesh != null)
                {
                    Console.WriteLine("=== Running TripoSF refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refined = manager.TripoSF?.RefineMesh(firstMesh, cancellationSource.Token);
                        bool ok = refined != null && refined.Vertices.Count > 0;
                        Console.WriteLine(ok
                            ? $"OK: TripoSF refined mesh with {refined!.Vertices.Count} vertices."
                            : "FAIL: TripoSF refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: TripoSF refinement threw: {ex.Message}");
                        allOk = false;
                    }

                    /*
                    Console.WriteLine("=== Running DeepMeshPrior refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refineScene = new SceneResult { Meshes = new List<MeshData> { firstMesh.Clone() } };
                        var dmpPipeline = new WorkflowPipeline
                        {
                            Name = "DeepMeshPrior Refinement",
                            Steps = new List<WorkflowStep> { WorkflowStep.DeepMeshPriorRefinement }
                        };
                        var dmpResult = manager.ExecuteWorkflowAsync(
                            dmpPipeline,
                            images,
                            refineScene,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{dmpPipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{dmpPipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        bool ok = dmpResult != null &&
                                  dmpResult.Meshes.Count > 0 &&
                                  dmpResult.Meshes.Any(m => m.Vertices.Count > 0);
                        Console.WriteLine(ok
                            ? $"OK: DeepMeshPrior refined mesh with {dmpResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                            : "FAIL: DeepMeshPrior refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: DeepMeshPrior refinement threw: {ex.Message}");
                        allOk = false;
                    }
                    */

                    Console.WriteLine("=== Running GaussianSDF refinement ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var refineScene = new SceneResult { Meshes = new List<MeshData> { firstMesh.Clone() } };
                        var gsdfPipeline = new WorkflowPipeline
                        {
                            Name = "GaussianSDF Refinement",
                            Steps = new List<WorkflowStep> { WorkflowStep.GaussianSDFRefinement }
                        };
                        var gsdfResult = manager.ExecuteWorkflowAsync(
                            gsdfPipeline,
                            images,
                            refineScene,
                            (msg, progress) =>
                            {
                                Console.WriteLine($"[{gsdfPipeline.Name}] {progress:P0} {msg}");
                                TuiStatusMonitor.Instance.UpdateProgress($"{gsdfPipeline.Name}: {msg}", progress);
                            },
                            cancellationSource.Token
                        ).GetAwaiter().GetResult();

                        bool ok = gsdfResult != null &&
                                  gsdfResult.Meshes.Count > 0 &&
                                  gsdfResult.Meshes.Any(m => m.Vertices.Count > 0);
                        Console.WriteLine(ok
                            ? $"OK: GaussianSDF refined mesh with {gsdfResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                            : "FAIL: GaussianSDF refinement produced no geometry.");
                        if (!ok)
                            allOk = false;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: GaussianSDF refinement threw: {ex.Message}");
                        allOk = false;
                    }

                    Console.WriteLine("=== Running UniRig auto-rig ===");
                    try
                    {
                        cancellationSource.Token.ThrowIfCancellationRequested();
                        var rig = manager.UniRig?.RigMesh(firstMesh);
                        bool ok = rig != null && rig.Success;
                        Console.WriteLine(ok
                            ? $"OK: UniRig produced {rig!.JointPositions?.Length ?? 0} joints."
                            : $"FAIL: UniRig rigging failed ({rig?.StatusMessage ?? "no result"}).");
                        if (!ok)
                            allOk = false;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"FAIL: UniRig rigging threw: {ex.Message}");
                        allOk = false;
                    }
                }
                else
                {
                    Console.WriteLine("Skipping refinements: no mesh with faces produced by earlier models.");
                    allOk = false;
                }

                Console.WriteLine("=== Running NeRF reconstruction (timeout 5m) ===");
                try
                {
                    var settings = IniSettings.Instance;
                    var nerfPipeline = new WorkflowPipeline
                    {
                        Name = "Images -> Reconstruction -> NeRF",
                        Steps = new List<WorkflowStep>
                        {
                            WorkflowStep.LoadImages,
                            settings.ReconstructionMethod switch
                            {
                                ReconstructionMethod.Mast3r => WorkflowStep.Mast3rReconstruction,
                                ReconstructionMethod.Must3r => WorkflowStep.Must3rReconstruction,
                                ReconstructionMethod.FeatureMatching => WorkflowStep.SfMReconstruction,
                                _ => WorkflowStep.Dust3rReconstruction
                            },
                            WorkflowStep.NeRFRefinement
                        }
                    };

                    using var nerfCts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(cancellationSource.Token);
                    nerfCts.CancelAfter(TimeSpan.FromMinutes(5));

                    var nerfResult = manager.ExecuteWorkflowAsync(
                        nerfPipeline,
                        images,
                        null,
                        (msg, progress) =>
                        {
                            Console.WriteLine($"[{nerfPipeline.Name}] {progress:P0} {msg}");
                            TuiStatusMonitor.Instance.UpdateProgress($"{nerfPipeline.Name}: {msg}", progress);
                        },
                        nerfCts.Token
                    ).GetAwaiter().GetResult();

                    bool ok = nerfResult != null &&
                              nerfResult.Meshes.Count > 0 &&
                              nerfResult.Meshes.Any(m => m.Vertices.Count > 0);
                    Console.WriteLine(ok
                        ? $"OK: NeRF produced mesh with {nerfResult!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                        : "FAIL: NeRF produced no geometry.");
                    if (!ok)
                        allOk = false;
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("NeRF reconstruction cancelled.");
                    exitCode = 1;
                    return exitCode;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"FAIL: NeRF reconstruction threw: {ex.Message}");
                    allOk = false;
                }

                exitCode = allOk ? 0 : 1;
                return exitCode;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private int RunNeRFWorkflow()
        {
            var images = ResolveInputImages();
            if (images.Count < 2)
            {
                Console.Error.WriteLine("NeRF requires at least 2 images.");
                return 1;
            }

            var settings = IniSettings.Instance;
            int? originalIterations = null;
            if (_options.NerfIterations.HasValue)
            {
                originalIterations = settings.NeRFIterations;
                settings.NeRFIterations = Math.Max(1, _options.NerfIterations.Value);
            }

            var cts = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cts);

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
                Console.WriteLine("Cancellation requested. Returning partial NeRF mesh...");
            };

            var pipeline = new WorkflowPipeline
            {
                Name = "Images -> Reconstruction -> NeRF",
                Steps = new List<WorkflowStep>
                {
                    WorkflowStep.LoadImages,
                    settings.ReconstructionMethod switch
                    {
                        ReconstructionMethod.Mast3r => WorkflowStep.Mast3rReconstruction,
                        ReconstructionMethod.Must3r => WorkflowStep.Must3rReconstruction,
                        ReconstructionMethod.FeatureMatching => WorkflowStep.SfMReconstruction,
                        _ => WorkflowStep.Dust3rReconstruction
                    },
                    WorkflowStep.NeRFRefinement
                }
            };

            var manager = AIModelManager.Instance;
            Console.WriteLine($"=== Running {pipeline.Name} (Ctrl+C to cancel) ===");
            try
            {
                var result = manager.ExecuteWorkflowAsync(
                    pipeline,
                    images,
                    null,
                    (msg, progress) =>
                    {
                        Console.WriteLine($"[{pipeline.Name}] {progress:P0} {msg}");
                        TuiStatusMonitor.Instance.UpdateProgress($"{pipeline.Name}: {msg}", progress);
                    },
                    cts.Token
                ).GetAwaiter().GetResult();

                bool ok = result != null &&
                          result.Meshes.Count > 0 &&
                          result.Meshes.Any(m => m.Vertices.Count > 0);

                Console.WriteLine(ok
                    ? $"OK: NeRF produced mesh with {result!.Meshes.Sum(m => m.Vertices.Count)} vertices."
                    : "FAIL: NeRF produced no geometry.");

                return ok ? 0 : 1;
            }
            finally
            {
                if (originalIterations.HasValue)
                    settings.NeRFIterations = originalIterations.Value;
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private static bool RequiresMultiView(WorkflowPipeline pipeline)
        {
            return pipeline.Steps.Contains(WorkflowStep.Dust3rReconstruction) ||
                   pipeline.Steps.Contains(WorkflowStep.Mast3rReconstruction) ||
                   pipeline.Steps.Contains(WorkflowStep.Must3rReconstruction) ||
                   pipeline.Steps.Contains(WorkflowStep.SfMReconstruction);
        }

        private List<string> ResolveInputImages()
        {
            var images = new List<string>();
            if (!string.IsNullOrWhiteSpace(_options.InputPath))
                images.AddRange(ExpandImagePaths(_options.InputPath));

            foreach (var arg in _options.ExtraArgs)
            {
                if (File.Exists(arg) && IsImageFile(arg))
                    images.Add(Path.GetFullPath(arg));
            }

            if (images.Count == 0)
            {
                var crocoDir = FindCrocoExamples();
                if (crocoDir != null)
                    images.AddRange(ExpandImagePaths(crocoDir));
            }

            return images.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        }

        private static IEnumerable<string> ExpandImagePaths(string path)
        {
            var fullPath = Path.GetFullPath(path);
            if (File.Exists(fullPath))
            {
                if (IsImageFile(fullPath))
                    return new[] { fullPath };
                return Array.Empty<string>();
            }

            if (Directory.Exists(fullPath))
            {
                return Directory.GetFiles(fullPath)
                    .Where(IsImageFile)
                    .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .Select(Path.GetFullPath);
            }

            return Array.Empty<string>();
        }

        private static bool IsImageFile(string path)
        {
            var ext = Path.GetExtension(path).ToLowerInvariant();
            return ext is ".png" or ".jpg" or ".jpeg" or ".bmp";
        }

        private static string? FindCrocoExamples()
        {
            var exeDir = AppDomain.CurrentDomain.BaseDirectory;
            var candidates = new[]
            {
                Path.Combine(exeDir, "Croco_Examples"),
                Path.Combine(exeDir, "..", "..", "..", "Croco_Examples"),
                Path.Combine(exeDir, "..", "..", "..", "..", "src", "Deep3DStudio", "Croco_Examples"),
                Path.Combine(Environment.CurrentDirectory, "Croco_Examples"),
                Path.Combine(Environment.CurrentDirectory, "src", "Deep3DStudio", "Croco_Examples")
            };

            foreach (var candidate in candidates)
            {
                var full = Path.GetFullPath(candidate);
                if (Directory.Exists(full))
                    return full;
            }

            return null;
        }

        private static string? FindUniRigExampleMesh()
        {
            var exeDir = AppDomain.CurrentDomain.BaseDirectory;
            var candidates = new[]
            {
                Path.Combine(exeDir, "Unirig_examples"),
                Path.Combine(exeDir, "..", "..", "..", "Unirig_examples"),
                Path.Combine(exeDir, "..", "..", "..", "..", "src", "Deep3DStudio", "Unirig_examples"),
                Path.Combine(Environment.CurrentDirectory, "Unirig_examples"),
                Path.Combine(Environment.CurrentDirectory, "src", "Deep3DStudio", "Unirig_examples")
            };

            foreach (var candidate in candidates)
            {
                var full = Path.GetFullPath(candidate);
                if (!Directory.Exists(full))
                    continue;

                var glb = Directory.GetFiles(full, "*.glb")
                    .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .FirstOrDefault();
                if (!string.IsNullOrEmpty(glb))
                    return glb;
            }

            return null;
        }

        private static void PrintHelp()
        {
            Console.WriteLine("Deep3DStudio CLI (placeholder)");
            Console.WriteLine("Usage:");
            Console.WriteLine("  --cli --command test-all [--input <file|dir>] [--verbose]");
            Console.WriteLine("  --cli --command nerf [--input <file|dir>] [--nerf-iterations N]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --cli, --headless   Run without GUI");
            Console.WriteLine("  --command, --mode   CLI command to run");
            Console.WriteLine("  --model             Model name to use");
            Console.WriteLine("  --input             Input path");
            Console.WriteLine("  --output            Output path");
            Console.WriteLine("  --nerf-iterations   Override NeRF iteration count (Ctrl+C cancels and returns partial mesh)");
            Console.WriteLine("  --verbose, -v       Verbose logging");
            Console.WriteLine("  --help, -h, -?      Show this help");
        }
    }
}
