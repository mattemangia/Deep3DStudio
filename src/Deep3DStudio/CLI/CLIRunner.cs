using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using Deep3DStudio.IO;
using Deep3DStudio.Scene;
using Deep3DStudio.Meshing;
using OpenTK.Mathematics;

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
                case "reconstruct":
                    return RunReconstructCommand();
                case "mesh":
                    return RunMeshCommand();
                case "refine":
                    return RunRefineCommand();
                case "export":
                    return RunExportCommand();
                case "project":
                    return RunProjectCommand();
                case "pipeline":
                    return RunPipelineCommand();
                case "reconstruct-pointcloud":
                    return RunReconstructPointCloud();
                case "mesh-from-pointcloud":
                    return RunMeshFromPointCloud();
                case "refine-mesh":
                    return RunRefineMesh();
                case "export-mesh":
                    return RunExportMesh();
                case "export-pointcloud":
                    return RunExportPointCloud();
                case "project-export-all":
                    return RunProjectExportAll();
                case "pipeline-run":
                    return RunPipelineRun();
                case "test-all":
                case "test-models":
                case "test":
                    return RunTestAllModels();
                case "test-problematic":
                    return RunTestProblematicModels();
                case "nerf":
                    return RunNeRFWorkflow();
                case "pc-mesh-triposf":
                case "pointcloud-mesh-triposf":
                    return RunPointCloudToMeshTripoSF();
                case "reextract-python":
                case "reextract-env":
                case "reextract-python-env":
                    return RunPythonReextract();
                default:
                    Console.Error.WriteLine($"Unknown CLI command: {_options.Command}");
                    PrintHelp();
                    return 1;
            }
        }

        private int RunReconstructCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "pointcloud" or "pc" => RunReconstructPointCloud(),
                _ => UnknownSubcommand("reconstruct", "pointcloud")
            };
        }

        private int RunMeshCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "from-pointcloud" or "frompc" or "pointcloud" => RunMeshFromPointCloud(),
                _ => UnknownSubcommand("mesh", "from-pointcloud")
            };
        }

        private int RunRefineCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "mesh" => RunRefineMesh(),
                _ => UnknownSubcommand("refine", "mesh")
            };
        }

        private int RunExportCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "mesh" => RunExportMesh(),
                "pointcloud" or "pc" => RunExportPointCloud(),
                _ => UnknownSubcommand("export", "mesh|pointcloud")
            };
        }

        private int RunProjectCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "export-all" => RunProjectExportAll(),
                _ => UnknownSubcommand("project", "export-all")
            };
        }

        private int RunPipelineCommand()
        {
            var subcommand = NormalizeToken(_options.Subcommand);
            return subcommand switch
            {
                "run" => RunPipelineRun(),
                _ => UnknownSubcommand("pipeline", "run")
            };
        }

        private int RunReconstructPointCloud()
        {
            var images = ResolveInputImages();
            if (images.Count < 2)
            {
                Console.Error.WriteLine("Reconstruction requires at least 2 images. Provide --input <dir|img1> [--input <img2> ...].");
                return 1;
            }

            var pipeline = NormalizeReconstructionPipeline(_options.Pipeline) ?? "mast3r";
            var fallback = NormalizeReconstructionPipeline(_options.FallbackPipeline);
            var fallbackEnabled = !string.IsNullOrWhiteSpace(fallback) &&
                                  !string.Equals(fallback, "none", StringComparison.OrdinalIgnoreCase) &&
                                  !string.Equals(fallback, "off", StringComparison.OrdinalIgnoreCase);

            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            var manager = AIModelManager.Instance;
            try
            {
                Console.WriteLine($"Running reconstruction pipeline: {pipeline}");
                var result = ExecuteReconstructionWorkflow(manager, pipeline, images, cancellationSource.Token);
                var hasGeometry = TryBuildMergedGeometry(result, out var merged, out var reason);
                var usedPipeline = pipeline;

                if (!hasGeometry && fallbackEnabled && !string.Equals(fallback, pipeline, StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine($"Primary reconstruction returned no valid geometry ({reason}). Falling back to: {fallback}");
                    var fallbackResult = ExecuteReconstructionWorkflow(manager, fallback!, images, cancellationSource.Token);
                    if (TryBuildMergedGeometry(fallbackResult, out merged, out reason))
                    {
                        result = fallbackResult;
                        hasGeometry = true;
                        usedPipeline = fallback!;
                    }
                }

                if (!hasGeometry || merged == null)
                {
                    Console.Error.WriteLine($"FAILED_NO_GEOMETRY: {reason}");
                    WriteRunManifest(images, pipeline, fallback, "FAILED_NO_GEOMETRY", reason, 0, 0, null);
                    return 1;
                }

                var outputDir = ResolveOutputDirectory(images.FirstOrDefault());
                Directory.CreateDirectory(outputDir);
                var pcFormats = ResolvePointCloudFormats("ply");
                var meshFormats = ResolveMeshFormats();
                var baseName = $"{usedPipeline}_reconstruction";
                var exportedPointCloud = new List<string>();
                var exportedMeshes = new List<string>();

                foreach (var format in pcFormats)
                {
                    var targetPath = Path.Combine(outputDir, $"{baseName}.{format}");
                    SavePointCloud(merged, targetPath, _options.IncludeColors);
                    exportedPointCloud.Add(targetPath);
                    Console.WriteLine($"Saved point cloud: {targetPath}");
                }

                foreach (var format in meshFormats)
                {
                    var targetPath = Path.Combine(outputDir, $"{baseName}.{format}");
                    SaveMesh(merged, targetPath);
                    exportedMeshes.Add(targetPath);
                    Console.WriteLine($"Saved mesh: {targetPath}");
                }

                WriteRunManifest(
                    images,
                    usedPipeline,
                    fallback,
                    "SUCCESS",
                    "Geometry reconstructed",
                    merged.Vertices.Count,
                    merged.Indices.Count / 3,
                    exportedPointCloud.Concat(exportedMeshes).ToList());

                Console.WriteLine($"SUCCESS: {merged.Vertices.Count} points reconstructed with {usedPipeline}.");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled.");
                return 1;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Reconstruction failed: {ex.Message}");
                return 1;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private int RunMeshFromPointCloud()
        {
            var inputPath = ResolveInputFilePath(new[] { ".ply", ".xyz" });
            if (string.IsNullOrWhiteSpace(inputPath) || !File.Exists(inputPath))
            {
                Console.Error.WriteLine("Point cloud input not found. Provide --input <file.ply|file.xyz>.");
                return 1;
            }

            PointCloudObject pc;
            try
            {
                pc = PointCloudImporter.Load(inputPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load point cloud: {ex.Message}");
                return 1;
            }

            if (pc.Points.Count == 0)
            {
                Console.Error.WriteLine("Point cloud is empty.");
                return 1;
            }

            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            var manager = AIModelManager.Instance;
            try
            {
                int targetRes = _options.VoxelResolution.HasValue ? Math.Clamp(_options.VoxelResolution.Value, 32, 512) : 256;
                Console.WriteLine($"Voxelizing point cloud (targetRes={targetRes})...");
                var voxelized = VoxelizePointCloud(pc.Points, targetRes);
                if (voxelized.grid == null)
                {
                    Console.Error.WriteLine("Voxelization failed.");
                    return 1;
                }

                float isoLevel = _options.IsoLevel.HasValue
                    ? Math.Clamp(_options.IsoLevel.Value, -2.0f, 2.0f)
                    : 0.5f;
                Console.WriteLine($"Running marching cubes (iso={isoLevel:F4})...");
                var mesher = new MarchingCubesMesher();
                Action<string, float> cliProgress = (msg, p) =>
                {
                    TuiStatusMonitor.Instance.UpdateProgress(msg, p);
                };
                var baseMesh = mesher.GenerateMesh(voxelized.grid, voxelized.origin, voxelized.voxelSize, isoLevel, cliProgress);
                if (baseMesh.Vertices.Count == 0 || baseMesh.Indices.Count == 0)
                {
                    Console.Error.WriteLine("Marching cubes produced no geometry.");
                    return 1;
                }

                var outputDir = ResolveOutputDirectory(inputPath);
                Directory.CreateDirectory(outputDir);
                var baseName = ResolveOutputBaseName(inputPath);

                var baseMeshPath = Path.Combine(outputDir, $"{baseName}_base.ply");
                SaveMesh(baseMesh, baseMeshPath);
                Console.WriteLine($"Saved base mesh: {baseMeshPath}");

                var current = baseMesh;
                var refiners = ResolveRefiners();
                var imageInputs = ResolveInputImages();
                foreach (var refiner in refiners)
                {
                    Console.WriteLine($"Running refiner: {refiner}");
                    var refined = ApplyRefiner(manager, current, refiner, imageInputs, cancellationSource.Token);
                    if (refined == null || refined.Vertices.Count == 0)
                    {
                        Console.WriteLine($"Skipping {refiner}: no geometry generated.");
                        continue;
                    }

                    current = refined;
                    var stagePath = Path.Combine(outputDir, $"{baseName}_{refiner}.ply");
                    SaveMesh(current, stagePath);
                    Console.WriteLine($"Saved {refiner} mesh: {stagePath}");
                }

                var meshFormats = ResolveMeshFormats("ply");
                ExportMeshToFormats(current, outputDir, baseName, meshFormats, _options.OutputPath);
                Console.WriteLine($"SUCCESS: final mesh has {current.Vertices.Count} vertices and {current.Indices.Count / 3} triangles.");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled.");
                return 1;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Mesh generation failed: {ex.Message}");
                return 1;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private int RunRefineMesh()
        {
            var inputPath = ResolveInputFilePath(new[] { ".obj", ".ply", ".stl" });
            if (string.IsNullOrWhiteSpace(inputPath) || !File.Exists(inputPath))
            {
                Console.Error.WriteLine("Mesh input not found. Provide --input <mesh file>.");
                return 1;
            }

            MeshData mesh;
            try
            {
                mesh = LoadMeshAnyFormat(inputPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load mesh: {ex.Message}");
                return 1;
            }

            if (mesh.Vertices.Count == 0)
            {
                Console.Error.WriteLine("Input mesh has no vertices.");
                return 1;
            }

            var refiners = ResolveRefiners();
            if (refiners.Count == 0)
            {
                Console.Error.WriteLine("No refiners specified. Use --refiners triposf|gaussiansdf|deepmeshprior|nerf.");
                return 1;
            }

            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            var manager = AIModelManager.Instance;
            try
            {
                var outputDir = ResolveOutputDirectory(inputPath);
                Directory.CreateDirectory(outputDir);
                var baseName = ResolveOutputBaseName(inputPath);
                var imageInputs = ResolveInputImages();
                var current = mesh;

                foreach (var refiner in refiners)
                {
                    Console.WriteLine($"Running refiner: {refiner}");
                    var refined = ApplyRefiner(manager, current, refiner, imageInputs, cancellationSource.Token);
                    if (refined == null || refined.Vertices.Count == 0)
                    {
                        Console.WriteLine($"Skipping {refiner}: no geometry generated.");
                        continue;
                    }

                    current = refined;
                    var stagePath = Path.Combine(outputDir, $"{baseName}_{refiner}.ply");
                    SaveMesh(current, stagePath);
                    Console.WriteLine($"Saved {refiner} mesh: {stagePath}");
                }

                var meshFormats = ResolveMeshFormats("ply");
                ExportMeshToFormats(current, outputDir, baseName, meshFormats, _options.OutputPath);
                Console.WriteLine($"SUCCESS: refined mesh has {current.Vertices.Count} vertices and {current.Indices.Count / 3} triangles.");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled.");
                return 1;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Refine failed: {ex.Message}");
                return 1;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private int RunExportMesh()
        {
            var inputPath = ResolveInputFilePath(new[] { ".obj", ".ply", ".stl" });
            if (string.IsNullOrWhiteSpace(inputPath) || !File.Exists(inputPath))
            {
                Console.Error.WriteLine("Mesh input not found. Provide --input <mesh file>.");
                return 1;
            }

            MeshData mesh;
            try
            {
                mesh = LoadMeshAnyFormat(inputPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load mesh: {ex.Message}");
                return 1;
            }

            if (mesh.Vertices.Count == 0)
            {
                Console.Error.WriteLine("Input mesh has no geometry.");
                return 1;
            }

            var outputDir = ResolveOutputDirectory(inputPath);
            Directory.CreateDirectory(outputDir);
            var baseName = ResolveOutputBaseName(inputPath);
            var formats = ResolveMeshFormats("obj", "ply");
            ExportMeshToFormats(mesh, outputDir, baseName, formats, _options.OutputPath);
            Console.WriteLine($"SUCCESS: exported mesh in {formats.Count} format(s).");
            return 0;
        }

        private int RunExportPointCloud()
        {
            var inputPath = ResolveInputFilePath(new[] { ".ply", ".xyz", ".obj", ".stl" });
            if (string.IsNullOrWhiteSpace(inputPath) || !File.Exists(inputPath))
            {
                Console.Error.WriteLine("Input not found. Provide --input <point cloud or mesh file>.");
                return 1;
            }

            MeshData sourceMesh;
            var ext = Path.GetExtension(inputPath).ToLowerInvariant();
            try
            {
                if (ext is ".ply" or ".xyz")
                {
                    var pc = PointCloudImporter.Load(inputPath);
                    sourceMesh = new MeshData
                    {
                        Vertices = new List<Vector3>(pc.Points),
                        Colors = new List<Vector3>(pc.Colors)
                    };
                }
                else
                {
                    sourceMesh = LoadMeshAnyFormat(inputPath);
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load input: {ex.Message}");
                return 1;
            }

            if (sourceMesh.Vertices.Count == 0)
            {
                Console.Error.WriteLine("Input has no points.");
                return 1;
            }

            var outputDir = ResolveOutputDirectory(inputPath);
            Directory.CreateDirectory(outputDir);
            var baseName = ResolveOutputBaseName(inputPath);
            var formats = ResolvePointCloudFormats("ply");
            foreach (var format in formats)
            {
                var outPath = Path.Combine(outputDir, $"{baseName}.{format}");
                SavePointCloud(sourceMesh, outPath, _options.IncludeColors);
                Console.WriteLine($"Saved point cloud: {outPath}");
            }

            Console.WriteLine($"SUCCESS: exported point cloud in {formats.Count} format(s).");
            return 0;
        }

        private int RunProjectExportAll()
        {
            if (string.IsNullOrWhiteSpace(_options.ProjectPath) || !File.Exists(_options.ProjectPath))
            {
                Console.Error.WriteLine("Project file not found. Provide --project <project.d3d>.");
                return 1;
            }

            ProjectState state;
            try
            {
                state = ProjectManager.LoadProject(_options.ProjectPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load project: {ex.Message}");
                return 1;
            }

            var outputDir = ResolveOutputDirectory(_options.ProjectPath);
            var meshOutDir = Path.Combine(outputDir, "meshes");
            var pointCloudOutDir = Path.Combine(outputDir, "pointclouds");
            Directory.CreateDirectory(meshOutDir);
            Directory.CreateDirectory(pointCloudOutDir);

            var meshFormats = ResolveMeshFormats("ply", "obj");
            var pointFormats = ResolvePointCloudFormats("ply");

            var allObjects = EnumerateSceneObjects(state.Scene.Objects);
            int meshCount = 0;
            int pointCloudCount = 0;
            var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var obj in allObjects)
            {
                if (!_options.IncludeHidden && !obj.Visible)
                    continue;

                var safeBaseName = GetUniqueName(SanitizeFileName(obj.Name), usedNames);
                if (obj is MeshObjectDTO meshObj)
                {
                    var mesh = ConvertMeshDto(meshObj);
                    if (mesh.Vertices.Count == 0)
                        continue;

                    foreach (var format in meshFormats)
                    {
                        var outPath = Path.Combine(meshOutDir, $"{safeBaseName}.{format}");
                        SaveMesh(mesh, outPath);
                        Console.WriteLine($"Saved mesh: {outPath}");
                    }

                    meshCount++;
                }
                else if (obj is PointCloudObjectDTO pointObj)
                {
                    var mesh = ConvertPointCloudDto(pointObj);
                    if (mesh.Vertices.Count == 0)
                        continue;

                    foreach (var format in pointFormats)
                    {
                        var outPath = Path.Combine(pointCloudOutDir, $"{safeBaseName}.{format}");
                        SavePointCloud(mesh, outPath, _options.IncludeColors);
                        Console.WriteLine($"Saved point cloud: {outPath}");
                    }

                    pointCloudCount++;
                }
            }

            Console.WriteLine($"SUCCESS: exported {meshCount} mesh object(s) and {pointCloudCount} point cloud object(s).");
            return meshCount > 0 || pointCloudCount > 0 ? 0 : 1;
        }

        private int RunPipelineRun()
        {
            var images = ResolveInputImages();
            if (images.Count < 2)
            {
                Console.Error.WriteLine("Pipeline requires at least 2 images.");
                return 1;
            }

            var reconstruction = NormalizeReconstructionPipeline(_options.Pipeline) ?? "mast3r";
            WorkflowPipeline pipeline = reconstruction switch
            {
                "dust3r" => WorkflowPipeline.ImageToDust3rToMesh,
                "must3r" => WorkflowPipeline.ImageToMust3rToMesh,
                "sfm" => WorkflowPipeline.ImageToSfM,
                _ => WorkflowPipeline.ImageToMast3rToMesh
            };

            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            var manager = AIModelManager.Instance;
            try
            {
                Console.WriteLine($"Running pipeline: {pipeline.Name}");
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

                if (!TryBuildMergedGeometry(result, out var merged, out var reason) || merged == null)
                {
                    Console.Error.WriteLine($"FAILED_NO_GEOMETRY: {reason}");
                    return 1;
                }

                var refiners = ResolveRefiners();
                var current = merged;
                foreach (var refiner in refiners)
                {
                    var refined = ApplyRefiner(manager, current, refiner, images, cancellationSource.Token);
                    if (refined != null && refined.Vertices.Count > 0)
                        current = refined;
                }

                var outputDir = ResolveOutputDirectory(images.FirstOrDefault());
                Directory.CreateDirectory(outputDir);
                var baseName = $"{reconstruction}_pipeline";
                var meshFormats = ResolveMeshFormats("ply");
                var pointFormats = ResolvePointCloudFormats("ply");

                ExportMeshToFormats(current, outputDir, baseName, meshFormats, _options.OutputPath);
                foreach (var format in pointFormats)
                {
                    var outPath = Path.Combine(outputDir, $"{baseName}.{format}");
                    SavePointCloud(current, outPath, _options.IncludeColors);
                }

                Console.WriteLine($"SUCCESS: pipeline completed with {current.Vertices.Count} vertices.");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled.");
                return 1;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Pipeline failed: {ex.Message}");
                return 1;
            }
            finally
            {
                manager.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private static string NormalizeToken(string? value)
        {
            return value?.Trim().ToLowerInvariant() ?? string.Empty;
        }

        private static string? NormalizeReconstructionPipeline(string? pipeline)
        {
            var token = NormalizeToken(pipeline);
            if (string.IsNullOrWhiteSpace(token))
                return null;

            return token switch
            {
                "dust3r" => "dust3r",
                "mast3r" => "mast3r",
                "must3r" => "must3r",
                "sfm" or "featurematching" or "feature-matching" => "sfm",
                "none" or "off" => "none",
                _ => token
            };
        }

        private int UnknownSubcommand(string command, string expected)
        {
            Console.Error.WriteLine($"Unknown subcommand for '{command}': {_options.Subcommand ?? "(missing)"}");
            Console.Error.WriteLine($"Expected: {expected}");
            PrintHelp();
            return 1;
        }

        private SceneResult ExecuteReconstructionWorkflow(
            AIModelManager manager,
            string pipeline,
            List<string> images,
            System.Threading.CancellationToken cancellationToken)
        {
            WorkflowStep reconstructionStep = pipeline switch
            {
                "dust3r" => WorkflowStep.Dust3rReconstruction,
                "must3r" => WorkflowStep.Must3rReconstruction,
                "sfm" => WorkflowStep.SfMReconstruction,
                _ => WorkflowStep.Mast3rReconstruction
            };

            var wf = new WorkflowPipeline
            {
                Name = $"Reconstruct ({pipeline})",
                Description = "CLI reconstruction workflow",
                Steps = new List<WorkflowStep>
                {
                    WorkflowStep.LoadImages,
                    reconstructionStep
                }
            };

            return manager.ExecuteWorkflowAsync(
                wf,
                images,
                null,
                (msg, progress) =>
                {
                    Console.WriteLine($"[{wf.Name}] {progress:P0} {msg}");
                    TuiStatusMonitor.Instance.UpdateProgress($"{wf.Name}: {msg}", progress);
                },
                cancellationToken
            ).GetAwaiter().GetResult();
        }

        private static bool TryBuildMergedGeometry(SceneResult? result, out MeshData? mergedMesh, out string reason)
        {
            mergedMesh = null;
            if (result == null)
            {
                reason = "null result";
                return false;
            }

            var validMeshes = result.Meshes
                .Where(m => m != null && m.Vertices.Count > 0)
                .ToList();

            if (validMeshes.Count == 0)
            {
                reason = "no meshes with vertices";
                return false;
            }

            var merged = MergeMeshes(validMeshes);
            if (merged.Vertices.Count < 16)
            {
                reason = $"too few points ({merged.Vertices.Count})";
                return false;
            }

            if (!HasUsableSpatialExtent(merged, out var maxDim))
            {
                reason = $"degenerate bounds (maxDim={maxDim:F8})";
                return false;
            }

            mergedMesh = merged;
            reason = "ok";
            return true;
        }

        private static MeshData MergeMeshes(IEnumerable<MeshData> meshes)
        {
            var merged = new MeshData();
            foreach (var mesh in meshes)
            {
                int baseIndex = merged.Vertices.Count;
                merged.Vertices.AddRange(mesh.Vertices);
                if (mesh.Colors.Count == mesh.Vertices.Count)
                {
                    merged.Colors.AddRange(mesh.Colors);
                }
                else
                {
                    for (int i = 0; i < mesh.Vertices.Count; i++)
                        merged.Colors.Add(new Vector3(1, 1, 1));
                }

                if (mesh.Indices.Count > 0)
                {
                    foreach (var idx in mesh.Indices)
                        merged.Indices.Add(baseIndex + idx);
                }
            }

            return merged;
        }

        private static bool HasUsableSpatialExtent(MeshData mesh, out float maxDimension)
        {
            maxDimension = 0f;
            if (mesh.Vertices.Count == 0)
                return false;

            var min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            var max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            foreach (var v in mesh.Vertices)
            {
                min = Vector3.ComponentMin(min, v);
                max = Vector3.ComponentMax(max, v);
            }

            var size = max - min;
            maxDimension = Math.Max(size.X, Math.Max(size.Y, size.Z));
            if (float.IsNaN(maxDimension) || float.IsInfinity(maxDimension))
                return false;
            return maxDimension > 1e-5f;
        }

        private string ResolveOutputDirectory(string? primaryInputPath)
        {
            if (!string.IsNullOrWhiteSpace(_options.OutputDirectory))
                return Path.GetFullPath(_options.OutputDirectory);

            if (!string.IsNullOrWhiteSpace(_options.OutputPath))
            {
                var outputPath = Path.GetFullPath(_options.OutputPath);
                var directory = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrWhiteSpace(directory))
                    return directory;
            }

            if (!string.IsNullOrWhiteSpace(primaryInputPath))
            {
                var fullInput = Path.GetFullPath(primaryInputPath);
                if (File.Exists(fullInput))
                    return Path.GetDirectoryName(fullInput) ?? Environment.CurrentDirectory;
                if (Directory.Exists(fullInput))
                    return fullInput;
            }

            return Environment.CurrentDirectory;
        }

        private string ResolveOutputBaseName(string? inputPath)
        {
            if (!string.IsNullOrWhiteSpace(_options.OutputPath))
            {
                var fileName = Path.GetFileNameWithoutExtension(_options.OutputPath);
                if (!string.IsNullOrWhiteSpace(fileName))
                    return SanitizeFileName(fileName);
            }

            if (!string.IsNullOrWhiteSpace(inputPath))
                return SanitizeFileName(Path.GetFileNameWithoutExtension(inputPath));

            return "output";
        }

        private string? ResolveInputFilePath(IReadOnlyCollection<string> allowedExtensions)
        {
            IEnumerable<string> candidates = _options.InputPaths
                .Concat(string.IsNullOrWhiteSpace(_options.InputPath) ? Array.Empty<string>() : new[] { _options.InputPath! })
                .Concat(_options.ExtraArgs)
                .Where(p => !string.IsNullOrWhiteSpace(p))
                .Select(Path.GetFullPath);

            foreach (var candidate in candidates)
            {
                if (!File.Exists(candidate))
                    continue;

                var ext = Path.GetExtension(candidate).ToLowerInvariant();
                if (allowedExtensions.Contains(ext, StringComparer.OrdinalIgnoreCase))
                    return candidate;
            }

            return null;
        }

        private List<string> ResolveRefiners()
        {
            var selected = new List<string>();
            void Add(string? token)
            {
                var normalized = NormalizeToken(token);
                if (normalized is "triposf" or "gaussiansdf" or "deepmeshprior" or "nerf")
                {
                    if (!selected.Contains(normalized, StringComparer.OrdinalIgnoreCase))
                        selected.Add(normalized);
                }
            }

            foreach (var refiner in _options.Refiners)
                Add(refiner);

            if (selected.Count == 0 && string.Equals(NormalizeToken(_options.Preset), "quality", StringComparison.Ordinal))
            {
                Add("triposf");
                Add("gaussiansdf");
                Add("deepmeshprior");
            }

            return selected;
        }

        private MeshData? ApplyRefiner(
            AIModelManager manager,
            MeshData inputMesh,
            string refiner,
            List<string> imageInputs,
            System.Threading.CancellationToken cancellationToken)
        {
            var normalized = NormalizeToken(refiner);
            if (normalized == "triposf")
            {
                return manager.TripoSF?.RefineMesh(inputMesh, cancellationToken);
            }

            if (normalized == "gaussiansdf")
            {
                return RunSingleStepRefiner(manager, WorkflowStep.GaussianSDFRefinement, "GaussianSDF refinement", inputMesh, imageInputs, cancellationToken);
            }

            if (normalized == "deepmeshprior")
            {
                var settings = IniSettings.Instance;
                int oldIterations = settings.DeepMeshPriorIterations;
                float oldLr = settings.DeepMeshPriorLearningRate;
                float oldLap = settings.DeepMeshPriorLaplacianWeight;
                try
                {
                    if (_options.DeepMeshPriorIterations.HasValue)
                        settings.DeepMeshPriorIterations = Math.Clamp(_options.DeepMeshPriorIterations.Value, 100, 5000);
                    if (_options.DeepMeshPriorLearningRate.HasValue)
                        settings.DeepMeshPriorLearningRate = Math.Clamp(_options.DeepMeshPriorLearningRate.Value, 0.0001f, 0.1f);
                    if (_options.DeepMeshPriorLaplacianWeight.HasValue)
                        settings.DeepMeshPriorLaplacianWeight = Math.Clamp(_options.DeepMeshPriorLaplacianWeight.Value, 0.0f, 10.0f);

                    return RunSingleStepRefiner(manager, WorkflowStep.DeepMeshPriorRefinement, "DeepMeshPrior refinement", inputMesh, imageInputs, cancellationToken);
                }
                finally
                {
                    settings.DeepMeshPriorIterations = oldIterations;
                    settings.DeepMeshPriorLearningRate = oldLr;
                    settings.DeepMeshPriorLaplacianWeight = oldLap;
                }
            }

            if (normalized == "nerf")
            {
                var settings = IniSettings.Instance;
                int oldIterations = settings.NeRFIterations;
                try
                {
                    if (_options.NerfIterations.HasValue)
                        settings.NeRFIterations = Math.Max(1, _options.NerfIterations.Value);
                    return RunSingleStepRefiner(manager, WorkflowStep.NeRFRefinement, "NeRF refinement", inputMesh, imageInputs, cancellationToken);
                }
                finally
                {
                    settings.NeRFIterations = oldIterations;
                }
            }

            return null;
        }

        private MeshData? RunSingleStepRefiner(
            AIModelManager manager,
            WorkflowStep step,
            string pipelineName,
            MeshData inputMesh,
            List<string> imageInputs,
            System.Threading.CancellationToken cancellationToken)
        {
            var pipeline = new WorkflowPipeline
            {
                Name = pipelineName,
                Steps = new List<WorkflowStep> { step }
            };
            var scene = new SceneResult
            {
                Meshes = new List<MeshData> { inputMesh.Clone() }
            };

            var result = manager.ExecuteWorkflowAsync(
                pipeline,
                imageInputs,
                scene,
                (msg, progress) =>
                {
                    Console.WriteLine($"[{pipeline.Name}] {progress:P0} {msg}");
                    TuiStatusMonitor.Instance.UpdateProgress($"{pipeline.Name}: {msg}", progress);
                },
                cancellationToken
            ).GetAwaiter().GetResult();

            return result.Meshes.FirstOrDefault(m => m.Vertices.Count > 0);
        }

        private List<string> ResolveMeshFormats(params string[] defaults)
        {
            var selected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var format in _options.MeshFormats)
            {
                var normalized = NormalizeMeshFormat(format);
                if (normalized != null) selected.Add(normalized);
            }

            foreach (var format in _options.GenericFormats)
            {
                var normalized = NormalizeMeshFormat(format);
                if (normalized != null) selected.Add(normalized);
            }

            if (selected.Count == 0)
            {
                foreach (var def in defaults)
                {
                    var normalized = NormalizeMeshFormat(def);
                    if (normalized != null) selected.Add(normalized);
                }
            }

            return selected.ToList();
        }

        private List<string> ResolvePointCloudFormats(params string[] defaults)
        {
            var selected = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var format in _options.PointCloudFormats)
            {
                var normalized = NormalizePointCloudFormat(format);
                if (normalized != null) selected.Add(normalized);
            }

            foreach (var format in _options.GenericFormats)
            {
                var normalized = NormalizePointCloudFormat(format);
                if (normalized != null) selected.Add(normalized);
            }

            if (selected.Count == 0)
            {
                foreach (var def in defaults)
                {
                    var normalized = NormalizePointCloudFormat(def);
                    if (normalized != null) selected.Add(normalized);
                }
            }

            return selected.ToList();
        }

        private static string? NormalizeMeshFormat(string? format)
        {
            return NormalizeToken(format) switch
            {
                "obj" => "obj",
                "gltf" => "gltf",
                "glb" => "glb",
                "ply" => "ply",
                "fbx" => "fbx",
                _ => null
            };
        }

        private static string? NormalizePointCloudFormat(string? format)
        {
            return NormalizeToken(format) switch
            {
                "ply" => "ply",
                "xyz" => "xyz",
                _ => null
            };
        }

        private void ExportMeshToFormats(
            MeshData mesh,
            string outputDir,
            string baseName,
            List<string> formats,
            string? explicitOutputPath = null)
        {
            if (!string.IsNullOrWhiteSpace(explicitOutputPath) && formats.Count == 1)
            {
                var singleFormat = formats[0];
                var target = Path.GetFullPath(explicitOutputPath);
                if (string.IsNullOrWhiteSpace(Path.GetExtension(target)))
                    target = $"{target}.{singleFormat}";
                else
                    target = Path.ChangeExtension(target, singleFormat);

                target = EnsureWritablePath(target);
                SaveMesh(mesh, target);
                Console.WriteLine($"Saved mesh: {target}");
                return;
            }

            foreach (var format in formats)
            {
                var outPath = Path.Combine(outputDir, $"{baseName}.{format}");
                outPath = EnsureWritablePath(outPath);
                SaveMesh(mesh, outPath);
                Console.WriteLine($"Saved mesh: {outPath}");
            }
        }

        private void SaveMesh(MeshData mesh, string path)
        {
            var fullPath = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Environment.CurrentDirectory);
            MeshExporter.Save(fullPath, mesh);
        }

        private void SavePointCloud(MeshData mesh, string path, bool includeColors)
        {
            var fullPath = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(fullPath) ?? Environment.CurrentDirectory);
            var ext = Path.GetExtension(fullPath).ToLowerInvariant();
            var format = ext == ".xyz" ? PointCloudExporter.ExportFormat.XYZ : PointCloudExporter.ExportFormat.PLY;
            PointCloudExporter.Export(fullPath, mesh, format, includeColors);
        }

        private string EnsureWritablePath(string desiredPath)
        {
            var fullPath = Path.GetFullPath(desiredPath);
            if (_options.Overwrite || !File.Exists(fullPath))
                return fullPath;

            var directory = Path.GetDirectoryName(fullPath) ?? Environment.CurrentDirectory;
            var fileName = Path.GetFileNameWithoutExtension(fullPath);
            var ext = Path.GetExtension(fullPath);
            int index = 2;
            string candidate;
            do
            {
                candidate = Path.Combine(directory, $"{fileName}_{index}{ext}");
                index++;
            } while (File.Exists(candidate));

            return candidate;
        }

        private static MeshData LoadMeshAnyFormat(string inputPath)
        {
            var ext = Path.GetExtension(inputPath).ToLowerInvariant();
            if (ext is ".obj" or ".ply" or ".stl")
                return MeshImporter.Load(inputPath);

            throw new NotSupportedException($"Mesh import format not supported by CLI yet: {ext}");
        }

        private void WriteRunManifest(
            List<string> inputImages,
            string pipeline,
            string? fallbackPipeline,
            string status,
            string message,
            int vertexCount,
            int triangleCount,
            List<string>? outputs)
        {
            try
            {
                var outputDir = ResolveOutputDirectory(inputImages.FirstOrDefault());
                Directory.CreateDirectory(outputDir);
                var manifestPath = Path.Combine(outputDir, "run.json");
                var manifest = new
                {
                    timestampUtc = DateTime.UtcNow,
                    status,
                    message,
                    pipeline,
                    fallbackPipeline,
                    inputs = inputImages,
                    vertexCount,
                    triangleCount,
                    outputs = outputs ?? new List<string>()
                };

                var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(manifestPath, json);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: could not write run manifest: {ex.Message}");
            }
        }

        private static IEnumerable<SceneObjectDTO> EnumerateSceneObjects(IEnumerable<SceneObjectDTO> roots)
        {
            foreach (var root in roots)
            {
                yield return root;
                foreach (var child in EnumerateSceneObjects(root.Children))
                    yield return child;
            }
        }

        private static MeshData ConvertMeshDto(MeshObjectDTO dto)
        {
            var mesh = new MeshData
            {
                Vertices = UnflattenVector3(dto.MeshData.Vertices),
                Normals = UnflattenVector3(dto.MeshData.Normals),
                Colors = UnflattenVector3(dto.MeshData.Colors),
                UVs = UnflattenVector2(dto.MeshData.UVs),
                Indices = dto.MeshData.Indices != null ? new List<int>(dto.MeshData.Indices) : new List<int>()
            };
            return mesh;
        }

        private static MeshData ConvertPointCloudDto(PointCloudObjectDTO dto)
        {
            var mesh = new MeshData
            {
                Vertices = UnflattenVector3(dto.Points),
                Colors = UnflattenVector3(dto.Colors)
            };
            return mesh;
        }

        private static List<Vector3> UnflattenVector3(List<float> values)
        {
            var result = new List<Vector3>(values.Count / 3);
            for (int i = 0; i + 2 < values.Count; i += 3)
            {
                result.Add(new Vector3(values[i], values[i + 1], values[i + 2]));
            }
            return result;
        }

        private static List<Vector2> UnflattenVector2(List<float> values)
        {
            var result = new List<Vector2>(values.Count / 2);
            for (int i = 0; i + 1 < values.Count; i += 2)
            {
                result.Add(new Vector2(values[i], values[i + 1]));
            }
            return result;
        }

        private static string SanitizeFileName(string? name)
        {
            var safe = string.IsNullOrWhiteSpace(name) ? "object" : name.Trim();
            foreach (var c in Path.GetInvalidFileNameChars())
            {
                safe = safe.Replace(c, '_');
            }

            return string.IsNullOrWhiteSpace(safe) ? "object" : safe;
        }

        private static string GetUniqueName(string baseName, HashSet<string> usedNames)
        {
            var candidate = baseName;
            int index = 2;
            while (!usedNames.Add(candidate))
            {
                candidate = $"{baseName}_{index}";
                index++;
            }
            return candidate;
        }

        private static int RunPythonReextract()
        {
            Console.WriteLine("Re-extracting Python environment...");
            bool ok = PythonService.Instance.ReextractPythonEnvironment();
            Console.WriteLine(ok
                ? "Python environment re-extracted successfully."
                : "Python environment re-extraction failed. Check logs for details.");
            return ok ? 0 : 1;
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

        private int RunPointCloudToMeshTripoSF()
        {
            if (string.IsNullOrWhiteSpace(_options.InputPath) || !File.Exists(_options.InputPath))
            {
                Console.Error.WriteLine("Point cloud input not found. Provide --input <file.ply|file.xyz>.");
                return 1;
            }

            var inputPath = Path.GetFullPath(_options.InputPath);
            Console.WriteLine($"Loading point cloud: {inputPath}");

            PointCloudObject pc;
            try
            {
                pc = PointCloudImporter.Load(inputPath);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed to load point cloud: {ex.Message}");
                return 1;
            }

            if (pc.Points.Count == 0)
            {
                Console.Error.WriteLine("Point cloud is empty. Aborting.");
                return 1;
            }

            using var cancellationSource = new System.Threading.CancellationTokenSource();
            TuiStatusMonitor.Instance.SetCancellationTokenSource(cancellationSource);
            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                cancellationSource.Cancel();
                Console.WriteLine("Cancellation requested. Stopping after current step...");
            };

            try
            {
                int targetRes = _options.VoxelResolution.HasValue ? Math.Clamp(_options.VoxelResolution.Value, 32, 512) : 256;
                Console.WriteLine($"Voxelizing point cloud (targetRes={targetRes})...");
                var voxelized = VoxelizePointCloud(pc.Points, targetRes);
                if (voxelized.grid == null)
                {
                    Console.Error.WriteLine("Voxelization failed.");
                    return 1;
                }

                Console.WriteLine("Running marching cubes...");
                var mesher = new MarchingCubesMesher();
                Action<string, float> cliProgress2 = (msg, p) =>
                {
                    TuiStatusMonitor.Instance.UpdateProgress(msg, p);
                };
                var baseMesh = mesher.GenerateMesh(voxelized.grid, voxelized.origin, voxelized.voxelSize, 0.5f, cliProgress2);
                Console.WriteLine($"Marching cubes: {baseMesh.Vertices.Count} vertices, {baseMesh.Indices.Count / 3} triangles");

                if (baseMesh.Vertices.Count == 0 || baseMesh.Indices.Count == 0)
                {
                    Console.Error.WriteLine("Marching cubes produced no geometry.");
                    return 1;
                }

                var baseOutputPath = GetDefaultOutputPath(inputPath, "_mc.ply");
                if (!string.IsNullOrWhiteSpace(_options.OutputPath))
                {
                    var dir = Path.GetDirectoryName(_options.OutputPath) ?? Environment.CurrentDirectory;
                    baseOutputPath = Path.Combine(dir, Path.GetFileNameWithoutExtension(_options.OutputPath) + "_mc.ply");
                }
                Console.WriteLine($"Saving base mesh to: {baseOutputPath}");
                MeshExporter.Save(baseOutputPath, baseMesh);

                Console.WriteLine("Running TripoSF refinement...");
                using var tripo = new TripoSFInference();
                var refined = tripo.RefineMesh(baseMesh, cancellationSource.Token);

                if (refined == null || refined.Vertices.Count == 0 || refined.Indices.Count == 0)
                {
                    Console.Error.WriteLine("TripoSF produced no geometry.");
                    return 1;
                }

                var outputPath = string.IsNullOrWhiteSpace(_options.OutputPath)
                    ? GetDefaultOutputPath(inputPath, "_tripoSF.ply")
                    : _options.OutputPath;

                Console.WriteLine($"Saving refined mesh to: {outputPath}");
                MeshExporter.Save(outputPath, refined);
                Console.WriteLine("Done.");
                return 0;
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("Cancelled.");
                return 1;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Failed: {ex.Message}");
                return 1;
            }
            finally
            {
                AIModelManager.Instance.UnloadAllModels();
                TuiStatusMonitor.Instance.SetCancellationTokenSource(null);
            }
        }

        private static string GetDefaultOutputPath(string inputPath, string suffix)
        {
            var dir = Path.GetDirectoryName(inputPath) ?? Environment.CurrentDirectory;
            var name = Path.GetFileNameWithoutExtension(inputPath);
            return Path.Combine(dir, $"{name}{suffix}");
        }

        private static (float[,,]? grid, OpenTK.Mathematics.Vector3 origin, float voxelSize) VoxelizePointCloud(
            IList<OpenTK.Mathematics.Vector3> points,
            int targetRes)
        {
            var result = VoxelizationUtils.Voxelize(points, targetRes);
            return (result.grid, result.origin, result.voxelSize);
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
                    // WorkflowPipeline.ImageToLGM,
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
                    /*
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
                    */

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

                    /*
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
                    */

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
            bool hasExplicitInput = _options.InputPaths.Count > 0 || !string.IsNullOrWhiteSpace(_options.InputPath);

            foreach (var input in _options.InputPaths)
                images.AddRange(ExpandImagePaths(input));

            if (!string.IsNullOrWhiteSpace(_options.InputPath))
                images.AddRange(ExpandImagePaths(_options.InputPath!));

            foreach (var arg in _options.ExtraArgs)
            {
                if (File.Exists(arg) && IsImageFile(arg))
                {
                    images.Add(Path.GetFullPath(arg));
                }
                else if (Directory.Exists(arg))
                {
                    images.AddRange(ExpandImagePaths(arg));
                }

                if (File.Exists(arg) || Directory.Exists(arg))
                    hasExplicitInput = true;
            }

            if (images.Count == 0 && !hasExplicitInput)
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
            Console.WriteLine("Deep3DStudio CLI");
            Console.WriteLine("Usage (new):");
            Console.WriteLine("  --cli reconstruct pointcloud --input <img|dir> [--input <img2>] [--pipeline mast3r|dust3r|must3r|sfm] [--fallback dust3r|none]");
            Console.WriteLine("  --cli mesh from-pointcloud --input <file.ply|file.xyz> [--voxel-res N] [--iso-level F] [--refiners triposf,gaussiansdf,deepmeshprior,nerf]");
            Console.WriteLine("  --cli refine mesh --input <file.obj|file.ply|file.stl> --refiners triposf,gaussiansdf,deepmeshprior,nerf");
            Console.WriteLine("  --cli export mesh --input <mesh> --mesh-formats obj,gltf,glb,ply,fbx");
            Console.WriteLine("  --cli export pointcloud --input <mesh|pointcloud> --pointcloud-formats ply,xyz");
            Console.WriteLine("  --cli project export-all --project <project.d3d> [--mesh-formats ...] [--pointcloud-formats ...]");
            Console.WriteLine("  --cli pipeline run --input <img|dir> [--pipeline mast3r|dust3r|must3r|sfm] [--refiners ...]");
            Console.WriteLine();
            Console.WriteLine("Usage (legacy still supported):");
            Console.WriteLine("  --cli --command test-all [--input <file|dir>] [--verbose]");
            Console.WriteLine("  --cli --command nerf [--input <file|dir>] [--nerf-iterations N]");
            Console.WriteLine("  --cli --command pc-mesh-triposf --input <file.ply|file.xyz> [--output <file.ply>] [--voxel-res N]");
            Console.WriteLine("  --cli --command reextract-python");
            Console.WriteLine();
            Console.WriteLine("Global options:");
            Console.WriteLine("  --cli, --headless      Run without GUI");
            Console.WriteLine("  --command, --mode      Legacy command selector");
            Console.WriteLine("  --input                Input path (repeatable)");
            Console.WriteLine("  --output               Output file path");
            Console.WriteLine("  --output-dir           Output directory");
            Console.WriteLine("  --pipeline             Reconstruction backend");
            Console.WriteLine("  --fallback             Fallback backend (use 'none' to disable)");
            Console.WriteLine("  --refiners             Comma-separated refiners");
            Console.WriteLine("  --mesh-formats         obj,gltf,glb,ply,fbx");
            Console.WriteLine("  --pointcloud-formats   ply,xyz");
            Console.WriteLine("  --export               Generic formats (auto-routed)");
            Console.WriteLine("  --voxel-res            Voxel resolution for meshing");
            Console.WriteLine("  --iso-level            Marching cubes iso level");
            Console.WriteLine("  --nerf-iterations      Override NeRF iterations");
            Console.WriteLine("  --deepmeshprior-iterations / --dmp-iterations");
            Console.WriteLine("  --deepmeshprior-learning-rate / --dmp-learning-rate");
            Console.WriteLine("  --deepmeshprior-laplacian-weight / --dmp-laplacian-weight");
            Console.WriteLine("  --include-colors       true|false (default true)");
            Console.WriteLine("  --include-hidden       Include hidden project objects");
            Console.WriteLine("  --overwrite            Overwrite existing files");
            Console.WriteLine("  --verbose, -v          Verbose logging");
            Console.WriteLine("  --help, -h, -?         Show this help");
        }
    }
}
