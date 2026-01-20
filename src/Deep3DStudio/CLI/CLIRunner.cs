using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;

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
            manager.ModelLoadProgress += (stage, progress, message) =>
            {
                Console.WriteLine($"[ModelLoad] {stage} {progress:P0} {message}");
            };

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
            MeshData? firstMesh = null;

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
                    (msg, progress) => Console.WriteLine($"[{pipeline.Name}] {progress:P0} {msg}")
                ).GetAwaiter().GetResult();

                var ok = result != null &&
                         result.Meshes.Count > 0 &&
                         result.Meshes.Any(m => m.Vertices.Count > 0);

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

            if (firstMesh != null)
            {
                Console.WriteLine("=== Running TripoSF refinement ===");
                try
                {
                    var refined = manager.TripoSF?.RefineMesh(firstMesh);
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

                Console.WriteLine("=== Running UniRig auto-rig ===");
                try
                {
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
                Console.WriteLine("Skipping TripoSF/UniRig: no mesh with faces produced by earlier models.");
                allOk = false;
            }

            manager.UnloadAllModels();
            return allOk ? 0 : 1;
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

        private static void PrintHelp()
        {
            Console.WriteLine("Deep3DStudio CLI (placeholder)");
            Console.WriteLine("Usage:");
            Console.WriteLine("  --cli --command test-all [--input <file|dir>] [--verbose]");
            Console.WriteLine("Options:");
            Console.WriteLine("  --cli, --headless   Run without GUI");
            Console.WriteLine("  --command, --mode   CLI command to run");
            Console.WriteLine("  --model             Model name to use");
            Console.WriteLine("  --input             Input path");
            Console.WriteLine("  --output            Output path");
            Console.WriteLine("  --verbose, -v       Verbose logging");
            Console.WriteLine("  --help, -h, -?      Show this help");
        }
    }
}
