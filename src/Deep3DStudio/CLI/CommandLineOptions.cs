using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Deep3DStudio.CLI
{
    public sealed class CommandLineOptions
    {
        public bool IsCLIMode { get; private set; }
        public bool Verbose { get; private set; }
        public bool ShowHelp { get; private set; }
        public string? Command { get; private set; }
        public string? Subcommand { get; private set; }
        public string? InputPath { get; private set; }
        public string? OutputPath { get; private set; }
        public string? OutputDirectory { get; private set; }
        public string? ProjectPath { get; private set; }
        public string? ModelName { get; private set; }
        public string? Pipeline { get; private set; }
        public string? FallbackPipeline { get; private set; }
        public string? Preset { get; private set; }
        public int? NerfIterations { get; private set; }
        public int? VoxelResolution { get; private set; }
        public float? IsoLevel { get; private set; }
        public int? SmoothingIterations { get; private set; }
        public int? DecimationTargetFaces { get; private set; }
        public int? DeepMeshPriorIterations { get; private set; }
        public float? DeepMeshPriorLearningRate { get; private set; }
        public float? DeepMeshPriorLaplacianWeight { get; private set; }
        public bool IncludeColors { get; private set; } = true;
        public bool IncludeHidden { get; private set; }
        public bool Overwrite { get; private set; }
        public IReadOnlyList<string> InputPaths => _inputPaths;
        public IReadOnlyList<string> Refiners => _refiners;
        public IReadOnlyList<string> MeshFormats => _meshFormats;
        public IReadOnlyList<string> PointCloudFormats => _pointCloudFormats;
        public IReadOnlyList<string> GenericFormats => _genericFormats;
        public IReadOnlyList<string> ExtraArgs => _extraArgs;

        private readonly List<string> _extraArgs = new();
        private readonly List<string> _inputPaths = new();
        private readonly List<string> _refiners = new();
        private readonly List<string> _meshFormats = new();
        private readonly List<string> _pointCloudFormats = new();
        private readonly List<string> _genericFormats = new();

        public static CommandLineOptions Parse(string[] args)
        {
            var options = new CommandLineOptions();

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                var key = arg;
                string? inlineValue = null;
                var equalsIdx = arg.IndexOf('=');
                if (arg.StartsWith("--", StringComparison.Ordinal) && equalsIdx > 2)
                {
                    key = arg.Substring(0, equalsIdx);
                    inlineValue = arg.Substring(equalsIdx + 1);
                }

                switch (key)
                {
                    case "--cli":
                    case "--headless":
                        options.IsCLIMode = true;
                        break;
                    case "--verbose":
                    case "-v":
                        options.Verbose = true;
                        break;
                    case "--help":
                    case "-h":
                    case "-?":
                        options.ShowHelp = true;
                        options.IsCLIMode = true;
                        break;
                    case "--command":
                    case "--mode":
                        if (TryGetValue(args, ref i, inlineValue, out var command))
                        {
                            options.Command = command;
                            options.IsCLIMode = true;
                        }
                        break;
                    case "--subcommand":
                        if (TryGetValue(args, ref i, inlineValue, out var subcommand))
                            options.Subcommand = subcommand;
                        break;
                    case "--input":
                        if (TryGetValue(args, ref i, inlineValue, out var input))
                        {
                            options._inputPaths.Add(input);
                            options.InputPath ??= input;
                        }
                        break;
                    case "--output":
                        if (TryGetValue(args, ref i, inlineValue, out var output))
                        {
                            options.OutputPath = output;
                        }
                        break;
                    case "--output-dir":
                    case "--outdir":
                        if (TryGetValue(args, ref i, inlineValue, out var outputDir))
                            options.OutputDirectory = outputDir;
                        break;
                    case "--project":
                        if (TryGetValue(args, ref i, inlineValue, out var projectPath))
                            options.ProjectPath = projectPath;
                        break;
                    case "--model":
                        if (TryGetValue(args, ref i, inlineValue, out var modelName))
                        {
                            options.ModelName = modelName;
                        }
                        break;
                    case "--pipeline":
                        if (TryGetValue(args, ref i, inlineValue, out var pipeline))
                            options.Pipeline = pipeline;
                        break;
                    case "--fallback":
                        if (TryGetValue(args, ref i, inlineValue, out var fallback))
                            options.FallbackPipeline = fallback;
                        break;
                    case "--preset":
                        if (TryGetValue(args, ref i, inlineValue, out var preset))
                            options.Preset = preset;
                        break;
                    case "--nerf-iterations":
                        if (TryGetInt(args, ref i, inlineValue, out var nerfIterations))
                        {
                            options.NerfIterations = nerfIterations;
                        }
                        break;
                    case "--voxel-res":
                        if (TryGetInt(args, ref i, inlineValue, out var voxelRes))
                        {
                            options.VoxelResolution = voxelRes;
                        }
                        break;
                    case "--iso-level":
                        if (TryGetFloat(args, ref i, inlineValue, out var isoLevel))
                            options.IsoLevel = isoLevel;
                        break;
                    case "--mesh-smoothing-iterations":
                    case "--smoothing-iterations":
                        if (TryGetInt(args, ref i, inlineValue, out var smoothing))
                            options.SmoothingIterations = smoothing;
                        break;
                    case "--mesh-decimation-target-faces":
                    case "--decimation-target-faces":
                        if (TryGetInt(args, ref i, inlineValue, out var faces))
                            options.DecimationTargetFaces = faces;
                        break;
                    case "--dmp-iterations":
                    case "--deepmeshprior-iterations":
                        if (TryGetInt(args, ref i, inlineValue, out var dmpIterations))
                            options.DeepMeshPriorIterations = dmpIterations;
                        break;
                    case "--dmp-learning-rate":
                    case "--deepmeshprior-learning-rate":
                        if (TryGetFloat(args, ref i, inlineValue, out var dmpLr))
                            options.DeepMeshPriorLearningRate = dmpLr;
                        break;
                    case "--dmp-laplacian-weight":
                    case "--deepmeshprior-laplacian-weight":
                        if (TryGetFloat(args, ref i, inlineValue, out var dmpLap))
                            options.DeepMeshPriorLaplacianWeight = dmpLap;
                        break;
                    case "--refiners":
                        if (TryGetValue(args, ref i, inlineValue, out var refiners))
                            AddCsvValues(options._refiners, refiners);
                        break;
                    case "--mesh-formats":
                    case "--export-mesh":
                        if (TryGetValue(args, ref i, inlineValue, out var meshFormats))
                            AddCsvValues(options._meshFormats, meshFormats);
                        break;
                    case "--pointcloud-formats":
                    case "--export-pointcloud":
                        if (TryGetValue(args, ref i, inlineValue, out var pointCloudFormats))
                            AddCsvValues(options._pointCloudFormats, pointCloudFormats);
                        break;
                    case "--formats":
                    case "--export":
                        if (TryGetValue(args, ref i, inlineValue, out var genericFormats))
                            AddCsvValues(options._genericFormats, genericFormats);
                        break;
                    case "--include-colors":
                        options.IncludeColors = ParseOptionalBool(args, ref i, inlineValue, true);
                        break;
                    case "--include-hidden":
                        options.IncludeHidden = ParseOptionalBool(args, ref i, inlineValue, true);
                        break;
                    case "--overwrite":
                        options.Overwrite = ParseOptionalBool(args, ref i, inlineValue, true);
                        break;
                    default:
                        if (arg.StartsWith("-", StringComparison.Ordinal))
                        {
                            options._extraArgs.Add(arg);
                        }
                        else if (string.IsNullOrWhiteSpace(options.Command))
                        {
                            options.Command = arg;
                        }
                        else if (string.IsNullOrWhiteSpace(options.Subcommand))
                        {
                            options.Subcommand = arg;
                        }
                        else
                        {
                            options._extraArgs.Add(arg);
                        }
                        break;
                }
            }

            if (string.IsNullOrWhiteSpace(options.FallbackPipeline))
                options.FallbackPipeline = "dust3r";

            return options;
        }

        private static bool TryGetValue(string[] args, ref int index, string? inlineValue, out string value)
        {
            if (!string.IsNullOrWhiteSpace(inlineValue))
            {
                value = inlineValue;
                return true;
            }

            if (index + 1 < args.Length)
            {
                value = args[++index];
                return true;
            }

            value = string.Empty;
            return false;
        }

        private static bool TryGetInt(string[] args, ref int index, string? inlineValue, out int value)
        {
            value = 0;
            return TryGetValue(args, ref index, inlineValue, out var str) &&
                   int.TryParse(str, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryGetFloat(string[] args, ref int index, string? inlineValue, out float value)
        {
            value = 0f;
            return TryGetValue(args, ref index, inlineValue, out var str) &&
                   float.TryParse(str, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out value);
        }

        private static bool ParseOptionalBool(string[] args, ref int index, string? inlineValue, bool defaultIfMissing)
        {
            if (!string.IsNullOrWhiteSpace(inlineValue))
            {
                if (bool.TryParse(inlineValue, out var parsed))
                    return parsed;
                return defaultIfMissing;
            }

            if (index + 1 < args.Length && !args[index + 1].StartsWith("-", StringComparison.Ordinal))
            {
                var next = args[++index];
                if (bool.TryParse(next, out var parsed))
                    return parsed;
                return defaultIfMissing;
            }

            return defaultIfMissing;
        }

        private static void AddCsvValues(List<string> target, string csv)
        {
            foreach (var token in csv.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries))
            {
                var value = token.Trim().ToLowerInvariant();
                if (!string.IsNullOrWhiteSpace(value) && !target.Contains(value, StringComparer.OrdinalIgnoreCase))
                    target.Add(value);
            }
        }
    }
}
