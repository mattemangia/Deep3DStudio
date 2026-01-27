using System;
using System.Collections.Generic;

namespace Deep3DStudio.CLI
{
    public sealed class CommandLineOptions
    {
        public bool IsCLIMode { get; private set; }
        public bool Verbose { get; private set; }
        public bool ShowHelp { get; private set; }
        public string? Command { get; private set; }
        public string? InputPath { get; private set; }
        public string? OutputPath { get; private set; }
        public string? ModelName { get; private set; }
        public int? NerfIterations { get; private set; }
        public int? VoxelResolution { get; private set; }
        public IReadOnlyList<string> ExtraArgs => _extraArgs;

        private readonly List<string> _extraArgs = new();

        public static CommandLineOptions Parse(string[] args)
        {
            var options = new CommandLineOptions();

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                switch (arg)
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
                        if (i + 1 < args.Length)
                        {
                            options.Command = args[++i];
                        }
                        break;
                    case "--input":
                        if (i + 1 < args.Length)
                        {
                            options.InputPath = args[++i];
                        }
                        break;
                    case "--output":
                        if (i + 1 < args.Length)
                        {
                            options.OutputPath = args[++i];
                        }
                        break;
                    case "--model":
                        if (i + 1 < args.Length)
                        {
                            options.ModelName = args[++i];
                        }
                        break;
                    case "--nerf-iterations":
                        if (i + 1 < args.Length && int.TryParse(args[++i], out var nerfIterations))
                        {
                            options.NerfIterations = nerfIterations;
                        }
                        break;
                    case "--voxel-res":
                        if (i + 1 < args.Length && int.TryParse(args[++i], out var voxelRes))
                        {
                            options.VoxelResolution = voxelRes;
                        }
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
                        else
                        {
                            options._extraArgs.Add(arg);
                        }
                        break;
                }
            }

            return options;
        }
    }
}
