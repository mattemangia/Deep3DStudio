using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.Common.Input;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using ImGuiNET;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using Deep3DStudio.CLI;
using System.Diagnostics;
using System.Runtime.InteropServices;
using SkiaSharp;
using Deep3DStudio.Model;

namespace Deep3DStudio
{
    class Program
    {
        [STAThread]
        public static int Main(string[] args)
        {
            // Initialize logging first - clears log file on each run
            Logger.Initialize();
            Logger.Info("Application starting...");
            Logger.Info($"Arguments: {string.Join(", ", args)}");

            // Parse command-line options
            var cliOptions = CommandLineOptions.Parse(args);

            // If CLI mode is detected, run in CLI mode without GUI
            if (cliOptions.IsCLIMode)
            {
                Console.WriteLine(ConsoleLogo.GenerateAsciiLogo());
                Logger.Info("CLI mode detected");
                return RunCLI(cliOptions);
            }
            else
            {
                // Start TUI Status Monitor for GUI mode
                TuiStatusMonitor.Instance.Start();

                // Hook Python extraction progress to TUI
                PythonService.Instance.OnExtractionProgress += (msg, prog) => {
                    TuiStatusMonitor.Instance.UpdateProgress(msg, prog);
                };

                // Hook AI Model Manager progress to TUI
                var aiManager = Model.AIModels.AIModelManager.Instance;
                aiManager.ProgressUpdated += (msg, prog) => {
                     TuiStatusMonitor.Instance.UpdateProgress(msg, prog);
                };
                aiManager.ModelLoadProgress += (stage, prog, msg) => {
                     TuiStatusMonitor.Instance.UpdateProgress($"{stage}: {msg}", prog);
                };
            }

            // Set up global exception handlers for crash logging
            AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
            {
                Logger.Error("=== UNHANDLED EXCEPTION ===");
                if (e.ExceptionObject is Exception ex)
                {
                    Logger.Exception(ex, "Fatal unhandled exception");
                }
                else
                {
                    Logger.Error($"Non-exception unhandled error: {e.ExceptionObject}");
                }
                Logger.Error($"IsTerminating: {e.IsTerminating}");
            };

            // Force legacy OpenGL support for ThreeDView
            Environment.SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3COMPAT");
            Logger.Debug("Set MESA_GL_VERSION_OVERRIDE=3.3COMPAT");

            var nativeWindowSettings = new NativeWindowSettings()
            {
                Size = new Vector2i(1600, 900),
                Title = "Deep3DStudio (Cross-Platform / ImGui)",
                Flags = ContextFlags.Default,
                Icon = LoadWindowIcon()
            };

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // macOS only supports legacy OpenGL up to 2.1.
                // Requesting 3.3 Compat causes a crash.
                nativeWindowSettings.APIVersion = new Version(2, 1);
                nativeWindowSettings.Profile = ContextProfile.Any;
                Logger.Info("Platform: macOS - using OpenGL 2.1");
            }
            else
            {
                // Windows/Linux support 3.3 Compatibility Profile
                nativeWindowSettings.APIVersion = new Version(3, 3);
                nativeWindowSettings.Profile = ContextProfile.Compatability;
                Logger.Info($"Platform: {(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Windows" : "Linux")} - using OpenGL 3.3 Compat");
            }

            Logger.Info("Creating MainWindow...");
            try
            {
                using (var window = new MainWindow(GameWindowSettings.Default, nativeWindowSettings))
                {
                    Logger.Info("MainWindow created successfully, starting run loop");
                    window.Run();
                    Logger.Info("MainWindow run loop ended normally");
                }
            }
            catch (Exception ex)
            {
                Logger.Exception(ex, "MainWindow crashed");
                throw;
            }
            Logger.Info("Application exiting normally");
            return 0;
        }

        /// <summary>
        /// Run in CLI mode for command-line inference testing.
        /// </summary>
        private static int RunCLI(CommandLineOptions options)
        {
            try
            {
                var runner = new CLIRunner(options);
                return runner.Run();
            }
            catch (Exception ex)
            {
                Logger.Exception(ex, "CLI mode crashed");
                Console.Error.WriteLine($"Fatal error: {ex.Message}");
                if (options.Verbose)
                {
                    Console.Error.WriteLine(ex.StackTrace);
                }
                return 1;
            }
        }

        /// <summary>
        /// Loads the application window icon from embedded resource.
        /// </summary>
        private static WindowIcon? LoadWindowIcon()
        {
            try
            {
                // Load logo.png from embedded resources
                var assembly = Assembly.GetExecutingAssembly();

                // Find resource ending with logo.png
                string? resourceName = null;
                foreach (var name in assembly.GetManifestResourceNames())
                {
                    if (name.EndsWith("logo.png"))
                    {
                        resourceName = name;
                        break;
                    }
                }

                if (resourceName == null)
                {
                    Console.WriteLine("Warning: logo.png not found in resources, using procedural icon.");
                    return CreateProceduralIcon();
                }

                using var stream = assembly.GetManifestResourceStream(resourceName);
                if (stream != null)
                {
                    // Decode using SkiaSharp to get RGBA pixels
                    using var originalBitmap = SKBitmap.Decode(stream);
                    if (originalBitmap != null)
                    {
                        // Create multiple icon sizes for better compatibility
                        var sizes = new[] { 16, 32, 48, 64, 128, 256 };
                        var images = new List<Image>();

                        foreach (var size in sizes)
                        {
                            // Resize and convert to RGBA8888 with Premul alpha (standard for window icons)
                            var info = new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Premul);
                            using var resizedBitmap = new SKBitmap(info);
                            using var canvas = new SKCanvas(resizedBitmap);

                            // Clear with transparent
                            canvas.Clear(SKColors.Transparent);

                            // Draw scaled image with high quality
                            var srcRect = new SKRect(0, 0, originalBitmap.Width, originalBitmap.Height);
                            var destRect = new SKRect(0, 0, size, size);
                            using var paint = new SKPaint
                            {
                                FilterQuality = SKFilterQuality.High,
                                IsAntialias = true
                            };
                            canvas.DrawBitmap(originalBitmap, srcRect, destRect, paint);
                            canvas.Flush();

                            // Extract pixels
                            var pixels = new byte[size * size * 4];
                            Marshal.Copy(resizedBitmap.GetPixels(), pixels, 0, pixels.Length);
                            images.Add(new Image(size, size, pixels));
                        }

                        if (images.Count > 0)
                        {
                            return new WindowIcon(images.ToArray());
                        }
                    }
                }

                return CreateProceduralIcon();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not load window icon: {ex.Message}");
                return CreateProceduralIcon();
            }
        }

        /// <summary>
        /// Creates a procedural icon matching the Deep3DStudio branding.
        /// </summary>
        private static WindowIcon CreateProceduralIcon()
        {
            var sizes = new[] { 16, 32, 48, 64, 128, 256 };
            var images = new Image[sizes.Length];

            for (int i = 0; i < sizes.Length; i++)
            {
                int size = sizes[i];
                var info = new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Premul);
                using (var bitmap = new SKBitmap(info))
                using (var canvas = new SKCanvas(bitmap))
                {
                    canvas.Clear(SKColors.Transparent);

                    float cx = size / 2f;
                    float cy = size / 2f;
                    float r = size * 0.4f;

                    // Draw a stylized 3D cube wireframe with gradient colors
                    var strokeWidth = Math.Max(1f, size / 16f);

                    // Create gradient colors (blue -> purple -> orange)
                    var blueColor = new SKColor(0, 150, 255);
                    var purpleColor = new SKColor(150, 100, 200);
                    var orangeColor = new SKColor(255, 120, 80);

                    // Draw background circle
                    var bgPaint = new SKPaint
                    {
                        Color = new SKColor(40, 40, 50, 230),
                        Style = SKPaintStyle.Fill,
                        IsAntialias = true
                    };
                    canvas.DrawCircle(cx, cy, r * 1.1f, bgPaint);

                    // Draw cube edges
                    float cubeSize = r * 0.7f;
                    float offset = cubeSize * 0.4f;

                    // Front face (blue)
                    var frontPaint = new SKPaint { Color = blueColor, StrokeWidth = strokeWidth, Style = SKPaintStyle.Stroke, IsAntialias = true };
                    canvas.DrawRect(cx - cubeSize/2 - offset/2, cy - cubeSize/2 + offset/2, cubeSize, cubeSize, frontPaint);

                    // Back face (orange)
                    var backPaint = new SKPaint { Color = orangeColor, StrokeWidth = strokeWidth, Style = SKPaintStyle.Stroke, IsAntialias = true };
                    canvas.DrawRect(cx - cubeSize/2 + offset/2, cy - cubeSize/2 - offset/2, cubeSize, cubeSize, backPaint);

                    // Connecting edges (purple)
                    var edgePaint = new SKPaint { Color = purpleColor, StrokeWidth = strokeWidth, Style = SKPaintStyle.Stroke, IsAntialias = true };
                    // Top-left
                    canvas.DrawLine(cx - cubeSize/2 - offset/2, cy - cubeSize/2 + offset/2, cx - cubeSize/2 + offset/2, cy - cubeSize/2 - offset/2, edgePaint);
                    // Top-right
                    canvas.DrawLine(cx + cubeSize/2 - offset/2, cy - cubeSize/2 + offset/2, cx + cubeSize/2 + offset/2, cy - cubeSize/2 - offset/2, edgePaint);
                    // Bottom-left
                    canvas.DrawLine(cx - cubeSize/2 - offset/2, cy + cubeSize/2 + offset/2, cx - cubeSize/2 + offset/2, cy + cubeSize/2 - offset/2, edgePaint);
                    // Bottom-right
                    canvas.DrawLine(cx + cubeSize/2 - offset/2, cy + cubeSize/2 + offset/2, cx + cubeSize/2 + offset/2, cy + cubeSize/2 - offset/2, edgePaint);

                    // Corner dots
                    var dotPaint = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true };
                    float dotR = Math.Max(1f, size / 20f);
                    canvas.DrawCircle(cx - cubeSize/2 - offset/2, cy - cubeSize/2 + offset/2, dotR, dotPaint);
                    canvas.DrawCircle(cx + cubeSize/2 + offset/2, cy - cubeSize/2 - offset/2, dotR, dotPaint);

                    // Convert to byte array (OpenTK expects RGBA)
                    var pixels = new byte[size * size * 4];
                    Marshal.Copy(bitmap.GetPixels(), pixels, 0, pixels.Length);

                    images[i] = new Image(size, size, pixels);
                }
            }

            return new WindowIcon(images);
        }
    }
}
