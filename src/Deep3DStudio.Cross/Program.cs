using System;
using System.IO;
using System.Reflection;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.Common.Input;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using ImGuiNET;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using System.Diagnostics;
using System.Runtime.InteropServices;
using SkiaSharp;

namespace Deep3DStudio
{
    class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            // Force legacy OpenGL support for ThreeDView
            Environment.SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3COMPAT");

            var nativeWindowSettings = new NativeWindowSettings()
            {
                Size = new Vector2i(1280, 720),
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
            }
            else
            {
                // Windows/Linux support 3.3 Compatibility Profile
                nativeWindowSettings.APIVersion = new Version(3, 3);
                nativeWindowSettings.Profile = ContextProfile.Compatability;
            }

            using (var window = new MainWindow(GameWindowSettings.Default, nativeWindowSettings))
            {
                window.Run();
            }
        }

        /// <summary>
        /// Loads the application window icon from embedded resource.
        /// Creates multiple sizes for best appearance on all platforms.
        /// </summary>
        private static WindowIcon? LoadWindowIcon()
        {
            try
            {
                // Create a procedural icon that matches the logo's design
                // (3D wireframe cube with blue-purple-orange gradient)
                return CreateProceduralIcon();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not load window icon: {ex.Message}");
                return null;
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
