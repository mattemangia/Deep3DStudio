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
using Deep3DStudio.Model;

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
                    using var bitmap = SKBitmap.Decode(stream);
                    if (bitmap != null)
                    {
                        // OpenTK WindowIcon expects Images array.
                        // We can provide the original size.
                        // Ensure RGBA and Unpremultiplied Alpha for Window Icon
                        // Using Unpremul to avoid dark halos on macOS/Windows
                        var info = new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
                        using var rgbaBitmap = new SKBitmap(info);

                        // We need to draw the original bitmap onto the new one, but Copy/Draw might premultiply.
                        // Instead, let's try to convert pixel data directly if possible, or redraw.
                        // SKCanvas usually works with Premul.
                        // For Unpremul, we might need to rely on the decode providing it or manual copy.
                        // SKBitmap.Decode usually provides Premul.

                        // Safe approach: Decode directly to desired info if possible, or copy pixels.
                        if (bitmap.ColorType == SKColorType.Rgba8888 && bitmap.AlphaType == SKAlphaType.Unpremul)
                        {
                             var pixels = new byte[bitmap.Width * bitmap.Height * 4];
                             Marshal.Copy(bitmap.GetPixels(), pixels, 0, pixels.Length);
                             return new WindowIcon(new Image(bitmap.Width, bitmap.Height, pixels));
                        }

                        // Force conversion
                        if (bitmap.CopyTo(rgbaBitmap))
                        {
                            var pixels = new byte[rgbaBitmap.Width * rgbaBitmap.Height * 4];
                            Marshal.Copy(rgbaBitmap.GetPixels(), pixels, 0, pixels.Length);
                            return new WindowIcon(new Image(rgbaBitmap.Width, rgbaBitmap.Height, pixels));
                        }
                    }
                }

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
