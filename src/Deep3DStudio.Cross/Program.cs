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
                var assembly = Assembly.GetExecutingAssembly();
                Stream? stream = null;

                // Find the logo resource
                foreach (var name in assembly.GetManifestResourceNames())
                {
                    if (name.EndsWith("logo.png"))
                    {
                        stream = assembly.GetManifestResourceStream(name);
                        break;
                    }
                }

                if (stream == null)
                {
                    Console.WriteLine("Warning: logo.png resource not found");
                    return null;
                }

                using (stream)
                using (var bitmap = SKBitmap.Decode(stream))
                {
                    if (bitmap == null)
                    {
                        Console.WriteLine("Warning: Failed to decode logo.png");
                        return null;
                    }

                    // Create multiple icon sizes for best appearance
                    var sizes = new[] { 16, 32, 48, 64, 128, 256 };
                    var images = new Image[sizes.Length];

                    for (int i = 0; i < sizes.Length; i++)
                    {
                        int size = sizes[i];
                        var info = new SKImageInfo(size, size, SKColorType.Rgba8888, SKAlphaType.Unpremul);
                        using (var scaled = new SKBitmap(info))
                        using (var canvas = new SKCanvas(scaled))
                        {
                            canvas.Clear(SKColors.Transparent);
                            var destRect = new SKRect(0, 0, size, size);
                            canvas.DrawBitmap(bitmap, destRect, new SKPaint { FilterQuality = SKFilterQuality.High });

                            // Convert to byte array (OpenTK expects RGBA)
                            var pixels = new byte[size * size * 4];
                            Marshal.Copy(scaled.GetPixels(), pixels, 0, pixels.Length);

                            images[i] = new Image(size, size, pixels);
                        }
                    }

                    return new WindowIcon(images);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not load window icon: {ex.Message}");
                return null;
            }
        }
    }
}
