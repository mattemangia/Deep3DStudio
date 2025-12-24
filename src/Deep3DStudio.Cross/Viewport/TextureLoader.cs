using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;
using SkiaSharp;

namespace Deep3DStudio.Viewport
{
    public static class TextureLoader
    {
        /// <summary>
        /// Loads a texture from an embedded resource.
        /// </summary>
        public static int LoadTextureFromResource(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            var stream = assembly.GetManifestResourceStream(resourceName);

            // Fallback to searching all names if exact match fails
            if (stream == null)
            {
                foreach (var name in assembly.GetManifestResourceNames())
                {
                    if (name.EndsWith(resourceName))
                    {
                        stream = assembly.GetManifestResourceStream(name);
                        break;
                    }
                }
            }

            if (stream == null)
            {
                Console.WriteLine($"Resource not found: {resourceName}");
                return -1;
            }

            using (stream)
            {
                return LoadTextureFromStream(stream);
            }
        }

        /// <summary>
        /// Loads a texture from a file path.
        /// </summary>
        public static int LoadTextureFromFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"File not found: {filePath}");
                return -1;
            }

            using (var stream = File.OpenRead(filePath))
            {
                return LoadTextureFromStream(stream);
            }
        }

        /// <summary>
        /// Loads a texture from a stream.
        /// </summary>
        public static int LoadTextureFromStream(Stream stream)
        {
            using (var memoryStream = new MemoryStream())
            {
                stream.CopyTo(memoryStream);
                memoryStream.Position = 0;

                using (var bitmap = SKBitmap.Decode(memoryStream))
                {
                    if (bitmap == null)
                    {
                        Console.WriteLine("Failed to decode image");
                        return -1;
                    }

                    return CreateTextureFromBitmap(bitmap);
                }
            }
        }

        /// <summary>
        /// Creates an OpenGL texture from an SKBitmap, handling platform-specific pixel format differences.
        /// </summary>
        public static int CreateTextureFromBitmap(SKBitmap bitmap)
        {
            // Convert to a consistent BGRA format which is what most platforms expect
            // SkiaSharp internally uses BGRA on most platforms
            SKBitmap convertedBitmap;
            bool needsDispose = false;

            if (bitmap.ColorType != SKColorType.Bgra8888)
            {
                var info = new SKImageInfo(bitmap.Width, bitmap.Height, SKColorType.Bgra8888, SKAlphaType.Premul);
                convertedBitmap = new SKBitmap(info);
                needsDispose = true;

                using (var canvas = new SKCanvas(convertedBitmap))
                {
                    canvas.DrawBitmap(bitmap, 0, 0);
                }
            }
            else
            {
                convertedBitmap = bitmap;
            }

            try
            {
                int tex;
                GL.GenTextures(1, out tex);
                GL.BindTexture(TextureTarget.Texture2D, tex);

                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                // Ensure proper alignment
                GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);
                GL.PixelStore(PixelStoreParameter.UnpackRowLength, 0);

                // Use BGRA format which matches SkiaSharp's internal format
                GL.TexImage2D(
                    TextureTarget.Texture2D,
                    0,
                    PixelInternalFormat.Rgba,
                    convertedBitmap.Width,
                    convertedBitmap.Height,
                    0,
                    PixelFormat.Bgra,
                    PixelType.UnsignedByte,
                    convertedBitmap.GetPixels());

                // Restore default alignment
                GL.PixelStore(PixelStoreParameter.UnpackAlignment, 4);

                GL.BindTexture(TextureTarget.Texture2D, 0);

                return tex;
            }
            finally
            {
                if (needsDispose)
                {
                    convertedBitmap.Dispose();
                }
            }
        }

        /// <summary>
        /// Creates a texture from raw pixel data.
        /// </summary>
        public static int CreateTextureFromPixels(IntPtr pixels, int width, int height, bool isBgra = true)
        {
            int tex;
            GL.GenTextures(1, out tex);
            GL.BindTexture(TextureTarget.Texture2D, tex);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

            GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);

            GL.TexImage2D(
                TextureTarget.Texture2D,
                0,
                PixelInternalFormat.Rgba,
                width,
                height,
                0,
                isBgra ? PixelFormat.Bgra : PixelFormat.Rgba,
                PixelType.UnsignedByte,
                pixels);

            GL.PixelStore(PixelStoreParameter.UnpackAlignment, 4);
            GL.BindTexture(TextureTarget.Texture2D, 0);

            return tex;
        }

        /// <summary>
        /// Creates a thumbnail texture from an image file.
        /// </summary>
        public static int CreateThumbnail(string filePath, int maxSize = 128)
        {
            if (!File.Exists(filePath)) return -1;

            try
            {
                using (var stream = File.OpenRead(filePath))
                using (var bitmap = SKBitmap.Decode(stream))
                {
                    if (bitmap == null) return -1;

                    // Calculate thumbnail size maintaining aspect ratio
                    float scale = Math.Min((float)maxSize / bitmap.Width, (float)maxSize / bitmap.Height);
                    int thumbWidth = (int)(bitmap.Width * scale);
                    int thumbHeight = (int)(bitmap.Height * scale);

                    // Create scaled bitmap
                    var info = new SKImageInfo(thumbWidth, thumbHeight, SKColorType.Bgra8888, SKAlphaType.Premul);
                    using (var thumbnail = new SKBitmap(info))
                    using (var canvas = new SKCanvas(thumbnail))
                    {
                        canvas.Clear(SKColors.Transparent);
                        var destRect = new SKRect(0, 0, thumbWidth, thumbHeight);
                        canvas.DrawBitmap(bitmap, destRect, new SKPaint { FilterQuality = SKFilterQuality.Medium });

                        return CreateTextureFromBitmap(thumbnail);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error creating thumbnail for {filePath}: {ex.Message}");
                return -1;
            }
        }

        /// <summary>
        /// Deletes a texture from GPU memory.
        /// </summary>
        public static void DeleteTexture(int textureId)
        {
            if (textureId > 0)
            {
                GL.DeleteTexture(textureId);
            }
        }

        /// <summary>
        /// Creates a runtime-generated logo texture using SkiaSharp.
        /// This is used as a fallback when the logo.png resource is not available,
        /// which can happen especially on macOS where embedded resources may not work correctly.
        /// </summary>
        public static int CreateRuntimeLogo(int size = 256)
        {
            var info = new SKImageInfo(size, size, SKColorType.Bgra8888, SKAlphaType.Premul);
            using (var bitmap = new SKBitmap(info))
            using (var canvas = new SKCanvas(bitmap))
            {
                // Clear with transparent background
                canvas.Clear(SKColors.Transparent);

                float cx = size / 2f;
                float cy = size / 2f;
                float radius = size * 0.42f;

                // Draw gradient background circle
                using (var bgPaint = new SKPaint())
                {
                    bgPaint.IsAntialias = true;
                    bgPaint.Shader = SKShader.CreateRadialGradient(
                        new SKPoint(cx, cy * 0.7f),
                        radius * 1.2f,
                        new SKColor[] {
                            new SKColor(60, 120, 200),    // Light blue
                            new SKColor(30, 60, 120)      // Dark blue
                        },
                        SKShaderTileMode.Clamp);
                    canvas.DrawCircle(cx, cy, radius, bgPaint);
                }

                // Draw 3D cube wireframe (representing 3D reconstruction)
                using (var linePaint = new SKPaint())
                {
                    linePaint.IsAntialias = true;
                    linePaint.Style = SKPaintStyle.Stroke;
                    linePaint.StrokeWidth = size * 0.02f;
                    linePaint.Color = SKColors.White;

                    float cubeSize = size * 0.25f;
                    float offset = size * 0.08f;

                    // Front face (lower-left)
                    float fx = cx - cubeSize / 2 - offset / 2;
                    float fy = cy + offset / 2;
                    canvas.DrawRect(fx, fy, cubeSize, cubeSize, linePaint);

                    // Back face (upper-right)
                    float bx = fx + offset;
                    float by = fy - offset;
                    canvas.DrawRect(bx, by, cubeSize, cubeSize, linePaint);

                    // Connect corners
                    canvas.DrawLine(fx, fy, bx, by, linePaint);
                    canvas.DrawLine(fx + cubeSize, fy, bx + cubeSize, by, linePaint);
                    canvas.DrawLine(fx, fy + cubeSize, bx, by + cubeSize, linePaint);
                    canvas.DrawLine(fx + cubeSize, fy + cubeSize, bx + cubeSize, by + cubeSize, linePaint);
                }

                // Draw neural network nodes (representing AI/Neural)
                using (var nodePaint = new SKPaint())
                {
                    nodePaint.IsAntialias = true;
                    nodePaint.Style = SKPaintStyle.Fill;

                    using (var connectionPaint = new SKPaint())
                    {
                        connectionPaint.IsAntialias = true;
                        connectionPaint.Style = SKPaintStyle.Stroke;
                        connectionPaint.StrokeWidth = size * 0.008f;
                        connectionPaint.Color = new SKColor(200, 200, 255, 150);

                        float nodeRadius = size * 0.025f;

                        // Left column nodes
                        SKPoint[] leftNodes = {
                            new SKPoint(cx - radius * 0.6f, cy - radius * 0.3f),
                            new SKPoint(cx - radius * 0.6f, cy),
                            new SKPoint(cx - radius * 0.6f, cy + radius * 0.3f)
                        };

                        // Right column nodes
                        SKPoint[] rightNodes = {
                            new SKPoint(cx + radius * 0.6f, cy - radius * 0.3f),
                            new SKPoint(cx + radius * 0.6f, cy),
                            new SKPoint(cx + radius * 0.6f, cy + radius * 0.3f)
                        };

                        // Draw connections
                        foreach (var left in leftNodes)
                        {
                            foreach (var right in rightNodes)
                            {
                                canvas.DrawLine(left, right, connectionPaint);
                            }
                        }

                        // Draw nodes with gradient
                        nodePaint.Shader = SKShader.CreateRadialGradient(
                            new SKPoint(0, 0), nodeRadius * 2,
                            new SKColor[] { SKColors.White, new SKColor(100, 180, 255) },
                            SKShaderTileMode.Clamp);

                        foreach (var node in leftNodes)
                        {
                            nodePaint.Shader = SKShader.CreateRadialGradient(
                                node, nodeRadius,
                                new SKColor[] { SKColors.White, new SKColor(100, 180, 255) },
                                SKShaderTileMode.Clamp);
                            canvas.DrawCircle(node, nodeRadius, nodePaint);
                        }
                        foreach (var node in rightNodes)
                        {
                            nodePaint.Shader = SKShader.CreateRadialGradient(
                                node, nodeRadius,
                                new SKColor[] { SKColors.White, new SKColor(100, 180, 255) },
                                SKShaderTileMode.Clamp);
                            canvas.DrawCircle(node, nodeRadius, nodePaint);
                        }
                    }
                }

                // Draw "D3D" text
                using (var textPaint = new SKPaint())
                {
                    textPaint.IsAntialias = true;
                    textPaint.Color = SKColors.White;
                    textPaint.TextSize = size * 0.12f;
                    textPaint.TextAlign = SKTextAlign.Center;
                    textPaint.FakeBoldText = true;

                    canvas.DrawText("D3D", cx, cy + radius * 0.8f, textPaint);
                }

                // Draw outer ring glow
                using (var ringPaint = new SKPaint())
                {
                    ringPaint.IsAntialias = true;
                    ringPaint.Style = SKPaintStyle.Stroke;
                    ringPaint.StrokeWidth = size * 0.015f;
                    ringPaint.Color = new SKColor(100, 180, 255, 200);
                    canvas.DrawCircle(cx, cy, radius * 0.95f, ringPaint);
                }

                return CreateTextureFromBitmap(bitmap);
            }
        }
    }
}
