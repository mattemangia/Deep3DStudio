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
    }
}
