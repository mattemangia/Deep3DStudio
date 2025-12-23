using System;
using System.IO;
using System.Reflection;
using OpenTK.Graphics.OpenGL;
using SkiaSharp;

namespace Deep3DStudio.Viewport
{
    public static class TextureLoader
    {
        public static int LoadTextureFromResource(string resourceName)
        {
            var assembly = Assembly.GetExecutingAssembly();
            // Try to find the resource in the Cross assembly or the referenced one
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
            using (var memoryStream = new MemoryStream())
            {
                stream.CopyTo(memoryStream);
                memoryStream.Position = 0;

                using (var bitmap = SKBitmap.Decode(memoryStream))
                {
                    if (bitmap == null) return -1;

                    int tex;
                    GL.GenTextures(1, out tex);
                    GL.BindTexture(TextureTarget.Texture2D, tex);

                    GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                    GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                    GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                    GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                    var data = bitmap.Pixels;
                    // Convert SKColor array to byte array if needed or use GetPixels
                    // SKBitmap.GetPixels() returns IntPtr

                    GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, bitmap.Width, bitmap.Height, 0,
                        PixelFormat.Bgra, PixelType.UnsignedByte, bitmap.GetPixels());

                    return tex;
                }
            }
        }
    }
}
