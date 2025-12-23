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

                using (var codec = SKCodec.Create(memoryStream))
                {
                    if (codec == null) return -1;

                    // Decode directly to Rgba8888 to ensure consistent format for OpenGL
                    var info = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888, SKAlphaType.Premul);
                    using (var bitmap = new SKBitmap(info))
                    {
                         var result = codec.GetPixels(info, bitmap.GetPixels());
                         if (result != SKCodecResult.Success) return -1;

                         int tex;
                         GL.GenTextures(1, out tex);
                         GL.BindTexture(TextureTarget.Texture2D, tex);

                         GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                         GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                         GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                         GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                         // Ensure 1-byte packing alignment for arbitrary widths
                         GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);

                         // Now we know it's Rgba8888, so we use PixelFormat.Rgba
                         GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, bitmap.Width, bitmap.Height, 0,
                             PixelFormat.Rgba, PixelType.UnsignedByte, bitmap.GetPixels());

                         // Restore default alignment
                         GL.PixelStore(PixelStoreParameter.UnpackAlignment, 4);

                         return tex;
                    }
                }
            }
        }
    }
}
