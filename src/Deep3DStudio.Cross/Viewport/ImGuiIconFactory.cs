using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using SkiaSharp;
using ImGuiNET;

namespace Deep3DStudio.Viewport
{
    public enum IconType
    {
        Select,
        Move,
        Rotate,
        Scale,
        Run,
        Mesh,
        Cloud,
        Camera,
        Settings,
        Delete,
        Save,
        Open,
        Bake,
        Clean
    }

    public class ImGuiIconFactory : IDisposable
    {
        private Dictionary<IconType, int> _icons = new Dictionary<IconType, int>();

        public ImGuiIconFactory()
        {
            LoadIcons();
        }

        public IntPtr GetIcon(IconType type)
        {
            if (_icons.TryGetValue(type, out int id))
                return (IntPtr)id;
            return IntPtr.Zero;
        }

        public void Dispose()
        {
            foreach (var id in _icons.Values)
            {
                GL.DeleteTexture(id);
            }
            _icons.Clear();
        }

        private void LoadIcons()
        {
            // Generate procedural icons using SkiaSharp to ensure they exist without external assets
            // This guarantees we have icons even if files are missing, and solves the request for "Where are the icons?"

            _icons[IconType.Select] = CreateIcon(SKColors.White, (canvas, w, h) => {
                // Cursor arrow
                var path = new SKPath();
                path.MoveTo(w*0.2f, h*0.2f);
                path.LineTo(w*0.2f, h*0.8f);
                path.LineTo(w*0.4f, h*0.6f);
                path.LineTo(w*0.6f, h*0.9f);
                path.LineTo(w*0.7f, h*0.8f);
                path.LineTo(w*0.5f, h*0.5f);
                path.LineTo(w*0.8f, h*0.5f);
                path.Close();
                canvas.DrawPath(path, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
                canvas.DrawPath(path, new SKPaint { Color = SKColors.Black, Style = SKPaintStyle.Stroke, StrokeWidth = 2 });
            });

            _icons[IconType.Move] = CreateIcon(SKColors.LightBlue, (canvas, w, h) => {
                // Cross arrows
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 4, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w*0.5f, h*0.2f, w*0.5f, h*0.8f, p);
                canvas.DrawLine(w*0.2f, h*0.5f, w*0.8f, h*0.5f, p);
            });

            _icons[IconType.Rotate] = CreateIcon(SKColors.LightGreen, (canvas, w, h) => {
                // Circle arrow
                var rect = new SKRect(w*0.2f, h*0.2f, w*0.8f, h*0.8f);
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 4, Style = SKPaintStyle.Stroke };
                canvas.DrawArc(rect, 45, 270, false, p);
            });

            _icons[IconType.Scale] = CreateIcon(SKColors.LightPink, (canvas, w, h) => {
                // Box expanding
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w*0.3f, h*0.3f, w*0.4f, h*0.4f, p);
                canvas.DrawLine(w*0.4f, h*0.4f, w*0.8f, h*0.8f, p); // Diagonal
            });

            _icons[IconType.Run] = CreateIcon(SKColors.LimeGreen, (canvas, w, h) => {
                 var path = new SKPath();
                 path.MoveTo(w*0.3f, h*0.2f);
                 path.LineTo(w*0.3f, h*0.8f);
                 path.LineTo(w*0.8f, h*0.5f);
                 path.Close();
                 canvas.DrawPath(path, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Clean] = CreateIcon(SKColors.Orange, (canvas, w, h) => {
                // Broom / Brush
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 5, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w*0.3f, h*0.7f, w*0.7f, h*0.3f, p);
            });

            _icons[IconType.Bake] = CreateIcon(SKColors.Purple, (canvas, w, h) => {
                // Texture / Grid
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w*0.2f, h*0.2f, w*0.6f, h*0.6f, p);
                canvas.DrawLine(w*0.5f, h*0.2f, w*0.5f, h*0.8f, p);
                canvas.DrawLine(w*0.2f, h*0.5f, w*0.8f, h*0.5f, p);
            });

            _icons[IconType.Delete] = CreateIcon(SKColors.Red, (canvas, w, h) => {
                 var p = new SKPaint { Color = SKColors.White, StrokeWidth = 4, Style = SKPaintStyle.Stroke };
                 canvas.DrawLine(w*0.2f, h*0.2f, w*0.8f, h*0.8f, p);
                 canvas.DrawLine(w*0.8f, h*0.2f, w*0.2f, h*0.8f, p);
            });

            _icons[IconType.Cloud] = CreateIcon(SKColors.Cyan, (canvas, w, h) => {
                 // Point Cloud Icon
                 var p = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                 canvas.DrawCircle(w*0.3f, h*0.5f, 4, p);
                 canvas.DrawCircle(w*0.5f, h*0.3f, 4, p);
                 canvas.DrawCircle(w*0.7f, h*0.5f, 4, p);
                 canvas.DrawCircle(w*0.5f, h*0.7f, 4, p);
                 canvas.DrawCircle(w*0.5f, h*0.5f, 4, p);
            });

            _icons[IconType.Mesh] = CreateIcon(SKColors.Magenta, (canvas, w, h) => {
                 // Mesh Icon (Wireframe triangle)
                 var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                 var path = new SKPath();
                 path.MoveTo(w*0.5f, h*0.2f);
                 path.LineTo(w*0.2f, h*0.8f);
                 path.LineTo(w*0.8f, h*0.8f);
                 path.Close();
                 canvas.DrawPath(path, p);
                 // Internal lines
                 canvas.DrawLine(w*0.5f, h*0.2f, w*0.5f, h*0.8f, p);
            });
        }

        private int CreateIcon(SKColor bg, Action<SKCanvas, int, int> drawAction)
        {
            int size = 64;
            using (var bitmap = new SKBitmap(size, size, SKColorType.Rgba8888, SKAlphaType.Premul))
            using (var canvas = new SKCanvas(bitmap))
            {
                canvas.Clear(SKColors.Transparent);

                // Draw simple background shape or keep transparent
                // drawAction should draw white/colored icon content

                drawAction(canvas, size, size);

                // Upload
                int tex;
                GL.GenTextures(1, out tex);
                GL.BindTexture(TextureTarget.Texture2D, tex);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

                // SKBitmap in Rgba8888 is R, G, B, A in memory
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, size, size, 0,
                    PixelFormat.Rgba, PixelType.UnsignedByte, bitmap.GetPixels());

                return tex;
            }
        }
    }
}
