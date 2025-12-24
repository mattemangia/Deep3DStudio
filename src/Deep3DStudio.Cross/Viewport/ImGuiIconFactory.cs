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
        Clean,
        Texture,
        Grid,
        Wireframe,
        Focus,
        Pen,
        Skeleton,
        Decimate,
        Optimize,
        PointCloudGen,
        MeshGen,
        Rig,
        FlipNormals,
        Subdivide,
        Weld,
        Paint,
        SelectAll,
        InvertSelection,
        GrowSelection,
        ClearSelection,
        Smooth,
        Fullscreen
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

            _icons[IconType.Camera] = CreateIcon(SKColors.Yellow, (canvas, w, h) => {
                 // Camera Icon
                 var p = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Stroke, StrokeWidth = 2 };
                 var box = new SKRect(w*0.2f, h*0.3f, w*0.8f, h*0.7f);
                 canvas.DrawRect(box, p);
                 // Lens
                 canvas.DrawCircle(w*0.5f, h*0.5f, w*0.15f, p);
                 // Flash/Viewfinder
                 canvas.DrawRect(w*0.6f, h*0.2f, w*0.15f, h*0.1f, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Texture] = CreateIcon(SKColors.Pink, (canvas, w, h) => {
                 // Checkerboard
                 var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                 float s = w * 0.5f;
                 canvas.DrawRect(0, 0, s, s, pFill);
                 canvas.DrawRect(s, s, s, s, pFill);
            });

            _icons[IconType.Grid] = CreateIcon(SKColors.Gray, (canvas, w, h) => {
                 // Grid lines
                 var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2 };
                 for(float i=0; i<=w; i+=w/3f) {
                     canvas.DrawLine(i, 0, i, h, p);
                     canvas.DrawLine(0, i, w, i, p);
                 }
            });

            _icons[IconType.Wireframe] = CreateIcon(SKColors.LightBlue, (canvas, w, h) => {
                 // Wireframe cube
                 var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                 var r = new SKRect(w*0.3f, h*0.3f, w*0.7f, h*0.7f);
                 canvas.DrawRect(r, p);
                 canvas.DrawLine(0,0, w*0.3f, h*0.3f, p);
                 canvas.DrawLine(w,0, w*0.7f, h*0.3f, p);
                 canvas.DrawLine(0,h, w*0.3f, h*0.7f, p);
                 canvas.DrawLine(w,h, w*0.7f, h*0.7f, p);
            });

            _icons[IconType.Focus] = CreateIcon(SKColors.LightBlue, (canvas, w, h) => {
                // Target / Focus icon
                float cx = w * 0.5f;
                float cy = h * 0.5f;
                float r = w * 0.3f;
                var p = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Stroke, StrokeWidth = 3 };
                canvas.DrawCircle(cx, cy, r, p);
                canvas.DrawLine(cx - r - 5, cy, cx + r + 5, cy, p);
                canvas.DrawLine(cx, cy - r - 5, cx, cy + r + 5, p);
                canvas.DrawCircle(cx, cy, 3, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Pen] = CreateIcon(SKColors.Orange, (canvas, w, h) => {
                // Pen / Brush icon
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                var path = new SKPath();
                path.MoveTo(w * 0.2f, h * 0.8f);
                path.LineTo(w * 0.3f, h * 0.6f);
                path.LineTo(w * 0.7f, h * 0.2f);
                path.LineTo(w * 0.8f, h * 0.3f);
                path.LineTo(w * 0.4f, h * 0.7f);
                path.Close();
                canvas.DrawPath(path, p);
                canvas.DrawPath(path, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true });
            });

            _icons[IconType.Skeleton] = CreateIcon(SKColors.LightGreen, (canvas, w, h) => {
                // Skeleton / Bone icon
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 4, Style = SKPaintStyle.Stroke };
                // Spine
                canvas.DrawLine(w * 0.5f, h * 0.15f, w * 0.5f, h * 0.5f, p);
                // Ribs
                canvas.DrawLine(w * 0.3f, h * 0.35f, w * 0.7f, h * 0.35f, p);
                // Pelvis
                canvas.DrawLine(w * 0.3f, h * 0.5f, w * 0.7f, h * 0.5f, p);
                // Legs
                canvas.DrawLine(w * 0.35f, h * 0.5f, w * 0.25f, h * 0.85f, p);
                canvas.DrawLine(w * 0.65f, h * 0.5f, w * 0.75f, h * 0.85f, p);
                // Head
                canvas.DrawCircle(w * 0.5f, h * 0.15f, 6, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Decimate] = CreateIcon(SKColors.Coral, (canvas, w, h) => {
                // Decimate icon - mesh simplification
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                var pStroke = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                // Original mesh triangles
                var path = new SKPath();
                path.MoveTo(w * 0.1f, h * 0.8f);
                path.LineTo(w * 0.5f, h * 0.2f);
                path.LineTo(w * 0.9f, h * 0.8f);
                path.Close();
                canvas.DrawPath(path, pStroke);
                // Arrow down
                canvas.DrawLine(w * 0.5f, h * 0.5f, w * 0.5f, h * 0.7f, new SKPaint { Color = SKColors.Yellow, StrokeWidth = 3 });
            });

            _icons[IconType.Optimize] = CreateIcon(SKColors.Gold, (canvas, w, h) => {
                // Optimize icon - gear/cog
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.25f, p);
                // Gear teeth
                for (int i = 0; i < 8; i++)
                {
                    float angle = i * 45 * (float)Math.PI / 180f;
                    float x1 = w * 0.5f + (float)Math.Cos(angle) * w * 0.25f;
                    float y1 = h * 0.5f + (float)Math.Sin(angle) * h * 0.25f;
                    float x2 = w * 0.5f + (float)Math.Cos(angle) * w * 0.35f;
                    float y2 = h * 0.5f + (float)Math.Sin(angle) * h * 0.35f;
                    canvas.DrawLine(x1, y1, x2, y2, p);
                }
            });

            _icons[IconType.PointCloudGen] = CreateIcon(SKColors.Cyan, (canvas, w, h) => {
                // Point Cloud Generation icon
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                // Multiple dots forming a cloud
                canvas.DrawCircle(w * 0.25f, h * 0.4f, 4, pFill);
                canvas.DrawCircle(w * 0.45f, h * 0.25f, 4, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.35f, 4, pFill);
                canvas.DrawCircle(w * 0.35f, h * 0.55f, 4, pFill);
                canvas.DrawCircle(w * 0.55f, h * 0.5f, 4, pFill);
                canvas.DrawCircle(w * 0.75f, h * 0.55f, 4, pFill);
                canvas.DrawCircle(w * 0.45f, h * 0.7f, 4, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.75f, 4, pFill);
                // Arrow
                var pArrow = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.15f, h * 0.85f, w * 0.4f, h * 0.6f, pArrow);
            });

            _icons[IconType.MeshGen] = CreateIcon(SKColors.Magenta, (canvas, w, h) => {
                // Mesh Generation icon
                var pStroke = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                // Wireframe mesh
                var path = new SKPath();
                path.MoveTo(w * 0.5f, h * 0.15f);
                path.LineTo(w * 0.15f, h * 0.85f);
                path.LineTo(w * 0.85f, h * 0.85f);
                path.Close();
                canvas.DrawPath(path, pStroke);
                canvas.DrawLine(w * 0.5f, h * 0.15f, w * 0.5f, h * 0.85f, pStroke);
                canvas.DrawLine(w * 0.325f, h * 0.5f, w * 0.675f, h * 0.5f, pStroke);
                // Arrow
                var pArrow = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.1f, h * 0.1f, w * 0.3f, h * 0.3f, pArrow);
            });

            _icons[IconType.Rig] = CreateIcon(SKColors.LightGreen, (canvas, w, h) => {
                // Auto Rig icon
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                // Simplified humanoid figure
                canvas.DrawCircle(w * 0.5f, h * 0.15f, 5, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
                canvas.DrawLine(w * 0.5f, h * 0.2f, w * 0.5f, h * 0.5f, p); // Body
                canvas.DrawLine(w * 0.25f, h * 0.35f, w * 0.75f, h * 0.35f, p); // Arms
                canvas.DrawLine(w * 0.5f, h * 0.5f, w * 0.3f, h * 0.85f, p); // Left leg
                canvas.DrawLine(w * 0.5f, h * 0.5f, w * 0.7f, h * 0.85f, p); // Right leg
                // Joints
                canvas.DrawCircle(w * 0.25f, h * 0.35f, 3, new SKPaint { Color = SKColors.Yellow, Style = SKPaintStyle.Fill });
                canvas.DrawCircle(w * 0.75f, h * 0.35f, 3, new SKPaint { Color = SKColors.Yellow, Style = SKPaintStyle.Fill });
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 3, new SKPaint { Color = SKColors.Yellow, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.FlipNormals] = CreateIcon(SKColors.SkyBlue, (canvas, w, h) => {
                // Flip normals icon - two opposing arrows
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.3f, h * 0.3f, w * 0.7f, h * 0.7f, p);
                canvas.DrawLine(w * 0.7f, h * 0.3f, w * 0.3f, h * 0.7f, p);
                // Arrowheads
                canvas.DrawLine(w * 0.3f, h * 0.3f, w * 0.4f, h * 0.3f, p);
                canvas.DrawLine(w * 0.3f, h * 0.3f, w * 0.3f, h * 0.4f, p);
                canvas.DrawLine(w * 0.7f, h * 0.7f, w * 0.6f, h * 0.7f, p);
                canvas.DrawLine(w * 0.7f, h * 0.7f, w * 0.7f, h * 0.6f, p);
            });

            _icons[IconType.Subdivide] = CreateIcon(SKColors.LightBlue, (canvas, w, h) => {
                // Subdivide icon - triangle split into 4
                var pStroke = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                // Outer triangle
                var path = new SKPath();
                path.MoveTo(w * 0.5f, h * 0.1f);
                path.LineTo(w * 0.1f, h * 0.9f);
                path.LineTo(w * 0.9f, h * 0.9f);
                path.Close();
                canvas.DrawPath(path, pStroke);
                // Inner lines for subdivision
                canvas.DrawLine(w * 0.3f, h * 0.5f, w * 0.7f, h * 0.5f, pStroke);
                canvas.DrawLine(w * 0.5f, h * 0.1f, w * 0.5f, h * 0.9f, pStroke);
                canvas.DrawLine(w * 0.3f, h * 0.5f, w * 0.5f, h * 0.9f, pStroke);
                canvas.DrawLine(w * 0.7f, h * 0.5f, w * 0.5f, h * 0.9f, pStroke);
            });

            _icons[IconType.Weld] = CreateIcon(SKColors.Orange, (canvas, w, h) => {
                // Weld icon - merging points
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                var pStroke = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                // Two dots converging
                canvas.DrawCircle(w * 0.25f, h * 0.5f, 6, pFill);
                canvas.DrawCircle(w * 0.75f, h * 0.5f, 6, pFill);
                // Arrow to center
                canvas.DrawLine(w * 0.35f, h * 0.5f, w * 0.45f, h * 0.5f, pStroke);
                canvas.DrawLine(w * 0.55f, h * 0.5f, w * 0.65f, h * 0.5f, pStroke);
                // Center merged point
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 4, new SKPaint { Color = SKColors.Yellow, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Paint] = CreateIcon(SKColors.HotPink, (canvas, w, h) => {
                // Paint bucket icon
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                // Bucket body
                canvas.DrawRect(w * 0.2f, h * 0.3f, w * 0.5f, h * 0.5f, p);
                // Handle
                canvas.DrawArc(new SKRect(w * 0.35f, h * 0.15f, w * 0.55f, h * 0.35f), 180, 180, false, p);
                // Paint drop
                var drop = new SKPath();
                drop.MoveTo(w * 0.75f, h * 0.4f);
                drop.QuadTo(w * 0.85f, h * 0.55f, w * 0.75f, h * 0.7f);
                drop.QuadTo(w * 0.65f, h * 0.55f, w * 0.75f, h * 0.4f);
                canvas.DrawPath(drop, pFill);
            });

            _icons[IconType.SelectAll] = CreateIcon(SKColors.LightGray, (canvas, w, h) => {
                // Select all icon - multiple selection boxes
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.15f, w * 0.4f, h * 0.4f, p);
                canvas.DrawRect(w * 0.3f, h * 0.3f, w * 0.4f, h * 0.4f, p);
                canvas.DrawRect(w * 0.45f, h * 0.45f, w * 0.4f, h * 0.4f, p);
            });

            _icons[IconType.InvertSelection] = CreateIcon(SKColors.LightGray, (canvas, w, h) => {
                // Invert selection icon - yin-yang style
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.35f, p);
                // Half filled
                canvas.DrawArc(new SKRect(w * 0.15f, h * 0.15f, w * 0.85f, h * 0.85f), 90, 180, true, pFill);
            });

            _icons[IconType.GrowSelection] = CreateIcon(SKColors.LightGray, (canvas, w, h) => {
                // Grow selection icon - expanding circles
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.15f, p);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.28f, p);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.4f, p);
            });

            _icons[IconType.ClearSelection] = CreateIcon(SKColors.LightGray, (canvas, w, h) => {
                // Clear selection icon - X mark
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.2f, h * 0.2f, w * 0.6f, h * 0.6f, p);
                canvas.DrawLine(w * 0.3f, h * 0.3f, w * 0.7f, h * 0.7f, new SKPaint { Color = SKColors.Red, StrokeWidth = 3, Style = SKPaintStyle.Stroke });
                canvas.DrawLine(w * 0.7f, h * 0.3f, w * 0.3f, h * 0.7f, new SKPaint { Color = SKColors.Red, StrokeWidth = 3, Style = SKPaintStyle.Stroke });
            });

            _icons[IconType.Smooth] = CreateIcon(SKColors.LightGreen, (canvas, w, h) => {
                // Smooth icon - wave becoming flat
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                var path = new SKPath();
                path.MoveTo(w * 0.1f, h * 0.5f);
                path.QuadTo(w * 0.3f, h * 0.2f, w * 0.5f, h * 0.5f);
                path.QuadTo(w * 0.7f, h * 0.8f, w * 0.9f, h * 0.5f);
                canvas.DrawPath(path, p);
            });

            _icons[IconType.Fullscreen] = CreateIcon(SKColors.DodgerBlue, (canvas, w, h) => {
                // Fullscreen icon - four arrows pointing to corners
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2.5f, Style = SKPaintStyle.Stroke, IsAntialias = true };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true };

                float margin = w * 0.15f;
                float arrowSize = w * 0.18f;

                // Top-left corner
                canvas.DrawLine(margin, margin, margin + arrowSize, margin, p);
                canvas.DrawLine(margin, margin, margin, margin + arrowSize, p);

                // Top-right corner
                canvas.DrawLine(w - margin, margin, w - margin - arrowSize, margin, p);
                canvas.DrawLine(w - margin, margin, w - margin, margin + arrowSize, p);

                // Bottom-left corner
                canvas.DrawLine(margin, h - margin, margin + arrowSize, h - margin, p);
                canvas.DrawLine(margin, h - margin, margin, h - margin - arrowSize, p);

                // Bottom-right corner
                canvas.DrawLine(w - margin, h - margin, w - margin - arrowSize, h - margin, p);
                canvas.DrawLine(w - margin, h - margin, w - margin, h - margin - arrowSize, p);

                // Center rectangle (representing the window)
                float rectMargin = w * 0.3f;
                canvas.DrawRect(rectMargin, rectMargin, w - 2 * rectMargin, h - 2 * rectMargin, p);
            });
        }

        private int CreateIcon(SKColor bg, Action<SKCanvas, int, int> drawAction)
        {
            int size = 64;
            // Use BGRA8888 which is the native format on most platforms
            using (var bitmap = new SKBitmap(size, size, SKColorType.Bgra8888, SKAlphaType.Premul))
            using (var canvas = new SKCanvas(bitmap))
            {
                canvas.Clear(SKColors.Transparent);

                // Draw icon content
                drawAction(canvas, size, size);

                // Upload texture
                int tex;
                GL.GenTextures(1, out tex);
                GL.BindTexture(TextureTarget.Texture2D, tex);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                GL.PixelStore(PixelStoreParameter.UnpackAlignment, 1);

                // Use BGRA format to match SkiaSharp's internal format
                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, size, size, 0,
                    PixelFormat.Bgra, PixelType.UnsignedByte, bitmap.GetPixels());

                GL.PixelStore(PixelStoreParameter.UnpackAlignment, 4);
                GL.BindTexture(TextureTarget.Texture2D, 0);

                return tex;
            }
        }
    }
}
