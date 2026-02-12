using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using SkiaSharp;
using ImGuiNET;
using Deep3DStudio.Model;

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
        Rgb,
        DepthMap,
        Confidence,
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
        Fullscreen,
        Link,
        TripoSR,
        LGM,
        Wonder3D,
        NeRF,
        Refine,
        Mast3r,     // MASt3R - Matching And Stereo 3D Reconstruction (metric pointmaps)
        Must3r,     // MUSt3R - Multi-view Network (>2 images, video support)
        Video,      // Video file icon for MUSt3R video input
        Plane,
        Cube,
        Sphere,
        Cylinder,
        Cone,
        Torus,
        Circle,
        Polygon,
        GridMesh,
        VertexMove,
        Extrude,
        Inset,
        Bridge,
        MergeMeshes,
        MergePointClouds,
        VoxelFilter,
        OutlierFilter,
        DuplicateFilter,
        Normals,
        AxisFilter,
        RadiusCrop,
        DenseCloud,
        Georef,
        Residuals,
        Dem,
        GeoExport
    }

    public class ImGuiIconFactory : IDisposable
    {
        private Dictionary<IconType, int> _icons = new Dictionary<IconType, int>();
        private int _fallbackIcon;

        public ImGuiIconFactory()
        {
            LoadIcons();
        }

        public IntPtr GetIcon(IconType type)
        {
            if (_icons.TryGetValue(type, out int id))
                return (IntPtr)id;
            return _fallbackIcon != 0 ? (IntPtr)_fallbackIcon : IntPtr.Zero;
        }

        public void Dispose()
        {
            foreach (var id in _icons.Values)
            {
                GL.DeleteTexture(id);
            }
            _icons.Clear();
            if (_fallbackIcon != 0)
            {
                GL.DeleteTexture(_fallbackIcon);
                _fallbackIcon = 0;
            }
        }

        private void LoadIcons()
        {
            // Generate procedural icons using SkiaSharp to ensure they exist without external assets
            // This guarantees we have icons even if files are missing, and solves the request for "Where are the icons?"
            _fallbackIcon = CreateIcon(SKColors.DimGray, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRect(w * 0.18f, h * 0.18f, w * 0.64f, h * 0.64f, p);
                canvas.DrawLine(w * 0.32f, h * 0.5f, w * 0.68f, h * 0.5f, p);
                canvas.DrawLine(w * 0.5f, h * 0.32f, w * 0.5f, h * 0.68f, p);
            });

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
                // Bold tray + heat mark for readability at small toolbar size.
                var fill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true };
                var stroke = new SKPaint { Color = SKColors.Black.WithAlpha(180), StrokeWidth = 2, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRoundRect(new SKRect(w * 0.16f, h * 0.38f, w * 0.84f, h * 0.84f), 6, 6, fill);
                canvas.DrawRoundRect(new SKRect(w * 0.16f, h * 0.38f, w * 0.84f, h * 0.84f), 6, 6, stroke);

                var heat = new SKPaint { Color = SKColors.Gold, StrokeWidth = 4, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawLine(w * 0.30f, h * 0.20f, w * 0.30f, h * 0.34f, heat);
                canvas.DrawLine(w * 0.50f, h * 0.16f, w * 0.50f, h * 0.34f, heat);
                canvas.DrawLine(w * 0.70f, h * 0.20f, w * 0.70f, h * 0.34f, heat);
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

            _icons[IconType.Settings] = CreateIcon(SKColors.SlateGray, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2.5f, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.2f, p);
                for (int i = 0; i < 8; i++)
                {
                    float a = i * (float)Math.PI / 4.0f;
                    float x1 = w * 0.5f + (float)Math.Cos(a) * w * 0.28f;
                    float y1 = h * 0.5f + (float)Math.Sin(a) * h * 0.28f;
                    float x2 = w * 0.5f + (float)Math.Cos(a) * w * 0.38f;
                    float y2 = h * 0.5f + (float)Math.Sin(a) * h * 0.38f;
                    canvas.DrawLine(x1, y1, x2, y2, p);
                }
            });

            _icons[IconType.Save] = CreateIcon(SKColors.SeaGreen, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRect(w * 0.18f, h * 0.18f, w * 0.64f, h * 0.64f, p);
                canvas.DrawRect(w * 0.3f, h * 0.24f, w * 0.4f, h * 0.18f, p);
                canvas.DrawRect(w * 0.3f, h * 0.52f, w * 0.4f, h * 0.2f, p);
                canvas.DrawRect(w * 0.58f, h * 0.3f, w * 0.12f, h * 0.12f, new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Open] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2.2f, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRect(w * 0.14f, h * 0.34f, w * 0.72f, h * 0.42f, p);
                canvas.DrawRect(w * 0.2f, h * 0.24f, w * 0.28f, h * 0.12f, p);
                var pArrow = new SKPaint { Color = SKColors.Gold, StrokeWidth = 3, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawLine(w * 0.5f, h * 0.18f, w * 0.5f, h * 0.5f, pArrow);
                canvas.DrawLine(w * 0.42f, h * 0.4f, w * 0.5f, h * 0.5f, pArrow);
                canvas.DrawLine(w * 0.58f, h * 0.4f, w * 0.5f, h * 0.5f, pArrow);
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

            _icons[IconType.Rgb] = CreateIcon(SKColors.IndianRed, (canvas, w, h) => {
                float r = w * 0.22f;
                var fill = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };
                fill.Color = new SKColor(240, 70, 70, 210);
                canvas.DrawCircle(w * 0.38f, h * 0.40f, r, fill);
                fill.Color = new SKColor(70, 220, 90, 210);
                canvas.DrawCircle(w * 0.62f, h * 0.40f, r, fill);
                fill.Color = new SKColor(80, 110, 240, 210);
                canvas.DrawCircle(w * 0.50f, h * 0.62f, r, fill);
            });

            _icons[IconType.DepthMap] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                int steps = 12;
                float x0 = w * 0.18f;
                float y0 = h * 0.26f;
                float bw = w * 0.64f;
                float bh = h * 0.48f;
                float sw = bw / steps;
                var p = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };
                for (int i = 0; i < steps; i++)
                {
                    float t = (float)i / (steps - 1);
                    var (r, g, b) = ImageUtils.TurboColormap(t);
                    p.Color = new SKColor((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
                    canvas.DrawRect(x0 + i * sw, y0, sw + 1, bh, p);
                }
                var border = new SKPaint { Color = SKColors.White.WithAlpha(210), StrokeWidth = 2, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRect(x0, y0, bw, bh, border);
            });

            _icons[IconType.Confidence] = CreateIcon(SKColors.SeaGreen, (canvas, w, h) => {
                float x = w * 0.24f;
                float y = h * 0.18f;
                float bw = w * 0.24f;
                float bh = h * 0.64f;
                int steps = 10;
                float sh = bh / steps;
                var p = new SKPaint { Style = SKPaintStyle.Fill, IsAntialias = true };
                for (int i = 0; i < steps; i++)
                {
                    float t = (float)i / (steps - 1);
                    var (r, g, b) = ImageUtils.TurboColormap(1.0f - t);
                    p.Color = new SKColor((byte)(r * 255), (byte)(g * 255), (byte)(b * 255));
                    canvas.DrawRect(x, y + i * sh, bw, sh + 1, p);
                }
                var border = new SKPaint { Color = SKColors.White.WithAlpha(210), StrokeWidth = 2, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawRect(x, y, bw, bh, border);
                var check = new SKPaint { Color = SKColors.White, StrokeWidth = 4, Style = SKPaintStyle.Stroke, IsAntialias = true };
                canvas.DrawLine(w * 0.56f, h * 0.60f, w * 0.68f, h * 0.72f, check);
                canvas.DrawLine(w * 0.68f, h * 0.72f, w * 0.86f, h * 0.36f, check);
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
                // Thick corner brackets + center block, visible even at 20px.
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 5, Style = SKPaintStyle.Stroke, IsAntialias = true, StrokeCap = SKStrokeCap.Round };
                float m = w * 0.16f;
                float s = w * 0.22f;

                canvas.DrawLine(m, m + s, m, m, p);
                canvas.DrawLine(m, m, m + s, m, p);
                canvas.DrawLine(w - m - s, m, w - m, m, p);
                canvas.DrawLine(w - m, m, w - m, m + s, p);
                canvas.DrawLine(m, h - m - s, m, h - m, p);
                canvas.DrawLine(m, h - m, m + s, h - m, p);
                canvas.DrawLine(w - m - s, h - m, w - m, h - m, p);
                canvas.DrawLine(w - m, h - m - s, w - m, h - m, p);

                var center = new SKPaint { Color = SKColors.Gold, Style = SKPaintStyle.Fill, IsAntialias = true };
                canvas.DrawRoundRect(new SKRect(w * 0.40f, h * 0.40f, w * 0.60f, h * 0.60f), 3, 3, center);
            });

            _icons[IconType.Link] = CreateIcon(SKColors.LimeGreen, (canvas, w, h) => {
                // Link/Chain icon for auto workflow toggle
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke, IsAntialias = true };
                // Two chain links
                canvas.DrawOval(new SKRect(w * 0.15f, h * 0.3f, w * 0.55f, h * 0.7f), p);
                canvas.DrawOval(new SKRect(w * 0.45f, h * 0.3f, w * 0.85f, h * 0.7f), p);
            });

            _icons[IconType.TripoSR] = CreateIcon(SKColors.DeepPink, (canvas, w, h) => {
                // Single-view input card -> 3D output block.
                var stroke = new SKPaint { Color = SKColors.White, StrokeWidth = 2.2f, Style = SKPaintStyle.Stroke, IsAntialias = true };
                var fill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true };
                var accent = new SKPaint { Color = SKColors.Gold, Style = SKPaintStyle.Fill, IsAntialias = true };

                canvas.DrawRoundRect(new SKRect(w * 0.10f, h * 0.22f, w * 0.42f, h * 0.78f), 4, 4, stroke);
                canvas.DrawCircle(w * 0.22f, h * 0.38f, 3, fill);
                canvas.DrawRect(w * 0.16f, h * 0.52f, w * 0.20f, h * 0.12f, fill);

                var path = new SKPath();
                path.MoveTo(w * 0.47f, h * 0.50f);
                path.LineTo(w * 0.63f, h * 0.40f);
                path.LineTo(w * 0.63f, h * 0.60f);
                path.Close();
                canvas.DrawPath(path, accent);

                canvas.DrawRoundRect(new SKRect(w * 0.68f, h * 0.32f, w * 0.90f, h * 0.68f), 3, 3, stroke);
                canvas.DrawLine(w * 0.68f, h * 0.32f, w * 0.78f, h * 0.22f, stroke);
                canvas.DrawLine(w * 0.90f, h * 0.32f, w * 0.78f, h * 0.22f, stroke);
                canvas.DrawLine(w * 0.78f, h * 0.22f, w * 0.78f, h * 0.56f, stroke);
            });

            _icons[IconType.LGM] = CreateIcon(SKColors.Purple, (canvas, w, h) => {
                // LGM icon - Large Gaussian Model
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                // Multiple gaussian dots
                canvas.DrawCircle(w * 0.3f, h * 0.3f, 6, pFill);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 8, pFill);
                canvas.DrawCircle(w * 0.7f, h * 0.35f, 5, pFill);
                canvas.DrawCircle(w * 0.4f, h * 0.7f, 7, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.65f, 6, pFill);
            });

            _icons[IconType.Wonder3D] = CreateIcon(SKColors.Teal, (canvas, w, h) => {
                // Multi-view stack with strong filled cards.
                var card = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill, IsAntialias = true };
                var border = new SKPaint { Color = SKColors.Black.WithAlpha(160), StrokeWidth = 1.8f, Style = SKPaintStyle.Stroke, IsAntialias = true };
                var dot = new SKPaint { Color = SKColors.Gold, Style = SKPaintStyle.Fill, IsAntialias = true };

                canvas.DrawRoundRect(new SKRect(w * 0.12f, h * 0.30f, w * 0.38f, h * 0.74f), 3, 3, card);
                canvas.DrawRoundRect(new SKRect(w * 0.12f, h * 0.30f, w * 0.38f, h * 0.74f), 3, 3, border);
                canvas.DrawRoundRect(new SKRect(w * 0.34f, h * 0.20f, w * 0.62f, h * 0.68f), 3, 3, card);
                canvas.DrawRoundRect(new SKRect(w * 0.34f, h * 0.20f, w * 0.62f, h * 0.68f), 3, 3, border);
                canvas.DrawRoundRect(new SKRect(w * 0.58f, h * 0.30f, w * 0.86f, h * 0.74f), 3, 3, card);
                canvas.DrawRoundRect(new SKRect(w * 0.58f, h * 0.30f, w * 0.86f, h * 0.74f), 3, 3, border);

                canvas.DrawCircle(w * 0.24f, h * 0.52f, 2.8f, dot);
                canvas.DrawCircle(w * 0.48f, h * 0.44f, 2.8f, dot);
                canvas.DrawCircle(w * 0.72f, h * 0.52f, 2.8f, dot);
            });

            _icons[IconType.NeRF] = CreateIcon(SKColors.OrangeRed, (canvas, w, h) => {
                // NeRF icon - neural radiance field
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                // Neural network representation
                canvas.DrawCircle(w * 0.2f, h * 0.3f, 5, pFill);
                canvas.DrawCircle(w * 0.2f, h * 0.7f, 5, pFill);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 6, pFill);
                canvas.DrawCircle(w * 0.8f, h * 0.5f, 5, pFill);
                // Connections
                canvas.DrawLine(w * 0.25f, h * 0.3f, w * 0.45f, h * 0.5f, p);
                canvas.DrawLine(w * 0.25f, h * 0.7f, w * 0.45f, h * 0.5f, p);
                canvas.DrawLine(w * 0.55f, h * 0.5f, w * 0.75f, h * 0.5f, p);
            });

            _icons[IconType.Refine] = CreateIcon(SKColors.Gold, (canvas, w, h) => {
                // Refine icon - polish/enhance
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                // Star/sparkle
                canvas.DrawLine(w * 0.5f, h * 0.15f, w * 0.5f, h * 0.85f, p);
                canvas.DrawLine(w * 0.15f, h * 0.5f, w * 0.85f, h * 0.5f, p);
                canvas.DrawLine(w * 0.25f, h * 0.25f, w * 0.75f, h * 0.75f, p);
                canvas.DrawLine(w * 0.75f, h * 0.25f, w * 0.25f, h * 0.75f, p);
            });

            _icons[IconType.Mast3r] = CreateIcon(SKColors.DodgerBlue, (canvas, w, h) => {
                // MASt3R icon - metric matching with two connected views
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };

                // Two image frames (stereo pair)
                canvas.DrawRect(w * 0.1f, h * 0.25f, w * 0.3f, h * 0.4f, p);
                canvas.DrawRect(w * 0.6f, h * 0.25f, w * 0.3f, h * 0.4f, p);

                // Connection lines (matching features)
                var pMatch = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.4f, h * 0.35f, w * 0.6f, h * 0.35f, pMatch);
                canvas.DrawLine(w * 0.4f, h * 0.5f, w * 0.6f, h * 0.5f, pMatch);
                canvas.DrawLine(w * 0.4f, h * 0.55f, w * 0.6f, h * 0.55f, pMatch);

                // Feature points
                canvas.DrawCircle(w * 0.25f, h * 0.35f, 3, pFill);
                canvas.DrawCircle(w * 0.25f, h * 0.5f, 3, pFill);
                canvas.DrawCircle(w * 0.75f, h * 0.35f, 3, pFill);
                canvas.DrawCircle(w * 0.75f, h * 0.5f, 3, pFill);

                // Metric indicator (ruler)
                canvas.DrawLine(w * 0.2f, h * 0.75f, w * 0.8f, h * 0.75f, p);
                canvas.DrawLine(w * 0.2f, h * 0.7f, w * 0.2f, h * 0.8f, p);
                canvas.DrawLine(w * 0.8f, h * 0.7f, w * 0.8f, h * 0.8f, p);
            });

            _icons[IconType.Must3r] = CreateIcon(SKColors.MediumPurple, (canvas, w, h) => {
                // MUSt3R icon - multi-view with video/streaming indicator
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };

                // Multiple overlapping image frames (multi-view)
                canvas.DrawRect(w * 0.1f, h * 0.2f, w * 0.25f, h * 0.35f, p);
                canvas.DrawRect(w * 0.25f, h * 0.15f, w * 0.25f, h * 0.35f, p);
                canvas.DrawRect(w * 0.4f, h * 0.2f, w * 0.25f, h * 0.35f, p);
                canvas.DrawRect(w * 0.55f, h * 0.15f, w * 0.25f, h * 0.35f, p);

                // Central 3D point cloud result
                canvas.DrawCircle(w * 0.5f, h * 0.65f, 5, pFill);
                canvas.DrawCircle(w * 0.35f, h * 0.7f, 4, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.7f, 4, pFill);
                canvas.DrawCircle(w * 0.4f, h * 0.8f, 3, pFill);
                canvas.DrawCircle(w * 0.6f, h * 0.8f, 3, pFill);

                // Flow/streaming arrows
                var pArrow = new SKPaint { Color = SKColors.LimeGreen, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.35f, h * 0.55f, w * 0.45f, h * 0.6f, pArrow);
                canvas.DrawLine(w * 0.65f, h * 0.55f, w * 0.55f, h * 0.6f, pArrow);
            });

            _icons[IconType.Video] = CreateIcon(SKColors.Crimson, (canvas, w, h) => {
                // Video icon - film strip with play button
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };

                // Film strip
                canvas.DrawRect(w * 0.15f, h * 0.2f, w * 0.7f, h * 0.6f, p);

                // Sprocket holes (top)
                canvas.DrawRect(w * 0.2f, h * 0.25f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.35f, h * 0.25f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.5f, h * 0.25f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.65f, h * 0.25f, w * 0.08f, h * 0.08f, pFill);

                // Sprocket holes (bottom)
                canvas.DrawRect(w * 0.2f, h * 0.67f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.35f, h * 0.67f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.5f, h * 0.67f, w * 0.08f, h * 0.08f, pFill);
                canvas.DrawRect(w * 0.65f, h * 0.67f, w * 0.08f, h * 0.08f, pFill);

                // Play button in center
                var playPath = new SKPath();
                playPath.MoveTo(w * 0.4f, h * 0.4f);
                playPath.LineTo(w * 0.4f, h * 0.6f);
                playPath.LineTo(w * 0.6f, h * 0.5f);
                playPath.Close();
                canvas.DrawPath(playPath, pFill);
            });

            _icons[IconType.Plane] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                var pStroke = new SKPaint { Color = SKColors.Black, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.3f, w * 0.7f, h * 0.4f, pFill);
                canvas.DrawRect(w * 0.15f, h * 0.3f, w * 0.7f, h * 0.4f, pStroke);
            });

            _icons[IconType.Cube] = CreateIcon(SKColors.SandyBrown, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.2f, h * 0.25f, w * 0.45f, h * 0.45f, p);
                canvas.DrawLine(w * 0.2f, h * 0.25f, w * 0.38f, h * 0.1f, p);
                canvas.DrawLine(w * 0.65f, h * 0.25f, w * 0.83f, h * 0.1f, p);
                canvas.DrawLine(w * 0.38f, h * 0.1f, w * 0.83f, h * 0.1f, p);
            });

            _icons[IconType.Sphere] = CreateIcon(SKColors.LightSkyBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.28f, p);
            });

            _icons[IconType.Cylinder] = CreateIcon(SKColors.YellowGreen, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.25f, h * 0.2f, w * 0.5f, h * 0.55f, p);
                canvas.DrawOval(new SKRect(w * 0.25f, h * 0.14f, w * 0.75f, h * 0.28f), p);
                canvas.DrawOval(new SKRect(w * 0.25f, h * 0.68f, w * 0.75f, h * 0.82f), p);
            });

            _icons[IconType.Cone] = CreateIcon(SKColors.Orange, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var tri = new SKPath();
                tri.MoveTo(w * 0.5f, h * 0.15f);
                tri.LineTo(w * 0.2f, h * 0.8f);
                tri.LineTo(w * 0.8f, h * 0.8f);
                tri.Close();
                canvas.DrawPath(tri, p);
            });

            _icons[IconType.Torus] = CreateIcon(SKColors.MediumSeaGreen, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.3f, p);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.15f, p);
            });

            _icons[IconType.Circle] = CreateIcon(SKColors.SlateBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.24f, pFill);
            });

            _icons[IconType.Polygon] = CreateIcon(SKColors.Plum, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                var path = new SKPath();
                for (int i = 0; i < 6; i++)
                {
                    float a = (float)(i * Math.PI * 2.0 / 6.0 - Math.PI / 2.0);
                    float x = w * 0.5f + (float)Math.Cos(a) * w * 0.28f;
                    float y = h * 0.5f + (float)Math.Sin(a) * h * 0.28f;
                    if (i == 0) path.MoveTo(x, y); else path.LineTo(x, y);
                }
                path.Close();
                canvas.DrawPath(path, p);
            });

            _icons[IconType.GridMesh] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 1.6f, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.15f, w * 0.7f, h * 0.7f, p);
                for (int i = 1; i < 4; i++)
                {
                    float t = i / 4.0f;
                    canvas.DrawLine(w * (0.15f + 0.7f * t), h * 0.15f, w * (0.15f + 0.7f * t), h * 0.85f, p);
                    canvas.DrawLine(w * 0.15f, h * (0.15f + 0.7f * t), w * 0.85f, h * (0.15f + 0.7f * t), p);
                }
            });

            _icons[IconType.VertexMove] = CreateIcon(SKColors.CornflowerBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.35f, h * 0.65f, 4, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.65f, 4, pFill);
                canvas.DrawCircle(w * 0.5f, h * 0.35f, 4, pFill);
                var p = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.5f, h * 0.2f, w * 0.5f, h * 0.48f, p);
            });

            _icons[IconType.Extrude] = CreateIcon(SKColors.Goldenrod, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.2f, h * 0.45f, w * 0.4f, h * 0.3f, p);
                var pa = new SKPaint { Color = SKColors.LimeGreen, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.72f, h * 0.72f, w * 0.72f, h * 0.2f, pa);
            });

            _icons[IconType.Inset] = CreateIcon(SKColors.DodgerBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.15f, w * 0.7f, h * 0.7f, p);
                canvas.DrawRect(w * 0.3f, h * 0.3f, w * 0.4f, h * 0.4f, p);
            });

            _icons[IconType.Bridge] = CreateIcon(SKColors.IndianRed, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.12f, h * 0.3f, w * 0.22f, h * 0.4f, p);
                canvas.DrawRect(w * 0.66f, h * 0.3f, w * 0.22f, h * 0.4f, p);
                canvas.DrawLine(w * 0.34f, h * 0.5f, w * 0.66f, h * 0.5f, p);
            });

            _icons[IconType.MergeMeshes] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.12f, h * 0.22f, w * 0.28f, h * 0.56f, p);
                canvas.DrawRect(w * 0.6f, h * 0.22f, w * 0.28f, h * 0.56f, p);
                canvas.DrawLine(w * 0.4f, h * 0.5f, w * 0.6f, h * 0.5f, p);

                var tri = new SKPath();
                tri.MoveTo(w * 0.5f, h * 0.28f);
                tri.LineTo(w * 0.38f, h * 0.6f);
                tri.LineTo(w * 0.62f, h * 0.6f);
                tri.Close();
                canvas.DrawPath(tri, new SKPaint { Color = SKColors.Gold, StrokeWidth = 1.5f, Style = SKPaintStyle.Stroke });
            });

            _icons[IconType.MergePointClouds] = CreateIcon(SKColors.Teal, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.12f, h * 0.22f, w * 0.28f, h * 0.56f, p);
                canvas.DrawRect(w * 0.6f, h * 0.22f, w * 0.28f, h * 0.56f, p);
                canvas.DrawLine(w * 0.4f, h * 0.5f, w * 0.6f, h * 0.5f, p);

                var dot = new SKPaint { Color = SKColors.DeepSkyBlue, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.3f, h * 0.36f, 3, dot);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 3, dot);
                canvas.DrawCircle(w * 0.7f, h * 0.36f, 3, dot);
                canvas.DrawCircle(w * 0.5f, h * 0.64f, 3, dot);
            });

            _icons[IconType.VoxelFilter] = CreateIcon(SKColors.CadetBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 1.5f, Style = SKPaintStyle.Stroke };
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        canvas.DrawRect(w * (0.2f + c * 0.18f), h * (0.2f + r * 0.18f), w * 0.14f, h * 0.14f, p);
                    }
                }
            });

            _icons[IconType.OutlierFilter] = CreateIcon(SKColors.MediumSlateBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                for (int i = 0; i < 8; i++)
                {
                    float a = (float)(i * Math.PI * 2.0 / 8.0);
                    canvas.DrawCircle(w * (0.5f + (float)Math.Cos(a) * 0.22f), h * (0.5f + (float)Math.Sin(a) * 0.22f), 3, pFill);
                }
                var p = new SKPaint { Color = SKColors.Red, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.74f, h * 0.2f, w * 0.9f, h * 0.36f, p);
            });

            _icons[IconType.DuplicateFilter] = CreateIcon(SKColors.MediumTurquoise, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.4f, h * 0.5f, 7, pFill);
                canvas.DrawCircle(w * 0.6f, h * 0.5f, 7, pFill);
                var p = new SKPaint { Color = SKColors.LimeGreen, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.5f, h * 0.2f, w * 0.5f, h * 0.8f, p);
            });

            _icons[IconType.Normals] = CreateIcon(SKColors.DarkSeaGreen, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.35f, h * 0.65f, 3, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.65f, 3, pFill);
                canvas.DrawCircle(w * 0.5f, h * 0.4f, 3, pFill);
                var p = new SKPaint { Color = SKColors.DeepSkyBlue, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.5f, h * 0.65f, w * 0.5f, h * 0.18f, p);
            });

            _icons[IconType.AxisFilter] = CreateIcon(SKColors.SteelBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.25f, h * 0.5f, 3, pFill);
                canvas.DrawCircle(w * 0.5f, h * 0.5f, 3, pFill);
                canvas.DrawCircle(w * 0.75f, h * 0.5f, 3, pFill);
                var p = new SKPaint { Color = SKColors.Yellow, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.35f, h * 0.35f, w * 0.3f, h * 0.3f, p);
            });

            _icons[IconType.RadiusCrop] = CreateIcon(SKColors.Teal, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.28f, p);
                var pFill = new SKPaint { Color = SKColors.DeepSkyBlue, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.5f, h * 0.5f, w * 0.12f, pFill);
            });

            _icons[IconType.DenseCloud] = CreateIcon(SKColors.MediumTurquoise, (canvas, w, h) => {
                var basePaint = new SKPaint { Color = SKColors.DeepSkyBlue, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.24f, h * 0.28f, 3, basePaint);
                canvas.DrawCircle(w * 0.52f, h * 0.22f, 3, basePaint);
                canvas.DrawCircle(w * 0.72f, h * 0.44f, 3, basePaint);
                canvas.DrawCircle(w * 0.32f, h * 0.68f, 3, basePaint);
                canvas.DrawCircle(w * 0.68f, h * 0.72f, 3, basePaint);

                var densePaint = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.38f, h * 0.3f, 2.5f, densePaint);
                canvas.DrawCircle(w * 0.6f, h * 0.33f, 2.5f, densePaint);
                canvas.DrawCircle(w * 0.48f, h * 0.52f, 2.5f, densePaint);
                canvas.DrawCircle(w * 0.56f, h * 0.62f, 2.5f, densePaint);

                var arrow = new SKPaint { Color = SKColors.Gold, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.12f, h * 0.82f, w * 0.12f, h * 0.54f, arrow);
                var path = new SKPath();
                path.MoveTo(w * 0.12f, h * 0.46f);
                path.LineTo(w * 0.06f, h * 0.56f);
                path.LineTo(w * 0.18f, h * 0.56f);
                path.Close();
                canvas.DrawPath(path, new SKPaint { Color = SKColors.Gold, Style = SKPaintStyle.Fill });
            });

            _icons[IconType.Georef] = CreateIcon(SKColors.CadetBlue, (canvas, w, h) => {
                var pFill = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Fill };
                canvas.DrawCircle(w * 0.35f, h * 0.35f, 6, pFill);
                canvas.DrawCircle(w * 0.65f, h * 0.35f, 6, pFill);
                var p = new SKPaint { Color = SKColors.DeepSkyBlue, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.35f, h * 0.45f, w * 0.35f, h * 0.82f, p);
                canvas.DrawLine(w * 0.65f, h * 0.45f, w * 0.65f, h * 0.82f, p);
            });

            _icons[IconType.Residuals] = CreateIcon(SKColors.SlateGray, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.2f, w * 0.7f, h * 0.6f, p);
                var p2 = new SKPaint { Color = SKColors.LimeGreen, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.2f, h * 0.65f, w * 0.45f, h * 0.45f, p2);
                canvas.DrawLine(w * 0.45f, h * 0.45f, w * 0.8f, h * 0.32f, p2);
            });

            _icons[IconType.Dem] = CreateIcon(SKColors.DarkOliveGreen, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 1.6f, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.2f, w * 0.7f, h * 0.6f, p);
                for (int i = 1; i < 4; i++)
                {
                    float yy = h * (0.2f + i * 0.15f);
                    canvas.DrawLine(w * 0.15f, yy, w * 0.85f, yy, p);
                }
            });

            _icons[IconType.GeoExport] = CreateIcon(SKColors.RoyalBlue, (canvas, w, h) => {
                var p = new SKPaint { Color = SKColors.White, StrokeWidth = 2, Style = SKPaintStyle.Stroke };
                canvas.DrawRect(w * 0.15f, h * 0.2f, w * 0.45f, h * 0.55f, p);
                var pArrow = new SKPaint { Color = SKColors.LimeGreen, StrokeWidth = 3, Style = SKPaintStyle.Stroke };
                canvas.DrawLine(w * 0.62f, h * 0.55f, w * 0.88f, h * 0.55f, pArrow);
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
