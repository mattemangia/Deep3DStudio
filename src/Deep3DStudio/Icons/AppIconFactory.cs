using System;
using Gtk;
using Cairo;

namespace Deep3DStudio.Icons
{
    public static class AppIconFactory
    {
        public static Image GenerateIcon(string name, int size)
        {
            using (var surface = new ImageSurface(Format.Argb32, size, size))
            {
                using (var cr = new Context(surface))
                {
                    cr.SetSourceRGBA(0, 0, 0, 0); // Transparent background
                    cr.Paint();

                    // Draw Icon based on name
                    switch (name)
                    {
                        case "open": DrawOpenIcon(cr, size); break;
                        case "settings": DrawSettingsIcon(cr, size); break;
                        case "run": DrawRunIcon(cr, size); break;
                        case "pointcloud": DrawPointCloudIcon(cr, size); break;
                        case "mesh": DrawMeshIcon(cr, size); break;
                        case "wireframe": DrawWireframeIcon(cr, size); break;
                        case "rgb": DrawRgbIcon(cr, size); break;
                        case "depthmap": DrawDepthMapIcon(cr, size); break;
                        case "select": DrawSelectIcon(cr, size); break;
                        case "texture": DrawTextureIcon(cr, size); break;
                    }
                }

                // Convert Surface to Pixbuf
                surface.Flush();
                var data = surface.Data;
                // Copy the data since surface will be disposed
                var dataCopy = new byte[data.Length];
                Array.Copy(data, dataCopy, data.Length);
                var pixbuf = new Gdk.Pixbuf(dataCopy, Gdk.Colorspace.Rgb, true, 8, size, size, surface.Stride);
                return new Image(pixbuf);
            }
        }

        private static void DrawOpenIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.9, 0.8, 0.2); // Folder color
            cr.MoveTo(size * 0.1, size * 0.2);
            cr.LineTo(size * 0.4, size * 0.2);
            cr.LineTo(size * 0.5, size * 0.3);
            cr.LineTo(size * 0.9, size * 0.3);
            cr.LineTo(size * 0.9, size * 0.8);
            cr.LineTo(size * 0.1, size * 0.8);
            cr.ClosePath();
            cr.FillPreserve();
            cr.SetSourceRGB(0.7, 0.6, 0.1);
            cr.Stroke();
        }

        private static void DrawSettingsIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.4, 0.4, 0.4);
            cr.Translate(size / 2.0, size / 2.0);

            // Gear
            for (int i = 0; i < 8; i++)
            {
                cr.Rotate(Math.PI / 4.0);
                cr.Rectangle(-size * 0.1, -size * 0.45, size * 0.2, size * 0.15);
                cr.Fill();
            }

            cr.Arc(0, 0, size * 0.3, 0, 2 * Math.PI);
            cr.Fill();

            cr.SetSourceRGB(1, 1, 1);
            cr.Arc(0, 0, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawRunIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.2, 0.8, 0.2);
            cr.MoveTo(size * 0.3, size * 0.2);
            cr.LineTo(size * 0.8, size * 0.5);
            cr.LineTo(size * 0.3, size * 0.8);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawPointCloudIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.2, 0.5, 0.9);
            // Dots
            double r = size * 0.1;
            cr.Arc(size * 0.3, size * 0.3, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.7, size * 0.3, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.5, size * 0.5, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.3, size * 0.7, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.7, size * 0.7, r, 0, 2 * Math.PI); cr.Fill();
        }

        private static void DrawMeshIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.4, 0.2);
            cr.MoveTo(size * 0.5, size * 0.2);
            cr.LineTo(size * 0.8, size * 0.8);
            cr.LineTo(size * 0.2, size * 0.8);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawWireframeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0, 0, 0);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.5, size * 0.2);
            cr.LineTo(size * 0.8, size * 0.8);
            cr.LineTo(size * 0.2, size * 0.8);
            cr.ClosePath();
            cr.Stroke();
        }

        private static void DrawRgbIcon(Context cr, int size)
        {
            // RGB circles (like a Venn diagram)
            double r = size * 0.25;
            double cx = size * 0.5;
            double cy = size * 0.45;

            // Red circle
            cr.SetSourceRGBA(1.0, 0.2, 0.2, 0.8);
            cr.Arc(cx - r * 0.5, cy - r * 0.3, r, 0, 2 * Math.PI);
            cr.Fill();

            // Green circle
            cr.SetSourceRGBA(0.2, 1.0, 0.2, 0.8);
            cr.Arc(cx + r * 0.5, cy - r * 0.3, r, 0, 2 * Math.PI);
            cr.Fill();

            // Blue circle
            cr.SetSourceRGBA(0.2, 0.2, 1.0, 0.8);
            cr.Arc(cx, cy + r * 0.5, r, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawDepthMapIcon(Context cr, int size)
        {
            // Gradient bar representing depth colormap (blue to red)
            double barHeight = size * 0.6;
            double barWidth = size * 0.7;
            double startX = size * 0.15;
            double startY = size * 0.2;

            int steps = 8;
            double stepWidth = barWidth / steps;

            for (int i = 0; i < steps; i++)
            {
                // Turbo-like colormap: blue -> cyan -> green -> yellow -> red
                double t = (double)i / (steps - 1);
                double r, g, b;
                TurboColormap(t, out r, out g, out b);

                cr.SetSourceRGB(r, g, b);
                cr.Rectangle(startX + i * stepWidth, startY, stepWidth + 1, barHeight);
                cr.Fill();
            }

            // Border
            cr.SetSourceRGB(0.3, 0.3, 0.3);
            cr.LineWidth = 1;
            cr.Rectangle(startX, startY, barWidth, barHeight);
            cr.Stroke();
        }

        private static void DrawSelectIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.9, 0.9, 0.9);
            cr.LineWidth = 2;

            // Cursor arrow
            cr.MoveTo(size * 0.3, size * 0.2);
            cr.LineTo(size * 0.3, size * 0.8);
            cr.LineTo(size * 0.45, size * 0.65);
            cr.LineTo(size * 0.6, size * 0.9);
            cr.LineTo(size * 0.7, size * 0.85);
            cr.LineTo(size * 0.55, size * 0.6);
            cr.LineTo(size * 0.8, size * 0.6);
            cr.ClosePath();

            cr.FillPreserve();
            cr.SetSourceRGB(0.2, 0.2, 0.2);
            cr.Stroke();
        }

        private static void DrawTextureIcon(Context cr, int size)
        {
            // Checkerboard pattern
            cr.SetSourceRGB(1.0, 1.0, 1.0);
            cr.Rectangle(size * 0.2, size * 0.2, size * 0.6, size * 0.6);
            cr.Fill();

            cr.SetSourceRGB(0.4, 0.4, 0.4);
            int rows = 2;
            int cols = 2;
            double w = (size * 0.6) / cols;
            double h = (size * 0.6) / rows;

            for(int r=0; r<rows; r++) {
                for(int c=0; c<cols; c++) {
                    if ((r + c) % 2 == 0) {
                        cr.Rectangle(size * 0.2 + c * w, size * 0.2 + r * h, w, h);
                        cr.Fill();
                    }
                }
            }

            cr.SetSourceRGB(0, 0, 0);
            cr.LineWidth = 1;
            cr.Rectangle(size * 0.2, size * 0.2, size * 0.6, size * 0.6);
            cr.Stroke();
        }

        private static void TurboColormap(double t, out double r, out double g, out double b)
        {
            // Simplified turbo colormap approximation
            t = Math.Max(0, Math.Min(1, t));

            if (t < 0.25)
            {
                // Blue to Cyan
                double s = t / 0.25;
                r = 0.0; g = s; b = 1.0;
            }
            else if (t < 0.5)
            {
                // Cyan to Green
                double s = (t - 0.25) / 0.25;
                r = 0.0; g = 1.0; b = 1.0 - s;
            }
            else if (t < 0.75)
            {
                // Green to Yellow
                double s = (t - 0.5) / 0.25;
                r = s; g = 1.0; b = 0.0;
            }
            else
            {
                // Yellow to Red
                double s = (t - 0.75) / 0.25;
                r = 1.0; g = 1.0 - s; b = 0.0;
            }
        }
    }
}
