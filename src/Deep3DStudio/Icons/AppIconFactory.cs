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
    }
}
