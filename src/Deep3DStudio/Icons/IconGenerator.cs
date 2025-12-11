using System;
using Cairo;

namespace Deep3DStudio.Icons
{
    public static class IconGenerator
    {
        public static Gtk.Image GenerateAddIcon(int size)
        {
            var surface = new ImageSurface(Format.Argb32, size, size);
            using (var cr = new Context(surface))
            {
                cr.SetSourceRGBA(0, 0, 0, 0); // Transparent
                cr.Paint();

                cr.SetSourceRGB(0.2, 0.6, 1.0);
                cr.LineWidth = 3;

                // Draw Plus
                cr.MoveTo(size / 2, 4);
                cr.LineTo(size / 2, size - 4);
                cr.MoveTo(4, size / 2);
                cr.LineTo(size - 4, size / 2);
                cr.Stroke();
            }
            return new Gtk.Image(new Gdk.Pixbuf(surface, 0, 0, size, size));
        }

        public static Gtk.Image GenerateRunIcon(int size)
        {
            var surface = new ImageSurface(Format.Argb32, size, size);
            using (var cr = new Context(surface))
            {
                cr.SetSourceRGBA(0, 0, 0, 0);
                cr.Paint();

                cr.SetSourceRGB(0.2, 0.8, 0.2);

                // Draw Play Triangle
                cr.MoveTo(6, 4);
                cr.LineTo(size - 6, size / 2);
                cr.LineTo(6, size - 4);
                cr.ClosePath();
                cr.Fill();
            }
            return new Gtk.Image(new Gdk.Pixbuf(surface, 0, 0, size, size));
        }
    }
}
