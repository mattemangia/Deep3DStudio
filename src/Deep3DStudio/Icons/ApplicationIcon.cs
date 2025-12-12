using System;
using Cairo;
using Gdk;

namespace Deep3DStudio.Icons
{
    /// <summary>
    /// Generates the application icon programmatically using Cairo graphics.
    /// The icon represents a stylized 3D reconstruction concept with depth layers.
    /// </summary>
    public static class ApplicationIcon
    {
        /// <summary>
        /// Creates a Pixbuf application icon at the specified size.
        /// The icon features a stylized 3D cube with depth gradient representing NeRF/3D reconstruction.
        /// </summary>
        public static Pixbuf Create(int size)
        {
            using (var surface = new ImageSurface(Format.Argb32, size, size))
            {
                using (var cr = new Context(surface))
                {
                    DrawApplicationIcon(cr, size);
                }

                surface.Flush();
                var data = surface.Data;
                var dataCopy = new byte[data.Length];
                Array.Copy(data, dataCopy, data.Length);

                return new Pixbuf(dataCopy, Colorspace.Rgb, true, 8, size, size, surface.Stride);
            }
        }

        /// <summary>
        /// Draws the complete application icon.
        /// Design: A stylized isometric 3D cube with depth gradient layers,
        /// representing the NeRF/3D reconstruction concept.
        /// </summary>
        private static void DrawApplicationIcon(Context cr, int size)
        {
            double s = size;
            double cx = s / 2;
            double cy = s / 2;
            double margin = s * 0.08;

            // Clear with transparent background
            cr.SetSourceRGBA(0, 0, 0, 0);
            cr.Paint();

            // Draw circular background with gradient (deep blue to purple)
            DrawBackgroundCircle(cr, cx, cy, s, margin);

            // Draw the stylized 3D cube/mesh representation
            Draw3DCube(cr, cx, cy, s, margin);

            // Draw depth scan lines effect
            DrawScanLines(cr, cx, cy, s, margin);

            // Draw point cloud dots
            DrawPointCloudDots(cr, cx, cy, s, margin);

            // Draw highlight/glow effect
            DrawHighlight(cr, cx, cy, s, margin);
        }

        private static void DrawBackgroundCircle(Context cr, double cx, double cy, double s, double margin)
        {
            double radius = (s - margin * 2) / 2;

            // Gradient from deep blue to purple
            using (var gradient = new RadialGradient(cx - radius * 0.3, cy - radius * 0.3, 0, cx, cy, radius * 1.2))
            {
                gradient.AddColorStop(0, new Cairo.Color(0.2, 0.3, 0.5, 1.0));    // Lighter blue center
                gradient.AddColorStop(0.5, new Cairo.Color(0.1, 0.15, 0.35, 1.0)); // Mid blue
                gradient.AddColorStop(1, new Cairo.Color(0.08, 0.08, 0.2, 1.0));   // Dark blue edge

                cr.Arc(cx, cy, radius, 0, 2 * Math.PI);
                cr.SetSource(gradient);
                cr.Fill();
            }

            // Subtle border
            cr.Arc(cx, cy, radius - 1, 0, 2 * Math.PI);
            cr.SetSourceRGBA(0.3, 0.4, 0.6, 0.5);
            cr.LineWidth = s * 0.015;
            cr.Stroke();
        }

        private static void Draw3DCube(Context cr, double cx, double cy, double s, double margin)
        {
            // Isometric cube parameters
            double cubeSize = s * 0.35;
            double isoAngle = Math.PI / 6; // 30 degrees for isometric
            double offsetY = s * 0.05;

            // Calculate isometric projections
            double dx = Math.Cos(isoAngle) * cubeSize;
            double dy = Math.Sin(isoAngle) * cubeSize;

            // Center the cube
            double baseCx = cx;
            double baseCy = cy + offsetY;

            // Front face vertices (parallelogram)
            PointD frontTL = new PointD(baseCx - dx, baseCy - cubeSize * 0.5 + dy);
            PointD frontTR = new PointD(baseCx + dx, baseCy - cubeSize * 0.5 - dy);
            PointD frontBR = new PointD(baseCx + dx, baseCy + cubeSize * 0.5 - dy);
            PointD frontBL = new PointD(baseCx - dx, baseCy + cubeSize * 0.5 + dy);

            // Top face (shift up)
            double topOffset = cubeSize * 0.6;
            PointD topTL = new PointD(frontTL.X, frontTL.Y - topOffset);
            PointD topTR = new PointD(frontTR.X, frontTR.Y - topOffset);
            PointD topBL = new PointD(baseCx, baseCy - cubeSize * 0.5 - topOffset + dy * 0.5);

            // Draw top face (brightest - teal/cyan)
            cr.MoveTo(topTL.X, topTL.Y);
            cr.LineTo(topBL.X, topBL.Y - dy);
            cr.LineTo(topTR.X, topTR.Y);
            cr.LineTo(baseCx, baseCy - cubeSize * 0.5 + dy * 0.3);
            cr.ClosePath();

            using (var gradient = new LinearGradient(topTL.X, topTL.Y, topTR.X, topTR.Y))
            {
                gradient.AddColorStop(0, new Cairo.Color(0.2, 0.8, 0.9, 0.9));   // Bright cyan
                gradient.AddColorStop(1, new Cairo.Color(0.1, 0.6, 0.7, 0.9));   // Teal
                cr.SetSource(gradient);
            }
            cr.Fill();

            // Draw left face (medium brightness - blue)
            cr.MoveTo(frontTL.X, frontTL.Y);
            cr.LineTo(topTL.X, topTL.Y);
            cr.LineTo(baseCx, baseCy - cubeSize * 0.5 + dy * 0.3);
            cr.LineTo(frontBL.X, frontBL.Y - topOffset * 0.3);
            cr.ClosePath();

            using (var gradient = new LinearGradient(frontTL.X, frontTL.Y, baseCx, baseCy))
            {
                gradient.AddColorStop(0, new Cairo.Color(0.15, 0.4, 0.7, 0.85));  // Blue
                gradient.AddColorStop(1, new Cairo.Color(0.1, 0.25, 0.5, 0.85));  // Dark blue
                cr.SetSource(gradient);
            }
            cr.Fill();

            // Draw right face (darkest - purple)
            cr.MoveTo(baseCx, baseCy - cubeSize * 0.5 + dy * 0.3);
            cr.LineTo(topTR.X, topTR.Y);
            cr.LineTo(frontTR.X, frontTR.Y);
            cr.LineTo(frontBR.X - dx * 0.5, frontBR.Y - topOffset * 0.3);
            cr.ClosePath();

            using (var gradient = new LinearGradient(baseCx, baseCy, frontTR.X, frontTR.Y))
            {
                gradient.AddColorStop(0, new Cairo.Color(0.3, 0.2, 0.5, 0.85));   // Purple
                gradient.AddColorStop(1, new Cairo.Color(0.15, 0.1, 0.3, 0.85));  // Dark purple
                cr.SetSource(gradient);
            }
            cr.Fill();

            // Draw cube edges with glow
            cr.LineWidth = s * 0.012;
            cr.SetSourceRGBA(0.5, 0.8, 1.0, 0.7);

            // Top edges
            cr.MoveTo(topTL.X, topTL.Y);
            cr.LineTo(topBL.X, topBL.Y - dy);
            cr.LineTo(topTR.X, topTR.Y);
            cr.Stroke();

            // Vertical edges
            cr.MoveTo(topTL.X, topTL.Y);
            cr.LineTo(frontTL.X, frontTL.Y);
            cr.Stroke();

            cr.MoveTo(topTR.X, topTR.Y);
            cr.LineTo(frontTR.X, frontTR.Y);
            cr.Stroke();
        }

        private static void DrawScanLines(Context cr, double cx, double cy, double s, double margin)
        {
            // Draw horizontal scan lines across the cube area (represents depth scanning)
            cr.LineWidth = s * 0.008;

            int numLines = 5;
            double startY = cy - s * 0.25;
            double endY = cy + s * 0.15;
            double lineSpacing = (endY - startY) / numLines;

            for (int i = 0; i <= numLines; i++)
            {
                double y = startY + i * lineSpacing;
                double alpha = 0.3 + 0.2 * Math.Sin(i * Math.PI / numLines); // Varying opacity

                // Gradient from cyan to magenta
                double t = (double)i / numLines;
                double r = 0.2 + 0.6 * t;
                double g = 0.8 - 0.4 * t;
                double b = 0.9;

                cr.SetSourceRGBA(r, g, b, alpha);

                // Curved scan line
                double curveAmount = s * 0.05 * Math.Sin((i + 1) * Math.PI / (numLines + 2));
                cr.MoveTo(cx - s * 0.25, y);
                cr.CurveTo(cx - s * 0.1, y - curveAmount, cx + s * 0.1, y + curveAmount, cx + s * 0.25, y);
                cr.Stroke();
            }
        }

        private static void DrawPointCloudDots(Context cr, double cx, double cy, double s, double margin)
        {
            // Draw scattered point cloud dots (represents 3D point reconstruction)
            var random = new Random(42); // Fixed seed for consistent icon

            int numPoints = 20;
            double regionRadius = s * 0.28;

            for (int i = 0; i < numPoints; i++)
            {
                double angle = random.NextDouble() * 2 * Math.PI;
                double dist = random.NextDouble() * regionRadius * 0.8;
                double px = cx + Math.Cos(angle) * dist;
                double py = cy + Math.Sin(angle) * dist - s * 0.05;

                // Skip points that would be outside the circle
                double distFromCenter = Math.Sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
                if (distFromCenter > regionRadius * 1.1) continue;

                double pointSize = s * 0.015 + random.NextDouble() * s * 0.015;
                double alpha = 0.4 + random.NextDouble() * 0.4;

                // Color varies from cyan to white
                double brightness = 0.7 + random.NextDouble() * 0.3;
                cr.SetSourceRGBA(brightness * 0.8, brightness, brightness, alpha);

                cr.Arc(px, py, pointSize, 0, 2 * Math.PI);
                cr.Fill();
            }
        }

        private static void DrawHighlight(Context cr, double cx, double cy, double s, double margin)
        {
            // Add a subtle highlight/lens flare in top-left
            double highlightX = cx - s * 0.2;
            double highlightY = cy - s * 0.2;
            double highlightRadius = s * 0.12;

            using (var gradient = new RadialGradient(highlightX, highlightY, 0, highlightX, highlightY, highlightRadius))
            {
                gradient.AddColorStop(0, new Cairo.Color(1, 1, 1, 0.25));
                gradient.AddColorStop(0.5, new Cairo.Color(0.8, 0.9, 1, 0.1));
                gradient.AddColorStop(1, new Cairo.Color(0.5, 0.7, 1, 0));

                cr.Arc(highlightX, highlightY, highlightRadius, 0, 2 * Math.PI);
                cr.SetSource(gradient);
                cr.Fill();
            }

            // Add "D3D" or stylized text hint (optional small detail)
            // Drawing a subtle "3D" using simple shapes
            double textX = cx + s * 0.08;
            double textY = cy + s * 0.28;
            double textScale = s * 0.06;

            cr.SetSourceRGBA(0.7, 0.85, 1.0, 0.6);
            cr.SelectFontFace("Sans", FontSlant.Normal, FontWeight.Bold);
            cr.SetFontSize(textScale);

            // Draw "3D" text
            cr.MoveTo(textX - textScale * 0.8, textY);
            cr.ShowText("3D");
        }

        /// <summary>
        /// Creates multiple icon sizes for use in window icon list.
        /// </summary>
        public static Pixbuf[] CreateIconSet()
        {
            return new Pixbuf[]
            {
                Create(16),
                Create(24),
                Create(32),
                Create(48),
                Create(64),
                Create(128),
                Create(256)
            };
        }
    }
}
