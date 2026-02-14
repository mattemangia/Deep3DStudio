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
                        case "confidence": DrawConfidenceIcon(cr, size); break;
                        case "select": DrawSelectIcon(cr, size); break;
                        case "texture": DrawTextureIcon(cr, size); break;
                        case "camera": DrawCameraIcon(cr, size); break;
                        case "rig": DrawRigIcon(cr, size); break;
                        case "refine": DrawRefineIcon(cr, size); break;
                        case "link": DrawLinkIcon(cr, size); break;
                        case "ai_single": DrawAISingleIcon(cr, size); break;
                        case "ai_gauss": DrawAIGaussIcon(cr, size); break;
                        case "ai_multi": DrawAIMultiIcon(cr, size); break;
                        case "nerf": DrawNeRFIcon(cr, size); break;
                        case "decimate": DrawDecimateIcon(cr, size); break;
                        case "smooth": DrawSmoothIcon(cr, size); break;
                        case "optimize": DrawOptimizeIcon(cr, size); break;
                        case "merge": DrawMergeIcon(cr, size); break;
                        case "merge_meshes": DrawMergeMeshesIcon(cr, size); break;
                        case "merge_pointclouds": DrawMergePointCloudsIcon(cr, size); break;
                        case "align": DrawAlignIcon(cr, size); break;
                        case "plane": DrawPlaneIcon(cr, size); break;
                        case "cube": DrawCubeIcon(cr, size); break;
                        case "sphere": DrawSphereIcon(cr, size); break;
                        case "cylinder": DrawCylinderIcon(cr, size); break;
                        case "cone": DrawConeIcon(cr, size); break;
                        case "torus": DrawTorusIcon(cr, size); break;
                        case "circle": DrawCircleIcon(cr, size); break;
                        case "polygon": DrawPolygonIcon(cr, size); break;
                        case "grid_mesh": DrawGridMeshIcon(cr, size); break;
                        case "vertex_move": DrawVertexMoveIcon(cr, size); break;
                        case "extrude": DrawExtrudeIcon(cr, size); break;
                        case "inset": DrawInsetIcon(cr, size); break;
                        case "bridge": DrawBridgeIcon(cr, size); break;
                        case "voxel_filter": DrawVoxelFilterIcon(cr, size); break;
                        case "outlier_filter": DrawOutlierFilterIcon(cr, size); break;
                        case "duplicate_filter": DrawDuplicateFilterIcon(cr, size); break;
                        case "normals": DrawNormalsIcon(cr, size); break;
                        case "axis_filter": DrawAxisFilterIcon(cr, size); break;
                        case "radius_crop": DrawRadiusCropIcon(cr, size); break;
                        case "dense_cloud": DrawDenseCloudIcon(cr, size); break;
                        case "georef": DrawGeorefIcon(cr, size); break;
                        case "residuals": DrawResidualsIcon(cr, size); break;
                        case "dem": DrawDemIcon(cr, size); break;
                        case "geo_export": DrawGeoExportIcon(cr, size); break;
                        case "fill_holes": DrawFillHolesIcon(cr, size); break;
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
                var (r, g, b) = Model.ImageUtils.TurboColormap((float)t);

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

        private static void DrawConfidenceIcon(Context cr, int size)
        {
            // Vertical confidence gauge with turbo gradient and check marker.
            double barWidth = size * 0.28;
            double barHeight = size * 0.66;
            double x = size * 0.2;
            double y = size * 0.17;

            int steps = 10;
            double stepHeight = barHeight / steps;
            for (int i = 0; i < steps; i++)
            {
                double t = (double)i / (steps - 1);
                var (r, g, b) = Model.ImageUtils.TurboColormap((float)(1.0 - t));
                cr.SetSourceRGB(r, g, b);
                cr.Rectangle(x, y + i * stepHeight, barWidth, stepHeight + 1);
                cr.Fill();
            }

            cr.SetSourceRGB(0.25, 0.25, 0.25);
            cr.LineWidth = 1;
            cr.Rectangle(x, y, barWidth, barHeight);
            cr.Stroke();

            cr.SetSourceRGB(0.95, 0.95, 0.95);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.55, size * 0.55);
            cr.LineTo(size * 0.68, size * 0.68);
            cr.LineTo(size * 0.84, size * 0.36);
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

        private static void DrawCameraIcon(Context cr, int size)
        {
            // Camera frustum icon - represents camera viewing cone
            cr.SetSourceRGB(1.0, 0.8, 0.0); // Yellow-orange (matches frustum color in viewport)
            cr.LineWidth = 2;

            // Camera body (small rectangle at top-left representing camera position)
            double camX = size * 0.2;
            double camY = size * 0.25;
            double camW = size * 0.2;
            double camH = size * 0.15;

            cr.Rectangle(camX, camY, camW, camH);
            cr.Fill();

            // Lens circle
            cr.Arc(camX + camW + size * 0.05, camY + camH / 2, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();

            // Frustum lines (viewing cone radiating from camera)
            double frustumStartX = camX + camW + size * 0.1;
            double frustumStartY = camY + camH / 2;
            double frustumEndX = size * 0.85;

            cr.SetSourceRGB(1.0, 0.8, 0.0);
            cr.LineWidth = 1.5;

            // Top frustum line
            cr.MoveTo(frustumStartX, frustumStartY);
            cr.LineTo(frustumEndX, size * 0.15);
            cr.Stroke();

            // Bottom frustum line
            cr.MoveTo(frustumStartX, frustumStartY);
            cr.LineTo(frustumEndX, size * 0.85);
            cr.Stroke();

            // Far plane (vertical line at end of frustum)
            cr.MoveTo(frustumEndX, size * 0.15);
            cr.LineTo(frustumEndX, size * 0.85);
            cr.Stroke();

            // Optional: Add a small viewfinder on top of camera
            cr.SetSourceRGB(0.8, 0.6, 0.0);
            cr.Rectangle(camX + camW * 0.3, camY - size * 0.08, camW * 0.4, size * 0.08);
            cr.Fill();
        }

        private static void DrawRigIcon(Context cr, int size)
        {
            // Simple bone + joint representation
            cr.SetSourceRGB(0.2, 0.6, 0.9);
            cr.LineWidth = 3;

            cr.MoveTo(size * 0.25, size * 0.2);
            cr.LineTo(size * 0.5, size * 0.5);
            cr.LineTo(size * 0.75, size * 0.2);
            cr.Stroke();

            cr.SetSourceRGB(0.1, 0.3, 0.6);
            cr.Arc(size * 0.5, size * 0.5, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawRefineIcon(Context cr, int size)
        {
            // Magic-wand style icon for refinement
            cr.SetSourceRGB(0.9, 0.7, 0.1);
            cr.LineWidth = 3;
            cr.MoveTo(size * 0.25, size * 0.75);
            cr.LineTo(size * 0.75, size * 0.25);
            cr.Stroke();

            cr.SetSourceRGB(1.0, 0.9, 0.4);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.55, size * 0.2);
            cr.LineTo(size * 0.6, size * 0.05);
            cr.LineTo(size * 0.65, size * 0.2);
            cr.ClosePath();
            cr.Fill();

            cr.MoveTo(size * 0.8, size * 0.4);
            cr.LineTo(size * 0.95, size * 0.45);
            cr.LineTo(size * 0.8, size * 0.5);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawLinkIcon(Context cr, int size)
        {
            // Chain link icon for auto workflow toggle
            cr.SetSourceRGB(0.2, 0.8, 0.2);
            cr.LineWidth = 3;

            // Left chain link (oval)
            double ovalW = size * 0.35;
            double ovalH = size * 0.4;

            cr.Save();
            cr.Translate(size * 0.3, size * 0.5);
            cr.Scale(ovalW / 2, ovalH / 2);
            cr.Arc(0, 0, 1, 0, 2 * Math.PI);
            cr.Restore();
            cr.Stroke();

            // Right chain link (oval)
            cr.Save();
            cr.Translate(size * 0.7, size * 0.5);
            cr.Scale(ovalW / 2, ovalH / 2);
            cr.Arc(0, 0, 1, 0, 2 * Math.PI);
            cr.Restore();
            cr.Stroke();
        }

        private static void DrawAISingleIcon(Context cr, int size)
        {
            // TripoSR icon - single image to 3D
            cr.SetSourceRGB(0.9, 0.2, 0.6);
            cr.LineWidth = 2;

            // Image frame
            cr.Rectangle(size * 0.1, size * 0.25, size * 0.35, size * 0.5);
            cr.Stroke();

            // Arrow
            cr.MoveTo(size * 0.5, size * 0.5);
            cr.LineTo(size * 0.7, size * 0.5);
            cr.Stroke();

            // 3D cube outline
            cr.Rectangle(size * 0.7, size * 0.3, size * 0.2, size * 0.4);
            cr.Stroke();
        }

        private static void DrawAIGaussIcon(Context cr, int size)
        {
            // LGM icon - Large Gaussian Model (multiple dots)
            cr.SetSourceRGB(0.6, 0.2, 0.8);

            // Multiple gaussian dots
            cr.Arc(size * 0.3, size * 0.3, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.5, size * 0.5, size * 0.12, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.7, size * 0.35, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.4, size * 0.7, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.65, size * 0.65, size * 0.09, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawAIMultiIcon(Context cr, int size)
        {
            // Wonder3D icon - multiple views
            cr.SetSourceRGB(0.0, 0.6, 0.6);
            cr.LineWidth = 2;

            // Multiple overlapping image frames
            cr.Rectangle(size * 0.15, size * 0.35, size * 0.25, size * 0.35);
            cr.Stroke();
            cr.Rectangle(size * 0.3, size * 0.25, size * 0.25, size * 0.35);
            cr.Stroke();
            cr.Rectangle(size * 0.45, size * 0.35, size * 0.25, size * 0.35);
            cr.Stroke();
        }

        private static void DrawNeRFIcon(Context cr, int size)
        {
            // NeRF icon - neural radiance field (network representation)
            cr.SetSourceRGB(1.0, 0.3, 0.1);

            // Input nodes
            cr.Arc(size * 0.2, size * 0.3, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.2, size * 0.7, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();

            // Middle node
            cr.Arc(size * 0.5, size * 0.5, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();

            // Output node
            cr.Arc(size * 0.8, size * 0.5, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();

            // Connections
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.28, size * 0.32);
            cr.LineTo(size * 0.42, size * 0.48);
            cr.Stroke();
            cr.MoveTo(size * 0.28, size * 0.68);
            cr.LineTo(size * 0.42, size * 0.52);
            cr.Stroke();
            cr.MoveTo(size * 0.58, size * 0.5);
            cr.LineTo(size * 0.72, size * 0.5);
            cr.Stroke();
        }

        private static void DrawPlaneIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 1.0);
            cr.Rectangle(size * 0.15, size * 0.3, size * 0.7, size * 0.4);
            cr.FillPreserve();
            cr.SetSourceRGB(0.2, 0.35, 0.5);
            cr.LineWidth = 1.5;
            cr.Stroke();
        }

        private static void DrawCubeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.95, 0.78, 0.45);
            cr.Rectangle(size * 0.2, size * 0.25, size * 0.5, size * 0.5);
            cr.FillPreserve();
            cr.SetSourceRGB(0.45, 0.3, 0.15);
            cr.LineWidth = 1.5;
            cr.Stroke();
            cr.MoveTo(size * 0.2, size * 0.25);
            cr.LineTo(size * 0.35, size * 0.12);
            cr.LineTo(size * 0.85, size * 0.12);
            cr.LineTo(size * 0.7, size * 0.25);
            cr.ClosePath();
            cr.Stroke();
        }

        private static void DrawSphereIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.6, 0.85, 1.0);
            cr.Arc(size * 0.5, size * 0.5, size * 0.3, 0, 2 * Math.PI);
            cr.FillPreserve();
            cr.SetSourceRGB(0.2, 0.45, 0.65);
            cr.LineWidth = 1.5;
            cr.Stroke();
        }

        private static void DrawCylinderIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.9, 0.65);
            cr.Rectangle(size * 0.25, size * 0.22, size * 0.5, size * 0.56);
            cr.FillPreserve();
            cr.SetSourceRGB(0.3, 0.45, 0.2);
            cr.LineWidth = 1.5;
            cr.Stroke();
            cr.Arc(size * 0.5, size * 0.22, size * 0.25, Math.PI, 2 * Math.PI);
            cr.Stroke();
            cr.Arc(size * 0.5, size * 0.78, size * 0.25, 0, Math.PI);
            cr.Stroke();
        }

        private static void DrawConeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.95, 0.7, 0.5);
            cr.MoveTo(size * 0.5, size * 0.15);
            cr.LineTo(size * 0.18, size * 0.78);
            cr.LineTo(size * 0.82, size * 0.78);
            cr.ClosePath();
            cr.FillPreserve();
            cr.SetSourceRGB(0.45, 0.2, 0.1);
            cr.LineWidth = 1.5;
            cr.Stroke();
        }

        private static void DrawTorusIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.95, 0.75);
            cr.Arc(size * 0.5, size * 0.5, size * 0.3, 0, 2 * Math.PI);
            cr.FillPreserve();
            cr.SetSourceRGB(0.2, 0.5, 0.2);
            cr.LineWidth = 1.5;
            cr.Stroke();
            cr.SetSourceRGBA(0, 0, 0, 0);
        }

        private static void DrawCircleIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.8, 1.0);
            cr.Arc(size * 0.5, size * 0.5, size * 0.28, 0, 2 * Math.PI);
            cr.FillPreserve();
            cr.SetSourceRGB(0.25, 0.25, 0.5);
            cr.LineWidth = 1.5;
            cr.Stroke();
        }

        private static void DrawPolygonIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.85, 0.75, 0.95);
            for (int i = 0; i < 6; i++)
            {
                double angle = (Math.PI * 2.0 * i / 6.0) - Math.PI / 2.0;
                double x = size * 0.5 + Math.Cos(angle) * size * 0.3;
                double y = size * 0.5 + Math.Sin(angle) * size * 0.3;
                if (i == 0) cr.MoveTo(x, y);
                else cr.LineTo(x, y);
            }
            cr.ClosePath();
            cr.FillPreserve();
            cr.SetSourceRGB(0.3, 0.2, 0.45);
            cr.LineWidth = 1.5;
            cr.Stroke();
        }

        private static void DrawGridMeshIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 0.95);
            cr.Rectangle(size * 0.15, size * 0.15, size * 0.7, size * 0.7);
            cr.Stroke();
            cr.LineWidth = 1.0;
            for (int i = 1; i < 4; i++)
            {
                double p = size * (0.15 + 0.7 * i / 4.0);
                cr.MoveTo(p, size * 0.15);
                cr.LineTo(p, size * 0.85);
                cr.Stroke();
                cr.MoveTo(size * 0.15, p);
                cr.LineTo(size * 0.85, p);
                cr.Stroke();
            }
        }

        private static void DrawVertexMoveIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.85, 0.9, 1.0);
            cr.Arc(size * 0.35, size * 0.65, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.65, size * 0.65, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.5, size * 0.35, size * 0.08, 0, 2 * Math.PI);
            cr.Fill();
            cr.SetSourceRGB(0.2, 0.65, 0.95);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.5, size * 0.15);
            cr.LineTo(size * 0.5, size * 0.45);
            cr.Stroke();
            cr.MoveTo(size * 0.5, size * 0.15);
            cr.LineTo(size * 0.45, size * 0.22);
            cr.LineTo(size * 0.55, size * 0.22);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawExtrudeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.95, 0.85, 0.55);
            cr.Rectangle(size * 0.2, size * 0.45, size * 0.4, size * 0.3);
            cr.FillPreserve();
            cr.SetSourceRGB(0.45, 0.35, 0.1);
            cr.Stroke();
            cr.SetSourceRGB(0.2, 0.75, 0.25);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.7, size * 0.7);
            cr.LineTo(size * 0.7, size * 0.22);
            cr.Stroke();
            cr.MoveTo(size * 0.7, size * 0.22);
            cr.LineTo(size * 0.64, size * 0.3);
            cr.LineTo(size * 0.76, size * 0.3);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawInsetIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 1.0);
            cr.Rectangle(size * 0.15, size * 0.15, size * 0.7, size * 0.7);
            cr.Stroke();
            cr.SetSourceRGB(0.2, 0.6, 0.95);
            cr.Rectangle(size * 0.3, size * 0.3, size * 0.4, size * 0.4);
            cr.Stroke();
            cr.MoveTo(size * 0.22, size * 0.22);
            cr.LineTo(size * 0.3, size * 0.3);
            cr.Stroke();
        }

        private static void DrawBridgeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.9, 0.8, 0.7);
            cr.Rectangle(size * 0.12, size * 0.3, size * 0.22, size * 0.4);
            cr.FillPreserve();
            cr.Stroke();
            cr.Rectangle(size * 0.66, size * 0.3, size * 0.22, size * 0.4);
            cr.FillPreserve();
            cr.Stroke();
            cr.SetSourceRGB(0.25, 0.7, 0.95);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.34, size * 0.5);
            cr.LineTo(size * 0.66, size * 0.5);
            cr.Stroke();
        }

        private static void DrawVoxelFilterIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.6, 0.9, 0.9);
            cr.LineWidth = 1.2;
            for (int r = 0; r < 3; r++)
            {
                for (int c = 0; c < 3; c++)
                {
                    cr.Rectangle(size * (0.2 + c * 0.18), size * (0.2 + r * 0.18), size * 0.14, size * 0.14);
                    cr.Stroke();
                }
            }
        }

        private static void DrawOutlierFilterIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.9, 1.0);
            for (int i = 0; i < 8; i++)
            {
                double angle = i * Math.PI * 2 / 8;
                cr.Arc(size * (0.5 + Math.Cos(angle) * 0.22), size * (0.5 + Math.Sin(angle) * 0.22), 2, 0, 2 * Math.PI);
                cr.Fill();
            }
            cr.SetSourceRGB(0.95, 0.3, 0.3);
            cr.MoveTo(size * 0.75, size * 0.2);
            cr.LineTo(size * 0.9, size * 0.35);
            cr.Stroke();
        }

        private static void DrawDuplicateFilterIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.85, 0.85, 1.0);
            cr.Arc(size * 0.4, size * 0.5, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.6, size * 0.5, size * 0.1, 0, 2 * Math.PI);
            cr.Fill();
            cr.SetSourceRGB(0.2, 0.75, 0.3);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.5, size * 0.2);
            cr.LineTo(size * 0.5, size * 0.8);
            cr.Stroke();
        }

        private static void DrawNormalsIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.9, 0.8);
            cr.Arc(size * 0.35, size * 0.65, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.65, size * 0.65, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.5, size * 0.38, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.SetSourceRGB(0.25, 0.75, 0.95);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.5, size * 0.65);
            cr.LineTo(size * 0.5, size * 0.2);
            cr.Stroke();
        }

        private static void DrawAxisFilterIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.85, 1.0);
            cr.Arc(size * 0.25, size * 0.5, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.5, size * 0.5, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.75, size * 0.5, 2.5, 0, 2 * Math.PI); cr.Fill();
            cr.SetSourceRGB(0.95, 0.75, 0.2);
            cr.Rectangle(size * 0.35, size * 0.35, size * 0.3, size * 0.3);
            cr.Stroke();
        }

        private static void DrawRadiusCropIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.85, 0.9, 1.0);
            cr.Arc(size * 0.5, size * 0.5, size * 0.3, 0, 2 * Math.PI);
            cr.Stroke();
            cr.SetSourceRGB(0.2, 0.7, 0.95);
            cr.Arc(size * 0.5, size * 0.5, size * 0.14, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawDenseCloudIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.25, 0.78, 1.0);
            double r = size * 0.06;

            cr.Arc(size * 0.24, size * 0.28, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.52, size * 0.22, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.72, size * 0.44, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.32, size * 0.68, r, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.68, size * 0.72, r, 0, 2 * Math.PI); cr.Fill();

            cr.SetSourceRGB(0.9, 0.95, 1.0);
            cr.Arc(size * 0.38, size * 0.30, r * 0.8, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.60, size * 0.33, r * 0.8, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.48, size * 0.52, r * 0.8, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.56, size * 0.62, r * 0.8, 0, 2 * Math.PI); cr.Fill();

            cr.SetSourceRGB(0.95, 0.85, 0.2);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.12, size * 0.82);
            cr.LineTo(size * 0.12, size * 0.54);
            cr.Stroke();
            cr.MoveTo(size * 0.12, size * 0.46);
            cr.LineTo(size * 0.06, size * 0.56);
            cr.LineTo(size * 0.18, size * 0.56);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawGeorefIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.9, 1.0);
            cr.Arc(size * 0.35, size * 0.4, size * 0.12, 0, 2 * Math.PI); cr.Fill();
            cr.Arc(size * 0.65, size * 0.4, size * 0.12, 0, 2 * Math.PI); cr.Fill();
            cr.SetSourceRGB(0.25, 0.7, 0.95);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.35, size * 0.52);
            cr.LineTo(size * 0.35, size * 0.85);
            cr.Stroke();
            cr.MoveTo(size * 0.65, size * 0.52);
            cr.LineTo(size * 0.65, size * 0.85);
            cr.Stroke();
        }

        private static void DrawResidualsIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.9, 1.0);
            cr.Rectangle(size * 0.15, size * 0.2, size * 0.7, size * 0.6);
            cr.Stroke();
            cr.SetSourceRGB(0.2, 0.8, 0.3);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.2, size * 0.65);
            cr.LineTo(size * 0.45, size * 0.45);
            cr.LineTo(size * 0.8, size * 0.3);
            cr.Stroke();
        }

        private static void DrawDemIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.8, 0.9, 0.75);
            cr.Rectangle(size * 0.15, size * 0.2, size * 0.7, size * 0.6);
            cr.Stroke();
            for (int i = 0; i < 4; i++)
            {
                double y = size * (0.25 + i * 0.12);
                cr.MoveTo(size * 0.2, y);
                cr.LineTo(size * 0.8, y);
                cr.Stroke();
            }
            cr.SetSourceRGB(0.1, 0.6, 0.2);
            cr.MoveTo(size * 0.2, size * 0.7);
            cr.LineTo(size * 0.45, size * 0.48);
            cr.LineTo(size * 0.65, size * 0.56);
            cr.LineTo(size * 0.8, size * 0.42);
            cr.Stroke();
        }

        private static void DrawGeoExportIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 1.0);
            cr.Rectangle(size * 0.15, size * 0.2, size * 0.45, size * 0.55);
            cr.Stroke();
            cr.SetSourceRGB(0.2, 0.75, 0.25);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.62, size * 0.55);
            cr.LineTo(size * 0.86, size * 0.55);
            cr.Stroke();
            cr.MoveTo(size * 0.86, size * 0.55);
            cr.LineTo(size * 0.78, size * 0.47);
            cr.LineTo(size * 0.78, size * 0.63);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawDecimateIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.95, 0.7, 0.35);
            cr.MoveTo(size * 0.5, size * 0.15);
            cr.LineTo(size * 0.85, size * 0.8);
            cr.LineTo(size * 0.15, size * 0.8);
            cr.ClosePath();
            cr.Stroke();
            cr.MoveTo(size * 0.5, size * 0.85);
            cr.LineTo(size * 0.4, size * 0.65);
            cr.LineTo(size * 0.6, size * 0.65);
            cr.ClosePath();
            cr.Fill();
        }

        private static void DrawSmoothIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.65, 0.9, 0.7);
            cr.LineWidth = 2;
            cr.MoveTo(size * 0.1, size * 0.55);
            cr.CurveTo(size * 0.3, size * 0.2, size * 0.7, size * 0.9, size * 0.9, size * 0.45);
            cr.Stroke();
        }

        private static void DrawOptimizeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.7, 0.95, 0.75);
            cr.Arc(size * 0.5, size * 0.5, size * 0.3, 0, 2 * Math.PI);
            cr.Stroke();
            cr.MoveTo(size * 0.32, size * 0.52);
            cr.LineTo(size * 0.45, size * 0.64);
            cr.LineTo(size * 0.7, size * 0.38);
            cr.Stroke();
        }

        private static void DrawMergeIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 1.0);
            cr.Rectangle(size * 0.14, size * 0.25, size * 0.34, size * 0.5);
            cr.Stroke();
            cr.Rectangle(size * 0.52, size * 0.25, size * 0.34, size * 0.5);
            cr.Stroke();
            cr.MoveTo(size * 0.45, size * 0.5);
            cr.LineTo(size * 0.55, size * 0.5);
            cr.Stroke();
        }

        private static void DrawFillHolesIcon(Context cr, int size)
        {
            // Draw a polygon with a 'hole' being filled
            cr.SetSourceRGB(0.7, 0.7, 0.7);
            cr.LineWidth = 1.5;

            // Main mesh part 1
            cr.MoveTo(size * 0.15, size * 0.15);
            cr.LineTo(size * 0.5, size * 0.35);
            cr.LineTo(size * 0.15, size * 0.6);
            cr.ClosePath();
            cr.Stroke();

            // Main mesh part 2
            cr.MoveTo(size * 0.85, size * 0.15);
            cr.LineTo(size * 0.5, size * 0.35);
            cr.LineTo(size * 0.85, size * 0.6);
            cr.ClosePath();
            cr.Stroke();

            // The 'Hole' being filled (highlighted triangle in the middle)
            cr.SetSourceRGB(0.3, 0.7, 1.0); // Bright blue for filling
            cr.MoveTo(size * 0.15, size * 0.6);
            cr.LineTo(size * 0.5, size * 0.35);
            cr.LineTo(size * 0.85, size * 0.6);
            cr.ClosePath();
            cr.FillPreserve();
            cr.SetSourceRGB(0.2, 0.5, 0.8);
            cr.Stroke();

            // Bottom base
            cr.SetSourceRGB(0.7, 0.7, 0.7);
            cr.MoveTo(size * 0.15, size * 0.6);
            cr.LineTo(size * 0.5, size * 0.85);
            cr.LineTo(size * 0.85, size * 0.6);
            cr.Stroke();
        }

        private static void DrawMergeMeshesIcon(Context cr, int size)
        {
            DrawMergeIcon(cr, size);

            // Overlay a small wireframe triangle to indicate mesh merge.
            cr.SetSourceRGB(0.95, 0.78, 0.25);
            cr.MoveTo(size * 0.32, size * 0.68);
            cr.LineTo(size * 0.5, size * 0.38);
            cr.LineTo(size * 0.68, size * 0.68);
            cr.ClosePath();
            cr.Stroke();
        }

        private static void DrawMergePointCloudsIcon(Context cr, int size)
        {
            DrawMergeIcon(cr, size);

            // Overlay point samples to indicate point-cloud merge.
            cr.SetSourceRGB(0.45, 0.9, 0.95);
            cr.Arc(size * 0.3, size * 0.35, size * 0.04, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.5, size * 0.58, size * 0.04, 0, 2 * Math.PI);
            cr.Fill();
            cr.Arc(size * 0.7, size * 0.35, size * 0.04, 0, 2 * Math.PI);
            cr.Fill();
        }

        private static void DrawAlignIcon(Context cr, int size)
        {
            cr.SetSourceRGB(0.75, 0.85, 1.0);
            cr.Rectangle(size * 0.15, size * 0.2, size * 0.28, size * 0.6);
            cr.Stroke();
            cr.Rectangle(size * 0.57, size * 0.2, size * 0.28, size * 0.6);
            cr.Stroke();
            cr.MoveTo(size * 0.43, size * 0.3);
            cr.LineTo(size * 0.57, size * 0.3);
            cr.Stroke();
            cr.MoveTo(size * 0.43, size * 0.7);
            cr.LineTo(size * 0.57, size * 0.7);
            cr.Stroke();
        }

    }
}
