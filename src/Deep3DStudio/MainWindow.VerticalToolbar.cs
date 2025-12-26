using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private Widget CreateVerticalToolbar()
        {
            var vbox = new Box(Orientation.Vertical, 2);
            vbox.MarginStart = 2;
            vbox.MarginEnd = 2;
            vbox.MarginTop = 5;

            int btnSize = 36;

            // Transform tools section
            var moveBtn = CreateIconButton("move", "Move (W)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Translate));
            vbox.PackStart(moveBtn, false, false, 1);

            var rotateBtn = CreateIconButton("rotate", "Rotate (E)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Rotate));
            vbox.PackStart(rotateBtn, false, false, 1);

            var scaleBtn = CreateIconButton("scale", "Scale (R)", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Scale));
            vbox.PackStart(scaleBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Pen tool for triangle editing
            var penBtn = CreateIconButton("pen", "Pen Tool (P) - Edit Triangles", btnSize, () => _viewport.SetGizmoMode(GizmoMode.Pen));
            vbox.PackStart(penBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // View tools
            var focusBtn = CreateIconButton("focus", "Focus (F)", btnSize, () => _viewport.FocusOnSelection());
            vbox.PackStart(focusBtn, false, false, 1);

            var cropBtn = CreateIconButton("crop", "Toggle Crop Box", btnSize, () => _viewport.ToggleCropBox(true));
            vbox.PackStart(cropBtn, false, false, 1);

            var applyCropBtn = CreateIconButton("apply_crop", "Apply Crop", btnSize, () => _viewport.ApplyCrop());
            vbox.PackStart(applyCropBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Mesh operations
            var decimateBtn = CreateIconButton("decimate", "Decimate", btnSize, () => OnDecimateClicked(null, EventArgs.Empty));
            vbox.PackStart(decimateBtn, false, false, 1);

            var smoothBtn = CreateIconButton("smooth", "Smooth", btnSize, () => OnSmoothClicked(null, EventArgs.Empty));
            vbox.PackStart(smoothBtn, false, false, 1);

            var optimizeBtn = CreateIconButton("optimize", "Optimize", btnSize, () => OnOptimizeClicked(null, EventArgs.Empty));
            vbox.PackStart(optimizeBtn, false, false, 1);

            var splitBtn = CreateIconButton("split", "Split", btnSize, () => OnSplitClicked(null, EventArgs.Empty));
            vbox.PackStart(splitBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Merge/Align
            var mergeBtn = CreateIconButton("merge", "Merge", btnSize, () => OnMergeClicked(null, EventArgs.Empty));
            vbox.PackStart(mergeBtn, false, false, 1);

            var alignBtn = CreateIconButton("align", "Align (ICP)", btnSize, () => OnAlignClicked(null, EventArgs.Empty));
            vbox.PackStart(alignBtn, false, false, 1);

            vbox.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            var cleanupBtn = CreateIconButton("cleanup", "Cleanup Mesh", btnSize, () => OnMeshCleanupClicked(null, EventArgs.Empty));
            vbox.PackStart(cleanupBtn, false, false, 1);

            var bakeBtn = CreateIconButton("bake", "Bake Textures", btnSize, () => OnBakeTexturesClicked(null, EventArgs.Empty));
            vbox.PackStart(bakeBtn, false, false, 1);

            return vbox;
        }

        private Button CreateIconButton(string iconType, string tooltip, int size, Action onClick)
        {
            var btn = new Button();
            btn.TooltipText = tooltip;
            btn.SetSizeRequest(size, size);
            btn.Relief = ReliefStyle.None;

            // Create custom icon
            var icon = CreateCustomIcon(iconType, size - 8);
            btn.Add(icon);

            btn.Clicked += (s, e) => onClick();

            return btn;
        }

        private Widget CreateCustomIcon(string iconType, int size)
        {
            // Create a DrawingArea with custom drawing for each icon type
            var drawingArea = new DrawingArea();
            drawingArea.SetSizeRequest(size, size);
            drawingArea.Drawn += (o, args) =>
            {
                var cr = args.Cr;
                DrawIconContent(cr, iconType, size);
            };
            return drawingArea;
        }

        private void DrawIconContent(Cairo.Context cr, string iconType, int size)
        {
            double s = size;
            double cx = s / 2;
            double cy = s / 2;

            cr.SetSourceRGB(0.8, 0.8, 0.8); // Light gray for icons

            switch (iconType)
            {
                case "move":
                    // Cross with arrows
                    cr.LineWidth = 2;
                    // Horizontal line
                    cr.MoveTo(2, cy);
                    cr.LineTo(s - 2, cy);
                    cr.Stroke();
                    // Vertical line
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s - 2);
                    cr.Stroke();
                    // Arrow heads
                    cr.MoveTo(s - 2, cy);
                    cr.LineTo(s - 6, cy - 3);
                    cr.LineTo(s - 6, cy + 3);
                    cr.ClosePath();
                    cr.Fill();
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx - 3, 6);
                    cr.LineTo(cx + 3, 6);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "rotate":
                    // Circular arrow
                    cr.LineWidth = 2;
                    cr.Arc(cx, cy, s / 3, 0.5, 5.5);
                    cr.Stroke();
                    // Arrow head
                    double angle = 5.5;
                    double ax = cx + Math.Cos(angle) * s / 3;
                    double ay = cy + Math.Sin(angle) * s / 3;
                    cr.MoveTo(ax, ay);
                    cr.LineTo(ax + 4, ay - 2);
                    cr.LineTo(ax + 2, ay + 4);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "scale":
                    // Box with diagonal arrow
                    cr.LineWidth = 1.5;
                    cr.Rectangle(4, 4, s / 2, s / 2);
                    cr.Stroke();
                    cr.MoveTo(s / 2, s / 2);
                    cr.LineTo(s - 4, s - 4);
                    cr.Stroke();
                    // Arrow head
                    cr.MoveTo(s - 4, s - 4);
                    cr.LineTo(s - 8, s - 4);
                    cr.LineTo(s - 4, s - 8);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "focus":
                    // Target/crosshair
                    cr.LineWidth = 1.5;
                    cr.Arc(cx, cy, s / 3, 0, 2 * Math.PI);
                    cr.Stroke();
                    cr.Arc(cx, cy, s / 6, 0, 2 * Math.PI);
                    cr.Fill();
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s / 3);
                    cr.Stroke();
                    cr.MoveTo(cx, s - 2);
                    cr.LineTo(cx, s - s / 3);
                    cr.Stroke();
                    cr.MoveTo(2, cy);
                    cr.LineTo(s / 3, cy);
                    cr.Stroke();
                    cr.MoveTo(s - 2, cy);
                    cr.LineTo(s - s / 3, cy);
                    cr.Stroke();
                    break;

                case "crop":
                    // Crop corners
                    cr.LineWidth = 2;
                    // Top-left
                    cr.MoveTo(4, s / 3);
                    cr.LineTo(4, 4);
                    cr.LineTo(s / 3, 4);
                    cr.Stroke();
                    // Top-right
                    cr.MoveTo(s - s / 3, 4);
                    cr.LineTo(s - 4, 4);
                    cr.LineTo(s - 4, s / 3);
                    cr.Stroke();
                    // Bottom-left
                    cr.MoveTo(4, s - s / 3);
                    cr.LineTo(4, s - 4);
                    cr.LineTo(s / 3, s - 4);
                    cr.Stroke();
                    // Bottom-right
                    cr.MoveTo(s - s / 3, s - 4);
                    cr.LineTo(s - 4, s - 4);
                    cr.LineTo(s - 4, s - s / 3);
                    cr.Stroke();
                    break;

                case "apply_crop":
                    // Scissors/cut icon
                    cr.LineWidth = 2;
                    // Crop box outline
                    cr.SetSourceRGB(0.6, 0.6, 0.6);
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    // Checkmark inside
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 2.5;
                    cr.MoveTo(cx - 5, cy);
                    cr.LineTo(cx - 1, cy + 4);
                    cr.LineTo(cx + 5, cy - 4);
                    cr.Stroke();
                    break;

                case "decimate":
                    // Triangle with down arrow (simplify)
                    cr.LineWidth = 1.5;
                    cr.MoveTo(cx, 3);
                    cr.LineTo(s - 4, s - 6);
                    cr.LineTo(4, s - 6);
                    cr.ClosePath();
                    cr.Stroke();
                    // Down arrow
                    cr.SetSourceRGB(0.9, 0.5, 0.3);
                    cr.MoveTo(cx - 4, s - 10);
                    cr.LineTo(cx + 4, s - 10);
                    cr.LineTo(cx, s - 3);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "smooth":
                    // Wavy line becoming smooth
                    cr.LineWidth = 2;
                    cr.MoveTo(3, cy);
                    cr.CurveTo(s / 4, cy - 6, s / 2, cy + 6, s - 3, cy);
                    cr.Stroke();
                    break;

                case "optimize":
                    // Checkmark in circle
                    cr.LineWidth = 1.5;
                    cr.Arc(cx, cy, s / 3, 0, 2 * Math.PI);
                    cr.Stroke();
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 2;
                    cr.MoveTo(cx - 5, cy);
                    cr.LineTo(cx - 1, cy + 4);
                    cr.LineTo(cx + 5, cy - 4);
                    cr.Stroke();
                    break;

                case "split":
                    // Rectangle splitting into two
                    cr.LineWidth = 1.5;
                    cr.Rectangle(3, 4, s / 2 - 3, s - 8);
                    cr.Stroke();
                    cr.Rectangle(s / 2 + 1, 4, s / 2 - 4, s - 8);
                    cr.Stroke();
                    // Scissors/cut line
                    cr.SetSourceRGB(0.9, 0.4, 0.4);
                    cr.SetDash(new double[] { 2, 2 }, 0);
                    cr.MoveTo(cx, 2);
                    cr.LineTo(cx, s - 2);
                    cr.Stroke();
                    cr.SetDash(new double[] { }, 0);
                    break;

                case "merge":
                    // Two shapes merging
                    cr.LineWidth = 1.5;
                    cr.SetSourceRGB(0.6, 0.8, 1.0);
                    cr.Rectangle(3, 6, s / 2 - 2, s - 12);
                    cr.Fill();
                    cr.SetSourceRGB(1.0, 0.8, 0.6);
                    cr.Rectangle(s / 2 - 2, 6, s / 2 - 2, s - 12);
                    cr.Fill();
                    cr.SetSourceRGB(0.8, 0.8, 0.8);
                    cr.Rectangle(3, 6, s - 6, s - 12);
                    cr.Stroke();
                    // Arrow
                    cr.SetSourceRGB(0.3, 0.7, 0.3);
                    cr.MoveTo(s / 2 - 6, cy);
                    cr.LineTo(s / 2 + 6, cy);
                    cr.Stroke();
                    break;

                case "align":
                    // Two shapes with alignment arrows
                    cr.LineWidth = 1.5;
                    cr.Rectangle(4, 4, 8, 10);
                    cr.Stroke();
                    cr.Rectangle(s - 12, s - 14, 8, 10);
                    cr.Stroke();
                    // Alignment arrows
                    cr.SetSourceRGB(0.3, 0.6, 1.0);
                    cr.MoveTo(12, 9);
                    cr.LineTo(s - 12, s - 9);
                    cr.Stroke();
                    cr.MoveTo(s - 12, s - 9);
                    cr.LineTo(s - 16, s - 6);
                    cr.Stroke();
                    cr.MoveTo(s - 12, s - 9);
                    cr.LineTo(s - 9, s - 13);
                    cr.Stroke();
                    break;

                case "cleanup":
                    // Broom icon
                    cr.LineWidth = 2;
                    // Handle
                    cr.SetSourceRGB(0.6, 0.4, 0.2);
                    cr.MoveTo(s - 4, 4);
                    cr.LineTo(s / 2, s / 2);
                    cr.Stroke();
                    // Bristles
                    cr.SetSourceRGB(0.9, 0.8, 0.4);
                    cr.MoveTo(s / 2, s / 2);
                    cr.LineTo(4, s - 8);
                    cr.LineTo(8, s - 4);
                    cr.ClosePath();
                    cr.Fill();
                    break;

                case "bake":
                    // Texture/image icon
                    cr.LineWidth = 1.5;
                    cr.SetSourceRGB(0.8, 0.8, 0.8);
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    // Mountains/Sun
                    cr.MoveTo(4, s - 8);
                    cr.LineTo(s / 3, s / 2);
                    cr.LineTo(s / 2, s - 6);
                    cr.LineTo(2 * s / 3, s / 3);
                    cr.LineTo(s - 4, s - 8);
                    cr.Stroke();
                    cr.Arc(s - 8, 8, 2, 0, 2 * Math.PI);
                    cr.Fill();
                    break;

                case "camera":
                    // Camera body
                    cr.LineWidth = 1.5;
                    cr.SetSourceRGB(0.8, 0.8, 0.8);
                    cr.Rectangle(4, 8, s - 8, s - 12);
                    cr.Stroke();
                    // Lens
                    cr.Arc(cx, cy + 2, s / 4, 0, 2 * Math.PI);
                    cr.Stroke();
                    // Viewfinder/Flash bump
                    cr.MoveTo(cx - 3, 8);
                    cr.LineTo(cx - 2, 5);
                    cr.LineTo(cx + 2, 5);
                    cr.LineTo(cx + 3, 8);
                    cr.Stroke();
                    break;

                case "pen":
                    // Pen/pencil icon for triangle editing
                    cr.LineWidth = 2;
                    cr.SetSourceRGB(1.0, 0.6, 0.2); // Orange color
                    // Pen body (diagonal)
                    cr.MoveTo(s - 4, 4);
                    cr.LineTo(6, s - 6);
                    cr.Stroke();
                    // Pen tip
                    cr.SetSourceRGB(0.4, 0.4, 0.4);
                    cr.MoveTo(6, s - 6);
                    cr.LineTo(4, s - 4);
                    cr.LineTo(8, s - 8);
                    cr.ClosePath();
                    cr.Fill();
                    // Triangle indicator
                    cr.SetSourceRGB(0.3, 0.8, 0.3);
                    cr.LineWidth = 1.5;
                    cr.MoveTo(s - 8, s / 2);
                    cr.LineTo(s - 4, s - 4);
                    cr.LineTo(s / 2, s - 4);
                    cr.ClosePath();
                    cr.Stroke();
                    break;

                default:
                    // Default: simple square
                    cr.Rectangle(4, 4, s - 8, s - 8);
                    cr.Stroke();
                    break;
            }
        }
    }
}
