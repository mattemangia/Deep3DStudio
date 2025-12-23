using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    public static class PointCloudExporter
    {
        public enum ExportFormat
        {
            PLY,
            XYZ
        }

        public static void Save(string filepath, PointCloudObject pointCloud)
        {
            string ext = Path.GetExtension(filepath).ToLower();
            ExportFormat format = ext == ".xyz" ? ExportFormat.XYZ : ExportFormat.PLY;
            Export(filepath, pointCloud, format, true);
        }

        public static void Export(string filepath, PointCloudObject pointCloud, ExportFormat format, bool includeColors)
        {
            switch (format)
            {
                case ExportFormat.PLY:
                    ExportPly(filepath, pointCloud.Points, includeColors ? pointCloud.Colors : null);
                    break;
                case ExportFormat.XYZ:
                    ExportXyz(filepath, pointCloud.Points, includeColors ? pointCloud.Colors : null);
                    break;
            }
        }

        public static void Export(string filepath, MeshData mesh, ExportFormat format, bool includeColors)
        {
            switch (format)
            {
                case ExportFormat.PLY:
                    ExportPly(filepath, mesh.Vertices, includeColors ? mesh.Colors : null);
                    break;
                case ExportFormat.XYZ:
                    ExportXyz(filepath, mesh.Vertices, includeColors ? mesh.Colors : null);
                    break;
            }
        }

        private static void ExportPly(string filepath, List<Vector3> points, List<Vector3>? colors)
        {
            using (var writer = new StreamWriter(filepath))
            {
                writer.WriteLine("ply");
                writer.WriteLine("format ascii 1.0");
                writer.WriteLine($"element vertex {points.Count}");
                writer.WriteLine("property float x");
                writer.WriteLine("property float y");
                writer.WriteLine("property float z");

                if (colors != null && colors.Count == points.Count)
                {
                    writer.WriteLine("property uchar red");
                    writer.WriteLine("property uchar green");
                    writer.WriteLine("property uchar blue");
                }

                writer.WriteLine("end_header");

                for (int i = 0; i < points.Count; i++)
                {
                    var p = points[i];
                    if (colors != null && colors.Count == points.Count)
                    {
                        var c = colors[i];
                        int r = (int)(Math.Clamp(c.X, 0, 1) * 255);
                        int g = (int)(Math.Clamp(c.Y, 0, 1) * 255);
                        int b = (int)(Math.Clamp(c.Z, 0, 1) * 255);
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2} {3} {4} {5}", p.X, p.Y, p.Z, r, g, b));
                    }
                    else
                    {
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2}", p.X, p.Y, p.Z));
                    }
                }
            }
        }

        private static void ExportXyz(string filepath, List<Vector3> points, List<Vector3>? colors)
        {
            using (var writer = new StreamWriter(filepath))
            {
                for (int i = 0; i < points.Count; i++)
                {
                    var p = points[i];
                    if (colors != null && colors.Count == points.Count)
                    {
                        var c = colors[i];
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2} {3} {4} {5}", p.X, p.Y, p.Z, c.X, c.Y, c.Z));
                    }
                    else
                    {
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} {1} {2}", p.X, p.Y, p.Z));
                    }
                }
            }
        }
    }
}
