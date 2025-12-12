using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using OpenTK.Mathematics;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;

namespace Deep3DStudio.IO
{
    public static class PointCloudImporter
    {
        public static PointCloudObject Load(string filepath)
        {
            string ext = Path.GetExtension(filepath).ToLower();
            string name = Path.GetFileNameWithoutExtension(filepath);

            var pc = new PointCloudObject(name);
            List<Vector3> points = new List<Vector3>();
            List<Vector3> colors = new List<Vector3>();

            switch (ext)
            {
                case ".xyz":
                    LoadXyz(filepath, points, colors);
                    break;
                case ".ply":
                    LoadPly(filepath, points, colors);
                    break;
                default: throw new NotSupportedException($"File format {ext} not supported for point cloud import.");
            }

            pc.Points = points;
            pc.Colors = colors;
            pc.UpdateBounds();
            return pc;
        }

        private static void LoadXyz(string filepath, List<Vector3> points, List<Vector3> colors)
        {
            foreach (var line in File.ReadLines(filepath))
            {
                var parts = line.Split(new[] { ' ', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 3) continue;

                if (float.TryParse(parts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out float x) &&
                    float.TryParse(parts[1], NumberStyles.Any, CultureInfo.InvariantCulture, out float y) &&
                    float.TryParse(parts[2], NumberStyles.Any, CultureInfo.InvariantCulture, out float z))
                {
                    points.Add(new Vector3(x, y, z));

                    if (parts.Length >= 6)
                    {
                        float r = float.Parse(parts[3], CultureInfo.InvariantCulture);
                        float g = float.Parse(parts[4], CultureInfo.InvariantCulture);
                        float b = float.Parse(parts[5], CultureInfo.InvariantCulture);

                        // Check if 0-255
                        if (r > 1.0f || g > 1.0f || b > 1.0f) { r /= 255f; g /= 255f; b /= 255f; }
                        colors.Add(new Vector3(r, g, b));
                    }
                    else
                    {
                        colors.Add(new Vector3(1, 1, 1)); // White default
                    }
                }
            }
        }

        private static void LoadPly(string filepath, List<Vector3> points, List<Vector3> colors)
        {
             var plyData = PlyParser.Parse(filepath);
             points.AddRange(plyData.Vertices);
             colors.AddRange(plyData.Colors);
        }
    }
}
