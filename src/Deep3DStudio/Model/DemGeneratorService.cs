using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using BitMiracle.LibTiff.Classic;
using Deep3DStudio.Scene;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public sealed class DemGrid
    {
        public int Width { get; init; }
        public int Height { get; init; }
        public double OriginX { get; init; }
        public double OriginY { get; init; }
        public double CellSize { get; init; }
        public float NoData { get; init; } = -9999f;
        public float[] Values { get; init; } = Array.Empty<float>(); // row-major Y+

        public float GetValue(int x, int y) => Values[y * Width + x];
        public void SetValue(int x, int y, float value) => Values[y * Width + x] = value;
    }

    public static class DemGeneratorService
    {
        public static bool TryCollectWorldPoints(SceneObject obj, bool applyGeoref, out List<Vector3> points, out string error)
        {
            points = new List<Vector3>();
            error = string.Empty;

            if (obj is MeshObject meshObj && meshObj.MeshData.Vertices.Count > 0)
            {
                var world = meshObj.GetWorldTransform();
                foreach (var v in meshObj.MeshData.Vertices)
                {
                    var p = GeoReferenceService.TransformPoint(v, world);
                    if (applyGeoref && GeoReferenceRuntime.HasActiveGeoreference)
                        p = GeoReferenceService.TransformModelToWorld(p);
                    points.Add(p);
                }
                return points.Count > 0;
            }

            if (obj is PointCloudObject pcObj && pcObj.Points.Count > 0)
            {
                var world = pcObj.GetWorldTransform();
                foreach (var v in pcObj.Points)
                {
                    var p = GeoReferenceService.TransformPoint(v, world);
                    if (applyGeoref && GeoReferenceRuntime.HasActiveGeoreference)
                        p = GeoReferenceService.TransformModelToWorld(p);
                    points.Add(p);
                }
                return points.Count > 0;
            }

            error = "Seleziona una Mesh o Point Cloud con punti validi.";
            return false;
        }

        public static DemGrid GenerateDemFromPoints(IReadOnlyList<Vector3> points, double cellSize, float noData = -9999f)
        {
            if (points == null || points.Count == 0)
                throw new ArgumentException("Point set vuoto.", nameof(points));
            if (cellSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(cellSize), "Cell size deve essere > 0.");

            double minX = double.MaxValue, minY = double.MaxValue;
            double maxX = double.MinValue, maxY = double.MinValue;
            foreach (var p in points)
            {
                minX = Math.Min(minX, p.X);
                minY = Math.Min(minY, p.Y);
                maxX = Math.Max(maxX, p.X);
                maxY = Math.Max(maxY, p.Y);
            }

            int width = Math.Max(1, (int)Math.Ceiling((maxX - minX) / cellSize) + 1);
            int height = Math.Max(1, (int)Math.Ceiling((maxY - minY) / cellSize) + 1);

            var sums = new double[width * height];
            var counts = new int[width * height];
            var values = new float[width * height];

            foreach (var p in points)
            {
                int ix = (int)Math.Floor((p.X - minX) / cellSize);
                int iy = (int)Math.Floor((p.Y - minY) / cellSize);
                ix = Math.Clamp(ix, 0, width - 1);
                iy = Math.Clamp(iy, 0, height - 1);
                int idx = iy * width + ix;
                sums[idx] += p.Z;
                counts[idx]++;
            }

            for (int i = 0; i < values.Length; i++)
                values[i] = counts[i] > 0 ? (float)(sums[i] / counts[i]) : noData;

            FillNoDataNearest(values, counts, width, height, noData);

            return new DemGrid
            {
                Width = width,
                Height = height,
                OriginX = minX,
                OriginY = minY,
                CellSize = cellSize,
                NoData = noData,
                Values = values
            };
        }

        public static void SaveAsciiGrid(string path, DemGrid grid)
        {
            using var writer = new StreamWriter(path);
            writer.WriteLine($"ncols {grid.Width}");
            writer.WriteLine($"nrows {grid.Height}");
            writer.WriteLine($"xllcorner {grid.OriginX.ToString("G17", CultureInfo.InvariantCulture)}");
            writer.WriteLine($"yllcorner {grid.OriginY.ToString("G17", CultureInfo.InvariantCulture)}");
            writer.WriteLine($"cellsize {grid.CellSize.ToString("G17", CultureInfo.InvariantCulture)}");
            writer.WriteLine($"NODATA_value {grid.NoData.ToString(CultureInfo.InvariantCulture)}");

            // ASCII grid expects first row as north/top row.
            for (int y = grid.Height - 1; y >= 0; y--)
            {
                for (int x = 0; x < grid.Width; x++)
                {
                    float v = grid.GetValue(x, y);
                    if (x > 0) writer.Write(' ');
                    writer.Write(v.ToString("G9", CultureInfo.InvariantCulture));
                }
                writer.WriteLine();
            }
        }

        public static void SaveGeoTiff(string path, DemGrid grid, string epsg)
        {
            using var tiff = Tiff.Open(path, "w");
            if (tiff == null)
                throw new IOException("Impossibile aprire output GeoTIFF.");

            tiff.SetField(TiffTag.IMAGEWIDTH, grid.Width);
            tiff.SetField(TiffTag.IMAGELENGTH, grid.Height);
            tiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
            tiff.SetField(TiffTag.BITSPERSAMPLE, 32);
            tiff.SetField(TiffTag.ROWSPERSTRIP, grid.Height);
            tiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
            tiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
            tiff.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.IEEEFP);
            tiff.SetField(TiffTag.COMPRESSION, Compression.LZW);
            tiff.SetField(TiffTag.FILLORDER, FillOrder.MSB2LSB);

            // GeoTIFF model tags.
            double topLeftY = grid.OriginY + grid.CellSize * grid.Height;
            tiff.SetField((TiffTag)33550, new double[] { grid.CellSize, grid.CellSize, 0.0 }); // ModelPixelScaleTag
            tiff.SetField((TiffTag)33922, new double[] { 0.0, 0.0, 0.0, grid.OriginX, topLeftY, 0.0 }); // ModelTiepointTag

            int epsgCode = ParseEpsgCode(epsg);
            bool geographic = epsgCode == 4326;
            ushort[] geoKeyDir = geographic
                ? new ushort[]
                {
                    1, 1, 0, 3,       // KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys
                    1024, 0, 1, 2,    // GTModelTypeGeoKey = Geographic
                    1025, 0, 1, 1,    // GTRasterTypeGeoKey = PixelIsArea
                    2048, 0, 1, (ushort)Math.Clamp(epsgCode, 0, ushort.MaxValue) // GeographicTypeGeoKey
                }
                : new ushort[]
                {
                    1, 1, 0, 4,
                    1024, 0, 1, 1,    // GTModelTypeGeoKey = Projected
                    1025, 0, 1, 1,    // PixelIsArea
                    3072, 0, 1, (ushort)Math.Clamp(epsgCode, 0, ushort.MaxValue), // ProjectedCSTypeGeoKey
                    3076, 0, 1, 9001  // ProjLinearUnitsGeoKey = metre
                };
            tiff.SetField((TiffTag)34735, geoKeyDir.Length, geoKeyDir); // GeoKeyDirectoryTag

            byte[] buffer = new byte[grid.Width * sizeof(float)];
            float[] row = new float[grid.Width];
            for (int rowTop = 0; rowTop < grid.Height; rowTop++)
            {
                int y = grid.Height - 1 - rowTop;
                for (int x = 0; x < grid.Width; x++)
                    row[x] = grid.GetValue(x, y);
                Buffer.BlockCopy(row, 0, buffer, 0, buffer.Length);
                tiff.WriteScanline(buffer, rowTop);
            }
            tiff.WriteDirectory();
        }

        private static void FillNoDataNearest(float[] values, int[] counts, int width, int height, float noData)
        {
            int maxRadius = 8;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;
                    if (counts[idx] > 0)
                        continue;

                    float chosen = noData;
                    bool found = false;
                    for (int r = 1; r <= maxRadius && !found; r++)
                    {
                        int minX = Math.Max(0, x - r);
                        int maxX = Math.Min(width - 1, x + r);
                        int minY = Math.Max(0, y - r);
                        int maxY = Math.Min(height - 1, y + r);
                        double bestDist = double.MaxValue;

                        for (int yy = minY; yy <= maxY; yy++)
                        {
                            for (int xx = minX; xx <= maxX; xx++)
                            {
                                int i2 = yy * width + xx;
                                if (counts[i2] <= 0)
                                    continue;
                                double dx = xx - x;
                                double dy = yy - y;
                                double d2 = dx * dx + dy * dy;
                                if (d2 < bestDist)
                                {
                                    bestDist = d2;
                                    chosen = values[i2];
                                    found = true;
                                }
                            }
                        }
                    }
                    values[idx] = chosen;
                }
            }
        }

        private static int ParseEpsgCode(string epsg)
        {
            if (string.IsNullOrWhiteSpace(epsg))
                return 0;
            string s = epsg.Trim().ToUpperInvariant();
            if (s.StartsWith("EPSG:"))
                s = s.Substring("EPSG:".Length);
            return int.TryParse(s, out int code) ? code : 0;
        }
    }
}
