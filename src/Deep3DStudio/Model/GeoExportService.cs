using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Deep3DStudio.Scene;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class GeoExportOptions
    {
        public bool ApplyGeoreferenceIfAvailable { get; set; } = true;
        public bool WriteGeoSidecar { get; set; } = true;
        public string? TargetEpsg { get; set; }
    }

    public static class GeoExportService
    {
        public static MeshData PrepareMeshForExport(MeshObject meshObj, GeoExportOptions? options = null)
        {
            options ??= new GeoExportOptions();
            var result = meshObj.MeshData.Clone();
            var world = meshObj.GetWorldTransform();
            bool applyGeo = options.ApplyGeoreferenceIfAvailable && GeoReferenceRuntime.HasActiveGeoreference;

            for (int i = 0; i < result.Vertices.Count; i++)
            {
                var p = GeoReferenceService.TransformPoint(result.Vertices[i], world);
                if (applyGeo)
                    p = GeoReferenceService.TransformModelToWorld(p);
                result.Vertices[i] = p;
            }

            if (result.Normals.Count == result.Vertices.Count)
                result.RecalculateNormals();

            return result;
        }

        public static PointCloudObject PreparePointCloudForExport(PointCloudObject pcObj, GeoExportOptions? options = null)
        {
            options ??= new GeoExportOptions();
            var result = new PointCloudObject(pcObj.Name)
            {
                PointSize = pcObj.PointSize,
                Colors = new List<Vector3>(pcObj.Colors),
                Normals = new List<Vector3>(pcObj.Normals),
                Points = new List<Vector3>(pcObj.Points.Count)
            };

            bool applyGeo = options.ApplyGeoreferenceIfAvailable && GeoReferenceRuntime.HasActiveGeoreference;
            var world = pcObj.GetWorldTransform();
            foreach (var pLocal in pcObj.Points)
            {
                var p = GeoReferenceService.TransformPoint(pLocal, world);
                if (applyGeo)
                    p = GeoReferenceService.TransformModelToWorld(p);
                result.Points.Add(p);
            }
            result.UpdateBounds();
            return result;
        }

        public static void TryWriteGeoSidecar(string exportPath, IEnumerable<Vector3> points, GeoExportOptions? options = null)
        {
            options ??= new GeoExportOptions();
            if (!options.WriteGeoSidecar)
                return;
            if (!GeoReferenceRuntime.HasActiveGeoreference)
                return;

            Vector3 min = new Vector3(float.MaxValue);
            Vector3 max = new Vector3(float.MinValue);
            int count = 0;
            foreach (var p in points)
            {
                min = Vector3.ComponentMin(min, p);
                max = Vector3.ComponentMax(max, p);
                count++;
            }
            if (count == 0)
                return;

            var sidecar = new
            {
                epsg = string.IsNullOrWhiteSpace(options.TargetEpsg)
                    ? GeoReferenceRuntime.GeoReference.ProjectCrsEpsg
                    : options.TargetEpsg,
                georeferenced = true,
                pointCount = count,
                boundsMin = new[] { min.X, min.Y, min.Z },
                boundsMax = new[] { max.X, max.Y, max.Z },
                modelToWorldMatrix = GeoReferenceRuntime.GeoReference.ModelToWorldMatrix
            };

            string sidecarPath = exportPath + ".geo.json";
            File.WriteAllText(sidecarPath, JsonSerializer.Serialize(sidecar, new JsonSerializerOptions
            {
                WriteIndented = true
            }));
        }
    }
}
