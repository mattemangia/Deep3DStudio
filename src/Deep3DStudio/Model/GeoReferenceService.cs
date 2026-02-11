using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Deep3DStudio.Scene;
using MathNet.Numerics.LinearAlgebra;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public static class GeoReferenceService
    {
        private const double Wgs84A = 6378137.0;
        private const double Wgs84F = 1.0 / 298.257223563;
        private const double K0 = 0.9996;

        public static Vector3 TransformModelToWorld(Vector3 pointModel)
        {
            return TransformPoint(pointModel, GeoReferenceRuntime.GetModelToWorldMatrix());
        }

        public static Vector3 TransformWorldToModel(Vector3 pointWorld)
        {
            var inverse = GeoReferenceRuntime.GetModelToWorldMatrix().Inverted();
            return TransformPoint(pointWorld, inverse);
        }

        public static Vector3 TransformPoint(Vector3 point, Matrix4 transform)
        {
            var v = new Vector4(point, 1f);
            var w = v * transform;
            if (Math.Abs(w.W) < 1e-8f)
                return new Vector3(w.X, w.Y, w.Z);
            return new Vector3(w.X / w.W, w.Y / w.W, w.Z / w.W);
        }

        public static bool TryNormalizeInputCoordinate(
            string projectEpsg,
            bool inputIsLatLon,
            double lonOrX,
            double latOrY,
            double z,
            out Vector3 worldPoint,
            out string error)
        {
            error = string.Empty;
            worldPoint = Vector3.Zero;

            if (!inputIsLatLon)
            {
                worldPoint = new Vector3((float)lonOrX, (float)latOrY, (float)z);
                return true;
            }

            if (double.IsNaN(latOrY) || double.IsNaN(lonOrX) ||
                latOrY < -90 || latOrY > 90 || lonOrX < -180 || lonOrX > 180)
            {
                error = "Lat/Lon non validi.";
                return false;
            }

            int epsgCode = ParseEpsgCode(projectEpsg);
            if (epsgCode == 0)
            {
                error = "EPSG non valido.";
                return false;
            }

            if (epsgCode == 4326)
            {
                worldPoint = new Vector3((float)lonOrX, (float)latOrY, (float)z);
                return true;
            }

            if (epsgCode >= 32601 && epsgCode <= 32660)
            {
                int zone = epsgCode - 32600;
                LatLonToUtm(latOrY, lonOrX, zone, true, out double easting, out double northing);
                worldPoint = new Vector3((float)easting, (float)northing, (float)z);
                return true;
            }

            if (epsgCode >= 32701 && epsgCode <= 32760)
            {
                int zone = epsgCode - 32700;
                LatLonToUtm(latOrY, lonOrX, zone, false, out double easting, out double northing);
                worldPoint = new Vector3((float)easting, (float)northing, (float)z);
                return true;
            }

            error = $"Supporto lat/lon non disponibile per {projectEpsg}. Usa EPSG:4326 o UTM WGS84 (326xx/327xx).";
            return false;
        }

        public static bool TrySolveModelToWorldFromGcps(
            IEnumerable<GcpEntryDTO> gcps,
            out Matrix4 modelToWorld,
            out double rms,
            out string error)
        {
            modelToWorld = Matrix4.Identity;
            rms = 0;
            error = string.Empty;

            var valid = gcps
                .Where(g => g.Enabled)
                .Select(g => (model: g.ModelPoint, world: g.WorldPoint))
                .ToList();

            if (valid.Count < 3)
            {
                error = "Servono almeno 3 GCP validi.";
                return false;
            }

            var src = valid.Select(v => new Vector3d(v.model.X, v.model.Y, v.model.Z)).ToList();
            var dst = valid.Select(v => new Vector3d(v.world.X, v.world.Y, v.world.Z)).ToList();

            if (!TryComputeSimilarityTransform(src, dst, out modelToWorld))
            {
                error = "Impossibile calcolare la trasformazione dai GCP.";
                return false;
            }

            UpdateResiduals(gcps, modelToWorld, out rms);
            return true;
        }

        public static void UpdateResiduals(IEnumerable<GcpEntryDTO> gcps, Matrix4 modelToWorld, out double rms)
        {
            double sumSq = 0;
            int count = 0;
            foreach (var g in gcps)
            {
                if (!g.Enabled)
                {
                    g.Residual = 0;
                    continue;
                }

                var transformed = TransformPoint(g.ModelPoint, modelToWorld);
                double dx = transformed.X - g.WorldPoint.X;
                double dy = transformed.Y - g.WorldPoint.Y;
                double dz = transformed.Z - g.WorldPoint.Z;
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);
                g.Residual = (float)dist;
                sumSq += dist * dist;
                count++;
            }

            rms = count > 0 ? Math.Sqrt(sumSq / count) : 0.0;
        }

        public static bool TryPickModelPointFromImagePixel(
            SceneGraph sceneGraph,
            string imagePath,
            float pixelX,
            float pixelY,
            out Vector3 modelPoint,
            out string error)
        {
            modelPoint = Vector3.Zero;
            error = string.Empty;

            var camera = FindCameraByImage(sceneGraph, imagePath);
            if (camera?.Pose == null)
            {
                error = "Camera pose non trovata per l'immagine selezionata.";
                return false;
            }

            if (camera.Pose.Width <= 0 || camera.Pose.Height <= 0)
            {
                error = "Dimensioni immagine camera non valide.";
                return false;
            }

            float fx = camera.Pose.GetEffectiveFocalLength();
            if (fx <= 0)
            {
                error = "Focale non valida per la camera.";
                return false;
            }

            float cx = camera.Pose.Width * 0.5f;
            float cy = camera.Pose.Height * 0.5f;
            float nx = (pixelX - cx) / fx;
            float ny = (cy - pixelY) / fx;

            var rayDirCam = new Vector3(nx, ny, -1f).Normalized();
            var rayDirWorld4 = new Vector4(rayDirCam, 0f) * camera.Pose.CameraToWorld;
            var rayDirWorld = new Vector3(rayDirWorld4.X, rayDirWorld4.Y, rayDirWorld4.Z).Normalized();
            var rayOriginWorld = camera.Pose.CameraToWorld.ExtractTranslation();

            bool hit = TryRaycastScene(sceneGraph, rayOriginWorld, rayDirWorld, out modelPoint);
            if (!hit)
            {
                error = "Nessuna intersezione trovata sulla scena (mesh/point cloud).";
                return false;
            }

            return true;
        }

        public static string FormatResidualStats(IEnumerable<GcpEntryDTO> gcps)
        {
            var residuals = gcps.Where(g => g.Enabled).Select(g => (double)g.Residual).ToList();
            if (residuals.Count == 0)
                return "Nessun GCP attivo";

            double rms = Math.Sqrt(residuals.Sum(v => v * v) / residuals.Count);
            double max = residuals.Max();
            double min = residuals.Min();
            double avg = residuals.Average();
            return string.Create(CultureInfo.InvariantCulture, $"RMS: {rms:F4} | AVG: {avg:F4} | MIN: {min:F4} | MAX: {max:F4}");
        }

        private static CameraObject? FindCameraByImage(SceneGraph sceneGraph, string imagePath)
        {
            string normalized = NormalizePath(imagePath);
            var byFullPath = sceneGraph.GetObjectsOfType<CameraObject>()
                .FirstOrDefault(c => c.Pose != null && NormalizePath(c.ImagePath) == normalized);
            if (byFullPath != null)
                return byFullPath;

            string fileName = System.IO.Path.GetFileName(imagePath);
            return sceneGraph.GetObjectsOfType<CameraObject>()
                .FirstOrDefault(c => c.Pose != null &&
                                     string.Equals(System.IO.Path.GetFileName(c.ImagePath), fileName, StringComparison.OrdinalIgnoreCase));
        }

        private static string NormalizePath(string path)
        {
            try
            {
                return System.IO.Path.GetFullPath(path).Replace('\\', '/').ToLowerInvariant();
            }
            catch
            {
                return (path ?? string.Empty).Replace('\\', '/').ToLowerInvariant();
            }
        }

        private static bool TryRaycastScene(SceneGraph sceneGraph, Vector3 rayOrigin, Vector3 rayDir, out Vector3 hitPoint)
        {
            hitPoint = Vector3.Zero;
            float bestDistance = float.MaxValue;
            bool found = false;

            IEnumerable<SceneObject> targets = sceneGraph.SelectedObjects
                .Where(o => o is MeshObject || o is PointCloudObject);
            if (!targets.Any())
                targets = sceneGraph.GetVisibleObjects().Where(o => o is MeshObject || o is PointCloudObject);

            foreach (var obj in targets)
            {
                if (obj is MeshObject meshObj && meshObj.MeshData.Indices.Count >= 3)
                {
                    if (TryRaycastMesh(meshObj, rayOrigin, rayDir, out var p, out float t) && t < bestDistance)
                    {
                        bestDistance = t;
                        hitPoint = p;
                        found = true;
                    }
                }
                else if (obj is PointCloudObject pcObj && pcObj.Points.Count > 0)
                {
                    if (TryRaycastPointCloud(pcObj, rayOrigin, rayDir, out var p, out float t) && t < bestDistance)
                    {
                        bestDistance = t;
                        hitPoint = p;
                        found = true;
                    }
                }
            }

            return found;
        }

        private static bool TryRaycastMesh(MeshObject meshObj, Vector3 rayOrigin, Vector3 rayDir, out Vector3 hitPoint, out float hitDistance)
        {
            hitPoint = Vector3.Zero;
            hitDistance = float.MaxValue;

            var mesh = meshObj.MeshData;
            var world = meshObj.GetWorldTransform();

            bool found = false;
            for (int i = 0; i + 2 < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];
                if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= mesh.Vertices.Count || i1 >= mesh.Vertices.Count || i2 >= mesh.Vertices.Count)
                    continue;

                var v0 = TransformPoint(mesh.Vertices[i0], world);
                var v1 = TransformPoint(mesh.Vertices[i1], world);
                var v2 = TransformPoint(mesh.Vertices[i2], world);

                if (RayIntersectsTriangle(rayOrigin, rayDir, v0, v1, v2, out float t) && t < hitDistance)
                {
                    hitDistance = t;
                    hitPoint = rayOrigin + rayDir * t;
                    found = true;
                }
            }

            return found;
        }

        private static bool TryRaycastPointCloud(PointCloudObject pcObj, Vector3 rayOrigin, Vector3 rayDir, out Vector3 hitPoint, out float hitDistance)
        {
            hitPoint = Vector3.Zero;
            hitDistance = float.MaxValue;
            bool found = false;

            var world = pcObj.GetWorldTransform();
            float maxPerpendicular = 0.03f;

            foreach (var pLocal in pcObj.Points)
            {
                var p = TransformPoint(pLocal, world);
                var v = p - rayOrigin;
                float t = Vector3.Dot(v, rayDir);
                if (t <= 0)
                    continue;

                var closest = rayOrigin + rayDir * t;
                float d = (p - closest).Length;
                if (d <= maxPerpendicular && t < hitDistance)
                {
                    hitDistance = t;
                    hitPoint = p;
                    found = true;
                }
            }

            return found;
        }

        private static bool RayIntersectsTriangle(
            Vector3 rayOrigin,
            Vector3 rayVector,
            Vector3 vertex0,
            Vector3 vertex1,
            Vector3 vertex2,
            out float t)
        {
            t = 0f;
            const float epsilon = 1e-6f;
            Vector3 edge1 = vertex1 - vertex0;
            Vector3 edge2 = vertex2 - vertex0;
            Vector3 h = Vector3.Cross(rayVector, edge2);
            float a = Vector3.Dot(edge1, h);
            if (a > -epsilon && a < epsilon)
                return false;

            float f = 1f / a;
            Vector3 s = rayOrigin - vertex0;
            float u = f * Vector3.Dot(s, h);
            if (u < 0f || u > 1f)
                return false;

            Vector3 q = Vector3.Cross(s, edge1);
            float v = f * Vector3.Dot(rayVector, q);
            if (v < 0f || u + v > 1f)
                return false;

            t = f * Vector3.Dot(edge2, q);
            return t > epsilon;
        }

        private static bool TryComputeSimilarityTransform(
            IReadOnlyList<Vector3d> sourcePoints,
            IReadOnlyList<Vector3d> targetPoints,
            out Matrix4 transform)
        {
            transform = Matrix4.Identity;
            int n = sourcePoints.Count;
            if (n < 3 || targetPoints.Count != n)
                return false;

            Vector<double> srcMean = Vector<double>.Build.Dense(3);
            Vector<double> dstMean = Vector<double>.Build.Dense(3);
            for (int i = 0; i < n; i++)
            {
                srcMean[0] += sourcePoints[i].X;
                srcMean[1] += sourcePoints[i].Y;
                srcMean[2] += sourcePoints[i].Z;
                dstMean[0] += targetPoints[i].X;
                dstMean[1] += targetPoints[i].Y;
                dstMean[2] += targetPoints[i].Z;
            }
            srcMean /= n;
            dstMean /= n;

            Matrix<double> covariance = Matrix<double>.Build.Dense(3, 3);
            double srcVariance = 0.0;
            for (int i = 0; i < n; i++)
            {
                var src = Vector<double>.Build.Dense(new[]
                {
                    sourcePoints[i].X - srcMean[0],
                    sourcePoints[i].Y - srcMean[1],
                    sourcePoints[i].Z - srcMean[2]
                });
                var dst = Vector<double>.Build.Dense(new[]
                {
                    targetPoints[i].X - dstMean[0],
                    targetPoints[i].Y - dstMean[1],
                    targetPoints[i].Z - dstMean[2]
                });
                covariance += dst.OuterProduct(src);
                srcVariance += src.DotProduct(src);
            }

            covariance /= n;
            srcVariance /= n;
            if (srcVariance < 1e-12)
                return false;

            var svd = covariance.Svd(true);
            var u = svd.U;
            var vt = svd.VT;
            var v = vt.Transpose();

            var s = Matrix<double>.Build.DenseIdentity(3);
            if (u.Determinant() * v.Determinant() < 0)
                s[2, 2] = -1;

            var r = u * s * vt;
            var singular = svd.S;
            double trace = singular[0] * s[0, 0] + singular[1] * s[1, 1] + singular[2] * s[2, 2];
            double scale = trace / srcVariance;

            double tx = dstMean[0] - scale * (r[0, 0] * srcMean[0] + r[0, 1] * srcMean[1] + r[0, 2] * srcMean[2]);
            double ty = dstMean[1] - scale * (r[1, 0] * srcMean[0] + r[1, 1] * srcMean[1] + r[1, 2] * srcMean[2]);
            double tz = dstMean[2] - scale * (r[2, 0] * srcMean[0] + r[2, 1] * srcMean[1] + r[2, 2] * srcMean[2]);

            transform = Matrix4.Identity;
            transform.M11 = (float)(scale * r[0, 0]); transform.M12 = (float)(scale * r[1, 0]); transform.M13 = (float)(scale * r[2, 0]);
            transform.M21 = (float)(scale * r[0, 1]); transform.M22 = (float)(scale * r[1, 1]); transform.M23 = (float)(scale * r[2, 1]);
            transform.M31 = (float)(scale * r[0, 2]); transform.M32 = (float)(scale * r[1, 2]); transform.M33 = (float)(scale * r[2, 2]);
            transform.M41 = (float)tx;
            transform.M42 = (float)ty;
            transform.M43 = (float)tz;
            transform.M44 = 1f;
            return true;
        }

        private static int ParseEpsgCode(string epsg)
        {
            if (string.IsNullOrWhiteSpace(epsg))
                return 0;

            string trimmed = epsg.Trim().ToUpperInvariant();
            if (trimmed.StartsWith("EPSG:"))
                trimmed = trimmed.Substring("EPSG:".Length);
            return int.TryParse(trimmed, out int code) ? code : 0;
        }

        private static void LatLonToUtm(double latDeg, double lonDeg, int zone, bool northernHemisphere, out double easting, out double northing)
        {
            double lat = MathHelper.DegreesToRadians((float)latDeg);
            double lon = MathHelper.DegreesToRadians((float)lonDeg);
            double lon0 = MathHelper.DegreesToRadians((float)(zone * 6 - 183));

            double e2 = Wgs84F * (2.0 - Wgs84F);
            double ep2 = e2 / (1.0 - e2);
            double sinLat = Math.Sin(lat);
            double cosLat = Math.Cos(lat);
            double tanLat = Math.Tan(lat);

            double n = Wgs84A / Math.Sqrt(1.0 - e2 * sinLat * sinLat);
            double t = tanLat * tanLat;
            double c = ep2 * cosLat * cosLat;
            double a = (lon - lon0) * cosLat;

            double m = Wgs84A * (
                (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * Math.Pow(e2, 3) / 256) * lat
                - (3 * e2 / 8 + 3 * e2 * e2 / 32 + 45 * Math.Pow(e2, 3) / 1024) * Math.Sin(2 * lat)
                + (15 * e2 * e2 / 256 + 45 * Math.Pow(e2, 3) / 1024) * Math.Sin(4 * lat)
                - (35 * Math.Pow(e2, 3) / 3072) * Math.Sin(6 * lat));

            easting = K0 * n * (a + (1 - t + c) * Math.Pow(a, 3) / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * Math.Pow(a, 5) / 120) + 500000.0;

            northing = K0 * (m + n * tanLat * (
                a * a / 2
                + (5 - t + 9 * c + 4 * c * c) * Math.Pow(a, 4) / 24
                + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * Math.Pow(a, 6) / 720));

            if (!northernHemisphere)
                northing += 10000000.0;
        }
    }
}
