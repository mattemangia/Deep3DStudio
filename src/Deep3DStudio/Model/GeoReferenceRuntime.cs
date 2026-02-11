using System;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public static class GeoReferenceRuntime
    {
        private static ProjectGeoReferenceDTO _geoReference = new ProjectGeoReferenceDTO();
        private static readonly List<GcpEntryDTO> _gcps = new List<GcpEntryDTO>();

        public static event EventHandler? Changed;

        public static ProjectGeoReferenceDTO GeoReference => _geoReference;
        public static IReadOnlyList<GcpEntryDTO> Gcps => _gcps;

        public static bool HasActiveGeoreference =>
            _geoReference.Enabled &&
            _geoReference.ModelToWorldMatrix != null &&
            _geoReference.ModelToWorldMatrix.Count >= 16;

        public static void Clear()
        {
            _geoReference = new ProjectGeoReferenceDTO();
            _gcps.Clear();
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static void LoadFromState(ProjectState state)
        {
            _geoReference = state.GeoReference ?? new ProjectGeoReferenceDTO();
            _gcps.Clear();
            if (state.Gcps != null)
                _gcps.AddRange(state.Gcps);
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static void ApplyToState(ProjectState state)
        {
            state.GeoReference = CloneGeoReference(_geoReference);
            state.Gcps = CloneGcps(_gcps);
        }

        public static void SetGeoReference(ProjectGeoReferenceDTO geoReference)
        {
            _geoReference = CloneGeoReference(geoReference);
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static void SetGcps(IEnumerable<GcpEntryDTO> gcps)
        {
            _gcps.Clear();
            _gcps.AddRange(gcps);
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static void AddOrUpdateGcp(GcpEntryDTO gcp)
        {
            int idx = _gcps.FindIndex(g => string.Equals(g.Id, gcp.Id, StringComparison.OrdinalIgnoreCase));
            if (idx >= 0)
                _gcps[idx] = gcp;
            else
                _gcps.Add(gcp);
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static void RemoveGcp(string id)
        {
            _gcps.RemoveAll(g => string.Equals(g.Id, id, StringComparison.OrdinalIgnoreCase));
            Changed?.Invoke(null, EventArgs.Empty);
        }

        public static Matrix4 GetModelToWorldMatrix()
        {
            if (_geoReference.ModelToWorldMatrix == null || _geoReference.ModelToWorldMatrix.Count < 16)
                return Matrix4.Identity;

            var d = _geoReference.ModelToWorldMatrix;
            return new Matrix4(
                d[0], d[1], d[2], d[3],
                d[4], d[5], d[6], d[7],
                d[8], d[9], d[10], d[11],
                d[12], d[13], d[14], d[15]);
        }

        public static void SetModelToWorldMatrix(Matrix4 matrix)
        {
            _geoReference.ModelToWorldMatrix = new List<float>(16)
            {
                matrix.M11, matrix.M12, matrix.M13, matrix.M14,
                matrix.M21, matrix.M22, matrix.M23, matrix.M24,
                matrix.M31, matrix.M32, matrix.M33, matrix.M34,
                matrix.M41, matrix.M42, matrix.M43, matrix.M44
            };
            _geoReference.Enabled = true;
            Changed?.Invoke(null, EventArgs.Empty);
        }

        private static ProjectGeoReferenceDTO CloneGeoReference(ProjectGeoReferenceDTO src)
        {
            return new ProjectGeoReferenceDTO
            {
                Enabled = src.Enabled,
                ProjectCrsEpsg = src.ProjectCrsEpsg,
                HorizontalUnit = src.HorizontalUnit,
                VerticalUnit = src.VerticalUnit,
                ModelToWorldMatrix = src.ModelToWorldMatrix != null
                    ? new List<float>(src.ModelToWorldMatrix)
                    : new List<float>()
            };
        }

        private static List<GcpEntryDTO> CloneGcps(IEnumerable<GcpEntryDTO> src)
        {
            var result = new List<GcpEntryDTO>();
            foreach (var g in src)
            {
                result.Add(new GcpEntryDTO
                {
                    Id = g.Id,
                    ImagePath = g.ImagePath,
                    PixelX = g.PixelX,
                    PixelY = g.PixelY,
                    InputIsLatLon = g.InputIsLatLon,
                    InputLonOrX = g.InputLonOrX,
                    InputLatOrY = g.InputLatOrY,
                    InputZ = g.InputZ,
                    ModelPoint = g.ModelPoint,
                    WorldPoint = g.WorldPoint,
                    Residual = g.Residual,
                    Enabled = g.Enabled
                });
            }
            return result;
        }
    }
}
