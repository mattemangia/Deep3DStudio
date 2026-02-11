using System;
using System.Collections.Generic;
using Deep3DStudio.Model;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    internal static class ReconstructionPoseNormalizer
    {
        private const int MaxSamplePoints = 2048;
        private const float Epsilon = 1e-6f;
        private static readonly Matrix4 OpenCvToViewport = new Matrix4(
            1f,  0f,  0f, 0f,
            0f, -1f,  0f, 0f,
            0f,  0f, -1f, 0f,
            0f,  0f,  0f, 1f);

        /// <summary>
        /// Detects if mesh vertices are still in camera-local frame and, if so, applies mesh pose to convert to world space.
        /// Returns true when transform was applied.
        /// </summary>
        public static bool TryApplyPoseIfLikelyLocal(MeshData mesh, out float localScore, out float worldScore)
        {
            localScore = 0f;
            worldScore = 0f;

            if (mesh == null || mesh.Vertices.Count < 64 || !mesh.Pose.HasValue)
                return false;

            Matrix4 pose = mesh.Pose.Value;
            Matrix4 worldToCamera;
            try
            {
                worldToCamera = pose.Inverted();
            }
            catch
            {
                return false;
            }

            var localSample = new List<Vector3>(Math.Min(mesh.Vertices.Count, MaxSamplePoints));
            var worldSampleInCam = new List<Vector3>(Math.Min(mesh.Vertices.Count, MaxSamplePoints));
            int step = Math.Max(1, mesh.Vertices.Count / MaxSamplePoints);

            for (int i = 0; i < mesh.Vertices.Count; i += step)
            {
                var p = mesh.Vertices[i];
                if (!IsFinite(p))
                    continue;

                localSample.Add(p);
                var pCam = Vector3.TransformPosition(p, worldToCamera);
                if (IsFinite(pCam))
                    worldSampleInCam.Add(pCam);
            }

            localScore = ComputeCameraFrameScore(localSample);
            worldScore = ComputeCameraFrameScore(worldSampleInCam);

            bool shouldApply = localScore > 0.82f && localScore > worldScore + 0.18f;
            if (!shouldApply)
                return false;

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i] = Vector3.TransformPosition(mesh.Vertices[i], pose);
            }

            if (mesh.Normals.Count == mesh.Vertices.Count)
            {
                for (int i = 0; i < mesh.Normals.Count; i++)
                {
                    var n = Vector3.TransformNormal(mesh.Normals[i], pose);
                    if (n.LengthSquared > Epsilon)
                        n.Normalize();
                    mesh.Normals[i] = n;
                }
            }

            return true;
        }

        /// <summary>
        /// Converts points and pose from OpenCV camera/world convention (+Z forward, Y down)
        /// to viewport convention (-Z forward, Y up).
        /// </summary>
        public static void ConvertOpenCvConventionToViewport(MeshData mesh)
        {
            if (mesh == null)
                return;

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i] = Vector3.TransformPosition(mesh.Vertices[i], OpenCvToViewport);
            }

            if (mesh.Normals.Count == mesh.Vertices.Count)
            {
                for (int i = 0; i < mesh.Normals.Count; i++)
                {
                    var n = Vector3.TransformNormal(mesh.Normals[i], OpenCvToViewport);
                    if (n.LengthSquared > Epsilon)
                        n.Normalize();
                    mesh.Normals[i] = n;
                }
            }

            if (mesh.Pose is Matrix4 pose)
            {
                // Row-vector convention: p_world = p_cam * T.
                // Convert both camera/world frames with C => T' = C * T * C.
                mesh.Pose = OpenCvToViewport * pose * OpenCvToViewport;
            }
        }

        private static float ComputeCameraFrameScore(List<Vector3> points)
        {
            if (points == null || points.Count < 32)
                return 0f;

            int pos = 0;
            int neg = 0;
            var absZ = new List<float>(points.Count);
            var radial = new List<float>(points.Count);

            foreach (var p in points)
            {
                float z = p.Z;
                if (MathF.Abs(z) > Epsilon)
                {
                    if (z > 0) pos++;
                    else neg++;
                    absZ.Add(MathF.Abs(z));
                    radial.Add(MathF.Sqrt(p.X * p.X + p.Y * p.Y));
                }
            }

            int total = pos + neg;
            if (total < 16 || absZ.Count < 16)
                return 0f;

            float dominantZ = MathF.Max(pos, neg) / total;

            float medZ = Median(absZ);
            float medR = Median(radial);
            if (medZ <= Epsilon)
                return 0f;

            float ratio = medR / medZ;
            float shapeScore = ratio switch
            {
                < 0.03f => 0.25f,
                < 6.0f => 1.0f,
                < 12.0f => 0.7f,
                _ => 0.4f
            };

            return 0.7f * dominantZ + 0.3f * shapeScore;
        }

        private static float Median(List<float> values)
        {
            values.Sort();
            int n = values.Count;
            if (n == 0) return 0f;
            if ((n & 1) == 1) return values[n / 2];
            return 0.5f * (values[n / 2 - 1] + values[n / 2]);
        }

        private static bool IsFinite(Vector3 v)
        {
            return float.IsFinite(v.X) && float.IsFinite(v.Y) && float.IsFinite(v.Z);
        }
    }
}
