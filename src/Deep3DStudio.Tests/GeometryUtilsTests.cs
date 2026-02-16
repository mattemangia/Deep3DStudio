using Xunit;
using Deep3DStudio.Model;
using OpenTK.Mathematics;
using System.Collections.Generic;
using System;

namespace Deep3DStudio.Tests
{
    public class GeometryUtilsTests
    {
        [Fact]
        public void ComputeAlignmentQuality_EmptyMeshes_ReturnsDefault()
        {
            var source = new MeshData();
            var target = new MeshData();
            var transform = Matrix4.Identity;

            var (overlap, rmse, correspondences) = GeometryUtils.ComputeAlignmentQuality(source, target, transform);

            Assert.Equal(0, overlap);
            Assert.Equal(float.MaxValue, rmse);
            Assert.Equal(0, correspondences);
        }

        [Fact]
        public void ComputeAlignmentQuality_IdenticalMeshes_ReturnsFullOverlapAndZeroRMSE()
        {
            var vertices = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 1, 1),
                new Vector3(2, 2, 2)
            };
            var source = new MeshData { Vertices = vertices };
            var target = new MeshData { Vertices = new List<Vector3>(vertices) };
            var transform = Matrix4.Identity;

            var (overlap, rmse, correspondences) = GeometryUtils.ComputeAlignmentQuality(source, target, transform);

            Assert.Equal(1.0f, overlap);
            Assert.Equal(3, correspondences);
            Assert.Equal(0.0f, rmse);
        }

        [Fact]
        public void ComputeAlignmentQuality_DisparateMeshes_ReturnsZeroOverlap()
        {
            var source = new MeshData { Vertices = new List<Vector3> { new Vector3(0, 0, 0) } };
            var target = new MeshData { Vertices = new List<Vector3> { new Vector3(10, 10, 10) } };
            var transform = Matrix4.Identity;

            var (overlap, rmse, correspondences) = GeometryUtils.ComputeAlignmentQuality(source, target, transform);

            Assert.Equal(0.0f, overlap);
            Assert.Equal(0, correspondences);
            Assert.Equal(float.MaxValue, rmse);
        }

        [Fact]
        public void ComputeAlignmentQuality_PartialOverlap_ReturnsCorrectMetrics()
        {
            var source = new MeshData
            {
                Vertices = new List<Vector3>
                {
                    new Vector3(0, 0, 0),
                    new Vector3(10, 10, 10)
                }
            };
            var target = new MeshData
            {
                Vertices = new List<Vector3>
                {
                    new Vector3(0, 0, 0),
                    new Vector3(5, 5, 5)
                }
            };
            var transform = Matrix4.Identity;

            var (overlap, rmse, correspondences) = GeometryUtils.ComputeAlignmentQuality(source, target, transform);

            Assert.Equal(0.5f, overlap);
            Assert.Equal(1, correspondences);
            // Current implementation will have non-zero RMSE here too.
        }

        [Fact]
        public void ComputeRigidTransform_Identity_ReturnsIdentity()
        {
            var points = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0),
                new Vector3(0, 0, 1)
            };

            var transform = GeometryUtils.ComputeRigidTransform(points, points);

            AssertMatrix4ApproxEqual(Matrix4.Identity, transform);
        }

        [Fact]
        public void ComputeRigidTransform_Translation_ReturnsTranslation()
        {
            var src = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0),
                new Vector3(0, 0, 1)
            };
            var offset = new Vector3(1, 2, 3);
            var dst = new List<Vector3>();
            foreach (var p in src) dst.Add(p + offset);

            var transform = GeometryUtils.ComputeRigidTransform(src, dst);
            var expected = Matrix4.CreateTranslation(offset);

            AssertMatrix4ApproxEqual(expected, transform);
        }

        [Fact]
        public void ComputeRigidTransform_RotationZ90_ReturnsRotation()
        {
            var src = new List<Vector3>
            {
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0),
                new Vector3(0, 0, 1),
                new Vector3(0, 0, 0)
            };
            // Rotate 90 degrees around Z: (x,y,z) -> (-y, x, z)
            var dst = new List<Vector3>
            {
                new Vector3(0, 1, 0),
                new Vector3(-1, 0, 0),
                new Vector3(0, 0, 1),
                new Vector3(0, 0, 0)
            };

            var transform = GeometryUtils.ComputeRigidTransform(src, dst);
            var expected = Matrix4.CreateRotationZ(MathHelper.PiOver2);

            AssertMatrix4ApproxEqual(expected, transform);
        }

        [Fact]
        public void ComputeRigidTransform_TranslationAndRotation_ReturnsCombined()
        {
            var src = new List<Vector3>
            {
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0),
                new Vector3(0, 0, 1),
                new Vector3(0, 0, 0)
            };

            // Rotate 90 deg Z then translate (10, 20, 30)
            var rotation = Matrix4.CreateRotationZ(MathHelper.PiOver2);
            var translation = Matrix4.CreateTranslation(10, 20, 30);
            var combined = rotation * translation; // OpenTK order: first rotation then translation

            var dst = new List<Vector3>();
            foreach (var p in src)
            {
                var v = new Vector4(p, 1.0f);
                var res = v * combined;
                dst.Add(new Vector3(res.X, res.Y, res.Z));
            }

            var transform = GeometryUtils.ComputeRigidTransform(src, dst);

            AssertMatrix4ApproxEqual(combined, transform);
        }

        [Fact]
        public void ComputeRigidTransform_InsufficientPoints_ReturnsIdentity()
        {
            var src = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0)
            };
            var dst = new List<Vector3>(src);

            var transform = GeometryUtils.ComputeRigidTransform(src, dst);

            Assert.Equal(Matrix4.Identity, transform);
        }

        [Fact]
        public void ComputeRigidTransform_UnequalPoints_ReturnsIdentity()
        {
            var src = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0)
            };
            var dst = new List<Vector3>
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0)
            };

            var transform = GeometryUtils.ComputeRigidTransform(src, dst);

            Assert.Equal(Matrix4.Identity, transform);
        }

        private void AssertMatrix4ApproxEqual(Matrix4 expected, Matrix4 actual, float tolerance = 1e-4f)
        {
            Assert.True(Math.Abs(expected.M11 - actual.M11) < tolerance, $"M11 expected {expected.M11} but got {actual.M11}");
            Assert.True(Math.Abs(expected.M12 - actual.M12) < tolerance, $"M12 expected {expected.M12} but got {actual.M12}");
            Assert.True(Math.Abs(expected.M13 - actual.M13) < tolerance, $"M13 expected {expected.M13} but got {actual.M13}");
            Assert.True(Math.Abs(expected.M14 - actual.M14) < tolerance, $"M14 expected {expected.M14} but got {actual.M14}");

            Assert.True(Math.Abs(expected.M21 - actual.M21) < tolerance, $"M21 expected {expected.M21} but got {actual.M21}");
            Assert.True(Math.Abs(expected.M22 - actual.M22) < tolerance, $"M22 expected {expected.M22} but got {actual.M22}");
            Assert.True(Math.Abs(expected.M23 - actual.M23) < tolerance, $"M23 expected {expected.M23} but got {actual.M23}");
            Assert.True(Math.Abs(expected.M24 - actual.M24) < tolerance, $"M24 expected {expected.M24} but got {actual.M24}");

            Assert.True(Math.Abs(expected.M31 - actual.M31) < tolerance, $"M31 expected {expected.M31} but got {actual.M31}");
            Assert.True(Math.Abs(expected.M32 - actual.M32) < tolerance, $"M32 expected {expected.M32} but got {actual.M32}");
            Assert.True(Math.Abs(expected.M33 - actual.M33) < tolerance, $"M33 expected {expected.M33} but got {actual.M33}");
            Assert.True(Math.Abs(expected.M34 - actual.M34) < tolerance, $"M34 expected {expected.M34} but got {actual.M34}");

            Assert.True(Math.Abs(expected.M41 - actual.M41) < tolerance, $"M41 expected {expected.M41} but got {actual.M41}");
            Assert.True(Math.Abs(expected.M42 - actual.M42) < tolerance, $"M42 expected {expected.M42} but got {actual.M42}");
            Assert.True(Math.Abs(expected.M43 - actual.M43) < tolerance, $"M43 expected {expected.M43} but got {actual.M43}");
            Assert.True(Math.Abs(expected.M44 - actual.M44) < tolerance, $"M44 expected {expected.M44} but got {actual.M44}");
        }
    }
}
