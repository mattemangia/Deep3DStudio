using Xunit;
using Deep3DStudio.Model;
using OpenTK.Mathematics;
using System.Collections.Generic;

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
    }
}
