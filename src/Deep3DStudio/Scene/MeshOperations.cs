using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    /// <summary>
    /// Provides advanced mesh operations like decimation, optimization, splitting, merging, and alignment
    /// </summary>
    public static class MeshOperations
    {
        #region Decimation

        /// <summary>
        /// Decimates a mesh to reach a target vertex count using adaptive voxel clustering.
        /// This approach is topologically robust and avoids holes.
        /// </summary>
        /// <param name="mesh">Input mesh</param>
        /// <param name="targetRatio">Target vertex ratio (0.1 = keep 10% of vertices)</param>
        /// <returns>Decimated mesh</returns>
        public static MeshData Decimate(MeshData mesh, float targetRatio)
        {
            if (mesh.Vertices.Count < 10 || targetRatio >= 1.0f)
                return CloneMesh(mesh);

            int targetCount = Math.Max(4, (int)(mesh.Vertices.Count * targetRatio));

            // Calculate mesh bounds to determine voxel size search range
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var v in mesh.Vertices)
            {
                min = Vector3.ComponentMin(min, v);
                max = Vector3.ComponentMax(max, v);
            }

            float maxDim = Math.Max(max.X - min.X, Math.Max(max.Y - min.Y, max.Z - min.Z));

            // Binary search for optimal voxel size
            float low = 0.0f;
            float high = maxDim; // Max possible voxel size (one vertex)
            float bestSize = 0.0f;
            int bestDiff = int.MaxValue;

            for (int i = 0; i < 15; i++)
            {
                float mid = (low + high) * 0.5f;
                if (mid < 0.00001f) mid = 0.00001f;

                int count = CountVerticesUniform(mesh, mid);

                int diff = Math.Abs(count - targetCount);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    bestSize = mid;
                }

                if (count > targetCount)
                {
                    // Too many vertices -> Need larger voxels
                    low = mid;
                }
                else
                {
                    // Too few vertices -> Need smaller voxels
                    high = mid;
                }
            }

            return DecimateUniform(mesh, bestSize);
        }

        private static int CountVerticesUniform(MeshData mesh, float voxelSize)
        {
             var voxelSet = new HashSet<(int, int, int)>();
             foreach (var v in mesh.Vertices)
             {
                 var key = (
                     (int)Math.Floor(v.X / voxelSize),
                     (int)Math.Floor(v.Y / voxelSize),
                     (int)Math.Floor(v.Z / voxelSize)
                 );
                 voxelSet.Add(key);
             }
             return voxelSet.Count;
        }

        /// <summary>
        /// Decimates by removing vertices based on spatial clustering
        /// </summary>
        public static MeshData DecimateUniform(MeshData mesh, float voxelSize)
        {
            if (mesh.Vertices.Count < 10)
                return CloneMesh(mesh);

            // Spatial hashing to cluster nearby vertices
            var voxelMap = new Dictionary<(int, int, int), List<int>>();

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                var key = (
                    (int)Math.Floor(v.X / voxelSize),
                    (int)Math.Floor(v.Y / voxelSize),
                    (int)Math.Floor(v.Z / voxelSize)
                );

                if (!voxelMap.ContainsKey(key))
                    voxelMap[key] = new List<int>();
                voxelMap[key].Add(i);
            }

            // Keep one vertex per voxel (centroid of cluster)
            var newVertices = new List<Vector3>();
            var newColors = new List<Vector3>();
            var oldToNew = new int[mesh.Vertices.Count];
            Array.Fill(oldToNew, -1);

            foreach (var kvp in voxelMap)
            {
                var indices = kvp.Value;
                var centroid = Vector3.Zero;
                var color = Vector3.Zero;

                foreach (int idx in indices)
                {
                    centroid += mesh.Vertices[idx];
                    color += mesh.Colors[idx];
                }

                centroid /= indices.Count;
                color /= indices.Count;

                int newIdx = newVertices.Count;
                newVertices.Add(centroid);
                newColors.Add(color);

                foreach (int idx in indices)
                    oldToNew[idx] = newIdx;
            }

            // Rebuild triangles
            var newIndices = new List<int>();
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = oldToNew[mesh.Indices[i]];
                int i1 = oldToNew[mesh.Indices[i + 1]];
                int i2 = oldToNew[mesh.Indices[i + 2]];

                // Skip degenerate triangles
                if (i0 != i1 && i1 != i2 && i2 != i0)
                {
                    newIndices.Add(i0);
                    newIndices.Add(i1);
                    newIndices.Add(i2);
                }
            }

            return new MeshData
            {
                Vertices = newVertices,
                Colors = newColors,
                Indices = newIndices
            };
        }

        private static float[] ComputeVertexImportance(MeshData mesh)
        {
            var importance = new float[mesh.Vertices.Count];

            // Count triangle participation
            var triangleCount = new int[mesh.Vertices.Count];
            for (int i = 0; i < mesh.Indices.Count; i++)
            {
                triangleCount[mesh.Indices[i]]++;
            }

            // Compute curvature estimate based on normal variation
            var normals = ComputeVertexNormals(mesh);

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                // More triangles = more important
                importance[i] = triangleCount[i];

                // Add curvature estimate (high curvature = important)
                // Using magnitude of normal as proxy for curvature
                importance[i] += normals[i].Length * 10f;
            }

            return importance;
        }

        #endregion

        #region Optimization

        /// <summary>
        /// Optimizes mesh by removing duplicate vertices and degenerate triangles
        /// </summary>
        public static MeshData Optimize(MeshData mesh)
        {
            const float epsilon = 0.0001f;

            // Spatial hash for duplicate detection
            var uniqueVertices = new List<Vector3>();
            var uniqueColors = new List<Vector3>();
            var oldToNew = new int[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                int foundIdx = -1;

                // Linear search for now (could use spatial hash for large meshes)
                for (int j = 0; j < uniqueVertices.Count; j++)
                {
                    if ((uniqueVertices[j] - v).LengthSquared < epsilon * epsilon)
                    {
                        foundIdx = j;
                        break;
                    }
                }

                if (foundIdx >= 0)
                {
                    oldToNew[i] = foundIdx;
                }
                else
                {
                    oldToNew[i] = uniqueVertices.Count;
                    uniqueVertices.Add(v);
                    uniqueColors.Add(mesh.Colors[i]);
                }
            }

            // Rebuild triangles, removing degenerates
            var newIndices = new List<int>();
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = oldToNew[mesh.Indices[i]];
                int i1 = oldToNew[mesh.Indices[i + 1]];
                int i2 = oldToNew[mesh.Indices[i + 2]];

                if (i0 != i1 && i1 != i2 && i2 != i0)
                {
                    var v0 = uniqueVertices[i0];
                    var v1 = uniqueVertices[i1];
                    var v2 = uniqueVertices[i2];

                    // Check for degenerate triangle (zero area)
                    var edge1 = v1 - v0;
                    var edge2 = v2 - v0;
                    var cross = Vector3.Cross(edge1, edge2);

                    if (cross.LengthSquared > epsilon * epsilon)
                    {
                        newIndices.Add(i0);
                        newIndices.Add(i1);
                        newIndices.Add(i2);
                    }
                }
            }

            return new MeshData
            {
                Vertices = uniqueVertices,
                Colors = uniqueColors,
                Indices = newIndices
            };
        }

        /// <summary>
        /// Removes isolated vertices that don't belong to any triangle
        /// </summary>
        public static MeshData RemoveIsolatedVertices(MeshData mesh)
        {
            var usedVertices = new bool[mesh.Vertices.Count];

            foreach (int idx in mesh.Indices)
            {
                usedVertices[idx] = true;
            }

            return RemapMesh(mesh, usedVertices);
        }

        #endregion

        #region Smoothing

        /// <summary>
        /// Applies Laplacian smoothing to the mesh
        /// </summary>
        public static MeshData Smooth(MeshData mesh, int iterations = 1, float lambda = 0.5f)
        {
            var result = CloneMesh(mesh);

            // Build adjacency
            var adjacency = BuildVertexAdjacency(result);

            for (int iter = 0; iter < iterations; iter++)
            {
                var newPositions = new Vector3[result.Vertices.Count];

                for (int i = 0; i < result.Vertices.Count; i++)
                {
                    if (adjacency[i].Count == 0)
                    {
                        newPositions[i] = result.Vertices[i];
                        continue;
                    }

                    // Compute centroid of neighbors
                    var centroid = Vector3.Zero;
                    foreach (int neighbor in adjacency[i])
                    {
                        centroid += result.Vertices[neighbor];
                    }
                    centroid /= adjacency[i].Count;

                    // Move vertex towards centroid
                    newPositions[i] = Vector3.Lerp(result.Vertices[i], centroid, lambda);
                }

                for (int i = 0; i < result.Vertices.Count; i++)
                {
                    result.Vertices[i] = newPositions[i];
                }
            }

            return result;
        }

        /// <summary>
        /// Applies Taubin smoothing (reduces shrinkage)
        /// </summary>
        public static MeshData SmoothTaubin(MeshData mesh, int iterations = 1, float lambda = 0.5f, float mu = -0.53f)
        {
            var result = CloneMesh(mesh);
            var adjacency = BuildVertexAdjacency(result);

            for (int iter = 0; iter < iterations; iter++)
            {
                // Shrink step
                ApplyLaplacianStep(result, adjacency, lambda);
                // Expand step
                ApplyLaplacianStep(result, adjacency, mu);
            }

            return result;
        }

        private static void ApplyLaplacianStep(MeshData mesh, List<HashSet<int>> adjacency, float factor)
        {
            var newPositions = new Vector3[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                if (adjacency[i].Count == 0)
                {
                    newPositions[i] = mesh.Vertices[i];
                    continue;
                }

                var centroid = Vector3.Zero;
                foreach (int neighbor in adjacency[i])
                {
                    centroid += mesh.Vertices[neighbor];
                }
                centroid /= adjacency[i].Count;

                newPositions[i] = mesh.Vertices[i] + factor * (centroid - mesh.Vertices[i]);
            }

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                mesh.Vertices[i] = newPositions[i];
            }
        }

        #endregion

        #region Splitting

        /// <summary>
        /// Splits mesh into connected components
        /// </summary>
        public static List<MeshData> SplitByConnectivity(MeshData mesh)
        {
            var results = new List<MeshData>();
            var visited = new bool[mesh.Vertices.Count];
            var adjacency = BuildVertexAdjacency(mesh);

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                if (visited[i]) continue;

                // BFS to find connected component
                var component = new HashSet<int>();
                var queue = new Queue<int>();
                queue.Enqueue(i);

                while (queue.Count > 0)
                {
                    int v = queue.Dequeue();
                    if (visited[v]) continue;

                    visited[v] = true;
                    component.Add(v);

                    foreach (int neighbor in adjacency[v])
                    {
                        if (!visited[neighbor])
                            queue.Enqueue(neighbor);
                    }
                }

                // Extract component mesh
                if (component.Count > 0)
                {
                    var componentMesh = ExtractSubMesh(mesh, component);
                    if (componentMesh.Vertices.Count > 0)
                        results.Add(componentMesh);
                }
            }

            return results;
        }

        /// <summary>
        /// Splits mesh by a plane
        /// </summary>
        public static (MeshData above, MeshData below) SplitByPlane(MeshData mesh, Vector3 planePoint, Vector3 planeNormal)
        {
            planeNormal = planeNormal.Normalized();

            var aboveVertices = new HashSet<int>();
            var belowVertices = new HashSet<int>();

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                float dist = Vector3.Dot(v - planePoint, planeNormal);

                if (dist >= 0)
                    aboveVertices.Add(i);
                else
                    belowVertices.Add(i);
            }

            return (
                ExtractSubMesh(mesh, aboveVertices),
                ExtractSubMesh(mesh, belowVertices)
            );
        }

        /// <summary>
        /// Splits mesh by bounding box
        /// </summary>
        public static (MeshData inside, MeshData outside) SplitByBounds(MeshData mesh, Vector3 min, Vector3 max)
        {
            var inside = new HashSet<int>();
            var outside = new HashSet<int>();

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                if (v.X >= min.X && v.X <= max.X &&
                    v.Y >= min.Y && v.Y <= max.Y &&
                    v.Z >= min.Z && v.Z <= max.Z)
                {
                    inside.Add(i);
                }
                else
                {
                    outside.Add(i);
                }
            }

            return (
                ExtractSubMesh(mesh, inside),
                ExtractSubMesh(mesh, outside)
            );
        }

        #endregion

        #region Merging

        /// <summary>
        /// Merges multiple meshes into one
        /// </summary>
        public static MeshData Merge(IEnumerable<MeshData> meshes)
        {
            var result = new MeshData();
            int vertexOffset = 0;

            foreach (var mesh in meshes)
            {
                // Add vertices and colors
                result.Vertices.AddRange(mesh.Vertices);
                result.Colors.AddRange(mesh.Colors);

                // Add indices with offset
                foreach (int idx in mesh.Indices)
                {
                    result.Indices.Add(idx + vertexOffset);
                }

                vertexOffset += mesh.Vertices.Count;
            }

            return result;
        }

        /// <summary>
        /// Merges point clouds into one
        /// </summary>
        public static PointCloudObject MergePointClouds(IEnumerable<PointCloudObject> pointClouds)
        {
            var result = new PointCloudObject("Merged Point Cloud");

            foreach (var pc in pointClouds)
            {
                var transform = pc.GetWorldTransform();

                for (int i = 0; i < pc.Points.Count; i++)
                {
                    var worldPoint = Vector3.TransformPosition(pc.Points[i], transform);
                    result.Points.Add(worldPoint);
                    result.Colors.Add(pc.Colors[i]);
                }
            }

            result.UpdateBounds();
            return result;
        }

        /// <summary>
        /// Merges meshes with duplicate vertex removal at seams
        /// </summary>
        public static MeshData MergeWithWelding(IEnumerable<MeshData> meshes, float weldDistance = 0.001f)
        {
            var merged = Merge(meshes);
            return Optimize(merged);
        }

        #endregion

        #region Alignment / ICP

        /// <summary>
        /// Aligns source mesh to target using ICP (Iterative Closest Point)
        /// </summary>
        public static Matrix4 AlignICP(MeshData source, MeshData target, int maxIterations = 50, float convergenceThreshold = 0.0001f)
        {
            return AlignICP(source.Vertices, target.Vertices, maxIterations, convergenceThreshold);
        }

        /// <summary>
        /// Aligns source point cloud to target using ICP (Iterative Closest Point)
        /// </summary>
        public static Matrix4 AlignICP(List<Vector3> sourceVertices, List<Vector3> targetVertices, int maxIterations = 50, float convergenceThreshold = 0.0001f)
        {
            var transform = Matrix4.Identity;

            // Sample points from source (use fewer for speed)
            var sourcePoints = SamplePoints(sourceVertices, Math.Min(1000, sourceVertices.Count));

            // Build KD-tree equivalent (spatial hash) for target
            var targetHash = BuildSpatialHash(targetVertices, 0.1f);

            float prevError = float.MaxValue;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Transform source points
                var transformedSource = new List<Vector3>();
                foreach (var p in sourcePoints)
                {
                    transformedSource.Add(Vector3.TransformPosition(p, transform));
                }

                // Find correspondences (closest points)
                var srcPts = new List<Vector3>();
                var dstPts = new List<Vector3>();

                foreach (var p in transformedSource)
                {
                    var closest = FindClosestPoint(p, targetVertices, targetHash, 0.1f);
                    if (closest.HasValue)
                    {
                        srcPts.Add(p);
                        dstPts.Add(closest.Value);
                    }
                }

                if (srcPts.Count < 4)
                    break;

                // Compute optimal alignment
                var deltaTransform = GeometryUtils.ComputeRigidTransform(srcPts, dstPts);
                transform = transform * deltaTransform;

                // Check convergence
                float error = 0;
                for (int i = 0; i < srcPts.Count; i++)
                {
                    var transformed = Vector3.TransformPosition(srcPts[i], deltaTransform);
                    error += (transformed - dstPts[i]).LengthSquared;
                }
                error /= srcPts.Count;

                if (Math.Abs(prevError - error) < convergenceThreshold)
                    break;

                prevError = error;
            }

            return transform;
        }

        /// <summary>
        /// Coarse alignment using feature-based matching (faster but less accurate)
        /// </summary>
        public static Matrix4 AlignFeatureBased(MeshData source, MeshData target)
        {
            // Use RANSAC-based alignment from GeometryUtils
            var srcSampled = SamplePoints(source.Vertices, Math.Min(500, source.Vertices.Count));
            var tgtSampled = SamplePoints(target.Vertices, Math.Min(500, target.Vertices.Count));

            return GeometryUtils.ComputeRigidTransformRANSAC(srcSampled, tgtSampled, out _, out _);
        }

        /// <summary>
        /// Aligns point clouds using 4PCS (4-Points Congruent Sets) algorithm
        /// </summary>
        public static Matrix4 Align4PCS(List<Vector3> source, List<Vector3> target, int samples = 200)
        {
            var rng = new Random(42);
            Matrix4 bestTransform = Matrix4.Identity;
            int bestInliers = 0;

            var srcSampled = SamplePoints(source, Math.Min(samples, source.Count));
            var tgtSampled = SamplePoints(target, Math.Min(samples, target.Count));

            // Try random 4-point samples
            for (int iter = 0; iter < 100; iter++)
            {
                // Pick 4 random points from source
                var srcIndices = Enumerable.Range(0, srcSampled.Count)
                    .OrderBy(x => rng.Next())
                    .Take(4)
                    .ToList();

                var srcPts = srcIndices.Select(i => srcSampled[i]).ToList();

                // Find congruent set in target
                // (Simplified - just use random points and check alignment)
                var tgtIndices = Enumerable.Range(0, tgtSampled.Count)
                    .OrderBy(x => rng.Next())
                    .Take(4)
                    .ToList();

                var tgtPts = tgtIndices.Select(i => tgtSampled[i]).ToList();

                // Compute transform
                var transform = GeometryUtils.ComputeRigidTransform(srcPts, tgtPts);

                // Count inliers
                int inliers = CountInliers(srcSampled, tgtSampled, transform, 0.05f);

                if (inliers > bestInliers)
                {
                    bestInliers = inliers;
                    bestTransform = transform;
                }
            }

            // Refine with ICP using best initial alignment
            // (Would need full mesh for this, returning coarse result)
            return bestTransform;
        }

        private static int CountInliers(List<Vector3> source, List<Vector3> target, Matrix4 transform, float threshold)
        {
            var targetHash = BuildSpatialHash(target, threshold * 2);
            int inliers = 0;

            foreach (var p in source)
            {
                var transformed = Vector3.TransformPosition(p, transform);
                var closest = FindClosestPoint(transformed, target, targetHash, threshold * 2);
                if (closest.HasValue && (transformed - closest.Value).Length < threshold)
                {
                    inliers++;
                }
            }

            return inliers;
        }

        #endregion

        #region Normal Computation

        /// <summary>
        /// Computes per-vertex normals
        /// </summary>
        public static Vector3[] ComputeVertexNormals(MeshData mesh)
        {
            var normals = new Vector3[mesh.Vertices.Count];

            // Accumulate face normals at each vertex
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                var edge1 = v1 - v0;
                var edge2 = v2 - v0;
                var faceNormal = Vector3.Cross(edge1, edge2);

                normals[i0] += faceNormal;
                normals[i1] += faceNormal;
                normals[i2] += faceNormal;
            }

            // Normalize
            for (int i = 0; i < normals.Length; i++)
            {
                if (normals[i].LengthSquared > 0.0001f)
                    normals[i] = normals[i].Normalized();
            }

            return normals;
        }

        /// <summary>
        /// Flips mesh normals (reverses winding order)
        /// </summary>
        public static MeshData FlipNormals(MeshData mesh)
        {
            var result = CloneMesh(mesh);

            // Reverse triangle winding
            for (int i = 0; i < result.Indices.Count; i += 3)
            {
                int temp = result.Indices[i + 1];
                result.Indices[i + 1] = result.Indices[i + 2];
                result.Indices[i + 2] = temp;
            }

            return result;
        }

        #endregion

        #region Helper Methods

        private static MeshData CloneMesh(MeshData mesh)
        {
            return new MeshData
            {
                Vertices = new List<Vector3>(mesh.Vertices),
                Colors = new List<Vector3>(mesh.Colors),
                Indices = new List<int>(mesh.Indices),
                PixelToVertexIndex = mesh.PixelToVertexIndex
            };
        }

        private static MeshData RemapMesh(MeshData mesh, bool[] keepVertex)
        {
            var newVertices = new List<Vector3>();
            var newColors = new List<Vector3>();
            var oldToNew = new int[mesh.Vertices.Count];
            Array.Fill(oldToNew, -1);

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                if (keepVertex[i])
                {
                    oldToNew[i] = newVertices.Count;
                    newVertices.Add(mesh.Vertices[i]);
                    newColors.Add(mesh.Colors[i]);
                }
            }

            var newIndices = new List<int>();
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = oldToNew[mesh.Indices[i]];
                int i1 = oldToNew[mesh.Indices[i + 1]];
                int i2 = oldToNew[mesh.Indices[i + 2]];

                if (i0 >= 0 && i1 >= 0 && i2 >= 0)
                {
                    newIndices.Add(i0);
                    newIndices.Add(i1);
                    newIndices.Add(i2);
                }
            }

            return new MeshData
            {
                Vertices = newVertices,
                Colors = newColors,
                Indices = newIndices
            };
        }

        private static MeshData ExtractSubMesh(MeshData mesh, HashSet<int> vertexIndices)
        {
            var keepVertex = new bool[mesh.Vertices.Count];
            foreach (int idx in vertexIndices)
                keepVertex[idx] = true;

            return RemapMesh(mesh, keepVertex);
        }

        private static List<HashSet<int>> BuildVertexAdjacency(MeshData mesh)
        {
            var adjacency = new List<HashSet<int>>();
            for (int i = 0; i < mesh.Vertices.Count; i++)
                adjacency.Add(new HashSet<int>());

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                adjacency[i0].Add(i1);
                adjacency[i0].Add(i2);
                adjacency[i1].Add(i0);
                adjacency[i1].Add(i2);
                adjacency[i2].Add(i0);
                adjacency[i2].Add(i1);
            }

            return adjacency;
        }

        private static List<Vector3> SamplePoints(List<Vector3> points, int count)
        {
            if (points.Count <= count)
                return new List<Vector3>(points);

            var rng = new Random(42);
            return points.OrderBy(x => rng.Next()).Take(count).ToList();
        }

        private static Dictionary<(int, int, int), List<int>> BuildSpatialHash(List<Vector3> points, float cellSize)
        {
            var hash = new Dictionary<(int, int, int), List<int>>();

            for (int i = 0; i < points.Count; i++)
            {
                var p = points[i];
                var key = (
                    (int)Math.Floor(p.X / cellSize),
                    (int)Math.Floor(p.Y / cellSize),
                    (int)Math.Floor(p.Z / cellSize)
                );

                if (!hash.ContainsKey(key))
                    hash[key] = new List<int>();
                hash[key].Add(i);
            }

            return hash;
        }

        private static Vector3? FindClosestPoint(Vector3 query, List<Vector3> points,
            Dictionary<(int, int, int), List<int>> spatialHash, float cellSize)
        {
            int cx = (int)Math.Floor(query.X / cellSize);
            int cy = (int)Math.Floor(query.Y / cellSize);
            int cz = (int)Math.Floor(query.Z / cellSize);

            Vector3? closest = null;
            float minDistSq = float.MaxValue;

            // Search neighboring cells
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        var key = (cx + dx, cy + dy, cz + dz);
                        if (spatialHash.TryGetValue(key, out var indices))
                        {
                            foreach (int idx in indices)
                            {
                                float distSq = (points[idx] - query).LengthSquared;
                                if (distSq < minDistSq)
                                {
                                    minDistSq = distSq;
                                    closest = points[idx];
                                }
                            }
                        }
                    }
                }
            }

            return closest;
        }

        #endregion
    }
}
