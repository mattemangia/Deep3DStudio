using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    /// <summary>
    /// Comprehensive mesh cleaning and repair tools for production-quality mesh processing
    /// </summary>
    public static class MeshCleaningTools
    {
        #region Remove Small Components

        /// <summary>
        /// Removes connected components smaller than a minimum vertex count
        /// </summary>
        /// <param name="mesh">Input mesh</param>
        /// <param name="minVertexCount">Minimum vertices to keep a component</param>
        /// <returns>Cleaned mesh with small components removed</returns>
        public static MeshData RemoveSmallComponents(MeshData mesh, int minVertexCount = 100)
        {
            var components = FindConnectedComponents(mesh);

            if (components.Count <= 1)
                return CloneMesh(mesh);

            // Keep only components above threshold
            var keepVertices = new bool[mesh.Vertices.Count];

            foreach (var component in components)
            {
                if (component.Count >= minVertexCount)
                {
                    foreach (int idx in component)
                        keepVertices[idx] = true;
                }
            }

            return RemapMesh(mesh, keepVertices);
        }

        /// <summary>
        /// Removes components smaller than a percentage of the largest component
        /// </summary>
        public static MeshData RemoveSmallComponentsByRatio(MeshData mesh, float minRatio = 0.01f)
        {
            var components = FindConnectedComponents(mesh);

            if (components.Count <= 1)
                return CloneMesh(mesh);

            int maxSize = components.Max(c => c.Count);
            int threshold = (int)(maxSize * minRatio);

            var keepVertices = new bool[mesh.Vertices.Count];

            foreach (var component in components)
            {
                if (component.Count >= threshold)
                {
                    foreach (int idx in component)
                        keepVertices[idx] = true;
                }
            }

            return RemapMesh(mesh, keepVertices);
        }

        /// <summary>
        /// Keeps only the N largest connected components
        /// </summary>
        public static MeshData KeepLargestComponents(MeshData mesh, int numComponents = 1)
        {
            var components = FindConnectedComponents(mesh);

            if (components.Count <= numComponents)
                return CloneMesh(mesh);

            var keepVertices = new bool[mesh.Vertices.Count];

            var largestComponents = components.OrderByDescending(c => c.Count).Take(numComponents);
            foreach (var component in largestComponents)
            {
                foreach (int idx in component)
                    keepVertices[idx] = true;
            }

            return RemapMesh(mesh, keepVertices);
        }

        #endregion

        #region Fill Holes

        /// <summary>
        /// Detects and fills holes in the mesh using triangulation
        /// </summary>
        /// <param name="mesh">Input mesh</param>
        /// <param name="maxHoleEdges">Maximum hole boundary edges to fill (larger holes skipped)</param>
        /// <returns>Mesh with holes filled</returns>
        public static MeshData FillHoles(MeshData mesh, int maxHoleEdges = 100)
        {
            var result = CloneMesh(mesh);
            var holes = FindHoleBoundaries(result);

            foreach (var hole in holes)
            {
                if (hole.Count <= 2 || hole.Count > maxHoleEdges)
                    continue;

                FillHole(result, hole);
            }

            return result;
        }

        /// <summary>
        /// Finds boundary loops (holes) in the mesh
        /// </summary>
        private static List<List<int>> FindHoleBoundaries(MeshData mesh)
        {
            // Build edge usage count
            var edgeCount = new Dictionary<(int, int), int>();

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                AddEdge(edgeCount, i0, i1);
                AddEdge(edgeCount, i1, i2);
                AddEdge(edgeCount, i2, i0);
            }

            // Find boundary edges (used only once)
            var boundaryEdges = new HashSet<(int, int)>();
            foreach (var kvp in edgeCount)
            {
                if (kvp.Value == 1)
                    boundaryEdges.Add(kvp.Key);
            }

            if (boundaryEdges.Count == 0)
                return new List<List<int>>();

            // Build adjacency for boundary vertices
            var boundaryAdjacency = new Dictionary<int, List<int>>();
            foreach (var edge in boundaryEdges)
            {
                if (!boundaryAdjacency.ContainsKey(edge.Item1))
                    boundaryAdjacency[edge.Item1] = new List<int>();
                if (!boundaryAdjacency.ContainsKey(edge.Item2))
                    boundaryAdjacency[edge.Item2] = new List<int>();

                boundaryAdjacency[edge.Item1].Add(edge.Item2);
                boundaryAdjacency[edge.Item2].Add(edge.Item1);
            }

            // Trace boundary loops
            var holes = new List<List<int>>();
            var visited = new HashSet<int>();

            foreach (int startVertex in boundaryAdjacency.Keys)
            {
                if (visited.Contains(startVertex))
                    continue;

                var loop = TraceBoundaryLoop(startVertex, boundaryAdjacency, visited);
                if (loop.Count >= 3)
                    holes.Add(loop);
            }

            return holes;
        }

        private static List<int> TraceBoundaryLoop(int start, Dictionary<int, List<int>> adjacency, HashSet<int> visited)
        {
            var loop = new List<int>();
            int current = start;
            int prev = -1;

            while (true)
            {
                if (visited.Contains(current) && current != start)
                    break;

                visited.Add(current);
                loop.Add(current);

                var neighbors = adjacency[current];
                int next = -1;

                foreach (int n in neighbors)
                {
                    if (n != prev && (!visited.Contains(n) || n == start))
                    {
                        next = n;
                        break;
                    }
                }

                if (next == -1 || next == start)
                    break;

                prev = current;
                current = next;
            }

            return loop;
        }

        /// <summary>
        /// Fills a single hole using ear clipping triangulation
        /// </summary>
        private static void FillHole(MeshData mesh, List<int> boundaryLoop)
        {
            if (boundaryLoop.Count < 3)
                return;

            // Use ear clipping for simple polygon triangulation
            var remaining = new List<int>(boundaryLoop);

            // Compute average color for new triangles
            var avgColor = Vector3.Zero;
            foreach (int idx in boundaryLoop)
                avgColor += mesh.Colors[idx];
            avgColor /= boundaryLoop.Count;

            while (remaining.Count >= 3)
            {
                bool earFound = false;

                for (int i = 0; i < remaining.Count && !earFound; i++)
                {
                    int prev = remaining[(i - 1 + remaining.Count) % remaining.Count];
                    int curr = remaining[i];
                    int next = remaining[(i + 1) % remaining.Count];

                    if (IsEar(mesh, remaining, prev, curr, next))
                    {
                        // Add triangle (with correct winding)
                        mesh.Indices.Add(prev);
                        mesh.Indices.Add(curr);
                        mesh.Indices.Add(next);

                        remaining.RemoveAt(i);
                        earFound = true;
                    }
                }

                if (!earFound)
                {
                    // Fallback: use fan triangulation from centroid
                    FillHoleWithFan(mesh, boundaryLoop, avgColor);
                    return;
                }
            }
        }

        private static bool IsEar(MeshData mesh, List<int> polygon, int prev, int curr, int next)
        {
            var vPrev = mesh.Vertices[prev];
            var vCurr = mesh.Vertices[curr];
            var vNext = mesh.Vertices[next];

            // Check if convex
            var edge1 = vCurr - vPrev;
            var edge2 = vNext - vCurr;
            var cross = Vector3.Cross(edge1, edge2);

            // Check that no other vertices are inside the triangle
            for (int i = 0; i < polygon.Count; i++)
            {
                int idx = polygon[i];
                if (idx == prev || idx == curr || idx == next)
                    continue;

                if (PointInTriangle(mesh.Vertices[idx], vPrev, vCurr, vNext))
                    return false;
            }

            return true;
        }

        private static bool PointInTriangle(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
        {
            var v0 = c - a;
            var v1 = b - a;
            var v2 = p - a;

            float dot00 = Vector3.Dot(v0, v0);
            float dot01 = Vector3.Dot(v0, v1);
            float dot02 = Vector3.Dot(v0, v2);
            float dot11 = Vector3.Dot(v1, v1);
            float dot12 = Vector3.Dot(v1, v2);

            float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);
            float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
            float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

            return (u >= 0) && (v >= 0) && (u + v <= 1);
        }

        private static void FillHoleWithFan(MeshData mesh, List<int> boundaryLoop, Vector3 color)
        {
            // Compute centroid
            var centroid = Vector3.Zero;
            foreach (int idx in boundaryLoop)
                centroid += mesh.Vertices[idx];
            centroid /= boundaryLoop.Count;

            // Add centroid vertex
            int centroidIdx = mesh.Vertices.Count;
            mesh.Vertices.Add(centroid);
            mesh.Colors.Add(color);

            // Create fan triangles
            for (int i = 0; i < boundaryLoop.Count; i++)
            {
                int curr = boundaryLoop[i];
                int next = boundaryLoop[(i + 1) % boundaryLoop.Count];

                mesh.Indices.Add(curr);
                mesh.Indices.Add(next);
                mesh.Indices.Add(centroidIdx);
            }
        }

        #endregion

        #region Fix Non-Manifold

        /// <summary>
        /// Detects and reports non-manifold geometry
        /// </summary>
        public static NonManifoldReport AnalyzeNonManifold(MeshData mesh)
        {
            var report = new NonManifoldReport();

            // Find edges used by more than 2 triangles
            var edgeTriangles = new Dictionary<(int, int), List<int>>();

            for (int t = 0; t < mesh.Indices.Count / 3; t++)
            {
                int i0 = mesh.Indices[t * 3];
                int i1 = mesh.Indices[t * 3 + 1];
                int i2 = mesh.Indices[t * 3 + 2];

                AddEdgeTriangle(edgeTriangles, i0, i1, t);
                AddEdgeTriangle(edgeTriangles, i1, i2, t);
                AddEdgeTriangle(edgeTriangles, i2, i0, t);
            }

            foreach (var kvp in edgeTriangles)
            {
                if (kvp.Value.Count > 2)
                {
                    report.NonManifoldEdges.Add(kvp.Key);
                }
            }

            // Find vertices that create non-manifold topology
            var vertexFans = BuildVertexFans(mesh);
            foreach (var kvp in vertexFans)
            {
                if (kvp.Value.Count > 1)
                {
                    report.NonManifoldVertices.Add(kvp.Key);
                }
            }

            return report;
        }

        /// <summary>
        /// Attempts to fix non-manifold geometry by duplicating vertices
        /// </summary>
        public static MeshData FixNonManifold(MeshData mesh)
        {
            var result = CloneMesh(mesh);

            // Fix non-manifold edges by duplicating shared vertices
            var edgeTriangles = new Dictionary<(int, int), List<int>>();

            for (int t = 0; t < result.Indices.Count / 3; t++)
            {
                int i0 = result.Indices[t * 3];
                int i1 = result.Indices[t * 3 + 1];
                int i2 = result.Indices[t * 3 + 2];

                AddEdgeTriangle(edgeTriangles, i0, i1, t);
                AddEdgeTriangle(edgeTriangles, i1, i2, t);
                AddEdgeTriangle(edgeTriangles, i2, i0, t);
            }

            var verticesToDuplicate = new Dictionary<int, List<int>>(); // vertex -> triangles that need new copy

            foreach (var kvp in edgeTriangles)
            {
                if (kvp.Value.Count > 2)
                {
                    // Keep first 2 triangles, duplicate for rest
                    for (int i = 2; i < kvp.Value.Count; i++)
                    {
                        int triIdx = kvp.Value[i];
                        int v1 = kvp.Key.Item1;
                        int v2 = kvp.Key.Item2;

                        if (!verticesToDuplicate.ContainsKey(v1))
                            verticesToDuplicate[v1] = new List<int>();
                        if (!verticesToDuplicate.ContainsKey(v2))
                            verticesToDuplicate[v2] = new List<int>();

                        verticesToDuplicate[v1].Add(triIdx);
                        verticesToDuplicate[v2].Add(triIdx);
                    }
                }
            }

            // Duplicate vertices and update triangle references
            foreach (var kvp in verticesToDuplicate)
            {
                int originalVertex = kvp.Key;
                var triangles = kvp.Value.Distinct().ToList();

                foreach (int triIdx in triangles)
                {
                    // Create new vertex
                    int newVertexIdx = result.Vertices.Count;
                    result.Vertices.Add(result.Vertices[originalVertex]);
                    result.Colors.Add(result.Colors[originalVertex]);

                    // Update triangle
                    for (int j = 0; j < 3; j++)
                    {
                        if (result.Indices[triIdx * 3 + j] == originalVertex)
                        {
                            result.Indices[triIdx * 3 + j] = newVertexIdx;
                        }
                    }
                }
            }

            return result;
        }

        private static Dictionary<int, List<HashSet<int>>> BuildVertexFans(MeshData mesh)
        {
            var vertexFans = new Dictionary<int, List<HashSet<int>>>();
            var vertexTriangles = new Dictionary<int, List<int>>();

            // Build vertex -> triangles map
            for (int t = 0; t < mesh.Indices.Count / 3; t++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int v = mesh.Indices[t * 3 + j];
                    if (!vertexTriangles.ContainsKey(v))
                        vertexTriangles[v] = new List<int>();
                    vertexTriangles[v].Add(t);
                }
            }

            // For each vertex, group triangles into connected fans
            foreach (var kvp in vertexTriangles)
            {
                int vertex = kvp.Key;
                var triangles = kvp.Value;
                var fans = GroupTrianglesIntoFans(mesh, vertex, triangles);
                vertexFans[vertex] = fans;
            }

            return vertexFans;
        }

        private static List<HashSet<int>> GroupTrianglesIntoFans(MeshData mesh, int vertex, List<int> triangles)
        {
            var fans = new List<HashSet<int>>();
            var visited = new HashSet<int>();

            foreach (int startTri in triangles)
            {
                if (visited.Contains(startTri))
                    continue;

                var fan = new HashSet<int>();
                var queue = new Queue<int>();
                queue.Enqueue(startTri);

                while (queue.Count > 0)
                {
                    int tri = queue.Dequeue();
                    if (visited.Contains(tri))
                        continue;

                    visited.Add(tri);
                    fan.Add(tri);

                    // Find adjacent triangles sharing an edge with this triangle at the vertex
                    foreach (int other in triangles)
                    {
                        if (visited.Contains(other))
                            continue;

                        if (SharesEdgeAtVertex(mesh, tri, other, vertex))
                        {
                            queue.Enqueue(other);
                        }
                    }
                }

                if (fan.Count > 0)
                    fans.Add(fan);
            }

            return fans;
        }

        private static bool SharesEdgeAtVertex(MeshData mesh, int tri1, int tri2, int vertex)
        {
            var verts1 = new int[] { mesh.Indices[tri1 * 3], mesh.Indices[tri1 * 3 + 1], mesh.Indices[tri1 * 3 + 2] };
            var verts2 = new int[] { mesh.Indices[tri2 * 3], mesh.Indices[tri2 * 3 + 1], mesh.Indices[tri2 * 3 + 2] };

            int sharedCount = 0;
            foreach (int v1 in verts1)
            {
                if (verts2.Contains(v1))
                    sharedCount++;
            }

            return sharedCount == 2 && verts1.Contains(vertex) && verts2.Contains(vertex);
        }

        #endregion

        #region Remove Noise / Statistical Outlier Removal

        /// <summary>
        /// Removes statistical outlier vertices (noise) based on mean distance to neighbors
        /// </summary>
        /// <param name="mesh">Input mesh</param>
        /// <param name="kNeighbors">Number of neighbors to consider</param>
        /// <param name="stdRatio">Standard deviation multiplier threshold</param>
        public static MeshData RemoveStatisticalOutliers(MeshData mesh, int kNeighbors = 10, float stdRatio = 2.0f)
        {
            if (mesh.Vertices.Count < kNeighbors + 1)
                return CloneMesh(mesh);

            // Build spatial index
            var spatialHash = BuildSpatialHash(mesh.Vertices, 0.1f);

            // Compute mean distance to k nearest neighbors for each vertex
            var meanDistances = new float[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var neighbors = FindKNearestNeighbors(mesh.Vertices, i, kNeighbors, spatialHash);
                float sumDist = 0;
                foreach (int n in neighbors)
                {
                    sumDist += (mesh.Vertices[i] - mesh.Vertices[n]).Length;
                }
                meanDistances[i] = neighbors.Count > 0 ? sumDist / neighbors.Count : 0;
            }

            // Compute global mean and standard deviation
            float globalMean = meanDistances.Average();
            float variance = meanDistances.Select(d => (d - globalMean) * (d - globalMean)).Average();
            float stdDev = (float)Math.Sqrt(variance);

            // Filter vertices within threshold
            float threshold = globalMean + stdRatio * stdDev;
            var keepVertices = new bool[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                keepVertices[i] = meanDistances[i] <= threshold;
            }

            return RemapMesh(mesh, keepVertices);
        }

        /// <summary>
        /// Removes vertices that are isolated (have no triangles or very few neighbors)
        /// </summary>
        public static MeshData RemoveIsolatedVertices(MeshData mesh, int minTriangles = 1)
        {
            var triangleCount = new int[mesh.Vertices.Count];

            for (int i = 0; i < mesh.Indices.Count; i++)
            {
                triangleCount[mesh.Indices[i]]++;
            }

            var keepVertices = new bool[mesh.Vertices.Count];
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                keepVertices[i] = triangleCount[i] >= minTriangles;
            }

            return RemapMesh(mesh, keepVertices);
        }

        #endregion

        #region Remove Degenerate Triangles

        /// <summary>
        /// Removes degenerate triangles (zero area, duplicate vertices, etc.)
        /// </summary>
        public static MeshData RemoveDegenerateTriangles(MeshData mesh, float minArea = 1e-8f)
        {
            var result = new MeshData
            {
                Vertices = new List<Vector3>(mesh.Vertices),
                Colors = new List<Vector3>(mesh.Colors)
            };

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                // Skip if duplicate vertices
                if (i0 == i1 || i1 == i2 || i2 == i0)
                    continue;

                // Check area
                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                var cross = Vector3.Cross(v1 - v0, v2 - v0);
                float area = cross.Length * 0.5f;

                if (area < minArea)
                    continue;

                // Keep this triangle
                result.Indices.Add(i0);
                result.Indices.Add(i1);
                result.Indices.Add(i2);
            }

            return RemoveUnusedVertices(result);
        }

        /// <summary>
        /// Removes long thin triangles (slivers)
        /// </summary>
        public static MeshData RemoveSliverTriangles(MeshData mesh, float minAspectRatio = 0.01f)
        {
            var result = new MeshData
            {
                Vertices = new List<Vector3>(mesh.Vertices),
                Colors = new List<Vector3>(mesh.Colors)
            };

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = mesh.Indices[i];
                int i1 = mesh.Indices[i + 1];
                int i2 = mesh.Indices[i + 2];

                var v0 = mesh.Vertices[i0];
                var v1 = mesh.Vertices[i1];
                var v2 = mesh.Vertices[i2];

                // Compute edge lengths
                float e0 = (v1 - v0).Length;
                float e1 = (v2 - v1).Length;
                float e2 = (v0 - v2).Length;

                float maxEdge = Math.Max(e0, Math.Max(e1, e2));
                float minEdge = Math.Min(e0, Math.Min(e1, e2));

                // Aspect ratio check
                if (maxEdge > 0 && minEdge / maxEdge < minAspectRatio)
                    continue;

                // Keep this triangle
                result.Indices.Add(i0);
                result.Indices.Add(i1);
                result.Indices.Add(i2);
            }

            return RemoveUnusedVertices(result);
        }

        #endregion

        #region Fix Normals

        /// <summary>
        /// Ensures consistent normal orientation across the mesh
        /// </summary>
        public static MeshData FixNormalOrientation(MeshData mesh)
        {
            if (mesh.Indices.Count < 3)
                return CloneMesh(mesh);

            var result = CloneMesh(mesh);

            // Build triangle adjacency
            var triangleAdjacency = BuildTriangleAdjacency(result);

            // BFS to propagate consistent orientation
            var visited = new bool[result.Indices.Count / 3];
            var flipped = new bool[result.Indices.Count / 3];

            for (int start = 0; start < visited.Length; start++)
            {
                if (visited[start])
                    continue;

                var queue = new Queue<int>();
                queue.Enqueue(start);
                visited[start] = true;

                while (queue.Count > 0)
                {
                    int tri = queue.Dequeue();

                    foreach (int neighbor in triangleAdjacency[tri])
                    {
                        if (visited[neighbor])
                            continue;

                        visited[neighbor] = true;

                        // Check if normals are consistent
                        if (!HaveConsistentOrientation(result, tri, neighbor, flipped[tri]))
                        {
                            flipped[neighbor] = true;
                        }

                        queue.Enqueue(neighbor);
                    }
                }
            }

            // Apply flips
            for (int t = 0; t < flipped.Length; t++)
            {
                if (flipped[t])
                {
                    int temp = result.Indices[t * 3 + 1];
                    result.Indices[t * 3 + 1] = result.Indices[t * 3 + 2];
                    result.Indices[t * 3 + 2] = temp;
                }
            }

            return result;
        }

        private static bool HaveConsistentOrientation(MeshData mesh, int tri1, int tri2, bool tri1Flipped)
        {
            var verts1 = new int[] { mesh.Indices[tri1 * 3], mesh.Indices[tri1 * 3 + 1], mesh.Indices[tri1 * 3 + 2] };
            var verts2 = new int[] { mesh.Indices[tri2 * 3], mesh.Indices[tri2 * 3 + 1], mesh.Indices[tri2 * 3 + 2] };

            // Find shared edge
            int shared1 = -1, shared2 = -1;
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    if (verts1[i] == verts2[j])
                    {
                        if (shared1 == -1)
                            shared1 = i;
                        else
                            shared2 = i;
                    }
                }
            }

            if (shared1 == -1 || shared2 == -1)
                return true; // No shared edge, assume consistent

            // Check edge direction in both triangles
            // Consistent orientation means the shared edge is traversed in opposite directions
            int idx1Next = (shared1 + 1) % 3;
            int idx2Next = (shared2 + 1) % 3;

            return verts1[shared1] != verts2[shared2] || verts1[idx1Next] != verts2[idx2Next];
        }

        #endregion

        #region Mesh Simplification

        /// <summary>
        /// Performs vertex clustering simplification
        /// </summary>
        public static MeshData SimplifyVertexClustering(MeshData mesh, float cellSize)
        {
            var voxelMap = new Dictionary<(int, int, int), List<int>>();

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                var v = mesh.Vertices[i];
                var key = (
                    (int)Math.Floor(v.X / cellSize),
                    (int)Math.Floor(v.Y / cellSize),
                    (int)Math.Floor(v.Z / cellSize)
                );

                if (!voxelMap.ContainsKey(key))
                    voxelMap[key] = new List<int>();
                voxelMap[key].Add(i);
            }

            var result = new MeshData();
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

                int newIdx = result.Vertices.Count;
                result.Vertices.Add(centroid);
                result.Colors.Add(color);

                foreach (int idx in indices)
                    oldToNew[idx] = newIdx;
            }

            // Rebuild triangles
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = oldToNew[mesh.Indices[i]];
                int i1 = oldToNew[mesh.Indices[i + 1]];
                int i2 = oldToNew[mesh.Indices[i + 2]];

                if (i0 != i1 && i1 != i2 && i2 != i0)
                {
                    result.Indices.Add(i0);
                    result.Indices.Add(i1);
                    result.Indices.Add(i2);
                }
            }

            return result;
        }

        #endregion

        #region Complete Mesh Cleanup

        /// <summary>
        /// Performs comprehensive mesh cleanup with all operations
        /// </summary>
        public static MeshData CleanupMesh(MeshData mesh, MeshCleanupOptions options)
        {
            var result = mesh;

            if (options.RemoveDegenerateTris)
            {
                result = RemoveDegenerateTriangles(result, options.MinTriangleArea);
            }

            if (options.RemoveSlivers)
            {
                result = RemoveSliverTriangles(result, options.MinAspectRatio);
            }

            if (options.RemoveSmallComponents)
            {
                result = RemoveSmallComponents(result, options.MinComponentSize);
            }

            if (options.RemoveStatisticalOutliers)
            {
                result = RemoveStatisticalOutliers(result, options.KNeighbors, options.StdRatio);
            }

            if (options.FixNonManifold)
            {
                result = FixNonManifold(result);
            }

            if (options.FillHoles)
            {
                result = FillHoles(result, options.MaxHoleEdges);
            }

            if (options.FixNormals)
            {
                result = FixNormalOrientation(result);
            }

            if (options.RemoveIsolatedVertices)
            {
                result = RemoveIsolatedVertices(result, 1);
            }

            return result;
        }

        #endregion

        #region Helper Methods

        private static void AddEdge(Dictionary<(int, int), int> edgeCount, int v1, int v2)
        {
            var key = v1 < v2 ? (v1, v2) : (v2, v1);
            if (!edgeCount.ContainsKey(key))
                edgeCount[key] = 0;
            edgeCount[key]++;
        }

        private static void AddEdgeTriangle(Dictionary<(int, int), List<int>> edgeTriangles, int v1, int v2, int triangle)
        {
            var key = v1 < v2 ? (v1, v2) : (v2, v1);
            if (!edgeTriangles.ContainsKey(key))
                edgeTriangles[key] = new List<int>();
            edgeTriangles[key].Add(triangle);
        }

        private static List<HashSet<int>> FindConnectedComponents(MeshData mesh)
        {
            var components = new List<HashSet<int>>();
            var visited = new bool[mesh.Vertices.Count];

            // Build adjacency
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

            for (int start = 0; start < mesh.Vertices.Count; start++)
            {
                if (visited[start])
                    continue;

                var component = new HashSet<int>();
                var queue = new Queue<int>();
                queue.Enqueue(start);

                while (queue.Count > 0)
                {
                    int v = queue.Dequeue();
                    if (visited[v])
                        continue;

                    visited[v] = true;
                    component.Add(v);

                    foreach (int neighbor in adjacency[v])
                    {
                        if (!visited[neighbor])
                            queue.Enqueue(neighbor);
                    }
                }

                if (component.Count > 0)
                    components.Add(component);
            }

            return components;
        }

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
            var result = new MeshData();
            var oldToNew = new int[mesh.Vertices.Count];
            Array.Fill(oldToNew, -1);

            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                if (keepVertex[i])
                {
                    oldToNew[i] = result.Vertices.Count;
                    result.Vertices.Add(mesh.Vertices[i]);
                    result.Colors.Add(mesh.Colors[i]);
                }
            }

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int i0 = oldToNew[mesh.Indices[i]];
                int i1 = oldToNew[mesh.Indices[i + 1]];
                int i2 = oldToNew[mesh.Indices[i + 2]];

                if (i0 >= 0 && i1 >= 0 && i2 >= 0)
                {
                    result.Indices.Add(i0);
                    result.Indices.Add(i1);
                    result.Indices.Add(i2);
                }
            }

            return result;
        }

        private static MeshData RemoveUnusedVertices(MeshData mesh)
        {
            var usedVertices = new bool[mesh.Vertices.Count];
            foreach (int idx in mesh.Indices)
                usedVertices[idx] = true;

            return RemapMesh(mesh, usedVertices);
        }

        private static List<HashSet<int>> BuildTriangleAdjacency(MeshData mesh)
        {
            int numTriangles = mesh.Indices.Count / 3;
            var adjacency = new List<HashSet<int>>();

            for (int i = 0; i < numTriangles; i++)
                adjacency.Add(new HashSet<int>());

            var edgeToTriangles = new Dictionary<(int, int), List<int>>();

            for (int t = 0; t < numTriangles; t++)
            {
                int i0 = mesh.Indices[t * 3];
                int i1 = mesh.Indices[t * 3 + 1];
                int i2 = mesh.Indices[t * 3 + 2];

                AddEdgeTriangle(edgeToTriangles, i0, i1, t);
                AddEdgeTriangle(edgeToTriangles, i1, i2, t);
                AddEdgeTriangle(edgeToTriangles, i2, i0, t);
            }

            foreach (var triangles in edgeToTriangles.Values)
            {
                for (int i = 0; i < triangles.Count; i++)
                {
                    for (int j = i + 1; j < triangles.Count; j++)
                    {
                        adjacency[triangles[i]].Add(triangles[j]);
                        adjacency[triangles[j]].Add(triangles[i]);
                    }
                }
            }

            return adjacency;
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

        private static List<int> FindKNearestNeighbors(List<Vector3> points, int queryIdx, int k,
            Dictionary<(int, int, int), List<int>> spatialHash)
        {
            var query = points[queryIdx];
            var candidates = new List<(int idx, float dist)>();

            // Search expanding radius
            for (int radius = 1; radius <= 5 && candidates.Count < k * 2; radius++)
            {
                int cx = (int)Math.Floor(query.X / 0.1f);
                int cy = (int)Math.Floor(query.Y / 0.1f);
                int cz = (int)Math.Floor(query.Z / 0.1f);

                for (int dx = -radius; dx <= radius; dx++)
                {
                    for (int dy = -radius; dy <= radius; dy++)
                    {
                        for (int dz = -radius; dz <= radius; dz++)
                        {
                            var key = (cx + dx, cy + dy, cz + dz);
                            if (spatialHash.TryGetValue(key, out var indices))
                            {
                                foreach (int idx in indices)
                                {
                                    if (idx != queryIdx)
                                    {
                                        float dist = (points[idx] - query).Length;
                                        candidates.Add((idx, dist));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return candidates.OrderBy(c => c.dist).Take(k).Select(c => c.idx).ToList();
        }

        #endregion
    }

    /// <summary>
    /// Report of non-manifold geometry
    /// </summary>
    public class NonManifoldReport
    {
        public List<(int, int)> NonManifoldEdges { get; } = new List<(int, int)>();
        public List<int> NonManifoldVertices { get; } = new List<int>();

        public bool IsManifold => NonManifoldEdges.Count == 0 && NonManifoldVertices.Count == 0;

        public override string ToString()
        {
            if (IsManifold)
                return "Mesh is manifold";

            return $"Non-manifold geometry detected:\n" +
                   $"  - {NonManifoldEdges.Count} non-manifold edges\n" +
                   $"  - {NonManifoldVertices.Count} non-manifold vertices";
        }
    }

    /// <summary>
    /// Options for comprehensive mesh cleanup
    /// </summary>
    public class MeshCleanupOptions
    {
        public bool RemoveDegenerateTris { get; set; } = true;
        public float MinTriangleArea { get; set; } = 1e-8f;

        public bool RemoveSlivers { get; set; } = true;
        public float MinAspectRatio { get; set; } = 0.01f;

        public bool RemoveSmallComponents { get; set; } = true;
        public int MinComponentSize { get; set; } = 100;

        public bool RemoveStatisticalOutliers { get; set; } = false;
        public int KNeighbors { get; set; } = 10;
        public float StdRatio { get; set; } = 2.0f;

        public bool FixNonManifold { get; set; } = true;

        public bool FillHoles { get; set; } = false;
        public int MaxHoleEdges { get; set; } = 100;

        public bool FixNormals { get; set; } = true;

        public bool RemoveIsolatedVertices { get; set; } = true;

        public static MeshCleanupOptions Default => new MeshCleanupOptions();

        public static MeshCleanupOptions Aggressive => new MeshCleanupOptions
        {
            RemoveStatisticalOutliers = true,
            FillHoles = true
        };
    }
}
