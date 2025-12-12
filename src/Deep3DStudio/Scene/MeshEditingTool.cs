using System;
using System.Collections.Generic;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    /// <summary>
    /// Tool mode for mesh editing operations
    /// </summary>
    public enum MeshEditMode
    {
        Select,
        Delete,
        Paint,
        Weld,
        Extrude
    }

    /// <summary>
    /// Represents a selected triangle with its mesh reference
    /// </summary>
    public class SelectedTriangle
    {
        public MeshObject Mesh { get; set; }
        public int TriangleIndex { get; set; }
        public Vector3[] Vertices { get; set; } = new Vector3[3];

        public SelectedTriangle(MeshObject mesh, int triangleIndex)
        {
            Mesh = mesh;
            TriangleIndex = triangleIndex;
            UpdateVertices();
        }

        public void UpdateVertices()
        {
            if (Mesh?.MeshData?.Indices == null || Mesh.MeshData.Vertices == null)
                return;

            int baseIdx = TriangleIndex * 3;
            if (baseIdx + 2 >= Mesh.MeshData.Indices.Count)
                return;

            var indices = Mesh.MeshData.Indices;
            var verts = Mesh.MeshData.Vertices;

            for (int i = 0; i < 3; i++)
            {
                int vIdx = indices[baseIdx + i];
                if (vIdx >= 0 && vIdx < verts.Count)
                    Vertices[i] = verts[vIdx];
            }
        }
    }

    /// <summary>
    /// Interactive mesh editing tool for triangle-level manipulation
    /// </summary>
    public class MeshEditingTool
    {
        private MeshEditMode _mode = MeshEditMode.Select;
        private HashSet<(MeshObject mesh, int triangleIndex)> _selectedTriangles = new();
        private Vector3 _paintColor = new Vector3(1, 0, 0);
        private float _brushRadius = 0.1f;

        /// <summary>
        /// Current editing mode
        /// </summary>
        public MeshEditMode Mode
        {
            get => _mode;
            set => _mode = value;
        }

        /// <summary>
        /// Currently selected triangles
        /// </summary>
        public IReadOnlyCollection<(MeshObject mesh, int triangleIndex)> SelectedTriangles => _selectedTriangles;

        /// <summary>
        /// Color used for paint mode
        /// </summary>
        public Vector3 PaintColor
        {
            get => _paintColor;
            set => _paintColor = value;
        }

        /// <summary>
        /// Brush radius for paint/select operations
        /// </summary>
        public float BrushRadius
        {
            get => _brushRadius;
            set => _brushRadius = Math.Max(0.01f, value);
        }

        /// <summary>
        /// Event fired when selection changes
        /// </summary>
        public event EventHandler? SelectionChanged;

        /// <summary>
        /// Event fired when mesh is modified
        /// </summary>
        public event EventHandler<MeshObject>? MeshModified;

        /// <summary>
        /// Clear all selected triangles
        /// </summary>
        public void ClearSelection()
        {
            _selectedTriangles.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Select a triangle by index
        /// </summary>
        public void SelectTriangle(MeshObject mesh, int triangleIndex, bool addToSelection = false)
        {
            if (!addToSelection)
                _selectedTriangles.Clear();

            _selectedTriangles.Add((mesh, triangleIndex));
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Toggle triangle selection
        /// </summary>
        public void ToggleTriangleSelection(MeshObject mesh, int triangleIndex)
        {
            var key = (mesh, triangleIndex);
            if (_selectedTriangles.Contains(key))
                _selectedTriangles.Remove(key);
            else
                _selectedTriangles.Add(key);

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Check if a triangle is selected
        /// </summary>
        public bool IsTriangleSelected(MeshObject mesh, int triangleIndex)
        {
            return _selectedTriangles.Contains((mesh, triangleIndex));
        }

        /// <summary>
        /// Performs ray-triangle intersection test using Moller-Trumbore algorithm
        /// </summary>
        public static bool RayTriangleIntersect(
            Vector3 rayOrigin,
            Vector3 rayDir,
            Vector3 v0, Vector3 v1, Vector3 v2,
            out float t,
            out float u,
            out float v)
        {
            t = u = v = 0;
            const float EPSILON = 1e-8f;

            Vector3 edge1 = v1 - v0;
            Vector3 edge2 = v2 - v0;
            Vector3 h = Vector3.Cross(rayDir, edge2);
            float a = Vector3.Dot(edge1, h);

            if (a > -EPSILON && a < EPSILON)
                return false; // Ray parallel to triangle

            float f = 1.0f / a;
            Vector3 s = rayOrigin - v0;
            u = f * Vector3.Dot(s, h);

            if (u < 0.0f || u > 1.0f)
                return false;

            Vector3 q = Vector3.Cross(s, edge1);
            v = f * Vector3.Dot(rayDir, q);

            if (v < 0.0f || u + v > 1.0f)
                return false;

            t = f * Vector3.Dot(edge2, q);

            return t > EPSILON;
        }

        /// <summary>
        /// Pick a triangle from scene meshes using ray casting
        /// </summary>
        public (MeshObject? mesh, int triangleIndex, float distance) PickTriangle(
            Vector3 rayOrigin,
            Vector3 rayDirection,
            IEnumerable<MeshObject> meshes)
        {
            MeshObject? closestMesh = null;
            int closestTriangle = -1;
            float closestDistance = float.MaxValue;

            foreach (var meshObj in meshes)
            {
                if (meshObj.MeshData == null || !meshObj.Visible)
                    continue;

                var meshData = meshObj.MeshData;
                var worldTransform = meshObj.GetWorldTransform();
                var inverseTransform = worldTransform.Inverted();

                // Transform ray to local space
                var localOrigin = Vector3.TransformPosition(rayOrigin, inverseTransform);
                var localDir = Vector3.TransformNormal(rayDirection, inverseTransform).Normalized();

                int triangleCount = meshData.Indices.Count / 3;

                for (int i = 0; i < triangleCount; i++)
                {
                    int i0 = meshData.Indices[i * 3];
                    int i1 = meshData.Indices[i * 3 + 1];
                    int i2 = meshData.Indices[i * 3 + 2];

                    if (i0 >= meshData.Vertices.Count || i1 >= meshData.Vertices.Count || i2 >= meshData.Vertices.Count)
                        continue;

                    var v0 = meshData.Vertices[i0];
                    var v1 = meshData.Vertices[i1];
                    var v2 = meshData.Vertices[i2];

                    if (RayTriangleIntersect(localOrigin, localDir, v0, v1, v2, out float t, out _, out _))
                    {
                        // Transform intersection point back to world space to get correct distance
                        var localHit = localOrigin + localDir * t;
                        var worldHit = Vector3.TransformPosition(localHit, worldTransform);
                        float worldDist = (worldHit - rayOrigin).Length;

                        if (worldDist < closestDistance)
                        {
                            closestDistance = worldDist;
                            closestMesh = meshObj;
                            closestTriangle = i;
                        }
                    }
                }
            }

            return (closestMesh, closestTriangle, closestDistance);
        }

        /// <summary>
        /// Delete selected triangles from their meshes
        /// </summary>
        public void DeleteSelectedTriangles()
        {
            if (_selectedTriangles.Count == 0)
                return;

            // Group by mesh
            var byMesh = _selectedTriangles.GroupBy(x => x.mesh);

            foreach (var group in byMesh)
            {
                var mesh = group.Key;
                var trianglesToDelete = group.Select(x => x.triangleIndex).OrderByDescending(x => x).ToList();

                DeleteTriangles(mesh, trianglesToDelete);
                MeshModified?.Invoke(this, mesh);
            }

            _selectedTriangles.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Delete specific triangles from a mesh
        /// </summary>
        private void DeleteTriangles(MeshObject meshObj, List<int> triangleIndices)
        {
            var meshData = meshObj.MeshData;
            if (meshData == null)
                return;

            // Remove indices in reverse order to avoid index shifting
            foreach (int triIdx in triangleIndices.OrderByDescending(x => x))
            {
                int baseIdx = triIdx * 3;
                if (baseIdx + 2 < meshData.Indices.Count)
                {
                    meshData.Indices.RemoveRange(baseIdx, 3);
                }
            }

            // Clean up orphaned vertices
            CleanupOrphanedVertices(meshData);
            meshObj.UpdateBounds();
        }

        /// <summary>
        /// Remove vertices that are no longer referenced by any triangle
        /// </summary>
        private void CleanupOrphanedVertices(MeshData meshData)
        {
            if (meshData.Vertices.Count == 0)
                return;

            // Find which vertices are still in use
            var usedVertices = new HashSet<int>(meshData.Indices);

            if (usedVertices.Count == meshData.Vertices.Count)
                return; // All vertices are still used

            // Create mapping from old to new indices
            var oldToNew = new Dictionary<int, int>();
            var newVertices = new List<Vector3>();
            var newColors = new List<Vector3>();
            var newUVs = new List<Vector2>();

            for (int i = 0; i < meshData.Vertices.Count; i++)
            {
                if (usedVertices.Contains(i))
                {
                    oldToNew[i] = newVertices.Count;
                    newVertices.Add(meshData.Vertices[i]);

                    if (i < meshData.Colors.Count)
                        newColors.Add(meshData.Colors[i]);

                    if (i < meshData.UVs.Count)
                        newUVs.Add(meshData.UVs[i]);
                }
            }

            // Update indices
            for (int i = 0; i < meshData.Indices.Count; i++)
            {
                meshData.Indices[i] = oldToNew[meshData.Indices[i]];
            }

            // Replace vertex data
            meshData.Vertices = newVertices;
            meshData.Colors = newColors;
            meshData.UVs = newUVs;
        }

        /// <summary>
        /// Paint color on selected triangles
        /// </summary>
        public void PaintSelectedTriangles()
        {
            foreach (var (mesh, triIdx) in _selectedTriangles)
            {
                PaintTriangle(mesh, triIdx, _paintColor);
                MeshModified?.Invoke(this, mesh);
            }
        }

        /// <summary>
        /// Paint a single triangle with a color
        /// </summary>
        public void PaintTriangle(MeshObject meshObj, int triangleIndex, Vector3 color)
        {
            var meshData = meshObj.MeshData;
            if (meshData == null)
                return;

            int baseIdx = triangleIndex * 3;
            if (baseIdx + 2 >= meshData.Indices.Count)
                return;

            // Ensure colors list is properly sized
            while (meshData.Colors.Count < meshData.Vertices.Count)
            {
                meshData.Colors.Add(new Vector3(0.8f));
            }

            // Set color for all three vertices of the triangle
            for (int i = 0; i < 3; i++)
            {
                int vIdx = meshData.Indices[baseIdx + i];
                if (vIdx >= 0 && vIdx < meshData.Colors.Count)
                {
                    meshData.Colors[vIdx] = color;
                }
            }
        }

        /// <summary>
        /// Flip normal direction of selected triangles
        /// </summary>
        public void FlipSelectedTriangles()
        {
            foreach (var (mesh, triIdx) in _selectedTriangles)
            {
                FlipTriangle(mesh, triIdx);
                MeshModified?.Invoke(this, mesh);
            }
        }

        /// <summary>
        /// Flip a single triangle (reverse winding order)
        /// </summary>
        private void FlipTriangle(MeshObject meshObj, int triangleIndex)
        {
            var meshData = meshObj.MeshData;
            if (meshData == null)
                return;

            int baseIdx = triangleIndex * 3;
            if (baseIdx + 2 >= meshData.Indices.Count)
                return;

            // Swap indices 1 and 2 to reverse winding
            (meshData.Indices[baseIdx + 1], meshData.Indices[baseIdx + 2]) =
                (meshData.Indices[baseIdx + 2], meshData.Indices[baseIdx + 1]);
        }

        /// <summary>
        /// Subdivide selected triangles (split each into 4)
        /// </summary>
        public void SubdivideSelectedTriangles()
        {
            var byMesh = _selectedTriangles.GroupBy(x => x.mesh);

            foreach (var group in byMesh)
            {
                var mesh = group.Key;
                var triangles = group.Select(x => x.triangleIndex).OrderByDescending(x => x).ToList();

                SubdivideTriangles(mesh, triangles);
                MeshModified?.Invoke(this, mesh);
            }

            _selectedTriangles.Clear();
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Subdivide triangles by adding midpoint vertices
        /// </summary>
        private void SubdivideTriangles(MeshObject meshObj, List<int> triangleIndices)
        {
            var meshData = meshObj.MeshData;
            if (meshData == null)
                return;

            // Process triangles in reverse order
            foreach (int triIdx in triangleIndices.OrderByDescending(x => x))
            {
                int baseIdx = triIdx * 3;
                if (baseIdx + 2 >= meshData.Indices.Count)
                    continue;

                int i0 = meshData.Indices[baseIdx];
                int i1 = meshData.Indices[baseIdx + 1];
                int i2 = meshData.Indices[baseIdx + 2];

                if (i0 >= meshData.Vertices.Count || i1 >= meshData.Vertices.Count || i2 >= meshData.Vertices.Count)
                    continue;

                var v0 = meshData.Vertices[i0];
                var v1 = meshData.Vertices[i1];
                var v2 = meshData.Vertices[i2];

                // Create midpoint vertices
                var m01 = (v0 + v1) * 0.5f;
                var m12 = (v1 + v2) * 0.5f;
                var m20 = (v2 + v0) * 0.5f;

                int im01 = meshData.Vertices.Count;
                int im12 = im01 + 1;
                int im20 = im01 + 2;

                meshData.Vertices.Add(m01);
                meshData.Vertices.Add(m12);
                meshData.Vertices.Add(m20);

                // Add colors for new vertices (average of edge colors)
                if (meshData.Colors.Count >= Math.Max(Math.Max(i0, i1), i2) + 1)
                {
                    var c0 = meshData.Colors[i0];
                    var c1 = meshData.Colors[i1];
                    var c2 = meshData.Colors[i2];

                    meshData.Colors.Add((c0 + c1) * 0.5f);
                    meshData.Colors.Add((c1 + c2) * 0.5f);
                    meshData.Colors.Add((c2 + c0) * 0.5f);
                }
                else
                {
                    meshData.Colors.Add(new Vector3(0.8f));
                    meshData.Colors.Add(new Vector3(0.8f));
                    meshData.Colors.Add(new Vector3(0.8f));
                }

                // Add UVs for new vertices if UVs exist
                if (meshData.UVs.Count >= Math.Max(Math.Max(i0, i1), i2) + 1)
                {
                    var uv0 = meshData.UVs[i0];
                    var uv1 = meshData.UVs[i1];
                    var uv2 = meshData.UVs[i2];

                    meshData.UVs.Add((uv0 + uv1) * 0.5f);
                    meshData.UVs.Add((uv1 + uv2) * 0.5f);
                    meshData.UVs.Add((uv2 + uv0) * 0.5f);
                }

                // Remove original triangle
                meshData.Indices.RemoveRange(baseIdx, 3);

                // Add 4 new triangles
                // Triangle 0: v0, m01, m20
                meshData.Indices.AddRange(new[] { i0, im01, im20 });
                // Triangle 1: m01, v1, m12
                meshData.Indices.AddRange(new[] { im01, i1, im12 });
                // Triangle 2: m20, m12, v2
                meshData.Indices.AddRange(new[] { im20, im12, i2 });
                // Triangle 3: m01, m12, m20 (center)
                meshData.Indices.AddRange(new[] { im01, im12, im20 });
            }

            meshObj.UpdateBounds();
        }

        /// <summary>
        /// Weld vertices within a threshold distance for selected triangles
        /// </summary>
        public void WeldSelectedVertices(float threshold = 0.001f)
        {
            var meshes = _selectedTriangles.Select(x => x.mesh).Distinct().ToList();

            foreach (var mesh in meshes)
            {
                WeldVertices(mesh, threshold);
                MeshModified?.Invoke(this, mesh);
            }
        }

        /// <summary>
        /// Weld duplicate vertices in a mesh
        /// </summary>
        private void WeldVertices(MeshObject meshObj, float threshold)
        {
            var meshData = meshObj.MeshData;
            if (meshData == null || meshData.Vertices.Count == 0)
                return;

            var uniqueVertices = new List<Vector3>();
            var uniqueColors = new List<Vector3>();
            var uniqueUVs = new List<Vector2>();
            var indexMap = new int[meshData.Vertices.Count];

            float thresholdSq = threshold * threshold;

            for (int i = 0; i < meshData.Vertices.Count; i++)
            {
                var v = meshData.Vertices[i];
                int foundIdx = -1;

                // Check if this vertex is close to an existing unique vertex
                for (int j = 0; j < uniqueVertices.Count; j++)
                {
                    if ((uniqueVertices[j] - v).LengthSquared < thresholdSq)
                    {
                        foundIdx = j;
                        break;
                    }
                }

                if (foundIdx >= 0)
                {
                    indexMap[i] = foundIdx;
                }
                else
                {
                    indexMap[i] = uniqueVertices.Count;
                    uniqueVertices.Add(v);

                    if (i < meshData.Colors.Count)
                        uniqueColors.Add(meshData.Colors[i]);
                    else
                        uniqueColors.Add(new Vector3(0.8f));

                    if (i < meshData.UVs.Count)
                        uniqueUVs.Add(meshData.UVs[i]);
                }
            }

            // Update indices
            for (int i = 0; i < meshData.Indices.Count; i++)
            {
                meshData.Indices[i] = indexMap[meshData.Indices[i]];
            }

            // Remove degenerate triangles (where indices are the same)
            for (int i = meshData.Indices.Count - 3; i >= 0; i -= 3)
            {
                int i0 = meshData.Indices[i];
                int i1 = meshData.Indices[i + 1];
                int i2 = meshData.Indices[i + 2];

                if (i0 == i1 || i1 == i2 || i2 == i0)
                {
                    meshData.Indices.RemoveRange(i, 3);
                }
            }

            meshData.Vertices = uniqueVertices;
            meshData.Colors = uniqueColors;
            meshData.UVs = uniqueUVs;
            meshObj.UpdateBounds();
        }

        /// <summary>
        /// Select all triangles in a mesh
        /// </summary>
        public void SelectAll(MeshObject mesh)
        {
            if (mesh?.MeshData == null)
                return;

            int triangleCount = mesh.MeshData.Indices.Count / 3;
            for (int i = 0; i < triangleCount; i++)
            {
                _selectedTriangles.Add((mesh, i));
            }

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Invert selection for a mesh
        /// </summary>
        public void InvertSelection(MeshObject mesh)
        {
            if (mesh?.MeshData == null)
                return;

            int triangleCount = mesh.MeshData.Indices.Count / 3;
            var currentSelection = _selectedTriangles.Where(x => x.mesh == mesh).Select(x => x.triangleIndex).ToHashSet();

            // Remove existing selection for this mesh
            _selectedTriangles.RemoveWhere(x => x.mesh == mesh);

            // Add non-selected triangles
            for (int i = 0; i < triangleCount; i++)
            {
                if (!currentSelection.Contains(i))
                {
                    _selectedTriangles.Add((mesh, i));
                }
            }

            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Grow selection to include adjacent triangles
        /// </summary>
        public void GrowSelection()
        {
            var newSelection = new HashSet<(MeshObject mesh, int triangleIndex)>(_selectedTriangles);

            foreach (var (mesh, triIdx) in _selectedTriangles)
            {
                var adjacent = GetAdjacentTriangles(mesh, triIdx);
                foreach (var adj in adjacent)
                {
                    newSelection.Add((mesh, adj));
                }
            }

            _selectedTriangles = newSelection;
            SelectionChanged?.Invoke(this, EventArgs.Empty);
        }

        /// <summary>
        /// Get triangles adjacent to a given triangle (share at least one vertex)
        /// </summary>
        private List<int> GetAdjacentTriangles(MeshObject meshObj, int triangleIndex)
        {
            var result = new List<int>();
            var meshData = meshObj.MeshData;

            if (meshData == null)
                return result;

            int baseIdx = triangleIndex * 3;
            if (baseIdx + 2 >= meshData.Indices.Count)
                return result;

            var triVertices = new HashSet<int>
            {
                meshData.Indices[baseIdx],
                meshData.Indices[baseIdx + 1],
                meshData.Indices[baseIdx + 2]
            };

            int triangleCount = meshData.Indices.Count / 3;
            for (int i = 0; i < triangleCount; i++)
            {
                if (i == triangleIndex)
                    continue;

                int bi = i * 3;
                if (triVertices.Contains(meshData.Indices[bi]) ||
                    triVertices.Contains(meshData.Indices[bi + 1]) ||
                    triVertices.Contains(meshData.Indices[bi + 2]))
                {
                    result.Add(i);
                }
            }

            return result;
        }

        /// <summary>
        /// Get triangle vertices for rendering highlights
        /// </summary>
        public List<Vector3> GetSelectedTriangleVertices()
        {
            var vertices = new List<Vector3>();

            foreach (var (mesh, triIdx) in _selectedTriangles)
            {
                var meshData = mesh.MeshData;
                if (meshData == null)
                    continue;

                int baseIdx = triIdx * 3;
                if (baseIdx + 2 >= meshData.Indices.Count)
                    continue;

                var worldTransform = mesh.GetWorldTransform();

                for (int i = 0; i < 3; i++)
                {
                    int vIdx = meshData.Indices[baseIdx + i];
                    if (vIdx >= 0 && vIdx < meshData.Vertices.Count)
                    {
                        var localPos = meshData.Vertices[vIdx];
                        var worldPos = Vector3.TransformPosition(localPos, worldTransform);
                        vertices.Add(worldPos);
                    }
                }
            }

            return vertices;
        }

        /// <summary>
        /// Get statistics about current selection
        /// </summary>
        public (int triangleCount, int vertexCount, int meshCount) GetSelectionStats()
        {
            int triangleCount = _selectedTriangles.Count;
            int meshCount = _selectedTriangles.Select(x => x.mesh).Distinct().Count();

            var uniqueVertices = new HashSet<(MeshObject, int)>();
            foreach (var (mesh, triIdx) in _selectedTriangles)
            {
                var meshData = mesh.MeshData;
                if (meshData == null)
                    continue;

                int baseIdx = triIdx * 3;
                if (baseIdx + 2 >= meshData.Indices.Count)
                    continue;

                for (int i = 0; i < 3; i++)
                {
                    uniqueVertices.Add((mesh, meshData.Indices[baseIdx + i]));
                }
            }

            return (triangleCount, uniqueVertices.Count, meshCount);
        }
    }
}
