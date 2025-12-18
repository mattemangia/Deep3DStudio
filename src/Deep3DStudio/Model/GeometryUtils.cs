using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using MathNet.Numerics.LinearAlgebra;
using SkiaSharp;
using System.Linq;

namespace Deep3DStudio.Model
{
    public class MeshData
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Vector3> Normals { get; set; } = new List<Vector3>();
        public List<Vector3> Colors { get; set; } = new List<Vector3>();
        public List<Vector2> UVs { get; set; } = new List<Vector2>();
        public List<int> Indices { get; set; } = new List<int>();

        // Texture data (if any)
        public SKBitmap? Texture { get; set; }
        public int TextureId { get; set; } = -1; // OpenGL Texture ID

        // Maps original pixel index (y * width + x) to vertex index in Vertices list.
        // Value is -1 if no vertex was generated for that pixel (filtered out).
        public int[]? PixelToVertexIndex { get; set; }

        public void ApplyTransform(Matrix4 transform)
        {
            for(int i=0; i<Vertices.Count; i++)
            {
                var v = new Vector4(Vertices[i], 1.0f);
                var res = v * transform;
                Vertices[i] = new Vector3(res.X / res.W, res.Y / res.W, res.Z / res.W);
            }
        }

        public MeshData Clone()
        {
            var clone = new MeshData();
            clone.Vertices = new List<Vector3>(Vertices);
            clone.Normals = new List<Vector3>(Normals);
            clone.Colors = new List<Vector3>(Colors);
            clone.UVs = new List<Vector2>(UVs);
            clone.Indices = new List<int>(Indices);
            clone.Texture = Texture; // Shallow copy bitmap ref
            clone.TextureId = TextureId;
            if (PixelToVertexIndex != null)
                clone.PixelToVertexIndex = (int[])PixelToVertexIndex.Clone();
            return clone;
        }

        public void RecalculateNormals()
        {
            Normals.Clear();
            if (Vertices.Count == 0) return;

            // Initialize normals with zero vectors
            for (int i = 0; i < Vertices.Count; i++)
            {
                Normals.Add(Vector3.Zero);
            }

            // Accumulate face normals
            for (int i = 0; i < Indices.Count; i += 3)
            {
                int i1 = Indices[i];
                int i2 = Indices[i + 1];
                int i3 = Indices[i + 2];

                if (i1 >= Vertices.Count || i2 >= Vertices.Count || i3 >= Vertices.Count) continue;

                Vector3 v1 = Vertices[i1];
                Vector3 v2 = Vertices[i2];
                Vector3 v3 = Vertices[i3];

                Vector3 edge1 = v2 - v1;
                Vector3 edge2 = v3 - v1;
                Vector3 normal = Vector3.Cross(edge1, edge2);

                // Add weighted by area (cross product magnitude)
                Normals[i1] += normal;
                Normals[i2] += normal;
                Normals[i3] += normal;
            }

            // Normalize
            for (int i = 0; i < Normals.Count; i++)
            {
                if (Normals[i].LengthSquared > 1e-6f)
                {
                    Normals[i] = Normals[i].Normalized();
                }
                else
                {
                    Normals[i] = Vector3.UnitY; // Default up
                }
            }
        }

        public Box3 GetBounds()
        {
            if (Vertices.Count == 0)
                return new Box3(Vector3.Zero, Vector3.Zero);

            Vector3 min = new Vector3(float.MaxValue);
            Vector3 max = new Vector3(float.MinValue);

            foreach (var v in Vertices)
            {
                min = Vector3.ComponentMin(min, v);
                max = Vector3.ComponentMax(max, v);
            }
            return new Box3(min, max);
        }
    }

    public static class GeometryUtils
    {
        /// <summary>
        /// Computes the optimal rigid transformation (Rotation + Translation) aligning source points to destination points
        /// using the Kabsch algorithm (SVD).
        /// </summary>
        public static Matrix4 ComputeRigidTransform(List<Vector3> srcPoints, List<Vector3> dstPoints)
        {
            if (srcPoints.Count != dstPoints.Count || srcPoints.Count < 3)
                return Matrix4.Identity;

            // 1. Compute Centroids
            Vector3 centroidSrc = Vector3.Zero;
            Vector3 centroidDst = Vector3.Zero;
            for(int i=0; i<srcPoints.Count; i++)
            {
                centroidSrc += srcPoints[i];
                centroidDst += dstPoints[i];
            }
            centroidSrc /= srcPoints.Count;
            centroidDst /= dstPoints.Count;

            // 2. Center points & Build Covariance Matrix H = Sum ( (p_src - c_src) * (p_dst - c_dst)^T )
            var matH = Matrix<double>.Build.Dense(3, 3);

            for(int i=0; i<srcPoints.Count; i++)
            {
                var pS = srcPoints[i] - centroidSrc;
                var pD = dstPoints[i] - centroidDst;

                matH[0,0] += pS.X * pD.X; matH[0,1] += pS.X * pD.Y; matH[0,2] += pS.X * pD.Z;
                matH[1,0] += pS.Y * pD.X; matH[1,1] += pS.Y * pD.Y; matH[1,2] += pS.Y * pD.Z;
                matH[2,0] += pS.Z * pD.X; matH[2,1] += pS.Z * pD.Y; matH[2,2] += pS.Z * pD.Z;
            }

            // 3. SVD
            var svd = matH.Svd(true);
            var U = svd.U;
            var Vt = svd.VT;
            var V = Vt.Transpose();

            // 4. Rotation R = V * U^T
            var R_mat = V * U.Transpose();

            // Check reflection
            if (R_mat.Determinant() < 0)
            {
                var V_prime = V.Clone();
                V_prime.SetColumn(2, V_prime.Column(2).Multiply(-1));
                R_mat = V_prime * U.Transpose();
            }

            // 5. Translation t = c_dst - R * c_src
            var M = Matrix4.Identity;
            M.M11 = (float)R_mat[0,0]; M.M12 = (float)R_mat[1,0]; M.M13 = (float)R_mat[2,0];
            M.M21 = (float)R_mat[0,1]; M.M22 = (float)R_mat[1,1]; M.M23 = (float)R_mat[2,1];
            M.M31 = (float)R_mat[0,2]; M.M32 = (float)R_mat[1,2]; M.M33 = (float)R_mat[2,2];

            double tx = centroidDst.X - (R_mat[0,0]*centroidSrc.X + R_mat[0,1]*centroidSrc.Y + R_mat[0,2]*centroidSrc.Z);
            double ty = centroidDst.Y - (R_mat[1,0]*centroidSrc.X + R_mat[1,1]*centroidSrc.Y + R_mat[1,2]*centroidSrc.Z);
            double tz = centroidDst.Z - (R_mat[2,0]*centroidSrc.X + R_mat[2,1]*centroidSrc.Y + R_mat[2,2]*centroidSrc.Z);

            M.M41 = (float)tx;
            M.M42 = (float)ty;
            M.M43 = (float)tz;

            return M;
        }

        public static Matrix4 ComputeTransformFromCorrespondences(MeshData source, MeshData target)
        {
            if (source.PixelToVertexIndex == null || target.PixelToVertexIndex == null)
                return Matrix4.Identity;

            if (source.PixelToVertexIndex.Length != target.PixelToVertexIndex.Length)
                return Matrix4.Identity;

            var srcPts = new List<Vector3>();
            var dstPts = new List<Vector3>();

            for(int i=0; i<source.PixelToVertexIndex.Length; i++)
            {
                int idxS = source.PixelToVertexIndex[i];
                int idxT = target.PixelToVertexIndex[i];

                if (idxS != -1 && idxT != -1)
                {
                    srcPts.Add(source.Vertices[idxS]);
                    dstPts.Add(target.Vertices[idxT]);
                }
            }

            if (srcPts.Count < 3) return Matrix4.Identity;
            return ComputeRigidTransformRANSAC(srcPts, dstPts, out _, out _);
        }

        public static Matrix4 ComputeRigidTransformRANSAC(
            List<Vector3> srcPoints,
            List<Vector3> dstPoints,
            out int inlierCount,
            out float rmse,
            int maxIterations = 100,
            float inlierThreshold = 0.05f)
        {
            inlierCount = 0;
            rmse = float.MaxValue;

            if (srcPoints.Count != dstPoints.Count || srcPoints.Count < 4)
                return ComputeRigidTransform(srcPoints, dstPoints);

            int n = srcPoints.Count;
            var rnd = new Random(42);
            Matrix4 bestTransform = Matrix4.Identity;
            int bestInliers = 0;
            float bestRMSE = float.MaxValue;

            if (n < 10)
            {
                bestTransform = ComputeRigidTransform(srcPoints, dstPoints);
                (bestInliers, bestRMSE) = CountInliersAndRMSE(srcPoints, dstPoints, bestTransform, inlierThreshold);
                inlierCount = bestInliers;
                rmse = bestRMSE;
                return bestTransform;
            }

            for (int iter = 0; iter < maxIterations; iter++)
            {
                var indices = new HashSet<int>();
                while (indices.Count < 4)
                    indices.Add(rnd.Next(n));

                var sampleSrc = new List<Vector3>();
                var sampleDst = new List<Vector3>();
                foreach (int idx in indices)
                {
                    sampleSrc.Add(srcPoints[idx]);
                    sampleDst.Add(dstPoints[idx]);
                }

                var candidateTransform = ComputeRigidTransform(sampleSrc, sampleDst);
                var (currentInliers, currentRMSE) = CountInliersAndRMSE(srcPoints, dstPoints, candidateTransform, inlierThreshold);

                if (currentInliers > bestInliers || (currentInliers == bestInliers && currentRMSE < bestRMSE))
                {
                    bestInliers = currentInliers;
                    bestRMSE = currentRMSE;
                    bestTransform = candidateTransform;
                }
                if (bestInliers > n * 0.8) break;
            }

            if (bestInliers >= 4)
            {
                var inlierSrc = new List<Vector3>();
                var inlierDst = new List<Vector3>();

                for (int i = 0; i < n; i++)
                {
                    var transformed = TransformPoint(srcPoints[i], bestTransform);
                    float dist = (transformed - dstPoints[i]).Length;
                    if (dist < inlierThreshold)
                    {
                        inlierSrc.Add(srcPoints[i]);
                        inlierDst.Add(dstPoints[i]);
                    }
                }

                if (inlierSrc.Count >= 4)
                {
                    bestTransform = ComputeRigidTransform(inlierSrc, inlierDst);
                    (bestInliers, bestRMSE) = CountInliersAndRMSE(srcPoints, dstPoints, bestTransform, inlierThreshold);
                }
            }

            inlierCount = bestInliers;
            rmse = bestRMSE;
            return bestTransform;
        }

        private static (int inliers, float rmse) CountInliersAndRMSE(
            List<Vector3> src, List<Vector3> dst, Matrix4 transform, float threshold)
        {
            int inliers = 0;
            float sumSqError = 0;

            for (int i = 0; i < src.Count; i++)
            {
                var transformed = TransformPoint(src[i], transform);
                float dist = (transformed - dst[i]).Length;
                if (dist < threshold)
                {
                    inliers++;
                    sumSqError += dist * dist;
                }
            }

            float rmse = inliers > 0 ? (float)Math.Sqrt(sumSqError / inliers) : float.MaxValue;
            return (inliers, rmse);
        }

        public static Vector3 TransformPoint(Vector3 point, Matrix4 transform)
        {
            var v = new Vector4(point, 1.0f);
            var result = v * transform;
            return new Vector3(result.X / result.W, result.Y / result.W, result.Z / result.W);
        }

        public static (float overlapRatio, float rmse, int correspondences) ComputeAlignmentQuality(
            MeshData source, MeshData target, Matrix4 transform, float distanceThreshold = 0.1f)
        {
            if (source.Vertices.Count == 0 || target.Vertices.Count == 0)
                return (0, float.MaxValue, 0);

            var targetSet = new HashSet<(int, int, int)>();
            float cellSize = distanceThreshold;

            foreach (var v in target.Vertices)
            {
                int cx = (int)(v.X / cellSize);
                int cy = (int)(v.Y / cellSize);
                int cz = (int)(v.Z / cellSize);
                targetSet.Add((cx, cy, cz));
            }

            int matches = 0;
            float sumSqDist = 0;

            foreach (var v in source.Vertices)
            {
                var transformed = TransformPoint(v, transform);
                int cx = (int)(transformed.X / cellSize);
                int cy = (int)(transformed.Y / cellSize);
                int cz = (int)(transformed.Z / cellSize);

                bool found = false;
                for (int dx = -1; dx <= 1 && !found; dx++)
                    for (int dy = -1; dy <= 1 && !found; dy++)
                        for (int dz = -1; dz <= 1 && !found; dz++)
                            if (targetSet.Contains((cx + dx, cy + dy, cz + dz)))
                                found = true;

                if (found)
                {
                    matches++;
                    sumSqDist += cellSize * cellSize * 0.5f;
                }
            }

            float overlap = (float)matches / source.Vertices.Count;
            float rmse = matches > 0 ? (float)Math.Sqrt(sumSqDist / matches) : float.MaxValue;

            return (overlap, rmse, matches);
        }

        public static MeshData GenerateMeshFromDepth(float[] pts, float[] conf, SKColor[] colors, int width, int height)
        {
            var mesh = new MeshData();
            mesh.PixelToVertexIndex = new int[width * height];

            float confThreshold = 1.1f;
            float edgeThreshold = 0.2f;

            // 1. Vertices
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pIdx = y * width + x;

                    // Access conf: [1, y, x] -> y * width + x
                    float confidence = conf[pIdx];

                    if (confidence > confThreshold)
                    {
                        // Access pts: [1, y, x, 3] -> (y * width + x) * 3 + c
                        float px = pts[pIdx * 3 + 0];
                        float py = pts[pIdx * 3 + 1];
                        float pz = pts[pIdx * 3 + 2];

                        var c = colors[pIdx];

                        mesh.Vertices.Add(new Vector3(px, py, pz));
                        mesh.Colors.Add(new Vector3(c.Red/255f, c.Green/255f, c.Blue/255f));
                        mesh.PixelToVertexIndex[pIdx] = mesh.Vertices.Count - 1;
                    }
                    else
                    {
                        mesh.PixelToVertexIndex[pIdx] = -1;
                    }
                }
            }

            // 2. Indices (Triangles)
            for (int y = 0; y < height - 1; y++)
            {
                for (int x = 0; x < width - 1; x++)
                {
                    int pTL = y * width + x;
                    int pTR = y * width + x + 1;
                    int pBL = (y+1) * width + x;
                    int pBR = (y+1) * width + x + 1;

                    int idxTL = mesh.PixelToVertexIndex[pTL];
                    int idxTR = mesh.PixelToVertexIndex[pTR];
                    int idxBL = mesh.PixelToVertexIndex[pBL];
                    int idxBR = mesh.PixelToVertexIndex[pBR];

                    bool hasTL = idxTL != -1;
                    bool hasTR = idxTR != -1;
                    bool hasBL = idxBL != -1;
                    bool hasBR = idxBR != -1;

                    if (hasTL && hasBL && hasTR)
                    {
                        if (IsValidTriangle(mesh.Vertices[idxTL], mesh.Vertices[idxBL], mesh.Vertices[idxTR], edgeThreshold))
                        {
                            mesh.Indices.Add(idxTL);
                            mesh.Indices.Add(idxBL);
                            mesh.Indices.Add(idxTR);
                        }
                    }

                    if (hasTR && hasBL && hasBR)
                    {
                        if (IsValidTriangle(mesh.Vertices[idxTR], mesh.Vertices[idxBL], mesh.Vertices[idxBR], edgeThreshold))
                        {
                            mesh.Indices.Add(idxTR);
                            mesh.Indices.Add(idxBL);
                            mesh.Indices.Add(idxBR);
                        }
                    }
                }
            }

            return mesh;
        }

        private static bool IsValidTriangle(Vector3 v1, Vector3 v2, Vector3 v3, float threshold)
        {
            if ((v1 - v2).LengthSquared > threshold * threshold) return false;
            if ((v2 - v3).LengthSquared > threshold * threshold) return false;
            if ((v3 - v1).LengthSquared > threshold * threshold) return false;
            return true;
        }

        public static void CropMesh(MeshData mesh, Vector3 min, Vector3 max)
        {
            int[] oldToNew = new int[mesh.Vertices.Count];
            var newVertices = new List<Vector3>();
            var newColors = new List<Vector3>();

            for(int i=0; i<mesh.Vertices.Count; i++)
            {
                Vector3 v = mesh.Vertices[i];
                if (v.X >= min.X && v.X <= max.X &&
                    v.Y >= min.Y && v.Y <= max.Y &&
                    v.Z >= min.Z && v.Z <= max.Z)
                {
                    oldToNew[i] = newVertices.Count;
                    newVertices.Add(v);
                    newColors.Add(mesh.Colors[i]);
                }
                else
                {
                    oldToNew[i] = -1;
                }
            }

            var newIndices = new List<int>();
            for(int i=0; i<mesh.Indices.Count; i+=3)
            {
                int i1 = mesh.Indices[i];
                int i2 = mesh.Indices[i+1];
                int i3 = mesh.Indices[i+2];

                if (oldToNew[i1] != -1 && oldToNew[i2] != -1 && oldToNew[i3] != -1)
                {
                    newIndices.Add(oldToNew[i1]);
                    newIndices.Add(oldToNew[i2]);
                    newIndices.Add(oldToNew[i3]);
                }
            }

            mesh.Vertices = newVertices;
            mesh.Colors = newColors;
            mesh.Indices = newIndices;
            mesh.PixelToVertexIndex = null;
        }

        public static bool RayBoxIntersection(Vector3 rayOrigin, Vector3 rayDir, Vector3 boxMin, Vector3 boxMax, out float tMin, out float tMax)
        {
            tMin = float.NegativeInfinity;
            tMax = float.PositiveInfinity;

            Vector3 invDir = new Vector3(1.0f / rayDir.X, 1.0f / rayDir.Y, 1.0f / rayDir.Z);

            float t1 = (boxMin.X - rayOrigin.X) * invDir.X;
            float t2 = (boxMax.X - rayOrigin.X) * invDir.X;
            tMin = Math.Max(tMin, Math.Min(t1, t2));
            tMax = Math.Min(tMax, Math.Max(t1, t2));

            t1 = (boxMin.Y - rayOrigin.Y) * invDir.Y;
            t2 = (boxMax.Y - rayOrigin.Y) * invDir.Y;
            tMin = Math.Max(tMin, Math.Min(t1, t2));
            tMax = Math.Min(tMax, Math.Max(t1, t2));

            t1 = (boxMin.Z - rayOrigin.Z) * invDir.Z;
            t2 = (boxMax.Z - rayOrigin.Z) * invDir.Z;
            tMin = Math.Max(tMin, Math.Min(t1, t2));
            tMax = Math.Min(tMax, Math.Max(t1, t2));

            return tMax >= tMin && tMax >= 0;
        }

        public static MeshData MarchingCubes(float[,,] density, Vector3[,,] color, Vector3 min, Vector3 voxelSize, float isoLevel)
        {
            var mesh = new MeshData();
            int resX = density.GetLength(0);
            int resY = density.GetLength(1);
            int resZ = density.GetLength(2);

            for (int x = 0; x < resX - 1; x++)
            {
                for (int y = 0; y < resY - 1; y++)
                {
                    for (int z = 0; z < resZ - 1; z++)
                    {
                        ProcessCube(x, y, z, density, color, min, voxelSize, isoLevel, mesh);
                    }
                }
            }
            return mesh;
        }

        private static void ProcessCube(int x, int y, int z, float[,,] density, Vector3[,,] color, Vector3 min, Vector3 voxelSize, float isoLevel, MeshData mesh)
        {
            float[] vals = new float[8];
            Vector3[] poss = new Vector3[8];
            Vector3[] cols = new Vector3[8];

            for(int i=0; i<8; i++)
            {
                int dx = (i & 1);
                int dy = (i & 2) >> 1;
                int dz = (i & 4) >> 2;
                vals[i] = density[x + dx, y + dy, z + dz];
                poss[i] = min + new Vector3((x + dx) * voxelSize.X, (y + dy) * voxelSize.Y, (z + dz) * voxelSize.Z);
                cols[i] = color[x + dx, y + dy, z + dz];
            }

            int cubeIndex = 0;
            if (vals[0] < isoLevel) cubeIndex |= 1;
            if (vals[1] < isoLevel) cubeIndex |= 2;
            if (vals[2] < isoLevel) cubeIndex |= 4;
            if (vals[3] < isoLevel) cubeIndex |= 8;
            if (vals[4] < isoLevel) cubeIndex |= 16;
            if (vals[5] < isoLevel) cubeIndex |= 32;
            if (vals[6] < isoLevel) cubeIndex |= 64;
            if (vals[7] < isoLevel) cubeIndex |= 128;

            if (edgeTable[cubeIndex] == 0) return;

            Vector3[] vertList = new Vector3[12];
            Vector3[] colList = new Vector3[12];

            if ((edgeTable[cubeIndex] & 1) != 0) VertexInterp(isoLevel, poss[0], poss[1], vals[0], vals[1], cols[0], cols[1], out vertList[0], out colList[0]);
            if ((edgeTable[cubeIndex] & 2) != 0) VertexInterp(isoLevel, poss[1], poss[2], vals[1], vals[2], cols[1], cols[2], out vertList[1], out colList[1]);
            if ((edgeTable[cubeIndex] & 4) != 0) VertexInterp(isoLevel, poss[2], poss[3], vals[2], vals[3], cols[2], cols[3], out vertList[2], out colList[2]);
            if ((edgeTable[cubeIndex] & 8) != 0) VertexInterp(isoLevel, poss[3], poss[0], vals[3], vals[0], cols[3], cols[0], out vertList[3], out colList[3]);
            if ((edgeTable[cubeIndex] & 16) != 0) VertexInterp(isoLevel, poss[4], poss[5], vals[4], vals[5], cols[4], cols[5], out vertList[4], out colList[4]);
            if ((edgeTable[cubeIndex] & 32) != 0) VertexInterp(isoLevel, poss[5], poss[6], vals[5], vals[6], cols[5], cols[6], out vertList[5], out colList[5]);
            if ((edgeTable[cubeIndex] & 64) != 0) VertexInterp(isoLevel, poss[6], poss[7], vals[6], vals[7], cols[6], cols[7], out vertList[6], out colList[6]);
            if ((edgeTable[cubeIndex] & 128) != 0) VertexInterp(isoLevel, poss[7], poss[4], vals[7], vals[4], cols[7], cols[4], out vertList[7], out colList[7]);
            if ((edgeTable[cubeIndex] & 256) != 0) VertexInterp(isoLevel, poss[0], poss[4], vals[0], vals[4], cols[0], cols[4], out vertList[8], out colList[8]);
            if ((edgeTable[cubeIndex] & 512) != 0) VertexInterp(isoLevel, poss[1], poss[5], vals[1], vals[5], cols[1], cols[5], out vertList[9], out colList[9]);
            if ((edgeTable[cubeIndex] & 1024) != 0) VertexInterp(isoLevel, poss[2], poss[6], vals[2], vals[6], cols[2], cols[6], out vertList[10], out colList[10]);
            if ((edgeTable[cubeIndex] & 2048) != 0) VertexInterp(isoLevel, poss[3], poss[7], vals[3], vals[7], cols[3], cols[7], out vertList[11], out colList[11]);

            for (int i = 0; i < 15 && triTable[cubeIndex, i] != -1; i += 3)
            {
                int i1 = triTable[cubeIndex, i];
                int i2 = triTable[cubeIndex, i + 1];
                int i3 = triTable[cubeIndex, i + 2];

                mesh.Vertices.Add(vertList[i1]);
                mesh.Colors.Add(colList[i1]);
                mesh.Indices.Add(mesh.Vertices.Count - 1);

                mesh.Vertices.Add(vertList[i2]);
                mesh.Colors.Add(colList[i2]);
                mesh.Indices.Add(mesh.Vertices.Count - 1);

                mesh.Vertices.Add(vertList[i3]);
                mesh.Colors.Add(colList[i3]);
                mesh.Indices.Add(mesh.Vertices.Count - 1);
            }
        }

        private static void VertexInterp(float isoLevel, Vector3 p1, Vector3 p2, float v1, float v2, Vector3 c1, Vector3 c2, out Vector3 p, out Vector3 c)
        {
             if (Math.Abs(isoLevel - v1) < 0.00001f)
             {
                 p = p1;
                 c = c1;
                 return;
             }
             if (Math.Abs(isoLevel - v2) < 0.00001f)
             {
                 p = p2;
                 c = c2;
                 return;
             }
             if (Math.Abs(v1 - v2) < 0.00001f)
             {
                 p = p1;
                 c = c1;
                 return;
             }
             float mu = (isoLevel - v1) / (v2 - v1);
             p = p1 + mu * (p2 - p1);
             c = c1 + mu * (c2 - c1);
        }

        public static int[] GetMarchingCubesEdges(int cubeIndex)
        {
            var edges = new List<int>();
            for (int i = 0; i < 15; i++)
            {
                int edge = triTable[cubeIndex, i];
                if (edge == -1) break;
                edges.Add(edge);
            }
            var result = new int[edges.Count + 1];
            for(int i=0; i<edges.Count; i++) result[i] = edges[i];
            result[edges.Count] = -1;
            return result;
        }

        public static int[] edgeTable = new int[]{
            0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
            0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
            0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
            0x3a0, 0x2a9, 0x1a3, 0xa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
            0x460, 0x569, 0x663, 0x76a, 0x6, 0x10f, 0x205, 0x30c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
            0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
            0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
            0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbc0, 0xac3, 0x9c9, 0x8c0,
            0x8c0, 0x9c9, 0xac3, 0xbc0, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
            0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
            0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
            0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x30c, 0x205, 0x10f, 0x6, 0x76a, 0x663, 0x569, 0x460,
            0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xa, 0x1a3, 0x2a9, 0x3a0,
            0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
            0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
            0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
        };

        private static int[,] triTable = new int[,]{
            {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
            {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
            {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
            {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
            {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
            {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
            {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
            {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
            {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
            {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
            {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
            {3, 10, 1, 3, 11, 10, 7, 5, 4, -1, -1, -1, -1, -1, -1, -1},
            {0, 10, 1, 0, 8, 10, 8, 11, 10, 4, 7, 8, -1, -1, -1, -1},
            {9, 0, 3, 9, 3, 11, 9, 11, 7, 9, 7, 4, 11, 10, 7, -1},
            {9, 8, 7, 9, 7, 10, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
            {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
            {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
            {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
            {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
            {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
            {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
            {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
            {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
            {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
            {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
            {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
            {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
            {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
            {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
            {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
            {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
            {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
            {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
            {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
            {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
            {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
            {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
            {9, 0, 1, 5, 7, 4, 8, 7, 4, 11, 7, 8, 11, 8, 10, -1},
            {3, 0, 1, 3, 1, 5, 3, 5, 7, 3, 7, 11, -1, -1, -1, -1},
            {11, 10, 1, 11, 1, 5, 11, 5, 7, -1, -1, -1, -1, -1, -1, -1},
            {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
            {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
            {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
            {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
            {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
            {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
            {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
            {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
            {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
            {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
            {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
            {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
            {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
            {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
            {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
            {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
            {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
            {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
            {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
            {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
            {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
            {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
            {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
            {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
            {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 8, 4, -1, -1, -1, -1},
            {5, 10, 6, 4, 7, 9, 9, 7, 11, 9, 11, 0, 11, 3, 0, -1},
            {7, 4, 9, 7, 9, 11, 9, 6, 11, 9, 5, 6, -1, -1, -1, -1},
            {9, 6, 10, 9, 7, 6, 7, 8, 6, -1, -1, -1, -1, -1, -1, -1},
            {0, 8, 3, 9, 7, 6, 9, 6, 10, -1, -1, -1, -1, -1, -1, -1},
            {0, 1, 10, 0, 10, 6, 6, 10, 7, 0, 6, 9, -1, -1, -1, -1},
            {0, 1, 3, 10, 6, 8, 10, 8, 7, 6, 3, 8, -1, -1, -1, -1},
            {7, 8, 6, 7, 6, 10, 6, 8, 2, 10, 6, 1, 2, 8, 1, -1},
            {1, 3, 8, 1, 8, 2, 10, 7, 6, -1, -1, -1, -1, -1, -1, -1},
            {10, 7, 6, 9, 0, 2, 9, 2, 8, 2, 0, 6, 0, 9, 6, -1},
            {6, 10, 7, 2, 5, 8, 2, 8, 1, 5, 8, 3, 5, 3, 9, -1},
            {2, 3, 11, 10, 7, 9, 10, 9, 6, 9, 7, 8, -1, -1, -1, -1},
            {0, 11, 2, 0, 3, 11, 10, 7, 9, 10, 9, 6, 9, 7, 8, -1},
            {6, 10, 7, 6, 7, 9, 9, 7, 8, 0, 1, 2, 2, 1, 3, 2},
            {11, 2, 8, 11, 8, 9, 8, 2, 1, 9, 8, 6, 7, 6, 8, 6},
            {6, 1, 10, 6, 5, 1, 5, 0, 1, 3, 11, 7, 3, 7, 2, -1},
            {11, 6, 5, 11, 5, 2, 2, 5, 0, 2, 0, 1, 6, 11, 7, 11},
            {9, 6, 5, 9, 5, 3, 3, 5, 2, 3, 2, 11, 5, 6, 10, -1},
            {5, 11, 2, 5, 2, 3, 3, 2, 8, 3, 8, 9, 5, 3, 6, 10},
            {11, 10, 6, 11, 6, 5, 11, 5, 4, 11, 4, 7, -1, -1, -1, -1},
            {0, 8, 3, 5, 4, 7, 5, 7, 6, 6, 7, 11, 6, 11, 10, -1},
            {11, 10, 6, 11, 6, 4, 4, 6, 5, 0, 1, 9, 4, 5, 7, -1},
            {6, 5, 4, 6, 4, 1, 1, 4, 8, 1, 8, 3, 10, 6, 11, 7},
            {6, 10, 1, 6, 1, 5, 1, 2, 5, 4, 7, 5, 7, 11, 5, 7},
            {8, 4, 7, 6, 0, 5, 6, 5, 11, 11, 5, 2, 0, 2, 5, 10},
            {6, 4, 9, 6, 9, 5, 9, 1, 5, 9, 0, 1, 4, 6, 7, 11},
            {11, 2, 6, 11, 6, 7, 7, 6, 4, 7, 4, 8, 6, 2, 10, 5},
            {4, 5, 7, 4, 7, 3, 3, 7, 11, 5, 6, 7, 6, 10, 7, -1},
            {0, 8, 5, 0, 5, 6, 6, 5, 10, 8, 7, 5, 10, 11, 6, 7},
            {1, 9, 4, 1, 4, 5, 1, 5, 3, 5, 4, 7, 5, 7, 6, 6},
            {3, 9, 4, 3, 4, 8, 4, 9, 5, 4, 5, 6, 7, 6, 5, 11},
            {1, 2, 10, 1, 10, 6, 1, 6, 5, 5, 6, 4, 6, 10, 11, 4},
            {0, 8, 4, 0, 4, 5, 0, 5, 6, 6, 5, 11, 4, 5, 7, 11},
            {2, 11, 4, 2, 4, 5, 2, 5, 1, 5, 4, 9, 5, 9, 0, 6},
            {6, 10, 1, 6, 1, 5, 1, 2, 5, 5, 9, 8, 5, 8, 4, 4}
        };
    }
}
