using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Deep3DStudio.Model
{
    public class MeshData
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Vector3> Colors { get; set; } = new List<Vector3>();
        public List<int> Indices { get; set; } = new List<int>();

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
            // Convert to OpenTK Matrix4. Note: OpenTK Matrix4 is Row-Major in memory (compatible with v * M).
            // We construct M such that M corresponds to the Transpose of the rotation matrix derived here (which assumes col vectors).

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

            return ComputeRigidTransform(srcPts, dstPts);
        }

        public static MeshData GenerateMeshFromDepth(Tensor<float> ptsTensor, Tensor<float> confTensor, SixLabors.ImageSharp.Color[] colors, int width, int height)
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
                    float conf = confTensor[0, y, x];

                    if (conf > confThreshold)
                    {
                        float px = ptsTensor[0, y, x, 0];
                        float py = ptsTensor[0, y, x, 1];
                        float pz = ptsTensor[0, y, x, 2];

                        var c = colors[pIdx];
                        var pixel = c.ToPixel<SixLabors.ImageSharp.PixelFormats.Rgb24>();

                        mesh.Vertices.Add(new Vector3(px, py, pz));
                        mesh.Colors.Add(new Vector3(pixel.R/255f, pixel.G/255f, pixel.B/255f));
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

                    // Triangle 1: TL, BL, TR
                    if (hasTL && hasBL && hasTR)
                    {
                        if (IsValidTriangle(mesh.Vertices[idxTL], mesh.Vertices[idxBL], mesh.Vertices[idxTR], edgeThreshold))
                        {
                            mesh.Indices.Add(idxTL);
                            mesh.Indices.Add(idxBL);
                            mesh.Indices.Add(idxTR);
                        }
                    }

                    // Triangle 2: TR, BL, BR
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
            // Check edge lengths to avoid connecting depth discontinuities
            if ((v1 - v2).LengthSquared > threshold * threshold) return false;
            if ((v2 - v3).LengthSquared > threshold * threshold) return false;
            if ((v3 - v1).LengthSquared > threshold * threshold) return false;
            return true;
        }

        public static void CropMesh(MeshData mesh, Vector3 min, Vector3 max)
        {
            // 1. Identify valid vertices
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

            // 2. Rebuild indices
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

            // 3. Update Mesh
            mesh.Vertices = newVertices;
            mesh.Colors = newColors;
            mesh.Indices = newIndices;
            mesh.PixelToVertexIndex = null; // Index map invalid after geometric modification
        }
    }
}
