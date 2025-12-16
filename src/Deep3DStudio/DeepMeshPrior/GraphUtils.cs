using System;
using System.Collections.Generic;
using System.Linq;
using Deep3DStudio.Scene;
using Deep3DStudio.Model;
using OpenTK.Mathematics;
using TorchSharp;
using static TorchSharp.torch;

namespace Deep3DStudio.DeepMeshPrior
{
    public static class GraphUtils
    {
        public static (Tensor edgeIndex, Tensor edgeValues) ComputeAdjacencyMatrix(MeshData mesh, Device device)
        {
            // Build edges from faces
            // We want undirected graph, so we add (i, j) and (j, i)
            var edges = new HashSet<(int, int)>();

            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int v0 = mesh.Indices[i];
                int v1 = mesh.Indices[i + 1];
                int v2 = mesh.Indices[i + 2];

                AddEdge(edges, v0, v1);
                AddEdge(edges, v1, v2);
                AddEdge(edges, v2, v0);
            }

            // Also add self-loops
            int numVerts = mesh.Vertices.Count;
            for(int i=0; i<numVerts; i++)
            {
                 edges.Add((i, i));
            }

            // Convert to tensor format (2 x E)
            int numEdges = edges.Count;
            long[] src = new long[numEdges];
            long[] dst = new long[numEdges];

            int idx = 0;
            foreach(var edge in edges)
            {
                src[idx] = edge.Item1;
                dst[idx] = edge.Item2;
                idx++;
            }

            var edgeIndex = torch.stack(new [] {
                torch.tensor(src, dtype: ScalarType.Int64, device: device),
                torch.tensor(dst, dtype: ScalarType.Int64, device: device)
            }, dim: 0);

            // Calculate Symmetric Normalized Laplacian: D^-0.5 * A * D^-0.5
            // But GCN usually uses A_hat = A + I
            // We already added self loops to edges, so A_hat is what we have.

            // Degrees
            // Count occurrences of each source node
            var deg = torch.zeros(new long[]{numVerts}, dtype: ScalarType.Float32, device: device);
            // We can't use index_add easily on old TorchSharp or limited ops, let's do it manually or via scatter
            // On CPU it's fast enough to do in C#
            float[] degrees = new float[numVerts];
            foreach(var edge in edges)
            {
                degrees[edge.Item1] += 1.0f;
            }

            // Normalize values: val = 1 / sqrt(deg[src] * deg[dst])
            float[] values = new float[numEdges];
            idx = 0;
            foreach(var edge in edges)
            {
                float d_src = degrees[edge.Item1];
                float d_dst = degrees[edge.Item2];
                values[idx] = 1.0f / (float)Math.Sqrt(d_src * d_dst);
                idx++;
            }

            var edgeValues = torch.tensor(values, dtype: ScalarType.Float32, device: device);

            return (edgeIndex, edgeValues);
        }

        private static void AddEdge(HashSet<(int, int)> edges, int u, int v)
        {
            edges.Add((u, v));
            edges.Add((v, u));
        }

        public static (int[][] ve, int[][] edges) BuildMeshConnectivity(MeshData mesh)
        {
            // For Laplacian loss: need mapping from vertex to edges
            // edges: [E, 2] array of vertex indices
            // ve: [V] list of edge indices connected to vertex v

            var uniqueEdges = new HashSet<(int, int)>();
            for (int i = 0; i < mesh.Indices.Count; i += 3)
            {
                int[] face = new[] { mesh.Indices[i], mesh.Indices[i + 1], mesh.Indices[i + 2] };
                for (int j = 0; j < 3; j++)
                {
                    int u = face[j];
                    int v = face[(j + 1) % 3];
                    if (u > v) (u, v) = (v, u); // Canonicalize
                    uniqueEdges.Add((u, v));
                }
            }

            var edgeList = uniqueEdges.Select(e => new int[] { e.Item1, e.Item2 }).ToArray();
            var ve = new List<int>[mesh.Vertices.Count];
            for (int i = 0; i < ve.Length; i++) ve[i] = new List<int>();

            for (int i = 0; i < edgeList.Length; i++)
            {
                ve[edgeList[i][0]].Add(i);
                ve[edgeList[i][1]].Add(i);
            }

            return (ve.Select(l => l.ToArray()).ToArray(), edgeList);
        }
    }
}
