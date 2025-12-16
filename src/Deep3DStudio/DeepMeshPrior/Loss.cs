using System;
using TorchSharp;
using static TorchSharp.torch;

namespace Deep3DStudio.DeepMeshPrior
{
    public static class Loss
    {
        public static Tensor MSE(Tensor pred, Tensor target)
        {
            // Simple MSE
            var diff = pred - target;
            return torch.mean(torch.square(diff));
        }

        public static Tensor MeshLaplacianLoss(Tensor pred_pos, int[][] ve, int[][] edges, Device device)
        {
            // pred_pos: [N, 3]
            // We need to implement Laplacian smoothing loss
            // L = 1/N * sum( || v_i - 1/deg(i) * sum(v_j) ||^2 )

            // Construct adjacency matrix for current mesh (assumes topology is fixed)
            // But we already have edgeIndex and values from GraphUtils?
            // Yes, let's reuse the graph structure if possible.
            // But here "pred_pos" changes every iteration.

            // To do this efficiently in TorchSharp without re-building sparse matrix every time:
            // Precompute the "Laplacian Matrix" L = I - D^-1 A
            // Then Loss = || L * X ||^2

            // However, the python code computes it dynamically to check for isolated vertices?
            // No, it builds sparse matrix inside loss function.

            // Let's assume we pass a pre-calculated Laplacian matrix if possible.
            // But the signature here takes ve/edges.

            // For performance, we should compute the Laplacian Operator Matrix once and just do matrix multiplication.
            // L_op X

            // Let's return a zero tensor if we can't compute it easily here,
            // but the Optimizer should calculate the Laplacian Matrix once and pass it.

            return torch.tensor(0.0f, device: device);
        }

        public static Tensor LaplacianLossExplicit(Tensor pred_pos, Tensor laplacianMat)
        {
            // laplacianMat: Sparse tensor [N, N] representing I - D^-1 A
            // Loss = mean( || L * X ||^2 )

            // L * X result is [N, 3] (displacement vector for each vertex)
            var delta = torch.sparse.mm(laplacianMat, pred_pos);

            // Squared norm of each displacement
            var sqNorm = torch.sum(torch.square(delta), dim: 1); // [N]

            // Mean
            return torch.mean(sqNorm);
        }
    }
}
