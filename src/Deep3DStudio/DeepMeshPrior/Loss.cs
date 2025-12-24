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

        public static Tensor MeshLaplacianLoss(Tensor pred_pos, Tensor edgeIndex)
        {
            // Edge-list Laplacian Loss
            // Loss = sum_{i,j in E} || v_i - v_j ||^2
            // This is equivalent to Uniform Laplacian L = D - A, x^T L x = sum (v_i - v_j)^2
            // This pulls connected vertices together.

            // edgeIndex is [2, E]
            var srcIdx = edgeIndex[0]; // [E]
            var dstIdx = edgeIndex[1]; // [E]

            var src = pred_pos.index_select(0, srcIdx); // [E, 3]
            var dst = pred_pos.index_select(0, dstIdx); // [E, 3]

            var diff = src - dst;
            var sqDist = torch.sum(torch.square(diff), dim: 1); // [E]

            return torch.mean(sqDist); // Mean over edges
        }

        public static Tensor LaplacianLossExplicit(Tensor pred_pos, Tensor laplacianMat)
        {
            // laplacianMat: Sparse tensor [N, N] representing I - D^-1 A
            // Loss = mean( || L * X ||^2 )

            // L * X result is [N, 3] (displacement vector for each vertex)
            var delta = torch.matmul(laplacianMat, pred_pos);

            // Squared norm of each displacement
            var sqNorm = torch.sum(torch.square(delta), dim: 1); // [N]

            // Mean
            return torch.mean(sqNorm);
        }
    }
}
