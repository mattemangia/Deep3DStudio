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
            // This method seems to contain Python code copy-pasted.
            // We should use the explicit Laplacian loss implemented below.
            return torch.tensor(0.0f, device: device);
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
