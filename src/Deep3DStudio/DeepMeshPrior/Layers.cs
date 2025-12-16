using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Deep3DStudio.DeepMeshPrior
{
    public class GCNConv : nn.Module<Tensor, Tensor, Tensor, Tensor>
    {
        private Linear _linear;

        public GCNConv(long inChannels, long outChannels, string name = "GCNConv") : base(name)
        {
            // Linear transformation: X * W
            // Bias is handled by Linear by default
            _linear = nn.Linear(inChannels, outChannels);
            RegisterModule("linear", _linear);
        }

        /// <summary>
        /// Forward pass.
        /// </summary>
        /// <param name="x">Node features [N, in_channels]</param>
        /// <param name="edgeIndex">Edge indices [2, E]</param>
        /// <param name="edgeWeight">Normalized edge weights [E]</param>
        public override Tensor forward(Tensor x, Tensor edgeIndex, Tensor edgeWeight)
        {
            // 1. Linearly transform node feature matrix.
            // X' = X * W
            var xTransformed = _linear.forward(x);

            // 2. Message passing (Matrix Multiplication with Normalized Adjacency)
            // out = A_hat * X'
            // A_hat is sparse.

            long numNodes = x.size(0);

            // Create sparse adjacency matrix
            var adj = torch.sparse_coo_tensor(edgeIndex, edgeWeight, new long[] { numNodes, numNodes }, dtype: ScalarType.Float32, device: x.device);

            // Sparse MM
            var outTensor = torch.sparse.mm(adj, xTransformed);

            return outTensor;
        }
    }
}
