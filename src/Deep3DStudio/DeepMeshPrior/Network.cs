using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Deep3DStudio.DeepMeshPrior
{
    public class DeepMeshPriorNetwork : nn.Module<Tensor, Tensor, Tensor, Tensor>
    {
        // GCN layers
        private List<GCNConv> _convs = new List<GCNConv>();
        private List<BatchNorm1d> _bns = new List<BatchNorm1d>();
        private LeakyReLU _activation;

        // Skip connection setting
        private bool _useSkipConnections;
        private Linear _finalLinear;
        private Linear _finalLinear2;

        public DeepMeshPriorNetwork(bool useSkipConnections = false, string name = "DeepMeshPriorNet") : base(name)
        {
            _useSkipConnections = useSkipConnections;

            // Channels configuration matching the python reference
            // h = [16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 16, 3] (Normal)
            long[] h;
            if (useSkipConnections)
            {
                h = new long[] { 16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 3 };
                // Encoder
                AddLayer(h[0], h[1]); // 0 -> 1
                AddLayer(h[1], h[2]); // 1 -> 2
                AddLayer(h[2], h[3]); // 2 -> 3
                AddLayer(h[3], h[4]); // 3 -> 4
                AddLayer(h[4], h[5]); // 4 -> 5
                AddLayer(h[5], h[6]); // 5 -> 6
                AddLayer(h[6], h[7]); // 6 -> 7 (Bottleneck)

                // Decoder with skip concatenation.
                AddLayer(h[7] + h[6], h[8]); // 7+6 -> 8
                AddLayer(h[8] + h[5], h[9]); // 8+5 -> 9
                AddLayer(h[9] + h[4], h[10]); // 9+4 -> 10
                AddLayer(h[10] + h[3], h[11]); // 10+3 -> 11
                AddLayer(h[11] + h[2], h[12]); // 11+2 -> 12
                AddLayer(h[12] + h[1], h[13]); // 12+1 -> 13

                _finalLinear = nn.Linear(h[13], h[14]);
                register_module("final_linear", _finalLinear);
            }
            else
            {
                h = new long[] { 16, 32, 64, 128, 256, 256, 512, 512, 256, 256, 128, 64, 32, 32, 16, 3 };
                // Stop at h[13] (32 channels) for Convs
                for(int i=0; i<h.Length-3; i++)
                {
                    AddLayer(h[i], h[i+1]);
                }

                // Two linear layers at the end: linear1 (32 -> 16) then linear2 (16 -> 3).
                _finalLinear = nn.Linear(h[h.Length-3], h[h.Length-2]); // 32 -> 16
                register_module("final_linear1", _finalLinear);

                _finalLinear2 = nn.Linear(h[h.Length-2], h[h.Length-1]); // 16 -> 3
                register_module("final_linear2", _finalLinear2);
            }

            _activation = nn.LeakyReLU();
            register_module("activation", _activation);
        }

        private void AddLayer(long inCh, long outCh)
        {
            var conv = new GCNConv(inCh, outCh);
            var bn = nn.BatchNorm1d(outCh);

            _convs.Add(conv);
            _bns.Add(bn);

            register_module($"conv_{_convs.Count}", conv);
            register_module($"bn_{_bns.Count}", bn);
        }

        public override Tensor forward(Tensor x, Tensor edgeIndex, Tensor edgeWeight)
        {
            Tensor dx = x;

            if (_useSkipConnections)
            {
                var skips = new List<Tensor>();

                // Encoder (7 layers: 0 to 6)
                for(int i=0; i<7; i++)
                {
                    dx = _convs[i].forward(dx, edgeIndex, edgeWeight);
                    dx = _bns[i].forward(dx);
                    dx = _activation.forward(dx);
                    if (i < 6) skips.Add(dx); // Store outputs of conv1..conv6 for skip
                }

                // Decoder (6 layers: 7 to 12)
                // Skips are used in reverse: skip6, skip5, ..., skip1
                for(int i=7; i<13; i++)
                {
                    var skip = skips[6 - (i - 6)];

                    dx = torch.cat(new [] { dx, skip }, dim: 1);
                    dx = _convs[i].forward(dx, edgeIndex, edgeWeight);
                    dx = _bns[i].forward(dx);
                    dx = _activation.forward(dx);
                }

                dx = _finalLinear.forward(dx);
            }
            else
            {
                // Normal sequential
                for(int i=0; i<_convs.Count; i++)
                {
                    dx = _convs[i].forward(dx, edgeIndex, edgeWeight);
                    dx = _bns[i].forward(dx);
                    dx = _activation.forward(dx);
                }

                dx = _finalLinear.forward(dx);
                dx = _activation.forward(dx);
                dx = _finalLinear2.forward(dx);
            }

            return dx; // Delta coordinates
        }
    }
}
