using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Deep3DStudio.Scene;
using Deep3DStudio.Model;
using TorchSharp;
using static TorchSharp.torch;

namespace Deep3DStudio.DeepMeshPrior
{
    public class DeepMeshPriorOptimizer
    {
        public async Task<MeshData> OptimizeAsync(MeshData inputMesh, int iterations, float lr, float lapWeight, Action<string, float>? progressCallback)
        {
            // Initialize Torch
            // torch.InitializeDeviceType(DeviceType.CUDA); // Optional, auto-detected

            // Resolve device based on settings
            var settings = Deep3DStudio.Configuration.IniSettings.Instance;
            Device device = torch.CPU;

            if (settings.AIDevice == Deep3DStudio.Configuration.AIComputeDevice.CUDA && torch.cuda.is_available())
            {
                device = torch.CUDA;
            }
            else if (settings.AIDevice == Deep3DStudio.Configuration.AIComputeDevice.MPS)
            {
                // Note: TorchSharp 0.102 might not expose backends.mps_is_available() directly.
                // We rely on the user selection and try to usage.
                // In newer versions: torch.backends.mps_is_available() or torch.mps_is_available()
                // For safety in this version, we assume if MPS is selected on Mac, it should work or fail gracefully.
                // But since 'torch.MPS' device exists in recent TorchSharp, we use it.
                // If this fails to compile due to missing 'torch.MPS', we will fallback to string "mps".
                try {
                     device = torch.MPS;
                } catch {
                     // Fallback if torch.MPS is not defined in this binding version
                     device = new Device("mps");
                }
            }
            // TorchSharp does not currently support DirectML backend natively in the same way PyTorch does.
            // DirectML support in C# would require a specific backend loaded.
            // For now, we fallback to CPU if DirectML is selected in C# native components to avoid crash.
            else if (settings.AIDevice == Deep3DStudio.Configuration.AIComputeDevice.DirectML)
            {
                // Warn user
                Console.WriteLine("Warning: DirectML selected but not supported in native DeepMeshPrior. Falling back to CPU.");
                device = torch.CPU;
            }

            Console.WriteLine($"DeepMeshPrior using device: {device}");

            // 1. Prepare Data
            int numVerts = inputMesh.Vertices.Count;
            int numFeatures = 16;

            // Random input features Z ~ N(0, 0.1)
            var x_np = torch.randn(new long[] { numVerts, numFeatures }, dtype: ScalarType.Float32, device: device) * 0.1f;
            var x = x_np.detach(); // Fixed input noise

            // Target positions Y (the noisy mesh vertices)
            float[] vertsFlat = new float[numVerts * 3];
            for(int i=0; i<numVerts; i++)
            {
                vertsFlat[i*3] = inputMesh.Vertices[i].X;
                vertsFlat[i*3+1] = inputMesh.Vertices[i].Y;
                vertsFlat[i*3+2] = inputMesh.Vertices[i].Z;
            }
            var y = torch.tensor(vertsFlat, new long[] { numVerts, 3 }, dtype: ScalarType.Float32, device: device);
            var x_pos = y.clone(); // Initial positions

            // Graph Structure
            var (edgeIndex, edgeWeight) = GraphUtils.ComputeAdjacencyMatrix(inputMesh, device);

            // Laplacian Matrix for Loss
            // L = I - D^-1 A
            // We have normalized A_hat = D^-0.5 A D^-0.5
            // Standard Laplacian loss usually uses Uniform Laplacian or Cotangent.
            // The uniform Laplacian uses v_i - mean(neighbors),
            // which corresponds to L_uniform = I - D^-1 A_binary.

            // Let's build L_uniform
            var L_uniform = BuildUniformLaplacian(inputMesh, device);

            // 2. Model
            var model = new DeepMeshPriorNetwork(useSkipConnections: false);
            model.to(device);
            model.train();

            var optimizer = torch.optim.Adam(model.parameters(), lr: lr);

            // 3. Loop
            for(int i=0; i<iterations; i++)
            {
                optimizer.zero_grad();

                // Forward
                // Model outputs a delta; we add it to the current positions.
                var output = model.forward(x, edgeIndex, edgeWeight);
                var pred_pos = x_pos + output;

                // Loss
                var loss_mse = Loss.MSE(pred_pos, y);
                var loss_lap = Loss.LaplacianLossExplicit(pred_pos, L_uniform);

                var total_loss = loss_mse + torch.tensor(lapWeight, device: device) * loss_lap;

                // Backward
                total_loss.backward();
                optimizer.step();

                if (i % 50 == 0)
                {
                    float lossVal = total_loss.item<float>();
                    progressCallback?.Invoke($"Iter {i}/{iterations} Loss: {lossVal:F6}", (float)i/iterations);
                }
            }

            // 4. Final Result
            model.eval();
            var final_out = model.forward(x, edgeIndex, edgeWeight);
            var final_pos = (x_pos + final_out).cpu();

            // Update mesh vertices
            var resultMesh = inputMesh.Clone();
            resultMesh.Vertices.Clear();

            float[] resultData = final_pos.data<float>().ToArray();
            for(int i=0; i<numVerts; i++)
            {
                resultMesh.Vertices.Add(new OpenTK.Mathematics.Vector3(
                    resultData[i*3],
                    resultData[i*3+1],
                    resultData[i*3+2]
                ));
            }

            // Recalculate normals
            resultMesh.RecalculateNormals();

            return resultMesh;
        }

        private Tensor BuildUniformLaplacian(MeshData mesh, Device device)
        {
            // L = I - D^-1 A
            // Rows sum to 0.

            int n = mesh.Vertices.Count;

            // 1. Build Adjacency (Binary)
            var edges = new List<(int, int)>();
            for(int i=0; i<mesh.Indices.Count; i+=3)
            {
                int v0 = mesh.Indices[i];
                int v1 = mesh.Indices[i+1];
                int v2 = mesh.Indices[i+2];
                edges.Add((v0, v1)); edges.Add((v1, v0));
                edges.Add((v1, v2)); edges.Add((v2, v1));
                edges.Add((v2, v0)); edges.Add((v0, v2));
            }
            var uniqueEdges = edges.Distinct().ToList();

            // Degrees
            int[] degrees = new int[n];
            foreach(var e in uniqueEdges) degrees[e.Item1]++;

            // Values:
            // Diag = 1
            // Off-diag = -1 / degree[row]

            var indicesList = new List<long>();
            var valuesList = new List<float>();

            // Diagonals
            for(int i=0; i<n; i++)
            {
                indicesList.Add(i); indicesList.Add(i);
                valuesList.Add(1.0f);
            }

            // Off-diagonals
            foreach(var e in uniqueEdges)
            {
                indicesList.Add(e.Item1); indicesList.Add(e.Item2);
                valuesList.Add(-1.0f / Math.Max(1, degrees[e.Item1]));
            }

            long[] indicesArr = indicesList.ToArray();
            float[] valuesArr = valuesList.ToArray();

            // Convert to 2xNNZ tensor
            var indicesTensor = torch.tensor(indicesArr, dtype: ScalarType.Int64, device: device).reshape(-1, 2).t();
            var valuesTensor = torch.tensor(valuesArr, dtype: ScalarType.Float32, device: device);

            return torch.sparse_coo_tensor(indicesTensor, valuesTensor, new long[] { n, n }, dtype: ScalarType.Float32, device: device);
        }
    }
}
