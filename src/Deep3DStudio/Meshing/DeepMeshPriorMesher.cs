using System;
using System.Threading.Tasks;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.DeepMeshPrior;

namespace Deep3DStudio.Meshing
{
    public class DeepMeshPriorMesher
    {
        public async Task<MeshData?> RefineMeshAsync(MeshData inputMesh, Action<string, float>? progressCallback = null)
        {
            try
            {
                var settings = IniSettings.Instance;
                var optimizer = new DeepMeshPriorOptimizer();

                // DeepMeshPrior optimization requires iterations and learning rate from settings
                int iterations = settings.DeepMeshPriorIterations;
                float lr = settings.DeepMeshPriorLearningRate;
                float lapWeight = settings.DeepMeshPriorLaplacianWeight;

                progressCallback?.Invoke("Starting DeepMeshPrior optimization...", 0.0f);

                // Run the optimization
                // Note: TorchSharp operations usually need to run on a thread that can access native libs?
                // TorchSharp is thread-safe for the most part but let's be careful.

                var result = await optimizer.OptimizeAsync(inputMesh, iterations, lr, lapWeight, progressCallback);

                progressCallback?.Invoke("Optimization complete.", 1.0f);
                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"DeepMeshPrior error: {ex.Message}");
                progressCallback?.Invoke($"Error: {ex.Message}", 0);
                return null;
            }
        }
    }
}
