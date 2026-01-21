using System;
using System.Threading.Tasks;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.DeepMeshPrior;

namespace Deep3DStudio.Meshing
{
    public class DeepMeshPriorMesher
    {
        public async Task<MeshData?> RefineMeshAsync(
            MeshData inputMesh,
            Action<string, float>? progressCallback = null,
            System.Threading.CancellationToken cancellationToken = default)
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

                var result = await optimizer.OptimizeAsync(inputMesh, iterations, lr, lapWeight, progressCallback, cancellationToken);

                progressCallback?.Invoke("Optimization complete.", 1.0f);
                return result;
            }
            catch (OperationCanceledException)
            {
                progressCallback?.Invoke("DeepMeshPrior cancelled.", 0.0f);
                throw;
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
