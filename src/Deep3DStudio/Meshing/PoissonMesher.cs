using System;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class PoissonMesher : IMesher
    {
        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel, Action<string, float>? progress = null)
        {
            int w = densityGrid.GetLength(0);
            int h = densityGrid.GetLength(1);
            int d = densityGrid.GetLength(2);

            // 1. Compute Gradients (Normals) from the input density grid
            // We use central differences to estimate normals.
            progress?.Invoke("Computing gradients...", 0.0f);
            Vector3[,,] vectorField = new Vector3[w, h, d];

            Parallel.For(1, w - 1, x =>
            {
                for (int y = 1; y < h - 1; y++)
                {
                    for (int z = 1; z < d - 1; z++)
                    {
                        float dx = (densityGrid[x + 1, y, z] - densityGrid[x - 1, y, z]) * 0.5f;
                        float dy = (densityGrid[x, y + 1, z] - densityGrid[x, y - 1, z]) * 0.5f;
                        float dz = (densityGrid[x, y, z + 1] - densityGrid[x, y, z - 1]) * 0.5f;

                        vectorField[x, y, z] = new Vector3(dx, dy, dz);
                    }
                }
            });
            progress?.Invoke("Gradients computed.", 0.15f);

            // 2. Smooth the Vector Field
            // Smoothing the vector field helps in reducing noise and providing a cleaner reconstruction,
            // which is the main advantage of Poisson-like approaches.
            int smoothingPasses = 2;
            for (int i = 0; i < smoothingPasses; i++)
            {
                progress?.Invoke($"Smoothing vector field (pass {i + 1}/{smoothingPasses})...", 0.15f + 0.10f * i / smoothingPasses);
                vectorField = SmoothVectorField(vectorField);
            }

            // 3. Compute Divergence of the Vector Field
            // We compute the divergence (div V) to set up the Poisson equation.
            // Equation: Laplacian(Phi) = Divergence(V)
            // However, to ensure stability in the Conjugate Gradient solver, we solve for the Negative Laplacian:
            // -Laplacian(Phi) = -Divergence(V)
            // The Negative Laplacian operator (6*center - sum_neighbors) is Symmetric Positive Definite (SPD),
            // which is a strict requirement for the Conjugate Gradient method.
            progress?.Invoke("Computing divergence...", 0.25f);
            float[,,] negDivergence = new float[w, h, d];
            Parallel.For(1, w - 1, x =>
            {
                for (int y = 1; y < h - 1; y++)
                {
                    for (int z = 1; z < d - 1; z++)
                    {
                        float dv_dx = (vectorField[x + 1, y, z].X - vectorField[x - 1, y, z].X) * 0.5f;
                        float dv_dy = (vectorField[x, y + 1, z].Y - vectorField[x, y - 1, z].Y) * 0.5f;
                        float dv_dz = (vectorField[x, y, z + 1].Z - vectorField[x, y, z - 1].Z) * 0.5f;

                        // Negate the divergence for the RHS
                        negDivergence[x, y, z] = -(dv_dx + dv_dy + dv_dz);
                    }
                }
            });

            // 4. Solve Poisson Equation: -Laplacian(Phi) = -Divergence
            // Using Matrix-Free Conjugate Gradient Solver.
            // Heuristic for iterations: scale with grid dimension to ensure convergence on larger grids.
            // Base iterations = 200, plus dimension-dependent factor.
            progress?.Invoke("Solving Poisson equation (CG)...", 0.35f);
            int maxDim = Math.Max(w, Math.Max(h, d));
            int iterations = Math.Max(200, maxDim * 4);

            float[,,] phi = SolvePoissonCG(negDivergence, maxIterations: iterations, progress: progress);

            // 5. Generate Mesh from the reconstructed potential field (Phi)
            // Use adaptive isoLevel based on phi statistics since the CG output
            // has arbitrary range unrelated to the caller's isoLevel.
            progress?.Invoke("Computing adaptive iso-level...", 0.85f);
            float adaptiveIso = ComputeAdaptiveIsoLevel(phi);

            // Default white color grid for Marching Cubes extraction
            // Colors are typically overwritten by the caller (e.g., pipeline ApplyColorsFromGrid)
            Vector3[,,] colorGrid = new Vector3[w, h, d];
            Parallel.For(0, w, x => {
                for(int y=0; y<h; y++)
                    for(int z=0; z<d; z++)
                        colorGrid[x,y,z] = new Vector3(1, 1, 1);
            });

            progress?.Invoke("Extracting surface...", 0.90f);
            var result = GeometryUtils.MarchingCubes(phi, colorGrid, origin, new Vector3(voxelSize), adaptiveIso);
            progress?.Invoke("Poisson meshing complete.", 1.0f);
            return result;
        }

        private float ComputeAdaptiveIsoLevel(float[,,] phi)
        {
            int w = phi.GetLength(0);
            int h = phi.GetLength(1);
            int d = phi.GetLength(2);

            // Use Otsu's method on the phi field to find the optimal threshold
            // that separates exterior (zero/low) from interior (high) regions.
            float minVal = float.MaxValue, maxVal = float.MinValue;
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 1; y++)
                    for (int z = 1; z < d - 1; z++)
                    {
                        float v = phi[x, y, z];
                        if (v < minVal) minVal = v;
                        if (v > maxVal) maxVal = v;
                    }

            if (maxVal - minVal < 1e-12f)
                return 0.0f;

            const int bins = 256;
            int[] histogram = new int[bins];
            int total = 0;
            float range = maxVal - minVal;

            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 1; y++)
                    for (int z = 1; z < d - 1; z++)
                    {
                        int bin = Math.Clamp((int)((phi[x, y, z] - minVal) / range * (bins - 1)), 0, bins - 1);
                        histogram[bin]++;
                        total++;
                    }

            if (total == 0) return 0.0f;

            double sumTotal = 0;
            for (int i = 0; i < bins; i++)
                sumTotal += (double)i * histogram[i];

            double sumB = 0;
            int wB = 0;
            double maxVariance = 0;
            int bestThreshold = 1;

            for (int t = 1; t < bins; t++)
            {
                wB += histogram[t - 1];
                if (wB == 0) continue;
                int wF = total - wB;
                if (wF == 0) break;

                sumB += (double)(t - 1) * histogram[t - 1];
                double mB = sumB / wB;
                double mF = (sumTotal - sumB) / wF;
                double variance = (double)wB * wF * (mB - mF) * (mB - mF);

                if (variance > maxVariance)
                {
                    maxVariance = variance;
                    bestThreshold = t;
                }
            }

            return minVal + ((float)bestThreshold / (bins - 1)) * range;
        }

        private Vector3[,,] SmoothVectorField(Vector3[,,] input)
        {
            int w = input.GetLength(0);
            int h = input.GetLength(1);
            int d = input.GetLength(2);
            Vector3[,,] output = (Vector3[,,])input.Clone(); // Clone to preserve boundary values

            Parallel.For(1, w - 1, x =>
            {
                for (int y = 1; y < h - 1; y++)
                {
                    for (int z = 1; z < d - 1; z++)
                    {
                        Vector3 sum = Vector3.Zero;
                        int count = 0;

                        for (int dx = -1; dx <= 1; dx++)
                        {
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                for (int dz = -1; dz <= 1; dz++)
                                {
                                    sum += input[x + dx, y + dy, z + dz];
                                    count++;
                                }
                            }
                        }
                        output[x, y, z] = sum / count;
                    }
                }
            });

            return output;
        }

        private float[,,] SolvePoissonCG(float[,,] rhs, int maxIterations, Action<string, float>? progress = null)
        {
            int w = rhs.GetLength(0);
            int h = rhs.GetLength(1);
            int d = rhs.GetLength(2);

            float[,,] x = new float[w, h, d]; // Initial guess 0 (Dirichlet boundary conditions implied)
            float[,,] r = (float[,,])rhs.Clone(); // Residual r = b - Ax. Since x=0, r = b (rhs)
            float[,,] p = (float[,,])r.Clone(); // Search direction
            float[,,] Ap = new float[w, h, d]; // Pre-allocate buffer for matrix-vector product

            double rsold = DotProduct(r, r);

            int reportInterval = Math.Max(1, maxIterations / 50);
            for (int i = 0; i < maxIterations; i++)
            {
                // Convergence check
                if (rsold < 1e-10) break;

                if (i % reportInterval == 0)
                    progress?.Invoke($"Solving Poisson (iter {i}/{maxIterations})...", 0.35f + 0.50f * i / maxIterations);

                // Apply Negative Laplacian operator (A * p)
                // This operator must be Positive Definite.
                ApplyNegativeLaplacian(p, Ap);

                double pAp = DotProduct(p, Ap);

                // Stability check: The operator should be positive definite, so pAp > 0.
                // If it's effectively zero or negative (due to float precision), we stop to avoid NaN.
                if (pAp <= 1e-20) break;

                double alpha = rsold / pAp;

                // Update solution x and residual r
                // x = x + alpha * p
                // r = r - alpha * Ap
                Parallel.For(1, w - 1, ix =>
                {
                    for (int iy = 1; iy < h - 1; iy++)
                    {
                        for (int iz = 1; iz < d - 1; iz++)
                        {
                            x[ix, iy, iz] += (float)(alpha * p[ix, iy, iz]);
                            r[ix, iy, iz] -= (float)(alpha * Ap[ix, iy, iz]);
                        }
                    }
                });

                double rsnew = DotProduct(r, r);

                // Convergence check
                if (rsnew < 1e-10) break;

                double beta = rsnew / rsold;

                // Update search direction p
                // p = r + beta * p
                Parallel.For(1, w - 1, ix =>
                {
                    for (int iy = 1; iy < h - 1; iy++)
                    {
                        for (int iz = 1; iz < d - 1; iz++)
                        {
                            p[ix, iy, iz] = r[ix, iy, iz] + (float)(beta * p[ix, iy, iz]);
                        }
                    }
                });

                rsold = rsnew;
            }

            return x;
        }

        private void ApplyNegativeLaplacian(float[,,] u, float[,,] result)
        {
            int w = u.GetLength(0);
            int h = u.GetLength(1);
            int d = u.GetLength(2);
            
            // Negative Laplacian Operator (-L)
            // Standard Discrete Laplacian L(u) = Sum(neighbors) - 6*u(center)
            // Negative Laplacian -L(u) = 6*u(center) - Sum(neighbors)
            // This matrix is Symmetric Positive Definite (SPD) for Dirichlet boundary conditions (u=0 at boundary),
            // ensuring convergence of the CG method.

            Parallel.For(1, w - 1, x =>
            {
                for (int y = 1; y < h - 1; y++)
                {
                    for (int z = 1; z < d - 1; z++)
                    {
                        float sumNeighbors = u[x + 1, y, z] + u[x - 1, y, z] +
                                             u[x, y + 1, z] + u[x, y - 1, z] +
                                             u[x, y, z + 1] + u[x, y, z - 1];

                        result[x, y, z] = 6.0f * u[x, y, z] - sumNeighbors;
                    }
                }
            });
        }

        private double DotProduct(float[,,] a, float[,,] b)
        {
            double sum = 0;
            object lockObj = new object();

            int w = a.GetLength(0);
            int h = a.GetLength(1);
            int d = a.GetLength(2);

            // Compute dot product of two fields
            // Using local accumulation to minimize lock contention
            Parallel.For(0, w, () => 0.0, (x, state, localSum) =>
            {
                for (int y = 0; y < h; y++)
                {
                    for (int z = 0; z < d; z++)
                    {
                        localSum += a[x, y, z] * b[x, y, z];
                    }
                }
                return localSum;
            },
            (localSum) => { lock(lockObj) sum += localSum; });

            return sum;
        }
    }
}
