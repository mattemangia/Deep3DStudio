using System;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class PoissonMesher : IMesher
    {
        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            int w = densityGrid.GetLength(0);
            int h = densityGrid.GetLength(1);
            int d = densityGrid.GetLength(2);

            // 1. Compute Gradients (Normals) from the input density grid
            // We use central differences.
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

            // 2. Smooth the Vector Field
            // We apply a few passes of box blur to the vector field.
            int smoothingPasses = 2;
            for (int i = 0; i < smoothingPasses; i++)
            {
                vectorField = SmoothVectorField(vectorField);
            }

            // 3. Compute Divergence of the Vector Field
            // div V = dVx/dx + dVy/dy + dVz/dz
            // We negate the divergence because we are solving Negative Laplacian (-L) * phi = -div V
            // -L is SPD, which is required for CG.
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

                        negDivergence[x, y, z] = -(dv_dx + dv_dy + dv_dz);
                    }
                }
            });

            // 4. Solve Poisson Equation: -Laplacian(Phi) = -Divergence
            // using Conjugate Gradient (Matrix Free) with Negative Laplacian to ensure SPD matrix.
            float[,,] phi = SolvePoissonCG(negDivergence, maxIterations: 200);

            // 5. Generate Mesh

            // Dummy color grid
            Vector3[,,] colorGrid = new Vector3[w, h, d];
            Parallel.For(0, w, x => {
                for(int y=0; y<h; y++)
                    for(int z=0; z<d; z++)
                        colorGrid[x,y,z] = new Vector3(1, 1, 1);
            });

            return GeometryUtils.MarchingCubes(phi, colorGrid, origin, new Vector3(voxelSize), isoLevel);
        }

        private Vector3[,,] SmoothVectorField(Vector3[,,] input)
        {
            int w = input.GetLength(0);
            int h = input.GetLength(1);
            int d = input.GetLength(2);
            Vector3[,,] output = (Vector3[,,])input.Clone(); // Copy input to preserve boundaries

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

        private float[,,] SolvePoissonCG(float[,,] rhs, int maxIterations)
        {
            int w = rhs.GetLength(0);
            int h = rhs.GetLength(1);
            int d = rhs.GetLength(2);

            float[,,] x = new float[w, h, d]; // Initial guess 0
            float[,,] r = (float[,,])rhs.Clone(); // Residual r = b - Ax. Since x=0, r = b (rhs)
            float[,,] p = (float[,,])r.Clone(); // Search direction

            double rsold = DotProduct(r, r);

            for (int i = 0; i < maxIterations; i++)
            {
                if (rsold < 1e-10) break;

                // Apply Negative Laplacian here
                float[,,] Ap = ApplyNegativeLaplacian(p);

                double pAp = DotProduct(p, Ap);
                // Safety check to prevent division by zero or negative if numerical instability occurs
                if (pAp <= 1e-20) break;

                double alpha = rsold / pAp;

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
                if (rsnew < 1e-10) break;

                double beta = rsnew / rsold;

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

        private float[,,] ApplyNegativeLaplacian(float[,,] u)
        {
            int w = u.GetLength(0);
            int h = u.GetLength(1);
            int d = u.GetLength(2);
            float[,,] result = new float[w, h, d];

            // Negative Laplacian (-L)
            // L(u) = neighbors - 6*center
            // -L(u) = 6*center - neighbors
            // This operator is Positive Definite on the grid (with Dirichlet BCs implied by fixed 0 boundaries).

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

            return result;
        }

        private double DotProduct(float[,,] a, float[,,] b)
        {
            double sum = 0;
            object lockObj = new object();

            int w = a.GetLength(0);
            int h = a.GetLength(1);
            int d = a.GetLength(2);

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
