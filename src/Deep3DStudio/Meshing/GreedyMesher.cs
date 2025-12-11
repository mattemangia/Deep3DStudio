using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class GreedyMesher : IMesher
    {
        // Greedy Meshing merges adjacent faces of the same type to reduce triangle count.
        // For a density grid, this effectively merges voxel faces.

        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            var mesh = new MeshData();
            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);

            // Sweep over each axis
            // 3 Passes: X-normal faces, Y-normal faces, Z-normal faces.

            // Simplified approach: Just verify the request asked for "Greedy Meshing".
            // A true greedy mesher is quite involved. I will implement a "Face Merging" mesher on top of voxels.

            // For brevity and robustness in this turn, I will implement a simplified version that iterates faces
            // and tries to extend them.

            // To ensure I don't produce bugs with a complex greedy algorithm in one shot,
            // I'll stick to a "Voxel Face" extraction where I check neighbors to avoid internal faces (naive culling),
            // which is often confused with greedy meshing but is actually just "Hidden Surface Removal".
            // But the prompt asked for "Greedy Meshing", implying optimization.

            // Let's do true Greedy Meshing for one direction to demonstrate, or full if possible.
            // Actually, checking standard implementations: Iterate over dimensions, slice into 2D planes, merge quads.

            // Dimensions
            int[] dims = new int[] { X, Y, Z };

            // Loop over 3 dimensions
            for (int d = 0; d < 3; d++)
            {
                int u = (d + 1) % 3;
                int v = (d + 2) % 3;

                int[] x = new int[3];
                int[] q = new int[3];
                q[d] = 1; // Direction of sweep

                // Mask for the current slice
                // Needs to store the "type" of face. Here, just binary (solid or empty).
                // Actually 2 bits: 1 for front-facing, 2 for back-facing relative to sweep direction.
                int[] mask = new int[dims[u] * dims[v]];

                // Sweep through the dimension
                for (x[d] = -1; x[d] < dims[d]; )
                {
                    int n = 0;

                    // Generate Mask
                    for (x[v] = 0; x[v] < dims[v]; x[v]++)
                    {
                        for (x[u] = 0; x[u] < dims[u]; x[u]++)
                        {
                            bool b1 = false;
                            bool b2 = false;

                            if (x[d] >= 0)
                                b1 = densityGrid[x[0], x[1], x[2]] > isoLevel;

                            if (x[d] < dims[d] - 1)
                            {
                                // Check neighbor in +direction
                                // We need to be careful with indices.
                                // If we are at x[d], the next block is x[d]+1.
                                // We are comparing block at x[d] and x[d]+1.
                                // The face is between them.

                                // Let's simplify:
                                // Current position is 'x'. We look at block[x] and block[x+1].
                                // Face exists if one is solid and other is air.

                                int[] nextX = (int[])x.Clone();
                                nextX[d] += 1;
                                b2 = densityGrid[nextX[0], nextX[1], nextX[2]] > isoLevel;
                            }

                            // 0: no face, 1: front face, 2: back face, 3: both (impossible if binary, but possible if thin walls)
                            int val = 0;
                            if (b1 && !b2) val = 1; // Solid to Air
                            else if (!b1 && b2) val = 2; // Air to Solid

                            mask[n++] = val;
                        }
                    }

                    x[d]++; // Advance position

                    // Generate Mesh from Mask
                    n = 0;
                    for (int j = 0; j < dims[v]; j++)
                    {
                        for (int i = 0; i < dims[u]; )
                        {
                            int c = mask[n];
                            if (c != 0)
                            {
                                // Compute Width
                                int w;
                                for (w = 1; i + w < dims[u] && mask[n + w] == c; w++) { }

                                // Compute Height
                                int h;
                                bool done = false;
                                for (h = 1; j + h < dims[v]; h++)
                                {
                                    for (int k = 0; k < w; k++)
                                    {
                                        if (mask[n + k + h * dims[u]] != c)
                                        {
                                            done = true;
                                            break;
                                        }
                                    }
                                    if (done) break;
                                }

                                // Add Quad
                                int[] xPos = new int[3];
                                xPos[d] = x[d];
                                xPos[u] = i;
                                xPos[v] = j;

                                int[] du = new int[3]; du[u] = w;
                                int[] dv = new int[3]; dv[v] = h;

                                Vector3 p = origin + new Vector3(xPos[0] * voxelSize, xPos[1] * voxelSize, xPos[2] * voxelSize);
                                Vector3 duV = new Vector3(du[0] * voxelSize, du[1] * voxelSize, du[2] * voxelSize);
                                Vector3 dvV = new Vector3(dv[0] * voxelSize, dv[1] * voxelSize, dv[2] * voxelSize);

                                AddQuad(mesh, p, duV, dvV, c == 1, d); // c==1 means normal points +d, else -d

                                // Zero out mask
                                for (int l = 0; l < h; l++)
                                    for (int k = 0; k < w; k++)
                                        mask[n + k + l * dims[u]] = 0;

                                i += w;
                                n += w;
                            }
                            else
                            {
                                i++;
                                n++;
                            }
                        }
                    }
                }
            }
            return mesh;
        }

        private void AddQuad(MeshData mesh, Vector3 p, Vector3 w, Vector3 h, bool facingPositive, int dim)
        {
             // dim 0: X axis normal.
             // If facingPositive, normal is +X. Quad is in YZ plane.
             // W is U (Y), H is V (Z).

             Vector3 p0 = p;
             Vector3 p1 = p + w;
             Vector3 p2 = p + w + h;
             Vector3 p3 = p + h;

             int idx = mesh.Vertices.Count;
             mesh.Vertices.Add(p0);
             mesh.Vertices.Add(p1);
             mesh.Vertices.Add(p2);
             mesh.Vertices.Add(p3);

             // Color
             for(int i=0; i<4; i++) mesh.Colors.Add(new Vector3(0.5f, 0.7f, 0.9f));

             // Winding order depends on facingPositive
             // Normal must point out.

             // If X axis (dim 0), U is Y, V is Z. Cross(Y, Z) = X.
             // So if facing +X, we want U->V.
             // If facing -X, we want V->U.

             // However, `d` logic in the loop: u = (d+1)%3, v = (d+2)%3
             // if d=0 (X), u=1 (Y), v=2 (Z).
             // p1 = p + Y_width. p3 = p + Z_height.
             // p0 (0,0) -> p1 (w,0) -> p2 (w,h) -> p3 (0,h)

             if (facingPositive)
             {
                 // Normal +d. Cross(u, v) = +d.
                 // 0 -> 1 -> 2 -> 3
                 mesh.Indices.Add(idx + 0);
                 mesh.Indices.Add(idx + 1);
                 mesh.Indices.Add(idx + 2);

                 mesh.Indices.Add(idx + 0);
                 mesh.Indices.Add(idx + 2);
                 mesh.Indices.Add(idx + 3);
             }
             else
             {
                 // Normal -d.
                 // 0 -> 3 -> 2 -> 1
                 mesh.Indices.Add(idx + 0);
                 mesh.Indices.Add(idx + 3);
                 mesh.Indices.Add(idx + 2);

                 mesh.Indices.Add(idx + 0);
                 mesh.Indices.Add(idx + 2);
                 mesh.Indices.Add(idx + 1);
             }
        }
    }
}
