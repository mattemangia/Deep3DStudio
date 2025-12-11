using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class SurfaceNetsMesher : IMesher
    {
        // Naive Surface Nets:
        // 1. For each cube, if it contains an edge intersection, create a vertex in the center (or average of intersections).
        // 2. For each edge crossing the surface, create a quad connecting the 4 adjacent cube vertices.

        // This produces a dual mesh compared to Marching Cubes and is generally smoother/more relaxed than MC but retains blocky topology if not smoothed.

        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            var mesh = new MeshData();
            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);

            // Map voxel indices (x,y,z) to a vertex index in the mesh
            // Use a hash map or a 3D array if size permits. 3D array is faster.
            int[,,] nodeIndices = new int[X, Y, Z];
            for(int i=0; i<X; i++) for(int j=0; j<Y; j++) for(int k=0; k<Z; k++) nodeIndices[i,j,k] = -1;

            // 1. Generate Vertices for cells that intersect the surface
            for (int x = 0; x < X - 1; x++)
            {
                for (int y = 0; y < Y - 1; y++)
                {
                    for (int z = 0; z < Z - 1; z++)
                    {
                        // Check 8 corners
                        int mask = 0;
                        if (densityGrid[x, y, z] < isoLevel) mask |= 1;
                        if (densityGrid[x+1, y, z] < isoLevel) mask |= 2;
                        if (densityGrid[x, y+1, z] < isoLevel) mask |= 4;
                        if (densityGrid[x, y, z+1] < isoLevel) mask |= 8;
                        if (densityGrid[x+1, y+1, z] < isoLevel) mask |= 16;
                        if (densityGrid[x+1, y, z+1] < isoLevel) mask |= 32;
                        if (densityGrid[x, y+1, z+1] < isoLevel) mask |= 64;
                        if (densityGrid[x+1, y+1, z+1] < isoLevel) mask |= 128;

                        if (mask != 0 && mask != 255) // Surface intersects this cell
                        {
                            // Calculate average position of edge intersections
                            // Simplified: Just place at center of voxel for now (Naive).
                            // Better: Average of edge crossings.

                            Vector3 avgPos = Vector3.Zero;
                            int count = 0;

                            // Edges: x-axis
                            ProcessEdge(x, y, z, 1, 0, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x, y+1, z, 1, 0, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x, y, z+1, 1, 0, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x, y+1, z+1, 1, 0, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);

                            // Y-axis
                            ProcessEdge(x, y, z, 0, 1, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x+1, y, z, 0, 1, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x, y, z+1, 0, 1, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x+1, y, z+1, 0, 1, 0, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);

                            // Z-axis
                            ProcessEdge(x, y, z, 0, 0, 1, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x+1, y, z, 0, 0, 1, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x, y+1, z, 0, 0, 1, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);
                            ProcessEdge(x+1, y+1, z, 0, 0, 1, densityGrid, isoLevel, voxelSize, origin, ref avgPos, ref count);

                            if (count > 0)
                            {
                                avgPos /= count;
                                nodeIndices[x, y, z] = mesh.Vertices.Count;
                                mesh.Vertices.Add(avgPos);
                                mesh.Colors.Add(new Vector3(1, 0.5f, 0.5f)); // Pinkish
                            }
                        }
                    }
                }
            }

            // 2. Generate Quads for edges crossing the surface
            // We iterate edges. If an edge crosses the surface (sign change), we connect the 4 surrounding cell centers.

            // X-parallel edges
            for (int x = 0; x < X - 1; x++)
            {
                for (int y = 1; y < Y - 1; y++)
                {
                    for (int z = 1; z < Z - 1; z++)
                    {
                        float v1 = densityGrid[x, y, z];
                        float v2 = densityGrid[x+1, y, z];

                        if ((v1 < isoLevel) != (v2 < isoLevel))
                        {
                            // Edge crosses surface.
                            // Surrounding cells: (x, y-1, z-1), (x, y, z-1), (x, y, z), (x, y-1, z)
                            // Note: The "cells" are defined by the min corner.
                            // The edge is shared by 4 cells.
                            // Edge is at (x,y,z) -> (x+1,y,z).
                            // Cells sharing this edge are defined by offsets relative to the edge.

                            // Cells:
                            // A: x, y-1, z-1
                            // B: x, y, z-1
                            // C: x, y, z
                            // D: x, y-1, z

                            int a = nodeIndices[x, y-1, z-1];
                            int b = nodeIndices[x, y, z-1];
                            int c = nodeIndices[x, y, z];
                            int d = nodeIndices[x, y-1, z];

                            if (a != -1 && b != -1 && c != -1 && d != -1)
                            {
                                if (v1 < v2) // Inside to Outside
                                    AddQuad(mesh, a, b, c, d);
                                else
                                    AddQuad(mesh, d, c, b, a);
                            }
                        }
                    }
                }
            }

            // Y-parallel edges
            for (int x = 1; x < X - 1; x++)
            {
                for (int y = 0; y < Y - 1; y++)
                {
                    for (int z = 1; z < Z - 1; z++)
                    {
                        float v1 = densityGrid[x, y, z];
                        float v2 = densityGrid[x, y+1, z];

                        if ((v1 < isoLevel) != (v2 < isoLevel))
                        {
                            // Cells sharing edge (x,y,z)->(x,y+1,z)
                            // A: x-1, y, z-1
                            // B: x, y, z-1
                            // C: x, y, z
                            // D: x-1, y, z

                            int a = nodeIndices[x-1, y, z-1];
                            int b = nodeIndices[x, y, z-1];
                            int c = nodeIndices[x, y, z];
                            int d = nodeIndices[x-1, y, z];

                            if (a != -1 && b != -1 && c != -1 && d != -1)
                            {
                                if (v1 < v2)
                                    AddQuad(mesh, d, c, b, a);
                                else
                                    AddQuad(mesh, a, b, c, d);
                            }
                        }
                    }
                }
            }

            // Z-parallel edges
            for (int x = 1; x < X - 1; x++)
            {
                for (int y = 1; y < Y - 1; y++)
                {
                    for (int z = 0; z < Z - 1; z++)
                    {
                        float v1 = densityGrid[x, y, z];
                        float v2 = densityGrid[x, y, z+1];

                        if ((v1 < isoLevel) != (v2 < isoLevel))
                        {
                            // Cells sharing edge (x,y,z)->(x,y,z+1)
                            // A: x-1, y-1, z
                            // B: x, y-1, z
                            // C: x, y, z
                            // D: x-1, y, z

                            int a = nodeIndices[x-1, y-1, z];
                            int b = nodeIndices[x, y-1, z];
                            int c = nodeIndices[x, y, z];
                            int d = nodeIndices[x-1, y, z];

                            if (a != -1 && b != -1 && c != -1 && d != -1)
                            {
                                if (v1 < v2)
                                    AddQuad(mesh, a, b, c, d);
                                else
                                    AddQuad(mesh, d, c, b, a);
                            }
                        }
                    }
                }
            }

            return mesh;
        }

        private void ProcessEdge(int x, int y, int z, int dx, int dy, int dz, float[,,] density, float isoLevel, float voxelSize, Vector3 origin, ref Vector3 avgPos, ref int count)
        {
            if (x+dx >= density.GetLength(0) || y+dy >= density.GetLength(1) || z+dz >= density.GetLength(2)) return;

            float v1 = density[x, y, z];
            float v2 = density[x+dx, y+dy, z+dz];

            if ((v1 < isoLevel) != (v2 < isoLevel))
            {
                float mu = (isoLevel - v1) / (v2 - v1);
                Vector3 p1 = origin + new Vector3(x*voxelSize, y*voxelSize, z*voxelSize);
                Vector3 p2 = origin + new Vector3((x+dx)*voxelSize, (y+dy)*voxelSize, (z+dz)*voxelSize);
                avgPos += p1 + mu * (p2 - p1);
                count++;
            }
        }

        private void AddQuad(MeshData mesh, int i0, int i1, int i2, int i3)
        {
            mesh.Indices.Add(i0);
            mesh.Indices.Add(i1);
            mesh.Indices.Add(i2);

            mesh.Indices.Add(i0);
            mesh.Indices.Add(i2);
            mesh.Indices.Add(i3);
        }
    }
}
