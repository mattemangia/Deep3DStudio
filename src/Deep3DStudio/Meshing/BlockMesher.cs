using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class BlockMesher : IMesher
    {
        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            var mesh = new MeshData();
            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);
            float hs = voxelSize / 2.0f; // half size

            // Offsets for 8 corners of a cube relative to center
            Vector3[] corners = new Vector3[]
            {
                new Vector3(-hs, -hs, -hs), new Vector3( hs, -hs, -hs),
                new Vector3( hs,  hs, -hs), new Vector3(-hs,  hs, -hs),
                new Vector3(-hs, -hs,  hs), new Vector3( hs, -hs,  hs),
                new Vector3( hs,  hs,  hs), new Vector3(-hs,  hs,  hs)
            };

            // Indices for 12 triangles (6 faces)
            // Front: 4,5,6, 4,6,7. Back: 1,0,3, 1,3,2 ... etc.
            // Let's just generate quads and triangulate them.
            // Faces: -Z, +Z, -Y, +Y, -X, +X

            for (int x = 0; x < X; x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    for (int z = 0; z < Z; z++)
                    {
                        if (densityGrid[x, y, z] > isoLevel)
                        {
                            Vector3 center = origin + new Vector3(x * voxelSize, y * voxelSize, z * voxelSize);
                            AddCube(mesh, center, hs);
                        }
                    }
                }
            }

            return mesh;
        }

        private void AddCube(MeshData mesh, Vector3 center, float hs)
        {
            int baseIdx = mesh.Vertices.Count;

            // Vertices
            mesh.Vertices.Add(center + new Vector3(-hs, -hs,  hs)); // 0: Front Bottom Left
            mesh.Vertices.Add(center + new Vector3( hs, -hs,  hs)); // 1: Front Bottom Right
            mesh.Vertices.Add(center + new Vector3( hs,  hs,  hs)); // 2: Front Top Right
            mesh.Vertices.Add(center + new Vector3(-hs,  hs,  hs)); // 3: Front Top Left
            mesh.Vertices.Add(center + new Vector3(-hs, -hs, -hs)); // 4: Back Bottom Left
            mesh.Vertices.Add(center + new Vector3( hs, -hs, -hs)); // 5: Back Bottom Right
            mesh.Vertices.Add(center + new Vector3( hs,  hs, -hs)); // 6: Back Top Right
            mesh.Vertices.Add(center + new Vector3(-hs,  hs, -hs)); // 7: Back Top Left

            // Colors (White)
            for(int i=0; i<8; i++) mesh.Colors.Add(new Vector3(0.8f, 0.8f, 0.8f));

            // Indices
            // Front
            AddQuad(mesh, baseIdx, 0, 1, 2, 3);
            // Back
            AddQuad(mesh, baseIdx, 5, 4, 7, 6);
            // Top
            AddQuad(mesh, baseIdx, 3, 2, 6, 7);
            // Bottom
            AddQuad(mesh, baseIdx, 4, 5, 1, 0);
            // Left
            AddQuad(mesh, baseIdx, 4, 0, 3, 7);
            // Right
            AddQuad(mesh, baseIdx, 1, 5, 6, 2);
        }

        private void AddQuad(MeshData mesh, int baseIdx, int i0, int i1, int i2, int i3)
        {
            mesh.Indices.Add(baseIdx + i0);
            mesh.Indices.Add(baseIdx + i1);
            mesh.Indices.Add(baseIdx + i2);

            mesh.Indices.Add(baseIdx + i0);
            mesh.Indices.Add(baseIdx + i2);
            mesh.Indices.Add(baseIdx + i3);
        }
    }
}
