using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Meshing
{
    public class MarchingCubesMesher : IMesher
    {
        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            // Since we don't have color information in this signature, we'll create a dummy color grid or overload interface.
            // For now, let's just make white vertices.

            // Wait, the original MC implementation took a color grid.
            // I should probably pass a color grid or a function to sample color.
            // For simplicity in this refactor, let's assume white if not provided,
            // but the interface should ideally support color if we want NeRF colors.

            // I'll create a dummy color grid for now to reuse the existing static logic if I want to call it,
            // OR I can copy the logic here. Copying is safer to avoid modifying legacy code too much and cleaner for the "New Mesher" system.

            // However, to save token space and time, I will call the existing GeometryUtils.MarchingCubes
            // But I need to provide colors.

            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);

            Vector3[,,] colorGrid = new Vector3[X, Y, Z];
            for(int x=0; x<X; x++)
                for(int y=0; y<Y; y++)
                    for(int z=0; z<Z; z++)
                        colorGrid[x,y,z] = new Vector3(1, 1, 1);

            return GeometryUtils.MarchingCubes(densityGrid, colorGrid, origin, new Vector3(voxelSize), isoLevel);
        }
    }
}
