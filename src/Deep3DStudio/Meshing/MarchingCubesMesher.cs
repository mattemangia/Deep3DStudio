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
            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);

            // Default white color grid - colors are typically post-processed by callers
            // (e.g., NeRFModel.GetMesh interpolates colors from voxel color field)
            Vector3[,,] colorGrid = new Vector3[X, Y, Z];
            for(int x=0; x<X; x++)
                for(int y=0; y<Y; y++)
                    for(int z=0; z<Z; z++)
                        colorGrid[x,y,z] = new Vector3(1, 1, 1);

            return GeometryUtils.MarchingCubes(densityGrid, colorGrid, origin, new Vector3(voxelSize), isoLevel);
        }
    }
}
