using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;
using System.Threading.Tasks;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.Meshing
{
    public class MarchingCubesMesher : IMesher
    {
        public MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel)
        {
            // Use Computation Device Setting
            // Although Marching Cubes is generally CPU bound unless using Compute Shaders,
            // we ensure we are using Parallel loops which respect CPU resources efficiently.

            var settings = IniSettings.Instance;
            bool useParallel = true; // Always parallelize on CPU

            int X = densityGrid.GetLength(0);
            int Y = densityGrid.GetLength(1);
            int Z = densityGrid.GetLength(2);

            // Default white color grid
            Vector3[,,] colorGrid = new Vector3[X, Y, Z];

            // Simple parallel fill
            Parallel.For(0, X, x => {
                for(int y=0; y<Y; y++)
                    for(int z=0; z<Z; z++)
                        colorGrid[x,y,z] = new Vector3(1, 1, 1);
            });

            return GeometryUtils.MarchingCubes(densityGrid, colorGrid, origin, new Vector3(voxelSize), isoLevel);
        }
    }
}
