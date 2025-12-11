using System.Collections.Generic;
using Deep3DStudio.Model; // Correct Namespace for MeshData
using OpenTK.Mathematics;

namespace Deep3DStudio.Meshing
{
    public interface IMesher
    {
        MeshData GenerateMesh(float[,,] densityGrid, Vector3 origin, float voxelSize, float isoLevel);
    }
}
