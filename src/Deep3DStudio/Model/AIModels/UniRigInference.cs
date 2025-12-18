using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using OpenTK.Mathematics;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Deep3DStudio.Model.AIModels
{
    public class UniRigInference : BasePythonInference
    {
        public UniRigInference() : base("unirig") { }

        public MeshData GenerateFromImage(string imagePath)
        {
            return new MeshData();
        }

        public RigResult RigMesh(MeshData mesh)
        {
            Initialize();
            var result = new RigResult();
            if (!_isLoaded) return result;

            try
            {
                // Serialize Vertices
                float[] vertsArr = new float[mesh.Vertices.Count * 3];
                for(int i=0; i<mesh.Vertices.Count; i++)
                {
                    vertsArr[i*3] = mesh.Vertices[i].X;
                    vertsArr[i*3+1] = mesh.Vertices[i].Y;
                    vertsArr[i*3+2] = mesh.Vertices[i].Z;
                }
                byte[] vertsBytes = new byte[vertsArr.Length * sizeof(float)];
                Buffer.BlockCopy(vertsArr, 0, vertsBytes, 0, vertsBytes.Length);

                // Serialize Faces
                int[] facesArr = mesh.Indices.ToArray();
                byte[] facesBytes = new byte[facesArr.Length * sizeof(int)];
                Buffer.BlockCopy(facesArr, 0, facesBytes, 0, facesBytes.Length);

                using (Py.GIL())
                {
                    // Pass bytes directly to Python
                    // ToPython() on byte[] creates a bytes object which is what we want
                    dynamic output = _bridgeModule.infer_unirig_mesh_bytes(vertsBytes.ToPython(), facesBytes.ToPython());

                    if (output != null)
                    {
                        dynamic joints = output["joint_positions"];
                        dynamic parents = output["parent_indices"];
                        dynamic weights = output["skinning_weights"];
                        dynamic names = output["joint_names"];

                        long jCount = (long)joints.shape[0];
                        result.JointPositions = new Vector3[jCount];
                        // Parsing float arrays from numpy is reasonably fast via indexing or we can implement reverse buffer copy
                        // For rig data (small), looping is fine.
                        for(int i=0; i<jCount; i++)
                        {
                            result.JointPositions[i] = new Vector3((float)joints[i][0], (float)joints[i][1], (float)joints[i][2]);
                        }

                        result.ParentIndices = new int[jCount];
                        for(int i=0; i<jCount; i++) result.ParentIndices[i] = (int)parents[i];

                        result.JointNames = new string[jCount];
                        for(int i=0; i<jCount; i++) result.JointNames[i] = (string)names[i];

                        long vCount = (long)weights.shape[0];
                        long wBones = (long)weights.shape[1];
                        result.SkinningWeights = new float[vCount, wBones];

                        // Weights can be large, but usually sparse.
                        // Optimization: Copy buffer if possible.
                        // But Py.NET dynamic access is okay for now given vertices < 100k usually.
                        for(int v=0; v<vCount; v++)
                        {
                            for(int b=0; b<wBones; b++)
                            {
                                result.SkinningWeights[v,b] = (float)weights[v][b];
                            }
                        }

                        result.Success = true;
                        result.StatusMessage = "Rigging successful";
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"UniRig Inference failed: {ex.Message}");
                result.Success = false;
                result.StatusMessage = ex.Message;
            }
            return result;
        }
    }
}
