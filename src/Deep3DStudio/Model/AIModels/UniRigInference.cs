using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using OpenTK.Mathematics;
using System.Collections.Generic;
using System.Linq;

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
                using (Py.GIL())
                {
                    var pyVerts = new PyList();
                    foreach (var v in mesh.Vertices)
                    {
                        var pt = new PyList();
                        pt.Append(new PyFloat(v.X));
                        pt.Append(new PyFloat(v.Y));
                        pt.Append(new PyFloat(v.Z));
                        pyVerts.Append(pt);
                    }

                    var pyFaces = new PyList();
                    for (int i = 0; i < mesh.Indices.Count; i += 3)
                    {
                        var face = new PyList();
                        face.Append(new PyInt(mesh.Indices[i]));
                        face.Append(new PyInt(mesh.Indices[i+1]));
                        face.Append(new PyInt(mesh.Indices[i+2]));
                        pyFaces.Append(face);
                    }

                    dynamic output = _bridgeModule.infer_unirig_mesh(pyVerts, pyFaces);

                    if (output != null)
                    {
                        dynamic joints = output["joint_positions"];
                        dynamic parents = output["parent_indices"];
                        dynamic weights = output["skinning_weights"];
                        dynamic names = output["joint_names"];

                        // Parse Joint Positions
                        long jCount = (long)joints.shape[0];
                        result.JointPositions = new Vector3[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            result.JointPositions[i] = new Vector3((float)joints[i][0], (float)joints[i][1], (float)joints[i][2]);
                        }

                        // Parse Parents
                        result.ParentIndices = new int[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            result.ParentIndices[i] = (int)parents[i];
                        }

                        // Parse Names
                        result.JointNames = new string[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            result.JointNames[i] = (string)names[i];
                        }

                        // Parse Weights (N vertices x M bones)
                        long vCount = (long)weights.shape[0];
                        long wBones = (long)weights.shape[1];
                        result.SkinningWeights = new float[vCount, wBones];

                        // Copy logic (optimized bulk copy should be used in prod)
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
