using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using Deep3DStudio.Configuration;
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
                // Get settings for model parameters
                var settings = IniSettings.Instance;
                int maxJoints = settings.UniRigMaxJoints;

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

                PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    // CRITICAL: Use explicit PyObject method calls instead of dynamic
                    // to prevent GC from collecting temporary arguments
                    using var inferMethod = _bridgeModule!.GetAttr("infer_unirig_mesh_bytes");
                    using var vertsBytesPy = vertsBytes.ToPython();
                    using var facesBytesPy = facesBytes.ToPython();
                    using var maxJointsPy = maxJoints.ToPython();
                    using var args = new PyTuple(new PyObject[] { vertsBytesPy, facesBytesPy, maxJointsPy });

                    using var output = inferMethod.Invoke(args);
                    if (output != null && !output.IsNone())
                    {
                        using var joints = output["joint_positions"];
                        using var parents = output["parent_indices"];
                        using var weights = output["skinning_weights"];
                        using var names = output["joint_names"];

                        using var jShapeObj = joints.GetAttr("shape");
                        long jCount = jShapeObj[0].As<long>();

                        // Import builtins once for conversions
                        using var builtins = Py.Import("builtins");
                        using var floatFunc = builtins.GetAttr("float");
                        using var intFunc = builtins.GetAttr("int");
                        using var strFunc = builtins.GetAttr("str");

                        result.JointPositions = new Vector3[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            using var jRow = joints[i];
                            using var jx0 = jRow[0];
                            using var jy0 = jRow[1];
                            using var jz0 = jRow[2];

                            using var jxArgs = new PyTuple(new PyObject[] { jx0 });
                            using var jyArgs = new PyTuple(new PyObject[] { jy0 });
                            using var jzArgs = new PyTuple(new PyObject[] { jz0 });

                            using var jxPy = floatFunc.Invoke(jxArgs);
                            using var jyPy = floatFunc.Invoke(jyArgs);
                            using var jzPy = floatFunc.Invoke(jzArgs);

                            result.JointPositions[i] = new Vector3(
                                (float)jxPy.As<double>(),
                                (float)jyPy.As<double>(),
                                (float)jzPy.As<double>()
                            );
                        }

                        result.ParentIndices = new int[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            using var pIdx = parents[i];
                            using var pIdxArgs = new PyTuple(new PyObject[] { pIdx });
                            using var pIdxPy = intFunc.Invoke(pIdxArgs);
                            result.ParentIndices[i] = (int)pIdxPy.As<long>();
                        }

                        result.JointNames = new string[jCount];
                        for(int i=0; i<jCount; i++)
                        {
                            using var nameObj = names[i];
                            using var nameArgs = new PyTuple(new PyObject[] { nameObj });
                            using var namePy = strFunc.Invoke(nameArgs);
                            result.JointNames[i] = namePy.As<string>() ?? $"joint_{i}";
                        }

                        using var wShapeObj = weights.GetAttr("shape");
                        long vCount = wShapeObj[0].As<long>();
                        long wBones = wShapeObj[1].As<long>();
                        result.SkinningWeights = new float[vCount, wBones];

                        for(int v=0; v<vCount; v++)
                        {
                            using var wRow = weights[v];
                            for(int b=0; b<wBones; b++)
                            {
                                using var wVal = wRow[b];
                                using var wValArgs = new PyTuple(new PyObject[] { wVal });
                                using var wValPy = floatFunc.Invoke(wValArgs);
                                result.SkinningWeights[v,b] = (float)wValPy.As<double>();
                            }
                        }

                        result.Success = true;
                        result.StatusMessage = "Rigging successful";
                    }
                });
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
