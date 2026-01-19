using System;
using System.IO;
using Python.Runtime;
using Deep3DStudio.Python;
using Deep3DStudio.Configuration;
using OpenTK.Mathematics;
using System.Collections.Generic;

namespace Deep3DStudio.Model.AIModels
{
    public class TripoSRInference : BasePythonInference
    {
        public TripoSRInference() : base("triposr") { }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();
            if (!_isLoaded) return mesh;

            try
            {
                byte[] imgBytes = File.ReadAllBytes(imagePath);

                // Get settings for model parameters
                var settings = IniSettings.Instance;
                int resolution = settings.TripoSRResolution;
                int mcResolution = settings.TripoSRMarchingCubesRes;

                Deep3DStudio.Python.PythonService.Instance.ExecuteWithGIL((scope) =>
                {
                    // CRITICAL: Use explicit PyObject method calls instead of dynamic
                    // to prevent GC from collecting temporary arguments
                    using var inferMethod = _bridgeModule!.GetAttr("infer_triposr");
                    using var imgBytesPy = imgBytes.ToPython();
                    using var resolutionPy = resolution.ToPython();
                    using var mcResolutionPy = mcResolution.ToPython();
                    using var args = new PyTuple(new PyObject[] { imgBytesPy, resolutionPy, mcResolutionPy });

                    using var output = inferMethod.Invoke(args);
                    if (output != null && !output.IsNone())
                    {
                        using var vertices = output["vertices"];
                        using var faces = output["faces"];
                        using var colors = output["colors"];

                        using var vShapeObj = vertices.GetAttr("shape");
                        using var fShapeObj = faces.GetAttr("shape");
                        long vCount = vShapeObj[0].As<long>();
                        long fCount = fShapeObj[0].As<long>();

                        // Import builtins once for conversions
                        using var builtins = Py.Import("builtins");
                        using var floatFunc = builtins.GetAttr("float");
                        using var intFunc = builtins.GetAttr("int");

                        for(int i=0; i<vCount; i++)
                        {
                            using var vRow = vertices[i];
                            using var cRow = colors[i];

                            using var vx0 = vRow[0];
                            using var vy0 = vRow[1];
                            using var vz0 = vRow[2];
                            using var cx0 = cRow[0];
                            using var cy0 = cRow[1];
                            using var cz0 = cRow[2];

                            using var vxArgs = new PyTuple(new PyObject[] { vx0 });
                            using var vyArgs = new PyTuple(new PyObject[] { vy0 });
                            using var vzArgs = new PyTuple(new PyObject[] { vz0 });
                            using var cxArgs = new PyTuple(new PyObject[] { cx0 });
                            using var cyArgs = new PyTuple(new PyObject[] { cy0 });
                            using var czArgs = new PyTuple(new PyObject[] { cz0 });

                            using var vxPy = floatFunc.Invoke(vxArgs);
                            using var vyPy = floatFunc.Invoke(vyArgs);
                            using var vzPy = floatFunc.Invoke(vzArgs);
                            using var cxPy = floatFunc.Invoke(cxArgs);
                            using var cyPy = floatFunc.Invoke(cyArgs);
                            using var czPy = floatFunc.Invoke(czArgs);

                            float x = (float)vxPy.As<double>();
                            float y = (float)vyPy.As<double>();
                            float z = (float)vzPy.As<double>();
                            mesh.Vertices.Add(new Vector3(x, y, z));

                            float r = (float)cxPy.As<double>();
                            float g = (float)cyPy.As<double>();
                            float b = (float)czPy.As<double>();
                            mesh.Colors.Add(new Vector3(r, g, b));
                        }

                        for(int i=0; i<fCount; i++)
                        {
                            using var fRow = faces[i];
                            using var f0 = fRow[0];
                            using var f1 = fRow[1];
                            using var f2 = fRow[2];

                            using var f0Args = new PyTuple(new PyObject[] { f0 });
                            using var f1Args = new PyTuple(new PyObject[] { f1 });
                            using var f2Args = new PyTuple(new PyObject[] { f2 });

                            using var i0Py = intFunc.Invoke(f0Args);
                            using var i1Py = intFunc.Invoke(f1Args);
                            using var i2Py = intFunc.Invoke(f2Args);

                            mesh.Indices.Add((int)i0Py.As<long>());
                            mesh.Indices.Add((int)i1Py.As<long>());
                            mesh.Indices.Add((int)i2Py.As<long>());
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                Console.WriteLine($"TripoSR Inference failed: {ex.Message}");
            }
            return mesh;
        }
    }
}
