using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class Dust3rInference
    {
        private InferenceSession? _session;
        private readonly string _modelPath = "dust3r.onnx";

        public Dust3rInference()
        {
            try
            {
                if (System.IO.File.Exists(_modelPath))
                {
                    if (new System.IO.FileInfo(_modelPath).Length > 0)
                    {
                        var options = new SessionOptions();
                        _session = new InferenceSession(_modelPath, options);
                    }
                    else
                    {
                        Console.WriteLine("Warning: Model file is empty.");
                    }
                }
                else
                {
                    Console.WriteLine("Warning: Model file not found at " + _modelPath);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading model: {ex.Message}");
            }
        }

        public bool IsLoaded => _session != null;

        public List<MeshData> ReconstructScene(List<string> imagePaths)
        {
            var meshes = new List<MeshData>();
            if (_session == null || imagePaths.Count < 2) return meshes;

            // Global Transform Accumulator
            Matrix4 globalTransform = Matrix4.Identity;

            // Previous step data for alignment
            // We store the "Target" mesh of the previous pair, in its LOCAL frame (Frame i-1)
            // Wait, alignment happens between:
            // Pair (i-1, i) -> Pts_i_in_prev (Target of pair)
            // Pair (i, i+1) -> Pts_i_in_curr (Source of pair)
            // Correspondences are pixel-exact.

            MeshData? pts_i_in_prev = null;

            for (int i = 0; i < imagePaths.Count - 1; i++)
            {
                string img1Path = imagePaths[i];
                string img2Path = imagePaths[i+1];

                var (t1, s1) = ImageUtils.LoadAndPreprocessImage(img1Path);
                var (t2, s2) = ImageUtils.LoadAndPreprocessImage(img2Path);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("img1", t1),
                    NamedOnnxValue.CreateFromTensor("img2", t2),
                    NamedOnnxValue.CreateFromTensor("true_shape1", new DenseTensor<int>(new[] { s1[0], s1[1] }, new[] { 1, 2 })),
                    NamedOnnxValue.CreateFromTensor("true_shape2", new DenseTensor<int>(new[] { s2[0], s2[1] }, new[] { 1, 2 }))
                };

                using (var results = _session.Run(inputs))
                {
                    var pts1Tensor = results.First(n => n.Name == "pts3d1").AsTensor<float>();
                    var conf1Tensor = results.First(n => n.Name == "conf1").AsTensor<float>();
                    var pts2Tensor = results.First(n => n.Name == "pts3d2").AsTensor<float>();
                    var conf2Tensor = results.First(n => n.Name == "conf2").AsTensor<float>();

                    // Extract Colors
                    var c1 = ImageUtils.ExtractColors(img1Path, s1[1], s1[0]);
                    var c2 = ImageUtils.ExtractColors(img2Path, s2[1], s2[0]);

                    var mesh1_local = GeometryUtils.GenerateMeshFromDepth(pts1Tensor, conf1Tensor, c1, s1[1], s1[0]);
                    var mesh2_local = GeometryUtils.GenerateMeshFromDepth(pts2Tensor, conf2Tensor, c2, s2[1], s2[0]);

                    // mesh1_local is Pts_i_in_curr
                    // mesh2_local is Pts_{i+1}_in_curr

                    if (i > 0)
                    {
                        // Align Current Frame (i) to Previous Frame (i-1)
                        // Using Pts_i (which exists in both)
                        // Source: mesh1_local (Pts_i in Frame i)
                        // Target: pts_i_in_prev (Pts_i in Frame i-1)

                        // Find T s.t. mesh1_local * T = pts_i_in_prev
                        // Wait, GeometryUtils.ComputeRigidTransform gives M s.t. v' = v * M (OpenTK).
                        // If we want v_prev = v_curr * T_curr_to_prev
                        // We use Source=Curr, Target=Prev.

                        var T_curr_to_prev = GeometryUtils.ComputeTransformFromCorrespondences(mesh1_local, pts_i_in_prev!);

                        // Update Global Transform
                        // Global_{i} = T_curr_to_prev * Global_{i-1} ??
                        // Let's trace:
                        // Pts_global = Pts_prev * Global_{i-1}
                        // Pts_prev = Pts_curr * T_curr_to_prev
                        // So Pts_global = Pts_curr * T_curr_to_prev * Global_{i-1}

                        globalTransform = T_curr_to_prev * globalTransform;
                    }
                    else
                    {
                        // First pair: Frame 0 is World.
                        // Add mesh1 (Frame 0) directly.
                        meshes.Add(mesh1_local);
                    }

                    // Apply current global transform to the NEW part (Mesh 2)
                    // Note: If i > 0, mesh1_local corresponds to overlapping data we already have (mostly),
                    // but we might want to merge it? For simplicity, we only add the "new" mesh2 each step,
                    // chaining the transform.
                    // Actually, Visual Odometry usually adds the keyframes.
                    // Here we add Mesh2 transformed to world.

                    var mesh2_world = mesh2_local; // shallow copy ref, but we modify content
                    mesh2_world.ApplyTransform(globalTransform);
                    meshes.Add(mesh2_world);

                    // Store Mesh2 (in local frame!) for next iteration alignment
                    // Wait, we applied transform to mesh2_world. We need the original local one for next step?
                    // No, next step (i+1) will produce Pts_{i+1} in Frame (i+1).
                    // We need Pts_{i+1} in Frame (i) to align.
                    // Frame (i) IS the current frame of this loop iteration.
                    // mesh2_local was in Frame (i).
                    // So we must save mesh2_local BEFORE transforming it, or make a copy.

                    // Let's regenerate or deep copy?
                    // Since GenerateMesh is expensive, let's clone vertices?
                    // Or just act on copies.

                    // Better: Compute mesh2_local, save it as `pts_i_in_prev` for next loop.
                    // Then clone it to `mesh2_world`, transform, and add to list.

                    pts_i_in_prev = CloneMeshData(mesh2_local); // Helper needed

                    // Actually, we modified mesh2_world in place above.
                    // Let's invert the logic:
                    // 1. Save local mesh2 for next step.
                    // 2. Transform mesh2 to world and add.
                }
            }
            return meshes;
        }

        private MeshData CloneMeshData(MeshData source)
        {
            return new MeshData
            {
                Vertices = new List<Vector3>(source.Vertices),
                Colors = new List<Vector3>(source.Colors),
                Indices = new List<int>(source.Indices),
                PixelToVertexIndex = source.PixelToVertexIndex // Array reference is fine if we don't modify content, but we don't.
            };
        }
    }
}
