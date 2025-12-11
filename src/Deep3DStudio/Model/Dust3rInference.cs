using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class CameraPose
    {
        public Matrix4 WorldToCamera { get; set; } // View Matrix
        public Matrix4 CameraToWorld { get; set; } // Pose Matrix
        public int ImageIndex { get; set; }
        public string ImagePath { get; set; } = string.Empty;
        public int Width { get; set; }
        public int Height { get; set; }
        // Intrinsics are implicit in Dust3r's output (rays per pixel), but we can store estimated ones if needed.
        // For NeRF from Dust3r output, we use the RayDataset constructed from Dust3r points.
    }

    public class SceneResult
    {
        public List<MeshData> Meshes { get; set; } = new List<MeshData>();
        public List<CameraPose> Poses { get; set; } = new List<CameraPose>();
    }

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

        /// <summary>
        /// Reconstructs a scene from a sequence of images using pairwise inference and global alignment.
        /// Returns both the geometry and the estimated camera poses.
        /// </summary>
        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            var result = new SceneResult();
            if (_session == null || imagePaths.Count < 2) return result;

            // Global Transform Accumulator: Frame i -> World Frame
            Matrix4 globalTransform = Matrix4.Identity;

            // Stores the target mesh of the previous pair in its local frame (Frame i-1) for alignment.
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

                    var c1 = ImageUtils.ExtractColors(img1Path, s1[1], s1[0]);
                    var c2 = ImageUtils.ExtractColors(img2Path, s2[1], s2[0]);

                    var mesh1_local = GeometryUtils.GenerateMeshFromDepth(pts1Tensor, conf1Tensor, c1, s1[1], s1[0]);
                    var mesh2_local = GeometryUtils.GenerateMeshFromDepth(pts2Tensor, conf2Tensor, c2, s2[1], s2[0]);

                    if (i > 0)
                    {
                        // Align Current Frame (i) to Previous Frame (i-1) using pixel-exact correspondence of Pts_i.
                        // mesh1_local represents Pts_i in the current pair's coordinate system.
                        // pts_i_in_prev represents Pts_i in the previous pair's coordinate system.

                        var T_curr_to_prev = GeometryUtils.ComputeTransformFromCorrespondences(mesh1_local, pts_i_in_prev!);
                        globalTransform = T_curr_to_prev * globalTransform;
                    }
                    else
                    {
                        // First pair sets the World Frame (Frame 0).
                        result.Meshes.Add(mesh1_local);
                        result.Poses.Add(new CameraPose
                        {
                            ImageIndex = 0,
                            ImagePath = img1Path,
                            Width = s1[1],
                            Height = s1[0],
                            CameraToWorld = Matrix4.Identity,
                            WorldToCamera = Matrix4.Identity
                        });
                    }

                    // Preserve local mesh2 for next iteration's alignment step
                    pts_i_in_prev = CloneMeshData(mesh2_local);

                    // Add Mesh 2 to the scene, transformed to World Coordinates
                    mesh2_local.ApplyTransform(globalTransform);
                    result.Meshes.Add(mesh2_local);

                    result.Poses.Add(new CameraPose
                    {
                        ImageIndex = i + 1,
                        ImagePath = img2Path,
                        Width = s2[1],
                        Height = s2[0],
                        CameraToWorld = globalTransform,
                        WorldToCamera = globalTransform.Inverted()
                    });
                }
            }
            return result;
        }

        private MeshData CloneMeshData(MeshData source)
        {
            return new MeshData
            {
                Vertices = new List<Vector3>(source.Vertices),
                Colors = new List<Vector3>(source.Colors),
                Indices = new List<int>(source.Indices),
                PixelToVertexIndex = source.PixelToVertexIndex // Reference copy is okay here as we don't modify the array content, just read it
            };
        }
    }
}
