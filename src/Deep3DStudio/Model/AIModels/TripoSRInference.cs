using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Deep3DStudio.Configuration;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// TripoSR inference for fast single-image to 3D reconstruction.
    /// Based on Large Reconstruction Model (LRM) architecture.
    /// </summary>
    public class TripoSRInference : IImageTo3DModel
    {
        private InferenceSession? _backboneSession;
        private bool _disposed;

        private const int DEFAULT_INPUT_SIZE = 256;

        public bool IsLoaded => _backboneSession != null;
        public string ModelName => "TripoSR";
        public string Description => "Fast feedforward 3D reconstruction from single image (<0.5s on A100)";

        public TripoSRInference()
        {
            // Try to auto-load from default path
            var settings = IniSettings.Instance;
            var modelPath = GetAbsoluteModelPath(settings.TripoSRModelPath);
            if (Directory.Exists(modelPath))
            {
                LoadModel(modelPath);
            }
        }

        public bool LoadModel(string modelPath)
        {
            try
            {
                UnloadModel();

                var backbonePath = Path.Combine(modelPath, "triposr_backbone.onnx");
                if (!File.Exists(backbonePath))
                {
                    Console.WriteLine($"TripoSR model not found at {backbonePath}");
                    return false;
                }

                var options = CreateSessionOptions();
                _backboneSession = new InferenceSession(backbonePath, options);

                Console.WriteLine($"TripoSR model loaded from {modelPath}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load TripoSR model: {ex.Message}");
                return false;
            }
        }

        public void UnloadModel()
        {
            _backboneSession?.Dispose();
            _backboneSession = null;
        }

        public AIModelResult GenerateFromImage(string imagePath)
        {
            if (!IsLoaded)
            {
                return new AIModelResult
                {
                    Success = false,
                    StatusMessage = "TripoSR model not loaded"
                };
            }

            try
            {
                var stopwatch = Stopwatch.StartNew();

                // Load and preprocess image
                var imageData = LoadAndPreprocessImage(imagePath);

                // Run inference
                var result = RunInference(imageData);

                stopwatch.Stop();
                result.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
                result.StatusMessage = $"Generated in {stopwatch.ElapsedMilliseconds}ms";

                return result;
            }
            catch (Exception ex)
            {
                return new AIModelResult
                {
                    Success = false,
                    StatusMessage = $"Error: {ex.Message}"
                };
            }
        }

        public AIModelResult GenerateFromImageData(byte[] imageData, int width, int height)
        {
            if (!IsLoaded)
            {
                return new AIModelResult
                {
                    Success = false,
                    StatusMessage = "TripoSR model not loaded"
                };
            }

            try
            {
                var stopwatch = Stopwatch.StartNew();

                // Preprocess the raw image data
                var tensor = PreprocessImageData(imageData, width, height);

                // Run inference
                var result = RunInferenceWithTensor(tensor);

                stopwatch.Stop();
                result.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
                result.StatusMessage = $"Generated in {stopwatch.ElapsedMilliseconds}ms";

                return result;
            }
            catch (Exception ex)
            {
                return new AIModelResult
                {
                    Success = false,
                    StatusMessage = $"Error: {ex.Message}"
                };
            }
        }

        private DenseTensor<float> LoadAndPreprocessImage(string imagePath)
        {
            var settings = IniSettings.Instance;
            int inputSize = settings.TripoSRResolution;

            // Load image using SkiaSharp or System.Drawing
            // For now, use ImageUtils from existing code
            return ImageUtils.LoadAndPreprocessImage(imagePath, inputSize, inputSize);
        }

        private DenseTensor<float> PreprocessImageData(byte[] imageData, int width, int height)
        {
            var settings = IniSettings.Instance;
            int inputSize = settings.TripoSRResolution;

            var tensor = new DenseTensor<float>(new[] { 1, 3, inputSize, inputSize });

            // Simple resize (nearest neighbor) and normalize
            float scaleX = (float)width / inputSize;
            float scaleY = (float)height / inputSize;

            for (int y = 0; y < inputSize; y++)
            {
                for (int x = 0; x < inputSize; x++)
                {
                    int srcX = Math.Min((int)(x * scaleX), width - 1);
                    int srcY = Math.Min((int)(y * scaleY), height - 1);
                    int srcIdx = (srcY * width + srcX) * 4; // RGBA

                    // Normalize to [0, 1]
                    tensor[0, 0, y, x] = imageData[srcIdx] / 255.0f;     // R
                    tensor[0, 1, y, x] = imageData[srcIdx + 1] / 255.0f; // G
                    tensor[0, 2, y, x] = imageData[srcIdx + 2] / 255.0f; // B
                }
            }

            return tensor;
        }

        private AIModelResult RunInference(DenseTensor<float> imageTensor)
        {
            return RunInferenceWithTensor(imageTensor);
        }

        private AIModelResult RunInferenceWithTensor(DenseTensor<float> imageTensor)
        {
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image", imageTensor)
            };

            using var results = _backboneSession!.Run(inputs);

            var triplaneTokens = results.First().AsTensor<float>().ToArray();

            // The backbone outputs triplane tokens
            // To get a mesh, we need to either:
            // 1. Query the decoder at grid points (if decoder is available)
            // 2. Return the tokens for later processing
            // For now, return the triplane tokens

            var settings = IniSettings.Instance;

            return new AIModelResult
            {
                Success = true,
                TriplaneTokens = triplaneTokens,
                StatusMessage = $"Generated {triplaneTokens.Length} triplane tokens"
            };
        }

        /// <summary>
        /// Generate mesh from triplane tokens using marching cubes.
        /// This requires either a decoder ONNX model or custom implementation.
        /// </summary>
        public AIModelResult GenerateMeshFromTriplane(float[] triplaneTokens)
        {
            var settings = IniSettings.Instance;
            int resolution = settings.TripoSRMarchingCubesRes;

            // This would require the decoder model to query SDF values
            // For now, return a placeholder

            return new AIModelResult
            {
                Success = false,
                StatusMessage = "Mesh extraction requires decoder model (not yet implemented)"
            };
        }

        private SessionOptions CreateSessionOptions()
        {
            var options = new SessionOptions();
            var settings = IniSettings.Instance;

            switch (settings.AIDevice)
            {
                case AIComputeDevice.CUDA:
                    try
                    {
                        options.AppendExecutionProvider_CUDA(0);
                        Console.WriteLine("Using CUDA execution provider for TripoSR");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"CUDA not available ({ex.Message}), trying DirectML...");
                        try
                        {
                            options.AppendExecutionProvider_DML(0);
                            Console.WriteLine("Using DirectML execution provider for TripoSR");
                        }
                        catch
                        {
                            Console.WriteLine("DirectML not available, falling back to CPU");
                        }
                    }
                    break;

                case AIComputeDevice.DirectML:
                    try
                    {
                        options.AppendExecutionProvider_DML(0);
                        Console.WriteLine("Using DirectML execution provider for TripoSR");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"DirectML not available ({ex.Message}), falling back to CPU");
                    }
                    break;

                default:
                    Console.WriteLine("Using CPU execution provider for TripoSR");
                    break;
            }

            return options;
        }

        private string GetAbsoluteModelPath(string path)
        {
            if (Path.IsPathRooted(path))
                return path;

            // Relative to application directory
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, path);
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                UnloadModel();
                _disposed = true;
            }
        }
    }
}
