using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model
{
    /// <summary>
    /// MASt3R (Matching And Stereo 3D Reconstruction) inference handler.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// This avoids all pythonnet memory corruption issues.
    /// </summary>
    public class Mast3rInference : IDisposable
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;
        private string? _lastError;

        public Mast3rInference()
        {
            // Initialization is lazy
        }

        public bool IsLoaded => _inference?.IsLoaded ?? false;
        public string? LastError => _lastError;

        public Action<string>? LogCallback
        {
            set
            {
                _logCallback = value;
                if (_inference != null)
                {
                    _inference.OnLog += msg => _logCallback?.Invoke(msg);
                }
            }
        }

        public event Action<string, float, string>? OnProgress;

        private void Log(string message)
        {
            Console.WriteLine(message);
            _logCallback?.Invoke(message);
        }

        private string GetDeviceString()
        {
            var settings = IniSettings.Instance;
            return settings.AIDevice switch
            {
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.ROCm => "cuda", // ROCm uses cuda-compatible API
                AIComputeDevice.MPS => "mps",
                _ => "cpu"
            };
        }

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.Mast3rModelPath;

            // Handle HuggingFace-style path
            if (configuredPath.Contains("/") && !Path.IsPathRooted(configuredPath) && !configuredPath.StartsWith("models"))
            {
                return configuredPath; // Return as-is for from_pretrained
            }

            // Handle local path
            if (Path.IsPathRooted(configuredPath) && File.Exists(configuredPath))
            {
                return configuredPath;
            }

            // Check common locations
            string[] possiblePaths = new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, configuredPath),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "mast3r_weights.pth"),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    return path;
                }
            }

            // Return default HuggingFace path
            return "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric";
        }

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[Mast3r] Initializing subprocess inference...");

                _inference = new SubprocessInference("mast3r");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) => OnProgress?.Invoke(stage, progress, message);

                string weightsPath = GetWeightsPath();
                string device = GetDeviceString();

                Log($"[Mast3r] Loading model from: {weightsPath}");
                Log($"[Mast3r] Device: {device}");

                OnProgress?.Invoke("load", 0.1f, "Loading MASt3R model...");

                bool loaded = _inference.Load(weightsPath, device);

                if (loaded)
                {
                    Log("[Mast3r] Model loaded successfully");
                    OnProgress?.Invoke("load", 1.0f, "MASt3R loaded");
                }
                else
                {
                    _lastError = "Failed to load MASt3R model";
                    Log($"[Mast3r] {_lastError}");
                }
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Mast3r: {ex.Message}";
                Log(_lastError);
            }
        }

        /// <summary>
        /// Reconstruct 3D scene from multiple images using MASt3R.
        /// </summary>
        public SceneResult ReconstructScene(List<string> imagePaths, bool useRetrieval = true)
        {
            Initialize();
            var result = new SceneResult();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log($"[Mast3r] Not loaded. {_lastError}");
                return result;
            }

            try
            {
                if (imagePaths == null || imagePaths.Count == 0)
                {
                    Log("[Mast3r] No images provided");
                    return result;
                }

                // Load image bytes
                var imagesBytes = new List<byte[]>();
                var validImagePaths = new List<string>();

                foreach (var path in imagePaths)
                {
                    if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                    {
                        Log($"[Mast3r] Skipping invalid path: {path}");
                        continue;
                    }
                    imagesBytes.Add(File.ReadAllBytes(path));
                    validImagePaths.Add(path);
                }

                if (imagesBytes.Count == 0)
                {
                    Log("[Mast3r] No valid images to process");
                    return result;
                }

                Log($"[Mast3r] Processing {imagesBytes.Count} images...");
                OnProgress?.Invoke("inference", 0.1f, $"Processing {imagesBytes.Count} images...");

                // Run inference
                var meshes = _inference.Infer(imagesBytes, useRetrieval);

                if (meshes.Count == 0)
                {
                    Log("[Mast3r] No results from inference");
                    return result;
                }

                // Convert results
                for (int i = 0; i < meshes.Count; i++)
                {
                    result.Meshes.Add(meshes[i]);

                    if (i < validImagePaths.Count)
                    {
                        result.Poses.Add(new CameraPose
                        {
                            ImageIndex = i,
                            ImagePath = validImagePaths[i]
                        });
                    }
                }

                Log($"[Mast3r] Inference complete. {result.Meshes.Count} meshes generated.");
                OnProgress?.Invoke("inference", 1.0f, "Complete");

                return result;
            }
            catch (Exception ex)
            {
                Log($"[Mast3r] Error: {ex.Message}");
                return result;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            _inference?.Dispose();
            _inference = null;
            _disposed = true;
        }
    }
}
