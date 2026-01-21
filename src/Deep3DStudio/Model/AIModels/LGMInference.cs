using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// LGM (Large Gaussian Model) - Fast 3D Gaussian generation from images.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class LGMInference : IDisposable, IInferenceWithProgress
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;

        public LGMInference() { }

        public bool IsLoaded => _inference?.IsLoaded ?? false;

        public Action<string>? LogCallback
        {
            set
            {
                _logCallback = value;
                if (_inference != null)
                    _inference.OnLog += msg => _logCallback?.Invoke(msg);
            }
        }

        public event Action<string, float, string>? OnProgress;
        public event Action<string, float, string>? OnLoadProgress;

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
                AIComputeDevice.Auto => "auto",
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.ROCm => "rocm",
                AIComputeDevice.DirectML => "directml",
                AIComputeDevice.MPS => "mps",
                _ => "cpu"
            };
        }

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.LGMModelPath; // Default: "models"
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Check common locations for local weights file
            string[] possiblePaths = new[]
            {
                // Primary: configured path + weights file (models/model_fp16_fixrot.safetensors)
                Path.Combine(baseDir, configuredPath, "model_fp16_fixrot.safetensors"),
                // Direct models folder
                Path.Combine(baseDir, "models", "model_fp16_fixrot.safetensors"),
                // Alternative names
                Path.Combine(baseDir, "models", "lgm_weights.safetensors"),
                Path.Combine(baseDir, "models", "lgm", "model_fp16_fixrot.safetensors"),
                // If configuredPath is a direct file path
                Path.IsPathRooted(configuredPath) ? configuredPath : Path.Combine(baseDir, configuredPath),
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    Log($"[LGM] Found weights at: {path}");
                    return path;
                }
            }

            // Fallback to HuggingFace model identifier
            Log("[LGM] Local weights not found, using HuggingFace model identifier");
            return "ashawkey/LGM";
        }

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[LGM] Initializing subprocess inference...");
                _inference = new SubprocessInference("lgm");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) =>
                {
                    OnProgress?.Invoke(stage, progress, message);
                    OnLoadProgress?.Invoke(stage, progress, message);
                };

                string weightsPath = GetWeightsPath();
                string device = GetDeviceString();
                Log($"[LGM] Loading model from: {weightsPath}");

                if (_inference.Load(weightsPath, device))
                    Log("[LGM] Model loaded successfully");
                else
                    Log("[LGM] Failed to load model");
            }
            catch (Exception ex)
            {
                Log($"[LGM] Error: {ex.Message}");
            }
        }

        public MeshData GenerateFromImage(string imagePath, System.Threading.CancellationToken cancellationToken = default)
        {
            Initialize();
            var mesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[LGM] Model not loaded");
                return mesh;
            }

            try
            {
                if (!File.Exists(imagePath))
                {
                    Log($"[LGM] Image not found: {imagePath}");
                    return mesh;
                }

                var imagesBytes = new List<byte[]> { File.ReadAllBytes(imagePath) };
                Log($"[LGM] Processing image...");

                cancellationToken.ThrowIfCancellationRequested();
                var meshes = _inference.Infer(imagesBytes, false, cancellationToken);

                if (meshes.Count > 0)
                {
                    mesh = meshes[0];
                    Log($"[LGM] Generated mesh with {mesh.Vertices.Count} vertices");
                }
            }
            catch (OperationCanceledException)
            {
                Log("[LGM] Inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                Log($"[LGM] Error: {ex.Message}");
            }

            return mesh;
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
