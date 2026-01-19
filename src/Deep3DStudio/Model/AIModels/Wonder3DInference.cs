using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// Wonder3D - Single image to consistent multi-view 3D generation.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class Wonder3DInference : IDisposable, IInferenceWithProgress
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;

        public Wonder3DInference() { }

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
                AIComputeDevice.CUDA => "cuda",
                AIComputeDevice.MPS => "mps",
                _ => "cpu"
            };
        }

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.Wonder3DModelPath; // Default: "models/wonder3d"
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Check common locations for local model folder (Wonder3D uses a folder structure)
            string[] possiblePaths = new[]
            {
                // Primary: configured path (models/wonder3d)
                Path.Combine(baseDir, configuredPath),
                // Direct models folder
                Path.Combine(baseDir, "models", "wonder3d"),
                // If configuredPath is rooted
                configuredPath,
            };

            foreach (var path in possiblePaths)
            {
                // Check for model_index.json which indicates a valid Wonder3D model folder
                string modelIndex = Path.Combine(path, "model_index.json");
                if (File.Exists(modelIndex))
                {
                    Log($"[Wonder3D] Found model at: {path}");
                    return path;
                }
            }

            // Fallback to HuggingFace model identifier
            Log("[Wonder3D] Local model not found, using HuggingFace model identifier");
            return "flamehaze1115/wonder3d-v1.0";
        }

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[Wonder3D] Initializing subprocess inference...");
                _inference = new SubprocessInference("wonder3d");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) =>
                {
                    OnProgress?.Invoke(stage, progress, message);
                    OnLoadProgress?.Invoke(stage, progress, message);
                };

                string weightsPath = GetWeightsPath();
                string device = GetDeviceString();
                Log($"[Wonder3D] Loading model from: {weightsPath}");

                if (_inference.Load(weightsPath, device))
                    Log("[Wonder3D] Model loaded successfully");
                else
                    Log("[Wonder3D] Failed to load model");
            }
            catch (Exception ex)
            {
                Log($"[Wonder3D] Error: {ex.Message}");
            }
        }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[Wonder3D] Model not loaded");
                return mesh;
            }

            try
            {
                if (!File.Exists(imagePath))
                {
                    Log($"[Wonder3D] Image not found: {imagePath}");
                    return mesh;
                }

                var imagesBytes = new List<byte[]> { File.ReadAllBytes(imagePath) };
                Log($"[Wonder3D] Processing image...");

                var meshes = _inference.Infer(imagesBytes, false);

                if (meshes.Count > 0)
                {
                    mesh = meshes[0];
                    Log($"[Wonder3D] Generated mesh with {mesh.Vertices.Count} vertices");
                }
            }
            catch (Exception ex)
            {
                Log($"[Wonder3D] Error: {ex.Message}");
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
