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
    /// DUSt3R (Dense Unconstrained Stereo 3D Reconstruction) inference handler.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class Dust3rInference : IDisposable
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;
        private string? _lastError;

        public Dust3rInference() { }

        public bool IsLoaded => _inference?.IsLoaded ?? false;
        public string? LastError => _lastError;

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
                AIComputeDevice.ROCm => "cuda",
                AIComputeDevice.MPS => "mps",
                _ => "cpu"
            };
        }

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.Dust3rModelPath;

            if (configuredPath.Contains("/") && !Path.IsPathRooted(configuredPath))
                return configuredPath;

            if (Path.IsPathRooted(configuredPath) && File.Exists(configuredPath))
                return configuredPath;

            string[] possiblePaths = new[]
            {
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, configuredPath),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "dust3r_weights.pth"),
            };

            foreach (var path in possiblePaths)
                if (File.Exists(path)) return path;

            return "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt";
        }

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[Dust3r] Initializing subprocess inference...");
                _inference = new SubprocessInference("dust3r");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) => OnProgress?.Invoke(stage, progress, message);

                string weightsPath = GetWeightsPath();
                string device = GetDeviceString();

                Log($"[Dust3r] Loading model from: {weightsPath}");
                OnProgress?.Invoke("load", 0.1f, "Loading DUSt3R model...");

                if (_inference.Load(weightsPath, device))
                {
                    Log("[Dust3r] Model loaded successfully");
                    OnProgress?.Invoke("load", 1.0f, "DUSt3R loaded");
                }
                else
                {
                    _lastError = "Failed to load DUSt3R model";
                    Log($"[Dust3r] {_lastError}");
                }
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Dust3r: {ex.Message}";
                Log(_lastError);
            }
        }

        public SceneResult ReconstructScene(List<string> imagePaths)
        {
            Initialize();
            var result = new SceneResult();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log($"[Dust3r] Not loaded. {_lastError}");
                return result;
            }

            try
            {
                if (imagePaths == null || imagePaths.Count == 0)
                {
                    Log("[Dust3r] No images provided");
                    return result;
                }

                var imagesBytes = new List<byte[]>();
                var validImagePaths = new List<string>();

                foreach (var path in imagePaths)
                {
                    if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) continue;
                    imagesBytes.Add(File.ReadAllBytes(path));
                    validImagePaths.Add(path);
                }

                if (imagesBytes.Count == 0)
                {
                    Log("[Dust3r] No valid images");
                    return result;
                }

                Log($"[Dust3r] Processing {imagesBytes.Count} images...");
                OnProgress?.Invoke("inference", 0.1f, $"Processing {imagesBytes.Count} images...");

                var meshes = _inference.Infer(imagesBytes, false);

                for (int i = 0; i < meshes.Count; i++)
                {
                    result.Meshes.Add(meshes[i]);
                    if (i < validImagePaths.Count)
                        result.Poses.Add(new CameraPose { ImageIndex = i, ImagePath = validImagePaths[i] });
                }

                Log($"[Dust3r] Complete. {result.Meshes.Count} meshes.");
                OnProgress?.Invoke("inference", 1.0f, "Complete");
                return result;
            }
            catch (Exception ex)
            {
                Log($"[Dust3r] Error: {ex.Message}");
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
