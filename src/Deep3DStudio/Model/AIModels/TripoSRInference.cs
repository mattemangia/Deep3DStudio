using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// TripoSR - Single image to 3D mesh generation.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class TripoSRInference : IDisposable, IInferenceWithProgress
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;

        public TripoSRInference() { }

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

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[TripoSR] Initializing subprocess inference...");
                _inference = new SubprocessInference("triposr");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) =>
                {
                    OnProgress?.Invoke(stage, progress, message);
                    OnLoadProgress?.Invoke(stage, progress, message);
                };

                var settings = IniSettings.Instance;
                string weightsPath = settings.TripoSRModelPath;
                if (string.IsNullOrEmpty(weightsPath))
                    weightsPath = "stabilityai/TripoSR";

                string device = GetDeviceString();
                Log($"[TripoSR] Loading model from: {weightsPath}");

                if (_inference.Load(weightsPath, device))
                    Log("[TripoSR] Model loaded successfully");
                else
                    Log("[TripoSR] Failed to load model");
            }
            catch (Exception ex)
            {
                Log($"[TripoSR] Error: {ex.Message}");
            }
        }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[TripoSR] Model not loaded");
                return mesh;
            }

            try
            {
                if (!File.Exists(imagePath))
                {
                    Log($"[TripoSR] Image not found: {imagePath}");
                    return mesh;
                }

                var imagesBytes = new List<byte[]> { File.ReadAllBytes(imagePath) };
                Log($"[TripoSR] Processing image...");

                var meshes = _inference.Infer(imagesBytes, false);

                if (meshes.Count > 0)
                {
                    mesh = meshes[0];
                    Log($"[TripoSR] Generated mesh with {mesh.Vertices.Count} vertices");
                }
            }
            catch (Exception ex)
            {
                Log($"[TripoSR] Error: {ex.Message}");
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
