using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// TripoSF - Single image to 3D mesh with refinement.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class TripoSFInference : IDisposable, IInferenceWithProgress
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;

        public TripoSFInference() { }

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
                Log("[TripoSF] Initializing subprocess inference...");
                _inference = new SubprocessInference("triposf");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) =>
                {
                    OnProgress?.Invoke(stage, progress, message);
                    OnLoadProgress?.Invoke(stage, progress, message);
                };

                var settings = IniSettings.Instance;
                string weightsPath = settings.TripoSFModelPath;
                if (string.IsNullOrEmpty(weightsPath))
                    weightsPath = "stabilityai/TripoSF";

                string device = GetDeviceString();
                Log($"[TripoSF] Loading model from: {weightsPath}");

                if (_inference.Load(weightsPath, device))
                    Log("[TripoSF] Model loaded successfully");
                else
                    Log("[TripoSF] Failed to load model");
            }
            catch (Exception ex)
            {
                Log($"[TripoSF] Error: {ex.Message}");
            }
        }

        public MeshData GenerateFromImage(string imagePath)
        {
            Initialize();
            var mesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[TripoSF] Model not loaded");
                return mesh;
            }

            try
            {
                if (!File.Exists(imagePath))
                {
                    Log($"[TripoSF] Image not found: {imagePath}");
                    return mesh;
                }

                var imagesBytes = new List<byte[]> { File.ReadAllBytes(imagePath) };
                Log($"[TripoSF] Processing image...");

                var meshes = _inference.Infer(imagesBytes, false);

                if (meshes.Count > 0)
                {
                    mesh = meshes[0];
                    Log($"[TripoSF] Generated mesh with {mesh.Vertices.Count} vertices");
                }
            }
            catch (Exception ex)
            {
                Log($"[TripoSF] Error: {ex.Message}");
            }

            return mesh;
        }

        public MeshData RefineFromPointCloud(List<Vector3> points)
        {
            // Placeholder - TripoSF is primarily image-to-mesh
            return new MeshData();
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
