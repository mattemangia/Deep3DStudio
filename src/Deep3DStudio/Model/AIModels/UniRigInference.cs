using System;
using System.IO;
using System.Collections.Generic;
using System.Text.Json;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// UniRig - Automatic mesh rigging/skinning.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class UniRigInference : IDisposable, IInferenceWithProgress
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;

        public UniRigInference() { }

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
                Log("[UniRig] Initializing subprocess inference...");
                _inference = new SubprocessInference("unirig");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) =>
                {
                    OnProgress?.Invoke(stage, progress, message);
                    OnLoadProgress?.Invoke(stage, progress, message);
                };

                var settings = IniSettings.Instance;
                string weightsPath = settings.UniRigModelPath;
                if (string.IsNullOrEmpty(weightsPath))
                    weightsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models", "unirig_weights.pth");

                string device = GetDeviceString();
                Log($"[UniRig] Loading model from: {weightsPath}");

                if (_inference.Load(weightsPath, device))
                    Log("[UniRig] Model loaded successfully");
                else
                    Log("[UniRig] Failed to load model");
            }
            catch (Exception ex)
            {
                Log($"[UniRig] Error: {ex.Message}");
            }
        }

        public MeshData GenerateFromImage(string imagePath)
        {
            // UniRig is for rigging, not generation
            return new MeshData();
        }

        public RigResult RigMesh(MeshData mesh)
        {
            Initialize();
            var result = new RigResult();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[UniRig] Model not loaded");
                return result;
            }

            try
            {
                if (mesh.Vertices.Count == 0)
                {
                    Log("[UniRig] No vertices in mesh");
                    return result;
                }

                // For UniRig, we need to pass mesh data differently
                // The subprocess will need special handling for mesh input
                Log($"[UniRig] Rigging mesh with {mesh.Vertices.Count} vertices...");

                // Create mesh data for subprocess
                var meshVertices = new List<List<float>>();
                foreach (var v in mesh.Vertices)
                    meshVertices.Add(new List<float> { v.X, v.Y, v.Z });

                var meshFaces = new List<int>(mesh.Indices);

                // Note: UniRig subprocess needs special input handling
                // This is a simplified version - actual implementation may need
                // to serialize mesh data to a temp file

                result.Success = false;
                result.StatusMessage = "UniRig subprocess inference not fully implemented";
                Log("[UniRig] Rigging via subprocess not fully implemented yet");
            }
            catch (Exception ex)
            {
                Log($"[UniRig] Error: {ex.Message}");
                result.Success = false;
                result.StatusMessage = ex.Message;
            }

            return result;
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
