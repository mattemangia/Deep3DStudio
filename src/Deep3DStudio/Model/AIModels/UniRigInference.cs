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

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.UniRigModelPath; // Default: "models"
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Check common locations for local weights file
            string[] possiblePaths = new[]
            {
                // Primary: configured path + weights file (models/unirig_weights.pth)
                Path.Combine(baseDir, configuredPath, "unirig_weights.pth"),
                // Direct models folder
                Path.Combine(baseDir, "models", "unirig_weights.pth"),
                // Alternative names
                Path.Combine(baseDir, "models", "unirig", "unirig_weights.pth"),
                Path.Combine(baseDir, "models", "unirig", "skeleton", "articulation-xl_quantization_256", "model.ckpt"),
                // If configuredPath is a direct file path
                Path.IsPathRooted(configuredPath) ? configuredPath : Path.Combine(baseDir, configuredPath),
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    Log($"[UniRig] Found weights at: {path}");
                    return path;
                }
            }

            // Fallback to default path
            Log("[UniRig] Local weights not found");
            return Path.Combine(baseDir, "models", "unirig_weights.pth");
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

                string weightsPath = GetWeightsPath();
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

                Log($"[UniRig] Rigging mesh with {mesh.Vertices.Count} vertices...");

                var settings = IniSettings.Instance;
                result = _inference.InferRig(mesh, settings.UniRigMaxJoints, settings.UniRigMaxBonesPerVertex);

                if (result.Success)
                {
                    Log($"[UniRig] Rigging complete: {result.JointPositions?.Length ?? 0} joints.");
                }
                else
                {
                    Log($"[UniRig] Rigging failed: {result.StatusMessage}");
                }
            }
            catch (Exception ex)
            {
                Log($"[UniRig] Error: {ex.Message}");
                result.Success = false;
                result.StatusMessage = ex.Message;
            }

            return result;
        }

        public RigResult RigMeshFromFile(string meshPath)
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
                var settings = IniSettings.Instance;
                result = _inference.InferRigFromFile(meshPath, settings.UniRigMaxJoints, settings.UniRigMaxBonesPerVertex);
                if (result.Success)
                {
                    Log($"[UniRig] Rigging complete: {result.JointPositions?.Length ?? 0} joints.");
                }
                else
                {
                    Log($"[UniRig] Rigging failed: {result.StatusMessage}");
                }
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
