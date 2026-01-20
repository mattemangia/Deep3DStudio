using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Configuration;
using Deep3DStudio.IO;
using Deep3DStudio.Python;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// TripoSF (SparseFlex) - High-resolution mesh refinement model.
    /// Takes a mesh as input and produces a refined, higher-resolution mesh.
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

        private string GetWeightsPath()
        {
            var settings = IniSettings.Instance;
            string configuredPath = settings.TripoSFModelPath; // Default: "models"
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Check common locations for local weights file
            string[] possiblePaths = new[]
            {
                // Primary: configured path + weights file (models/triposf_weights.pth)
                Path.Combine(baseDir, configuredPath, "triposf_weights.pth"),
                // Direct models folder
                Path.Combine(baseDir, "models", "triposf_weights.pth"),
                // Alternative name
                Path.Combine(baseDir, "models", "pretrained_TripoSFVAE_256i1024o.safetensors"),
                // If configuredPath is a direct file path
                Path.IsPathRooted(configuredPath) ? configuredPath : Path.Combine(baseDir, configuredPath),
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    Log($"[TripoSF] Found weights at: {path}");
                    return path;
                }
            }

            // Fallback to HuggingFace model identifier
            Log("[TripoSF] Local weights not found, using HuggingFace model identifier");
            return "VAST-AI/TripoSF";
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

                string weightsPath = GetWeightsPath();
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

        /// <summary>
        /// Refine an existing mesh using TripoSF to generate a higher-resolution mesh.
        /// </summary>
        /// <param name="meshPath">Path to the input mesh file (.obj, .ply, .glb)</param>
        /// <returns>Refined mesh</returns>
        public MeshData RefineMesh(string meshPath)
        {
            Initialize();
            var resultMesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[TripoSF] Model not loaded");
                return resultMesh;
            }

            try
            {
                if (!File.Exists(meshPath))
                {
                    Log($"[TripoSF] Mesh not found: {meshPath}");
                    return resultMesh;
                }

                Log($"[TripoSF] Refining mesh: {meshPath}");

                // For TripoSF we pass the mesh path directly - the Python side handles loading
                var meshes = _inference.InferMesh(meshPath);

                if (meshes.Count > 0)
                {
                    resultMesh = meshes[0];
                    Log($"[TripoSF] Refined mesh: {resultMesh.Vertices.Count} vertices, {resultMesh.Indices.Count / 3} triangles");
                }
            }
            catch (Exception ex)
            {
                Log($"[TripoSF] Error: {ex.Message}");
            }

            return resultMesh;
        }

        /// <summary>
        /// Refine an existing mesh using TripoSF.
        /// </summary>
        /// <param name="inputMesh">Input mesh data</param>
        /// <returns>Refined mesh</returns>
        public MeshData RefineMesh(MeshData inputMesh)
        {
            Initialize();
            var resultMesh = new MeshData();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log("[TripoSF] Model not loaded");
                return resultMesh;
            }

            try
            {
                // Save input mesh to temp file
                string tempMeshPath = Path.GetTempFileName() + ".obj";
                MeshExporter.Save(tempMeshPath, inputMesh);
                Log($"[TripoSF] Saved input mesh to: {tempMeshPath}");

                var meshes = _inference.InferMesh(tempMeshPath);

                if (meshes.Count > 0)
                {
                    resultMesh = meshes[0];
                    Log($"[TripoSF] Refined mesh: {resultMesh.Vertices.Count} vertices, {resultMesh.Indices.Count / 3} triangles");
                }

                // Cleanup
                try { File.Delete(tempMeshPath); } catch { }
            }
            catch (Exception ex)
            {
                Log($"[TripoSF] Error: {ex.Message}");
            }

            return resultMesh;
        }

        [Obsolete("TripoSF takes mesh input, not images. Use RefineMesh instead.")]
        public MeshData GenerateFromImage(string imagePath)
        {
            Log("[TripoSF] Warning: GenerateFromImage is deprecated. TripoSF requires mesh input.");
            return new MeshData();
        }

        [Obsolete("Use RefineMesh with mesh input instead")]
        public MeshData RefineFromPointCloud(List<Vector3> points)
        {
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
