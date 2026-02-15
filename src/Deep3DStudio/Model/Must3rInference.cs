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
    /// MUSt3R (Multi-view Stereo 3D Reconstruction) inference handler.
    /// Optimized for many images/video, runs at 8-11 FPS.
    /// Uses subprocess-based Python inference for complete process isolation.
    /// </summary>
    public class Must3rInference : IDisposable
    {
        private SubprocessInference? _inference;
        private bool _disposed = false;
        private Action<string>? _logCallback;
        private string? _lastError;

        public Must3rInference() { }

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
            string configuredPath = settings.Must3rModelPath; // Default: "models/must3r"
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;

            // Check common locations for local weights file
            string[] possiblePaths = new[]
            {
                // Primary: configured path + weights file (models/must3r/must3r_weights.pth)
                Path.Combine(baseDir, configuredPath, "must3r_weights.pth"),
                // Direct models folder
                Path.Combine(baseDir, "models", "must3r", "must3r_weights.pth"),
                Path.Combine(baseDir, "models", "must3r_weights.pth"),
                // Alternative name
                Path.Combine(baseDir, "models", "MUSt3R_512.pth"),
                // If configuredPath is a direct file path
                Path.IsPathRooted(configuredPath) ? configuredPath : Path.Combine(baseDir, configuredPath),
            };

            foreach (var path in possiblePaths)
            {
                if (File.Exists(path))
                {
                    Log($"[Must3r] Found weights at: {path}");
                    return path;
                }
            }

            // Fallback to HuggingFace model identifier
            Log("[Must3r] Local weights not found, using HuggingFace model identifier");
            return "naver/MUSt3R";
        }

        private void Initialize()
        {
            if (_inference != null && _inference.IsLoaded) return;

            try
            {
                Log("[Must3r] Initializing subprocess inference...");
                _inference = new SubprocessInference("must3r");
                _inference.OnLog += msg => Log(msg);
                _inference.OnProgress += (stage, progress, message) => OnProgress?.Invoke(stage, progress, message);

                string weightsPath = GetWeightsPath();
                string device = GetDeviceString();

                Log($"[Must3r] Loading model from: {weightsPath}");
                OnProgress?.Invoke("load", 0.1f, "Loading MUSt3R model...");

                if (_inference.Load(weightsPath, device))
                {
                    Log("[Must3r] Model loaded successfully");
                    OnProgress?.Invoke("load", 1.0f, "MUSt3R loaded");
                }
                else
                {
                    _lastError = _inference.LastOpenMpShmError
                        ? "Failed to load MUSt3R model due to OpenMP shared-memory initialization on macOS (OMP #179)."
                        : (_inference.LastErrorMessage ?? "Failed to load MUSt3R model");
                    Log($"[Must3r] {_lastError}");
                }
            }
            catch (Exception ex)
            {
                _lastError = $"Error initializing Must3r: {ex.Message}";
                Log(_lastError);
            }
        }

        public SceneResult ReconstructScene(List<string> imagePaths, bool useRetrieval = true, System.Threading.CancellationToken cancellationToken = default)
        {
            Initialize();
            var result = new SceneResult();

            if (_inference == null || !_inference.IsLoaded)
            {
                Log($"[Must3r] Not loaded. {_lastError}");
                return result;
            }

            try
            {
                if (imagePaths == null || imagePaths.Count == 0)
                {
                    Log("[Must3r] No images provided");
                    return result;
                }

                var imagesBytes = new List<byte[]>();
                var validImagePaths = new List<string>();
                var validImageSizes = new List<(int Width, int Height)>();

                foreach (var path in imagePaths)
                {
                    if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) continue;
                    imagesBytes.Add(File.ReadAllBytes(path));
                    validImagePaths.Add(path);
                    if (ImageUtils.TryGetImageDimensions(path, out int width, out int height))
                    {
                        validImageSizes.Add((width, height));
                    }
                    else
                    {
                        validImageSizes.Add((0, 0));
                        Log($"[Must3r] Warning: Could not read image size for {Path.GetFileName(path)}");
                    }
                }

                if (imagesBytes.Count == 0)
                {
                    Log("[Must3r] No valid images");
                    return result;
                }

                Log($"[Must3r] Processing {imagesBytes.Count} images...");
                OnProgress?.Invoke("inference", 0.1f, $"Processing {imagesBytes.Count} images...");

                cancellationToken.ThrowIfCancellationRequested();
                var meshes = _inference.Infer(imagesBytes, useRetrieval, cancellationToken);

                if (meshes.Count == 0)
                {
                    Log("[Must3r] No meshes returned from inference.");
                }

                for (int i = 0; i < meshes.Count; i++)
                {
                    var mesh = meshes[i];
                    
                    if (mesh.Vertices.Count > 0)
                    {
                        var min = new Vector3(float.MaxValue);
                        var max = new Vector3(float.MinValue);
                        foreach (var v in mesh.Vertices)
                        {
                            min = Vector3.ComponentMin(min, v);
                            max = Vector3.ComponentMax(max, v);
                        }
                        Log($"[Must3r] Mesh {i}: {mesh.Vertices.Count} points, bounds min({min.X:F2},{min.Y:F2},{min.Z:F2}) max({max.X:F2},{max.Y:F2},{max.Z:F2})");
                    }
                    else
                    {
                        Log($"[Must3r] Mesh {i} has 0 vertices.");
                    }

                    if (ReconstructionPoseNormalizer.TryApplyPoseIfLikelyLocal(mesh, out float localScore, out float worldScore))
                    {
                        Log($"[Must3r] Applied pose to point cloud {i}: localScore={localScore:F2}, worldScore={worldScore:F2}");
                    }
                    ReconstructionPoseNormalizer.ConvertOpenCvConventionToViewport(mesh);

                    result.Meshes.Add(mesh);
                    if (i < validImagePaths.Count)
                    {
                        var pose = new CameraPose
                        {
                            ImageIndex = i,
                            ImagePath = validImagePaths[i]
                        };
                        if (i < validImageSizes.Count)
                        {
                            pose.Width = validImageSizes[i].Width;
                            pose.Height = validImageSizes[i].Height;
                        }

                        if (mesh.Pose is Matrix4 poseMatrix)
                        {
                            pose.CameraToWorld = poseMatrix;
                            try
                            {
                                pose.WorldToCamera = pose.CameraToWorld.Inverted();
                            }
                            catch
                            {
                                pose.WorldToCamera = Matrix4.Identity;
                                Log($"[Must3r] Warning: Could not invert pose matrix for image {i}");
                            }
                        }

                        result.Poses.Add(pose);
                    }
                }

                Log($"[Must3r] Complete. {result.Meshes.Count} meshes.");
                OnProgress?.Invoke("inference", 1.0f, "Complete");
                return result;
            }
            catch (OperationCanceledException)
            {
                Log("[Must3r] Inference cancelled.");
                throw;
            }
            catch (Exception ex)
            {
                Log($"[Must3r] Error: {ex.Message}");
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
