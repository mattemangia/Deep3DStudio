using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ImGuiNET;
using Deep3DStudio.Python;
using Deep3DStudio.Scene;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio
{
    public class DrawDiagnosticsWindow
    {
        private bool _visible = false;
        private string _logText = "";
        private float _progress = 0.0f;
        private string _status = "Ready";
        private bool _isRunning = false;

        public bool Visible
        {
            get => _visible;
            set => _visible = value;
        }

        public void Draw()
        {
            if (!_visible) return;

            ImGui.SetNextWindowSize(new System.Numerics.Vector2(600, 400), ImGuiCond.FirstUseEver);
            if (ImGui.Begin("AI Diagnostics", ref _visible))
            {
                // Toolbar
                if (ImGui.Button("Run Diagnostics") && !_isRunning)
                {
                    RunDiagnostics();
                }
                ImGui.SameLine();
                if (ImGui.Button("Export Log"))
                {
                    ExportLog();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close")) _visible = false;

                ImGui.Separator();

                // Progress
                ImGui.Text(_status);
                ImGui.ProgressBar(_progress, new System.Numerics.Vector2(-1, 0));

                ImGui.Separator();

                // Log
                // ImGui.NET 1.91+ uses ImGuiChildFlags
                ImGui.BeginChild("LogRegion", new System.Numerics.Vector2(0, 0), ImGuiChildFlags.None, ImGuiWindowFlags.HorizontalScrollbar);
                ImGui.TextUnformatted(_logText);
                if (_isRunning) ImGui.SetScrollHereY(1.0f);
                ImGui.EndChild();
            }
            ImGui.End();
        }

        private async void RunDiagnostics()
        {
            _isRunning = true;
            _logText = "";
            _progress = 0;
            _status = "Starting...";

            await Task.Run(() =>
            {
                Log("=== AI Diagnostics Started ===");
                Log($"Time: {DateTime.Now}");
                Log($"OS: {Environment.OSVersion}");
                Log($"64-bit: {Environment.Is64BitProcess}");

                SetProgress(0.1f, "Checking Python Environment...");

                // 1. Check Python
                try
                {
                    string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                    string targetDir = Path.Combine(appData, "Deep3DStudio", "python");
                    string localDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python");

                    if (Directory.Exists(targetDir)) Log($"[OK] Python found in AppData: {targetDir}");
                    else if (Directory.Exists(localDir)) Log($"[OK] Python found in Local Directory: {localDir}");
                    else
                    {
                        Log($"[FAIL] Python environment not found.");
                        string zipPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_env.zip");
                        if (File.Exists(zipPath)) Log($"[INFO] python_env.zip exists at {zipPath}.");
                        else Log($"[FAIL] python_env.zip also missing.");
                    }
                }
                catch (Exception ex) { Log($"[ERROR] Python check: {ex.Message}"); }

                SetProgress(0.3f, "Checking Models...");

                // 2. Check Models
                string modelsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");
                var expectedFiles = new List<string>
                {
                    "dust3r_weights.pth",
                    "mast3r/mast3r_weights.pth",        // MASt3R - Matching And Stereo 3D Reconstruction
                    "mast3r/mast3r_retrieval.pth",      // MASt3R retrieval for unordered images (optional)
                    "mast3r/mast3r_retrieval_codebook.pkl",
                    "must3r/must3r_weights.pth",        // MUSt3R - Multi-view Network (video support)
                    "must3r/must3r_retrieval.pth",      // MUSt3R retrieval for unordered images (optional)
                    "must3r/must3r_retrieval_codebook.pkl",
                    "model_fp16_fixrot.safetensors",
                    "triposr_weights.pth",
                    "wonder3d/model_index.json"
                };

                if (Directory.Exists(modelsDir))
                {
                    Log($"[OK] Models dir: {modelsDir}");
                    foreach (var f in expectedFiles)
                    {
                        if (File.Exists(Path.Combine(modelsDir, f))) Log($"[OK] Found {f}");
                        else Log($"[FAIL] Missing {f}");
                    }
                }
                else Log($"[FAIL] Models directory missing.");

                SetProgress(0.5f, "Initializing Python...");

                // 3. Import Libs
                try
                {
                    PythonService.Instance.Initialize();

                    if (!PythonService.Instance.IsInitialized)
                    {
                        Log($"[FAIL] Python not initialized: {PythonService.Instance.InitializationError}");
                    }
                    else
                    {
                        Log("[OK] Python Engine Initialized.");

                        using (global::Python.Runtime.Py.GIL())
                        {
                            // Show sys.path for debugging
                            try
                            {
                                dynamic sys = global::Python.Runtime.Py.Import("sys");
                                Log("[INFO] Python sys.path:");
                                foreach (var p in sys.path)
                                {
                                    string pathStr = p.ToString();
                                    bool exists = Directory.Exists(pathStr) || File.Exists(pathStr);
                                    Log($"  {(exists ? "[OK]" : "[MISSING]")} {pathStr}");
                                }

                                // Check site-packages specifically
                                string? sitePackagesPath = null;
                                foreach (var p in sys.path)
                                {
                                    string pathStr = p.ToString();
                                    if (pathStr.Contains("site-packages") && Directory.Exists(pathStr))
                                    {
                                        sitePackagesPath = pathStr;
                                        break;
                                    }
                                }

                                if (sitePackagesPath != null)
                                {
                                    Log($"[INFO] site-packages contents ({sitePackagesPath}):");
                                    var dirs = Directory.GetDirectories(sitePackagesPath).Take(15);
                                    foreach (var dir in dirs)
                                    {
                                        Log($"  - {Path.GetFileName(dir)}");
                                    }
                                    if (Directory.GetDirectories(sitePackagesPath).Length > 15)
                                        Log($"  ... and {Directory.GetDirectories(sitePackagesPath).Length - 15} more");
                                }
                                else
                                {
                                    Log("[WARN] No valid site-packages directory found in sys.path!");
                                }
                            }
                            catch (Exception ex)
                            {
                                Log($"[WARN] Could not inspect sys.path: {ex.Message}");
                            }

                            string[] libs = { "numpy", "torch", "cv2", "PIL" };
                            foreach(var lib in libs)
                            {
                                try {
                                    global::Python.Runtime.Py.Import(lib);
                                    Log($"[OK] Import {lib} success.");
                                } catch (Exception ex) {
                                    Log($"[FAIL] Import {lib} failed: {ex.Message}");
                                }
                            }
                        }
                    }
                }
                catch (Exception ex) { Log($"[FAIL] Python Init failed: {ex.Message}"); }

                SetProgress(0.7f, "Checking C# Libraries...");

                // 4. C# Libs
                try
                {
                    var nerf = new VoxelGridNeRF();
                    Log("[OK] VoxelGridNeRF instantiated.");

                    var ptsA = new List<Vector3> { Vector3.Zero };
                    var ptsB = new List<Vector3> { Vector3.UnitX };
                    MeshOperations.AlignICP(ptsA, ptsB);
                    Log("[OK] MeshOperations.AlignICP executed.");
                }
                catch (Exception ex) { Log($"[FAIL] C# Lib check failed: {ex.Message}"); }

                SetProgress(1.0f, "Done.");
                Log("=== Diagnostics Completed ===");
            });

            _isRunning = false;
            _status = "Ready";
        }

        private void Log(string msg)
        {
            _logText += msg + "\n";
        }

        private void SetProgress(float p, string s)
        {
            _progress = p;
            _status = s;
        }

        private void ExportLog()
        {
            // Simple save to file in app dir for cross-platform simplicity without Nfd for now if lazy,
            // but we have Nfd.
            string path;
            var res = NativeFileDialogs.Net.Nfd.SaveDialog(out path, new Dictionary<string,string>{{"Text", "txt"}});
            if (res == NativeFileDialogs.Net.NfdStatus.Ok)
            {
                File.WriteAllText(path, _logText);
            }
        }
    }
}
