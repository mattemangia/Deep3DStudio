using System;
using Gtk;
using System.IO;
using System.Text;
using Deep3DStudio.Python;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;
using System.Threading.Tasks;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace Deep3DStudio.UI
{
    public class AIDiagnosticWindow : Window
    {
        private TextView _logView;
        private TextBuffer _buffer;
        private Button _runBtn;
        private Button _exportBtn;
        private ProgressBar _progressBar;

        public AIDiagnosticWindow() : base(WindowType.Toplevel)
        {
            this.Title = "AI Diagnostics";
            this.SetDefaultSize(600, 400);
            this.WindowPosition = WindowPosition.Center;
            this.Modal = true;

            var vbox = new Box(Orientation.Vertical, 5);
            vbox.Margin = 10;
            this.Add(vbox);

            // Log View
            var scroll = new ScrolledWindow();
            scroll.ShadowType = ShadowType.In;
            _buffer = new TextBuffer(new TextTagTable());
            _logView = new TextView(_buffer);
            _logView.Editable = false;
            _logView.WrapMode = WrapMode.Word;
            _logView.Monospace = true;
            scroll.Add(_logView);
            vbox.PackStart(scroll, true, true, 0);

            // Progress
            _progressBar = new ProgressBar();
            vbox.PackStart(_progressBar, false, false, 0);

            // Buttons
            var btnBox = new Box(Orientation.Horizontal, 5);
            btnBox.Halign = Align.End;

            _runBtn = new Button("Run Diagnostics");
            _runBtn.Clicked += OnRunDiagnostics;
            btnBox.PackStart(_runBtn, false, false, 0);

            _exportBtn = new Button("Export Log...");
            _exportBtn.Clicked += OnExportLog;
            btnBox.PackStart(_exportBtn, false, false, 0);

            var closeBtn = new Button("Close");
            closeBtn.Clicked += (s, e) => this.Destroy();
            btnBox.PackStart(closeBtn, false, false, 0);

            vbox.PackStart(btnBox, false, false, 0);

            this.ShowAll();

            // Auto-run on open; provide a button for manual re-run.
            Task.Run(RunDiagnosticsAsync);
        }

        private async void OnRunDiagnostics(object? sender, EventArgs e)
        {
            await RunDiagnosticsAsync();
        }

        private async Task RunDiagnosticsAsync()
        {
            Application.Invoke((s, e) => {
                _buffer.Text = "";
                _runBtn.Sensitive = false;
                _progressBar.Fraction = 0.0;
                _progressBar.Text = "Starting...";
            });

            await Task.Run(() => {
                Log("=== AI Diagnostics Started ===");
                Log($"Time: {DateTime.Now}");
                Log($"OS: {Environment.OSVersion}");
                Log($"64-bit: {Environment.Is64BitProcess}");

                UpdateProgress(0.1, "Checking Python Environment...");

                // 1. Check Python Environment
                string pythonHome = "";
                try
                {
                    string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                    string targetDir = System.IO.Path.Combine(appData, "Deep3DStudio", "python");
                    string localDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python");

                    if (System.IO.Directory.Exists(targetDir))
                    {
                         pythonHome = targetDir;
                         Log($"[OK] Python found in AppData: {targetDir}");
                    }
                    else if (System.IO.Directory.Exists(localDir))
                    {
                         pythonHome = localDir;
                         Log($"[OK] Python found in Local Directory: {localDir}");
                    }
                    else
                    {
                         Log($"[FAIL] Python environment not found.");
                         // Check for zip
                         string zipPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_env.zip");
                         if (System.IO.File.Exists(zipPath))
                             Log($"[INFO] python_env.zip exists at {zipPath}. It may need extraction.");
                         else
                             Log($"[FAIL] python_env.zip also missing.");
                    }
                }
                catch (Exception ex)
                {
                    Log($"[ERROR] Checking python env: {ex.Message}");
                }

                UpdateProgress(0.3, "Checking Model Files...");

                // 2. Check Models
                string modelsDir = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");

                // List of expected files/folders based on user input
                var expectedFiles = new Dictionary<string, bool>
                {
                    { "dust3r_weights.pth", false },
                    { "model_fp16_fixrot.safetensors", false },  // LGM (Large Multi-View Gaussian) weights
                    { "triposf_config.yaml", false },
                    { "triposf_weights.pth", false },
                    { "triposr_config.yaml", false },
                    { "triposr_weights.pth", false },
                    { "unirig_weights.pth", false },
                    // Wonder3D structure
                    { "wonder3d/model_index.json", false },
                    { "wonder3d/feature_extractor/preprocessor_config.json", false },
                    { "wonder3d/image_encoder/config.json", false },
                    { "wonder3d/image_encoder/pytorch_model.bin", false },
                    { "wonder3d/scheduler/scheduler_config.json", false },
                    { "wonder3d/unet/config.json", false },
                    { "wonder3d/unet/diffusion_pytorch_model.bin", false },
                    { "wonder3d/vae/config.json", false },
                    { "wonder3d/vae/diffusion_pytorch_model.bin", false }
                };

                if (System.IO.Directory.Exists(modelsDir))
                {
                    Log($"[OK] Models directory found at: {modelsDir}");

                    // Iterate and check
                    int foundCount = 0;
                    foreach(var key in new List<string>(expectedFiles.Keys))
                    {
                        string fullPath = System.IO.Path.Combine(modelsDir, key);
                        if (System.IO.File.Exists(fullPath))
                        {
                            expectedFiles[key] = true;
                            foundCount++;
                            Log($"[OK] Found: {key}");
                        }
                        else
                        {
                            Log($"[FAIL] Missing: {key}");
                        }
                    }

                    Log($"Summary: Found {foundCount} / {expectedFiles.Count} required model files.");
                }
                else
                {
                    Log($"[FAIL] Models directory not found at {modelsDir}");
                }

                UpdateProgress(0.5, "Initializing Python & Imports...");

                // 3. Import Libraries
                try
                {
                    var service = PythonService.Instance;
                    service.Initialize(); // Ensure init

                    if (!service.IsInitialized)
                    {
                        Log($"[FAIL] Python not initialized: {service.InitializationError}");
                    }
                    else
                    {
                        Log("[OK] Python Engine Initialized.");

                        // Log sys.path for debugging
                        using (global::Python.Runtime.Py.GIL())
                        {
                            try
                            {
                                dynamic sys = global::Python.Runtime.Py.Import("sys");
                                Log("[INFO] Current sys.path:");
                                string? sitePackagesPath = null;
                                foreach (var path in sys.path)
                                {
                                    string pathStr = path.ToString();
                                    bool exists = System.IO.Directory.Exists(pathStr) || System.IO.File.Exists(pathStr);
                                    Log($"  {(exists ? "[EXISTS]" : "[MISSING]")} {pathStr}");
                                    if (pathStr.Contains("site-packages") && System.IO.Directory.Exists(pathStr))
                                    {
                                        sitePackagesPath = pathStr;
                                    }
                                }

                                // Show site-packages contents
                                if (sitePackagesPath != null)
                                {
                                    Log($"[INFO] site-packages contents ({sitePackagesPath}):");
                                    var dirs = System.IO.Directory.GetDirectories(sitePackagesPath);
                                    int shown = 0;
                                    foreach (var dir in dirs)
                                    {
                                        if (shown++ < 15)
                                            Log($"  - {System.IO.Path.GetFileName(dir)}");
                                    }
                                    if (dirs.Length > 15)
                                        Log($"  ... and {dirs.Length - 15} more packages");
                                }
                                else
                                {
                                    Log("[WARN] No valid site-packages directory found in sys.path!");
                                }
                            }
                            catch (Exception ex)
                            {
                                Log($"[WARN] Could not retrieve sys.path: {ex.Message}");
                            }
                        }

                        // Try importing key libraries
                        string[] libs = new string[] { "numpy", "torch", "cv2", "PIL" };

                        using (global::Python.Runtime.Py.GIL())
                        {
                            foreach(var lib in libs)
                            {
                                try {
                                    global::Python.Runtime.Py.Import(lib);
                                    Log($"[OK] Import {lib} successful.");
                                } catch (Exception ex) {
                                    Log($"[FAIL] Import {lib} failed: {ex.Message}");
                                }
                            }

                            try {
                                dynamic sys = global::Python.Runtime.Py.Import("sys");

                                string bridgePath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "inference_bridge.py");

                                Log("[INFO] Checking specific model dependencies...");
                            }
                            catch(Exception ex)
                            {
                                 Log($"[WARN] Bridge check issue: {ex.Message}");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log($"[FAIL] Python initialization or imports failed: {ex.Message}");
                }

                UpdateProgress(0.7, "Checking C# Libraries (NeRF, ICP)...");

                // 4. Check C# Libraries
                try
                {
                    // Check NeRF
                    Log("[INFO] Testing VoxelGridNeRF instantiation...");
                    var nerf = new VoxelGridNeRF();
                    if (nerf != null) Log("[OK] VoxelGridNeRF instantiated.");
                    else Log("[FAIL] VoxelGridNeRF is null.");

                    // Check ICP / Point Cloud Alignment
                    Log("[INFO] Testing MeshOperations.AlignICP...");
                    var ptsA = new List<Vector3> { new Vector3(0,0,0), new Vector3(1,0,0) };
                    var ptsB = new List<Vector3> { new Vector3(0,1,0), new Vector3(1,1,0) }; // Shifted by Y=1
                    try {
                        var mat = MeshOperations.AlignICP(ptsA, ptsB, maxIterations: 5);
                        Log($"[OK] AlignICP executed. Result translation Y: {mat.Row3.Y}");
                    } catch (Exception ex) {
                        Log($"[FAIL] AlignICP execution failed: {ex.Message}");
                    }

                    // Check NeRF initialization with dummy data
                    try {
                        var dummyMesh = new MeshData();
                        dummyMesh.Vertices.Add(new Vector3(0,0,0));
                        dummyMesh.Colors.Add(new Vector3(1,1,1));
                        nerf.InitializeFromMesh(new List<MeshData> { dummyMesh });
                        Log("[OK] VoxelGridNeRF initialized with dummy data.");
                    } catch (Exception ex) {
                        Log($"[FAIL] VoxelGridNeRF initialization failed: {ex.Message}");
                    }
                }
                catch (Exception ex)
                {
                    Log($"[FAIL] C# Library check failed: {ex.Message}");
                }

                UpdateProgress(0.9, "Cleaning up...");

                // 5. Free Memory
                try
                {
                    // Force GC in C#
                    GC.Collect();
                    GC.WaitForPendingFinalizers();

                    // Force GC in Python
                    if (PythonService.Instance != null && PythonService.Instance.IsInitialized)
                    {
                        using (global::Python.Runtime.Py.GIL())
                        {
                            dynamic gc = global::Python.Runtime.Py.Import("gc");
                            gc.collect();
                            Log("[OK] Python GC collected.");
                        }
                    }
                    Log("[OK] Memory cleanup finished.");
                }
                catch(Exception ex)
                {
                    Log($"[WARN] Cleanup failed: {ex.Message}");
                }

                UpdateProgress(1.0, "Done.");
                Log("=== Diagnostics Completed ===");
            });

            Application.Invoke((s, e) => {
                _runBtn.Sensitive = true;
                _progressBar.Text = "Ready";
            });
        }

        private void Log(string message)
        {
            Application.Invoke((s, e) => {
                _buffer.InsertAtCursor(message + "\n");
                // Scroll to bottom
                var end = _buffer.EndIter;
                _logView.ScrollToIter(end, 0, false, 0, 0);
            });
        }

        private void UpdateProgress(double fraction, string text)
        {
            Application.Invoke((s, e) => {
                _progressBar.Fraction = fraction;
                _progressBar.Text = text;
            });
        }

        private void OnExportLog(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Export Log", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            fc.CurrentName = $"DiagnosticLog_{DateTime.Now:yyyyMMdd_HHmmss}.txt";

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string text = _buffer.Text;
                    System.IO.File.WriteAllText(fc.Filename, text);
                }
                catch (Exception ex)
                {
                    var err = new MessageDialog(this, DialogFlags.Modal, MessageType.Error, ButtonsType.Ok, $"Failed to save: {ex.Message}");
                    err.Run();
                    err.Destroy();
                }
            }
            fc.Destroy();
        }
    }
}
