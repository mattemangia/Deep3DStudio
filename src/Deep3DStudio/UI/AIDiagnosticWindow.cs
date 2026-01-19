using System;
using Gtk;
using System.Diagnostics;
using System.IO;
using System.Linq;
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

                // 2. Check Models using IniSettings paths
                var settings = Configuration.IniSettings.Instance;
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;

                // Build expected files list based on IniSettings paths
                var expectedModels = new List<(string name, string path, string file)>
                {
                    ("Dust3r", settings.Dust3rModelPath, "dust3r_weights.pth"),
                    ("MASt3R", settings.Mast3rModelPath, "mast3r_weights.pth"),
                    ("MASt3R Retrieval", settings.Mast3rModelPath, "mast3r_retrieval.pth"),
                    ("MASt3R Codebook", settings.Mast3rModelPath, "mast3r_retrieval_codebook.pkl"),
                    ("MUSt3R", settings.Must3rModelPath, "must3r_weights.pth"),
                    ("MUSt3R Retrieval", settings.Must3rModelPath, "must3r_retrieval.pth"),
                    ("MUSt3R Codebook", settings.Must3rModelPath, "must3r_retrieval_codebook.pkl"),
                    ("TripoSR", settings.TripoSRModelPath, "triposr_weights.pth"),
                    ("TripoSF", settings.TripoSFModelPath, "triposf_weights.pth"),
                    ("LGM", settings.LGMModelPath, "model_fp16_fixrot.safetensors"),
                    ("UniRig", settings.UniRigModelPath, "unirig_weights.pth"),
                    ("Wonder3D", settings.Wonder3DModelPath, "model_index.json"),
                    ("Wonder3D Feature Extractor", settings.Wonder3DModelPath, "feature_extractor/preprocessor_config.json"),
                    ("Wonder3D Image Encoder Config", settings.Wonder3DModelPath, "image_encoder/config.json"),
                    ("Wonder3D Image Encoder Model", settings.Wonder3DModelPath, "image_encoder/pytorch_model.bin"),
                    ("Wonder3D Scheduler", settings.Wonder3DModelPath, "scheduler/scheduler_config.json"),
                    ("Wonder3D UNet Config", settings.Wonder3DModelPath, "unet/config.json"),
                    ("Wonder3D UNet Model", settings.Wonder3DModelPath, "unet/diffusion_pytorch_model.bin"),
                    ("Wonder3D VAE Config", settings.Wonder3DModelPath, "vae/config.json"),
                    ("Wonder3D VAE Model", settings.Wonder3DModelPath, "vae/diffusion_pytorch_model.bin"),
                };

                Log($"[INFO] Base directory: {baseDir}");
                Log($"[INFO] Settings paths:");
                Log($"       Dust3r: {settings.Dust3rModelPath}");
                Log($"       TripoSR: {settings.TripoSRModelPath}");
                Log($"       TripoSF: {settings.TripoSFModelPath}");
                Log($"       LGM: {settings.LGMModelPath}");
                Log($"       UniRig: {settings.UniRigModelPath}");
                Log($"[INFO] Checking model weights based on Settings paths:");
                int foundCount = 0;
                foreach (var (name, modelPath, file) in expectedModels)
                {
                    string fullPath = System.IO.Path.IsPathRooted(modelPath)
                        ? System.IO.Path.Combine(modelPath, file)
                        : System.IO.Path.Combine(baseDir, modelPath, file);

                    if (System.IO.File.Exists(fullPath))
                    {
                        foundCount++;
                        Log($"[OK] {name}: {file}");
                    }
                    else
                    {
                        Log($"[FAIL] {name}: Missing {file}");
                        Log($"       Expected at: {fullPath}");
                    }
                }

                Log($"Summary: Found {foundCount} / {expectedModels.Count} model files.");

                UpdateProgress(0.4, "Checking Subprocess Inference...");

                // 3. Check Subprocess Inference System (new subprocess-based approach)
                try
                {
                    Log("[INFO] Checking subprocess inference system...");

                    // Check for subprocess_inference.py script
                    string exeDir = AppDomain.CurrentDomain.BaseDirectory;
                    string[] scriptPaths = new[]
                    {
                        System.IO.Path.Combine(exeDir, "subprocess_inference.py"),
                        System.IO.Path.Combine(exeDir, "Embedded", "Python", "subprocess_inference.py")
                    };

                    bool scriptFound = false;
                    foreach (var scriptPath in scriptPaths)
                    {
                        if (System.IO.File.Exists(scriptPath))
                        {
                            Log($"[OK] Subprocess script found: {scriptPath}");
                            scriptFound = true;
                            break;
                        }
                    }

                    if (!scriptFound)
                    {
                        Log("[INFO] Subprocess script not found in filesystem, checking embedded resources...");
                        var assembly = System.Reflection.Assembly.GetExecutingAssembly();
                        var resources = assembly.GetManifestResourceNames();
                        bool embeddedFound = resources.Any(r => r.EndsWith("subprocess_inference.py"));
                        if (embeddedFound)
                            Log("[OK] Subprocess script found as embedded resource.");
                        else
                            Log("[WARN] Subprocess script not found. It may be embedded in main assembly.");
                    }

                    // Test subprocess Python execution
                    string appDataDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                    string appDataPythonDir = System.IO.Path.Combine(appDataDir, "Deep3DStudio", "python");
                    string localPythonDir = System.IO.Path.Combine(exeDir, "python");

                    string? pythonExe = FindPythonExecutable(appDataPythonDir) ??
                                        FindPythonExecutable(localPythonDir);

                    if (pythonExe != null)
                    {
                        Log($"[OK] Python executable for subprocess: {pythonExe}");

                        // Test Python can run
                        try
                        {
                            var psi = new ProcessStartInfo
                            {
                                FileName = pythonExe,
                                Arguments = "--version",
                                RedirectStandardOutput = true,
                                RedirectStandardError = true,
                                UseShellExecute = false,
                                CreateNoWindow = true
                            };

                            using var proc = Process.Start(psi);
                            if (proc != null)
                            {
                                string output = proc.StandardOutput.ReadToEnd() + proc.StandardError.ReadToEnd();
                                proc.WaitForExit(5000);
                                if (proc.ExitCode == 0)
                                    Log($"[OK] Python subprocess test: {output.Trim()}");
                                else
                                    Log($"[WARN] Python subprocess returned exit code {proc.ExitCode}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Log($"[FAIL] Could not test Python subprocess: {ex.Message}");
                        }
                    }
                    else
                    {
                        Log("[WARN] Could not find Python executable for subprocess inference.");
                        Log("       Subprocess-based AI models may not work.");
                    }
                }
                catch (Exception ex)
                {
                    Log($"[ERROR] Subprocess inference check: {ex.Message}");
                }

                UpdateProgress(0.5, "Initializing Python (pythonnet)...");

                // 4. Import Libraries (pythonnet - used for some features)
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

        /// <summary>
        /// Recursively search for Python executable in the given directory.
        /// </summary>
        private string? FindPythonExecutable(string rootDir)
        {
            if (!System.IO.Directory.Exists(rootDir)) return null;

            bool isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;

            var stack = new Stack<string>();
            stack.Push(rootDir);
            int safetyCounter = 0;

            while (stack.Count > 0 && safetyCounter++ < 500)
            {
                string currentDir = stack.Pop();

                string[] candidates = isWindows
                    ? new[] { System.IO.Path.Combine(currentDir, "python.exe") }
                    : new[] {
                        System.IO.Path.Combine(currentDir, "bin", "python3"),
                        System.IO.Path.Combine(currentDir, "bin", "python"),
                        System.IO.Path.Combine(currentDir, "python3"),
                        System.IO.Path.Combine(currentDir, "python")
                    };

                foreach (var candidate in candidates)
                {
                    if (System.IO.File.Exists(candidate))
                        return candidate;
                }

                try
                {
                    foreach (string dir in System.IO.Directory.GetDirectories(currentDir))
                    {
                        string dirName = System.IO.Path.GetFileName(dir);
                        if (dirName != "__pycache__" && dirName != "site-packages" && !dirName.StartsWith("."))
                            stack.Push(dir);
                    }
                }
                catch { }
            }

            return null;
        }
    }
}
