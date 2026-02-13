using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using ImGuiNET;
using Deep3DStudio.Viewport;
using Deep3DStudio.Scene;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using System.Drawing;
using Deep3DStudio.IO;
using NativeFileDialogs.Net;
using Deep3DStudio.Model;
using Deep3DStudio.Model.AIModels;
using Deep3DStudio.Meshing;
using System.Threading.Tasks;
using System.IO;
using System.Linq;
using Deep3DStudio.UI;
using System.Reflection;
using SkiaSharp;

namespace Deep3DStudio
{
    public partial class MainWindow : GameWindow
    {
        private enum PendingUnsavedAction
        {
            None,
            Exit,
            NewProject,
            OpenProject
        }

        private enum TransformDialogMode
        {
            Move,
            Rotate,
            Scale
        }

        private enum ImagePanelMode
        {
            RGB,
            DepthMap
        }

        private ImGuiController _controller;
        private ThreeDView _viewport;
        private SceneGraph _sceneGraph;
        private ImGuiIconFactory _iconFactory;
        private SceneResult? _lastSceneResult;

        // State
        private int _selectedWorkflow = 0;
        private int _selectedQuality = 1;
        private bool _ctrlModifierDown;
        private bool _shiftModifierDown;
        private bool _superModifierDown;
        // Dynamic workflow names - first entry uses Settings reconstruction method
        private string[] _workflowsBase = {
            "Multi-View (Settings)", // Uses IniSettings.ReconstructionMethod
            "Feature Matching (SfM)",
            "TripoSR (Single Image)",
            "LGM (Single Image)",
            "Wonder3D (Single Image)"
        };

        /// <summary>
        /// Gets the current reconstruction engine display name from settings.
        /// </summary>
        private string GetCurrentEngineName()
        {
            return IniSettings.Instance.ReconstructionMethod switch
            {
                ReconstructionMethod.Dust3r => "Dust3R",
                ReconstructionMethod.Mast3r => "MASt3R",
                ReconstructionMethod.Must3r => "MUSt3R",
                ReconstructionMethod.FeatureMatching => "SfM",
                ReconstructionMethod.TripoSR => "TripoSR",
                ReconstructionMethod.Wonder3D => "Wonder3D",
                _ => "Dust3R"
            };
        }

        /// <summary>
        /// Gets the workflow names with dynamic first entry based on current settings.
        /// </summary>
        private string[] GetWorkflowNames()
        {
            var result = (string[])_workflowsBase.Clone();
            result[0] = $"Multi-View ({GetCurrentEngineName()})";
            return result;
        }

        /// <summary>
        /// Gets the correct reconstruction workflow step based on settings.
        /// </summary>
        private WorkflowStep GetReconstructionStep()
        {
            return IniSettings.Instance.ReconstructionMethod switch
            {
                ReconstructionMethod.Mast3r => WorkflowStep.Mast3rReconstruction,
                ReconstructionMethod.Must3r => WorkflowStep.Must3rReconstruction,
                ReconstructionMethod.FeatureMatching => WorkflowStep.SfMReconstruction,
                _ => WorkflowStep.Dust3rReconstruction
            };
        }
        private string[] _qualities = { "Fast", "Balanced", "High" };
        private string _videoFilePath = ""; // For MUSt3R video input
        private bool _hasVideoInput = false;

        // Auto Workflow Toggle - when enabled, Play button runs the full selected workflow
        // When disabled, user can manually trigger each step (e.g., Dust3R -> then LGM -> then UniRig)
        private bool _autoWorkflowEnabled = true;
        private bool _workflowInProgress = false;
        private string _logBuffer = "";
        private bool _autoScroll = true;
        private int _lastLogLength = 0;
        private float _lastLogWidth = 0;
        private float _cachedLogHeight = 0;

        // Selection tracking for Log
        private int _logSelectionStart = 0;
        private int _logSelectionEnd = 0;
        private int _savedSelectionStart = 0;
        private int _savedSelectionEnd = 0;
        private ImGuiInputTextCallback _logCallback;

        // UI Windows
        private bool _showSettings = false;
        private bool _showAbout = false;
        private bool _showImagePreview = false;
        private string _previewImagePath = "";
        private int _previewTexture = -1;
        private bool _previewShowsDepth = false;
        private const int MaxDepthPreviewTextureSize = 2048;
        private DrawDiagnosticsWindow _diagnosticsWindow = new DrawDiagnosticsWindow();
        private int _logoTexture = -1;

        // Error Display
        private bool _showError = false;
        private string _errorTitle = "";
        private string _errorMessage = "";
        private string _errorStackTrace = "";
        private bool _errorExpanded = false;

        // Image List with Thumbnails
        private List<ProjectImage> _loadedImages = new List<ProjectImage>();
        private Dictionary<string, int> _imageThumbnails = new Dictionary<string, int>();
        private Dictionary<string, int> _imageDepthThumbnails = new Dictionary<string, int>();
        private int _selectedImageIndex = -1;
        private ImagePanelMode _imagePanelMode = ImagePanelMode.RGB;

        // Renaming state
        private SceneObject _renamingObject = null;
        private string _renameBuffer = "";
        private ProjectImage _renamingImage = null;
        private string _imageRenameBuffer = "";

        // Layout
        private float _leftPanelWidth = 280;
        private float _rightPanelWidth = 280;
        private float _logPanelHeight = 150;
        private float _toolbarHeight = 42;
        private float _auxToolbarHeight = 34;
        private float _verticalToolbarWidth = 38;
        private static readonly System.Numerics.Vector2 ToolbarIconSize = new System.Numerics.Vector2(20, 20);
        private static readonly System.Numerics.Vector2 ToolbarWindowPadding = new System.Numerics.Vector2(6, 6);
        private static readonly System.Numerics.Vector2 ToolbarItemSpacing = new System.Numerics.Vector2(6, 4);

        // View State
        private bool _showTopToolbar = true;
        private bool _showMeshEditorToolbar = false;
        private bool _showPointCloudToolbar = false;
        private bool _showGeoreferenceToolbar = false;
        private bool _showLeftPanel = true;
        private bool _showRightPanel = true;
        private bool _showLogPanel = true;
        private bool _showVerticalToolbar = true;

        // Splash State
        private bool _showSplash = true;
        private bool _pythonReady = false;

        // Project State
        private bool _isDirty = false;
        private string _currentProjectPath = "";
        private bool _showUnsavedChangesPrompt = false;
        private PendingUnsavedAction _pendingUnsavedAction = PendingUnsavedAction.None;
        private string? _pendingOpenProjectPath = null;
        private bool _executePendingActionAfterSave = false;

        // Popup Management
        private string? _popupToOpen = null;
        private bool _showTransformDialog = false;
        private TransformDialogMode _transformDialogMode = TransformDialogMode.Move;
        private System.Numerics.Vector3 _transformDialogValue = System.Numerics.Vector3.Zero;

        // Pen editing parameters
        private float _penExtrudeDistance = 0.05f;
        private float _penInsetAmount = 0.2f;
        private System.Numerics.Vector3 _penMoveDelta = System.Numerics.Vector3.Zero;

        // Primitive creation presets
        private float _primSize = 1.0f;
        private float _primRadius = 0.5f;
        private float _primHeight = 1.0f;
        private int _primSegments = 24;
        private int _primRings = 16;
        private int _primMinorSegments = 16;
        private int _primPolygonSides = 6;
        private int _primGridCells = 12;
        private float _primCellSize = 0.1f;
        private bool _primCapEnds = true;
        private bool _showPrimitiveDialog = false;
        private MeshPrimitiveType _primitiveDialogType = MeshPrimitiveType.Cube;

        // Point cloud editor parameters
        private float _pcVoxelSize = 0.02f;
        private int _pcOutlierK = 20;
        private float _pcOutlierStdRatio = 2.0f;
        private float _pcDuplicateThreshold = 0.001f;
        private int _pcNormalK = 30;
        private int _pcPassAxis = 2;
        private float _pcPassMin = -0.5f;
        private float _pcPassMax = 0.5f;
        private System.Numerics.Vector3 _pcRadiusCenter = System.Numerics.Vector3.Zero;
        private float _pcRadius = 1.0f;
        private float _pcDenseRadius = 0.03f;
        private int _pcDensePointsPerSeed = 2;
        private float _pcSkyMinBlue = 0.45f;
        private float _pcSkyMaxRed = 0.60f;
        private float _pcSkyMaxGreen = 0.75f;
        private float _pcSkyBlueDominance = 0.08f;

        // Rigging state
        private SkeletonObject? _activeSkeletonObject = null;

        // Toolbar option popups
        private bool _showRunOptionsDialog = false;
        private bool _showMeshingOptionsDialog = false;
        private bool _showRefinementOptionsDialog = false;
        private bool _showPcVoxelDialog = false;
        private bool _showPcOutliersDialog = false;
        private bool _showPcDuplicatesDialog = false;
        private bool _showPcNormalsDialog = false;
        private bool _showPcPassDialog = false;
        private bool _showPcRadiusDialog = false;
        private bool _showPcDenseDialog = false;
        private bool _showPcSkyBlueDialog = false;
        private bool _showPenMoveDialog = false;
        private bool _showPenExtrudeDialog = false;
        private bool _showPenInsetDialog = false;

        // Threading
        private readonly System.Collections.Concurrent.ConcurrentQueue<Action> _pendingActions = new System.Collections.Concurrent.ConcurrentQueue<Action>();
        // Display metrics (logical client points vs framebuffer pixels)
        private int _clientWidth = 1;
        private int _clientHeight = 1;
        private int _framebufferWidth = 1;
        private int _framebufferHeight = 1;
        private float _framebufferScaleX = 1.0f;
        private float _framebufferScaleY = 1.0f;

        public MainWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
            _sceneGraph = new SceneGraph();
            _sceneGraph.SceneChanged += (s, e) => { _isDirty = true; UpdateTitle(); };
            _viewport = new ThreeDView(_sceneGraph);

            // Keep callback alive
            unsafe
            {
                _logCallback = OnLogCallback;
            }
        }

        protected override void OnLoad()
        {
            base.OnLoad();

            Title += ": OpenGL Version: " + GL.GetString(StringName.Version);
            UpdateTitle();

            RefreshDisplayMetrics();
            _controller = new ImGuiController(_clientWidth, _clientHeight, _framebufferWidth, _framebufferHeight);

            // Configure ImGui style
            ConfigureImGuiStyle();

            // Init Python extraction progress hook
            PythonService.Instance.OnExtractionProgress += (message, progress) => {
                EnqueueAction(() => {
                    // Start dialog if this is the beginning of extraction
                    if (progress < 0.1f && !ProgressDialog.Instance.IsVisible)
                    {
                        ProgressDialog.Instance.Start("Extracting Python Environment", OperationType.Processing);
                    }

                    // Update progress
                    if (ProgressDialog.Instance.IsVisible)
                    {
                        ProgressDialog.Instance.Update(progress, message);
                    }

                    // Complete when done
                    if (progress >= 1.0f && ProgressDialog.Instance.IsVisible &&
                        ProgressDialog.Instance.State == ProgressState.Running)
                    {
                        ProgressDialog.Instance.Complete();
                    }
                });
            };

            // Init Python service log hook
            PythonService.Instance.OnLogOutput += (msg) => {
                _logBuffer += msg + "\n";
                // Also forward to progress dialog if active
                if (ProgressDialog.Instance.IsVisible)
                {
                    ProgressDialog.Instance.Log(msg);
                }
            };

            // Init AI Manager hooks
            AIModelManager.Instance.ProgressUpdated += (status, progress) => {
                ProgressDialog.Instance.Update(progress, status);
            };

            // Hook up model loading progress for progress bar during model initialization
            AIModelManager.Instance.ModelLoadProgress += (stage, progress, message) => {
                // Start progress dialog if not visible and we're starting to load
                if (!ProgressDialog.Instance.IsVisible && stage == "init")
                {
                    EnqueueAction(() => {
                        ProgressDialog.Instance.Start("Loading AI Model...", OperationType.Processing);
                    });
                }

                // Update progress
                EnqueueAction(() => {
                    if (ProgressDialog.Instance.IsVisible)
                    {
                        ProgressDialog.Instance.Update(progress, message);
                        ProgressDialog.Instance.Log($"[{stage}] {message}");

                        // Complete dialog when fully loaded
                        if (stage == "load" && progress >= 1.0f)
                        {
                            ProgressDialog.Instance.Complete();
                        }
                        else if (stage == "error")
                        {
                            ProgressDialog.Instance.Fail(new Exception(message));
                        }
                    }
                });
            };

            // Init Viewport GL state
            _viewport.InitGL();

            // Init Icons
            _iconFactory = new ImGuiIconFactory();

            // Restore toolbar visibility preferences
            var settings = IniSettings.Instance;
            _showTopToolbar = settings.ShowTopToolbar;
            _showVerticalToolbar = settings.ShowVerticalToolbar;
            _showMeshEditorToolbar = settings.ShowMeshEditorToolbar;
            _showPointCloudToolbar = settings.ShowPointCloudToolbar;
            _showGeoreferenceToolbar = settings.ShowGeoreferenceToolbar;

            // In ImGui mode, non-primary toolbars start disabled by default at app launch.
            // This is enforced even when an older settings file had them enabled.
            _showMeshEditorToolbar = false;
            _showPointCloudToolbar = false;
            _showGeoreferenceToolbar = false;
            settings.ShowMeshEditorToolbar = false;
            settings.ShowPointCloudToolbar = false;
            settings.ShowGeoreferenceToolbar = false;

            // Load Logo - try embedded resource first, fallback to runtime-generated logo
            _logoTexture = TextureLoader.LoadTextureFromResource("logo.png");
            if (_logoTexture == -1)
            {
                // Fallback to runtime-generated logo (especially needed on macOS)
                _logoTexture = TextureLoader.CreateRuntimeLogo(256);
            }

            // Force Python Init if not started
            Task.Run(() => {
                try {
                    PythonService.Instance.Initialize();
                    if (!PythonService.Instance.IsInitialized)
                    {
                        string error = PythonService.Instance.InitializationError;
                        if (string.IsNullOrEmpty(error)) error = "Python environment not found.";

                        EnqueueAction(() => {
                            ShowError("Python Environment Missing",
                                "The Python environment required for AI features could not be loaded.\n\n" +
                                error + "\n\n" +
                                "Please run 'setup_deployment.py' to install the required dependencies.\n" +
                                "AI features will be disabled.");
                        });
                    }
                } catch(Exception ex) {
                    _logBuffer += $"Python Init Error: {ex.Message}\n";
                    EnqueueAction(() => {
                        ShowError("Python Initialization Error", "An error occurred while initializing Python:\n" + ex.Message, ex);
                    });
                }
                _pythonReady = true;
                // Auto-close splash after a minimum time or when ready
                System.Threading.Thread.Sleep(1000);
                _showSplash = false;
            });
        }

        private void ConfigureImGuiStyle()
        {
            var style = ImGui.GetStyle();

            // Colors - Dark theme with blue accents
            var colors = style.Colors;
            colors[(int)ImGuiCol.WindowBg] = new System.Numerics.Vector4(0.12f, 0.12f, 0.12f, 1.0f);
            colors[(int)ImGuiCol.ChildBg] = new System.Numerics.Vector4(0.14f, 0.14f, 0.14f, 1.0f);
            colors[(int)ImGuiCol.PopupBg] = new System.Numerics.Vector4(0.08f, 0.08f, 0.08f, 0.98f);
            // Keep toolbar option dialogs readable without dimming the whole UI.
            colors[(int)ImGuiCol.ModalWindowDimBg] = new System.Numerics.Vector4(0.0f, 0.0f, 0.0f, 0.0f);
            colors[(int)ImGuiCol.Border] = new System.Numerics.Vector4(0.25f, 0.25f, 0.25f, 1.0f);
            colors[(int)ImGuiCol.FrameBg] = new System.Numerics.Vector4(0.18f, 0.18f, 0.18f, 1.0f);
            colors[(int)ImGuiCol.FrameBgHovered] = new System.Numerics.Vector4(0.25f, 0.25f, 0.25f, 1.0f);
            colors[(int)ImGuiCol.FrameBgActive] = new System.Numerics.Vector4(0.30f, 0.30f, 0.30f, 1.0f);
            colors[(int)ImGuiCol.TitleBg] = new System.Numerics.Vector4(0.08f, 0.08f, 0.08f, 1.0f);
            colors[(int)ImGuiCol.TitleBgActive] = new System.Numerics.Vector4(0.15f, 0.35f, 0.55f, 1.0f);
            colors[(int)ImGuiCol.MenuBarBg] = new System.Numerics.Vector4(0.14f, 0.14f, 0.14f, 1.0f);
            colors[(int)ImGuiCol.Header] = new System.Numerics.Vector4(0.20f, 0.40f, 0.60f, 0.8f);
            colors[(int)ImGuiCol.HeaderHovered] = new System.Numerics.Vector4(0.25f, 0.50f, 0.75f, 0.9f);
            colors[(int)ImGuiCol.HeaderActive] = new System.Numerics.Vector4(0.30f, 0.55f, 0.80f, 1.0f);
            colors[(int)ImGuiCol.Button] = new System.Numerics.Vector4(0.20f, 0.40f, 0.60f, 1.0f);
            colors[(int)ImGuiCol.ButtonHovered] = new System.Numerics.Vector4(0.25f, 0.50f, 0.75f, 1.0f);
            colors[(int)ImGuiCol.ButtonActive] = new System.Numerics.Vector4(0.30f, 0.55f, 0.80f, 1.0f);
            colors[(int)ImGuiCol.Tab] = new System.Numerics.Vector4(0.15f, 0.15f, 0.15f, 1.0f);
            colors[(int)ImGuiCol.TabHovered] = new System.Numerics.Vector4(0.25f, 0.50f, 0.75f, 1.0f);
            colors[(int)ImGuiCol.TabSelected] = new System.Numerics.Vector4(0.20f, 0.40f, 0.60f, 1.0f);
            colors[(int)ImGuiCol.ScrollbarBg] = new System.Numerics.Vector4(0.10f, 0.10f, 0.10f, 1.0f);
            colors[(int)ImGuiCol.ScrollbarGrab] = new System.Numerics.Vector4(0.30f, 0.30f, 0.30f, 1.0f);
            colors[(int)ImGuiCol.ScrollbarGrabHovered] = new System.Numerics.Vector4(0.40f, 0.40f, 0.40f, 1.0f);
            colors[(int)ImGuiCol.ScrollbarGrabActive] = new System.Numerics.Vector4(0.50f, 0.50f, 0.50f, 1.0f);

            // Sizing
            style.WindowRounding = 4.0f;
            style.FrameRounding = 3.0f;
            style.GrabRounding = 2.0f;
            style.WindowPadding = new System.Numerics.Vector2(8, 8);
            style.FramePadding = new System.Numerics.Vector2(4, 3);
            style.ItemSpacing = new System.Numerics.Vector2(6, 4);
        }

        private void UpdateTitle()
        {
            string title = "Deep3DStudio (Cross-Platform / ImGui)";
            if (!string.IsNullOrEmpty(_currentProjectPath))
                title = $"Deep3DStudio - {Path.GetFileName(_currentProjectPath)}";
            if (_isDirty) title += " *";
            Title = title + ": OpenGL Version: " + GL.GetString(StringName.Version);
        }

        private static string GetAppVersionText()
        {
            var assembly = Assembly.GetExecutingAssembly();
            var informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
            if (!string.IsNullOrWhiteSpace(informational))
            {
                var clean = informational.Split('+')[0];
                return $"Version {clean}";
            }

            var version = assembly.GetName().Version;
            return version != null ? $"Version {version}" : "Version unknown";
        }

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            if (_showUnsavedChangesPrompt)
            {
                e.Cancel = true;
                base.OnClosing(e);
                return;
            }

            if (_isDirty && !_showUnsavedChangesPrompt)
            {
                e.Cancel = true;
                _pendingUnsavedAction = PendingUnsavedAction.Exit;
                _showUnsavedChangesPrompt = true;
            }
            base.OnClosing(e);
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            _clientWidth = Math.Max(1, e.Width);
            _clientHeight = Math.Max(1, e.Height);
            UpdateFramebufferScale();
            GL.Viewport(0, 0, _framebufferWidth, _framebufferHeight);
            _controller?.UpdateDisplayMetrics(_clientWidth, _clientHeight, _framebufferWidth, _framebufferHeight);
        }

        protected override void OnFramebufferResize(FramebufferResizeEventArgs e)
        {
            base.OnFramebufferResize(e);
            _framebufferWidth = Math.Max(1, e.Width);
            _framebufferHeight = Math.Max(1, e.Height);
            UpdateFramebufferScale();
            GL.Viewport(0, 0, _framebufferWidth, _framebufferHeight);
            _controller?.UpdateDisplayMetrics(_clientWidth, _clientHeight, _framebufferWidth, _framebufferHeight);
        }

        protected override void OnTextInput(TextInputEventArgs e)
        {
            base.OnTextInput(e);
            _controller.PressChar((char)e.Unicode);
        }

        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            _controller.MouseScroll(new System.Numerics.Vector2(e.Offset.X, e.Offset.Y));

            if (!ImGui.GetIO().WantCaptureMouse && !_showSplash)
            {
                _viewport.OnMouseWheel(e.OffsetY);
            }
        }

        protected override void OnFileDrop(FileDropEventArgs e)
        {
            base.OnFileDrop(e);
            foreach (var file in e.FileNames)
            {
                ImportFile(file);
            }
        }

        private void ImportFile(string file)
        {
            Logger.Info($"ImportFile called: {file}");
            string ext = Path.GetExtension(file).ToLower();

            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            {
                Logger.Debug($"File is an image (ext: {ext})");
                if (!_loadedImages.Any(i => i.FilePath == file))
                {
                    var pImg = new ProjectImage { FilePath = file, Alias = Path.GetFileName(file) };
                    _loadedImages.Add(pImg);
                    Logger.Info($"Image added to list: {Path.GetFileName(file)}");

                    // Queue thumbnail creation on the main thread via pending actions
                    // OpenGL calls MUST happen on the main thread to avoid segfault
                    Logger.Debug("Queueing thumbnail creation on main thread...");
                    EnqueueAction(() => {
                        Logger.Debug($"Executing thumbnail creation for: {Path.GetFileName(file)}");
                        try
                        {
                            var thumb = TextureLoader.CreateThumbnail(file, 64);
                            if (thumb > 0)
                            {
                                lock (_imageThumbnails)
                                {
                                    _imageThumbnails[file] = thumb;
                                }
                                Logger.Info($"Thumbnail created successfully for: {Path.GetFileName(file)}");
                            }
                            else
                            {
                                Logger.Warn($"Thumbnail creation returned invalid ID for: {Path.GetFileName(file)}");
                            }
                        }
                        catch (Exception ex)
                        {
                            Logger.Exception(ex, $"Failed to create thumbnail for: {Path.GetFileName(file)}");
                        }
                    });
                    _logBuffer += $"Added image: {Path.GetFileName(file)}\n";
                }
                else
                {
                    Logger.Debug($"Image already loaded, skipping: {file}");
                }
            }
            else if (ext == ".obj" || ext == ".glb" || ext == ".stl")
            {
                ProgressDialog.Instance.Start($"Importing {Path.GetFileName(file)}...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var mesh = MeshImporter.Load(file);
                        if (mesh != null)
                        {
                            var obj = new MeshObject(Path.GetFileName(file), mesh);
                            EnqueueAction(() =>
                            {
                                _sceneGraph.AddObject(obj);
                                ProgressDialog.Instance.Log($"Imported mesh: {Path.GetFileName(file)}");
                                ProgressDialog.Instance.Complete();
                            });
                        }
                        else
                        {
                            throw new Exception("Failed to load mesh data.");
                        }
                    }
                    catch (Exception ex)
                    {
                        EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                    }
                });
            }
            else if (ext == ".xyz" || ext == ".ply")
            {
                ProgressDialog.Instance.Start($"Importing {Path.GetFileName(file)}...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var pc = PointCloudImporter.Load(file);
                        if (pc != null)
                        {
                            EnqueueAction(() =>
                            {
                                IniSettings.Instance.ShowPointCloud = true;
                                _sceneGraph.AddObject(pc);
                                ProgressDialog.Instance.Log($"Imported point cloud: {Path.GetFileName(file)}");
                                ProgressDialog.Instance.Complete();
                            });
                        }
                        else
                        {
                            throw new Exception("Failed to load point cloud data.");
                        }
                    }
                    catch (Exception ex)
                    {
                        EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                    }
                });
            }
            else if (ext == ".d3d")
            {
                OnOpenProject(file);
            }
        }

        protected override void OnKeyDown(KeyboardKeyEventArgs e)
        {
            base.OnKeyDown(e);
            UpdateModifierState(e);

            if (!ImGui.GetIO().WantCaptureKeyboard)
            {
                // Keyboard shortcuts
                switch (e.Key)
                {
                    case Keys.Q: _viewport.CurrentGizmoMode = GizmoMode.Select; break;
                    case Keys.W: _viewport.CurrentGizmoMode = GizmoMode.Translate; break;
                    case Keys.E: _viewport.CurrentGizmoMode = GizmoMode.Rotate; break;
                    case Keys.R: _viewport.CurrentGizmoMode = GizmoMode.Scale; break;
                    case Keys.P: _viewport.CurrentGizmoMode = GizmoMode.Pen; break;
                    case Keys.T: _viewport.CurrentGizmoMode = GizmoMode.Rigging; break;
                    case Keys.F: _viewport.FocusOnSelection(); break;
                    case Keys.F11: ToggleFullscreen(); break;
                    case Keys.Delete:
                        // In Pen mode, delete selected triangles
                        if (_viewport.CurrentGizmoMode == GizmoMode.Pen && _viewport.MeshEditingTool.SelectedTriangles.Count > 0)
                        {
                            _viewport.MeshEditingTool.DeleteSelectedTriangles();
                            _logBuffer += "Deleted selected triangles.\n";
                        }
                        else
                        {
                            OnDeleteSelected();
                        }
                        break;
                    case Keys.Escape:
                        // In Pen mode, clear triangle selection first
                        if (_viewport.CurrentGizmoMode == GizmoMode.Pen && _viewport.MeshEditingTool.SelectedTriangles.Count > 0)
                        {
                            _viewport.MeshEditingTool.ClearSelection();
                        }
                        else
                        {
                            _sceneGraph.ClearSelection();
                        }
                        break;
                }

                // Ctrl shortcuts
                if (e.Control)
                {
                    switch (e.Key)
                    {
                        case Keys.N: OnNewProject(); break;
                        case Keys.O: OnOpenProject(); break;
                        case Keys.S: OnSaveProject(); break;
                        case Keys.A: _sceneGraph.SelectAll(); break;
                        case Keys.D: OnDuplicateSelected(); break;
                    }
                }
            }
        }

        protected override void OnKeyUp(KeyboardKeyEventArgs e)
        {
            base.OnKeyUp(e);
            UpdateModifierState(e);
        }

        private void UpdateModifierState(KeyboardKeyEventArgs e)
        {
            _ctrlModifierDown = e.Control ||
                                e.Key == Keys.LeftControl || e.Key == Keys.RightControl ||
                                KeyboardState.IsKeyDown(Keys.LeftControl) || KeyboardState.IsKeyDown(Keys.RightControl);
            _shiftModifierDown = e.Shift ||
                                 e.Key == Keys.LeftShift || e.Key == Keys.RightShift ||
                                 KeyboardState.IsKeyDown(Keys.LeftShift) || KeyboardState.IsKeyDown(Keys.RightShift);
            _superModifierDown = e.Key == Keys.LeftSuper || e.Key == Keys.RightSuper ||
                                 KeyboardState.IsKeyDown(Keys.LeftSuper) || KeyboardState.IsKeyDown(Keys.RightSuper);
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            // Process pending actions
            while (_pendingActions.TryDequeue(out var action))
            {
                action();
            }

            _controller.Update(this, (float)e.Time);

            var s = IniSettings.Instance;
            GL.ClearColor(s.ViewportBgR, s.ViewportBgG, s.ViewportBgB, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit | ClearBufferMask.StencilBufferBit);

            if (_showSplash)
            {
                RenderSplash();
            }
            else
            {
                // Update Input
                if (!ImGui.GetIO().WantCaptureMouse && !ImGui.GetIO().WantCaptureKeyboard && !ProgressDialog.Instance.IsVisible)
                {
                    var mouseState = MouseState;
                    var keyboardState = KeyboardState;
                    _viewport.UpdateInput(mouseState, keyboardState, (float)e.Time, _clientWidth, _clientHeight);
                }

                // Render Viewport
                float vpX = _showVerticalToolbar ? _verticalToolbarWidth : 0;
                float vpY = GetTopUiHeight();
                float vpW = _clientWidth - vpX - (_showRightPanel ? _rightPanelWidth : 0);
                float vpH = _clientHeight - vpY - (_showLogPanel ? _logPanelHeight : 0);

                if (_showLeftPanel)
                {
                    vpX += _leftPanelWidth;
                    vpW -= _leftPanelWidth;
                }

                int logicalVpX = Math.Max(0, (int)MathF.Round(vpX));
                int logicalVpY = Math.Max(0, (int)MathF.Round(vpY));
                int logicalVpW = Math.Max(1, (int)MathF.Round(vpW));
                int logicalVpH = Math.Max(1, (int)MathF.Round(vpH));

                int fbVpX = LogicalToFramebufferX(vpX);
                int fbVpY = LogicalToFramebufferY(vpY);
                int fbVpRight = LogicalToFramebufferX(vpX + vpW);
                int fbVpBottom = LogicalToFramebufferY(vpY + vpH);
                int fbVpW = Math.Max(1, fbVpRight - fbVpX);
                int fbVpH = Math.Max(1, fbVpBottom - fbVpY);

                _viewport.Render(
                    logicalVpX, logicalVpY, logicalVpW, logicalVpH,
                    fbVpX, fbVpY, fbVpW, fbVpH,
                    _framebufferWidth, _framebufferHeight);
                CheckError("After Viewport");

                // Render UI
                RenderUI();
            }

            CheckError("Before ImGui");
            _controller.Render();
            CheckError("After ImGui");

            // Drain any remaining OpenGL errors silently before buffer swap
            while (GL.GetError() != OpenTK.Graphics.OpenGL.ErrorCode.NoError) { }

            SwapBuffers();
        }

        private float GetTopUiHeight()
        {
            float height = 20.0f; // Main menu bar
            if (_showTopToolbar)
                height += _toolbarHeight;
            if (_showMeshEditorToolbar)
                height += _auxToolbarHeight;
            if (_showPointCloudToolbar)
                height += _auxToolbarHeight;
            if (_showGeoreferenceToolbar)
                height += _auxToolbarHeight;
            return height;
        }

        private void RefreshDisplayMetrics()
        {
            _clientWidth = Math.Max(1, ClientSize.X);
            _clientHeight = Math.Max(1, ClientSize.Y);
            _framebufferWidth = Math.Max(1, FramebufferSize.X);
            _framebufferHeight = Math.Max(1, FramebufferSize.Y);
            UpdateFramebufferScale();
        }

        private void UpdateFramebufferScale()
        {
            _framebufferScaleX = _clientWidth > 0 ? (float)_framebufferWidth / _clientWidth : 1.0f;
            _framebufferScaleY = _clientHeight > 0 ? (float)_framebufferHeight / _clientHeight : 1.0f;

            if (_framebufferScaleX <= 0.0f) _framebufferScaleX = 1.0f;
            if (_framebufferScaleY <= 0.0f) _framebufferScaleY = 1.0f;
        }

        private int LogicalToFramebufferX(float logicalX)
        {
            return Math.Clamp((int)MathF.Round(logicalX * _framebufferScaleX), 0, _framebufferWidth);
        }

        private int LogicalToFramebufferY(float logicalY)
        {
            return Math.Clamp((int)MathF.Round(logicalY * _framebufferScaleY), 0, _framebufferHeight);
        }

        // Error tracking to avoid spamming console
        private static DateTime _lastErrorLog = DateTime.MinValue;
        private static int _errorCount = 0;

        private void CheckError(string stage)
        {
            // Drain all errors from the queue
            OpenTK.Graphics.OpenGL.ErrorCode err;
            while ((err = GL.GetError()) != OpenTK.Graphics.OpenGL.ErrorCode.NoError)
            {
                // Skip InvalidFramebufferOperation and InvalidOperation caused by legacy/modern GL switching
                if (err == OpenTK.Graphics.OpenGL.ErrorCode.InvalidFramebufferOperation ||
                    err == OpenTK.Graphics.OpenGL.ErrorCode.InvalidOperation)
                    continue;

                // Rate limit error logging
                _errorCount++;
                if ((DateTime.Now - _lastErrorLog).TotalSeconds > 5)
                {
                    Console.WriteLine($"OpenGL Error at MainWindow {stage}: {err} (count: {_errorCount})");
                    _lastErrorLog = DateTime.Now;
                    _errorCount = 0;
                }
            }
        }

        protected override void OnUnload()
        {
            base.OnUnload();

            try
            {
                var settings = IniSettings.Instance;
                settings.ShowTopToolbar = _showTopToolbar;
                settings.ShowVerticalToolbar = _showVerticalToolbar;
                settings.ShowMeshEditorToolbar = _showMeshEditorToolbar;
                settings.ShowPointCloudToolbar = _showPointCloudToolbar;
                settings.ShowGeoreferenceToolbar = _showGeoreferenceToolbar;
                settings.Save();
            }
            catch
            {
                // Ignore settings write errors during shutdown.
            }

            // Clean up thumbnails
            foreach (var thumb in _imageThumbnails.Values)
            {
                TextureLoader.DeleteTexture(thumb);
            }
            _imageThumbnails.Clear();
            foreach (var thumb in _imageDepthThumbnails.Values)
            {
                TextureLoader.DeleteTexture(thumb);
            }
            _imageDepthThumbnails.Clear();

            if (_previewTexture > 0)
                TextureLoader.DeleteTexture(_previewTexture);
            if (_geoPreviewTexture > 0)
                TextureLoader.DeleteTexture(_geoPreviewTexture);

            _iconFactory?.Dispose();
        }

        #region Error Display
        #endregion

        #region Splash Screen

        private void RenderSplash()
        {
            ImGui.SetNextWindowPos(System.Numerics.Vector2.Zero);
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.X, ClientSize.Y));
            ImGui.Begin("Splash", ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoSavedSettings);

            var drawList = ImGui.GetWindowDrawList();
            var center = new System.Numerics.Vector2(ClientSize.X * 0.5f, ClientSize.Y * 0.5f);

            drawList.AddRectFilledMultiColor(
                System.Numerics.Vector2.Zero,
                new System.Numerics.Vector2(ClientSize.X, ClientSize.Y),
                0xFF1A1A2E, 0xFF1A1A2E, 0xFF0F0F1A, 0xFF0F0F1A);

            if (_logoTexture != -1)
            {
                float size = 200;
                ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - size * 0.5f, center.Y - size * 0.5f - 80));
                ImGui.Image((IntPtr)_logoTexture, new System.Numerics.Vector2(size, size));
            }

            ImGui.PushFont(ImGui.GetIO().Fonts.Fonts[0]);
            string text = "Deep3DStudio";
            var textSize = ImGui.CalcTextSize(text);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - textSize.X * 0.5f, center.Y + 50));
            ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(0.9f, 0.9f, 0.95f, 1.0f));
            ImGui.Text(text);
            ImGui.PopStyleColor();
            ImGui.PopFont();

            string subtitle = "Neural 3D Reconstruction Studio";
            var subSize = ImGui.CalcTextSize(subtitle);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - subSize.X * 0.5f, center.Y + 75));
            ImGui.TextDisabled(subtitle);

            string authorLine1 = "Matteo Mangiagalli - m.mangiagalli@campus.uniurb.it";
            string authorLine2 = "Università degli Studi di Urbino - Carlo Bo";
            string authorLine3 = "2026";
            var authorSize1 = ImGui.CalcTextSize(authorLine1);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - authorSize1.X * 0.5f, center.Y + 100));
            ImGui.TextDisabled(authorLine1);
            var authorSize2 = ImGui.CalcTextSize(authorLine2);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - authorSize2.X * 0.5f, center.Y + 120));
            ImGui.TextDisabled(authorLine2);
            var authorSize3 = ImGui.CalcTextSize(authorLine3);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - authorSize3.X * 0.5f, center.Y + 140));
            ImGui.TextDisabled(authorLine3);

            string status = _pythonReady ? "Ready" : "Initializing AI Engine...";
            var statusSize = ImGui.CalcTextSize(status);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - statusSize.X * 0.5f, center.Y + 175));

            if (!_pythonReady)
            {
                float time = (float)(DateTime.Now.TimeOfDay.TotalSeconds % 1.0);
                ImGui.TextColored(new System.Numerics.Vector4(0.4f, 0.6f, 0.9f, 0.5f + 0.5f * (float)Math.Sin(time * Math.PI * 2)), status);
            }
            else
            {
                ImGui.TextColored(new System.Numerics.Vector4(0.4f, 0.9f, 0.4f, 1.0f), status);
            }

            string version = GetAppVersionText();
            var verSize = ImGui.CalcTextSize(version);
            ImGui.SetCursorPos(new System.Numerics.Vector2(ClientSize.X - verSize.X - 10, ClientSize.Y - 25));
            ImGui.TextDisabled(version);

            ImGui.End();
        }

        #endregion

        #region Main UI

        private void RenderUI()
        {
            // Progress Dialog (renders on top)
            ProgressDialog.Instance.Draw();

            // Handle popup requests
            ProcessPendingPopupRequest();

            // Unsaved changes prompt
            if (_showUnsavedChangesPrompt)
            {
                ImGui.OpenPopup("Unsaved Changes");

                // Center the modal
                var io = ImGui.GetIO();
                ImGui.SetNextWindowPos(new System.Numerics.Vector2(io.DisplaySize.X * 0.5f, io.DisplaySize.Y * 0.5f), ImGuiCond.Always, new System.Numerics.Vector2(0.5f, 0.5f));

                if (ImGui.BeginPopupModal("Unsaved Changes", ref _showUnsavedChangesPrompt, ImGuiWindowFlags.AlwaysAutoResize))
                {
                    ImGui.Text("You have unsaved changes. Do you want to save before continuing?");
                    ImGui.Separator();

                    if (ImGui.Button("Save", new System.Numerics.Vector2(120, 0)))
                    {
                        ImGui.CloseCurrentPopup();
                        _showUnsavedChangesPrompt = false;
                        if (_pendingUnsavedAction == PendingUnsavedAction.Exit)
                        {
                            ClearPendingUnsavedAction();
                            OnSaveProject(true);
                        }
                        else
                        {
                            _executePendingActionAfterSave = true;
                            OnSaveProject();
                        }
                    }
                    ImGui.SameLine();

                    if (ImGui.Button("Discard", new System.Numerics.Vector2(120, 0)))
                    {
                        ImGui.CloseCurrentPopup();
                        _showUnsavedChangesPrompt = false;
                        _isDirty = false;
                        ExecutePendingUnsavedActionAndClear();
                    }
                    ImGui.SetItemDefaultFocus();
                    ImGui.SameLine();
                    if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0)))
                    {
                        ImGui.CloseCurrentPopup();
                        _showUnsavedChangesPrompt = false;
                        _executePendingActionAfterSave = false;
                        ClearPendingUnsavedAction();
                    }

                    ImGui.EndPopup();
                }
            }

            // Error Dialog (renders on top)
            RenderErrorDialog();

            // Main Menu
            RenderMainMenu();

            float toolbarY = 20.0f;
            if (_showTopToolbar)
            {
                RenderTopToolbar(toolbarY);
                toolbarY += _toolbarHeight;
            }

            if (_showMeshEditorToolbar)
            {
                RenderMeshEditorToolbar(toolbarY);
                toolbarY += _auxToolbarHeight;
            }

            if (_showPointCloudToolbar)
            {
                RenderPointCloudToolbar(toolbarY);
                toolbarY += _auxToolbarHeight;
            }

            if (_showGeoreferenceToolbar)
            {
                RenderGeoreferenceToolbar(toolbarY);
                toolbarY += _auxToolbarHeight;
            }

            // Vertical Toolbar
            if (_showVerticalToolbar)
                RenderVerticalToolbar();

            // Left Panel
            if (_showLeftPanel)
                RenderLeftPanel();

            // Right Panel
            if (_showRightPanel)
                RenderRightPanel();

            // Log Panel
            if (_showLogPanel)
                RenderLogPanel();

            // Info Overlay
            if (IniSettings.Instance.ShowInfoOverlay)
            {
                RenderInfoOverlay();
            }

            // A second pass allows right-click popup requests raised during toolbar rendering
            // to be opened in the same frame (avoids requiring an extra click).
            ProcessPendingPopupRequest();

            // Dialogs
            if (_showSettings) DrawSettingsWindow();
            if (_showAbout) DrawAboutWindow();
            if (_showImagePreview) DrawImagePreviewWindow();
            if (_showGeoreferenceWindow) DrawGeoreferenceWindow();
            if (_showDecimateDialog) DrawDecimateDialog();
            if (_showSmoothDialog) DrawSmoothDialog();
            if (_showOptimizeDialog) DrawOptimizeDialog();
            if (_showMergeDialog) DrawMergeDialog();
            if (_showAlignDialog) DrawAlignDialog();
            if (_showMeshToolbarMergeDialog) DrawMeshToolbarMergeDialog();
            if (_showPointCloudToolbarMergeDialog) DrawPointCloudToolbarMergeDialog();
            if (_showCleanupDialog) DrawCleanupDialog();
            if (_showBakeDialog) DrawBakeDialog();
            if (_showTransformDialog) DrawTransformDialog();
            if (_showPrimitiveDialog) DrawPrimitiveDialog();
            if (_showRunOptionsDialog) DrawRunOptionsDialog();
            if (_showMeshingOptionsDialog) DrawMeshingOptionsDialog();
            if (_showRefinementOptionsDialog) DrawRefinementOptionsDialog();
            if (_showPcVoxelDialog) DrawPointCloudVoxelDialog();
            if (_showPcOutliersDialog) DrawPointCloudOutliersDialog();
            if (_showPcDuplicatesDialog) DrawPointCloudDuplicatesDialog();
            if (_showPcSkyBlueDialog) DrawPointCloudSkyBlueDialog();
            if (_showPcNormalsDialog) DrawPointCloudNormalsDialog();
            if (_showPcPassDialog) DrawPointCloudPassDialog();
            if (_showPcRadiusDialog) DrawPointCloudRadiusDialog();
            if (_showPcDenseDialog) DrawPointCloudDenseDialog();
            if (_showPenMoveDialog) DrawPenMoveDialog();
            if (_showPenExtrudeDialog) DrawPenExtrudeDialog();
            if (_showPenInsetDialog) DrawPenInsetDialog();
            _diagnosticsWindow.Draw();
        }

        private void RenderMainMenu()
        {
            if (ImGui.BeginMainMenuBar())
            {
                // File Menu
                if (ImGui.BeginMenu("File"))
                {
                    if (ImGui.MenuItem("New Project", "Ctrl+N")) OnNewProject();
                    if (ImGui.MenuItem("Open Project...", "Ctrl+O")) OnOpenProject();
                    if (ImGui.MenuItem("Save Project", "Ctrl+S")) OnSaveProject();
                    if (ImGui.MenuItem("Save Project As...")) OnSaveProjectAs();
                    ImGui.Separator();
                    if (ImGui.MenuItem("Open Images...")) OnAddImages();
                    if (ImGui.MenuItem("Import Mesh...")) OnImportMesh();
                    if (ImGui.MenuItem("Import Point Cloud...")) OnImportPointCloud();
                    ImGui.Separator();
                    if (ImGui.MenuItem("Export Mesh...")) OnExportMesh();
                    if (ImGui.MenuItem("Export Point Cloud...")) OnExportPointCloud();
                    if (ImGui.MenuItem("Export DEM...")) OnExportDemImGui();
                    ImGui.Separator();
                    if (ImGui.MenuItem("Settings...")) _showSettings = true;
                    ImGui.Separator();
                    if (ImGui.MenuItem("Quit")) Close();
                    ImGui.EndMenu();
                }

                // Edit Menu
                if (ImGui.BeginMenu("Edit"))
                {
                    if (ImGui.MenuItem("Select All", "Ctrl+A")) _sceneGraph.SelectAll();
                    if (ImGui.MenuItem("Deselect All")) _sceneGraph.ClearSelection();
                    ImGui.Separator();
                    if (ImGui.MenuItem("Delete", "Delete")) OnDeleteSelected();
                    if (ImGui.MenuItem("Duplicate", "Ctrl+D")) OnDuplicateSelected();
                    ImGui.Separator();

                    if (ImGui.BeginMenu("Transform"))
                    {
                        if (ImGui.MenuItem("Move (W)", "", _viewport.CurrentGizmoMode == GizmoMode.Translate))
                            _viewport.CurrentGizmoMode = GizmoMode.Translate;
                        if (ImGui.MenuItem("Rotate (E)", "", _viewport.CurrentGizmoMode == GizmoMode.Rotate))
                            _viewport.CurrentGizmoMode = GizmoMode.Rotate;
                        if (ImGui.MenuItem("Scale (R)", "", _viewport.CurrentGizmoMode == GizmoMode.Scale))
                            _viewport.CurrentGizmoMode = GizmoMode.Scale;
                        ImGui.Separator();
                        if (ImGui.MenuItem("Reset Transform")) OnResetTransform();
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Mesh Operations"))
                    {
                        if (ImGui.MenuItem("Decimate (50%)")) OnDecimate();
                        if (ImGui.MenuItem("Smooth")) OnSmooth();
                        if (ImGui.MenuItem("Optimize")) OnOptimize();
                        if (ImGui.MenuItem("Split by Connectivity")) OnSplit();
                        if (ImGui.MenuItem("Flip Normals")) OnFlipNormals();
                        ImGui.Separator();
                        if (ImGui.MenuItem("Merge Selected")) OnMerge();
                        if (ImGui.MenuItem("Align (ICP)")) OnAlign();
                        ImGui.Separator();
                        if (ImGui.MenuItem("Cleanup Mesh...")) OnCleanup();
                        if (ImGui.MenuItem("Bake Textures...")) OnBakeTextures();
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Create Primitive"))
                    {
                        if (ImGui.MenuItem("Plane")) OnCreatePrimitive(MeshPrimitiveType.Plane);
                        if (ImGui.MenuItem("Cube")) OnCreatePrimitive(MeshPrimitiveType.Cube);
                        if (ImGui.MenuItem("UV Sphere")) OnCreatePrimitive(MeshPrimitiveType.UVSphere);
                        if (ImGui.MenuItem("Cylinder")) OnCreatePrimitive(MeshPrimitiveType.Cylinder);
                        if (ImGui.MenuItem("Cone")) OnCreatePrimitive(MeshPrimitiveType.Cone);
                        if (ImGui.MenuItem("Torus")) OnCreatePrimitive(MeshPrimitiveType.Torus);
                        if (ImGui.MenuItem("Circle")) OnCreatePrimitive(MeshPrimitiveType.Circle);
                        if (ImGui.MenuItem("Polygon")) OnCreatePrimitive(MeshPrimitiveType.Polygon);
                        if (ImGui.MenuItem("Grid")) OnCreatePrimitive(MeshPrimitiveType.Grid);
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Triangle Editing (Pen)"))
                    {
                        bool hasSelection = _viewport.MeshEditingTool.SelectedTriangles.Count > 0;
                        bool canBridge = _viewport.MeshEditingTool.SelectedTriangles.Count == 2;

                        if (ImGui.MenuItem("Move Vertices", "", false, hasSelection))
                            ApplyPenMoveVertices();
                        if (ImGui.MenuItem("Extrude", "", false, hasSelection))
                            ApplyPenExtrude();
                        if (ImGui.MenuItem("Inset", "", false, hasSelection))
                            ApplyPenInset();
                        if (ImGui.MenuItem("Bridge 2 Triangles", "", false, canBridge))
                            ApplyPenBridge();
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Point Cloud Filters"))
                    {
                        bool hasPc = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().Any();
                        if (ImGui.MenuItem("Voxel Downsample", "", false, hasPc)) ApplyPointCloudVoxel();
                        if (ImGui.MenuItem("Remove Outliers", "", false, hasPc)) ApplyPointCloudOutliers();
                        if (ImGui.MenuItem("Remove Duplicates", "", false, hasPc)) ApplyPointCloudDuplicates();
                        if (ImGui.MenuItem("Remove Sky/Blue", "", false, hasPc)) ApplyPointCloudSkyBlue();
                        if (ImGui.MenuItem("Estimate Normals", "", false, hasPc)) ApplyPointCloudNormals();
                        if (ImGui.MenuItem("Pass-through Axis", "", false, hasPc)) ApplyPointCloudPassThrough();
                        if (ImGui.MenuItem("Radius Crop", "", false, hasPc)) ApplyPointCloudRadiusCrop();
                        if (ImGui.MenuItem("Point Cloud -> Dense Cloud", "", false, hasPc)) ApplyPointCloudDenseCloud();
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Georeferencing"))
                    {
                        if (ImGui.MenuItem("GCP Editor")) _showGeoreferenceWindow = true;
                        if (ImGui.MenuItem("Solve from GCP")) SolveGeoFromRuntime();
                        if (ImGui.MenuItem("Resolve Pending GCP")) ResolvePendingGeoGcpsFromRuntime();
                        if (ImGui.MenuItem("Export Georeferenced Selection")) OnExportGeoreferencedSelectionImGui();
                        if (ImGui.MenuItem("Generate DEM...")) OnExportDemImGui();
                        ImGui.EndMenu();
                    }

                    ImGui.EndMenu();
                }

                // View Menu
                if (ImGui.BeginMenu("View"))
                {
                    var s = IniSettings.Instance;
                    bool sm = s.ShowMesh; if (ImGui.MenuItem("Show Mesh", "", sm)) s.ShowMesh = !sm;
                    bool sp = s.ShowPointCloud; if (ImGui.MenuItem("Show Point Cloud", "", sp)) s.ShowPointCloud = !sp;
                    bool st = s.ShowTexture; if (ImGui.MenuItem("Show Texture", "", st)) s.ShowTexture = !st;
                    bool sw = s.ShowWireframe; if (ImGui.MenuItem("Show Wireframe", "", sw)) s.ShowWireframe = !sw;
                    ImGui.Separator();
                    bool sc = s.ShowCameras; if (ImGui.MenuItem("Show Cameras", "", sc)) s.ShowCameras = !sc;
                    bool sg = s.ShowGrid; if (ImGui.MenuItem("Show Grid", "", sg)) s.ShowGrid = !sg;
                    bool sa = s.ShowAxes; if (ImGui.MenuItem("Show Axes", "", sa)) s.ShowAxes = !sa;
                    ImGui.Separator();
                    if (ImGui.MenuItem("Point Colors: RGB", "", s.PointCloudColor == PointCloudColorMode.RGB))
                        s.PointCloudColor = PointCloudColorMode.RGB;
                    if (ImGui.MenuItem("Point Colors: Depth", "", s.PointCloudColor == PointCloudColorMode.DistanceMap))
                        s.PointCloudColor = PointCloudColorMode.DistanceMap;
                    if (ImGui.MenuItem("Point Colors: Confidence", "", s.PointCloudColor == PointCloudColorMode.Confidence))
                        s.PointCloudColor = PointCloudColorMode.Confidence;
                    ImGui.Separator();
                    if (ImGui.MenuItem("Focus on Selection", "F")) _viewport.FocusOnSelection();
                    if (ImGui.MenuItem("Reset Camera")) _viewport.ResetCamera();
                    ImGui.EndMenu();
                }

                // AI Models Menu
                if (ImGui.BeginMenu("AI Models"))
                {
                    if (ImGui.BeginMenu("Image to 3D"))
                    {
                        if (ImGui.MenuItem("TripoSR (Fast)")) RunAIModel("TripoSR");
                        if (ImGui.MenuItem("LGM (Single Image)")) RunAIModel("LGM");
                        if (ImGui.MenuItem("Wonder3D (Single-Image Multi-View)")) RunAIModel("Wonder3D");
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Mesh Processing"))
                    {
                        if (ImGui.MenuItem("DeepMeshPrior Optimization")) RunAIModel("DeepMeshPrior");
                        if (ImGui.MenuItem("TripoSF Refinement")) RunAIModel("TripoSF");
                        if (ImGui.MenuItem("GaussianSDF Refinement")) RunAIModel("GaussianSDF");
                        ImGui.EndMenu();
                    }

                    if (ImGui.BeginMenu("Rigging"))
                    {
                        if (ImGui.MenuItem("UniRig Auto Rig")) RunAIModel("UniRig");
                        ImGui.EndMenu();
                    }

                    if (ImGui.MenuItem("Point Cloud -> Dense Cloud"))
                    {
                        ApplyPointCloudDenseCloud();
                    }

                    ImGui.Separator();
                    if (ImGui.MenuItem("AI Model Settings...")) _showSettings = true;
                    ImGui.EndMenu();
                }

                // Window Menu
                if (ImGui.BeginMenu("Window"))
                {
                    if (ImGui.MenuItem("Top Toolbar", "", _showTopToolbar)) _showTopToolbar = !_showTopToolbar;
                    if (ImGui.MenuItem("Mesh Editor Toolbar", "", _showMeshEditorToolbar)) _showMeshEditorToolbar = !_showMeshEditorToolbar;
                    if (ImGui.MenuItem("Point Cloud Toolbar", "", _showPointCloudToolbar)) _showPointCloudToolbar = !_showPointCloudToolbar;
                    if (ImGui.MenuItem("Georeference Toolbar", "", _showGeoreferenceToolbar)) _showGeoreferenceToolbar = !_showGeoreferenceToolbar;
                    if (ImGui.MenuItem("Left Panel", "", _showLeftPanel)) _showLeftPanel = !_showLeftPanel;
                    if (ImGui.MenuItem("Right Panel", "", _showRightPanel)) _showRightPanel = !_showRightPanel;
                    if (ImGui.MenuItem("Log Panel", "", _showLogPanel)) _showLogPanel = !_showLogPanel;
                    if (ImGui.MenuItem("Vertical Toolbar", "", _showVerticalToolbar)) _showVerticalToolbar = !_showVerticalToolbar;
                    ImGui.Separator();
                    if (ImGui.MenuItem("Full Viewport Mode"))
                    {
                        _showTopToolbar = false;
                        _showMeshEditorToolbar = false;
                        _showPointCloudToolbar = false;
                        _showGeoreferenceToolbar = false;
                        _showLeftPanel = false;
                        _showRightPanel = false;
                        _showLogPanel = false;
                        _showVerticalToolbar = false;
                    }
                    if (ImGui.MenuItem("Restore All Panels"))
                    {
                        _showTopToolbar = true;
                        _showMeshEditorToolbar = true;
                        _showPointCloudToolbar = true;
                        _showGeoreferenceToolbar = true;
                        _showLeftPanel = true;
                        _showRightPanel = true;
                        _showLogPanel = true;
                        _showVerticalToolbar = true;
                    }
                    ImGui.EndMenu();
                }

                // Help Menu
                if (ImGui.BeginMenu("Help"))
                {
                    if (ImGui.MenuItem("AI Diagnostics")) _diagnosticsWindow.Visible = true;
                    ImGui.Separator();
                    if (ImGui.MenuItem("About")) _showAbout = true;
                    ImGui.EndMenu();
                }

                ImGui.EndMainMenuBar();
            }
        }

        private void RenderTopToolbar(float yPos)
        {
            bool shown = BeginStyledToolbarWindow("##Toolbar", new System.Numerics.Vector2(0, yPos), new System.Numerics.Vector2(ClientSize.X, _toolbarHeight));
            if (shown)
            {
                var size = ToolbarIconSize;

                // Gizmo Modes
                DrawToolbarButton("##Select", IconType.Select, _viewport.CurrentGizmoMode == GizmoMode.Select,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Select, "Select (Q)", size);
                ImGui.SameLine();
                DrawToolbarButton("##Move", IconType.Move, _viewport.CurrentGizmoMode == GizmoMode.Translate,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Translate, "Move (W)", size);
                ImGui.SameLine();
                DrawToolbarButton("##Rotate", IconType.Rotate, _viewport.CurrentGizmoMode == GizmoMode.Rotate,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Rotate, "Rotate (E)", size);
                ImGui.SameLine();
                DrawToolbarButton("##Scale", IconType.Scale, _viewport.CurrentGizmoMode == GizmoMode.Scale,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Scale, "Scale (R)", size);
                ImGui.SameLine();
                DrawToolbarButton("##Pen", IconType.Pen, _viewport.CurrentGizmoMode == GizmoMode.Pen,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Pen, "Pen / Triangle Edit (P)", size);
                ImGui.SameLine();
                DrawToolbarButton("##Rigging", IconType.Skeleton, _viewport.CurrentGizmoMode == GizmoMode.Rigging,
                    () => _viewport.CurrentGizmoMode = GizmoMode.Rigging, "Rigging (T)", size);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();

                // Workflow Selection with recommendation for MASt3R/MUSt3R
                ImGui.Text("Workflow:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(190);

                // Get dynamic workflow names (first option shows current engine from settings)
                var workflowNames = GetWorkflowNames();

                // Show recommendation hint if video loaded but MUSt3R not selected
                string workflowHint = "";
                bool isMust3rSelected = IniSettings.Instance.ReconstructionMethod == ReconstructionMethod.Must3r;
                if (_hasVideoInput && !isMust3rSelected)
                {
                    workflowHint = " (MUSt3R recommended for video - change in Settings)";
                    ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(0.6f, 0.8f, 1.0f, 1.0f));
                }

                ImGui.Combo("##Workflow", ref _selectedWorkflow, workflowNames, workflowNames.Length);
                if (!string.IsNullOrEmpty(workflowHint))
                {
                    ImGui.PopStyleColor();
                    if (ImGui.IsItemHovered())
                    {
                        ImGui.SetTooltip(workflowHint.Trim());
                    }
                }
                ImGui.SameLine();

                // Video input button - show when using Multi-View workflow with MUSt3R
                if (_selectedWorkflow == 0 && isMust3rSelected)
                {
                    if (ImGui.ImageButton("##VideoInput", _iconFactory.GetIcon(IconType.Video), ToolbarIconSize))
                    {
                        LoadVideoFile();
                    }
                    if (ImGui.IsItemHovered())
                    {
                        string videoTip = _hasVideoInput ? $"Video: {Path.GetFileName(_videoFilePath)}" : "Load Video for MUSt3R";
                        ImGui.SetTooltip(videoTip);
                    }
                    ImGui.SameLine();
                }

                ImGui.Text("Quality:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(100);
                ImGui.Combo("##Quality", ref _selectedQuality, _qualities, _qualities.Length);
                ImGui.SameLine();

                // Auto Workflow Toggle
                DrawToggleBtn("##AutoWF", IconType.Link, _autoWorkflowEnabled, v => _autoWorkflowEnabled = v,
                    _autoWorkflowEnabled ? "Auto Workflow: ON (Play runs full pipeline)" : "Auto Workflow: OFF (Manual step-by-step)", size);
                ImGui.SameLine();

                // Run Button - behavior depends on _autoWorkflowEnabled
                DrawToolbarButton("##Run", IconType.Run, false, () => {
                    if (_autoWorkflowEnabled)
                        RunReconstruction(); // Run full workflow
                    else
                        RunSingleStep(GetReconstructionStep()); // Run the selected engine's reconstruction step
                }, _autoWorkflowEnabled ? "Run Full Workflow" : "Run Selected Step", size,
                    OpenRunOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##Points", IconType.Cloud, false, () => RunSingleStep(GetReconstructionStep()), $"Generate Point Cloud ({GetCurrentEngineName()})", size,
                    OpenRunOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##Mesh", IconType.Mesh, false, RunMeshFromSelectedPointClouds, "Generate Mesh from Points", size,
                    OpenMeshingOptionsDialog);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();

                // Visibility Toggles
                var s = IniSettings.Instance;
                DrawToggleBtn("##TglMesh", IconType.Mesh, s.ShowMesh, v => s.ShowMesh = v, "Show/Hide Mesh", size);
                ImGui.SameLine();
                DrawToggleBtn("##TglCloud", IconType.Cloud, s.ShowPointCloud, v => s.ShowPointCloud = v, "Show/Hide Point Cloud", size);
                ImGui.SameLine();
                DrawToggleBtn("##TglTex", IconType.Texture, s.ShowTexture, v => s.ShowTexture = v, "Show/Hide Texture", size);
                ImGui.SameLine();
                DrawToggleBtn("##TglWire", IconType.Wireframe, s.ShowWireframe, v => s.ShowWireframe = v, "Show/Hide Wireframe", size);
                ImGui.SameLine();
                DrawToggleBtn("##TglCam", IconType.Camera, s.ShowCameras, v => s.ShowCameras = v, "Show/Hide Cameras", size);
                ImGui.SameLine();
                DrawToggleBtn("##TglGrid", IconType.Grid, s.ShowGrid, v => s.ShowGrid = v, "Show/Hide Grid", size);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();

                DrawToolbarButton("##PcRgb", IconType.Rgb, s.PointCloudColor == PointCloudColorMode.RGB,
                    () => s.PointCloudColor = PointCloudColorMode.RGB, "Point colors: RGB", size);
                ImGui.SameLine();
                DrawToolbarButton("##PcDepth", IconType.DepthMap, s.PointCloudColor == PointCloudColorMode.DistanceMap,
                    () => s.PointCloudColor = PointCloudColorMode.DistanceMap, "Point colors: Depth", size);
                ImGui.SameLine();
                DrawToolbarButton("##PcConf", IconType.Confidence, s.PointCloudColor == PointCloudColorMode.Confidence,
                    () => s.PointCloudColor = PointCloudColorMode.Confidence, "Point colors: Confidence", size);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();

                // Fullscreen toggle
                bool isFullscreen = WindowState == OpenTK.Windowing.Common.WindowState.Fullscreen;
                DrawToolbarButton("##Fullscreen", IconType.Fullscreen, isFullscreen, ToggleFullscreen,
                    isFullscreen ? "Exit Fullscreen (F11)" : "Fullscreen (F11)", size);
            }
            EndStyledToolbarWindow();
        }

        private void RenderMeshEditorToolbar(float yPos)
        {
            bool shown = BeginStyledToolbarWindow("##MeshEditorToolbar", new System.Numerics.Vector2(0, yPos), new System.Numerics.Vector2(ClientSize.X, _auxToolbarHeight));
            if (shown)
            {
                var size = ToolbarIconSize;

                ImGui.TextDisabled("Create:");
                ImGui.SameLine();
                DrawToolbarButton("##PrimPlane", IconType.Plane, false, () => OnCreatePrimitive(MeshPrimitiveType.Plane), "Create Plane", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Plane));
                ImGui.SameLine();
                DrawToolbarButton("##PrimCube", IconType.Cube, false, () => OnCreatePrimitive(MeshPrimitiveType.Cube), "Create Cube", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Cube));
                ImGui.SameLine();
                DrawToolbarButton("##PrimSphere", IconType.Sphere, false, () => OnCreatePrimitive(MeshPrimitiveType.UVSphere), "Create UV Sphere", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.UVSphere));
                ImGui.SameLine();
                DrawToolbarButton("##PrimCyl", IconType.Cylinder, false, () => OnCreatePrimitive(MeshPrimitiveType.Cylinder), "Create Cylinder", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Cylinder));
                ImGui.SameLine();
                DrawToolbarButton("##PrimCone", IconType.Cone, false, () => OnCreatePrimitive(MeshPrimitiveType.Cone), "Create Cone", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Cone));
                ImGui.SameLine();
                DrawToolbarButton("##PrimTorus", IconType.Torus, false, () => OnCreatePrimitive(MeshPrimitiveType.Torus), "Create Torus", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Torus));
                ImGui.SameLine();
                DrawToolbarButton("##PrimCircle", IconType.Circle, false, () => OnCreatePrimitive(MeshPrimitiveType.Circle), "Create Circle", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Circle));
                ImGui.SameLine();
                DrawToolbarButton("##PrimPoly", IconType.Polygon, false, () => OnCreatePrimitive(MeshPrimitiveType.Polygon), "Create Polygon", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Polygon));
                ImGui.SameLine();
                DrawToolbarButton("##PrimGrid", IconType.GridMesh, false, () => OnCreatePrimitive(MeshPrimitiveType.Grid), "Create Grid", size,
                    () => OpenPrimitiveOptionsDialog(MeshPrimitiveType.Grid));

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();

                ImGui.TextDisabled("Edit:");
                ImGui.SameLine();
                DrawToolbarButton("##MeshDec", IconType.Decimate, false, ApplyDecimatePreset, "Decimate", size,
                    OnDecimate);
                ImGui.SameLine();
                DrawToolbarButton("##MeshSm", IconType.Smooth, false, ApplySmoothPreset, "Smooth", size,
                    OnSmooth);
                ImGui.SameLine();
                DrawToolbarButton("##MeshOpt", IconType.Optimize, false, ApplyOptimizePreset, "Optimize", size,
                    OnOptimize);
                ImGui.SameLine();
                DrawToolbarButton("##PenMove", IconType.VertexMove, false, ApplyPenMoveVertices, "Move selected vertices", size,
                    OpenPenMoveOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PenExtrude", IconType.Extrude, false, ApplyPenExtrude, "Extrude selected triangles", size,
                    OpenPenExtrudeOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PenInset", IconType.Inset, false, ApplyPenInset, "Inset selected triangles", size,
                    OpenPenInsetOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PenBridge", IconType.Bridge, false, ApplyPenBridge, "Bridge two selected triangles", size);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();
                DrawToolbarButton("##MeshMergeToolbar", IconType.MergeMeshes, false, OpenMeshToolbarMergeDialog,
                    "Merge selected meshes (Merge / Align+Merge)", size);
            }
            EndStyledToolbarWindow();
        }

        private void RenderPointCloudToolbar(float yPos)
        {
            bool shown = BeginStyledToolbarWindow("##PointCloudToolbar", new System.Numerics.Vector2(0, yPos), new System.Numerics.Vector2(ClientSize.X, _auxToolbarHeight));
            if (shown)
            {
                var size = ToolbarIconSize;
                ImGui.TextDisabled("Point Cloud:");
                ImGui.SameLine();

                DrawToolbarButton("##PcVoxel", IconType.VoxelFilter, false, () => ApplyPointCloudVoxel(), "Voxel Downsample", size,
                    OpenPointCloudVoxelOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcOutliers", IconType.OutlierFilter, false, () => ApplyPointCloudOutliers(), "Remove Outliers", size,
                    OpenPointCloudOutlierOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcDup", IconType.DuplicateFilter, false, () => ApplyPointCloudDuplicates(), "Remove Duplicates", size,
                    OpenPointCloudDuplicateOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcSky", IconType.OutlierFilter, false, () => ApplyPointCloudSkyBlue(), "Remove Sky/Blue", size,
                    OpenPointCloudSkyBlueOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcNormals", IconType.Normals, false, () => ApplyPointCloudNormals(), "Estimate Normals", size,
                    OpenPointCloudNormalOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcAxis", IconType.AxisFilter, false, () => ApplyPointCloudPassThrough(), "Pass-through Axis", size,
                    OpenPointCloudPassOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcRadius", IconType.RadiusCrop, false, () => ApplyPointCloudRadiusCrop(), "Radius Crop", size,
                    OpenPointCloudRadiusOptionsDialog);
                ImGui.SameLine();
                DrawToolbarButton("##PcDense", IconType.DenseCloud, false, () => ApplyPointCloudDenseCloud(), "Point Cloud to Dense Cloud", size,
                    OpenPointCloudDenseOptionsDialog);

                ImGui.SameLine();
                ImGui.Text("|");
                ImGui.SameLine();
                DrawToolbarButton("##PcMergeToolbar", IconType.MergePointClouds, false, OpenPointCloudToolbarMergeDialog,
                    "Merge selected point clouds (Merge / Align+Merge)", size);
            }
            EndStyledToolbarWindow();
        }

        private void DrawToolbarButton(
            string id,
            IconType icon,
            bool active,
            Action onClick,
            string tooltip,
            System.Numerics.Vector2 size,
            Action? onRightClick = null,
            string? rightClickHint = null)
        {
            if (active)
            {
                ImGui.PushStyleColor(ImGuiCol.Button, new System.Numerics.Vector4(0.3f, 0.5f, 0.7f, 1f));
                ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new System.Numerics.Vector4(0.35f, 0.55f, 0.75f, 1f));
            }

            if (ImGui.ImageButton(id, _iconFactory.GetIcon(icon), size))
            {
                onClick();
            }

            if (onRightClick != null && IsLastItemRightClicked())
            {
                onRightClick();
            }

            if (active)
            {
                ImGui.PopStyleColor(2);
            }

            if (ImGui.IsItemHovered())
            {
                if (onRightClick == null)
                {
                    ImGui.SetTooltip(tooltip);
                }
                else
                {
                    var suffix = string.IsNullOrWhiteSpace(rightClickHint) ? "Right click for options" : rightClickHint;
                    ImGui.SetTooltip($"{tooltip}\n{suffix}");
                }
            }
        }

        private static bool IsLastItemRightClicked() =>
            ImGui.IsItemClicked(ImGuiMouseButton.Right) ||
            (ImGui.IsItemHovered() && ImGui.IsMouseReleased(ImGuiMouseButton.Right));

        private void ProcessPendingPopupRequest()
        {
            if (_popupToOpen == null) return;
            ImGui.OpenPopup(_popupToOpen);
            _popupToOpen = null;
        }

        /// <summary>
        /// Helper to draw a button with an icon and text label
        /// </summary>
        private bool DrawIconTextButton(string id, IconType icon, string text, System.Numerics.Vector2 iconSize)
        {
            bool clicked = false;
            float availWidth = ImGui.GetContentRegionAvail().X;

            ImGui.PushID(id);

            // Draw icon
            ImGui.Image(_iconFactory.GetIcon(icon), iconSize);
            ImGui.SameLine();

            // Draw button with remaining width
            float buttonWidth = availWidth - iconSize.X - ImGui.GetStyle().ItemSpacing.X;
            if (ImGui.Button(text, new System.Numerics.Vector2(buttonWidth, iconSize.Y)))
            {
                clicked = true;
            }

            ImGui.PopID();

            return clicked;
        }

        private const ImGuiWindowFlags ToolbarWindowFlags =
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings;

        private bool BeginStyledToolbarWindow(string id, System.Numerics.Vector2 position, System.Numerics.Vector2 size)
        {
            ImGui.SetNextWindowPos(position);
            ImGui.SetNextWindowSize(size);
            ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, ToolbarWindowPadding);
            ImGui.PushStyleVar(ImGuiStyleVar.ItemSpacing, ToolbarItemSpacing);
            return ImGui.Begin(id, ToolbarWindowFlags);
        }

        private static void EndStyledToolbarWindow()
        {
            ImGui.End();
            ImGui.PopStyleVar(2);
        }

        private void RenderVerticalToolbar()
        {
            float startY = GetTopUiHeight();
            float height = ClientSize.Y - startY - (_showLogPanel ? _logPanelHeight : 0);

            bool shown = BeginStyledToolbarWindow("##VToolbar", new System.Numerics.Vector2(0, startY), new System.Numerics.Vector2(_verticalToolbarWidth, height));
            if (shown)
            {
                var size = ToolbarIconSize;

                // Focus
                if (ImGui.ImageButton("##Focus", _iconFactory.GetIcon(IconType.Focus), size))
                    _viewport.FocusOnSelection();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Focus on Selection (F)");

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // Standalone AI Actions - each can be run independently
                ImGui.TextDisabled("AI Steps");
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("AI Processing Steps");

                // Point Cloud Generation (uses engine from Settings)
                if (ImGui.ImageButton("##PointCloud", _iconFactory.GetIcon(IconType.PointCloudGen), size))
                    RunSingleStep(GetReconstructionStep());
                if (ImGui.IsItemHovered()) ImGui.SetTooltip($"{GetCurrentEngineName()} Point Cloud");
                if (IsLastItemRightClicked()) OpenRunOptionsDialog();

                // Single-image 3D generation models
                if (ImGui.ImageButton("##TripoSR", _iconFactory.GetIcon(IconType.TripoSR), size))
                    RunSingleStep(WorkflowStep.TripoSRGeneration);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("TripoSR (Single Image)");
                if (IsLastItemRightClicked()) OpenRunOptionsDialog();

                if (ImGui.ImageButton("##LGM", _iconFactory.GetIcon(IconType.LGM), size))
                    RunSingleStep(WorkflowStep.LGMGeneration);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("LGM Gaussian (Single Image)");
                if (IsLastItemRightClicked()) OpenRunOptionsDialog();

                if (ImGui.ImageButton("##Wonder3D", _iconFactory.GetIcon(IconType.Wonder3D), size))
                    RunSingleStep(WorkflowStep.Wonder3DGeneration);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Wonder3D (Single-Image Multi-View)");
                if (IsLastItemRightClicked()) OpenRunOptionsDialog();

                ImGui.Spacing();

                // Refinement models
                ImGui.TextDisabled("Refine");
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Refinement Models");

                if (ImGui.ImageButton("##NeRF", _iconFactory.GetIcon(IconType.NeRF), size))
                    RunNeRFRefinementFromSelection();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("NeRF Refinement");
                if (IsLastItemRightClicked()) OpenRefinementOptionsDialog();

                if (ImGui.ImageButton("##DeepMeshPrior", _iconFactory.GetIcon(IconType.Refine), size))
                    RunDeepMeshPriorRefinement();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("DeepMeshPrior Refinement");
                if (IsLastItemRightClicked()) OpenRefinementOptionsDialog();

                if (ImGui.ImageButton("##TripoSF", _iconFactory.GetIcon(IconType.Optimize), size))
                    RunTripoSFRefinement();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("TripoSF Refinement");
                if (IsLastItemRightClicked()) OpenRefinementOptionsDialog();

                ImGui.Spacing();

                // Meshing
                ImGui.TextDisabled("Mesh Gen");
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Mesh Generation");

                if (ImGui.ImageButton("##Poisson", _iconFactory.GetIcon(IconType.MeshGen), size))
                    RunMeshFromSelectedPointClouds();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Poisson Mesh Reconstruction");
                if (IsLastItemRightClicked()) OpenMeshingOptionsDialog();

                if (ImGui.ImageButton("##AutoRig", _iconFactory.GetIcon(IconType.Rig), size))
                    RunSingleStep(WorkflowStep.UniRigAutoRig);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("UniRig Auto-Rig");
                if (IsLastItemRightClicked()) OpenRefinementOptionsDialog();

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // Mesh Operations Section
                ImGui.TextDisabled("Mesh");
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Mesh Operations");

                if (ImGui.ImageButton("##Decimate", _iconFactory.GetIcon(IconType.Decimate), size))
                    ApplyDecimatePreset();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Decimate Mesh (50%)");
                if (IsLastItemRightClicked()) OnDecimate();

                if (ImGui.ImageButton("##Optimize", _iconFactory.GetIcon(IconType.Optimize), size))
                    ApplyOptimizePreset();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Optimize Mesh");
                if (IsLastItemRightClicked()) OnOptimize();

                if (ImGui.ImageButton("##Clean", _iconFactory.GetIcon(IconType.Clean), size))
                    OnCleanup();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Cleanup Mesh");
                if (IsLastItemRightClicked()) OnCleanup();

                if (ImGui.ImageButton("##Bake", _iconFactory.GetIcon(IconType.Bake), size))
                    OnBakeTextures();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Bake Textures");
                if (IsLastItemRightClicked()) OnBakeTextures();

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                if (ImGui.ImageButton("##Delete", _iconFactory.GetIcon(IconType.Delete), size))
                {
                    // In Pen mode, delete selected triangles
                    if (_viewport.CurrentGizmoMode == GizmoMode.Pen && _viewport.MeshEditingTool.SelectedTriangles.Count > 0)
                    {
                        _viewport.MeshEditingTool.DeleteSelectedTriangles();
                        _logBuffer += "Deleted selected triangles.\n";
                    }
                    else
                    {
                        OnDeleteSelected();
                    }
                }
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Delete Selected");
            }
            EndStyledToolbarWindow();
        }

        private void RenderLeftPanel()
        {
            float startX = _showVerticalToolbar ? _verticalToolbarWidth : 0;
            float startY = GetTopUiHeight();
            float height = ClientSize.Y - startY - (_showLogPanel ? _logPanelHeight : 0);

            ImGui.SetNextWindowPos(new System.Numerics.Vector2(startX, startY));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(_leftPanelWidth, height));

            ImGui.Begin("Project", ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize);
            {
                if (ImGui.BeginTabBar("ProjectTabs"))
                {
                    if (ImGui.BeginTabItem("Images"))
                    {
                        RenderImagesPanel();
                        ImGui.EndTabItem();
                    }
                    if (ImGui.BeginTabItem("Scene"))
                    {
                        RenderSceneGraph();
                        ImGui.EndTabItem();
                    }
                    ImGui.EndTabBar();
                }
            }
            ImGui.End();
        }

        private void RenderImagesPanel()
        {
            int thumbnailCount;
            lock (_imageThumbnails)
            {
                thumbnailCount = _imageThumbnails.Count;
            }

            ImGui.Text($"Loaded: {_loadedImages.Count}  Thumbs: {thumbnailCount}");

            if (ImGui.Button("Add Images..."))
            {
                OnAddImages();
            }
            ImGui.SameLine();
            if (ImGui.Button("Clear"))
            {
                ClearImages();
            }

            ImGui.SameLine();
            bool hasAnyDepth = _loadedImages.Any(i => HasRenderableDepthMap(i.DepthMap));
            bool rgbMode = _imagePanelMode == ImagePanelMode.RGB;
            bool depthMode = _imagePanelMode == ImagePanelMode.DepthMap;

            if (rgbMode)
            {
                ImGui.PushStyleColor(ImGuiCol.Button, new System.Numerics.Vector4(0.3f, 0.55f, 0.7f, 1f));
                ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new System.Numerics.Vector4(0.35f, 0.6f, 0.75f, 1f));
            }
            if (ImGui.ImageButton("##ImgModeRgb", _iconFactory.GetIcon(IconType.Rgb), new System.Numerics.Vector2(20, 20)))
            {
                _imagePanelMode = ImagePanelMode.RGB;
            }
            if (rgbMode) ImGui.PopStyleColor(2);
            if (ImGui.IsItemHovered()) ImGui.SetTooltip("RGB thumbnails");

            ImGui.SameLine();
            if (!hasAnyDepth)
            {
                ImGui.BeginDisabled();
            }
            if (depthMode)
            {
                ImGui.PushStyleColor(ImGuiCol.Button, new System.Numerics.Vector4(0.3f, 0.55f, 0.7f, 1f));
                ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new System.Numerics.Vector4(0.35f, 0.6f, 0.75f, 1f));
            }
            if (ImGui.ImageButton("##ImgModeDepth", _iconFactory.GetIcon(IconType.DepthMap), new System.Numerics.Vector2(20, 20)))
            {
                if (hasAnyDepth)
                    _imagePanelMode = ImagePanelMode.DepthMap;
            }
            if (depthMode) ImGui.PopStyleColor(2);
            if (!hasAnyDepth)
            {
                ImGui.EndDisabled();
            }
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(hasAnyDepth ? "Depth map thumbnails" : "Depth maps not available");

            ImGui.Separator();

            if (_imagePanelMode == ImagePanelMode.DepthMap && hasAnyDepth)
            {
                ImGui.TextDisabled("Depth Legend");
                DrawColormapLegend(180.0f, 12.0f, "Near", "Far");
                ImGui.Separator();
            }

            // Thumbnail grid
            float thumbSize = 64;
            float availWidth = ImGui.GetContentRegionAvail().X;
            int columns = Math.Max(1, (int)(availWidth / (thumbSize + 8)));

            ImGui.BeginChild("ImageGrid", new System.Numerics.Vector2(0, 0), ImGuiChildFlags.None);

            int col = 0;
            for (int i = 0; i < _loadedImages.Count; i++)
            {
                var pImg = _loadedImages[i];
                string path = pImg.FilePath;
                string displayName = pImg.Alias;

                ImGui.PushID(i);

                // Draw thumbnail or placeholder
                int thumbTex = -1;
                bool hasDepth = HasRenderableDepthMap(pImg.DepthMap);
                if (_imagePanelMode == ImagePanelMode.DepthMap && hasDepth)
                {
                    lock (_imageDepthThumbnails)
                    {
                        _imageDepthThumbnails.TryGetValue(path, out thumbTex);
                    }
                    if (thumbTex <= 0)
                    {
                        thumbTex = EnsureDepthThumbnail(path, pImg.DepthMap, (int)thumbSize);
                    }
                }
                else
                {
                    lock (_imageThumbnails)
                    {
                        _imageThumbnails.TryGetValue(path, out thumbTex);
                    }
                }

                bool isSelected = i == _selectedImageIndex;
                if (isSelected)
                {
                    ImGui.PushStyleColor(ImGuiCol.Button, new System.Numerics.Vector4(0.3f, 0.5f, 0.7f, 1f));
                }

                if (thumbTex > 0)
                {
                    if (ImGui.ImageButton($"##img{i}", (IntPtr)thumbTex, new System.Numerics.Vector2(thumbSize, thumbSize)))
                    {
                        _selectedImageIndex = i;
                    }
                }
                else
                {
                    // Placeholder button
                    string shortName = displayName.Length > 6 ? displayName.Substring(0, 6) : displayName;
                    string label = _imagePanelMode == ImagePanelMode.DepthMap && !hasDepth
                        ? "[NoDepth]"
                        : $"[{shortName}...]";
                    if (ImGui.Button(label, new System.Numerics.Vector2(thumbSize, thumbSize)))
                    {
                        _selectedImageIndex = i;
                    }
                }

                if (isSelected)
                {
                    ImGui.PopStyleColor();
                }

                if (ImGui.IsItemHovered())
                {
                    ImGui.SetTooltip($"{displayName}\n({Path.GetFileName(path)})");
                }

                // Handle Renaming Input
                if (_renamingImage == pImg)
                {
                    ImGui.SetKeyboardFocusHere();
                    if (ImGui.InputText("##renameImg", ref _imageRenameBuffer, 64, ImGuiInputTextFlags.EnterReturnsTrue | ImGuiInputTextFlags.AutoSelectAll))
                    {
                        pImg.Alias = _imageRenameBuffer;
                        _renamingImage = null;
                        _isDirty = true;
                    }
                    if (ImGui.IsItemDeactivated() && ImGui.IsKeyPressed(ImGuiKey.Escape))
                    {
                        _renamingImage = null;
                    }
                    if (ImGui.IsItemDeactivatedAfterEdit())
                    {
                        pImg.Alias = _imageRenameBuffer;
                        _renamingImage = null;
                        _isDirty = true;
                    }
                }

                // Double click to preview
                if (ImGui.IsItemHovered() && ImGui.IsMouseDoubleClicked(ImGuiMouseButton.Left))
                {
                    bool showDepth = _imagePanelMode == ImagePanelMode.DepthMap && hasDepth;
                    OpenImagePreview(path, pImg.DepthMap, showDepth);
                }

                // Context menu
                if (ImGui.BeginPopupContextItem())
                {
                    if (ImGui.MenuItem("Preview"))
                    {
                        bool showDepth = _imagePanelMode == ImagePanelMode.DepthMap && hasDepth;
                        OpenImagePreview(path, pImg.DepthMap, showDepth);
                    }

                    if (ImGui.MenuItem("Depth View", "", false, hasDepth))
                    {
                        OpenImagePreview(path, pImg.DepthMap, showDepth: true);
                    }

                    if (ImGui.MenuItem("Rename"))
                    {
                        _renamingImage = pImg;
                        _imageRenameBuffer = pImg.Alias;
                    }

                    if (ImGui.MenuItem("Remove"))
                    {
                        lock (_imageThumbnails)
                        {
                            if (_imageThumbnails.TryGetValue(path, out int t))
                            {
                                TextureLoader.DeleteTexture(t);
                                _imageThumbnails.Remove(path);
                            }
                        }
                        lock (_imageDepthThumbnails)
                        {
                            if (_imageDepthThumbnails.TryGetValue(path, out int t))
                            {
                                TextureLoader.DeleteTexture(t);
                                _imageDepthThumbnails.Remove(path);
                            }
                        }
                        _loadedImages.RemoveAt(i);
                        i--;
                    }
                    ImGui.EndPopup();
                }

                ImGui.PopID();

                col++;
                if (col < columns)
                {
                    ImGui.SameLine();
                }
                else
                {
                    col = 0;
                }
            }

            ImGui.EndChild();
        }

        private static bool HasRenderableDepthMap(float[,]? depthMap)
        {
            return depthMap != null && depthMap.GetLength(0) > 0 && depthMap.GetLength(1) > 0;
        }

        private static bool TryResolvePoseImageSize(CameraPose pose, out int width, out int height)
        {
            width = pose.Width;
            height = pose.Height;
            if (width > 0 && height > 0)
            {
                return true;
            }

            if (ImageUtils.TryGetImageDimensions(pose.ImagePath, out width, out height))
            {
                pose.Width = width;
                pose.Height = height;
                return true;
            }

            width = 0;
            height = 0;
            return false;
        }

        private int EnsureDepthThumbnail(string path, float[,]? depthMap, int maxSize)
        {
            if (!HasRenderableDepthMap(depthMap))
                return -1;

            int existing = -1;
            lock (_imageDepthThumbnails)
            {
                if (_imageDepthThumbnails.TryGetValue(path, out existing) && existing > 0)
                    return existing;
            }

            try
            {
                using var thumb = CreateDepthThumbnailBitmap(depthMap, maxSize);
                if (thumb == null || thumb.Width <= 0 || thumb.Height <= 0 || thumb.GetPixels() == IntPtr.Zero)
                {
                    Logger.Warn($"Depth thumbnail skipped for {Path.GetFileName(path)}: invalid thumbnail bitmap.");
                    return -1;
                }

                int tex = TextureLoader.CreateTextureFromBitmap(thumb);
                if (tex > 0)
                {
                    lock (_imageDepthThumbnails)
                    {
                        _imageDepthThumbnails[path] = tex;
                    }
                }
                return tex;
            }
            catch (Exception ex)
            {
                Logger.Exception(ex, $"Failed to create depth thumbnail for {path}");
                return -1;
            }
        }

        private void OpenImagePreview(string path, float[,]? depthMap, bool showDepth)
        {
            _previewImagePath = path;
            _showImagePreview = true;
            _previewShowsDepth = showDepth && HasRenderableDepthMap(depthMap);

            if (_previewTexture > 0)
            {
                TextureLoader.DeleteTexture(_previewTexture);
                _previewTexture = -1;
            }

            if (_previewShowsDepth && depthMap != null)
            {
                int maxTextureSize = GetSafePreviewTextureLimit();
                using var depthBitmap = CreateDepthPreviewBitmap(depthMap, maxTextureSize);
                _previewTexture = depthBitmap != null ? TextureLoader.CreateTextureFromBitmap(depthBitmap) : -1;
                if (_previewTexture <= 0)
                {
                    Logger.Warn($"Depth preview unavailable for {Path.GetFileName(path)}. Falling back to RGB.");
                    _previewShowsDepth = false;
                    _previewTexture = TextureLoader.LoadTextureFromFile(path);
                }
            }
            else
            {
                _previewTexture = TextureLoader.LoadTextureFromFile(path);
            }
        }

        private static int GetSafePreviewTextureLimit()
        {
            int glMax = MaxDepthPreviewTextureSize;
            try
            {
                GL.GetInteger(GetPName.MaxTextureSize, out int queriedMax);
                if (queriedMax > 0)
                {
                    glMax = Math.Min(queriedMax, MaxDepthPreviewTextureSize);
                }
            }
            catch
            {
                // Ignore and keep fallback limit.
            }

            return Math.Clamp(glMax, 256, MaxDepthPreviewTextureSize);
        }

        private static SKBitmap? CreateDepthPreviewBitmap(float[,] depthMap, int maxTextureSize)
        {
            var colorized = ImageUtils.ColorizeDepthMap(depthMap);
            if (colorized == null || colorized.Width <= 0 || colorized.Height <= 0 || colorized.GetPixels() == IntPtr.Zero)
            {
                colorized?.Dispose();
                return null;
            }

            if (colorized.Width <= maxTextureSize && colorized.Height <= maxTextureSize)
            {
                return colorized;
            }

            float scale = Math.Min((float)maxTextureSize / colorized.Width, (float)maxTextureSize / colorized.Height);
            int targetWidth = Math.Max(1, (int)(colorized.Width * scale));
            int targetHeight = Math.Max(1, (int)(colorized.Height * scale));
            var resizedInfo = new SKImageInfo(targetWidth, targetHeight, SKColorType.Rgba8888, SKAlphaType.Premul);
            var resized = new SKBitmap(resizedInfo);
            using var canvas = new SKCanvas(resized);
            canvas.Clear(SKColors.Transparent);
            canvas.DrawBitmap(colorized, new SKRect(0, 0, targetWidth, targetHeight));
            canvas.Flush();
            colorized.Dispose();
            return resized;
        }

        private static SKBitmap? CreateDepthThumbnailBitmap(float[,] depthMap, int maxSize)
        {
            int srcWidth = depthMap.GetLength(0);
            int srcHeight = depthMap.GetLength(1);
            if (srcWidth <= 0 || srcHeight <= 0 || maxSize <= 0)
                return null;

            float minDepth = float.MaxValue;
            float maxDepth = float.MinValue;
            bool hasValidDepth = false;

            for (int y = 0; y < srcHeight; y++)
            {
                for (int x = 0; x < srcWidth; x++)
                {
                    float d = depthMap[x, y];
                    if (float.IsFinite(d) && d > 0.0f)
                    {
                        hasValidDepth = true;
                        if (d < minDepth) minDepth = d;
                        if (d > maxDepth) maxDepth = d;
                    }
                }
            }

            if (!hasValidDepth)
                return null;

            float range = maxDepth - minDepth;
            if (!float.IsFinite(range) || range < 0.0001f)
                range = 1.0f;

            float scale = Math.Min((float)maxSize / srcWidth, (float)maxSize / srcHeight);
            int width = Math.Max(1, (int)MathF.Round(srcWidth * scale));
            int height = Math.Max(1, (int)MathF.Round(srcHeight * scale));

            var bitmap = new SKBitmap(new SKImageInfo(width, height, SKColorType.Bgra8888, SKAlphaType.Premul));
            if (bitmap.GetPixels() == IntPtr.Zero)
            {
                bitmap.Dispose();
                return null;
            }

            for (int y = 0; y < height; y++)
            {
                int srcY = Math.Clamp((int)(y * (float)srcHeight / height), 0, srcHeight - 1);
                for (int x = 0; x < width; x++)
                {
                    int srcX = Math.Clamp((int)(x * (float)srcWidth / width), 0, srcWidth - 1);
                    float d = depthMap[srcX, srcY];
                    if (!float.IsFinite(d) || d <= 0.0f)
                    {
                        bitmap.SetPixel(x, y, new SKColor(0, 0, 0, 0));
                        continue;
                    }

                    float t = (d - minDepth) / range;
                    t = Math.Clamp(t, 0.0f, 1.0f);
                    var (r, g, b) = ImageUtils.TurboColormap(t);
                    bitmap.SetPixel(x, y, new SKColor((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255));
                }
            }

            return bitmap;
        }

        private void DrawColormapLegend(float width, float height, string leftLabel, string rightLabel)
        {
            width = Math.Max(1.0f, width);
            height = Math.Max(1.0f, height);
            ImGui.TextDisabled(leftLabel);
            ImGui.SameLine();
            var p = ImGui.GetCursorScreenPos();
            var drawList = ImGui.GetWindowDrawList();
            int steps = 48;
            float stepWidth = width / steps;
            for (int i = 0; i < steps; i++)
            {
                float t0 = (float)i / steps;
                float t1 = (float)(i + 1) / steps;
                var (r0, g0, b0) = ImageUtils.TurboColormap(t0);
                var (r1, g1, b1) = ImageUtils.TurboColormap(t1);
                uint c0 = ImGui.ColorConvertFloat4ToU32(new System.Numerics.Vector4(r0, g0, b0, 1));
                uint c1 = ImGui.ColorConvertFloat4ToU32(new System.Numerics.Vector4(r1, g1, b1, 1));
                float x0 = p.X + i * stepWidth;
                float x1 = p.X + (i + 1) * stepWidth;
                drawList.AddRectFilledMultiColor(
                    new System.Numerics.Vector2(x0, p.Y),
                    new System.Numerics.Vector2(x1, p.Y + height),
                    c0, c1, c1, c0);
            }
            drawList.AddRect(
                p,
                new System.Numerics.Vector2(p.X + width, p.Y + height),
                ImGui.ColorConvertFloat4ToU32(new System.Numerics.Vector4(0.2f, 0.2f, 0.2f, 1f)));
            ImGui.Dummy(new System.Numerics.Vector2(width, height));
            ImGui.SameLine();
            ImGui.TextDisabled(rightLabel);
        }

        private void ClearImages()
        {
            foreach (var kv in _imageThumbnails)
            {
                TextureLoader.DeleteTexture(kv.Value);
            }
            _imageThumbnails.Clear();
            foreach (var kv in _imageDepthThumbnails)
            {
                TextureLoader.DeleteTexture(kv.Value);
            }
            _imageDepthThumbnails.Clear();
            _loadedImages.Clear();
            _selectedImageIndex = -1;
        }

        private void RenderSceneGraph()
        {
            SceneObject? objectToDelete = null;
            SceneObject? objectToDuplicate = null;

            var allObjects = _sceneGraph.GetAllObjects().ToList();
            int visibleCount = allObjects.Count(o => o.Visible);
            int selectedCount = _sceneGraph.SelectedObjects.Count;

            ImGui.Text($"Total: {allObjects.Count}  Visible: {visibleCount}  Selected: {selectedCount}");
            ImGui.Separator();

            ImGui.BeginChild("SceneList", new System.Numerics.Vector2(0, 0), ImGuiChildFlags.Borders);
            if (allObjects.Count == 0)
            {
                ImGui.TextDisabled("No objects in scene");
                ImGui.EndChild();
                return;
            }

            const ImGuiTableFlags tableFlags =
                ImGuiTableFlags.RowBg |
                ImGuiTableFlags.BordersInnerV |
                ImGuiTableFlags.BordersOuter |
                ImGuiTableFlags.SizingFixedFit;

            if (ImGui.BeginTable("SceneObjectTable", 3, tableFlags))
            {
                ImGui.TableSetupColumn("Object", ImGuiTableColumnFlags.WidthStretch);
                ImGui.TableSetupColumn("Visible", ImGuiTableColumnFlags.WidthFixed, 64f);
                ImGui.TableSetupColumn("Camera", ImGuiTableColumnFlags.WidthFixed, 64f);
                ImGui.TableHeadersRow();

                foreach (var obj in allObjects)
                {
                    bool selected = obj.Selected;
                    string name = obj.Name ?? $"Object {obj.Id}";
                    int depth = GetSceneObjectDepth(obj);
                    string indent = depth > 0 ? new string(' ', depth * 2) : string.Empty;
                    string icon = obj switch
                    {
                        MeshObject => "[M] ",
                        PointCloudObject => "[P] ",
                        CameraObject => "[C] ",
                        SkeletonObject => "[S] ",
                        GroupObject => "[G] ",
                        _ => "[O] "
                    };
                    string displayName = obj.Visible ? $"{indent}{icon}{name}" : $"{indent}{icon}{name} (hidden)";

                    ImGui.PushID(obj.Id);
                    ImGui.TableNextRow();

                    ImGui.TableSetColumnIndex(0);
                    if (_renamingObject == obj)
                    {
                        ImGui.SetKeyboardFocusHere();
                        if (ImGui.InputText("##renameObj", ref _renameBuffer, 64, ImGuiInputTextFlags.EnterReturnsTrue | ImGuiInputTextFlags.AutoSelectAll))
                        {
                            obj.Name = _renameBuffer;
                            _renamingObject = null;
                            _isDirty = true;
                        }
                        if (ImGui.IsItemDeactivated() && ImGui.IsKeyPressed(ImGuiKey.Escape))
                        {
                            _renamingObject = null;
                        }
                        if (ImGui.IsItemDeactivatedAfterEdit())
                        {
                            obj.Name = _renameBuffer;
                            _renamingObject = null;
                            _isDirty = true;
                        }
                    }
                    else
                    {
                        if (!obj.Visible)
                            ImGui.PushStyleVar(ImGuiStyleVar.Alpha, 0.55f);

                        if (ImGui.Selectable($"{displayName}##sel", selected))
                        {
                            bool addToSelection = IsMultiSelectModifierDown();
                            if (!addToSelection)
                                _sceneGraph.ClearSelection();

                            if (addToSelection && selected)
                                _sceneGraph.Deselect(obj);
                            else
                                _sceneGraph.Select(obj, addToSelection);
                        }

                        if (!obj.Visible)
                            ImGui.PopStyleVar();

                        if (ImGui.BeginPopupContextItem("SceneGraphItemCtx"))
                        {
                            if (!obj.Selected)
                            {
                                bool addToSelection = IsMultiSelectModifierDown();
                                if (!addToSelection)
                                    _sceneGraph.ClearSelection();
                                _sceneGraph.Select(obj, addToSelection);
                            }

                            if (ImGui.MenuItem("Rename"))
                            {
                                _renamingObject = obj;
                                _renameBuffer = obj.Name ?? "";
                            }
                            if (ImGui.MenuItem("Focus")) _viewport.FocusOnObject(obj);
                            if (ImGui.MenuItem(obj.Visible ? "Hide" : "Show"))
                            {
                                obj.Visible = !obj.Visible;
                                _isDirty = true;
                            }
                            if (ImGui.BeginMenu("Render Mode"))
                            {
                                _isDirty |= DrawRenderModeMenuItem(obj, "Inherit (Global)", ObjectRenderMode.InheritGlobal);
                                _isDirty |= DrawRenderModeMenuItem(obj, "Shading", ObjectRenderMode.Shaded);
                                _isDirty |= DrawRenderModeMenuItem(obj, "Wireframe", ObjectRenderMode.Wireframe);
                                _isDirty |= DrawRenderModeMenuItem(obj, "No Textures", ObjectRenderMode.NoTexture);
                                _isDirty |= DrawRenderModeMenuItem(obj, "Textures", ObjectRenderMode.Texture);
                                _isDirty |= DrawRenderModeMenuItem(obj, "Bounding Box Only", ObjectRenderMode.BoundingBoxOnly);
                                ImGui.EndMenu();
                            }
                            ImGui.Separator();
                            if (ImGui.MenuItem("Move...")) OpenTransformDialog(TransformDialogMode.Move);
                            if (ImGui.MenuItem("Rotate...")) OpenTransformDialog(TransformDialogMode.Rotate);
                            if (ImGui.MenuItem("Scale...")) OpenTransformDialog(TransformDialogMode.Scale);
                            if (ImGui.MenuItem("Reset Transform"))
                            {
                                OnResetTransform();
                                _isDirty = true;
                            }
                            ImGui.Separator();
                            if (obj is CameraObject contextCam && ImGui.MenuItem(contextCam.ShowFrustum ? "Hide Frustum" : "Show Frustum"))
                            {
                                contextCam.ShowFrustum = !contextCam.ShowFrustum;
                                _isDirty = true;
                            }
                            if (ImGui.MenuItem("Delete")) objectToDelete = obj;
                            if (ImGui.MenuItem("Duplicate")) objectToDuplicate = obj;
                            ImGui.EndPopup();
                        }
                    }

                    ImGui.TableSetColumnIndex(1);
                    if (ImGui.SmallButton($"{(obj.Visible ? "Hide" : "Show")}##vis"))
                    {
                        obj.Visible = !obj.Visible;
                        _isDirty = true;
                    }

                    ImGui.TableSetColumnIndex(2);
                    if (obj is CameraObject cam)
                    {
                        if (ImGui.SmallButton($"{(cam.ShowFrustum ? "FrOff" : "FrOn")}##fr"))
                        {
                            cam.ShowFrustum = !cam.ShowFrustum;
                            _isDirty = true;
                        }
                    }
                    else
                    {
                        ImGui.TextDisabled("-");
                    }

                    ImGui.PopID();
                }

                ImGui.EndTable();
            }

            if (objectToDelete != null)
            {
                _sceneGraph.RemoveObject(objectToDelete);
                _isDirty = true;
            }
            if (objectToDuplicate != null)
            {
                OnDuplicateObject(objectToDuplicate);
            }

            ImGui.EndChild();
        }

        private bool IsMultiSelectModifierDown()
        {
            var io = ImGui.GetIO();
            if (ImGui.IsKeyDown(ImGuiKey.LeftCtrl) || ImGui.IsKeyDown(ImGuiKey.RightCtrl) ||
                ImGui.IsKeyDown(ImGuiKey.LeftShift) || ImGui.IsKeyDown(ImGuiKey.RightShift) ||
                ImGui.IsKeyDown(ImGuiKey.LeftSuper) || ImGui.IsKeyDown(ImGuiKey.RightSuper))
                return true;

            if (io.KeyCtrl || io.KeyShift || io.KeySuper)
                return true;

            if (_ctrlModifierDown || _shiftModifierDown || _superModifierDown)
                return true;

            var keyboard = KeyboardState;
            return keyboard.IsKeyDown(Keys.LeftControl) || keyboard.IsKeyDown(Keys.RightControl) ||
                   keyboard.IsKeyDown(Keys.LeftShift) || keyboard.IsKeyDown(Keys.RightShift) ||
                   keyboard.IsKeyDown(Keys.LeftSuper) || keyboard.IsKeyDown(Keys.RightSuper);
        }

        private static bool DrawRenderModeMenuItem(SceneObject obj, string label, ObjectRenderMode mode)
        {
            bool selected = obj.RenderMode == mode;
            if (!ImGui.MenuItem(label, "", selected))
                return false;

            obj.RenderMode = mode;
            return true;
        }

        private static int GetSceneObjectDepth(SceneObject obj)
        {
            int depth = 0;
            var parent = obj.Parent;
            while (parent != null && parent.Parent != null)
            {
                depth++;
                parent = parent.Parent;
            }
            return depth;
        }

        private void RenderRightPanel()
        {
            float startY = GetTopUiHeight();
            float height = ClientSize.Y - startY - (_showLogPanel ? _logPanelHeight : 0);

            ImGui.SetNextWindowPos(new System.Numerics.Vector2(ClientSize.X - _rightPanelWidth, startY));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(_rightPanelWidth, height));

            ImGui.Begin("Properties", ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize);
            {
                // Show different panels based on mode
                if (_viewport.CurrentGizmoMode == GizmoMode.Pen)
                {
                    RenderPenModePanel();
                }
                else if (_viewport.CurrentGizmoMode == GizmoMode.Rigging)
                {
                    RenderRiggingPanel();
                }
                else
                {
                    RenderProperties();
                }
            }
            ImGui.End();
        }

        private void RenderPenModePanel()
        {
            ImGui.TextColored(new System.Numerics.Vector4(1f, 0.6f, 0.2f, 1f), "Triangle Editing Mode");
            ImGui.Separator();

            var tool = _viewport.MeshEditingTool;
            var (triCount, vertCount, meshCount) = tool.GetSelectionStats();

            ImGui.Text($"Selected: {triCount} triangles");
            ImGui.Text($"Vertices: {vertCount}");
            ImGui.Text($"Meshes: {meshCount}");

            ImGui.Separator();

            // Edit Mode
            ImGui.Text("Edit Mode:");
            int mode = (int)tool.Mode;
            string[] modes = { "Select", "Delete", "Paint", "Weld", "Extrude", "Inset", "MoveVertices", "Bridge" };
            if (ImGui.Combo("##EditMode", ref mode, modes, modes.Length))
            {
                tool.Mode = (MeshEditMode)mode;
            }

            ImGui.Separator();

            // Paint Color (if in Paint mode)
            if (tool.Mode == MeshEditMode.Paint)
            {
                var color = new System.Numerics.Vector3(tool.PaintColor.X, tool.PaintColor.Y, tool.PaintColor.Z);
                if (ImGui.ColorEdit3("Paint Color", ref color))
                {
                    tool.PaintColor = new Vector3(color.X, color.Y, color.Z);
                }
            }

            ImGui.Separator();
            ImGui.Text("Actions:");

            var iconSize = new System.Numerics.Vector2(20, 20);

            ImGui.InputFloat3("Move Delta", ref _penMoveDelta);
            if (DrawIconTextButton("##PenMove", IconType.VertexMove, "Move Vertices", iconSize))
            {
                ApplyPenMoveVertices();
            }

            ImGui.SliderFloat("Extrude Dist", ref _penExtrudeDistance, -1.0f, 1.0f, "%.3f");
            if (DrawIconTextButton("##PenExtr", IconType.Extrude, "Extrude", iconSize))
            {
                ApplyPenExtrude();
            }

            ImGui.SliderFloat("Inset Amount", ref _penInsetAmount, 0.01f, 0.95f, "%.2f");
            if (DrawIconTextButton("##PenInset", IconType.Inset, "Inset", iconSize))
            {
                ApplyPenInset();
            }

            if (DrawIconTextButton("##PenBridge", IconType.Bridge, "Bridge 2 Triangles", iconSize))
            {
                ApplyPenBridge();
            }

            // Delete Selected button with icon
            if (DrawIconTextButton("##PenDel", IconType.Delete, "Delete Selected", iconSize))
            {
                if (triCount > 0)
                {
                    tool.DeleteSelectedTriangles();
                    _logBuffer += $"Deleted {triCount} triangles.\n";
                }
            }

            // Flip Normals button with icon
            if (DrawIconTextButton("##PenFlip", IconType.FlipNormals, "Flip Normals", iconSize))
            {
                if (triCount > 0)
                {
                    tool.FlipSelectedTriangles();
                    _logBuffer += $"Flipped {triCount} triangle normals.\n";
                }
            }

            // Subdivide button with icon
            if (DrawIconTextButton("##PenSub", IconType.Subdivide, "Subdivide", iconSize))
            {
                if (triCount > 0)
                {
                    tool.SubdivideSelectedTriangles();
                    _logBuffer += $"Subdivided {triCount} triangles.\n";
                }
            }

            // Weld Vertices button with icon
            if (DrawIconTextButton("##PenWeld", IconType.Weld, "Weld Vertices", iconSize))
            {
                if (triCount > 0)
                {
                    tool.WeldSelectedVertices();
                    _logBuffer += "Welded duplicate vertices.\n";
                }
            }

            // Paint Selected button with icon
            if (DrawIconTextButton("##PenPaint", IconType.Paint, "Paint Selected", iconSize))
            {
                if (triCount > 0)
                {
                    tool.PaintSelectedTriangles();
                    _logBuffer += $"Painted {triCount} triangles.\n";
                }
            }

            ImGui.Separator();
            ImGui.Text("Selection:");

            // Select All on selected mesh
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count > 0)
            {
                if (DrawIconTextButton("##PenSelAll", IconType.SelectAll, "Select All Tris", iconSize))
                {
                    foreach (var m in meshes)
                        tool.SelectAll(m);
                }

                if (DrawIconTextButton("##PenInvert", IconType.InvertSelection, "Invert Selection", iconSize))
                {
                    foreach (var m in meshes)
                        tool.InvertSelection(m);
                }
            }

            if (DrawIconTextButton("##PenGrow", IconType.GrowSelection, "Grow Selection", iconSize))
            {
                tool.GrowSelection();
            }

            if (DrawIconTextButton("##PenClear", IconType.ClearSelection, "Clear Selection", iconSize))
            {
                tool.ClearSelection();
            }

            ImGui.Separator();
            ImGui.TextDisabled("Tip: Shift+Click to multi-select");
            ImGui.TextDisabled("Press Escape to clear selection");
            ImGui.TextDisabled("Press Delete to remove triangles");
        }

        private void RenderRiggingPanel()
        {
            ImGui.TextColored(new System.Numerics.Vector4(0.4f, 1f, 0.4f, 1f), "Rigging Mode");
            ImGui.Separator();

            ImGui.Text("Skeleton Operations:");

            var iconSize = new System.Numerics.Vector2(20, 20);
            var selectedSkeleton = _sceneGraph.SelectedObjects.OfType<SkeletonObject>().FirstOrDefault();
            if (selectedSkeleton != null)
            {
                _activeSkeletonObject = selectedSkeleton;
            }
            else if (_activeSkeletonObject != null && !_sceneGraph.GetAllObjects().Contains(_activeSkeletonObject))
            {
                _activeSkeletonObject = null;
            }

            if (DrawIconTextButton("##RigAuto", IconType.Rig, "Auto Rig (UniRig)", iconSize))
            {
                OnAutoRig();
            }

            if (DrawIconTextButton("##RigNew", IconType.Skeleton, "Create Skeleton", iconSize))
            {
                OnCreateNewSkeletonImGui();
            }

            if (DrawIconTextButton("##RigHum", IconType.Skeleton, "Create Humanoid Skeleton", iconSize))
            {
                OnCreateHumanoidSkeletonImGui();
            }

            if (DrawIconTextButton("##RigSkel", IconType.Skeleton, "View Skeleton", iconSize))
            {
                // Toggle skeleton visibility in the scene
                var allSkeletons = _sceneGraph.GetAllObjects().OfType<SkeletonObject>().ToList();
                foreach (var skel in allSkeletons)
                {
                    skel.Visible = !skel.Visible;
                }
                _logBuffer += $"Toggled visibility for {allSkeletons.Count} skeleton(s).\n";
            }

            ImGui.Separator();
            ImGui.TextDisabled("Select mesh + Auto Rig, or create skeleton manually.");

            if (_activeSkeletonObject == null)
            {
                ImGui.TextDisabled("No active skeleton selected.");
                return;
            }

            var skeleton = _activeSkeletonObject.Skeleton;

            ImGui.Separator();
            ImGui.Text($"Active: {_activeSkeletonObject.Name}");
            ImGui.Text($"Joints: {skeleton.Joints.Count}");
            ImGui.Text($"Bones: {skeleton.Bones.Count}");

            if (DrawIconTextButton("##RigAddJoint", IconType.VertexMove, "Add Joint", iconSize))
            {
                OnAddJointImGui();
            }

            if (DrawIconTextButton("##RigAddBone", IconType.Link, "Add Bone", iconSize))
            {
                OnAddBoneImGui();
            }

            if (DrawIconTextButton("##RigDelJoint", IconType.Delete, "Delete Selected Joint(s)", iconSize))
            {
                OnDeleteSelectedJointsImGui();
            }

            ImGui.Separator();
            ImGui.Text("Joint List");

            ImGui.BeginChild("##RigJointList", new System.Numerics.Vector2(0, 170), ImGuiChildFlags.Borders);
            foreach (var joint in skeleton.GetJointsHierarchical())
            {
                int depth = GetJointDepth(joint);
                if (depth > 0) ImGui.Indent(depth * 12.0f);

                bool isSelected = joint.IsSelected;
                if (ImGui.Selectable($"{joint.Name}##rigJoint{joint.Id}", isSelected))
                {
                    bool addToSelection = IsMultiSelectModifierDown();
                    skeleton.SelectJoint(joint, addToSelection);
                    _sceneGraph.ClearSelection();
                    _sceneGraph.Select(_activeSkeletonObject, false);
                    _isDirty = true;
                }

                if (ImGui.BeginPopupContextItem($"RigJointCtx##{joint.Id}"))
                {
                    if (ImGui.MenuItem("Select"))
                    {
                        skeleton.SelectJoint(joint, false);
                        _isDirty = true;
                    }
                    if (ImGui.MenuItem("Select Additive"))
                    {
                        skeleton.SelectJoint(joint, true);
                        _isDirty = true;
                    }
                    if (ImGui.MenuItem("Toggle Visibility"))
                    {
                        joint.IsVisible = !joint.IsVisible;
                        _isDirty = true;
                    }
                    ImGui.EndPopup();
                }

                if (depth > 0) ImGui.Unindent(depth * 12.0f);
            }
            ImGui.EndChild();

            var selectedJoints = skeleton.GetSelectedJoints().ToList();
            if (selectedJoints.Count == 0)
            {
                ImGui.TextDisabled("Select a joint to edit properties.");
                return;
            }

            var activeJoint = selectedJoints[0];
            ImGui.Separator();
            ImGui.Text($"Editing Joint: {activeJoint.Name}");

            string jointName = activeJoint.Name;
            if (ImGui.InputText("Joint Name", ref jointName, 96))
            {
                activeJoint.Name = jointName;
                _isDirty = true;
            }

            var jointPos = new System.Numerics.Vector3(activeJoint.Position.X, activeJoint.Position.Y, activeJoint.Position.Z);
            if (ImGui.DragFloat3("Joint Position", ref jointPos, 0.01f))
            {
                activeJoint.Position = new Vector3(jointPos.X, jointPos.Y, jointPos.Z);
                _isDirty = true;
            }

            bool jointVisible = activeJoint.IsVisible;
            if (ImGui.Checkbox("Joint Visible", ref jointVisible))
            {
                activeJoint.IsVisible = jointVisible;
                _isDirty = true;
            }
        }

        private void RenderProperties()
        {
            if (_sceneGraph.SelectedObjects.Count == 0)
            {
                ImGui.TextDisabled("No object selected.");
                return;
            }

            var obj = _sceneGraph.SelectedObjects[0];

            // Name
            ImGui.Text($"ID: {obj.Id}");
            string name = obj.Name ?? "";
            if (ImGui.InputText("Name", ref name, 64)) obj.Name = name;

            ImGui.Separator();

            // Transform
            if (ImGui.CollapsingHeader("Transform", ImGuiTreeNodeFlags.DefaultOpen))
            {
                var pos = new System.Numerics.Vector3(obj.Position.X, obj.Position.Y, obj.Position.Z);
                if (ImGui.DragFloat3("Position", ref pos, 0.1f))
                    obj.Position = new Vector3(pos.X, pos.Y, pos.Z);

                var rot = new System.Numerics.Vector3(obj.Rotation.X, obj.Rotation.Y, obj.Rotation.Z);
                if (ImGui.DragFloat3("Rotation", ref rot, 1.0f))
                    obj.Rotation = new Vector3(rot.X, rot.Y, rot.Z);

                var scale = new System.Numerics.Vector3(obj.Scale.X, obj.Scale.Y, obj.Scale.Z);
                if (ImGui.DragFloat3("Scale", ref scale, 0.1f))
                    obj.Scale = ClampScale(new Vector3(scale.X, scale.Y, scale.Z));

                if (ImGui.Button("Reset Transform"))
                {
                    obj.Position = Vector3.Zero;
                    obj.Rotation = Vector3.Zero;
                    obj.Scale = Vector3.One;
                }
            }

            // Object-specific properties
            if (obj is PointCloudObject pc)
            {
                ImGui.Separator();
                if (ImGui.CollapsingHeader("Point Cloud", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    float ps = pc.PointSize;
                    if (ImGui.SliderFloat("Point Size", ref ps, 1.0f, 20.0f)) pc.PointSize = ps;
                    float visiblePct = pc.VisibleFraction * 100.0f;
                    if (ImGui.SliderFloat("Visible Points (%)", ref visiblePct, 0.0f, 100.0f, "%.1f%%"))
                        pc.VisibleFraction = Math.Clamp(visiblePct / 100.0f, 0.0f, 1.0f);
                    ImGui.Text($"Points: {pc.VisiblePointCount:N0} / {pc.PointCount:N0} visible");
                    ImGui.Text($"Normals: {(pc.Normals.Count == pc.PointCount ? "Estimated" : "None")}");

                    ImGui.Separator();
                    ImGui.Text("Filters");

                    ImGui.InputFloat("Voxel Size", ref _pcVoxelSize, 0.001f, 0.01f, "%.4f");
                    if (ImGui.Button("Apply Voxel Downsample"))
                        ApplyPointCloudVoxel();

                    ImGui.InputInt("Outlier K", ref _pcOutlierK);
                    ImGui.SliderFloat("Outlier Std", ref _pcOutlierStdRatio, 0.1f, 5.0f, "%.2f");
                    if (ImGui.Button("Apply Outlier Filter"))
                        ApplyPointCloudOutliers();

                    ImGui.InputFloat("Duplicate Threshold", ref _pcDuplicateThreshold, 0.0001f, 0.001f, "%.6f");
                    if (ImGui.Button("Remove Duplicates"))
                        ApplyPointCloudDuplicates();

                    ImGui.InputFloat("Sky Min Blue", ref _pcSkyMinBlue, 0.01f, 0.05f, "%.3f");
                    ImGui.InputFloat("Sky Max Red", ref _pcSkyMaxRed, 0.01f, 0.05f, "%.3f");
                    ImGui.InputFloat("Sky Max Green", ref _pcSkyMaxGreen, 0.01f, 0.05f, "%.3f");
                    ImGui.InputFloat("Sky Blue Dominance", ref _pcSkyBlueDominance, 0.01f, 0.05f, "%.3f");
                    _pcSkyMinBlue = Math.Clamp(_pcSkyMinBlue, 0.0f, 1.0f);
                    _pcSkyMaxRed = Math.Clamp(_pcSkyMaxRed, 0.0f, 1.0f);
                    _pcSkyMaxGreen = Math.Clamp(_pcSkyMaxGreen, 0.0f, 1.0f);
                    _pcSkyBlueDominance = Math.Max(0.0f, _pcSkyBlueDominance);
                    if (ImGui.Button("Remove Sky/Blue"))
                        ApplyPointCloudSkyBlue();

                    ImGui.InputInt("Normal K", ref _pcNormalK);
                    if (ImGui.Button("Estimate Normals"))
                        ApplyPointCloudNormals();

                    string[] axes = { "X", "Y", "Z" };
                    ImGui.Combo("Pass Axis", ref _pcPassAxis, axes, axes.Length);
                    ImGui.InputFloat("Pass Min", ref _pcPassMin);
                    ImGui.InputFloat("Pass Max", ref _pcPassMax);
                    if (ImGui.Button("Apply Pass-through"))
                        ApplyPointCloudPassThrough();

                    ImGui.InputFloat3("Radius Center", ref _pcRadiusCenter);
                    ImGui.InputFloat("Radius", ref _pcRadius, 0.01f, 0.1f, "%.3f");
                    if (ImGui.Button("Apply Radius Crop"))
                        ApplyPointCloudRadiusCrop();

                    ImGui.InputFloat("Dense Radius", ref _pcDenseRadius, 0.001f, 0.01f, "%.4f");
                    ImGui.InputInt("Dense Points/Seed", ref _pcDensePointsPerSeed);
                    if (ImGui.Button("Point Cloud -> Dense Cloud"))
                        ApplyPointCloudDenseCloud();
                }
            }
            else if (obj is MeshObject mo)
            {
                ImGui.Separator();
                if (ImGui.CollapsingHeader("Mesh", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    ImGui.Text($"Vertices: {mo.MeshData.Vertices.Count:N0}");
                    ImGui.Text($"Triangles: {mo.MeshData.Indices.Count / 3:N0}");
                    ImGui.Text($"Has Texture: {mo.MeshData.HasTexture}");

                    if (ImGui.Button("Recalculate Normals"))
                    {
                        mo.MeshData.RecalculateNormals();
                    }
                }
            }
        }

        private void RenderLogPanel()
        {
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, ClientSize.Y - _logPanelHeight));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.X, _logPanelHeight));

            ImGui.Begin("Log", ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize);
            {
                const int maxLogChars = 1024 * 1024;
                if (_logBuffer.Length >= maxLogChars)
                {
                    _logBuffer = _logBuffer[^ (maxLogChars - 1)..];
                }

                if (ImGui.Button("Clear"))
                {
                    _logBuffer = "";
                }
                ImGui.SameLine();
                if (ImGui.Button("Copy"))
                {
                    if (!string.IsNullOrEmpty(_logBuffer))
                    {
                        ClipboardString = _logBuffer;
                        Logger.Info("Log copied to clipboard.");
                    }
                }
                ImGui.SameLine();
                ImGui.Checkbox("Auto-scroll", ref _autoScroll);

                ImGui.Separator();

                ImGui.BeginChild("LogScroll", new System.Numerics.Vector2(0, 0), ImGuiChildFlags.None, ImGuiWindowFlags.HorizontalScrollbar);

                float width = ImGui.GetContentRegionAvail().X;
                bool newText = _logBuffer.Length > _lastLogLength;

                if (Math.Abs(width - _lastLogWidth) > 1.0f || newText)
                {
                    _cachedLogHeight = ImGui.CalcTextSize(_logBuffer, width).Y + ImGui.GetTextLineHeight() * 2;
                    _lastLogWidth = width;
                }

                // Ensure minimum height to fill the view if log is short
                float minHeight = ImGui.GetContentRegionAvail().Y;
                float height = Math.Max(minHeight, _cachedLogHeight);

                ImGui.InputTextMultiline("##LogBuffer", ref _logBuffer, maxLogChars, new System.Numerics.Vector2(-1, height),
                    ImGuiInputTextFlags.ReadOnly | ImGuiInputTextFlags.CallbackAlways, _logCallback);

                if (_autoScroll && newText)
                {
                    ImGui.SetScrollY(ImGui.GetScrollMaxY());
                }

                _lastLogLength = _logBuffer.Length;

                // Context Menu for Copy/Clear
                if (ImGui.BeginPopupContextItem("LogContext"))
                {
                    bool hasSelection = _savedSelectionStart != _savedSelectionEnd;

                    if (ImGui.MenuItem("Copy Selected", "", false, hasSelection))
                    {
                        CopySelectedLogText();
                    }
                    if (ImGui.MenuItem("Copy All"))
                    {
                        if (!string.IsNullOrEmpty(_logBuffer))
                        {
                            ClipboardString = _logBuffer;
                            Logger.Info("Log copied to clipboard via context menu.");
                        }
                    }
                    if (ImGui.MenuItem("Clear Log"))
                    {
                        _logBuffer = "";
                    }
                    ImGui.EndPopup();
                }

                ImGui.EndChild();
            }
            ImGui.End();
        }

        private void RenderInfoOverlay()
        {
            float padding = 10.0f;
            // Move overlay to the right side of the screen, accounting for the right panel if visible
            float xPos = ClientSize.X - (_showRightPanel ? _rightPanelWidth : 0) - 200 - padding;
            var windowPos = new System.Numerics.Vector2(xPos, GetTopUiHeight() + 10);

            ImGui.SetNextWindowPos(windowPos, ImGuiCond.Always);
            ImGui.SetNextWindowBgAlpha(0.35f); // Transparent background

            if (ImGui.Begin("InfoOverlay", ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.AlwaysAutoResize | ImGuiWindowFlags.NoSavedSettings | ImGuiWindowFlags.NoFocusOnAppearing | ImGuiWindowFlags.NoNav | ImGuiWindowFlags.NoMove))
            {
                float fps = _viewport.FPS;
                ImGui.Text($"FPS: {fps:F1}");
                ImGui.Separator();

                int objCount = _sceneGraph.GetVisibleObjects().Count();
                int selCount = _sceneGraph.SelectedObjects.Count;

                ImGui.Text($"Objects: {objCount}");
                if (selCount > 0)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1.0f, 0.8f, 0.2f, 1.0f), $"Selected: {selCount}");
                }

                // Gizmo mode
                string mode = _viewport.CurrentGizmoMode.ToString();
                ImGui.TextDisabled($"Mode: {mode}");

                if (_viewport.TryGetHoveredSourceCameraDistance(out float sourceDistance, out bool isMeters, out string sourceCameraName))
                {
                    ImGui.Separator();
                    ImGui.TextDisabled($"SrcCam: {sourceCameraName}");
                    string unit = isMeters ? "m" : "px";
                    ImGui.TextColored(new System.Numerics.Vector4(1.0f, 0.85f, 0.35f, 1.0f), $"Dist: {sourceDistance:F3} {unit}");
                }

                if (_viewport.TryGetPointCloudLegend(out var legendMode, out float minValue, out float maxValue, out bool depthFallback))
                {
                    ImGui.Separator();
                    string modeText = legendMode == PointCloudColorMode.Confidence
                        ? (depthFallback ? "Point Colors: Confidence (Depth Fallback)" : "Point Colors: Confidence")
                        : "Point Colors: Depth";
                    ImGui.TextDisabled(modeText);
                    DrawColormapLegend(140.0f, 10.0f, $"{minValue:F2}", $"{maxValue:F2}");
                }
            }
            ImGui.End();
        }

        #endregion

        #region Dialogs

        private void DrawSettingsWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(550, 650), ImGuiCond.FirstUseEver);

            if (ImGui.Begin("Settings", ref _showSettings))
            {
                var s = IniSettings.Instance;

                if (ImGui.BeginTabBar("SettingsTabs"))
                {
                    // --- General Settings ---
                    if (ImGui.BeginTabItem("General"))
                    {
                        ImGui.Spacing();
                        if (ImGui.CollapsingHeader("System & Compute", ImGuiTreeNodeFlags.DefaultOpen))
                        {
                            // Compute Device
                            int device = (int)s.Device;
                            string[] devices = Enum.GetNames(typeof(ComputeDevice));
                            if (ImGui.Combo("Meshing Device", ref device, devices, devices.Length))
                                s.Device = (ComputeDevice)device;

                            // Meshing Algorithm
                            int algo = (int)s.MeshingAlgo;
                            string[] algos = Enum.GetNames(typeof(MeshingAlgorithm));
                            if (ImGui.Combo("Default Meshing", ref algo, algos, algos.Length))
                                s.MeshingAlgo = (MeshingAlgorithm)algo;

                            // Coordinate System
                            int coord = (int)s.CoordSystem;
                            string[] coords = Enum.GetNames(typeof(CoordinateSystem));
                            if (ImGui.Combo("Coordinate System", ref coord, coords, coords.Length))
                                s.CoordSystem = (CoordinateSystem)coord;
                        }

                        ImGui.Spacing();
                        if (ImGui.CollapsingHeader("Reconstruction", ImGuiTreeNodeFlags.DefaultOpen))
                        {
                            int method = (int)s.ReconstructionMethod;
                            string[] methods = Enum.GetNames(typeof(ReconstructionMethod));
                            if (ImGui.Combo("Default Method", ref method, methods, methods.Length))
                                s.ReconstructionMethod = (ReconstructionMethod)method;

                            int bbox = (int)s.BoundingBoxStyle;
                            string[] bboxStyles = Enum.GetNames(typeof(BoundingBoxMode));
                            if (ImGui.Combo("Bounding Box", ref bbox, bboxStyles, bboxStyles.Length))
                                s.BoundingBoxStyle = (BoundingBoxMode)bbox;
                        }

                        ImGui.Spacing();
                        if (ImGui.CollapsingHeader("Viewport Appearance"))
                        {
                            // Background Color
                            var bg = new System.Numerics.Vector3(s.ViewportBgR, s.ViewportBgG, s.ViewportBgB);
                            if (ImGui.ColorEdit3("Background Color", ref bg))
                            {
                                s.ViewportBgR = bg.X; s.ViewportBgG = bg.Y; s.ViewportBgB = bg.Z;
                            }

                            // Grid Color
                            var gridCol = new System.Numerics.Vector3(s.GridColorR, s.GridColorG, s.GridColorB);
                            if (ImGui.ColorEdit3("Grid Color", ref gridCol))
                            {
                                s.GridColorR = gridCol.X; s.GridColorG = gridCol.Y; s.GridColorB = gridCol.Z;
                            }

                            bool grid = s.ShowGrid;
                            if (ImGui.Checkbox("Show Grid", ref grid)) s.ShowGrid = grid;

                            bool axes = s.ShowAxes;
                            if (ImGui.Checkbox("Show Axes", ref axes)) s.ShowAxes = axes;

                            bool cameras = s.ShowCameras;
                            if (ImGui.Checkbox("Show Cameras", ref cameras)) s.ShowCameras = cameras;

                            bool gizmo = s.ShowGizmo;
                            if (ImGui.Checkbox("Show Gizmo", ref gizmo)) s.ShowGizmo = gizmo;

                            bool info = s.ShowInfoOverlay;
                            if (ImGui.Checkbox("Show Info Overlay", ref info)) s.ShowInfoOverlay = info;
                        }

                        ImGui.EndTabItem();
                    }

                    // --- AI Models Settings ---
                    if (ImGui.BeginTabItem("AI Models"))
                    {
                        ImGui.Spacing();

                        // Global AI Settings
                        ImGui.TextColored(new System.Numerics.Vector4(0.4f, 0.8f, 1.0f, 1.0f), "Global AI Configuration");

                        int aiDevice = (int)s.AIDevice;
                        string[] aiDevices = Enum.GetNames(typeof(AIComputeDevice));
                        if (ImGui.Combo("AI Compute Device", ref aiDevice, aiDevices, aiDevices.Length))
                            s.AIDevice = (AIComputeDevice)aiDevice;

                        int img3d = (int)s.ImageTo3D;
                        string[] img3dModels = Enum.GetNames(typeof(ImageTo3DModel));
                        if (ImGui.Combo("Image-to-3D Model", ref img3d, img3dModels, img3dModels.Length))
                            s.ImageTo3D = (ImageTo3DModel)img3d;

                        int meshEx = (int)s.MeshExtraction;
                        string[] meshExMethods = Enum.GetNames(typeof(MeshExtractionMethod));
                        if (ImGui.Combo("Mesh Extraction", ref meshEx, meshExMethods, meshExMethods.Length))
                            s.MeshExtraction = (MeshExtractionMethod)meshEx;

                        ImGui.Separator();
                        ImGui.Spacing();

                        if (ImGui.CollapsingHeader("Dust3r (Multi-View)"))
                        {
                            ImGui.TextWrapped("Dust3r is the default multi-view reconstruction engine. It requires a GPU with significant VRAM for best performance.");
                            ImGui.Spacing();

                            string dust3rPath = s.Dust3rModelPath;
                            if (ImGui.InputText("Model Path##Dust3r", ref dust3rPath, 256)) s.Dust3rModelPath = dust3rPath;
                            ImGui.TextDisabled("Folder containing dust3r_weights.pth");
                        }

                        if (ImGui.CollapsingHeader("MASt3R (Metric Multi-View)"))
                        {
                            ImGui.TextWrapped("MASt3R provides metric 3D reconstruction with dense feature matching. Best for 2+ images requiring accurate scale.");
                            ImGui.Spacing();

                            string mast3rPath = s.Mast3rModelPath;
                            if (ImGui.InputText("Model Path##Mast3r", ref mast3rPath, 256)) s.Mast3rModelPath = mast3rPath;
                            ImGui.TextDisabled("Folder containing mast3r_weights.pth (models/mast3r/)");
                        }

                        if (ImGui.CollapsingHeader("MUSt3R (Video/Many Images)"))
                        {
                            ImGui.TextWrapped("MUSt3R is optimized for many images and video input. Supports 8-11 FPS video processing with memory mechanism.");
                            ImGui.Spacing();

                            string must3rPath = s.Must3rModelPath;
                            if (ImGui.InputText("Model Path##Must3r", ref must3rPath, 256)) s.Must3rModelPath = must3rPath;
                            ImGui.TextDisabled("Folder containing must3r_weights.pth (models/must3r/)");

                            ImGui.Spacing();
                            ImGui.Text("Video Settings:");

                            int maxFrames = s.Must3rMaxFrames;
                            if (ImGui.SliderInt("Max Frames", ref maxFrames, 10, 500)) s.Must3rMaxFrames = maxFrames;
                            ImGui.TextDisabled("Maximum frames to extract from video");

                            int frameInterval = s.Must3rFrameInterval;
                            if (ImGui.SliderInt("Frame Interval", ref frameInterval, 1, 30)) s.Must3rFrameInterval = frameInterval;
                            ImGui.TextDisabled("Extract every Nth frame");
                        }

                        if (ImGui.CollapsingHeader("SfM (Feature Matching)"))
                        {
                            ImGui.TextWrapped("Structure from Motion using OpenCV feature matching. Works without AI models but may be less accurate.");
                            ImGui.Spacing();
                            ImGui.TextDisabled("SfM uses SIFT/ORB features and doesn't require model downloads.");
                        }

                        if (ImGui.CollapsingHeader("TripoSR (Single Image)"))
                        {
                            int res = s.TripoSRResolution;
                            if (ImGui.SliderInt("Resolution", ref res, 128, 1024)) s.TripoSRResolution = res;

                            int mcRes = s.TripoSRMarchingCubesRes;
                            if (ImGui.SliderInt("Marching Cubes Res", ref mcRes, 32, 512)) s.TripoSRMarchingCubesRes = mcRes;

                            string path = s.TripoSRModelPath;
                            if (ImGui.InputText("Model Path##TripoSR", ref path, 256)) s.TripoSRModelPath = path;
                            ImGui.TextDisabled("Folder containing triposr_weights.pth");
                        }

                        if (ImGui.CollapsingHeader("LGM (Large Gaussian Model)"))
                        {
                            int flow = s.LGMFlowSteps;
                            if (ImGui.SliderInt("Flow Steps", ref flow, 10, 100)) s.LGMFlowSteps = flow;

                            int qRes = s.LGMQueryResolution;
                            if (ImGui.SliderInt("Query Resolution", ref qRes, 64, 512)) s.LGMQueryResolution = qRes;

                            int lgmRes = s.LGMResolution;
                            if (ImGui.SliderInt("Resolution", ref lgmRes, 256, 1024)) s.LGMResolution = lgmRes;

                            string path = s.LGMModelPath;
                            if (ImGui.InputText("Model Path##LGM", ref path, 256)) s.LGMModelPath = path;
                            ImGui.TextDisabled("Folder containing model_fp16_fixrot.safetensors");
                        }

                        if (ImGui.CollapsingHeader("Wonder3D"))
                        {
                            int steps = s.Wonder3DSteps;
                            if (ImGui.SliderInt("Steps", ref steps, 10, 100)) s.Wonder3DSteps = steps;

                            float guidance = s.Wonder3DGuidanceScale;
                            if (ImGui.SliderFloat("Guidance Scale", ref guidance, 1.0f, 10.0f)) s.Wonder3DGuidanceScale = guidance;

                            int diffSteps = s.Wonder3DDiffusionSteps;
                            if (ImGui.SliderInt("Diffusion Steps", ref diffSteps, 10, 100)) s.Wonder3DDiffusionSteps = diffSteps;

                            string path = s.Wonder3DModelPath;
                            if (ImGui.InputText("Model Path##Wonder3D", ref path, 256)) s.Wonder3DModelPath = path;
                            ImGui.TextDisabled("Folder containing model_index.json (models/wonder3d/)");
                        }

                        if (ImGui.CollapsingHeader("UniRig (Auto Rigging)"))
                        {
                            int rigMethod = (int)s.RiggingModel;
                            string[] rigMethods = Enum.GetNames(typeof(RiggingMethod));
                            if (ImGui.Combo("Rigging Method", ref rigMethod, rigMethods, rigMethods.Length))
                                s.RiggingModel = (RiggingMethod)rigMethod;

                            int joints = s.UniRigMaxJoints;
                            if (ImGui.SliderInt("Max Joints", ref joints, 16, 256)) s.UniRigMaxJoints = joints;

                            int bones = s.UniRigMaxBonesPerVertex;
                            if (ImGui.SliderInt("Bones Per Vertex", ref bones, 1, 8)) s.UniRigMaxBonesPerVertex = bones;

                            string path = s.UniRigModelPath;
                            if (ImGui.InputText("Model Path##UniRig", ref path, 256)) s.UniRigModelPath = path;
                            ImGui.TextDisabled("Folder containing unirig_weights.pth");
                        }

                        ImGui.EndTabItem();
                    }

                    // --- Refinement Settings ---
                    if (ImGui.BeginTabItem("Refinement"))
                    {
                        ImGui.Spacing();

                        int meshRefine = (int)s.MeshRefinement;
                        string[] meshRefineMethods = Enum.GetNames(typeof(MeshRefinementMethod));
                        if (ImGui.Combo("Mesh Refinement", ref meshRefine, meshRefineMethods, meshRefineMethods.Length))
                            s.MeshRefinement = (MeshRefinementMethod)meshRefine;

                        ImGui.Spacing();

                        if (ImGui.CollapsingHeader("DeepMeshPrior (Optimization)"))
                        {
                            ImGui.TextWrapped("Optimization of existing meshes using Graph Convolutional Networks.");

                            int iter = s.DeepMeshPriorIterations;
                            if (ImGui.InputInt("Iterations", ref iter)) s.DeepMeshPriorIterations = Math.Max(100, iter);

                            float lr = s.DeepMeshPriorLearningRate;
                            if (ImGui.InputFloat("Learning Rate", ref lr, 0.001f, 0.01f, "%.4f")) s.DeepMeshPriorLearningRate = lr;

                            float lap = s.DeepMeshPriorLaplacianWeight;
                            if (ImGui.InputFloat("Laplacian Weight", ref lap, 0.1f)) s.DeepMeshPriorLaplacianWeight = lap;
                        }

                        if (ImGui.CollapsingHeader("Gaussian SDF Refiner"))
                        {
                            int grid = s.GaussianSDFGridResolution;
                            if (ImGui.SliderInt("Grid Resolution", ref grid, 64, 512)) s.GaussianSDFGridResolution = grid;

                            float sigma = s.GaussianSDFSigma;
                            if (ImGui.SliderFloat("Sigma", ref sigma, 0.1f, 5.0f)) s.GaussianSDFSigma = sigma;

                            int iterations = s.GaussianSDFIterations;
                            if (ImGui.SliderInt("Iterations", ref iterations, 1, 10)) s.GaussianSDFIterations = iterations;

                            float iso = s.GaussianSDFIsoLevel;
                            if (ImGui.SliderFloat("Iso Level", ref iso, -1.0f, 1.0f)) s.GaussianSDFIsoLevel = iso;
                        }

                        if (ImGui.CollapsingHeader("TripoSF (Refinement)"))
                        {
                            int res = s.TripoSFResolution;
                            if (ImGui.SliderInt("Resolution", ref res, 256, 1024)) s.TripoSFResolution = res;

                            int dil = s.TripoSFSparseDilation;
                            if (ImGui.SliderInt("Sparse Dilation", ref dil, 0, 5)) s.TripoSFSparseDilation = dil;

                            string path = s.TripoSFModelPath;
                            if (ImGui.InputText("Model Path##TripoSF", ref path, 256)) s.TripoSFModelPath = path;
                        }

                        if (ImGui.CollapsingHeader("Point Cloud Merger"))
                        {
                            float vox = s.MergerVoxelSize;
                            if (ImGui.InputFloat("Voxel Size", ref vox, 0.001f)) s.MergerVoxelSize = Math.Max(0.001f, vox);

                            int iter = s.MergerMaxIterations;
                            if (ImGui.InputInt("Max Iterations", ref iter)) s.MergerMaxIterations = iter;

                            float outlier = s.MergerOutlierThreshold;
                            if (ImGui.SliderFloat("Outlier Threshold", ref outlier, 0.5f, 5.0f)) s.MergerOutlierThreshold = outlier;
                        }

                        if (ImGui.CollapsingHeader("NeRF (Legacy)"))
                        {
                            bool unlimited = s.NeRFUnlimited;
                            if (ImGui.Checkbox("Unlimited (Stop to finish)", ref unlimited))
                            {
                                s.NeRFUnlimited = unlimited;
                            }

                            int iter = s.NeRFIterations;
                            if (s.NeRFUnlimited)
                            {
                                ImGui.BeginDisabled();
                            }
                            if (ImGui.InputInt("Iterations", ref iter))
                            {
                                s.NeRFIterations = Math.Clamp(iter, 1, 500);
                            }
                            if (s.NeRFUnlimited)
                            {
                                ImGui.EndDisabled();
                            }

                            int grid = s.VoxelGridSize;
                            if (ImGui.InputInt("Voxel Grid Size", ref grid)) s.VoxelGridSize = grid;

                            float lr = s.NeRFLearningRate;
                            if (ImGui.InputFloat("Learning Rate", ref lr, 0.01f)) s.NeRFLearningRate = lr;
                        }

                        ImGui.EndTabItem();
                    }

                    ImGui.EndTabBar();
                }

                ImGui.Separator();

                // Bottom buttons
                if (ImGui.Button("Save Settings", new System.Numerics.Vector2(120, 30)))
                {
                    s.Save();
                    _logBuffer += "Settings saved.\n";
                }
                ImGui.SameLine();
                if (ImGui.Button("Reset to Defaults", new System.Numerics.Vector2(140, 30)))
                {
                    s.Reset();
                }

                // Tech Info at bottom
                ImGui.Spacing();
                ImGui.Separator();
                ImGui.TextDisabled($"OpenGL: {GL.GetString(StringName.Version)}");
                ImGui.TextDisabled($"Renderer: {GL.GetString(StringName.Renderer)}");
            }
            ImGui.End();
        }

        private void DrawAboutWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(400, 500), ImGuiCond.Always);

            if (ImGui.Begin("About Deep3DStudio", ref _showAbout, ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize))
            {
                // Center content
                float windowWidth = ImGui.GetWindowWidth();

                if (_logoTexture != -1)
                {
                    float logoSize = 128;
                    ImGui.SetCursorPosX((windowWidth - logoSize) * 0.5f);
                    ImGui.Image((IntPtr)_logoTexture, new System.Numerics.Vector2(logoSize, logoSize));
                }

                ImGui.Spacing();

                // Center title
                string title = "Deep3DStudio";
                var titleSize = ImGui.CalcTextSize(title);
                ImGui.SetCursorPosX((windowWidth - titleSize.X) * 0.5f);
                ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(0.9f, 0.9f, 0.95f, 1.0f));
                ImGui.Text(title);
                ImGui.PopStyleColor();

                // Subtitle
                string subtitle = "Cross-Platform Edition";
                var subSize = ImGui.CalcTextSize(subtitle);
                ImGui.SetCursorPosX((windowWidth - subSize.X) * 0.5f);
                ImGui.TextDisabled(subtitle);

                ImGui.Spacing();

                string version = GetAppVersionText();
                var verSize = ImGui.CalcTextSize(version);
                ImGui.SetCursorPosX((windowWidth - verSize.X) * 0.5f);
                ImGui.Text(version);

                ImGui.Separator();

                ImGui.TextWrapped("A Neural Rendering & Reconstruction Studio for creating 3D models from images using AI.");

                ImGui.Spacing();
                ImGui.Text("Powered by:");
                ImGui.BulletText("OpenTK & ImGui.NET");
                ImGui.BulletText("SkiaSharp");
                ImGui.BulletText("Dust3r, TripoSR, LGM AI Models");

                ImGui.Separator();
                ImGui.Text("Author:");
                ImGui.TextWrapped("Matteo Mangiagalli - m.mangiagalli@campus.uniurb.it");
                ImGui.TextWrapped("Università degli Studi di Urbino - Carlo Bo");
                ImGui.Text("2026");

                ImGui.Separator();

                float buttonWidth = 120;
                ImGui.SetCursorPosX((windowWidth - buttonWidth) * 0.5f);
                if (ImGui.Button("Close", new System.Numerics.Vector2(buttonWidth, 30)))
                {
                    _showAbout = false;
                }
            }
            ImGui.End();
        }

        private void DrawImagePreviewWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(600, 500), ImGuiCond.FirstUseEver);

            if (ImGui.Begin($"Image Preview - {Path.GetFileName(_previewImagePath)}###ImagePreview", ref _showImagePreview))
            {
                if (_previewTexture > 0)
                {
                    var avail = ImGui.GetContentRegionAvail();
                    if (_previewShowsDepth)
                    {
                        float legendHeight = 28.0f;
                        var imageSize = new System.Numerics.Vector2(Math.Max(10.0f, avail.X), Math.Max(10.0f, avail.Y - legendHeight));
                        ImGui.Image((IntPtr)_previewTexture, imageSize);
                        float legendWidth = Math.Max(80.0f, Math.Min(220.0f, ImGui.GetContentRegionAvail().X - 70.0f));
                        DrawColormapLegend(legendWidth, 12.0f, "Near", "Far");
                    }
                    else
                    {
                        ImGui.Image((IntPtr)_previewTexture, new System.Numerics.Vector2(Math.Max(10.0f, avail.X), Math.Max(10.0f, avail.Y)));
                    }
                }
                else
                {
                    ImGui.Text("Loading image...");
                }
            }
            ImGui.End();

            if (!_showImagePreview && _previewTexture > 0)
            {
                TextureLoader.DeleteTexture(_previewTexture);
                _previewTexture = -1;
                _previewShowsDepth = false;
            }
        }

        #endregion

        #region UI Helpers

        private void DrawToggleBtn(string id, IconType icon, bool active, Action<bool> setter, string tooltip, System.Numerics.Vector2 size)
        {
            if (active)
            {
                ImGui.PushStyleColor(ImGuiCol.Button, new System.Numerics.Vector4(0.3f, 0.6f, 0.3f, 1f));
                ImGui.PushStyleColor(ImGuiCol.ButtonHovered, new System.Numerics.Vector4(0.35f, 0.65f, 0.35f, 1f));
            }

            if (ImGui.ImageButton(id, _iconFactory.GetIcon(icon), size))
            {
                setter(!active);
            }

            if (active) ImGui.PopStyleColor(2);
            if (ImGui.IsItemHovered()) ImGui.SetTooltip(tooltip);
        }

        // Store window state before fullscreen for restoration
        private OpenTK.Windowing.Common.WindowState _previousWindowState = OpenTK.Windowing.Common.WindowState.Normal;
        private OpenTK.Mathematics.Vector2i _previousWindowSize;
        private OpenTK.Mathematics.Vector2i _previousWindowLocation;

        /// <summary>
        /// Toggles fullscreen mode. Works on Windows, macOS, and Linux.
        /// Saves window state before entering fullscreen for proper restoration.
        /// </summary>
        private void ToggleFullscreen()
        {
            if (WindowState == OpenTK.Windowing.Common.WindowState.Fullscreen)
            {
                // Exit fullscreen - restore previous state
                WindowState = _previousWindowState;
                if (_previousWindowState == OpenTK.Windowing.Common.WindowState.Normal)
                {
                    // Restore window size and position
                    Size = _previousWindowSize;
                    Location = _previousWindowLocation;
                }
                _logBuffer += "Exited fullscreen mode.\n";
            }
            else
            {
                // Save current state before going fullscreen
                _previousWindowState = WindowState;
                _previousWindowSize = Size;
                _previousWindowLocation = Location;

                // Enter fullscreen
                WindowState = OpenTK.Windowing.Common.WindowState.Fullscreen;
                _logBuffer += "Entered fullscreen mode. Press F11 or click the button to exit.\n";
            }
        }

        #endregion

        #region Project Operations

        private void OnNewProject()
        {
            if (!EnsureCanProceedWithUnsavedChanges(PendingUnsavedAction.NewProject))
                return;

            PerformNewProject();
        }

        private void PerformNewProject()
        {
            _sceneGraph.Clear();
            ClearImages();
            GeoReferenceRuntime.Clear();
            _logBuffer = "";
            _currentProjectPath = "";
            _isDirty = false;
            UpdateTitle();
            _logBuffer += "New project created.\n";
        }

        private void OnOpenProject(string? path = null)
        {
            if (!EnsureCanProceedWithUnsavedChanges(PendingUnsavedAction.OpenProject, path))
                return;

            PerformOpenProject(path);
        }

        private void PerformOpenProject(string? path = null)
        {
            if (path == null)
            {
                var result = Nfd.OpenDialog(out path, new Dictionary<string, string> { { "Deep3D Project", "d3d" } });
                if (result != NfdStatus.Ok || string.IsNullOrEmpty(path)) return;
            }

            ProgressDialog.Instance.Start("Loading Project...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    var state = CrossProjectManager.LoadProject(path);

                    // We must be careful with GL operations on background thread.
                    // Ideally we should enqueue this to the main thread.
                    EnqueueAction(() => {
                        try
                        {
                            ClearImages();
                            CrossProjectManager.RestoreSceneFromState(state, _sceneGraph);
                            GeoReferenceRuntime.LoadFromState(state);

                            // Restore images
                            if (state.Images != null && state.Images.Count > 0)
                            {
                                foreach (var pImg in state.Images)
                                {
                                    if (File.Exists(pImg.FilePath))
                                    {
                                        if (!_loadedImages.Any(x => x.FilePath == pImg.FilePath))
                                        {
                                            _loadedImages.Add(pImg);
                                            // Thumbnail
                                            try
                                            {
                                                var thumb = TextureLoader.CreateThumbnail(pImg.FilePath, 64);
                                                if (thumb > 0) lock (_imageThumbnails) _imageThumbnails[pImg.FilePath] = thumb;
                                            }
                                            catch { }
                                        }
                                    }
                                }
                            }
                            else if (state.ImagePaths != null) // Legacy fallback
                            {
                                foreach (var img in state.ImagePaths)
                                {
                                    if (File.Exists(img))
                                    {
                                        if (!_loadedImages.Any(x => x.FilePath == img))
                                        {
                                            _loadedImages.Add(new ProjectImage { FilePath = img, Alias = Path.GetFileName(img) });
                                            try
                                            {
                                                var thumb = TextureLoader.CreateThumbnail(img, 64);
                                                if (thumb > 0) lock (_imageThumbnails) _imageThumbnails[img] = thumb;
                                            }
                                            catch { }
                                        }
                                    }
                                }
                            }

                            _currentProjectPath = path;
                            _isDirty = false;
                            UpdateTitle();
                            ProgressDialog.Instance.Log($"Project loaded: {path}");
                            ProgressDialog.Instance.Complete();
                        }
                        catch(Exception innerEx)
                        {
                            ProgressDialog.Instance.Fail(innerEx);
                        }
                    });
                }
                catch (Exception ex)
                {
                    ProgressDialog.Instance.Fail(ex);
                }
            });
        }

        private bool EnsureCanProceedWithUnsavedChanges(PendingUnsavedAction action, string? openProjectPath = null)
        {
            if (!_isDirty)
                return true;

            _pendingUnsavedAction = action;
            _pendingOpenProjectPath = openProjectPath;
            _showUnsavedChangesPrompt = true;
            return false;
        }

        private void ExecutePendingUnsavedActionAndClear()
        {
            var action = _pendingUnsavedAction;
            var openPath = _pendingOpenProjectPath;
            ClearPendingUnsavedAction();

            switch (action)
            {
                case PendingUnsavedAction.Exit:
                    Close();
                    break;
                case PendingUnsavedAction.NewProject:
                    PerformNewProject();
                    break;
                case PendingUnsavedAction.OpenProject:
                    PerformOpenProject(openPath);
                    break;
            }
        }

        private void ClearPendingUnsavedAction()
        {
            _pendingUnsavedAction = PendingUnsavedAction.None;
            _pendingOpenProjectPath = null;
        }

        private void OnSaveProject(bool exitAfter = false)
        {
            if (string.IsNullOrEmpty(_currentProjectPath))
            {
                OnSaveProjectAs(exitAfter);
                return;
            }

            ProgressDialog.Instance.Start("Saving Project...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    lock (_sceneGraph)
                    {
                        CrossProjectManager.SaveProject(_currentProjectPath, _sceneGraph, _loadedImages);
                    }

                    EnqueueAction(() => {
                        _isDirty = false;
                        UpdateTitle();
                        ProgressDialog.Instance.Log($"Project saved: {_currentProjectPath}");
                        ProgressDialog.Instance.Complete();
                        if (exitAfter)
                        {
                            Close();
                        }
                        else if (_executePendingActionAfterSave)
                        {
                            _executePendingActionAfterSave = false;
                            ExecutePendingUnsavedActionAndClear();
                        }
                    });
                }
                catch (Exception ex)
                {
                    ProgressDialog.Instance.Fail(ex);
                }
            });
        }

        private void OnSaveProjectAs(bool exitAfter = false)
        {
            var result = Nfd.SaveDialog(out string path, new Dictionary<string, string> { { "Deep3D Project", "d3d" } });
            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                if (!path.EndsWith(".d3d")) path += ".d3d";
                _currentProjectPath = path;
                OnSaveProject(exitAfter);
            }
            else
            {
                _executePendingActionAfterSave = false;
                ClearPendingUnsavedAction();
            }
        }

        private void EnqueueAction(Action action)
        {
            _pendingActions.Enqueue(action);
        }

        #endregion

        #region File Operations

        private void OnAddImages()
        {
            Logger.Info("OnAddImages: Opening file dialog...");
            var result = Nfd.OpenDialogMultiple(out string[] paths, new Dictionary<string, string>
            {
                { "Images", "jpg,jpeg,png,bmp,tif,tiff" }
            });

            Logger.Debug($"OnAddImages: Dialog result: {result}");
            if (result == NfdStatus.Ok && paths != null)
            {
                Logger.Info($"OnAddImages: {paths.Length} file(s) selected");
                foreach (var path in paths)
                {
                    ImportFile(path);
                }
            }
            else
            {
                Logger.Debug("OnAddImages: Dialog cancelled or no files selected");
            }
        }

        private void OnImportMesh()
        {
            var result = Nfd.OpenDialog(out string path, new Dictionary<string, string>
            {
                { "3D Mesh", "obj,ply,stl,glb" }
            });

            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                ImportFile(path);
            }
        }

        private void OnImportPointCloud()
        {
            var result = Nfd.OpenDialog(out string path, new Dictionary<string, string>
            {
                { "Point Cloud", "ply,xyz" }
            });

            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                ImportFile(path);
            }
        }

        private void LoadVideoFile()
        {
            var result = Nfd.OpenDialog(out string path, new Dictionary<string, string>
            {
                { "Video Files", "mp4,avi,mov,mkv,webm" }
            });

            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                _videoFilePath = path;
                _hasVideoInput = true;
                _logBuffer += $"[MUSt3R] Video loaded: {Path.GetFileName(path)}\n";

                // Auto-select MUSt3R as reconstruction method and first workflow when video is loaded
                IniSettings.Instance.ReconstructionMethod = ReconstructionMethod.Must3r;
                _selectedWorkflow = 0; // Multi-View (will use MUSt3R from settings)
            }
        }

        private void OnExportMesh()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "No mesh selected for export.\n";
                return;
            }

            var result = Nfd.SaveDialog(out string path, new Dictionary<string, string>
            {
                { "OBJ Mesh", "obj" },
                { "PLY Mesh", "ply" },
                { "STL Mesh", "stl" }
            });

            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                bool applyGeo = GeoReferenceRuntime.HasActiveGeoreference;
                ProgressDialog.Instance.Start("Exporting Mesh...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var mesh = applyGeo
                            ? GeoExportService.PrepareMeshForExport(meshes[0])
                            : meshes[0].MeshData;
                        MeshExporter.Save(path, mesh);
                        if (applyGeo)
                            GeoExportService.TryWriteGeoSidecar(path, mesh.Vertices);
                        ProgressDialog.Instance.Log($"Mesh exported: {path}");
                        ProgressDialog.Instance.Complete();
                    }
                    catch (Exception ex)
                    {
                        ProgressDialog.Instance.Fail(ex);
                    }
                });
            }
        }

        private void OnExportPointCloud()
        {
            var pcs = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (pcs.Count == 0)
            {
                _logBuffer += "No point cloud selected for export.\n";
                return;
            }

            var result = Nfd.SaveDialog(out string path, new Dictionary<string, string>
            {
                { "PLY Point Cloud", "ply" },
                { "XYZ Point Cloud", "xyz" }
            });

            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                bool applyGeo = GeoReferenceRuntime.HasActiveGeoreference;
                ProgressDialog.Instance.Start("Exporting Point Cloud...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var pc = applyGeo
                            ? GeoExportService.PreparePointCloudForExport(pcs[0])
                            : pcs[0];
                        PointCloudExporter.Save(path, pc);
                        if (applyGeo)
                            GeoExportService.TryWriteGeoSidecar(path, pc.Points);
                        ProgressDialog.Instance.Log($"Point cloud exported: {path}");
                        ProgressDialog.Instance.Complete();
                    }
                    catch (Exception ex)
                    {
                        ProgressDialog.Instance.Fail(ex);
                    }
                });
            }
        }

        #endregion

        #region Edit Operations

        private void OnDeleteSelected()
        {
            var selected = _sceneGraph.SelectedObjects.ToList();
            foreach (var obj in selected)
            {
                _sceneGraph.RemoveObject(obj);
            }
            if (selected.Count > 0)
                _logBuffer += $"Deleted {selected.Count} object(s).\n";
        }

        private void OnDuplicateSelected()
        {
            var selected = _sceneGraph.SelectedObjects.ToList();
            foreach (var obj in selected)
            {
                OnDuplicateObject(obj);
            }
        }

        private void OnDuplicateObject(SceneObject obj)
        {
            var clone = obj.Clone();
            clone.Position += new Vector3(0.5f, 0, 0);
            _sceneGraph.AddObject(clone);
            _logBuffer += $"Duplicated: {obj.Name}\n";
        }

        private void OnResetTransform()
        {
            foreach (var obj in _sceneGraph.SelectedObjects)
            {
                obj.Position = Vector3.Zero;
                obj.Rotation = Vector3.Zero;
                obj.Scale = Vector3.One;
            }
        }

        private void OpenTransformDialog(TransformDialogMode mode)
        {
            _transformDialogMode = mode;
            _transformDialogValue = mode == TransformDialogMode.Scale
                ? new System.Numerics.Vector3(1f, 1f, 1f)
                : System.Numerics.Vector3.Zero;
            _showTransformDialog = true;
            _popupToOpen = "Transform Objects";
        }

        private void DrawTransformDialog()
        {
            if (!_showTransformDialog) return;
            if (ImGui.BeginPopup("Transform Objects", ImGuiWindowFlags.AlwaysAutoResize))
            {
                var selected = _sceneGraph.SelectedObjects.ToList();
                bool hasSelection = selected.Count > 0;

                string modeLabel = _transformDialogMode switch
                {
                    TransformDialogMode.Move => "Move (delta world units)",
                    TransformDialogMode.Rotate => "Rotate (delta degrees)",
                    TransformDialogMode.Scale => "Scale (multiplicative factors)",
                    _ => "Transform"
                };

                ImGui.Text(modeLabel);
                ImGui.InputFloat3("X / Y / Z", ref _transformDialogValue);
                ImGui.Separator();

                if (!hasSelection)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No object selected");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showTransformDialog = false;
                    ApplyTransformDialogValue(selected, _transformDialogValue);
                }

                if (!hasSelection)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0)))
                {
                    _showTransformDialog = false;
                }

                ImGui.EndPopup();
            }
        }

        private void ApplyTransformDialogValue(List<SceneObject> selected, System.Numerics.Vector3 value)
        {
            if (selected.Count == 0)
                return;

            var delta = new Vector3(value.X, value.Y, value.Z);
            switch (_transformDialogMode)
            {
                case TransformDialogMode.Move:
                    foreach (var obj in selected)
                        obj.Position += delta;
                    _logBuffer += $"Moved {selected.Count} object(s): ({delta.X:F3}, {delta.Y:F3}, {delta.Z:F3}).\n";
                    break;

                case TransformDialogMode.Rotate:
                    foreach (var obj in selected)
                        obj.Rotation += delta;
                    _logBuffer += $"Rotated {selected.Count} object(s): ({delta.X:F2}, {delta.Y:F2}, {delta.Z:F2}) deg.\n";
                    break;

                case TransformDialogMode.Scale:
                    var factor = ClampScale(delta);
                    foreach (var obj in selected)
                    {
                        obj.Scale = ClampScale(new Vector3(
                            obj.Scale.X * factor.X,
                            obj.Scale.Y * factor.Y,
                            obj.Scale.Z * factor.Z));
                    }
                    _logBuffer += $"Scaled {selected.Count} object(s): ({factor.X:F3}, {factor.Y:F3}, {factor.Z:F3}).\n";
                    break;
            }

            _isDirty = true;
            UpdateTitle();
        }

        private static Vector3 ClampScale(Vector3 scale)
        {
            const float minScale = 0.001f;
            return new Vector3(
                Math.Max(minScale, scale.X),
                Math.Max(minScale, scale.Y),
                Math.Max(minScale, scale.Z));
        }

        #endregion

        #region Mesh Operations

        private void OnCreatePrimitive(MeshPrimitiveType type)
        {
            var mesh = CreatePrimitiveFromPreset(type);
            string name = type switch
            {
                MeshPrimitiveType.UVSphere => "UV Sphere",
                _ => type.ToString()
            };

            var obj = new MeshObject(name, mesh);
            _sceneGraph.AddObject(obj);
            _sceneGraph.ClearSelection();
            _sceneGraph.Select(obj);
            _viewport.FocusOnSelection();
            _isDirty = true;
            UpdateTitle();
            _logBuffer += $"Created primitive: {name}\n";
        }

        private MeshData CreatePrimitiveFromPreset(MeshPrimitiveType type)
        {
            return type switch
            {
                MeshPrimitiveType.Plane => MeshPrimitiveFactory.CreatePlane(_primSize, _primSize),
                MeshPrimitiveType.Cube => MeshPrimitiveFactory.CreateCube(_primSize),
                MeshPrimitiveType.UVSphere => MeshPrimitiveFactory.CreateUVSphere(_primRadius, _primSegments, _primRings),
                MeshPrimitiveType.Cylinder => MeshPrimitiveFactory.CreateCylinder(_primRadius, _primHeight, _primSegments, _primCapEnds),
                MeshPrimitiveType.Cone => MeshPrimitiveFactory.CreateCone(_primRadius, _primHeight, _primSegments, _primCapEnds),
                MeshPrimitiveType.Torus => MeshPrimitiveFactory.CreateTorus(Math.Max(_primRadius, 0.05f), Math.Max(_primRadius * 0.35f, 0.02f), _primSegments, _primMinorSegments),
                MeshPrimitiveType.Circle => MeshPrimitiveFactory.CreateCircle(Math.Max(_primRadius, 0.01f), _primSegments),
                MeshPrimitiveType.Polygon => MeshPrimitiveFactory.CreatePolygon(_primPolygonSides, Math.Max(_primRadius, 0.01f)),
                MeshPrimitiveType.Grid => MeshPrimitiveFactory.CreateGrid(_primGridCells, Math.Max(_primCellSize, 0.001f)),
                _ => MeshPrimitiveFactory.CreatePrimitive(type)
            };
        }

        private void OpenPrimitiveOptionsDialog(MeshPrimitiveType type)
        {
            _primitiveDialogType = type;
            _showPrimitiveDialog = true;
            _popupToOpen = "Primitive Options";
        }

        private void DrawPrimitiveDialog()
        {
            if (!_showPrimitiveDialog) return;
            if (ImGui.BeginPopup("Primitive Options", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.Text($"Primitive: {_primitiveDialogType}");
                ImGui.Separator();

                switch (_primitiveDialogType)
                {
                    case MeshPrimitiveType.Plane:
                    case MeshPrimitiveType.Cube:
                        ImGui.InputFloat("Size", ref _primSize, 0.01f, 0.1f, "%.3f");
                        break;
                    case MeshPrimitiveType.UVSphere:
                        ImGui.InputFloat("Radius", ref _primRadius, 0.01f, 0.1f, "%.3f");
                        ImGui.InputInt("Segments", ref _primSegments);
                        ImGui.InputInt("Rings", ref _primRings);
                        break;
                    case MeshPrimitiveType.Cylinder:
                    case MeshPrimitiveType.Cone:
                        ImGui.InputFloat("Radius", ref _primRadius, 0.01f, 0.1f, "%.3f");
                        ImGui.InputFloat("Height", ref _primHeight, 0.01f, 0.1f, "%.3f");
                        ImGui.InputInt("Segments", ref _primSegments);
                        ImGui.Checkbox("Cap Ends", ref _primCapEnds);
                        break;
                    case MeshPrimitiveType.Torus:
                        ImGui.InputFloat("Major Radius", ref _primRadius, 0.01f, 0.1f, "%.3f");
                        ImGui.InputInt("Major Segments", ref _primSegments);
                        ImGui.InputInt("Minor Segments", ref _primMinorSegments);
                        break;
                    case MeshPrimitiveType.Circle:
                        ImGui.InputFloat("Radius", ref _primRadius, 0.01f, 0.1f, "%.3f");
                        ImGui.InputInt("Segments", ref _primSegments);
                        break;
                    case MeshPrimitiveType.Polygon:
                        ImGui.InputFloat("Radius", ref _primRadius, 0.01f, 0.1f, "%.3f");
                        ImGui.InputInt("Sides", ref _primPolygonSides);
                        break;
                    case MeshPrimitiveType.Grid:
                        ImGui.InputInt("Cells Per Side", ref _primGridCells);
                        ImGui.InputFloat("Cell Size", ref _primCellSize, 0.01f, 0.1f, "%.3f");
                        break;
                }

                _primSegments = Math.Clamp(_primSegments, 3, 256);
                _primRings = Math.Clamp(_primRings, 3, 256);
                _primMinorSegments = Math.Clamp(_primMinorSegments, 3, 256);
                _primPolygonSides = Math.Clamp(_primPolygonSides, 3, 64);
                _primGridCells = Math.Clamp(_primGridCells, 1, 512);
                _primSize = Math.Max(_primSize, 0.001f);
                _primRadius = Math.Max(_primRadius, 0.001f);
                _primHeight = Math.Max(_primHeight, 0.001f);
                _primCellSize = Math.Max(_primCellSize, 0.001f);

                ImGui.Separator();
                if (ImGui.Button("Create", new System.Numerics.Vector2(120, 0)))
                {
                    _showPrimitiveDialog = false;
                    OnCreatePrimitive(_primitiveDialogType);
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPrimitiveDialog = false;

                ImGui.EndPopup();
            }
        }

        private void OpenPenMoveOptionsDialog()
        {
            _showPenMoveDialog = true;
            _popupToOpen = "Pen Move";
        }

        private void OpenPenExtrudeOptionsDialog()
        {
            _showPenExtrudeDialog = true;
            _popupToOpen = "Pen Extrude";
        }

        private void OpenPenInsetOptionsDialog()
        {
            _showPenInsetDialog = true;
            _popupToOpen = "Pen Inset";
        }

        private void DrawPenExtrudeDialog()
        {
            if (!_showPenExtrudeDialog) return;
            if (ImGui.BeginPopup("Pen Extrude", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat("Extrude Distance", ref _penExtrudeDistance, 0.01f, 0.1f, "%.4f");
                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenExtrudeDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenExtrudeDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void DrawPenMoveDialog()
        {
            if (!_showPenMoveDialog) return;
            if (ImGui.BeginPopup("Pen Move", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat3("Move Delta", ref _penMoveDelta);
                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenMoveDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenMoveDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void DrawPenInsetDialog()
        {
            if (!_showPenInsetDialog) return;
            if (ImGui.BeginPopup("Pen Inset", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.SliderFloat("Inset Amount", ref _penInsetAmount, 0.01f, 0.95f, "%.3f");
                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenInsetDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showPenInsetDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void OpenRunOptionsDialog()
        {
            _showRunOptionsDialog = true;
            _popupToOpen = "Run Options";
        }

        private void OpenMeshingOptionsDialog()
        {
            _showMeshingOptionsDialog = true;
            _popupToOpen = "Meshing Options";
        }

        private void OpenRefinementOptionsDialog()
        {
            _showRefinementOptionsDialog = true;
            _popupToOpen = "Refinement Options";
        }

        private void OpenPointCloudVoxelOptionsDialog()
        {
            _showPcVoxelDialog = true;
            _popupToOpen = "Point Cloud Voxel";
        }

        private void OpenPointCloudOutlierOptionsDialog()
        {
            _showPcOutliersDialog = true;
            _popupToOpen = "Point Cloud Outliers";
        }

        private void OpenPointCloudDuplicateOptionsDialog()
        {
            _showPcDuplicatesDialog = true;
            _popupToOpen = "Point Cloud Duplicates";
        }

        private void OpenPointCloudSkyBlueOptionsDialog()
        {
            _showPcSkyBlueDialog = true;
            _popupToOpen = "Point Cloud Sky/Blue";
        }

        private void OpenPointCloudNormalOptionsDialog()
        {
            _showPcNormalsDialog = true;
            _popupToOpen = "Point Cloud Normals";
        }

        private void OpenPointCloudPassOptionsDialog()
        {
            _showPcPassDialog = true;
            _popupToOpen = "Point Cloud Pass";
        }

        private void OpenPointCloudRadiusOptionsDialog()
        {
            _showPcRadiusDialog = true;
            _popupToOpen = "Point Cloud Radius";
        }

        private void OpenPointCloudDenseOptionsDialog()
        {
            _showPcDenseDialog = true;
            _popupToOpen = "Point Cloud Dense";
        }

        private void DrawRunOptionsDialog()
        {
            if (!_showRunOptionsDialog) return;
            if (ImGui.BeginPopup("Run Options", ImGuiWindowFlags.AlwaysAutoResize))
            {
                var settings = IniSettings.Instance;

                bool auto = _autoWorkflowEnabled;
                if (ImGui.Checkbox("Auto Workflow", ref auto))
                {
                    _autoWorkflowEnabled = auto;
                }

                int rec = (int)settings.ReconstructionMethod;
                string[] methods = Enum.GetNames(typeof(ReconstructionMethod));
                if (ImGui.Combo("Reconstruction Method", ref rec, methods, methods.Length))
                {
                    settings.ReconstructionMethod = (ReconstructionMethod)rec;
                }

                var workflowNames = GetWorkflowNames();
                ImGui.Combo("Workflow Preset", ref _selectedWorkflow, workflowNames, workflowNames.Length);

                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showRunOptionsDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showRunOptionsDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void DrawMeshingOptionsDialog()
        {
            if (!_showMeshingOptionsDialog) return;
            if (ImGui.BeginPopup("Meshing Options", ImGuiWindowFlags.AlwaysAutoResize))
            {
                var settings = IniSettings.Instance;
                int algo = (int)settings.MeshingAlgo;
                string[] algos = Enum.GetNames(typeof(MeshingAlgorithm));
                if (ImGui.Combo("Meshing Algorithm", ref algo, algos, algos.Length))
                {
                    settings.MeshingAlgo = (MeshingAlgorithm)algo;
                }

                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showMeshingOptionsDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showMeshingOptionsDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void DrawRefinementOptionsDialog()
        {
            if (!_showRefinementOptionsDialog) return;
            if (ImGui.BeginPopup("Refinement Options", ImGuiWindowFlags.AlwaysAutoResize))
            {
                var settings = IniSettings.Instance;
                int method = (int)settings.MeshRefinement;
                string[] methods = Enum.GetNames(typeof(MeshRefinementMethod));
                if (ImGui.Combo("Refinement Method", ref method, methods, methods.Length))
                {
                    settings.MeshRefinement = (MeshRefinementMethod)method;
                }

                ImGui.Separator();
                if (ImGui.Button("Apply", new System.Numerics.Vector2(120, 0)))
                {
                    _showRefinementOptionsDialog = false;
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                {
                    _showRefinementOptionsDialog = false;
                }
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudVoxelDialog()
        {
            if (!_showPcVoxelDialog) return;
            if (ImGui.BeginPopup("Point Cloud Voxel", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat("Voxel Size", ref _pcVoxelSize, 0.001f, 0.01f, "%.4f");
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcVoxelDialog = false;
                    ApplyPointCloudVoxel();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcVoxelDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudOutliersDialog()
        {
            if (!_showPcOutliersDialog) return;
            if (ImGui.BeginPopup("Point Cloud Outliers", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputInt("K Neighbors", ref _pcOutlierK);
                ImGui.SliderFloat("Std Ratio", ref _pcOutlierStdRatio, 0.1f, 10f, "%.2f");
                _pcOutlierK = Math.Clamp(_pcOutlierK, 2, 200);
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcOutliersDialog = false;
                    ApplyPointCloudOutliers();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcOutliersDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudDuplicatesDialog()
        {
            if (!_showPcDuplicatesDialog) return;
            if (ImGui.BeginPopup("Point Cloud Duplicates", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat("Distance Threshold", ref _pcDuplicateThreshold, 0.0001f, 0.001f, "%.6f");
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcDuplicatesDialog = false;
                    ApplyPointCloudDuplicates();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcDuplicatesDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudSkyBlueDialog()
        {
            if (!_showPcSkyBlueDialog) return;
            if (ImGui.BeginPopup("Point Cloud Sky/Blue", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.SliderFloat("Min Blue", ref _pcSkyMinBlue, 0.0f, 1.0f, "%.3f");
                ImGui.SliderFloat("Max Red", ref _pcSkyMaxRed, 0.0f, 1.0f, "%.3f");
                ImGui.SliderFloat("Max Green", ref _pcSkyMaxGreen, 0.0f, 1.0f, "%.3f");
                ImGui.SliderFloat("Min Blue Dominance", ref _pcSkyBlueDominance, 0.0f, 1.0f, "%.3f");
                _pcSkyBlueDominance = Math.Max(0.0f, _pcSkyBlueDominance);

                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcSkyBlueDialog = false;
                    ApplyPointCloudSkyBlue();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcSkyBlueDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudNormalsDialog()
        {
            if (!_showPcNormalsDialog) return;
            if (ImGui.BeginPopup("Point Cloud Normals", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputInt("K Neighbors", ref _pcNormalK);
                _pcNormalK = Math.Clamp(_pcNormalK, 3, 200);
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcNormalsDialog = false;
                    ApplyPointCloudNormals();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcNormalsDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudPassDialog()
        {
            if (!_showPcPassDialog) return;
            if (ImGui.BeginPopup("Point Cloud Pass", ImGuiWindowFlags.AlwaysAutoResize))
            {
                string[] axes = { "X", "Y", "Z" };
                ImGui.Combo("Axis", ref _pcPassAxis, axes, axes.Length);
                ImGui.InputFloat("Min", ref _pcPassMin, 0.01f, 0.1f, "%.3f");
                ImGui.InputFloat("Max", ref _pcPassMax, 0.01f, 0.1f, "%.3f");
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcPassDialog = false;
                    ApplyPointCloudPassThrough();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcPassDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudRadiusDialog()
        {
            if (!_showPcRadiusDialog) return;
            if (ImGui.BeginPopup("Point Cloud Radius", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat3("Center", ref _pcRadiusCenter);
                ImGui.InputFloat("Radius", ref _pcRadius, 0.01f, 0.1f, "%.3f");
                _pcRadius = Math.Max(_pcRadius, 0.0001f);
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcRadiusDialog = false;
                    ApplyPointCloudRadiusCrop();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcRadiusDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudDenseDialog()
        {
            if (!_showPcDenseDialog) return;
            if (ImGui.BeginPopup("Point Cloud Dense", ImGuiWindowFlags.AlwaysAutoResize))
            {
                ImGui.InputFloat("Neighbor Radius", ref _pcDenseRadius, 0.001f, 0.01f, "%.4f");
                ImGui.InputInt("Points Per Seed", ref _pcDensePointsPerSeed);
                _pcDenseRadius = Math.Max(_pcDenseRadius, 0.0001f);
                _pcDensePointsPerSeed = Math.Clamp(_pcDensePointsPerSeed, 1, 8);
                if (ImGui.Button("Apply & Run", new System.Numerics.Vector2(120, 0)))
                {
                    _showPcDenseDialog = false;
                    ApplyPointCloudDenseCloud();
                }
                ImGui.SameLine();
                if (ImGui.Button("Close", new System.Numerics.Vector2(120, 0)))
                    _showPcDenseDialog = false;
                ImGui.EndPopup();
            }
        }

        private void ApplyPenMoveVertices()
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
                return;

            int moved = tool.MoveSelectedVertices(new Vector3(_penMoveDelta.X, _penMoveDelta.Y, _penMoveDelta.Z));
            _logBuffer += $"Moved {moved} selected vertices.\n";
            _isDirty = true;
            UpdateTitle();
        }

        private void ApplyPenExtrude()
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
                return;

            tool.ExtrudeSelectedTriangles(_penExtrudeDistance);
            _logBuffer += $"Extruded selected triangles by {_penExtrudeDistance:F3}.\n";
            _isDirty = true;
            UpdateTitle();
        }

        private void ApplyPenInset()
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
                return;

            tool.InsetSelectedTriangles(_penInsetAmount);
            _logBuffer += $"Inset selected triangles by {_penInsetAmount:F3}.\n";
            _isDirty = true;
            UpdateTitle();
        }

        private void ApplyPenBridge()
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.BridgeSelectedTrianglesSimple())
            {
                _logBuffer += "Bridge created between selected triangles.\n";
                _isDirty = true;
                UpdateTitle();
            }
            else
            {
                _logBuffer += "Bridge failed. Select exactly 2 triangles on the same mesh.\n";
            }
        }

        private List<PointCloudObject> GetSelectedPointCloudObjects()
        {
            return _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
        }

        private void ApplyPointCloudVoxel()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            float voxelSize = _pcVoxelSize;
            ProgressDialog.Instance.Start("Voxel Downsampling...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.VoxelDownsampleAsync(
                            selected[i],
                            voxelSize,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Voxel downsample removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Voxel downsampling cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudOutliers()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            int kNeighbors = _pcOutlierK;
            float stdRatio = _pcOutlierStdRatio;
            ProgressDialog.Instance.Start("Removing Outliers...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveStatisticalOutliersAsync(
                            selected[i],
                            kNeighbors,
                            stdRatio,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Outlier filter removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Outlier removal cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudDuplicates()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            float threshold = _pcDuplicateThreshold;
            ProgressDialog.Instance.Start("Removing Duplicates...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveDuplicatesAsync(
                            selected[i],
                            threshold,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Duplicate removal removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Duplicate removal cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudSkyBlue()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            var options = new PointCloudBlueFilterOptions
            {
                MinBlue = _pcSkyMinBlue,
                MaxRed = _pcSkyMaxRed,
                MaxGreen = _pcSkyMaxGreen,
                MinBlueDominance = _pcSkyBlueDominance
            };

            ProgressDialog.Instance.Start("Removing Sky/Blue Points...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RemoveBlueDominantPointsAsync(
                            selected[i],
                            options,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Sky/blue filter removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Sky/blue filter cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudNormals()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            int kNeighbors = _pcNormalK;
            ProgressDialog.Instance.Start("Estimating Normals...", OperationType.Processing);

            Task.Run(() => {
                try {
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        PointCloudOperations.EstimateNormalsAsync(
                            selected[i],
                            kNeighbors,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Wait();
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Estimated normals for {selected.Count} point cloud(s).\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Normal estimation cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudPassThrough()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            int axis = _pcPassAxis;
            float minValue = _pcPassMin;
            float maxValue = _pcPassMax;
            ProgressDialog.Instance.Start("Pass-Through Filter...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.PassThroughAxisAsync(
                            selected[i],
                            axis,
                            minValue,
                            maxValue,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Pass-through removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Pass-through filter cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudRadiusCrop()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            var center = new Vector3(_pcRadiusCenter.X, _pcRadiusCenter.Y, _pcRadiusCenter.Z);
            float radius = _pcRadius;
            ProgressDialog.Instance.Start("Radius Crop...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalRemoved = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var removed = PointCloudOperations.RadiusCropAsync(
                            selected[i],
                            center,
                            radius,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalRemoved += removed;
                    }

                    EnqueueAction(() => {
                        _logBuffer += $"Radius crop removed {totalRemoved:N0} points.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Radius crop cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyPointCloudDenseCloud()
        {
            var selected = GetSelectedPointCloudObjects();
            if (selected.Count == 0)
            {
                _logBuffer += "No point cloud selected.\n";
                return;
            }

            float radius = _pcDenseRadius;
            int seeds = _pcDensePointsPerSeed;
            int before = selected.Sum(pc => pc.PointCount);

            ProgressDialog.Instance.Start("Densifying Point Cloud...", OperationType.Processing);

            Task.Run(() => {
                try {
                    int totalAdded = 0;
                    for (int i = 0; i < selected.Count; i++) {
                        var progress = new Progress<string>(msg => ProgressDialog.Instance.Log(msg));
                        var cloudProgress = new Progress<float>(p => {
                            ProgressDialog.Instance.Update((i + p) / (float)selected.Count, $"Processing {selected[i].Name}... {(int)(p*100)}%");
                        });

                        var added = PointCloudOperations.DensifyAsync(
                            selected[i],
                            radius,
                            seeds,
                            ProgressDialog.Instance.CancellationTokenSource!.Token,
                            progress).Result;

                        totalAdded += added;
                    }

                    EnqueueAction(() => {
                        int after = selected.Sum(pc => pc.PointCount);
                        if (totalAdded > 0)
                            _logBuffer += $"Dense cloud added {totalAdded:N0} points ({before:N0} -> {after:N0}).\n";
                        else
                            _logBuffer += $"Dense cloud added 0 points. Increase radius (current {radius:F4}) or reduce filtering.\n";
                        _isDirty = true;
                        UpdateTitle();
                        ProgressDialog.Instance.Complete();
                    });
                } catch (AggregateException ae) when (ae.InnerException is OperationCanceledException) {
                    EnqueueAction(() => {
                        _logBuffer += "Densification cancelled.\n";
                        ProgressDialog.Instance.Close();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void ApplyDecimatePreset()
        {
            if (!_sceneGraph.SelectedObjects.OfType<MeshObject>().Any())
            {
                _logBuffer += "No mesh selected.\n";
                return;
            }

            PerformDecimation();
        }

        private void ApplySmoothPreset()
        {
            if (!_sceneGraph.SelectedObjects.OfType<MeshObject>().Any())
            {
                _logBuffer += "No mesh selected.\n";
                return;
            }

            PerformSmooth();
        }

        private void ApplyOptimizePreset()
        {
            if (!_sceneGraph.SelectedObjects.OfType<MeshObject>().Any())
            {
                _logBuffer += "No mesh selected.\n";
                return;
            }

            PerformOptimize();
        }

        private bool _showDecimateDialog = false;
        private float _decimateRatio = 0.5f;
        private float _decimateVoxelSize = 0.01f;
        private int _decimateMethod = 0; // 0 = Ratio, 1 = Uniform

        private void OnDecimate()
        {
            _showDecimateDialog = true;
            _popupToOpen = "Decimate Mesh";
        }

        private void DrawDecimateDialog()
        {
             if (!_showDecimateDialog) return;

             if (ImGui.BeginPopup("Decimate Mesh", ImGuiWindowFlags.AlwaysAutoResize))
             {
                 bool hasMesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().Any();

                 ImGui.Text("Choose Decimation Method:");
                 ImGui.RadioButton("Target Ratio (Adaptive)", ref _decimateMethod, 0);
                 ImGui.RadioButton("Voxel Grid (Uniform)", ref _decimateMethod, 1);
                 ImGui.Separator();

                 if (_decimateMethod == 0)
                 {
                     ImGui.SliderFloat("Target Ratio", ref _decimateRatio, 0.01f, 0.99f, "%.2f");
                 }
                 else
                 {
                     ImGui.InputFloat("Voxel Size", ref _decimateVoxelSize, 0.001f, 0.01f, "%.4f");
                 }

                 ImGui.Separator();

                 if (!hasMesh)
                 {
                     ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                     ImGui.BeginDisabled();
                 }

                 if (ImGui.Button("Decimate", new System.Numerics.Vector2(120, 0)))
                 {
                     _showDecimateDialog = false;
                     PerformDecimation();
                 }

                 if (!hasMesh)
                 {
                     ImGui.EndDisabled();
                 }

                 ImGui.SameLine();
                 if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showDecimateDialog = false;
                 ImGui.EndPopup();
             }
        }

        private void PerformDecimation()
        {
            var objects = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            float ratio = _decimateRatio; float voxelSize = _decimateVoxelSize; bool uniform = _decimateMethod == 1;
            ProgressDialog.Instance.Start("Decimating Mesh...", OperationType.Processing);
            Task.Run(() => {
                try {
                    var results = new List<(MeshObject obj, MeshData newData)>();
                    foreach (var mo in objects) {
                        var newData = uniform
                            ? MeshOperations.DecimateUniform(mo.MeshData, voxelSize)
                            : MeshOperations.Decimate(mo.MeshData, ratio);
                        results.Add((mo, newData));
                    }
                    EnqueueAction(() => {
                        foreach (var res in results) {
                            res.obj.MeshData = res.newData;
                            ProgressDialog.Instance.Log($"Decimated: {res.obj.Name}");
                        }
                        ProgressDialog.Instance.Complete();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private bool _showSmoothDialog = false;
        private int _smoothIter = 2;
        private float _smoothLambda = 0.5f;
        private float _smoothMu = -0.53f;
        private int _smoothMethod = 1; // 0=Laplacian, 1=Taubin

        private void OnSmooth()
        {
            _showSmoothDialog = true;
            _popupToOpen = "Smooth Mesh";
        }

        private void DrawSmoothDialog()
        {
            if (!_showSmoothDialog) return;
            if (ImGui.BeginPopup("Smooth Mesh", ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool hasMesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().Any();

                ImGui.RadioButton("Laplacian", ref _smoothMethod, 0); ImGui.SameLine();
                ImGui.RadioButton("Taubin", ref _smoothMethod, 1);
                ImGui.InputInt("Iterations", ref _smoothIter);
                ImGui.SliderFloat("Lambda", ref _smoothLambda, 0.01f, 1.0f);
                if (_smoothMethod == 1) ImGui.SliderFloat("Mu", ref _smoothMu, -1.0f, -0.01f);

                ImGui.Separator();

                if (!hasMesh)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Smooth", new System.Numerics.Vector2(120, 0)))
                {
                    _showSmoothDialog = false;
                    PerformSmooth();
                }

                if (!hasMesh)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showSmoothDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformSmooth()
        {
            var objects = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            int iter = _smoothIter; float lam = _smoothLambda; float mu = _smoothMu; bool taubin = _smoothMethod == 1;
            ProgressDialog.Instance.Start("Smoothing Mesh...", OperationType.Processing);
            Task.Run(() => {
                try {
                    var results = new List<(MeshObject obj, MeshData newData)>();
                    foreach (var mo in objects) {
                        var newData = taubin
                            ? MeshOperations.SmoothTaubin(mo.MeshData, iter, lam, mu)
                            : MeshOperations.Smooth(mo.MeshData, iter, lam);
                        results.Add((mo, newData));
                    }
                    EnqueueAction(() => {
                        foreach (var res in results) {
                            res.obj.MeshData = res.newData;
                            ProgressDialog.Instance.Log($"Smoothed: {res.obj.Name}");
                        }
                        ProgressDialog.Instance.Complete();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private bool _showOptimizeDialog = false;
        private float _optimizeEpsilon = 0.0001f;

        private void OnOptimize()
        {
            _showOptimizeDialog = true;
            _popupToOpen = "Optimize Mesh";
        }

        private void DrawOptimizeDialog()
        {
            if (!_showOptimizeDialog) return;
            if (ImGui.BeginPopup("Optimize Mesh", ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool hasMesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().Any();

                ImGui.InputFloat("Weld Distance", ref _optimizeEpsilon, 0.00001f, 0.0001f, "%.6f");
                ImGui.Separator();

                if (!hasMesh)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Optimize", new System.Numerics.Vector2(120, 0)))
                {
                    _showOptimizeDialog = false;
                    PerformOptimize();
                }

                if (!hasMesh)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showOptimizeDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformOptimize()
        {
            var objects = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            float eps = _optimizeEpsilon;
            ProgressDialog.Instance.Start("Optimizing Mesh...", OperationType.Processing);
            Task.Run(() => {
                try {
                    var results = new List<(MeshObject obj, MeshData newData)>();
                    foreach (var mo in objects) {
                        var newData = MeshOperations.Optimize(mo.MeshData, eps);
                        results.Add((mo, newData));
                    }
                    EnqueueAction(() => {
                        foreach (var res in results) {
                            res.obj.MeshData = res.newData;
                            ProgressDialog.Instance.Log($"Optimized: {res.obj.Name}");
                        }
                        ProgressDialog.Instance.Complete();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void OnSplit()
        {
            var objects = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (objects.Count == 0) return;

            ProgressDialog.Instance.Start("Splitting Mesh...", OperationType.Processing);
            Task.Run(() => {
                foreach (var mo in objects)
                {
                    try
                    {
                        var parts = MeshOperations.SplitByConnectivity(mo.MeshData);
                        if (parts.Count > 1)
                        {
                            lock (_sceneGraph)
                            {
                                _sceneGraph.RemoveObject(mo);
                                int i = 1;
                                foreach (var part in parts)
                                {
                                    var newObj = new MeshObject($"{mo.Name}_part{i}", part);
                                    _sceneGraph.AddObject(newObj);
                                    i++;
                                }
                            }
                            ProgressDialog.Instance.Log($"Split {mo.Name} into {parts.Count} parts.");
                        }
                        else
                        {
                            ProgressDialog.Instance.Log($"{mo.Name} has only one connected component.");
                        }
                    }
                    catch (Exception ex)
                    {
                        ProgressDialog.Instance.Fail(ex);
                        return;
                    }
                }
                ProgressDialog.Instance.Complete();
            });
        }

        private void OnFlipNormals()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                mo.MeshData = MeshOperations.FlipNormals(mo.MeshData);
                _logBuffer += $"Flipped normals: {mo.Name}\n";
            }
        }

        private bool _showMergeDialog = false;
        private bool _showMeshToolbarMergeDialog = false;
        private bool _showPointCloudToolbarMergeDialog = false;
        private float _mergeDist = 0.001f;

        private void OnMerge()
        {
            _showMergeDialog = true;
            _popupToOpen = "Merge Objects";
        }

        private void OpenMeshToolbarMergeDialog()
        {
            _showMeshToolbarMergeDialog = true;
            _popupToOpen = "Merge Meshes";
        }

        private void OpenPointCloudToolbarMergeDialog()
        {
            _showPointCloudToolbarMergeDialog = true;
            _popupToOpen = "Merge Point Clouds";
        }

        private void DrawMergeDialog()
        {
            if (!_showMergeDialog) return;
            if (ImGui.BeginPopup("Merge Objects", ImGuiWindowFlags.AlwaysAutoResize))
            {
                int meshCount = _sceneGraph.SelectedObjects.OfType<MeshObject>().Count();
                int pointCloudCount = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().Count();
                bool mixedSelection = meshCount > 0 && pointCloudCount > 0;
                bool canMerge = !mixedSelection && (meshCount >= 2 || pointCloudCount >= 2);

                ImGui.InputFloat("Weld Distance", ref _mergeDist, 0.0001f, 0.001f, "%.5f");
                ImGui.Separator();

                if (!canMerge)
                {
                    string msg = mixedSelection
                        ? "Mixed selection not supported. Select only meshes or only point clouds."
                        : "Select at least 2 meshes or 2 point clouds.";
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), msg);
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Merge", new System.Numerics.Vector2(120, 0)))
                {
                    _showMergeDialog = false;
                    PerformMerge();
                }

                if (!canMerge)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showMergeDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformMerge()
        {
             var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
             var pcs = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
             float dist = _mergeDist;

             if (meshes.Count > 0 && pcs.Count > 0)
             {
                 _logBuffer += "Merge aborted: mixed selection (mesh + point cloud) is not supported.\n";
                 return;
             }

             ProgressDialog.Instance.Start("Merging...", OperationType.Processing);
             Task.Run(() => {
                 try {
                     if (meshes.Count >= 2) {
                         var merged = MeshOperations.MergeWithWelding(meshes.Select(m => m.MeshData).ToList(), dist);
                         EnqueueAction(() => {
                             var newObj = new MeshObject("Merged", merged);
                             foreach(var m in meshes) _sceneGraph.RemoveObject(m);
                             _sceneGraph.AddObject(newObj);
                             ProgressDialog.Instance.Log("Merged meshes.");
                             ProgressDialog.Instance.Complete();
                         });
                     }
                     else if (pcs.Count >= 2) {
                         var merged = MeshOperations.MergePointClouds(pcs);
                         EnqueueAction(() => {
                             foreach(var p in pcs) _sceneGraph.RemoveObject(p);
                             _sceneGraph.AddObject(merged);
                             ProgressDialog.Instance.Log("Merged point clouds.");
                             ProgressDialog.Instance.Complete();
                         });
                     }
                 } catch (Exception ex) {
                     EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                 }
             });
        }

        private bool _showAlignDialog = false;
        private int _alignIter = 50;
        private float _alignThreshold = 0.0001f;

        private void OnAlign()
        {
            _showAlignDialog = true;
            _popupToOpen = "Align Objects";
        }

        private void DrawAlignDialog()
        {
            if (!_showAlignDialog) return;
            if (ImGui.BeginPopup("Align Objects", ImGuiWindowFlags.AlwaysAutoResize))
            {
                int meshCount = _sceneGraph.SelectedObjects.OfType<MeshObject>().Count();
                int pointCloudCount = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().Count();
                bool mixedSelection = meshCount > 0 && pointCloudCount > 0;
                bool canAlign = !mixedSelection && (meshCount >= 2 || pointCloudCount >= 2);

                ImGui.InputInt("Max Iterations", ref _alignIter);
                ImGui.InputFloat("Convergence Threshold", ref _alignThreshold, 0.00001f, 0.0001f, "%.6f");
                ImGui.Separator();

                if (!canAlign)
                {
                    string msg = mixedSelection
                        ? "Mixed selection not supported. Select only meshes or only point clouds."
                        : "Select at least 2 meshes or 2 point clouds.";
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), msg);
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Align", new System.Numerics.Vector2(120, 0)))
                {
                    _showAlignDialog = false;
                    PerformAlign();
                }

                if (!canAlign)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showAlignDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformAlign()
        {
             var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
             var pcs = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
             int iter = _alignIter; float thresh = _alignThreshold;

             if (meshes.Count > 0 && pcs.Count > 0)
             {
                 _logBuffer += "Align aborted: mixed selection (mesh + point cloud) is not supported.\n";
                 return;
             }

             ProgressDialog.Instance.Start("Aligning...", OperationType.Processing);
             Task.Run(() => {
                 try {
                     if (meshes.Count >= 2) {
                         var target = meshes[0].MeshData;
                         var transforms = new List<(MeshObject obj, Matrix4 transform)>();
                         for(int i=1; i<meshes.Count; i++) {
                             var transform = MeshOperations.AlignICP(meshes[i].MeshData, target, iter, thresh);
                             transforms.Add((meshes[i], transform));
                         }
                         EnqueueAction(() => {
                             foreach(var t in transforms) t.obj.MeshData.ApplyTransform(t.transform);
                             ProgressDialog.Instance.Log("Aligned meshes.");
                             ProgressDialog.Instance.Complete();
                         });
                     }
                     else if (pcs.Count >= 2) {
                         var target = pcs[0];
                         var tPoints = target.Points.Select(p => Vector3.TransformPosition(p, target.GetWorldTransform())).ToList();
                         var transforms = new List<(PointCloudObject obj, Matrix4 transform)>();
                         for(int i=1; i<pcs.Count; i++) {
                             var sPoints = pcs[i].Points.Select(p => Vector3.TransformPosition(p, pcs[i].GetWorldTransform())).ToList();
                             var transform = MeshOperations.AlignICP(sPoints, tPoints, iter, thresh);
                             transforms.Add((pcs[i], transform));
                         }
                         EnqueueAction(() => {
                             foreach(var t in transforms) t.obj.ApplyTransform(t.transform);
                             ProgressDialog.Instance.Log("Aligned point clouds.");
                             ProgressDialog.Instance.Complete();
                         });
                     }
                 } catch (Exception ex) {
                     EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                 }
             });
        }

        private void DrawMeshToolbarMergeDialog()
        {
            if (!_showMeshToolbarMergeDialog) return;
            if (ImGui.BeginPopup("Merge Meshes", ImGuiWindowFlags.AlwaysAutoResize))
            {
                int meshCount = _sceneGraph.SelectedObjects.OfType<MeshObject>().Count();
                int pointCloudCount = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().Count();
                bool canMerge = meshCount >= 2 && pointCloudCount == 0;

                ImGui.InputFloat("Weld Distance", ref _mergeDist, 0.0001f, 0.001f, "%.5f");
                ImGui.InputInt("Align Iterations", ref _alignIter);
                ImGui.InputFloat("Align Threshold", ref _alignThreshold, 0.00001f, 0.0001f, "%.6f");
                ImGui.Separator();

                if (!canMerge)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1),
                        pointCloudCount > 0
                            ? "Select only meshes for this command."
                            : "Select at least 2 meshes.");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Merge", new System.Numerics.Vector2(120, 0)))
                {
                    _showMeshToolbarMergeDialog = false;
                    PerformMeshToolbarMerge(alignFirst: false);
                }
                ImGui.SameLine();
                if (ImGui.Button("Align + Merge", new System.Numerics.Vector2(140, 0)))
                {
                    _showMeshToolbarMergeDialog = false;
                    PerformMeshToolbarMerge(alignFirst: true);
                }

                if (!canMerge)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(100, 0))) _showMeshToolbarMergeDialog = false;
                ImGui.EndPopup();
            }
        }

        private void DrawPointCloudToolbarMergeDialog()
        {
            if (!_showPointCloudToolbarMergeDialog) return;
            if (ImGui.BeginPopup("Merge Point Clouds", ImGuiWindowFlags.AlwaysAutoResize))
            {
                int meshCount = _sceneGraph.SelectedObjects.OfType<MeshObject>().Count();
                int pointCloudCount = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().Count();
                bool canMerge = pointCloudCount >= 2 && meshCount == 0;

                ImGui.InputInt("Align Iterations", ref _alignIter);
                ImGui.InputFloat("Align Threshold", ref _alignThreshold, 0.00001f, 0.0001f, "%.6f");
                ImGui.Separator();

                if (!canMerge)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1),
                        meshCount > 0
                            ? "Select only point clouds for this command."
                            : "Select at least 2 point clouds.");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Merge", new System.Numerics.Vector2(120, 0)))
                {
                    _showPointCloudToolbarMergeDialog = false;
                    PerformPointCloudToolbarMerge(alignFirst: false);
                }
                ImGui.SameLine();
                if (ImGui.Button("Align + Merge", new System.Numerics.Vector2(140, 0)))
                {
                    _showPointCloudToolbarMergeDialog = false;
                    PerformPointCloudToolbarMerge(alignFirst: true);
                }

                if (!canMerge)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(100, 0))) _showPointCloudToolbarMergeDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformMeshToolbarMerge(bool alignFirst)
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var pointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (pointClouds.Count > 0 || meshes.Count < 2)
            {
                ShowError("Invalid Selection", "Select only meshes and at least two items to merge.");
                return;
            }

            int iter = Math.Max(1, _alignIter);
            float thresh = Math.Max(1e-6f, _alignThreshold);
            float dist = Math.Max(0.0f, _mergeDist);

            ProgressDialog.Instance.Start(alignFirst ? "Aligning and merging meshes..." : "Merging meshes...", OperationType.Processing);
            Task.Run(() =>
            {
                try
                {
                    var mergeInputs = meshes.Select(m => m.MeshData.Clone()).ToList();
                    if (alignFirst)
                    {
                        var target = mergeInputs[0];
                        for (int i = 1; i < mergeInputs.Count; i++)
                        {
                            var transform = MeshOperations.AlignICP(mergeInputs[i], target, iter, thresh);
                            mergeInputs[i].ApplyTransform(transform);
                        }
                    }

                    var merged = MeshOperations.MergeWithWelding(mergeInputs, dist);
                    EnqueueAction(() =>
                    {
                        foreach (var m in meshes) _sceneGraph.RemoveObject(m);
                        var newObj = new MeshObject("Merged Mesh", merged);
                        _sceneGraph.AddObject(newObj);
                        _sceneGraph.Select(newObj);
                        _viewport.FocusOnSelection();
                        ProgressDialog.Instance.Log(alignFirst ? "Aligned and merged meshes." : "Merged meshes.");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void PerformPointCloudToolbarMerge(bool alignFirst)
        {
            var pcs = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count > 0 || pcs.Count < 2)
            {
                ShowError("Invalid Selection", "Select only point clouds and at least two items to merge.");
                return;
            }

            int iter = Math.Max(1, _alignIter);
            float thresh = Math.Max(1e-6f, _alignThreshold);

            ProgressDialog.Instance.Start(alignFirst ? "Aligning and merging point clouds..." : "Merging point clouds...", OperationType.Processing);
            Task.Run(() =>
            {
                try
                {
                    if (alignFirst)
                    {
                        var target = pcs[0];
                        var tPoints = target.Points.Select(p => Vector3.TransformPosition(p, target.GetWorldTransform())).ToList();
                        for (int i = 1; i < pcs.Count; i++)
                        {
                            var sPoints = pcs[i].Points.Select(p => Vector3.TransformPosition(p, pcs[i].GetWorldTransform())).ToList();
                            var transform = MeshOperations.AlignICP(sPoints, tPoints, iter, thresh);
                            pcs[i].ApplyTransform(transform);
                            pcs[i].UpdateBounds();
                        }
                    }

                    var merged = MeshOperations.MergePointClouds(pcs);
                    EnqueueAction(() =>
                    {
                        foreach (var pc in pcs) _sceneGraph.RemoveObject(pc);
                        _sceneGraph.AddObject(merged);
                        _sceneGraph.Select(merged);
                        _viewport.FocusOnSelection();
                        ProgressDialog.Instance.Log(alignFirst ? "Aligned and merged point clouds." : "Merged point clouds.");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private bool _showCleanupDialog = false;
        // Cleanup options flags
        private bool _cleanIsolated = true;
        private bool _cleanNormals = true;
        private bool _cleanHoles = true;

        private void OnCleanup()
        {
            _showCleanupDialog = true;
            _popupToOpen = "Cleanup Mesh";
        }

        private void DrawCleanupDialog()
        {
            if (!_showCleanupDialog) return;
            if (ImGui.BeginPopup("Cleanup Mesh", ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool hasMesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().Any();

                ImGui.Checkbox("Remove Isolated Vertices", ref _cleanIsolated);
                ImGui.Checkbox("Recalculate Normals", ref _cleanNormals);
                ImGui.Checkbox("Fill Holes", ref _cleanHoles);

                ImGui.Separator();

                if (!hasMesh)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Cleanup", new System.Numerics.Vector2(120, 0)))
                {
                    _showCleanupDialog = false;
                    PerformCleanup();
                }

                if (!hasMesh)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showCleanupDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformCleanup()
        {
            var objects = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            bool cleanHoles = _cleanHoles; bool cleanNormals = _cleanNormals; bool cleanIsolated = _cleanIsolated;

            ProgressDialog.Instance.Start("Cleaning Mesh...", OperationType.Processing);
            Task.Run(() => {
                try {
                    var results = new List<(MeshObject obj, MeshData newData)>();
                    foreach (var mo in objects)
                    {
                        var opts = cleanHoles ? MeshCleanupOptions.All : MeshCleanupOptions.Default;
                        var newData = MeshCleaningTools.CleanupMesh(mo.MeshData, opts);
                        if (cleanIsolated) newData = MeshOperations.RemoveIsolatedVertices(newData);
                        if (cleanNormals) newData.RecalculateNormals();
                        results.Add((mo, newData));
                    }
                    EnqueueAction(() => {
                        foreach(var res in results) {
                            res.obj.MeshData = res.newData;
                            ProgressDialog.Instance.Log($"Cleaned up: {res.obj.Name}");
                        }
                        ProgressDialog.Instance.Complete();
                    });
                } catch (Exception ex) {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private bool _showBakeDialog = false;
        private int _bakeSize = 2048;
        private int _bakeIslandMargin = 4;
        private bool _bakeFromCameras = true;
        private int _bakeUVMethod = 0; // 0 Smart, 1 Lightmap, 2 Box, 3 Cyl, 4 Sph
        private int _bakeBlendMode = 2; // 0 Replace, 1 Average, 2 ViewAngle, 3 Distance
        private float _bakeMinViewAngle = 0.1f;
        private bool _bakeBlendSeams = true;
        private int _bakeDilationPasses = 4;
        private int _bakeExportFormat = 0; // 0 OBJ, 1 GLTF, 2 GLB, 3 FBX, 4 PLY
        private int _bakeTextureFormat = 0; // 0 PNG, 1 JPEG, 2 BMP
        private int _bakeJpegQuality = 90;
        private bool _bakeExportNormals = true;
        private bool _bakeSwapYZ = false;
        private readonly HashSet<int> _bakeSelectedCameraIds = new HashSet<int>();

        private void OnBakeTextures()
        {
            SyncBakeCameraSelection();
            _showBakeDialog = true;
            _popupToOpen = "Bake Textures";
        }

        private void DrawBakeDialog()
        {
            if (!_showBakeDialog) return;
            if (ImGui.BeginPopup("Bake Textures", ImGuiWindowFlags.AlwaysAutoResize))
            {
                var mesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().FirstOrDefault();
                var validCameras = GetBakeableCameras();
                var selectedCameras = GetSelectedBakeCameras();

                bool hasMesh = mesh != null;
                bool hasAnyValidCamera = validCameras.Count > 0;
                bool hasSelectedCamera = selectedCameras.Count > 0;
                bool canBake = hasMesh && (!_bakeFromCameras || hasSelectedCamera);

                ImGui.Text("Source");
                if (ImGui.RadioButton("Project from camera images", _bakeFromCameras)) _bakeFromCameras = true;
                if (ImGui.RadioButton("Bake vertex colors to texture", !_bakeFromCameras)) _bakeFromCameras = false;

                ImGui.Spacing();
                ImGui.Text("UV Settings");
                string[] uvMethods = { "Smart UV Project", "Lightmap Pack", "Box Projection", "Cylindrical Projection", "Spherical Projection" };
                ImGui.Combo("UV Method", ref _bakeUVMethod, uvMethods, uvMethods.Length);
                ImGui.InputInt("Texture Size", ref _bakeSize);
                ImGui.InputInt("Island Margin", ref _bakeIslandMargin);
                _bakeSize = Math.Clamp(_bakeSize, 256, 8192);
                _bakeIslandMargin = Math.Clamp(_bakeIslandMargin, 0, 64);

                ImGui.Spacing();
                ImGui.Text("Baking Settings");
                string[] blendModes = { "Replace", "Average", "View Angle Weighted", "Distance Weighted" };
                ImGui.Combo("Blend Mode", ref _bakeBlendMode, blendModes, blendModes.Length);
                ImGui.SliderFloat("Min View Angle", ref _bakeMinViewAngle, 0.0f, 0.95f, "%.2f");
                ImGui.Checkbox("Blend Seams", ref _bakeBlendSeams);
                ImGui.InputInt("Dilation Passes", ref _bakeDilationPasses);
                _bakeDilationPasses = Math.Clamp(_bakeDilationPasses, 0, 16);

                ImGui.Spacing();
                ImGui.Text("Export Settings");
                string[] exportFormats = { "OBJ", "GLTF", "GLB", "FBX (ASCII)", "PLY" };
                ImGui.Combo("Mesh Format", ref _bakeExportFormat, exportFormats, exportFormats.Length);

                string[] textureFormats = { "PNG", "JPEG", "BMP" };
                ImGui.Combo("Texture Format", ref _bakeTextureFormat, textureFormats, textureFormats.Length);
                ImGui.BeginDisabled(_bakeTextureFormat != 1);
                ImGui.SliderInt("JPEG Quality", ref _bakeJpegQuality, 50, 100);
                ImGui.EndDisabled();
                _bakeJpegQuality = Math.Clamp(_bakeJpegQuality, 50, 100);
                ImGui.Checkbox("Export Normals", ref _bakeExportNormals);
                ImGui.Checkbox("Swap Y/Z", ref _bakeSwapYZ);

                if (_bakeFromCameras)
                {
                    ImGui.Spacing();
                    ImGui.Text("Camera Selection");
                    if (ImGui.Button("Select All")) SelectAllBakeCameras();
                    ImGui.SameLine();
                    if (ImGui.Button("Select None")) _bakeSelectedCameraIds.Clear();

                    ImGui.BeginChild("BakeCameraList", new System.Numerics.Vector2(460, 140), ImGuiChildFlags.Borders);
                    foreach (var cam in validCameras)
                    {
                        bool selected = _bakeSelectedCameraIds.Contains(cam.Id);
                        if (ImGui.Checkbox($"##cam_{cam.Id}", ref selected))
                        {
                            if (selected) _bakeSelectedCameraIds.Add(cam.Id);
                            else _bakeSelectedCameraIds.Remove(cam.Id);
                        }
                        ImGui.SameLine();
                        string imageName = string.IsNullOrEmpty(cam.ImagePath) ? "(no image)" : Path.GetFileName(cam.ImagePath);
                        ImGui.TextUnformatted($"{cam.Name} - {imageName}");
                    }
                    ImGui.EndChild();
                }

                ImGui.Separator();

                if (!canBake)
                {
                    if (!hasMesh)
                        ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                    if (_bakeFromCameras && !hasAnyValidCamera)
                        ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No cameras with pose+image in scene");
                    if (_bakeFromCameras && hasAnyValidCamera && !hasSelectedCamera)
                        ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "Select at least one camera");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Bake && Export", new System.Numerics.Vector2(140, 0)))
                {
                    _showBakeDialog = false;
                    PerformBake();
                }

                if (!canBake)
                {
                    ImGui.EndDisabled();
                }

                ImGui.SameLine();
                if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0))) _showBakeDialog = false;
                ImGui.EndPopup();
            }
        }

        private void PerformBake()
        {
            var meshObj = _sceneGraph.SelectedObjects.OfType<MeshObject>().FirstOrDefault();
            if (meshObj == null)
            {
                _logBuffer += "No mesh selected for texture baking.\n";
                return;
            }

            var selectedCameras = GetSelectedBakeCameras();
            if (_bakeFromCameras && selectedCameras.Count == 0)
            {
                _logBuffer += "No valid cameras selected for texture baking.\n";
                return;
            }

            var exportOptions = BuildBakeExportOptions();
            string extension = GetBakeFileExtension(exportOptions.Format);
            var saveFilter = new Dictionary<string, string> { { GetBakeFilterLabel(exportOptions.Format), extension.TrimStart('.') } };
            var saveResult = Nfd.SaveDialog(out string exportPath, saveFilter);
            if (saveResult != NfdStatus.Ok || string.IsNullOrEmpty(exportPath))
                return;

            if (string.IsNullOrEmpty(Path.GetExtension(exportPath)))
                exportPath += extension;

            int size = _bakeSize;
            int islandMargin = _bakeIslandMargin;
            int uvMethodIndex = _bakeUVMethod;
            int blendModeIndex = _bakeBlendMode;
            float minViewAngle = _bakeMinViewAngle;
            bool blendSeams = _bakeBlendSeams;
            int dilationPasses = _bakeDilationPasses;
            bool bakeFromCameras = _bakeFromCameras;

            ProgressDialog.Instance.Start("Baking Textures...", OperationType.Processing);

            Task.Run(() =>
            {
                try
                {
                    var baker = new Deep3DStudio.Texturing.TextureBaker();
                    baker.TextureSize = size;
                    baker.IslandMargin = islandMargin;
                    baker.BlendMode = MapBakeBlendMode(blendModeIndex);
                    baker.MinViewAngleCosine = minViewAngle;
                    baker.BlendSeams = blendSeams;
                    baker.DilationPasses = dilationPasses;

                    var mesh = meshObj.MeshData;
                    ProgressDialog.Instance.Update(0.1f, "Generating UVs...");
                    var uvData = baker.GenerateUVs(mesh, MapBakeUvMethod(uvMethodIndex));

                    Deep3DStudio.Texturing.BakedTextureResult baked;
                    if (bakeFromCameras)
                    {
                        ProgressDialog.Instance.Update(0.3f, "Projecting images...");
                        var progress = new Progress<float>(p => ProgressDialog.Instance.Update(0.3f + p * 0.5f, $"Baking... {(int)(p * 100)}%"));
                        baked = baker.BakeTextures(mesh, uvData, selectedCameras, progress);
                    }
                    else
                    {
                        ProgressDialog.Instance.Update(0.3f, "Baking vertex colors...");
                        var tex = baker.BakeVertexColorsToTexture(mesh, uvData);
                        baked = new Deep3DStudio.Texturing.BakedTextureResult
                        {
                            DiffuseMap = tex,
                            TextureSize = baker.TextureSize,
                            WeightMap = new float[baker.TextureSize, baker.TextureSize]
                        };
                    }

                    ProgressDialog.Instance.Update(0.85f, "Exporting textured mesh...");
                    TexturedMeshExporter.Export(exportPath, mesh, uvData, baked, exportOptions);
                    baked.Dispose();

                    ProgressDialog.Instance.Log($"Textured mesh exported: {exportPath}");
                    ProgressDialog.Instance.Complete();
                }
                catch (Exception ex)
                {
                    EnqueueAction(() =>
                    {
                        _executePendingActionAfterSave = false;
                        ProgressDialog.Instance.Fail(ex);
                    });
                }
            });
        }

        private List<CameraObject> GetBakeableCameras()
        {
            return _sceneGraph.GetObjectsOfType<CameraObject>()
                .Where(c => c.Pose != null && !string.IsNullOrEmpty(c.ImagePath))
                .ToList();
        }

        private List<CameraObject> GetSelectedBakeCameras()
        {
            var cameras = GetBakeableCameras();
            return cameras.Where(c => _bakeSelectedCameraIds.Contains(c.Id)).ToList();
        }

        private void SelectAllBakeCameras()
        {
            _bakeSelectedCameraIds.Clear();
            foreach (var cam in GetBakeableCameras())
                _bakeSelectedCameraIds.Add(cam.Id);
        }

        private void SyncBakeCameraSelection()
        {
            var validIds = GetBakeableCameras().Select(c => c.Id).ToHashSet();
            _bakeSelectedCameraIds.RemoveWhere(id => !validIds.Contains(id));
            if (_bakeSelectedCameraIds.Count == 0)
            {
                foreach (var id in validIds)
                    _bakeSelectedCameraIds.Add(id);
            }
        }

        private Deep3DStudio.Texturing.UVUnwrapMethod MapBakeUvMethod(int index)
        {
            return index switch
            {
                1 => Deep3DStudio.Texturing.UVUnwrapMethod.LightmapPack,
                2 => Deep3DStudio.Texturing.UVUnwrapMethod.BoxProject,
                3 => Deep3DStudio.Texturing.UVUnwrapMethod.CylindricalProject,
                4 => Deep3DStudio.Texturing.UVUnwrapMethod.SphericalProject,
                _ => Deep3DStudio.Texturing.UVUnwrapMethod.SmartProject
            };
        }

        private Deep3DStudio.Texturing.TextureBlendMode MapBakeBlendMode(int index)
        {
            return index switch
            {
                0 => Deep3DStudio.Texturing.TextureBlendMode.Replace,
                1 => Deep3DStudio.Texturing.TextureBlendMode.Average,
                3 => Deep3DStudio.Texturing.TextureBlendMode.DistanceWeighted,
                _ => Deep3DStudio.Texturing.TextureBlendMode.ViewAngleWeighted
            };
        }

        private MeshExportOptions BuildBakeExportOptions()
        {
            return new MeshExportOptions
            {
                Format = _bakeExportFormat switch
                {
                    1 => TexturedMeshFormat.GLTF,
                    2 => TexturedMeshFormat.GLB,
                    3 => TexturedMeshFormat.FBX_ASCII,
                    4 => TexturedMeshFormat.PLY,
                    _ => TexturedMeshFormat.OBJ
                },
                TextureFormat = _bakeTextureFormat switch
                {
                    1 => TextureFormat.JPEG,
                    2 => TextureFormat.BMP,
                    _ => TextureFormat.PNG
                },
                JpegQuality = _bakeJpegQuality,
                ExportNormals = _bakeExportNormals,
                SwapYZ = _bakeSwapYZ,
                ExportUVs = true,
                ExportTextures = true
            };
        }

        private string GetBakeFileExtension(TexturedMeshFormat format)
        {
            return format switch
            {
                TexturedMeshFormat.GLTF => ".gltf",
                TexturedMeshFormat.GLB => ".glb",
                TexturedMeshFormat.FBX_ASCII => ".fbx",
                TexturedMeshFormat.PLY => ".ply",
                _ => ".obj"
            };
        }

        private string GetBakeFilterLabel(TexturedMeshFormat format)
        {
            return format switch
            {
                TexturedMeshFormat.GLTF => "GLTF Mesh",
                TexturedMeshFormat.GLB => "GLB Mesh",
                TexturedMeshFormat.FBX_ASCII => "FBX Mesh",
                TexturedMeshFormat.PLY => "PLY Mesh",
                _ => "OBJ Mesh"
            };
        }

        #endregion

        #region AI Operations

        private void RunAIModel(string modelName)
        {
            _logBuffer += $"Running {modelName}...\n";
            // Map to appropriate workflow (indices match _workflowsBase array)
            switch (modelName)
            {
                case "TripoSR":
                    _selectedWorkflow = 2; // TripoSR (Single Image)
                    RunReconstruction();
                    break;
                case "LGM":
                    _selectedWorkflow = 3; // LGM (Single Image)
                    RunReconstruction();
                    break;
                case "Wonder3D":
                    _selectedWorkflow = 4; // Wonder3D (Single Image)
                    RunReconstruction();
                    break;
                case "DeepMeshPrior":
                    RunDeepMeshPriorRefinement();
                    break;
                case "GaussianSDF":
                    RunGaussianSDFRefinement();
                    break;
                case "TripoSF":
                    RunTripoSFRefinement();
                    break;
                case "UniRig":
                    OnAutoRig();
                    break;
                default:
                    _logBuffer += $"Model {modelName} not yet implemented.\n";
                    break;
            }
        }

        private void RunDeepMeshPriorRefinement()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "Error: No mesh selected for DeepMeshPrior refinement.\n";
                return;
            }

            ProgressDialog.Instance.Start("DeepMeshPrior Optimization...", OperationType.Processing);
            Task.Run(async () => {
                try
                {
                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var refiner = new Deep3DStudio.Meshing.DeepMeshPriorMesher();
                    foreach (var mesh in meshes)
                    {
                        var refined = await refiner.RefineMeshAsync(
                            mesh.MeshData,
                            (status, progress) => ProgressDialog.Instance.Update(progress, status),
                            cancellationToken);
                        if (refined != null)
                        {
                            EnqueueAction(() => {
                                mesh.MeshData = refined;
                                mesh.UpdateBounds();
                                ProgressDialog.Instance.Log($"Refined: {mesh.Name}");
                            });
                        }
                    }
                    EnqueueAction(() =>
                    {
                        TryAutoRefineGeoreferenceFromScene("DeepMeshPrior refinement");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (OperationCanceledException)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Log("DeepMeshPrior cancelled."));
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void RunGaussianSDFRefinement()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "Error: No mesh selected for GaussianSDF refinement.\n";
                return;
            }

            ProgressDialog.Instance.Start("GaussianSDF Refinement...", OperationType.Processing);
            Task.Run(async () => {
                try
                {
                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var refiner = new Deep3DStudio.Meshing.GaussianSDFRefiner();
                    foreach (var mesh in meshes)
                    {
                        var refined = await refiner.RefineMeshAsync(
                            mesh.MeshData,
                            (status, progress) => ProgressDialog.Instance.Update(progress, status),
                            cancellationToken);
                        if (refined != null)
                        {
                            EnqueueAction(() => {
                                mesh.MeshData = refined;
                                mesh.UpdateBounds();
                                ProgressDialog.Instance.Log($"Refined: {mesh.Name}");
                            });
                        }
                    }
                    EnqueueAction(() =>
                    {
                        TryAutoRefineGeoreferenceFromScene("GaussianSDF refinement");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (OperationCanceledException)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Log("GaussianSDF cancelled."));
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void RunTripoSFRefinement()
        {
            // TripoSF (SparseFlex) is a mesh refinement model
            // It takes an existing mesh and produces a higher-resolution refined mesh
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "Error: TripoSF requires a loaded mesh. Generate or load a mesh first.\n";
                return;
            }

            ProgressDialog.Instance.Start("TripoSF Mesh Refinement...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var triposf = new Deep3DStudio.Model.AIModels.TripoSFInference();
                    foreach (var mesh in meshes)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var refinedMesh = triposf.RefineMesh(mesh.MeshData, cancellationToken);
                        if (refinedMesh.Vertices.Count > 0)
                        {
                            EnqueueAction(() => {
                                mesh.MeshData = refinedMesh;
                                mesh.UpdateBounds();
                                ProgressDialog.Instance.Log($"TripoSF refinement complete: {mesh.Name}");
                            });
                        }
                    }
                    EnqueueAction(() =>
                    {
                        TryAutoRefineGeoreferenceFromScene("TripoSF refinement");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (OperationCanceledException)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Log("TripoSF cancelled."));
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private async void RunReconstruction(bool generateMesh = true, bool generateCloud = true)
        {
            if (_workflowInProgress)
            {
                _logBuffer += "AI workflow already in progress.\n";
                return;
            }

            if (_loadedImages.Count == 0)
            {
                _logBuffer += "Error: No images loaded.\n";
                return;
            }

            _workflowInProgress = true;
            var workflowNames = GetWorkflowNames();
            string resultLabel = "Reconstruction";

            try
            {
                ProgressDialog.Instance.Start($"Running {workflowNames[_selectedWorkflow]}...", OperationType.Processing);
                SceneResult? result = null;

                await Task.Run(async () =>
                {
                    WorkflowPipeline pipeline;

                    // First workflow option uses the engine from settings
                    if (_selectedWorkflow == 0)
                    {
                        // Use the reconstruction method from settings
                        pipeline = IniSettings.Instance.ReconstructionMethod switch
                        {
                            ReconstructionMethod.Mast3r => WorkflowPipeline.ImageToMast3rToMesh,
                            ReconstructionMethod.Must3r => WorkflowPipeline.ImageToMust3rToMesh,
                            ReconstructionMethod.FeatureMatching => WorkflowPipeline.ImageToSfM,
                            ReconstructionMethod.TripoSR => WorkflowPipeline.ImageToTripoSR,
                            ReconstructionMethod.Wonder3D => WorkflowPipeline.ImageToWonder3D,
                            _ => WorkflowPipeline.ImageToDust3rToMesh // Default to Dust3r
                        };
                    }
                    else if (workflowNames[_selectedWorkflow].Contains("SfM"))
                        pipeline = WorkflowPipeline.ImageToSfM;
                    else if (workflowNames[_selectedWorkflow].Contains("TripoSR"))
                        pipeline = WorkflowPipeline.ImageToTripoSR;
                    else if (workflowNames[_selectedWorkflow].Contains("LGM"))
                        pipeline = WorkflowPipeline.ImageToLGM;
                    else if (workflowNames[_selectedWorkflow].Contains("Wonder3D"))
                        pipeline = WorkflowPipeline.ImageToWonder3D;
                    else
                        pipeline = WorkflowPipeline.ImageToDust3rToMesh;

                    resultLabel = pipeline.Name;

                    // Convert ProjectImage list to string list
                    var imagePaths = _loadedImages.Select(i => i.FilePath).ToList();

                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    result = await AIModelManager.Instance.ExecuteWorkflowAsync(pipeline, imagePaths, null, (s, p) =>
                    {
                        ProgressDialog.Instance.Update(p, s);
                    }, cancellationToken);
                });
                EnqueueAction(() =>
                {
                    try
                    {
                        if (result != null)
                        {
                            bool hasPointCloud = HasPointCloudData(result);
                            bool hasTriangulatedMesh = HasTriangulatedMesh(result);
                            if (!hasPointCloud)
                            {
                                ProgressDialog.Instance.Fail(new Exception(
                                    $"Reconstruction completed but no geometry was generated. {DescribeGeometry(result)}"));
                                return;
                            }

                            if (!hasTriangulatedMesh)
                            {
                                ApplyPointCloudResultToScene(result, resultLabel);
                                ProgressDialog.Instance.Log($"Reconstruction complete. Result classified as point cloud: {result.Meshes.Count} cloud(s), {result.Poses.Count} camera(s).");
                            }
                            else
                            {
                                MeshObject? firstMeshObject = null;
                                int meshCount = 0;

                                foreach (var mesh in result.Meshes)
                                {
                                    if (mesh.Vertices.Count > 0 && mesh.Indices.Count >= 3)
                                    {
                                        var obj = new MeshObject($"Reconstructed Mesh {meshCount + 1}", mesh);
                                        _sceneGraph.AddObject(obj);
                                        if (firstMeshObject == null)
                                            firstMeshObject = obj;
                                        meshCount++;
                                    }
                                }

                                if (firstMeshObject != null)
                                {
                                    _sceneGraph.Select(firstMeshObject);
                                    _viewport.FocusOnSelection();
                                }

                                // Populate depth maps for visualization
                                PopulateDepthData(result);
                                TryAutoRefineGeoreferenceFromScene("reconstruction mesh result");

                                ProgressDialog.Instance.Log($"Reconstruction complete. Result classified as mesh: {meshCount} mesh(es).");
                            }
                            if (!(ProgressDialog.Instance.CancellationTokenSource?.IsCancellationRequested ?? false))
                                ProgressDialog.Instance.Complete();
                        }
                        else
                        {
                            // If no result but no exception, maybe cancelled or empty?
                            if (ProgressDialog.Instance.State == ProgressState.Running)
                                ProgressDialog.Instance.Fail(new Exception("Unknown failure: No result returned."));
                        }
                    }
                    finally
                    {
                        _workflowInProgress = false;
                    }
                });
            }
            catch (OperationCanceledException)
            {
                EnqueueAction(() =>
                {
                    ProgressDialog.Instance.Log("Reconstruction cancelled.");
                    _workflowInProgress = false;
                });
            }
            catch (Exception ex)
            {
                EnqueueAction(() =>
                {
                    ProgressDialog.Instance.Fail(ex);
                    _workflowInProgress = false;
                });
            }
        }

        /// <summary>
        /// Run a single workflow step standalone (without running the full workflow).
        /// This allows users to manually control each step of the pipeline.
        /// </summary>
        private async void RunSingleStep(WorkflowStep step)
        {
            if (step == WorkflowStep.PoissonReconstruction)
            {
                RunMeshFromSelectedPointClouds();
                return;
            }

            if (_workflowInProgress)
            {
                _logBuffer += $"AI workflow already in progress. Cannot start {GetStepDisplayName(step)}.\n";
                return;
            }

            // Validate prerequisites for each step
            switch (step)
            {
                case WorkflowStep.Dust3rReconstruction:
                case WorkflowStep.Mast3rReconstruction:
                case WorkflowStep.Must3rReconstruction:
                case WorkflowStep.SfMReconstruction:
                    if (_loadedImages.Count < 2)
                    {
                        ShowError("Need More Images", "Please load at least 2 images for reconstruction.");
                        return;
                    }
                    break;

                case WorkflowStep.TripoSRGeneration:
                case WorkflowStep.LGMGeneration:
                case WorkflowStep.Wonder3DGeneration:
                    if (_loadedImages.Count == 0)
                    {
                        ShowError("No Images", "Please load at least one image.");
                        return;
                    }
                    break;

                case WorkflowStep.NeRFRefinement:
                case WorkflowStep.DeepMeshPriorRefinement:
                case WorkflowStep.TripoSFRefinement:
                case WorkflowStep.GaussianSDFRefinement:
                case WorkflowStep.PoissonReconstruction:
                case WorkflowStep.MeshSmoothing:
                case WorkflowStep.MeshDecimation:
                case WorkflowStep.UniRigAutoRig:
                    if (_sceneGraph.GetAllObjects().Count == 0)
                    {
                        ShowError("No Geometry", "Please generate or load geometry first.");
                        return;
                    }
                    break;
            }

            string stepName = GetStepDisplayName(step);
            _workflowInProgress = true;

            try
            {
                ProgressDialog.Instance.Start($"Running {stepName}...", OperationType.Processing);
                SceneResult? result = null;
                var imagePaths = _loadedImages.Select(i => i.FilePath).ToList();

                // Get current scene result from existing meshes
                SceneResult? currentResult = null;
                var existingMeshes = _sceneGraph.GetAllObjects().OfType<MeshObject>().ToList();
                if (existingMeshes.Count > 0)
                {
                    currentResult = new SceneResult
                    {
                        Meshes = existingMeshes.Select(m => m.MeshData).ToList(),
                        Poses = new List<CameraPose>()
                    };
                }

                await Task.Run(async () =>
                {
                    // Create a single-step pipeline
                    var pipeline = new WorkflowPipeline
                    {
                        Name = stepName,
                        Steps = new List<WorkflowStep> { step }
                    };

                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    result = await AIModelManager.Instance.ExecuteWorkflowAsync(pipeline, imagePaths, currentResult, (s, p) =>
                    {
                        ProgressDialog.Instance.Update(p, s);
                    }, cancellationToken);
                });
                EnqueueAction(() =>
                {
                    try
                    {
                        if (result != null)
                        {
                            bool hasPointCloud = HasPointCloudData(result);
                            bool hasTriangulatedMesh = HasTriangulatedMesh(result);
                            if (!hasPointCloud)
                            {
                                ProgressDialog.Instance.Fail(new Exception(
                                    $"{stepName} completed but no geometry was generated. {DescribeGeometry(result)}"));
                                return;
                            }

                            bool isReconstructionStep =
                                step == WorkflowStep.Dust3rReconstruction ||
                                step == WorkflowStep.Mast3rReconstruction ||
                                step == WorkflowStep.Must3rReconstruction ||
                                step == WorkflowStep.SfMReconstruction;

                            if (isReconstructionStep || !hasTriangulatedMesh)
                            {
                                ApplyPointCloudResultToScene(result, stepName);
                                ProgressDialog.Instance.Log($"{stepName} complete. Added {result.Meshes.Count} point cloud(s).");
                                if (!(ProgressDialog.Instance.CancellationTokenSource?.IsCancellationRequested ?? false))
                                    ProgressDialog.Instance.Complete();
                                return;
                            }

                            foreach (var mesh in result.Meshes)
                            {
                                if (mesh.Vertices.Count > 0 && mesh.Indices.Count >= 3)
                                {
                                    var obj = new MeshObject($"{stepName} Result", mesh);
                                    _sceneGraph.AddObject(obj);
                                }
                            }

                            PopulateDepthData(result);
                            TryAutoRefineGeoreferenceFromScene(stepName);
                            ProgressDialog.Instance.Log($"{stepName} complete. Added {result.Meshes.Count} objects.");
                            if (!(ProgressDialog.Instance.CancellationTokenSource?.IsCancellationRequested ?? false))
                                ProgressDialog.Instance.Complete();
                        }
                        else
                        {
                            if (ProgressDialog.Instance.State == ProgressState.Running)
                                ProgressDialog.Instance.Fail(new Exception($"{stepName} failed: No result returned."));
                        }
                    }
                    finally
                    {
                        _workflowInProgress = false;
                    }
                });
            }
            catch (Exception ex)
            {
                EnqueueAction(() =>
                {
                    ProgressDialog.Instance.Fail(ex);
                    _workflowInProgress = false;
                });
            }
        }

        /// <summary>
        /// Returns true when at least one output mesh has valid triangle indices.
        /// </summary>
        private static bool HasTriangulatedMesh(SceneResult result)
        {
            return result.Meshes.Any(m => m.Vertices.Count > 0 && m.Indices.Count >= 3);
        }

        /// <summary>
        /// Returns true when at least one output contains points.
        /// </summary>
        private static bool HasPointCloudData(SceneResult result)
        {
            return result.Meshes.Any(m => m.Vertices.Count > 0);
        }

        private static string DescribeGeometry(SceneResult result)
        {
            int meshCount = result.Meshes.Count;
            int vertexCount = result.Meshes.Sum(m => m.Vertices.Count);
            int triangleMeshCount = result.Meshes.Count(m => m.Vertices.Count > 0 && m.Indices.Count >= 3);
            int poseCount = result.Poses.Count;
            return $"meshes={meshCount}, vertices={vertexCount}, triangleMeshes={triangleMeshCount}, poses={poseCount}";
        }

        /// <summary>
        /// Get a human-readable name for a workflow step
        /// </summary>
        private string GetStepDisplayName(WorkflowStep step)
        {
            return step switch
            {
                WorkflowStep.Dust3rReconstruction => "Dust3R Reconstruction",
                WorkflowStep.SfMReconstruction => "Feature Matching (SfM)",
                WorkflowStep.TripoSRGeneration => "TripoSR Generation",
                WorkflowStep.LGMGeneration => "LGM Generation",
                WorkflowStep.Wonder3DGeneration => "Wonder3D Generation",
                WorkflowStep.NeRFRefinement => "NeRF Refinement",
                WorkflowStep.DeepMeshPriorRefinement => "DeepMeshPrior Refinement",
                WorkflowStep.TripoSFRefinement => "TripoSF Refinement",
                WorkflowStep.GaussianSDFRefinement => "GaussianSDF Refinement",
                WorkflowStep.PoissonReconstruction => "Poisson Reconstruction",
                WorkflowStep.MarchingCubes => "Marching Cubes",
                WorkflowStep.MeshSmoothing => "Mesh Smoothing",
                WorkflowStep.MeshDecimation => "Mesh Decimation",
                WorkflowStep.UniRigAutoRig => "UniRig Auto-Rig",
                WorkflowStep.VoxelizePointCloud => "Voxelize Point Cloud",
                WorkflowStep.MergePointClouds => "Merge Point Clouds",
                WorkflowStep.AlignPointClouds => "Align Point Clouds",
                WorkflowStep.FilterPointCloud => "Filter Point Cloud",
                _ => step.ToString()
            };
        }

        private void ApplyPointCloudResultToScene(SceneResult result, string stepName)
        {
            _lastSceneResult = result;
            ClearReconstructionObjects();

            IniSettings.Instance.ShowPointCloud = true;
            IniSettings.Instance.ShowCameras = true;

            PointCloudObject? firstPc = null;
            for (int i = 0; i < result.Meshes.Count; i++)
            {
                var mesh = result.Meshes[i];
                if (mesh.Vertices.Count == 0) continue;
                var pcObj = new PointCloudObject($"{stepName} Points {i + 1}", mesh);
                _sceneGraph.AddObject(pcObj);
                if (firstPc == null) firstPc = pcObj;
            }

            if (firstPc != null)
            {
                _sceneGraph.Select(firstPc);
            }

            for (int i = 0; i < result.Poses.Count; i++)
            {
                var pose = result.Poses[i];
                var camObj = new CameraObject($"Camera {i + 1}", pose);
                _sceneGraph.AddObject(camObj);
            }

            PopulateDepthData(result);
            _viewport.FocusOnSelection();
            TryAutoRefineGeoreferenceFromScene(stepName);
        }

        private void ClearReconstructionObjects()
        {
            var toRemove = _sceneGraph.GetAllObjects()
                .Where(o => o is PointCloudObject || o is CameraObject)
                .ToList();

            foreach (var obj in toRemove)
            {
                _sceneGraph.RemoveObject(obj);
            }
        }

        private void RunMeshFromSelectedPointClouds()
        {
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (selectedPointClouds.Count == 0)
            {
                ShowError("No Point Cloud Selected", "Please select a point cloud to generate a mesh.");
                return;
            }

            int totalPoints = selectedPointClouds.Sum(pc => pc.PointCount);
            int visiblePoints = selectedPointClouds.Sum(pc => pc.VisiblePointCount);
            if (visiblePoints == 0 && totalPoints > 0)
            {
                ShowError("No Visible Points", "All selected point clouds currently expose 0 visible points. Increase the Visible Points slider before meshing.");
                return;
            }

            var meshingAlgo = IniSettings.Instance.MeshingAlgo;
            if (meshingAlgo == MeshingAlgorithm.LGM)
            {
                ShowError("LGM Is Image-Based", "Use the Image -> LGM workflow instead of point cloud meshing.");
                return;
            }

            ProgressDialog.Instance.Start("Meshing point cloud...", OperationType.Processing);
            Task.Run(async () =>
            {
                try
                {
                    var cancellationToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var baseMesh = GenerateMeshFromPointClouds(selectedPointClouds, MeshingAlgorithm.MarchingCubes);
                    MeshData? finalMesh = baseMesh;

                    switch (meshingAlgo)
                    {
                        case MeshingAlgorithm.DeepMeshPrior:
                            var deep = new Deep3DStudio.Meshing.DeepMeshPriorMesher();
                            finalMesh = await deep.RefineMeshAsync(
                                baseMesh,
                                (status, progress) => ProgressDialog.Instance.Update(progress, status),
                                cancellationToken);
                            break;
                        case MeshingAlgorithm.TripoSF:
                            using (var triposf = new Deep3DStudio.Model.AIModels.TripoSFInference())
                            {
                                cancellationToken.ThrowIfCancellationRequested();
                                finalMesh = triposf.RefineMesh(baseMesh, cancellationToken);
                            }
                            break;
                        case MeshingAlgorithm.GaussianSDF:
                            var gaussian = new Deep3DStudio.Meshing.GaussianSDFRefiner();
                            finalMesh = await gaussian.RefineMeshAsync(
                                baseMesh,
                                (status, progress) => ProgressDialog.Instance.Update(progress, status),
                                cancellationToken);
                            break;
                        default:
                            finalMesh = GenerateMeshFromPointClouds(selectedPointClouds, meshingAlgo);
                            break;
                    }

                    if (finalMesh == null || finalMesh.Vertices.Count == 0)
                    {
                        EnqueueAction(() => ProgressDialog.Instance.Fail(new Exception("Meshing returned empty geometry.")));
                        return;
                    }

                    EnqueueAction(() =>
                    {
                        var obj = new MeshObject("Reconstructed Mesh", finalMesh);
                        lock (_sceneGraph)
                        {
                            _sceneGraph.AddObject(obj);
                        }
                        _sceneGraph.Select(obj);
                        TryAutoRefineGeoreferenceFromScene("point cloud meshing");
                        _viewport.FocusOnSelection();
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void RunNeRFRefinementFromSelection()
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var selectedPointClouds = _sceneGraph.SelectedObjects.OfType<PointCloudObject>().ToList();
            if (selectedMeshes.Count == 0 && selectedPointClouds.Count == 0)
            {
                _logBuffer += "Error: No mesh or point cloud selected for NeRF refinement.\n";
                return;
            }

            var poses = _sceneGraph.GetObjectsOfType<CameraObject>()
                .Select(c => c.Pose)
                .Where(p => p != null)
                .Select(p => p!)
                .ToList();

            if (poses.Count == 0)
            {
                _logBuffer += "Error: No camera poses available for NeRF refinement.\n";
                return;
            }

            ProgressDialog.Instance.Start("NeRF Refinement...", OperationType.Processing);
            Task.Run(() =>
            {
                try
                {
                    var nerfToken = ProgressDialog.Instance.CancellationTokenSource?.Token ?? System.Threading.CancellationToken.None;
                    var inputMeshes = new List<MeshData>();
                    foreach (var mesh in selectedMeshes)
                    {
                        EnsureMeshColors(mesh.MeshData);
                        inputMeshes.Add(mesh.MeshData);
                    }

                    foreach (var pc in selectedPointClouds)
                    {
                        var mesh = ToMeshData(pc, visibleOnly: true);
                        EnsureMeshColors(mesh);
                        inputMeshes.Add(mesh);
                    }

                    var nerf = new VoxelGridNeRF();
                    nerf.InitializeFromMesh(inputMeshes);
                    bool cancelled = nerf.Train(poses, iterations: IniSettings.Instance.NeRFIterations, cancellationToken: nerfToken);
                    if (cancelled)
                    {
                        ProgressDialog.Instance.Log("NeRF cancelled. Returning partial mesh.");
                    }
                    var refined = nerf.GetMesh(GetMesher(IniSettings.Instance.MeshingAlgo));

                    if (refined.Vertices.Count == 0)
                    {
                        EnqueueAction(() => ProgressDialog.Instance.Fail(new Exception("NeRF returned empty geometry.")));
                        return;
                    }

                    EnqueueAction(() =>
                    {
                        var obj = new MeshObject("NeRF Mesh", refined);
                        lock (_sceneGraph)
                        {
                            _sceneGraph.AddObject(obj);
                        }
                        _sceneGraph.Select(obj);
                        TryAutoRefineGeoreferenceFromScene("NeRF refinement");
                        _viewport.FocusOnSelection();
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private MeshData GenerateMeshFromPointClouds(List<PointCloudObject> pointClouds, MeshingAlgorithm algorithm, int maxRes = 200)
        {
            var meshes = pointClouds.Select(pc => ToMeshData(pc, visibleOnly: true)).ToList();
            if (meshes.Sum(m => m.Vertices.Count) == 0)
                return new MeshData();

            var (grid, min, size) = VoxelizePoints(meshes, maxRes);
            var mesher = GetMesher(algorithm);
            return mesher.GenerateMesh(grid, min, size, 0.5f);
        }

        private MeshData ToMeshData(PointCloudObject pointCloud, bool visibleOnly = false)
        {
            return PointCloudOperations.ToMeshData(pointCloud, visibleOnly);
        }

        private void EnsureMeshColors(MeshData mesh)
        {
            while (mesh.Colors.Count < mesh.Vertices.Count)
            {
                mesh.Colors.Add(new Vector3(1f, 1f, 1f));
            }
        }

        private IMesher GetMesher(MeshingAlgorithm algo)
        {
            return algo switch
            {
                MeshingAlgorithm.Poisson => new PoissonMesher(),
                MeshingAlgorithm.GreedyMeshing => new GreedyMesher(),
                MeshingAlgorithm.SurfaceNets => new SurfaceNetsMesher(),
                MeshingAlgorithm.Blocky => new BlockMesher(),
                _ => new MarchingCubesMesher()
            };
        }

        private (float[,,], Vector3 min, float voxelSize) VoxelizePoints(List<MeshData> meshes, int maxRes = 200)
        {
            var min = new Vector3(float.MaxValue);
            var max = new Vector3(float.MinValue);
            foreach (var m in meshes)
            {
                foreach (var v in m.Vertices)
                {
                    min = Vector3.ComponentMin(min, v);
                    max = Vector3.ComponentMax(max, v);
                }
            }

            float voxelSize = 0.02f;
            int w = (int)((max.X - min.X) / voxelSize) + 5;
            int h = (int)((max.Y - min.Y) / voxelSize) + 5;
            int d = (int)((max.Z - min.Z) / voxelSize) + 5;

            if (w > maxRes)
            {
                voxelSize *= (w / (float)maxRes);
                w = maxRes;
                h = (int)((max.Y - min.Y) / voxelSize) + 5;
                d = (int)((max.Z - min.Z) / voxelSize) + 5;
            }

            float[,,] grid = new float[w, h, d];

            foreach (var m in meshes)
            {
                foreach (var v in m.Vertices)
                {
                    int x = (int)((v.X - min.X) / voxelSize);
                    int y = (int)((v.Y - min.Y) / voxelSize);
                    int z = (int)((v.Z - min.Z) / voxelSize);
                    if (x >= 0 && x < w && y >= 0 && y < h && z >= 0 && z < d)
                    {
                        grid[x, y, z] = 1.0f;
                    }
                }
            }

            float[,,] smooth = new float[w, h, d];
            for (int x = 1; x < w - 1; x++)
                for (int y = 1; y < h - 1; y++)
                    for (int z = 1; z < d - 1; z++)
                    {
                        if (grid[x, y, z] > 0)
                        {
                            smooth[x, y, z] = 1;
                            smooth[x + 1, y, z] = 1; smooth[x - 1, y, z] = 1;
                            smooth[x, y + 1, z] = 1; smooth[x, y - 1, z] = 1;
                            smooth[x, y, z + 1] = 1; smooth[x, y, z - 1] = 1;
                        }
                    }

            return (smooth, min, voxelSize);
        }

        private void PopulateDepthData(SceneResult result)
        {
            if (result.Poses.Count == 0 || result.Meshes.Count == 0) return;

            lock (_imageDepthThumbnails)
            {
                foreach (var tex in _imageDepthThumbnails.Values)
                    TextureLoader.DeleteTexture(tex);
                _imageDepthThumbnails.Clear();
            }

            // Combine meshes if multiple, similar to GTK implementation
            var combinedMesh = result.Meshes[0];
            if (result.Meshes.Count > 1)
            {
                combinedMesh = new MeshData();
                foreach (var m in result.Meshes)
                {
                    combinedMesh.Vertices.AddRange(m.Vertices);
                    combinedMesh.Colors.AddRange(m.Colors);
                }
            }

            // Generate depth maps for each pose
            // Parallelize this as it can be slow
            Parallel.ForEach(result.Poses, pose =>
            {
                try
                {
                    if (!TryResolvePoseImageSize(pose, out int poseWidth, out int poseHeight))
                    {
                        Logger.Warn($"Skipping depth map for {Path.GetFileName(pose.ImagePath)}: invalid image dimensions.");
                        return;
                    }

                    // Find corresponding ProjectImage
                    var pImg = _loadedImages.FirstOrDefault(i => Path.GetFullPath(i.FilePath) == Path.GetFullPath(pose.ImagePath));
                    if (pImg != null)
                    {
                        float focal = pose.GetEffectiveFocalLength();
                        var depthMap = ExtractDepthMap(combinedMesh, poseWidth, poseHeight, pose.WorldToCamera, focal);
                        if (HasRenderableDepthMap(depthMap))
                        {
                            pImg.DepthMap = depthMap;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Logger.Exception(ex, $"Failed to generate depth map for {pose.ImagePath}");
                }
            });
        }

        private float[,] ExtractDepthMap(MeshData mesh, int width, int height, Matrix4 worldToCamera, float focalLength = 0)
        {
            if (width <= 0 || height <= 0)
            {
                return new float[0, 0];
            }

            float[,] depthMap = new float[width, height];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    depthMap[x, y] = -1.0f;

            if (mesh.PixelToVertexIndex != null && mesh.PixelToVertexIndex.Length == width * height)
            {
                // Dense mesh logic
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pIdx = y * width + x;
                        int vertIdx = mesh.PixelToVertexIndex[pIdx];
                        if (vertIdx >= 0 && vertIdx < mesh.Vertices.Count)
                        {
                            var v = mesh.Vertices[vertIdx];
                            var vCam = Vector3.TransformPosition(v, worldToCamera);
                            depthMap[x, y] = Math.Abs(vCam.Z);
                        }
                    }
                }
            }
            else
            {
                // Sparse Point Cloud Logic (simplified splatting)
                float focal = focalLength > 0 ? focalLength : Math.Max(width, height) * 0.85f;
                float cx = width / 2.0f;
                float cy = height / 2.0f;
                int splatRadius = 3;

                foreach (var v in mesh.Vertices)
                {
                    var vCam = Vector3.TransformPosition(v, worldToCamera);
                    float depth = Math.Abs(vCam.Z);
                    if (depth < 0.1f) continue;

                    int px, py;
                    if (vCam.Z < 0)
                    {
                        px = (int)(-focal * vCam.X / vCam.Z + cx);
                        py = (int)(-focal * vCam.Y / vCam.Z + cy);
                    }
                    else
                    {
                        px = (int)(focal * vCam.X / vCam.Z + cx);
                        py = (int)(focal * vCam.Y / vCam.Z + cy);
                    }

                    for (int dy = -splatRadius; dy <= splatRadius; dy++)
                    {
                        for (int dx = -splatRadius; dx <= splatRadius; dx++)
                        {
                            if (dx*dx + dy*dy > splatRadius*splatRadius) continue;
                            int nx = px + dx;
                            int ny = py + dy;

                            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                            {
                                if (depthMap[nx, ny] < 0 || depth < depthMap[nx, ny])
                                {
                                    depthMap[nx, ny] = depth;
                                }
                            }
                        }
                    }
                }
            }
            return depthMap;
        }

        private unsafe int OnLogCallback(ImGuiInputTextCallbackData* data)
        {
            // Always update current selection
            _logSelectionStart = data->SelectionStart;
            _logSelectionEnd = data->SelectionEnd;

            // Update saved selection only if we have a valid selection
            if (data->SelectionStart != data->SelectionEnd)
            {
                _savedSelectionStart = data->SelectionStart;
                _savedSelectionEnd = data->SelectionEnd;
            }
            // Only clear saved selection if user left-clicks (intentional deselect) AND context menu is not open
            else if (ImGui.IsMouseClicked(ImGuiMouseButton.Left) && !ImGui.IsPopupOpen("LogContext"))
            {
                _savedSelectionStart = 0;
                _savedSelectionEnd = 0;
            }
            // If right-click happens, we do nothing here, preserving the last non-zero saved selection

            return 0;
        }

        private void CopySelectedLogText()
        {
            try
            {
                // Use saved selection values which persist through right-click
                int start = Math.Min(_savedSelectionStart, _savedSelectionEnd);
                int length = Math.Abs(_savedSelectionStart - _savedSelectionEnd);

                if (length > 0 && !string.IsNullOrEmpty(_logBuffer))
                {
                    // Convert to UTF-8 to handle indices correctly
                    byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(_logBuffer);

                    if (start >= 0 && start + length <= utf8.Length)
                    {
                        string selected = System.Text.Encoding.UTF8.GetString(utf8, start, length);
                        ClipboardString = selected;
                        Logger.Info($"Copied {length} bytes of log text to clipboard.");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Exception(ex, "Failed to copy selected text");
            }
        }

        private MeshObject? GetSelectedMeshObjectImGui()
        {
            return _sceneGraph.SelectedObjects.OfType<MeshObject>().FirstOrDefault();
        }

        private GroupObject FindOrCreateRiggingGroup()
        {
            var existing = _sceneGraph.GetObjectsOfType<GroupObject>()
                .FirstOrDefault(g => string.Equals(g.Name, "Rigging", StringComparison.OrdinalIgnoreCase));
            if (existing != null)
            {
                return existing;
            }

            var group = new GroupObject("Rigging");
            _sceneGraph.AddObject(group);
            return group;
        }

        private void SelectSkeletonObject(SkeletonObject skelObj)
        {
            _activeSkeletonObject = skelObj;
            _sceneGraph.ClearSelection();
            _sceneGraph.Select(skelObj, false);
            _isDirty = true;
        }

        private void OnCreateNewSkeletonImGui()
        {
            var selectedMesh = GetSelectedMeshObjectImGui();
            var position = selectedMesh?.GetCentroid() ?? Vector3.Zero;

            var skeleton = new SkeletonData { Name = "New Skeleton" };
            skeleton.AddJoint("Root", position);

            var skelObj = new SkeletonObject("Skeleton", skeleton);
            if (selectedMesh != null)
            {
                skelObj.TargetMesh = selectedMesh;
            }

            var riggingGroup = FindOrCreateRiggingGroup();
            _sceneGraph.AddObject(skelObj, riggingGroup);
            SelectSkeletonObject(skelObj);
            _logBuffer += "Created new manual skeleton.\n";
        }

        private void OnCreateHumanoidSkeletonImGui()
        {
            var selectedMesh = GetSelectedMeshObjectImGui();
            var position = selectedMesh?.GetCentroid() ?? Vector3.Zero;

            float scale = 1.0f;
            if (selectedMesh != null)
            {
                var (min, max) = selectedMesh.GetWorldBounds();
                float height = max.Y - min.Y;
                scale = height > 0.1f ? height : 1.0f;
            }

            var skeleton = SkeletonData.CreateHumanoidTemplate(position, scale);
            var skelObj = new SkeletonObject("Humanoid Skeleton", skeleton);
            if (selectedMesh != null)
            {
                skelObj.TargetMesh = selectedMesh;
            }

            var riggingGroup = FindOrCreateRiggingGroup();
            _sceneGraph.AddObject(skelObj, riggingGroup);
            SelectSkeletonObject(skelObj);
            _logBuffer += $"Created humanoid skeleton ({skeleton.Joints.Count} joints).\n";
        }

        private void OnAddJointImGui()
        {
            if (_activeSkeletonObject?.Skeleton == null)
            {
                _logBuffer += "No active skeleton. Create or select a skeleton first.\n";
                return;
            }

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();

            Joint? parent = selectedJoints.FirstOrDefault() ?? skeleton.RootJoint;
            Vector3 position = parent != null
                ? parent.Position + new Vector3(0, 0.1f, 0)
                : _activeSkeletonObject.Position;

            string name = $"Joint_{skeleton.Joints.Count}";
            var newJoint = skeleton.AddJoint(name, position, parent);
            skeleton.SelectJoint(newJoint, false);
            _isDirty = true;
            _logBuffer += $"Added joint '{name}'.\n";
        }

        private void OnAddBoneImGui()
        {
            if (_activeSkeletonObject?.Skeleton == null)
                return;

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();
            if (selectedJoints.Count < 2)
            {
                _logBuffer += "Select at least 2 joints to create a bone.\n";
                return;
            }

            int created = 0;
            for (int i = 0; i < selectedJoints.Count - 1; i++)
            {
                skeleton.AddBone(selectedJoints[i], selectedJoints[i + 1]);
                created++;
            }

            _isDirty = true;
            _logBuffer += $"Created {created} bone(s).\n";
        }

        private void OnDeleteSelectedJointsImGui()
        {
            if (_activeSkeletonObject?.Skeleton == null)
                return;

            var skeleton = _activeSkeletonObject.Skeleton;
            var selectedJoints = skeleton.GetSelectedJoints().ToList();
            if (selectedJoints.Count == 0)
                return;

            foreach (var joint in selectedJoints)
            {
                skeleton.RemoveJoint(joint);
            }

            _isDirty = true;
            _logBuffer += $"Deleted {selectedJoints.Count} joint(s).\n";
        }

        private static int GetJointDepth(Joint joint)
        {
            int depth = 0;
            var parent = joint.Parent;
            while (parent != null)
            {
                depth++;
                parent = parent.Parent;
            }
            return depth;
        }

        private void OnAutoRig()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "Error: No mesh selected for rigging.\n";
                return;
            }

            ProgressDialog.Instance.Start("Auto Rigging with UniRig...", OperationType.Processing);
            _ = Task.Run(async () => {
                try
                {
                    foreach (var mesh in meshes)
                    {
                        ProgressDialog.Instance.Update(0.1f, $"Rigging {mesh.Name}...");

                        // Try to use UniRig AI model first
                        var rigResult = await AIModelManager.Instance.RigMeshAsync(
                            mesh.MeshData.Vertices.ToArray(),
                            mesh.MeshData.Indices.ToArray(),
                            msg => EnqueueAction(() => ProgressDialog.Instance.Log(msg)));

                        SkeletonData skeleton;

                        if (rigResult != null && rigResult.Success && rigResult.JointPositions?.Length > 0)
                        {
                            // Use UniRig result - use the existing helper method
                            ProgressDialog.Instance.Log($"UniRig generated {rigResult.JointPositions.Length} joints.");
                            skeleton = SkeletonData.FromRigResult(rigResult);
                        }
                        else
                        {
                            // Fall back to humanoid template
                            ProgressDialog.Instance.Log("UniRig not available, using humanoid template...");

                            var (min, max) = mesh.GetWorldBounds();
                            var center = (min + max) * 0.5f;
                            var size = max - min;
                            float height = Math.Max(size.Y, 0.1f);
                            float scale = height;
                            var rootPosition = new Vector3(center.X, min.Y + height * 0.5f, center.Z);
                            skeleton = SkeletonData.CreateHumanoidTemplate(rootPosition, scale);
                        }

                        // Create skeleton object and add to scene
                        var skelObj = new SkeletonObject($"Rig_{mesh.Name}", skeleton);
                        skelObj.TargetMesh = mesh;
                        skelObj.Position = Vector3.Zero;

                        EnqueueAction(() =>
                        {
                            var riggingGroup = FindOrCreateRiggingGroup();
                            _sceneGraph.AddObject(skelObj, riggingGroup);
                            SelectSkeletonObject(skelObj);
                            ProgressDialog.Instance.Log($"Created skeleton for '{mesh.Name}' with {skeleton.Joints.Count} joints.");
                        });
                    }
                    EnqueueAction(() => ProgressDialog.Instance.Complete());
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        #endregion
    }
}
