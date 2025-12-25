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

namespace Deep3DStudio
{
    public class MainWindow : GameWindow
    {
        private ImGuiController _controller;
        private ThreeDView _viewport;
        private SceneGraph _sceneGraph;
        private ImGuiIconFactory _iconFactory;

        // State
        private int _selectedWorkflow = 0;
        private int _selectedQuality = 1;
        private string[] _workflows = { "Dust3r (Multi-View)", "TripoSR (Single Image)", "LGM (Gaussian)", "Wonder3D" };
        private string[] _qualities = { "Fast", "Balanced", "High" };
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
        private int _selectedImageIndex = -1;

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
        private float _verticalToolbarWidth = 42;

        // View State
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
        private bool _showExitConfirmation = false;

        // Popup Management
        private string? _popupToOpen = null;

        // Threading
        private readonly System.Collections.Concurrent.ConcurrentQueue<Action> _pendingActions = new System.Collections.Concurrent.ConcurrentQueue<Action>();

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

            _controller = new ImGuiController(ClientSize.X, ClientSize.Y);

            // Configure ImGui style
            ConfigureImGuiStyle();

            // Init Python service hook
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
            colors[(int)ImGuiCol.PopupBg] = new System.Numerics.Vector4(0.10f, 0.10f, 0.10f, 0.95f);
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

        protected override void OnClosing(System.ComponentModel.CancelEventArgs e)
        {
            if (_isDirty && !_showExitConfirmation)
            {
                e.Cancel = true;
                _showExitConfirmation = true;
            }
            base.OnClosing(e);
        }

        protected override void OnResize(ResizeEventArgs e)
        {
            base.OnResize(e);
            GL.Viewport(0, 0, ClientSize.X, ClientSize.Y);
            _controller.WindowResized(ClientSize.X, ClientSize.Y);
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
            else if (ext == ".obj" || ext == ".ply" || ext == ".glb" || ext == ".stl")
            {
                ProgressDialog.Instance.Start($"Importing {Path.GetFileName(file)}...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var mesh = MeshImporter.Load(file);
                        if (mesh != null)
                        {
                            var obj = new MeshObject(Path.GetFileName(file), mesh);
                            lock (_sceneGraph)
                            {
                                _sceneGraph.AddObject(obj);
                            }
                            ProgressDialog.Instance.Log($"Imported mesh: {Path.GetFileName(file)}");
                            ProgressDialog.Instance.Complete();
                        }
                        else
                        {
                            throw new Exception("Failed to load mesh data.");
                        }
                    }
                    catch (Exception ex)
                    {
                        ProgressDialog.Instance.Fail(ex);
                    }
                });
            }
            else if (ext == ".xyz")
            {
                ProgressDialog.Instance.Start($"Importing {Path.GetFileName(file)}...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        var pc = PointCloudImporter.Load(file);
                        if (pc != null)
                        {
                            lock (_sceneGraph)
                            {
                                _sceneGraph.AddObject(pc);
                            }
                            ProgressDialog.Instance.Log($"Imported point cloud: {Path.GetFileName(file)}");
                            ProgressDialog.Instance.Complete();
                        }
                        else
                        {
                            throw new Exception("Failed to load point cloud data.");
                        }
                    }
                    catch (Exception ex)
                    {
                        ProgressDialog.Instance.Fail(ex);
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
                    _viewport.UpdateInput(mouseState, keyboardState, (float)e.Time, ClientSize.X, ClientSize.Y);
                }

                // Render Viewport
                float vpX = _showVerticalToolbar ? _verticalToolbarWidth : 0;
                float vpY = _toolbarHeight + 20; // +20 for menu bar
                float vpW = ClientSize.X - vpX - (_showRightPanel ? _rightPanelWidth : 0);
                float vpH = ClientSize.Y - vpY - (_showLogPanel ? _logPanelHeight : 0);

                if (_showLeftPanel)
                {
                    vpX += _leftPanelWidth;
                    vpW -= _leftPanelWidth;
                }

                _viewport.Render((int)vpX, (int)vpY, (int)vpW, (int)vpH, ClientSize.X, ClientSize.Y);
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

            // Clean up thumbnails
            foreach (var thumb in _imageThumbnails.Values)
            {
                TextureLoader.DeleteTexture(thumb);
            }
            _imageThumbnails.Clear();

            if (_previewTexture > 0)
                TextureLoader.DeleteTexture(_previewTexture);

            _iconFactory?.Dispose();
        }

        #region Error Display

        private void ShowError(string title, string message, Exception? ex = null)
        {
            _errorTitle = title;
            _errorMessage = message;
            _errorStackTrace = ex?.ToString() ?? "";
            _errorExpanded = false;
            _showError = true;
            _logBuffer += $"ERROR: {title} - {message}\n";
            if (ex != null)
                _logBuffer += $"  Details: {ex.Message}\n";
        }

        private void RenderErrorDialog()
        {
            if (!_showError) return;

            ImGui.SetNextWindowSize(new System.Numerics.Vector2(600, 300), ImGuiCond.FirstUseEver);
            ImGui.SetNextWindowPos(
                new System.Numerics.Vector2(ClientSize.X / 2 - 300, ClientSize.Y / 2 - 150),
                ImGuiCond.FirstUseEver);

            // Red title bar color
            ImGui.PushStyleColor(ImGuiCol.TitleBgActive, new System.Numerics.Vector4(0.8f, 0.2f, 0.2f, 1.0f));
            ImGui.PushStyleColor(ImGuiCol.TitleBg, new System.Numerics.Vector4(0.6f, 0.1f, 0.1f, 1.0f));

            if (ImGui.Begin($"Error: {_errorTitle}###ErrorWindow", ref _showError, ImGuiWindowFlags.NoCollapse))
            {
                ImGui.PopStyleColor(2);

                // Error icon and message
                ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(1.0f, 0.4f, 0.4f, 1.0f));
                ImGui.TextWrapped(_errorMessage);
                ImGui.PopStyleColor();

                if (!string.IsNullOrEmpty(_errorStackTrace))
                {
                    ImGui.Separator();

                    if (ImGui.CollapsingHeader("Stack Trace (click to expand)", ref _errorExpanded))
                    {
                        ImGui.BeginChild("StackTrace", new System.Numerics.Vector2(0, 150), ImGuiChildFlags.Borders);
                        ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(0.7f, 0.7f, 0.7f, 1.0f));
                        ImGui.TextUnformatted(_errorStackTrace);
                        ImGui.PopStyleColor();
                        ImGui.EndChild();
                    }
                }

                ImGui.Separator();

                if (ImGui.Button("OK", new System.Numerics.Vector2(100, 30)))
                {
                    _showError = false;
                }

                ImGui.SameLine();

                if (ImGui.Button("Copy to Clipboard", new System.Numerics.Vector2(150, 30)))
                {
                    var text = $"{_errorTitle}\n\n{_errorMessage}\n\n{_errorStackTrace}";
                    ImGui.SetClipboardText(text);
                }

                ImGui.End();
            }
            else
            {
                ImGui.PopStyleColor(2);
            }
        }

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
            string authorLine3 = "2025";
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

            string version = "Version 1.0.0";
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
            if (_popupToOpen != null)
            {
                ImGui.OpenPopup(_popupToOpen);
                _popupToOpen = null;
            }

            // Exit Confirmation
            if (_showExitConfirmation)
            {
                ImGui.OpenPopup("Really Exit?");

                // Center the modal
                var io = ImGui.GetIO();
                ImGui.SetNextWindowPos(new System.Numerics.Vector2(io.DisplaySize.X * 0.5f, io.DisplaySize.Y * 0.5f), ImGuiCond.Always, new System.Numerics.Vector2(0.5f, 0.5f));

                if (ImGui.BeginPopupModal("Really Exit?", ref _showExitConfirmation, ImGuiWindowFlags.AlwaysAutoResize))
                {
                    ImGui.Text("You have unsaved changes. Are you sure you want to exit?");
                    ImGui.Separator();

                    if (ImGui.Button("Yes, Exit", new System.Numerics.Vector2(120, 0)))
                    {
                        ImGui.CloseCurrentPopup();
                        _isDirty = false; // Prevent loop
                        Close();
                    }
                    ImGui.SetItemDefaultFocus();
                    ImGui.SameLine();
                    if (ImGui.Button("Cancel", new System.Numerics.Vector2(120, 0)))
                    {
                        ImGui.CloseCurrentPopup();
                        _showExitConfirmation = false;
                    }

                    ImGui.EndPopup();
                }
            }

            // Error Dialog (renders on top)
            RenderErrorDialog();

            // Main Menu
            RenderMainMenu();

            // Top Toolbar
            RenderTopToolbar();

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

            // Dialogs
            if (_showSettings) DrawSettingsWindow();
            if (_showAbout) DrawAboutWindow();
            if (_showImagePreview) DrawImagePreviewWindow();
            if (_showDecimateDialog) DrawDecimateDialog();
            if (_showSmoothDialog) DrawSmoothDialog();
            if (_showOptimizeDialog) DrawOptimizeDialog();
            if (_showMergeDialog) DrawMergeDialog();
            if (_showAlignDialog) DrawAlignDialog();
            if (_showCleanupDialog) DrawCleanupDialog();
            if (_showBakeDialog) DrawBakeDialog();
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
                        if (ImGui.MenuItem("LGM (High Quality)")) RunAIModel("LGM");
                        if (ImGui.MenuItem("Wonder3D (Multi-View)")) RunAIModel("Wonder3D");
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

                    ImGui.Separator();
                    if (ImGui.MenuItem("AI Model Settings...")) _showSettings = true;
                    ImGui.EndMenu();
                }

                // Window Menu
                if (ImGui.BeginMenu("Window"))
                {
                    if (ImGui.MenuItem("Left Panel", "", _showLeftPanel)) _showLeftPanel = !_showLeftPanel;
                    if (ImGui.MenuItem("Right Panel", "", _showRightPanel)) _showRightPanel = !_showRightPanel;
                    if (ImGui.MenuItem("Log Panel", "", _showLogPanel)) _showLogPanel = !_showLogPanel;
                    if (ImGui.MenuItem("Vertical Toolbar", "", _showVerticalToolbar)) _showVerticalToolbar = !_showVerticalToolbar;
                    ImGui.Separator();
                    if (ImGui.MenuItem("Full Viewport Mode"))
                    {
                        _showLeftPanel = false;
                        _showRightPanel = false;
                        _showLogPanel = false;
                        _showVerticalToolbar = false;
                    }
                    if (ImGui.MenuItem("Restore All Panels"))
                    {
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

        private void RenderTopToolbar()
        {
            float menuBarHeight = 20;
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, menuBarHeight));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.X, _toolbarHeight));

            ImGui.Begin("##Toolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
                       ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings);
            {
                var size = new System.Numerics.Vector2(22, 22);

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

                // Workflow Selection
                ImGui.Text("Workflow:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(170);
                ImGui.Combo("##Workflow", ref _selectedWorkflow, _workflows, _workflows.Length);
                ImGui.SameLine();

                ImGui.Text("Quality:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(100);
                ImGui.Combo("##Quality", ref _selectedQuality, _qualities, _qualities.Length);
                ImGui.SameLine();

                // Run Button
                DrawToolbarButton("##Run", IconType.Run, false, () => RunReconstruction(), "Run Reconstruction", size);
                ImGui.SameLine();
                DrawToolbarButton("##Points", IconType.Cloud, false, () => RunReconstruction(false, true), "Generate Point Cloud", size);
                ImGui.SameLine();
                DrawToolbarButton("##Mesh", IconType.Mesh, false, () => RunReconstruction(true, false), "Generate Mesh from Points", size);

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

                // Fullscreen toggle
                bool isFullscreen = WindowState == OpenTK.Windowing.Common.WindowState.Fullscreen;
                DrawToolbarButton("##Fullscreen", IconType.Fullscreen, isFullscreen, ToggleFullscreen,
                    isFullscreen ? "Exit Fullscreen (F11)" : "Fullscreen (F11)", size);
            }
            ImGui.End();
        }

        private void DrawToolbarButton(string id, IconType icon, bool active, Action onClick, string tooltip, System.Numerics.Vector2 size)
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

            if (active)
            {
                ImGui.PopStyleColor(2);
            }

            if (ImGui.IsItemHovered())
            {
                ImGui.SetTooltip(tooltip);
            }
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

        private void RenderVerticalToolbar()
        {
            float menuBarHeight = 20;
            float startY = menuBarHeight + _toolbarHeight;
            float height = ClientSize.Y - startY - (_showLogPanel ? _logPanelHeight : 0);

            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, startY));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(_verticalToolbarWidth, height));

            ImGui.Begin("##VToolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
                       ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoSavedSettings);
            {
                var size = new System.Numerics.Vector2(22, 22);

                // Focus
                if (ImGui.ImageButton("##Focus", _iconFactory.GetIcon(IconType.Focus), size))
                    _viewport.FocusOnSelection();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Focus on Selection (F)");

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // Processing Operations Section
                ImGui.TextDisabled("AI");

                if (ImGui.ImageButton("##GenCloud", _iconFactory.GetIcon(IconType.PointCloudGen), size))
                    RunReconstruction(false, true);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Generate Point Cloud");

                if (ImGui.ImageButton("##GenMesh", _iconFactory.GetIcon(IconType.MeshGen), size))
                    RunReconstruction(true, false);
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Generate Mesh");

                if (ImGui.ImageButton("##AutoRig", _iconFactory.GetIcon(IconType.Rig), size))
                    OnAutoRig();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Auto Rig (UniRig)");

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // Mesh Operations Section
                ImGui.TextDisabled("Mesh");

                if (ImGui.ImageButton("##Decimate", _iconFactory.GetIcon(IconType.Decimate), size))
                    OnDecimate();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Decimate Mesh (50%)");

                if (ImGui.ImageButton("##Optimize", _iconFactory.GetIcon(IconType.Optimize), size))
                    OnOptimize();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Optimize Mesh");

                if (ImGui.ImageButton("##Clean", _iconFactory.GetIcon(IconType.Clean), size))
                    OnCleanup();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Cleanup Mesh");

                if (ImGui.ImageButton("##Bake", _iconFactory.GetIcon(IconType.Bake), size))
                    OnBakeTextures();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Bake Textures");

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
            ImGui.End();
        }

        private void RenderLeftPanel()
        {
            float menuBarHeight = 20;
            float startX = _showVerticalToolbar ? _verticalToolbarWidth : 0;
            float startY = menuBarHeight + _toolbarHeight;
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
            ImGui.Text($"Loaded: {_loadedImages.Count}");

            if (ImGui.Button("Add Images..."))
            {
                OnAddImages();
            }
            ImGui.SameLine();
            if (ImGui.Button("Clear"))
            {
                ClearImages();
            }

            ImGui.Separator();

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
                lock (_imageThumbnails)
                {
                    _imageThumbnails.TryGetValue(path, out thumbTex);
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
                    if (ImGui.Button($"[{displayName.Substring(0, Math.Min(6, displayName.Length))}...]", new System.Numerics.Vector2(thumbSize, thumbSize)))
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
                    _previewImagePath = path;
                    _showImagePreview = true;
                    if (_previewTexture > 0)
                    {
                        TextureLoader.DeleteTexture(_previewTexture);
                    }
                    _previewTexture = TextureLoader.LoadTextureFromFile(path);
                }

                // Context menu
                if (ImGui.BeginPopupContextItem())
                {
                    if (ImGui.MenuItem("Preview"))
                    {
                        _previewImagePath = path;
                        _showImagePreview = true;
                        if (_previewTexture > 0) TextureLoader.DeleteTexture(_previewTexture);
                        _previewTexture = TextureLoader.LoadTextureFromFile(path);
                    }

                    bool hasDepth = pImg.DepthMap != null;
                    if (ImGui.MenuItem("Depth View", "", false, hasDepth))
                    {
                        _previewImagePath = path;
                        _showImagePreview = true;
                        if (_previewTexture > 0) TextureLoader.DeleteTexture(_previewTexture);
                        // Generate depth visualization
                        using (var bmp = Deep3DStudio.Model.ImageUtils.ColorizeDepthMap(pImg.DepthMap))
                        {
                            string temp = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".png");
                            try
                            {
                                using (var image = SkiaSharp.SKImage.FromBitmap(bmp))
                                using (var data = image.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100))
                                using (var stream = File.OpenWrite(temp))
                                {
                                    data.SaveTo(stream);
                                }
                                _previewTexture = TextureLoader.LoadTextureFromFile(temp);
                            }
                            finally
                            {
                                try { File.Delete(temp); } catch { }
                            }
                        }
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

        private void ClearImages()
        {
            foreach (var kv in _imageThumbnails)
            {
                TextureLoader.DeleteTexture(kv.Value);
            }
            _imageThumbnails.Clear();
            _loadedImages.Clear();
            _selectedImageIndex = -1;
        }

        private void RenderSceneGraph()
        {
            int i = 0;
            foreach (var obj in _sceneGraph.GetVisibleObjects())
            {
                bool selected = obj.Selected;
                string name = obj.Name ?? $"Object {obj.Id}";
                string icon = obj is MeshObject ? "[M] " : obj is PointCloudObject ? "[P] " : "[O] ";

                if (_renamingObject == obj)
                {
                    ImGui.SetKeyboardFocusHere();
                    if (ImGui.InputText($"##renameObj{i}", ref _renameBuffer, 64, ImGuiInputTextFlags.EnterReturnsTrue | ImGuiInputTextFlags.AutoSelectAll))
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
                    if (ImGui.Selectable($"{icon}{name}##{i}", selected))
                    {
                        if (!ImGui.GetIO().KeyCtrl) _sceneGraph.ClearSelection();
                        _sceneGraph.Select(obj, !selected);
                    }

                    // Context menu
                    if (ImGui.BeginPopupContextItem())
                    {
                        if (ImGui.MenuItem("Rename"))
                        {
                            _renamingObject = obj;
                            _renameBuffer = obj.Name ?? "";
                        }
                        if (ImGui.MenuItem("Focus")) _viewport.FocusOnObject(obj);
                        if (ImGui.MenuItem("Delete")) _sceneGraph.RemoveObject(obj);
                        if (ImGui.MenuItem("Duplicate")) OnDuplicateObject(obj);
                        ImGui.EndPopup();
                    }
                }

                i++;
            }

            if (_sceneGraph.ObjectCount == 0)
            {
                ImGui.TextDisabled("No objects in scene");
            }
        }

        private void RenderRightPanel()
        {
            float menuBarHeight = 20;
            float startY = menuBarHeight + _toolbarHeight;
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
            string[] modes = { "Select", "Delete", "Paint", "Weld", "Extrude" };
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

            if (DrawIconTextButton("##RigAuto", IconType.Rig, "Auto Rig (UniRig)", iconSize))
            {
                OnAutoRig();
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
            ImGui.TextDisabled("Select a mesh and click");
            ImGui.TextDisabled("'Auto Rig' to generate skeleton");

            // Show selected skeleton info if any
            var selectedSkeletons = _sceneGraph.SelectedObjects.OfType<SkeletonObject>().ToList();
            if (selectedSkeletons.Count > 0)
            {
                ImGui.Separator();
                ImGui.Text($"Selected: {selectedSkeletons[0].Name}");
                ImGui.Text($"Joints: {selectedSkeletons[0].Skeleton?.Joints.Count ?? 0}");
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
                    obj.Scale = new Vector3(scale.X, scale.Y, scale.Z);

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
                    ImGui.Text($"Points: {pc.PointCount:N0}");
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
                        ImGui.SetClipboardText(_logBuffer);
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
                    ImGui.SetScrollHereY(1.0f);
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
                            ImGui.SetClipboardText(_logBuffer);
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
            var windowPos = new System.Numerics.Vector2(xPos, _toolbarHeight + 30);

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
                            // Dust3r doesn't have many exposed parameters in IniSettings,
                            // but usually runs based on reconstruction method.
                            ImGui.TextWrapped("Dust3r is the default multi-view reconstruction engine. It requires a GPU with significant VRAM for best performance.");
                        }

                        if (ImGui.CollapsingHeader("TripoSR (Single Image)"))
                        {
                            int res = s.TripoSRResolution;
                            if (ImGui.SliderInt("Resolution", ref res, 128, 1024)) s.TripoSRResolution = res;

                            int mcRes = s.TripoSRMarchingCubesRes;
                            if (ImGui.SliderInt("Marching Cubes Res", ref mcRes, 32, 512)) s.TripoSRMarchingCubesRes = mcRes;

                            string path = s.TripoSRModelPath;
                            if (ImGui.InputText("Model Path##TripoSR", ref path, 256)) s.TripoSRModelPath = path;
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
                            int iter = s.NeRFIterations;
                            if (ImGui.InputInt("Iterations", ref iter)) s.NeRFIterations = iter;

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

                string version = "Version 1.0.0";
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
                ImGui.Text("2025");

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
                    ImGui.Image((IntPtr)_previewTexture, avail);
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
            _sceneGraph.Clear();
            ClearImages();
            _logBuffer = "";
            _currentProjectPath = "";
            _isDirty = false;
            UpdateTitle();
            _logBuffer += "New project created.\n";
        }

        private void OnOpenProject(string? path = null)
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

        private void OnSaveProject()
        {
            if (string.IsNullOrEmpty(_currentProjectPath))
            {
                OnSaveProjectAs();
                return;
            }

            ProgressDialog.Instance.Start("Saving Project...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    // Copy lists to avoid collection modification exceptions if scene changes during save
                    SceneGraph graphSnapshot;
                    List<string> imagesSnapshot;
                    lock (_sceneGraph)
                    {
                        CrossProjectManager.SaveProject(_currentProjectPath, _sceneGraph, _loadedImages);
                    }

                    EnqueueAction(() => {
                        _isDirty = false;
                        UpdateTitle();
                        ProgressDialog.Instance.Log($"Project saved: {_currentProjectPath}");
                        ProgressDialog.Instance.Complete();
                    });
                }
                catch (Exception ex)
                {
                    ProgressDialog.Instance.Fail(ex);
                }
            });
        }

        private void OnSaveProjectAs()
        {
            var result = Nfd.SaveDialog(out string path, new Dictionary<string, string> { { "Deep3D Project", "d3d" } });
            if (result == NfdStatus.Ok && !string.IsNullOrEmpty(path))
            {
                if (!path.EndsWith(".d3d")) path += ".d3d";
                _currentProjectPath = path;
                OnSaveProject();
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
                ProgressDialog.Instance.Start("Exporting Mesh...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        MeshExporter.Save(path, meshes[0].MeshData);
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
                ProgressDialog.Instance.Start("Exporting Point Cloud...", OperationType.ImportExport);
                Task.Run(() => {
                    try
                    {
                        PointCloudExporter.Save(path, pcs[0]);
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

        #endregion

        #region Mesh Operations

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

             if (ImGui.BeginPopupModal("Decimate Mesh", ref _showDecimateDialog, ImGuiWindowFlags.AlwaysAutoResize))
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
            if (ImGui.BeginPopupModal("Smooth Mesh", ref _showSmoothDialog, ImGuiWindowFlags.AlwaysAutoResize))
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
            if (ImGui.BeginPopupModal("Optimize Mesh", ref _showOptimizeDialog, ImGuiWindowFlags.AlwaysAutoResize))
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
        private float _mergeDist = 0.001f;

        private void OnMerge()
        {
            _showMergeDialog = true;
            _popupToOpen = "Merge Objects";
        }

        private void DrawMergeDialog()
        {
            if (!_showMergeDialog) return;
            if (ImGui.BeginPopupModal("Merge Objects", ref _showMergeDialog, ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool canMerge = _sceneGraph.SelectedObjects.Count >= 2;

                ImGui.InputFloat("Weld Distance", ref _mergeDist, 0.0001f, 0.001f, "%.5f");
                ImGui.Separator();

                if (!canMerge)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "Select at least 2 objects");
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
            if (ImGui.BeginPopupModal("Align Objects", ref _showAlignDialog, ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool canAlign = _sceneGraph.SelectedObjects.Count >= 2;

                ImGui.InputInt("Max Iterations", ref _alignIter);
                ImGui.InputFloat("Convergence Threshold", ref _alignThreshold, 0.00001f, 0.0001f, "%.6f");
                ImGui.Separator();

                if (!canAlign)
                {
                    ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "Select at least 2 objects");
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
            if (ImGui.BeginPopupModal("Cleanup Mesh", ref _showCleanupDialog, ImGuiWindowFlags.AlwaysAutoResize))
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
        private int _bakeSize = 1024;

        private void OnBakeTextures()
        {
            _showBakeDialog = true;
            _popupToOpen = "Bake Textures";
        }

        private void DrawBakeDialog()
        {
            if (!_showBakeDialog) return;
            if (ImGui.BeginPopupModal("Bake Textures", ref _showBakeDialog, ImGuiWindowFlags.AlwaysAutoResize))
            {
                bool hasMesh = _sceneGraph.SelectedObjects.OfType<MeshObject>().Any();
                bool hasCameras = _sceneGraph.GetObjectsOfType<CameraObject>().Any();
                bool canBake = hasMesh && hasCameras;

                ImGui.InputInt("Texture Size", ref _bakeSize);
                ImGui.Separator();

                if (!canBake)
                {
                    if (!hasMesh)
                        ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No mesh selected");
                    if (!hasCameras)
                        ImGui.TextColored(new System.Numerics.Vector4(1, 0.5f, 0.5f, 1), "No cameras in scene");
                    ImGui.BeginDisabled();
                }

                if (ImGui.Button("Bake", new System.Numerics.Vector2(120, 0)))
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
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            var cameras = _sceneGraph.GetObjectsOfType<CameraObject>().ToList();
            int size = _bakeSize;

            ProgressDialog.Instance.Start("Baking Textures...", OperationType.Processing);

            Task.Run(() =>
            {
                try
                {
                    var baker = new Deep3DStudio.Texturing.TextureBaker();
                    baker.TextureSize = size;
                    var mesh = meshes[0].MeshData;

                    if (mesh.UVs.Count == 0)
                    {
                        ProgressDialog.Instance.Update(0.1f, "Generating UVs...");
                        var uvData = baker.GenerateUVs(mesh, Deep3DStudio.Texturing.UVUnwrapMethod.SmartProject);
                        mesh.UVs = uvData.UVs;
                    }

                    ProgressDialog.Instance.Update(0.3f, "Projecting Images...");
                    var uvDataForBake = new Deep3DStudio.Texturing.UVData { UVs = mesh.UVs };

                    var progress = new Progress<float>(p => ProgressDialog.Instance.Update(0.3f + p * 0.7f, $"Baking... {(int)(p * 100)}%"));
                    var result = baker.BakeTextures(mesh, uvDataForBake, cameras, progress);

                    mesh.Texture = result.DiffuseMap;
                    mesh.TextureId = -1;

                    ProgressDialog.Instance.Log("Texture baking complete.");
                    ProgressDialog.Instance.Complete();
                }
                catch (Exception ex)
                {
                    ProgressDialog.Instance.Fail(ex);
                }
            });
        }

        #endregion

        #region AI Operations

        private void RunAIModel(string modelName)
        {
            _logBuffer += $"Running {modelName}...\n";
            // Map to appropriate workflow
            switch (modelName)
            {
                case "TripoSR":
                    _selectedWorkflow = 1;
                    RunReconstruction();
                    break;
                case "LGM":
                    _selectedWorkflow = 2;
                    RunReconstruction();
                    break;
                case "Wonder3D":
                    _selectedWorkflow = 3;
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
                    var refiner = new Deep3DStudio.Meshing.DeepMeshPriorMesher();
                    foreach (var mesh in meshes)
                    {
                        var refined = await refiner.RefineMeshAsync(mesh.MeshData, (status, progress) => {
                            ProgressDialog.Instance.Update(progress, status);
                        });
                        if (refined != null)
                        {
                            EnqueueAction(() => {
                                mesh.MeshData = refined;
                                ProgressDialog.Instance.Log($"Refined: {mesh.Name}");
                            });
                        }
                    }
                    EnqueueAction(() => ProgressDialog.Instance.Complete());
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
                    var refiner = new Deep3DStudio.Meshing.GaussianSDFRefiner();
                    foreach (var mesh in meshes)
                    {
                        var refined = await refiner.RefineMeshAsync(mesh.MeshData, (status, progress) => {
                            ProgressDialog.Instance.Update(progress, status);
                        });
                        if (refined != null)
                        {
                            EnqueueAction(() => {
                                mesh.MeshData = refined;
                                ProgressDialog.Instance.Log($"Refined: {mesh.Name}");
                            });
                        }
                    }
                    EnqueueAction(() => ProgressDialog.Instance.Complete());
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private void RunTripoSFRefinement()
        {
            // TripoSF is an image-to-mesh model, not a mesh refiner
            // It generates mesh from images using feed-forward inference
            if (_loadedImages.Count == 0)
            {
                _logBuffer += "Error: TripoSF requires loaded images. Load an image first.\n";
                return;
            }

            ProgressDialog.Instance.Start("TripoSF Generation...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    var triposf = new Deep3DStudio.Model.AIModels.TripoSFInference();
                    var mesh = triposf.GenerateFromImage(_loadedImages[0].FilePath);

                    if (mesh.Vertices.Count > 0)
                    {
                        EnqueueAction(() => {
                            var obj = new MeshObject("TripoSF_Mesh", mesh);
                            lock (_sceneGraph)
                            {
                                _sceneGraph.AddObject(obj);
                            }
                            ProgressDialog.Instance.Log("TripoSF mesh generated.");
                            ProgressDialog.Instance.Complete();
                        });
                    }
                    else
                    {
                        EnqueueAction(() => ProgressDialog.Instance.Fail(new Exception("TripoSF returned empty mesh")));
                    }
                }
                catch (Exception ex)
                {
                    EnqueueAction(() => ProgressDialog.Instance.Fail(ex));
                }
            });
        }

        private async void RunReconstruction(bool generateMesh = true, bool generateCloud = true)
        {
            if (_loadedImages.Count == 0)
            {
                _logBuffer += "Error: No images loaded.\n";
                return;
            }

            ProgressDialog.Instance.Start($"Running {_workflows[_selectedWorkflow]}...", OperationType.Processing);

            try
            {
                SceneResult? result = null;

                await Task.Run(async () =>
                {
                    WorkflowPipeline pipeline = WorkflowPipeline.ImageToDust3rToMesh;

                    if (_workflows[_selectedWorkflow].Contains("TripoSR"))
                        pipeline = WorkflowPipeline.ImageToTripoSR;
                    else if (_workflows[_selectedWorkflow].Contains("LGM"))
                        pipeline = WorkflowPipeline.ImageToLGM;
                    else if (_workflows[_selectedWorkflow].Contains("Wonder3D"))
                        pipeline = WorkflowPipeline.ImageToWonder3D;

                    // Convert ProjectImage list to string list
                    var imagePaths = _loadedImages.Select(i => i.FilePath).ToList();

                    result = await AIModelManager.Instance.ExecuteWorkflowAsync(pipeline, imagePaths, null, (s, p) =>
                    {
                        ProgressDialog.Instance.Update(p, s);
                    });
                });

                if (result != null)
                {
                    foreach (var mesh in result.Meshes)
                    {
                        if (mesh.Vertices.Count > 0)
                        {
                            var obj = new MeshObject("Reconstructed Mesh", mesh);
                            lock (_sceneGraph)
                            {
                                _sceneGraph.AddObject(obj);
                            }
                        }
                    }

                    // Populate depth maps for visualization
                    PopulateDepthData(result);

                    ProgressDialog.Instance.Log($"Reconstruction complete. Added {result.Meshes.Count} objects.");
                    ProgressDialog.Instance.Complete();
                }
                else
                {
                    // If no result but no exception, maybe cancelled or empty?
                    if (ProgressDialog.Instance.State == ProgressState.Running)
                        ProgressDialog.Instance.Fail(new Exception("Unknown failure: No result returned."));
                }
            }
            catch (Exception ex)
            {
                ProgressDialog.Instance.Fail(ex);
            }
        }

        private void PopulateDepthData(SceneResult result)
        {
            if (result.Poses.Count == 0 || result.Meshes.Count == 0) return;

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
                    // Find corresponding ProjectImage
                    var pImg = _loadedImages.FirstOrDefault(i => Path.GetFullPath(i.FilePath) == Path.GetFullPath(pose.ImagePath));
                    if (pImg != null)
                    {
                        float focal = pose.GetEffectiveFocalLength();
                        pImg.DepthMap = ExtractDepthMap(combinedMesh, pose.Width, pose.Height, pose.WorldToCamera, focal);
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
                        ImGui.SetClipboardText(selected);
                        Logger.Info($"Copied {length} bytes of log text to clipboard.");
                    }
                }
            }
            catch (Exception ex)
            {
                Logger.Exception(ex, "Failed to copy selected text");
            }
        }

        private void OnAutoRig()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count == 0)
            {
                _logBuffer += "Error: No mesh selected for rigging.\n";
                return;
            }

            ProgressDialog.Instance.Start("Auto Rigging...", OperationType.Processing);
            Task.Run(() => {
                try
                {
                    foreach (var mesh in meshes)
                    {
                        // Calculate mesh bounds to size and position the skeleton
                        var (min, max) = mesh.GetWorldBounds();
                        var center = (min + max) * 0.5f;
                        var size = max - min;
                        float height = Math.Max(size.Y, 0.1f);
                        float scale = height; // Scale skeleton to match mesh height

                        // Position root at the center-bottom of the mesh
                        var rootPosition = new Vector3(center.X, min.Y + height * 0.5f, center.Z);

                        // Create humanoid skeleton template scaled to mesh size
                        var skeleton = SkeletonData.CreateHumanoidTemplate(rootPosition, scale);

                        // Create skeleton object and add to scene
                        var skelObj = new SkeletonObject($"Rig_{mesh.Name}", skeleton);
                        skelObj.TargetMesh = mesh;
                        skelObj.Position = Vector3.Zero;

                        lock (_sceneGraph)
                        {
                            _sceneGraph.AddObject(skelObj);
                        }
                        ProgressDialog.Instance.Log($"Created humanoid skeleton for '{mesh.Name}' with {skeleton.Joints.Count} joints.");
                    }
                    ProgressDialog.Instance.Complete();
                }
                catch (Exception ex)
                {
                    ProgressDialog.Instance.Fail(ex);
                }
            });
        }

        #endregion
    }
}
