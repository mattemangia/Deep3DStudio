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
        private bool _isBusy = false;
        private string _busyStatus = "";
        private float _busyProgress = 0.0f;

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
        private List<string> _loadedImages = new List<string>();
        private Dictionary<string, int> _imageThumbnails = new Dictionary<string, int>();
        private int _selectedImageIndex = -1;

        // Layout
        private float _leftPanelWidth = 280;
        private float _rightPanelWidth = 280;
        private float _logPanelHeight = 150;
        private float _toolbarHeight = 45;
        private float _verticalToolbarWidth = 45;

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

        public MainWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
            _sceneGraph = new SceneGraph();
            _sceneGraph.SceneChanged += (s, e) => { _isDirty = true; UpdateTitle(); };
            _viewport = new ThreeDView(_sceneGraph);
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
            };

            // Init AI Manager hooks
            AIModelManager.Instance.ProgressUpdated += (status, progress) => {
                _busyStatus = status;
                _busyProgress = progress;
            };

            // Init Viewport GL state
            _viewport.InitGL();

            // Init Icons
            _iconFactory = new ImGuiIconFactory();

            // Load Logo
            _logoTexture = TextureLoader.LoadTextureFromResource("logo.png");

            // Force Python Init if not started
            Task.Run(() => {
                try {
                    PythonService.Instance.Initialize();
                } catch(Exception ex) {
                    _logBuffer += $"Python Init Error: {ex.Message}\n";
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
            string ext = Path.GetExtension(file).ToLower();
            try
            {
                if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
                {
                    if (!_loadedImages.Contains(file))
                    {
                        _loadedImages.Add(file);
                        // Create thumbnail asynchronously
                        Task.Run(() => {
                            var thumb = TextureLoader.CreateThumbnail(file, 64);
                            if (thumb > 0)
                            {
                                lock (_imageThumbnails)
                                {
                                    _imageThumbnails[file] = thumb;
                                }
                            }
                        });
                        _logBuffer += $"Added image: {Path.GetFileName(file)}\n";
                    }
                }
                else if (ext == ".obj" || ext == ".ply" || ext == ".glb" || ext == ".stl")
                {
                    var mesh = MeshImporter.Load(file);
                    if (mesh != null)
                    {
                        var obj = new MeshObject(Path.GetFileName(file), mesh);
                        _sceneGraph.AddObject(obj);
                        _logBuffer += $"Imported mesh: {Path.GetFileName(file)}\n";
                    }
                }
                else if (ext == ".xyz")
                {
                    var pc = PointCloudImporter.Load(file);
                    if (pc != null)
                    {
                        _sceneGraph.AddObject(pc);
                        _logBuffer += $"Imported point cloud: {Path.GetFileName(file)}\n";
                    }
                }
                else if (ext == ".d3d")
                {
                    OnOpenProject(file);
                }
            }
            catch (Exception ex)
            {
                ShowError("Import Error", $"Failed to import {Path.GetFileName(file)}", ex);
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
                    case Keys.F: _viewport.FocusOnSelection(); break;
                    case Keys.Delete: OnDeleteSelected(); break;
                    case Keys.Escape: _sceneGraph.ClearSelection(); break;
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

            _controller.Update(this, (float)e.Time);

            GL.ClearColor(0.15f, 0.15f, 0.15f, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit | ClearBufferMask.StencilBufferBit);

            if (_showSplash)
            {
                RenderSplash();
            }
            else
            {
                // Update Input
                if (!ImGui.GetIO().WantCaptureMouse && !ImGui.GetIO().WantCaptureKeyboard && !_isBusy)
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

                // Render UI
                RenderUI();
            }

            _controller.Render();

            // Check for OpenGL errors
            var err = GL.GetError();
            if (err != OpenTK.Graphics.OpenGL.ErrorCode.NoError && err != OpenTK.Graphics.OpenGL.ErrorCode.InvalidFramebufferOperation)
            {
                Console.WriteLine($"OpenGL Error: {err}");
            }

            SwapBuffers();
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

            // Gradient background
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

            // Title
            ImGui.PushFont(ImGui.GetIO().Fonts.Fonts[0]);
            string text = "Deep3DStudio";
            var textSize = ImGui.CalcTextSize(text);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - textSize.X * 0.5f, center.Y + 50));
            ImGui.PushStyleColor(ImGuiCol.Text, new System.Numerics.Vector4(0.9f, 0.9f, 0.95f, 1.0f));
            ImGui.Text(text);
            ImGui.PopStyleColor();
            ImGui.PopFont();

            // Subtitle
            string subtitle = "Neural 3D Reconstruction Studio";
            var subSize = ImGui.CalcTextSize(subtitle);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - subSize.X * 0.5f, center.Y + 75));
            ImGui.TextDisabled(subtitle);

            // Status
            string status = _pythonReady ? "Ready" : "Initializing AI Engine...";
            var statusSize = ImGui.CalcTextSize(status);
            ImGui.SetCursorPos(new System.Numerics.Vector2(center.X - statusSize.X * 0.5f, center.Y + 110));

            if (!_pythonReady)
            {
                // Simple loading animation
                float time = (float)(DateTime.Now.TimeOfDay.TotalSeconds % 1.0);
                ImGui.TextColored(new System.Numerics.Vector4(0.4f, 0.6f, 0.9f, 0.5f + 0.5f * (float)Math.Sin(time * Math.PI * 2)), status);
            }
            else
            {
                ImGui.TextColored(new System.Numerics.Vector4(0.4f, 0.9f, 0.4f, 1.0f), status);
            }

            // Version
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
            // Busy Overlay
            if (_isBusy)
            {
                RenderBusyOverlay();
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

            // Dialogs
            if (_showSettings) DrawSettingsWindow();
            if (_showAbout) DrawAboutWindow();
            if (_showImagePreview) DrawImagePreviewWindow();
            _diagnosticsWindow.Draw();
        }

        private void RenderBusyOverlay()
        {
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(ClientSize.X / 2 - 200, ClientSize.Y / 2 - 50));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(400, 100));

            ImGui.Begin("Processing", ImGuiWindowFlags.NoDecoration | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoResize);

            ImGui.Text("Processing...");
            ImGui.Separator();
            ImGui.TextWrapped(_busyStatus);

            if (_busyProgress > 0)
            {
                ImGui.ProgressBar(_busyProgress, new System.Numerics.Vector2(-1, 0));
            }

            ImGui.End();
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
                var size = new System.Numerics.Vector2(28, 28);

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
                var size = new System.Numerics.Vector2(28, 28);

                // Focus
                if (ImGui.ImageButton("##Focus", _iconFactory.GetIcon(IconType.Select), size))
                    _viewport.FocusOnSelection();
                if (ImGui.IsItemHovered()) ImGui.SetTooltip("Focus on Selection (F)");

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // Mesh Operations
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
                    OnDeleteSelected();
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
                string path = _loadedImages[i];
                string filename = Path.GetFileName(path);

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
                    if (ImGui.Button($"[{filename.Substring(0, Math.Min(6, filename.Length))}...]", new System.Numerics.Vector2(thumbSize, thumbSize)))
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
                    ImGui.SetTooltip(filename);
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

                if (ImGui.Selectable($"{icon}{name}##{i}", selected))
                {
                    if (!ImGui.GetIO().KeyCtrl) _sceneGraph.ClearSelection();
                    _sceneGraph.Select(obj, !selected);
                }

                // Context menu
                if (ImGui.BeginPopupContextItem())
                {
                    if (ImGui.MenuItem("Focus")) _viewport.FocusOnObject(obj);
                    if (ImGui.MenuItem("Delete")) _sceneGraph.RemoveObject(obj);
                    if (ImGui.MenuItem("Duplicate")) OnDuplicateObject(obj);
                    ImGui.EndPopup();
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
                RenderProperties();
            }
            ImGui.End();
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
                if (ImGui.Button("Clear"))
                {
                    _logBuffer = "";
                }
                ImGui.SameLine();
                if (ImGui.Button("Copy"))
                {
                    ImGui.SetClipboardText(_logBuffer);
                }

                ImGui.Separator();

                ImGui.BeginChild("LogScroll");
                ImGui.TextUnformatted(_logBuffer);
                if (ImGui.GetScrollY() >= ImGui.GetScrollMaxY())
                    ImGui.SetScrollHereY(1.0f);
                ImGui.EndChild();
            }
            ImGui.End();
        }

        #endregion

        #region Dialogs

        private void DrawSettingsWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(400, 500), ImGuiCond.FirstUseEver);

            if (ImGui.Begin("Settings", ref _showSettings))
            {
                var s = IniSettings.Instance;

                if (ImGui.CollapsingHeader("Viewport", ImGuiTreeNodeFlags.DefaultOpen))
                {
                    bool grid = s.ShowGrid;
                    if (ImGui.Checkbox("Show Grid", ref grid)) s.ShowGrid = grid;

                    bool axes = s.ShowAxes;
                    if (ImGui.Checkbox("Show Axes", ref axes)) s.ShowAxes = axes;

                    bool cameras = s.ShowCameras;
                    if (ImGui.Checkbox("Show Cameras", ref cameras)) s.ShowCameras = cameras;
                }

                if (ImGui.CollapsingHeader("Reconstruction"))
                {
                    int method = (int)s.ReconstructionMethod;
                    string[] methods = Enum.GetNames(typeof(ReconstructionMethod));
                    if (ImGui.Combo("Default Method", ref method, methods, methods.Length))
                        s.ReconstructionMethod = (ReconstructionMethod)method;
                }

                if (ImGui.CollapsingHeader("Performance"))
                {
                    ImGui.Text("OpenGL Version: " + GL.GetString(StringName.Version));
                    ImGui.Text("Renderer: " + GL.GetString(StringName.Renderer));
                }

                ImGui.Separator();

                if (ImGui.Button("Save Settings"))
                {
                    s.Save();
                    _logBuffer += "Settings saved.\n";
                }
                ImGui.SameLine();
                if (ImGui.Button("Reset to Defaults"))
                {
                    s.Reset();
                }
            }
            ImGui.End();
        }

        private void DrawAboutWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(400, 350), ImGuiCond.FirstUseEver);

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

            try
            {
                var state = CrossProjectManager.LoadProject(path);
                CrossProjectManager.RestoreSceneFromState(state, _sceneGraph);

                ClearImages();
                if (state.ImagePaths != null)
                {
                    foreach (var img in state.ImagePaths)
                    {
                        if (File.Exists(img))
                        {
                            ImportFile(img);
                        }
                    }
                }

                _currentProjectPath = path;
                _isDirty = false;
                UpdateTitle();
                _logBuffer += $"Project loaded: {path}\n";
            }
            catch (Exception ex)
            {
                ShowError("Load Error", $"Failed to load project: {path}", ex);
            }
        }

        private void OnSaveProject()
        {
            if (string.IsNullOrEmpty(_currentProjectPath))
            {
                OnSaveProjectAs();
                return;
            }

            try
            {
                CrossProjectManager.SaveProject(_currentProjectPath, _sceneGraph, _loadedImages);
                _isDirty = false;
                UpdateTitle();
                _logBuffer += $"Project saved: {_currentProjectPath}\n";
            }
            catch (Exception ex)
            {
                ShowError("Save Error", $"Failed to save project", ex);
            }
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

        #endregion

        #region File Operations

        private void OnAddImages()
        {
            var result = Nfd.OpenDialogMultiple(out string[] paths, new Dictionary<string, string>
            {
                { "Images", "jpg,jpeg,png,bmp,tif,tiff" }
            });

            if (result == NfdStatus.Ok && paths != null)
            {
                foreach (var path in paths)
                {
                    ImportFile(path);
                }
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
                try
                {
                    MeshExporter.Save(path, meshes[0].MeshData);
                    _logBuffer += $"Mesh exported: {path}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Export Error", $"Failed to export mesh", ex);
                }
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
                try
                {
                    PointCloudExporter.Save(path, pcs[0]);
                    _logBuffer += $"Point cloud exported: {path}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Export Error", $"Failed to export point cloud", ex);
                }
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

        private void OnDecimate()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                try
                {
                    mo.MeshData = MeshOperations.Decimate(mo.MeshData, 0.5f);
                    _logBuffer += $"Decimated: {mo.Name}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Decimate Error", $"Failed to decimate {mo.Name}", ex);
                }
            }
        }

        private void OnSmooth()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                try
                {
                    mo.MeshData = MeshOperations.Smooth(mo.MeshData, 1);
                    _logBuffer += $"Smoothed: {mo.Name}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Smooth Error", $"Failed to smooth {mo.Name}", ex);
                }
            }
        }

        private void OnOptimize()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                try
                {
                    mo.MeshData = MeshCleaningTools.CleanupMesh(mo.MeshData, MeshCleanupOptions.Default);
                    _logBuffer += $"Optimized: {mo.Name}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Optimize Error", $"Failed to optimize {mo.Name}", ex);
                }
            }
        }

        private void OnSplit()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList())
            {
                try
                {
                    var parts = MeshOperations.SplitByConnectivity(mo.MeshData);
                    if (parts.Count > 1)
                    {
                        _sceneGraph.RemoveObject(mo);
                        int i = 1;
                        foreach (var part in parts)
                        {
                            var newObj = new MeshObject($"{mo.Name}_part{i}", part);
                            _sceneGraph.AddObject(newObj);
                            i++;
                        }
                        _logBuffer += $"Split {mo.Name} into {parts.Count} parts.\n";
                    }
                    else
                    {
                        _logBuffer += $"{mo.Name} has only one connected component.\n";
                    }
                }
                catch (Exception ex)
                {
                    ShowError("Split Error", $"Failed to split {mo.Name}", ex);
                }
            }
        }

        private void OnFlipNormals()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                mo.MeshData = MeshOperations.FlipNormals(mo.MeshData);
                _logBuffer += $"Flipped normals: {mo.Name}\n";
            }
        }

        private void OnMerge()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count < 2)
            {
                _logBuffer += "Select at least 2 meshes to merge.\n";
                return;
            }

            try
            {
                var merged = MeshOperations.Merge(meshes.Select(m => m.MeshData).ToList());
                var newObj = new MeshObject("Merged", merged);

                foreach (var mo in meshes)
                {
                    _sceneGraph.RemoveObject(mo);
                }

                _sceneGraph.AddObject(newObj);
                _logBuffer += $"Merged {meshes.Count} meshes.\n";
            }
            catch (Exception ex)
            {
                ShowError("Merge Error", "Failed to merge meshes", ex);
            }
        }

        private void OnAlign()
        {
            var meshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (meshes.Count < 2)
            {
                _logBuffer += "Select at least 2 meshes to align.\n";
                return;
            }

            try
            {
                // Use ICP to align second mesh to first
                var transform = MeshOperations.AlignICP(meshes[1].MeshData.Vertices, meshes[0].MeshData.Vertices);
                meshes[1].MeshData.ApplyTransform(transform);
                _logBuffer += $"Aligned {meshes[1].Name} to {meshes[0].Name}.\n";
            }
            catch (Exception ex)
            {
                ShowError("Align Error", "Failed to align meshes", ex);
            }
        }

        private void OnCleanup()
        {
            foreach (var mo in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                try
                {
                    mo.MeshData = MeshCleaningTools.CleanupMesh(mo.MeshData, MeshCleanupOptions.All);
                    mo.MeshData.RecalculateNormals();
                    _logBuffer += $"Cleaned up: {mo.Name}\n";
                }
                catch (Exception ex)
                {
                    ShowError("Cleanup Error", $"Failed to cleanup {mo.Name}", ex);
                }
            }
        }

        private void OnBakeTextures()
        {
            _logBuffer += "Texture baking not yet fully implemented in Cross version.\n";
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
                default:
                    _logBuffer += $"Model {modelName} not yet implemented.\n";
                    break;
            }
        }

        private async void RunReconstruction(bool generateMesh = true, bool generateCloud = true)
        {
            if (_loadedImages.Count == 0)
            {
                _logBuffer += "Error: No images loaded.\n";
                return;
            }

            _isBusy = true;
            _busyStatus = $"Running {_workflows[_selectedWorkflow]}...";
            _busyProgress = 0.0f;

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

                    result = await AIModelManager.Instance.ExecuteWorkflowAsync(pipeline, _loadedImages, null, (s, p) =>
                    {
                        _busyStatus = s;
                        _busyProgress = p;
                    });
                });

                if (result != null)
                {
                    foreach (var mesh in result.Meshes)
                    {
                        if (mesh.Vertices.Count > 0)
                        {
                            var obj = new MeshObject("Reconstructed Mesh", mesh);
                            _sceneGraph.AddObject(obj);
                        }
                    }
                    _logBuffer += $"Reconstruction complete. Added {result.Meshes.Count} objects.\n";
                }
            }
            catch (Exception ex)
            {
                ShowError("Reconstruction Error", "AI reconstruction failed", ex);
            }
            finally
            {
                _isBusy = false;
            }
        }

        #endregion
    }
}
