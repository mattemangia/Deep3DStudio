using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using ImGuiNET;
using Deep3DStudio.Viewport;
using Deep3DStudio.Scene;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using System.Drawing;
using Deep3DStudio.IO;
using NativeFileDialogs.Net;

namespace Deep3DStudio
{
    public class MainWindow : GameWindow
    {
        private ImGuiController _controller;
        private ThreeDView _viewport;
        private SceneGraph _sceneGraph;
        private bool _showDemoWindow = false;

        // State
        private int _selectedWorkflow = 0;
        private int _selectedQuality = 1;
        private string[] _workflows = { "Dust3r (Multi-View)", "TripoSR (Single Image)", "LGM (Gaussian)", "Wonder3D" };
        private string[] _qualities = { "Fast", "Balanced", "High" };
        private string _logBuffer = "";

        // Layout
        private float _leftPanelWidth = 300;
        private float _rightPanelWidth = 300;
        private float _logPanelHeight = 150;

        public MainWindow(GameWindowSettings gameWindowSettings, NativeWindowSettings nativeWindowSettings)
            : base(gameWindowSettings, nativeWindowSettings)
        {
            _sceneGraph = new SceneGraph();
            _viewport = new ThreeDView(_sceneGraph);
        }

        protected override void OnLoad()
        {
            base.OnLoad();

            Title += ": OpenGL Version: " + GL.GetString(StringName.Version);

            _controller = new ImGuiController(ClientSize.X, ClientSize.Y);

            // Init Python service hook
            PythonService.Instance.OnLogOutput += (msg) => {
                _logBuffer += msg + "\n";
            };

            // Init Viewport GL state
            _viewport.InitGL();
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

            // If mouse is not over UI, pass to viewport
            if (!ImGui.GetIO().WantCaptureMouse)
            {
                _viewport.OnMouseWheel(e.OffsetY);
            }
        }

        protected override void OnRenderFrame(FrameEventArgs e)
        {
            base.OnRenderFrame(e);

            _controller.Update(this, (float)e.Time);

            GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit | ClearBufferMask.StencilBufferBit);

            // Render 3D Viewport in the background (or full window)
            // We need to calculate the viewport rect based on UI layout if we want it "docked"
            // For simplicity in ImGui, we can render the 3D scene to the full window behind the UI
            // OR render it to a Framebuffer and display it as an Image in ImGui (more complex but better).
            // Let's render to full window for now, but respect UI overlay.

            // But wait, ImGui windows are opaque usually.
            // A better "Studio" layout is to have the 3D view fill the center area.
            // Since we don't have docking (unless we enable it in ImGui config, which requires downloading imgui docking branch usually, but ImGui.NET might include it).
            // Standard ImGui.NET doesn't have docking enabled by default.

            // Strategy:
            // 1. Render 3D View to the whole screen.
            // 2. Render ImGui windows floating over it.

            // Pass Input to Viewport only if ImGui doesn't want it.
            if (!ImGui.GetIO().WantCaptureMouse && !ImGui.GetIO().WantCaptureKeyboard)
            {
                 var mouseState = MouseState;
                 var keyboardState = KeyboardState;
                 _viewport.UpdateInput(mouseState, keyboardState, (float)e.Time, ClientSize.X, ClientSize.Y);
            }

            _viewport.Render(ClientSize.X, ClientSize.Y);

            // Render UI
            RenderUI();

            _controller.Render();

            SwapBuffers();
        }

        private void OnOpenProject()
        {
             var result = Nfd.OpenDialog("d3d", null);
             if (result != null && result.Path != null)
             {
                 _logBuffer += $"Loaded project: {result.Path}\n";
             }
        }

        private void OnSaveProject()
        {
             var result = Nfd.SaveDialog("d3d", null);
             if (result != null && result.Path != null)
             {
                 _logBuffer += $"Saved project: {result.Path}\n";
             }
        }

        private void RunReconstruction()
        {
            _logBuffer += $"Starting {_workflows[_selectedWorkflow]} reconstruction ({_qualities[_selectedQuality]})...\n";

            // In a real scenario, we would trigger the async task here.
            // Since we are in the main thread, we just log.
            // Calling Python logic here might block, so it should be threaded.

            System.Threading.Tasks.Task.Run(() =>
            {
                 // This is where we would call logic
                 // e.g. Dust3rInference.Run(...)
                 // But for this task, ensuring the UI is wired up is key.
            });
        }

        private void RenderUI()
        {
            // Main Menu Bar
            if (ImGui.BeginMainMenuBar())
            {
                if (ImGui.BeginMenu("File"))
                {
                    if (ImGui.MenuItem("New Project"))
                    {
                        _sceneGraph.Clear();
                        _logBuffer = "";
                    }
                    if (ImGui.MenuItem("Open Project")) { OnOpenProject(); }
                    if (ImGui.MenuItem("Save Project")) { OnSaveProject(); }
                    ImGui.Separator();
                    if (ImGui.MenuItem("Quit")) { Close(); }
                    ImGui.EndMenu();
                }
                if (ImGui.BeginMenu("Edit"))
                {
                    if (ImGui.MenuItem("Settings")) { /* Settings window not implemented yet */ }
                    ImGui.EndMenu();
                }
                if (ImGui.BeginMenu("Help"))
                {
                    if (ImGui.MenuItem("About"))
                    {
                        _logBuffer += "Deep3DStudio Cross-Platform\nVersion 1.0\n";
                    }
                    ImGui.EndMenu();
                }
                ImGui.EndMainMenuBar();
            }

            // Toolbar
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, 20)); // Below menu
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.X, 40));
            ImGui.Begin("Toolbar", ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoScrollbar);
            {
                ImGui.Text("Workflow:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(150);
                ImGui.Combo("##Workflow", ref _selectedWorkflow, _workflows, _workflows.Length);
                ImGui.SameLine();
                ImGui.Text("Quality:"); ImGui.SameLine();
                ImGui.SetNextItemWidth(100);
                ImGui.Combo("##Quality", ref _selectedQuality, _qualities, _qualities.Length);
                ImGui.SameLine();
                if (ImGui.Button("Run Reconstruction"))
                {
                    RunReconstruction();
                }
            }
            ImGui.End();

            // Left Panel (Scene/Images)
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, 60));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(_leftPanelWidth, ClientSize.Y - 60 - _logPanelHeight));
            ImGui.Begin("Project", ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse);
            {
                if (ImGui.BeginTabBar("ProjectTabs"))
                {
                    if (ImGui.BeginTabItem("Images"))
                    {
                        ImGui.Text("Drag & Drop images here");
                        // List images
                        ImGui.EndTabItem();
                    }
                    if (ImGui.BeginTabItem("Scene Graph"))
                    {
                        foreach(var obj in _sceneGraph.GetVisibleObjects())
                        {
                            bool selected = obj.Selected;
                            if (ImGui.Selectable(obj.Name ?? $"Object {obj.Id}", selected))
                            {
                                if (!ImGui.GetIO().KeyCtrl) _sceneGraph.ClearSelection();
                                _sceneGraph.Select(obj, !selected); // Toggle if ctrl
                            }
                        }
                        ImGui.EndTabItem();
                    }
                    ImGui.EndTabBar();
                }
            }
            ImGui.End();

            // Log Panel
            ImGui.SetNextWindowPos(new System.Numerics.Vector2(0, ClientSize.Y - _logPanelHeight));
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(ClientSize.X, _logPanelHeight));
            ImGui.Begin("Log", ImGuiWindowFlags.NoMove | ImGuiWindowFlags.NoCollapse);
            {
                ImGui.TextUnformatted(_logBuffer);
                if (ImGui.GetScrollY() >= ImGui.GetScrollMaxY())
                    ImGui.SetScrollHereY(1.0f);
            }
            ImGui.End();

            // Right Panel (Properties)
            // ...
        }
    }
}
