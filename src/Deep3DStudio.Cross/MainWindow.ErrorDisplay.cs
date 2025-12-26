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
    public partial class MainWindow
    {
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
                    ClipboardString = text;
                }

                ImGui.End();
            }
            else
            {
                ImGui.PopStyleColor(2);
            }
        }
    }
}
