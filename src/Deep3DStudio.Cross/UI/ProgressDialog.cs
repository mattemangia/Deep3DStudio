using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using System.Threading;
using ImGuiNET;

namespace Deep3DStudio.UI
{
    public enum OperationType
    {
        ImportExport,
        Processing
    }

    public enum ProgressState
    {
        Idle,
        Running,
        Success,
        Error,
        Cancelled
    }

    public class ProgressDialog
    {
        private static ProgressDialog? _instance;
        public static ProgressDialog Instance => _instance ??= new ProgressDialog();

        // State
        public bool IsVisible { get; private set; }
        public string Title { get; private set; } = "Processing";
        public float Progress { get; private set; }
        public string StatusText { get; private set; } = "";
        public OperationType OpType { get; private set; }
        public ProgressState State { get; private set; } = ProgressState.Idle;
        public CancellationTokenSource? CancellationTokenSource { get; private set; }

        private StringBuilder _logBuffer = new StringBuilder();
        private bool _verboseExpanded = false;
        private string _errorMessage = "";

        public void Start(string title, OperationType type)
        {
            Title = title;
            OpType = type;
            State = ProgressState.Running;
            Progress = 0.0f;
            StatusText = "Starting...";
            _logBuffer.Clear();
            _errorMessage = "";
            IsVisible = true;
            _verboseExpanded = false;
            CancellationTokenSource = new CancellationTokenSource();
        }

        public void Update(float progress, string status)
        {
            Progress = progress;
            StatusText = status;
        }

        public void Log(string message)
        {
            _logBuffer.AppendLine(message);
        }

        public void Complete()
        {
            State = ProgressState.Success;
            Progress = 1.0f;
            StatusText = "Operation successfully complete";

            if (OpType == OperationType.ImportExport)
            {
                Close();
            }
        }

        public void Fail(Exception ex)
        {
            State = ProgressState.Error;
            _errorMessage = ex.Message;
            StatusText = "Error occurred";
            Log($"ERROR: {ex.Message}");
            Log(ex.StackTrace ?? "");
            _verboseExpanded = true; // Auto expand on error
        }

        public void Close()
        {
            IsVisible = false;
            State = ProgressState.Idle;
            CancellationTokenSource?.Dispose();
            CancellationTokenSource = null;
        }

        public void Draw()
        {
            if (!IsVisible) return;

            // Make the dialog resizable, bigger on error, with minimum size constraints
            float defaultWidth = State == ProgressState.Error ? 700 : 500;
            float defaultHeight = _verboseExpanded ? (State == ProgressState.Error ? 500 : 400) : 180;

            ImGui.SetNextWindowSizeConstraints(new Vector2(400, 150), new Vector2(1200, 800));
            ImGui.SetNextWindowSize(new Vector2(defaultWidth, defaultHeight), ImGuiCond.Appearing);

            // Center only on first appearance
            var viewport = ImGui.GetMainViewport();
            var center = viewport.GetCenter();
            ImGui.SetNextWindowPos(center, ImGuiCond.Appearing, new Vector2(0.5f, 0.5f));

            if (ImGui.Begin(Title, ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoDocking | ImGuiWindowFlags.NoSavedSettings))
            {
                // 1. Progress Bar
                Vector4 barColor;
                if (State == ProgressState.Error) barColor = new Vector4(1.0f, 0.2f, 0.2f, 1.0f); // Red
                else if (State == ProgressState.Success) barColor = new Vector4(0.2f, 0.8f, 0.2f, 1.0f); // Green
                else barColor = new Vector4(0.2f, 0.4f, 0.8f, 1.0f); // Blue

                ImGui.PushStyleColor(ImGuiCol.PlotHistogram, barColor);
                string overlay = $"{StatusText} {(int)(Progress * 100)}%";
                ImGui.ProgressBar(Progress, new Vector2(-1, 25), overlay);
                ImGui.PopStyleColor();

                // 2. Status Text
                if (State == ProgressState.Error)
                {
                    ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(1.0f, 0.4f, 0.4f, 1.0f));
                    ImGui.TextWrapped(_errorMessage);
                    ImGui.PopStyleColor();
                }
                else
                {
                    ImGui.TextWrapped(StatusText);
                }

                ImGui.Spacing();
                ImGui.Separator();
                ImGui.Spacing();

                // 3. Controls (Cancel, Arrow)
                // Cancel Button
                if (State == ProgressState.Running)
                {
                    if (ImGui.Button("Cancel"))
                    {
                        State = ProgressState.Cancelled;
                        StatusText = "Cancelling...";
                        CancellationTokenSource?.Cancel();
                    }
                }
                else if (State == ProgressState.Success || State == ProgressState.Error || State == ProgressState.Cancelled)
                {
                    if (OpType == OperationType.Processing || State == ProgressState.Error || State == ProgressState.Cancelled)
                    {
                        if (ImGui.Button("OK"))
                        {
                            Close();
                        }
                    }
                }

                ImGui.SameLine();

                // Arrow for verbose
                if (ImGui.ArrowButton("##Verbose", _verboseExpanded ? ImGuiDir.Down : ImGuiDir.Right))
                {
                    _verboseExpanded = !_verboseExpanded;
                }
                ImGui.SameLine();
                ImGui.Text("Details");

                // 4. Verbose Output
                if (_verboseExpanded)
                {
                    ImGui.Separator();
                    ImGui.BeginChild("LogRegion", new Vector2(0, -1), ImGuiChildFlags.Borders);

                    if (State == ProgressState.Error)
                    {
                        ImGui.PushStyleColor(ImGuiCol.Text, new Vector4(1.0f, 0.4f, 0.4f, 1.0f)); // Red text
                        ImGui.TextUnformatted(_logBuffer.ToString());
                        ImGui.PopStyleColor();
                    }
                    else
                    {
                        ImGui.TextUnformatted(_logBuffer.ToString());
                    }

                    // Auto scroll
                    if (State == ProgressState.Running && ImGui.GetScrollY() >= ImGui.GetScrollMaxY())
                        ImGui.SetScrollHereY(1.0f);

                    ImGui.EndChild();
                }
            }
            ImGui.End();
        }
    }
}
