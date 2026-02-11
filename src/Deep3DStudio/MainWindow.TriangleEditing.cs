using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnDeleteSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.DeleteSelectedTriangles();
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Deleted {stats.triangleCount} triangles from {stats.meshCount} mesh(es)";
        }

        private void OnFlipSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.FlipSelectedTriangles();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Flipped {stats.triangleCount} triangles";
        }

        private void OnSubdivideSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.SubdivideSelectedTriangles();
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Subdivided {stats.triangleCount} triangles (each into 4)";
        }

        private void OnSelectAllTriangles(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                _statusLabel.Text = "No mesh selected. Select a mesh first.";
                return;
            }

            var tool = _viewport.MeshEditingTool;
            foreach (var mesh in selectedMeshes)
            {
                tool.SelectAll(mesh);
            }
            _viewport.QueueDraw();

            var stats = tool.GetSelectionStats();
            _statusLabel.Text = $"Selected all {stats.triangleCount} triangles";
        }

        private void OnInvertTriangleSelection(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                _statusLabel.Text = "No mesh selected. Select a mesh first.";
                return;
            }

            var tool = _viewport.MeshEditingTool;
            foreach (var mesh in selectedMeshes)
            {
                tool.InvertSelection(mesh);
            }
            _viewport.QueueDraw();

            var stats = tool.GetSelectionStats();
            _statusLabel.Text = $"Inverted selection: {stats.triangleCount} triangles now selected";
        }

        private void OnGrowTriangleSelection(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            int beforeCount = tool.SelectedTriangles.Count;
            tool.GrowSelection();
            int afterCount = tool.SelectedTriangles.Count;
            _viewport.QueueDraw();
            _statusLabel.Text = $"Selection grown: {beforeCount} -> {afterCount} triangles";
        }

        private void OnClearTriangleSelection(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            tool.ClearSelection();
            _viewport.QueueDraw();
            _statusLabel.Text = "Triangle selection cleared";
        }

        private void OnWeldSelectedVertices(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            tool.WeldSelectedVertices(0.001f);
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = "Welded duplicate vertices in selected area";
        }

        private void OnMoveSelectedVertices(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            int moved = tool.MoveSelectedVertices(_penMoveDelta);
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Moved {moved} selected vertices";
            _isDirty = true;
            UpdateTitle();
        }

        private void ConfigureMoveSelectedVertices()
        {
            var dialog = new Vector3InputDialog(
                this,
                "Move Selected Vertices",
                "Delta X", "Delta Y", "Delta Z",
                _penMoveDelta,
                -10000f, 10000f, 0.01f, 4);

            if (dialog.Run() == (int)ResponseType.Ok)
            {
                _penMoveDelta = dialog.Value;
            }

            dialog.Destroy();
        }

        private void OnExtrudeSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.ExtrudeSelectedTriangles(_penExtrudeDistance);
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Extruded {stats.triangleCount} selected triangles";
            _isDirty = true;
            UpdateTitle();
        }

        private void ConfigureExtrudeSelectedTriangles()
        {
            var dlg = new NumericInputDialog(this, "Extrude Triangles", "Extrusion Distance:", _penExtrudeDistance, -10f, 10f, 0.01f, 4);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _penExtrudeDistance = dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnInsetSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count == 0)
            {
                _statusLabel.Text = "No triangles selected. Use Pen tool (P) to select triangles first.";
                return;
            }

            var stats = tool.GetSelectionStats();
            tool.InsetSelectedTriangles(_penInsetAmount);
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
            _statusLabel.Text = $"Inset {stats.triangleCount} selected triangles";
            _isDirty = true;
            UpdateTitle();
        }

        private void ConfigureInsetSelectedTriangles()
        {
            var dlg = new NumericInputDialog(this, "Inset Triangles", "Inset Amount (0-1):", _penInsetAmount, 0.01f, 0.95f, 0.01f, 3);
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _penInsetAmount = dlg.Value;
            }
            dlg.Destroy();
        }

        private void OnBridgeSelectedTriangles(object? sender, EventArgs e)
        {
            var tool = _viewport.MeshEditingTool;
            if (tool.SelectedTriangles.Count != 2)
            {
                _statusLabel.Text = "Bridge requires exactly 2 selected triangles on the same mesh.";
                return;
            }

            if (tool.BridgeSelectedTrianglesSimple())
            {
                _sceneTreeView.RefreshTree();
                _viewport.QueueDraw();
                _statusLabel.Text = "Created bridge faces between selected triangles";
                _isDirty = true;
                UpdateTitle();
            }
            else
            {
                _statusLabel.Text = "Could not bridge selection. Ensure 2 valid triangles from the same mesh are selected.";
            }
        }
    }
}
