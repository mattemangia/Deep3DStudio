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
        private Widget CreateSceneTreePanel()
        {
            var panel = new Box(Orientation.Vertical, 0);
            panel.SetSizeRequest(250, -1);

            // Scene Tree (top)
            _sceneTreeView = new SceneTreeView();
            _sceneTreeView.SetSceneGraph(_sceneGraph);
            _sceneTreeView.ObjectSelected += OnSceneObjectSelected;
            _sceneTreeView.ObjectDoubleClicked += OnSceneObjectDoubleClicked;
            _sceneTreeView.ObjectActionRequested += OnSceneObjectAction;
            panel.PackStart(_sceneTreeView, true, true, 0);

            panel.PackStart(new Separator(Orientation.Horizontal), false, false, 5);

            // Image Browser (bottom)
            var imagesLabel = new Label("Input Images");
            imagesLabel.Attributes = new Pango.AttrList();
            imagesLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            panel.PackStart(imagesLabel, false, false, 2);

            _imageBrowser = new ImageBrowserPanel();
            _imageBrowser.ImageDoubleClicked += OnImageDoubleClicked;
            _imageBrowser.SetSizeRequest(-1, 150);
            panel.PackStart(_imageBrowser, false, false, 0);

            // Clear button
            var clearBtn = new Button("Clear Images");
            clearBtn.Clicked += (s, e) => {
                _imagePaths.Clear();
                _imageBrowser.Clear();
                _lastSceneResult = null;
            };
            panel.PackStart(clearBtn, false, false, 2);

            return panel;
        }

        private void OnToggleSceneTree(object? sender, EventArgs e)
        {
            if (_showSceneTreeMenuItem != null && _leftPanel != null)
            {
                _leftPanel.Visible = _showSceneTreeMenuItem.Active;
            }
        }

        private void OnToggleVerticalToolbar(object? sender, EventArgs e)
        {
            if (_showVerticalToolbarMenuItem != null && _verticalToolbar != null)
            {
                _verticalToolbar.Visible = _showVerticalToolbarMenuItem.Active;
                IniSettings.Instance.ShowVerticalToolbar = _showVerticalToolbarMenuItem.Active;
            }
        }

        private void OnToggleTopToolbar(object? sender, EventArgs e)
        {
            if (_showTopToolbarMenuItem != null && _topToolbar != null)
            {
                _topToolbar.Visible = _showTopToolbarMenuItem.Active;
                IniSettings.Instance.ShowTopToolbar = _showTopToolbarMenuItem.Active;
            }
        }

        private void OnToggleMeshEditorToolbar(object? sender, EventArgs e)
        {
            if (_showMeshEditorToolbarMenuItem != null && _meshEditorToolbar != null)
            {
                _meshEditorToolbar.Visible = _showMeshEditorToolbarMenuItem.Active;
                IniSettings.Instance.ShowMeshEditorToolbar = _showMeshEditorToolbarMenuItem.Active;
            }
        }

        private void OnTogglePointCloudToolbar(object? sender, EventArgs e)
        {
            if (_showPointCloudToolbarMenuItem != null && _pointCloudToolbar != null)
            {
                _pointCloudToolbar.Visible = _showPointCloudToolbarMenuItem.Active;
                IniSettings.Instance.ShowPointCloudToolbar = _showPointCloudToolbarMenuItem.Active;
            }
        }

        private void OnToggleGeoreferenceToolbar(object? sender, EventArgs e)
        {
            if (_showGeoreferenceToolbarMenuItem != null && _georeferenceToolbar != null)
            {
                _georeferenceToolbar.Visible = _showGeoreferenceToolbarMenuItem.Active;
                IniSettings.Instance.ShowGeoreferenceToolbar = _showGeoreferenceToolbarMenuItem.Active;
            }
        }

        private void OnFullViewportMode(object? sender, EventArgs e)
        {
            _leftPanel.Visible = false;
            _topToolbar.Visible = false;
            _verticalToolbar.Visible = false;
            _meshEditorToolbar.Visible = false;
            _pointCloudToolbar.Visible = false;
            _georeferenceToolbar.Visible = false;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = false;
            if (_showTopToolbarMenuItem != null) _showTopToolbarMenuItem.Active = false;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = false;
            if (_showMeshEditorToolbarMenuItem != null) _showMeshEditorToolbarMenuItem.Active = false;
            if (_showPointCloudToolbarMenuItem != null) _showPointCloudToolbarMenuItem.Active = false;
            if (_showGeoreferenceToolbarMenuItem != null) _showGeoreferenceToolbarMenuItem.Active = false;
        }

        private void OnRestoreAllPanels(object? sender, EventArgs e)
        {
            _leftPanel.Visible = true;
            _topToolbar.Visible = true;
            _verticalToolbar.Visible = true;
            _meshEditorToolbar.Visible = true;
            _pointCloudToolbar.Visible = true;
            _georeferenceToolbar.Visible = true;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = true;
            if (_showTopToolbarMenuItem != null) _showTopToolbarMenuItem.Active = true;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = true;
            if (_showMeshEditorToolbarMenuItem != null) _showMeshEditorToolbarMenuItem.Active = true;
            if (_showPointCloudToolbarMenuItem != null) _showPointCloudToolbarMenuItem.Active = true;
            if (_showGeoreferenceToolbarMenuItem != null) _showGeoreferenceToolbarMenuItem.Active = true;
        }
    }
}
