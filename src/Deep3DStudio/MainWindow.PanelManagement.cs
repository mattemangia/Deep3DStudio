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
            if (_showSceneTreeMenuItem != null)
            {
                _leftPanel.Visible = _showSceneTreeMenuItem.Active;
            }
        }

        private void OnToggleVerticalToolbar(object? sender, EventArgs e)
        {
            if (_showVerticalToolbarMenuItem != null)
            {
                _verticalToolbar.Visible = _showVerticalToolbarMenuItem.Active;
            }
        }

        private void OnFullViewportMode(object? sender, EventArgs e)
        {
            _leftPanel.Visible = false;
            _verticalToolbar.Visible = false;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = false;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = false;
        }

        private void OnRestoreAllPanels(object? sender, EventArgs e)
        {
            _leftPanel.Visible = true;
            _verticalToolbar.Visible = true;

            if (_showSceneTreeMenuItem != null) _showSceneTreeMenuItem.Active = true;
            if (_showVerticalToolbarMenuItem != null) _showVerticalToolbarMenuItem.Active = true;
        }
    }
}
