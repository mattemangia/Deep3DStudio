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
        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 24;

            // Select Tool
            var selectBtn = new ToolButton(AppIconFactory.GenerateIcon("select", iconSize), "Select");
            selectBtn.TooltipText = "Select Objects (Q)";
            selectBtn.Clicked += (s, e) => _viewport.SetGizmoMode(GizmoMode.Select);
            toolbar.Insert(selectBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Open Files
            var openBtn = new ToolButton(AppIconFactory.GenerateIcon("open", iconSize), "Open Images");
            openBtn.TooltipText = "Load input images for reconstruction";
            openBtn.Clicked += OnAddImages;
            toolbar.Insert(openBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Settings
            var settingsBtn = new ToolButton(AppIconFactory.GenerateIcon("settings", iconSize), "Settings");
            settingsBtn.TooltipText = "Configure Processing, Meshing, and GPU";
            settingsBtn.Clicked += OnOpenSettings;
            toolbar.Insert(settingsBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // View Toggles
            _meshToggle = new ToggleToolButton();
            _meshToggle.IconWidget = AppIconFactory.GenerateIcon("mesh", iconSize);
            _meshToggle.Label = "Mesh";
            _meshToggle.TooltipText = "Show Solid Mesh";
            _meshToggle.Active = IniSettings.Instance.ShowMesh;
            _meshToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowMesh = _meshToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_meshToggle, -1);

            _pointsToggle = new ToggleToolButton();
            _pointsToggle.IconWidget = AppIconFactory.GenerateIcon("pointcloud", iconSize);
            _pointsToggle.Label = "Points";
            _pointsToggle.TooltipText = "Show Point Cloud";
            _pointsToggle.Active = IniSettings.Instance.ShowPointCloud;
            _pointsToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowPointCloud = _pointsToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_pointsToggle, -1);

            _wireToggle = new ToggleToolButton();
            _wireToggle.IconWidget = AppIconFactory.GenerateIcon("wireframe", iconSize);
            _wireToggle.Label = "Wireframe";
            _wireToggle.TooltipText = "Toggle Wireframe Overlay";
            _wireToggle.Active = IniSettings.Instance.ShowWireframe;
            _wireToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowWireframe = _wireToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_wireToggle, -1);

            _textureToggle = new ToggleToolButton();
            _textureToggle.IconWidget = AppIconFactory.GenerateIcon("texture", iconSize);
            _textureToggle.Label = "Texture";
            _textureToggle.TooltipText = "Toggle Texture Display";
            _textureToggle.Active = IniSettings.Instance.ShowTexture;
            _textureToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowTexture = _textureToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_textureToggle, -1);

            _camerasToggle = new ToggleToolButton();
            _camerasToggle.IconWidget = AppIconFactory.GenerateIcon("camera", iconSize);
            _camerasToggle.Label = "Cameras";
            _camerasToggle.TooltipText = "Toggle Camera Frustums";
            _camerasToggle.Active = IniSettings.Instance.ShowCameras;
            _camerasToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowCameras = _camerasToggle.Active;
                _viewport.ShowCameras = _camerasToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_camerasToggle, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Point Cloud Color Mode Toggles
            _rgbColorToggle = new ToggleToolButton();
            _rgbColorToggle.IconWidget = AppIconFactory.GenerateIcon("rgb", iconSize);
            _rgbColorToggle.Label = "RGB";
            _rgbColorToggle.TooltipText = "Show original RGB colors";
            _rgbColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            _rgbColorToggle.Toggled += (s, e) => {
                if (_rgbColorToggle.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
                    _depthColorToggle.Active = false;
                    _viewport.QueueDraw();
                }
                else if (!_depthColorToggle.Active)
                {
                    _rgbColorToggle.Active = true;
                }
            };
            toolbar.Insert(_rgbColorToggle, -1);

            _depthColorToggle = new ToggleToolButton();
            _depthColorToggle.IconWidget = AppIconFactory.GenerateIcon("depthmap", iconSize);
            _depthColorToggle.Label = "Depth";
            _depthColorToggle.TooltipText = "Show distance map with colormap";
            _depthColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            _depthColorToggle.Toggled += (s, e) => {
                if (_depthColorToggle.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
                    _rgbColorToggle.Active = false;
                    _viewport.QueueDraw();
                }
                else if (!_rgbColorToggle.Active)
                {
                    _depthColorToggle.Active = true;
                }
            };
            toolbar.Insert(_depthColorToggle, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Workflow
            var wfItem = new ToolItem();
            var wfBox = new Box(Orientation.Horizontal, 5);
            wfBox.PackStart(new Label("Workflow: "), false, false, 0);
            _workflowCombo = new ComboBoxText();
            _workflowCombo.AppendText("Dust3r (Fast)");
            _workflowCombo.AppendText("NeRF (Refined)");
            _workflowCombo.AppendText("Interior Scan");
            _workflowCombo.AppendText("TripoSR (Single Image)");
            _workflowCombo.AppendText("LGM (High Quality)");
            _workflowCombo.AppendText("Wonder3D (Multi-View)");
            _workflowCombo.AppendText("Dust3r + DeepMeshPrior");
            _workflowCombo.AppendText("Dust3r + NeRF + DeepMeshPrior");
            _workflowCombo.AppendText("Full Pipeline (Mesh Only)");
            _workflowCombo.Active = 0;
            wfBox.PackStart(_workflowCombo, false, false, 0);
            wfItem.Add(wfBox);
            toolbar.Insert(wfItem, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Run
            var runPointsBtn = new ToolButton(AppIconFactory.GenerateIcon("pointcloud", iconSize), "Gen Points");
            runPointsBtn.TooltipText = "Generate Point Cloud Only";
            runPointsBtn.Clicked += OnGeneratePointCloud;
            toolbar.Insert(runPointsBtn, -1);

            var runMeshBtn = new ToolButton(AppIconFactory.GenerateIcon("mesh", iconSize), "Gen Mesh");
            runMeshBtn.TooltipText = "Generate Mesh from existing Point Cloud";
            runMeshBtn.Clicked += OnGenerateMesh;
            toolbar.Insert(runMeshBtn, -1);

            var runBtn = new ToolButton(AppIconFactory.GenerateIcon("run", iconSize), "Run All");
            runBtn.TooltipText = "Start Full Reconstruction Process";
            runBtn.Clicked += OnRunInference;
            toolbar.Insert(runBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // AI Model Operations
            var aiRigBtn = new ToolButton(AppIconFactory.GenerateIcon("rig", iconSize), "Auto Rig");
            aiRigBtn.TooltipText = "Auto-rig mesh with UniRig";
            aiRigBtn.Clicked += OnAutoRig;
            toolbar.Insert(aiRigBtn, -1);

            var aiRefineBtn = new ToolButton(AppIconFactory.GenerateIcon("refine", iconSize), "AI Refine");
            aiRefineBtn.TooltipText = "Refine mesh with AI models (TripoSF/DeepMeshPrior/GaussianSDF)";
            aiRefineBtn.Clicked += OnAIRefine;
            toolbar.Insert(aiRefineBtn, -1);

            return toolbar;
        }
    }
}
