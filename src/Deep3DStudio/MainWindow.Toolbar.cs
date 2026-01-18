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

            // Auto Workflow Toggle
            _autoWorkflowToggle = new ToggleToolButton();
            _autoWorkflowToggle.IconWidget = AppIconFactory.GenerateIcon("link", iconSize);
            _autoWorkflowToggle.Label = "Auto";
            _autoWorkflowToggle.TooltipText = "Auto Workflow: When ON, Play runs full pipeline. When OFF, run each step manually.";
            _autoWorkflowToggle.Active = _autoWorkflowEnabled;
            _autoWorkflowToggle.Toggled += (s, e) => {
                _autoWorkflowEnabled = _autoWorkflowToggle.Active;
                _autoWorkflowToggle.TooltipText = _autoWorkflowEnabled
                    ? "Auto Workflow: ON (Play runs full pipeline)"
                    : "Auto Workflow: OFF (Run each step manually)";
            };
            toolbar.Insert(_autoWorkflowToggle, -1);

            // Workflow - first option uses engine from Settings
            var wfItem = new ToolItem();
            var wfBox = new Box(Orientation.Horizontal, 5);
            wfBox.PackStart(new Label("Workflow: "), false, false, 0);
            _workflowCombo = new ComboBoxText();
            _workflowCombo.AppendText($"Multi-View ({GetCurrentEngineName()})"); // Uses Settings engine
            _workflowCombo.AppendText("Feature Matching (SfM)");
            _workflowCombo.AppendText("TripoSR (Single Image)");
            _workflowCombo.AppendText("LGM (Gaussian)");
            _workflowCombo.AppendText("Wonder3D");
            _workflowCombo.Active = 0;
            wfBox.PackStart(_workflowCombo, false, false, 0);
            wfItem.Add(wfBox);
            toolbar.Insert(wfItem, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Run Point Cloud - uses engine from Settings
            var runPointsBtn = new ToolButton(AppIconFactory.GenerateIcon("pointcloud", iconSize), "Points");
            runPointsBtn.TooltipText = $"Generate Point Cloud with {GetCurrentEngineName()} (standalone)";
            runPointsBtn.Clicked += (s, e) => OnRunSingleStep(GetReconstructionStep());
            toolbar.Insert(runPointsBtn, -1);

            var runMeshBtn = new ToolButton(AppIconFactory.GenerateIcon("mesh", iconSize), "Mesh");
            runMeshBtn.TooltipText = "Generate Mesh from existing Point Cloud (standalone)";
            runMeshBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.PoissonReconstruction);
            toolbar.Insert(runMeshBtn, -1);

            var runBtn = new ToolButton(AppIconFactory.GenerateIcon("run", iconSize), "Run");
            runBtn.TooltipText = "Run (Auto: full workflow / Manual: selected step)";
            runBtn.Clicked += OnRunInference;
            toolbar.Insert(runBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Standalone AI Model Operations
            var tripoSRBtn = new ToolButton(AppIconFactory.GenerateIcon("ai_single", iconSize), "TripoSR");
            tripoSRBtn.TooltipText = "TripoSR: Single image to 3D (standalone)";
            tripoSRBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.TripoSRGeneration);
            toolbar.Insert(tripoSRBtn, -1);

            var lgmBtn = new ToolButton(AppIconFactory.GenerateIcon("ai_gauss", iconSize), "LGM");
            lgmBtn.TooltipText = "LGM: Large Gaussian Model (standalone)";
            lgmBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.LGMGeneration);
            toolbar.Insert(lgmBtn, -1);

            var wonder3DBtn = new ToolButton(AppIconFactory.GenerateIcon("ai_multi", iconSize), "Wonder3D");
            wonder3DBtn.TooltipText = "Wonder3D: Multi-view 3D generation (standalone)";
            wonder3DBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.Wonder3DGeneration);
            toolbar.Insert(wonder3DBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Refinement Operations
            var nerfBtn = new ToolButton(AppIconFactory.GenerateIcon("nerf", iconSize), "NeRF");
            nerfBtn.TooltipText = "NeRF Refinement (standalone)";
            nerfBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.NeRFRefinement);
            toolbar.Insert(nerfBtn, -1);

            var aiRefineBtn = new ToolButton(AppIconFactory.GenerateIcon("refine", iconSize), "Refine");
            aiRefineBtn.TooltipText = "Refine mesh with AI models (TripoSF/DeepMeshPrior/GaussianSDF)";
            aiRefineBtn.Clicked += OnAIRefine;
            toolbar.Insert(aiRefineBtn, -1);

            var aiRigBtn = new ToolButton(AppIconFactory.GenerateIcon("rig", iconSize), "UniRig");
            aiRigBtn.TooltipText = "Auto-rig mesh with UniRig (standalone)";
            aiRigBtn.Clicked += (s, e) => OnRunSingleStep(AIModels.WorkflowStep.UniRigAutoRig);
            toolbar.Insert(aiRigBtn, -1);

            return toolbar;
        }
    }
}
