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
        private Dictionary<Toolbar, List<ToolItem>> _originalToolbarItems = new Dictionary<Toolbar, List<ToolItem>>();
        private Widget? _currentTooltipTarget = null;
        private HashSet<Widget> _hookedWidgets = new HashSet<Widget>();

        private void UpdateStatusText(string text)
        {
            if (_statusLabel == null) return;
            string escaped = System.Security.SecurityElement.Escape(text);
            // Black bold text for active tooltips
            _statusLabel.Markup = $"<span foreground='#000000'><b>{escaped}</b></span>";
        }

        private void ResetStatusText()
        {
            if (_statusLabel == null) return;
            // Dark gray for Ready state
            _statusLabel.Markup = "<span foreground='#333333'><b>Ready</b></span>";
        }

        private void ApplyStatusBarFeedback(Widget widget, string text)
        {
            if (widget == null) return;

            void Hook(Widget w)
            {
                if (_hookedWidgets.Contains(w)) return;
                _hookedWidgets.Add(w);

                w.Events |= EventMask.EnterNotifyMask | EventMask.LeaveNotifyMask;
                
                string targetText = text;
                w.EnterNotifyEvent += (o, args) => {
                    _currentTooltipTarget = widget; 
                    UpdateStatusText(targetText);
                };
                
                w.LeaveNotifyEvent += (o, args) => {
                    if (_currentTooltipTarget == widget)
                    {
                        _currentTooltipTarget = null;
                        ResetStatusText();
                    }
                };

                if (w is Container container)
                {
                    foreach (var child in container.Children)
                        Hook(child);
                }
                
                if (w is ToolButton tb)
                {
                    if (tb.IconWidget != null) Hook(tb.IconWidget);
                    if (tb.LabelWidget != null) Hook(tb.LabelWidget);
                }
            }

            Hook(widget);
        }

        private void RegisterToolbarItems(Toolbar toolbar)
        {
            var items = new List<ToolItem>();
            for (int i = 0; i < toolbar.NItems; i++)
                items.Add(toolbar.GetNthItem(i));
            _originalToolbarItems[toolbar] = items;
        }

        private void SetTooltip(Widget widget, string text)
        {
            widget.TooltipText = text;
            widget.HasTooltip = true;
            ApplyStatusBarFeedback(widget, text);
        }

        private void UpdateAllToolbarsOverflow()
        {
            if (_topToolbar is Toolbar top) UpdateToolbarOverflow(top);
            if (_meshEditorToolbar is Toolbar mesh) UpdateToolbarOverflow(mesh);
            if (_pointCloudToolbar is Toolbar pc) UpdateToolbarOverflow(pc);
            if (_georeferenceToolbar is Toolbar geo) UpdateToolbarOverflow(geo);
        }

        private void UpdateToolbarOverflow(Toolbar toolbar)
        {
            if (!_originalToolbarItems.TryGetValue(toolbar, out var originalItems))
                return;

            int windowWidth = this.Allocation.Width;
            int availableWidth = windowWidth - 60; 
            if (_verticalToolbar.Visible) availableWidth -= 40;
            
            int currentWidth = 20;

            for (int i = toolbar.NItems - 1; i >= 0; i--)
                toolbar.Remove(toolbar.GetNthItem(i));

            var overflowItems = new List<ToolItem>();

            foreach (var item in originalItems)
            {
                int itemWidth = item.SizeRequest().Width;
                if (itemWidth <= 0) itemWidth = 32;

                if (currentWidth + itemWidth + 45 < availableWidth)
                {
                    toolbar.Insert(item, -1);
                    currentWidth += itemWidth + 4;
                }
                else
                {
                    overflowItems.Add(item);
                }
            }

            // Ensure events are always active (but protection logic prevents duplicates)
            foreach (var item in originalItems)
            {
                if (!string.IsNullOrEmpty(item.TooltipText))
                    ApplyStatusBarFeedback(item, item.TooltipText);
            }

            if (overflowItems.Count > 0)
            {
                var overflowMenu = new Menu();
                foreach (var item in overflowItems)
                {
                    if (item is SeparatorToolItem)
                    {
                        overflowMenu.Append(new SeparatorMenuItem());
                        continue;
                    }

                    MenuItem mi;
                    string menuText = item.TooltipText ?? "";
                    
                    if (item is ToolButton tb)
                    {
                        mi = new ImageMenuItem(tb.Label);
                        if (tb.IconWidget is Image img)
                        {
                            var menuImg = new Image();
                            if (img.Pixbuf != null) menuImg.Pixbuf = img.Pixbuf.ScaleSimple(16, 16, InterpType.Bilinear);
                            else if (!string.IsNullOrEmpty(img.IconName)) menuImg.IconName = img.IconName;
                            ((ImageMenuItem)mi).Image = menuImg;
                        }
                        mi.Activated += (s, e) => tb.Activate();
                    }
                    else if (item is ToggleToolButton ttb)
                    {
                        mi = new CheckMenuItem(ttb.Label);
                        ((CheckMenuItem)mi).Active = ttb.Active;
                        mi.Activated += (s, e) => { ttb.Active = ((CheckMenuItem)mi).Active; };
                    }
                    else
                    {
                        mi = new MenuItem(item.GetType().Name);
                    }
                    
                    if (!string.IsNullOrEmpty(menuText))
                    {
                        mi.EnterNotifyEvent += (s, e) => UpdateStatusText(menuText);
                        mi.LeaveNotifyEvent += (s, e) => ResetStatusText();
                    }
                    
                    overflowMenu.Append(mi);
                }

                var overflowBtn = new MenuToolButton(AppIconFactory.GenerateIcon("settings", 20), "More...");
                overflowBtn.Menu = overflowMenu;
                toolbar.Insert(overflowBtn, -1);
                overflowMenu.ShowAll();
            }

            toolbar.ShowAll();
        }

        private ToolButton CreateActionButton(
            string iconKey,
            int iconSize,
            string label,
            string tooltip,
            EventHandler onClick,
            Action? onRightClickOptions = null)
        {
            var btn = new ToolButton(AppIconFactory.GenerateIcon(iconKey, iconSize), label);
            string finalTooltip = onRightClickOptions == null ? tooltip : $"{tooltip} (Right click for options)";
            SetTooltip(btn, finalTooltip);
            btn.Clicked += onClick;

            if (onRightClickOptions != null)
            {
                btn.Events |= EventMask.ButtonPressMask;
                btn.ButtonPressEvent += (s, e) =>
                {
                    if (e.Event.Button == 3)
                    {
                        onRightClickOptions();
                        e.RetVal = true;
                    }
                };
            }

            return btn;
        }

        private void ConfigureRunOptions()
        {
            using var dlg = new Dialog("Run Options", this, DialogFlags.Modal);
            dlg.SetDefaultSize(420, 220);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Apply", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var autoCheck = new CheckButton("Auto workflow (Run executes full pipeline)") { Active = _autoWorkflowEnabled };
            area.PackStart(autoCheck, false, false, 0);

            var recCombo = new ComboBoxText();
            var methods = Enum.GetValues(typeof(ReconstructionMethod)).Cast<ReconstructionMethod>().ToList();
            foreach (var method in methods)
                recCombo.AppendText(method.ToString());
            recCombo.Active = methods.IndexOf(IniSettings.Instance.ReconstructionMethod);

            var row = new Box(Orientation.Horizontal, 8);
            row.PackStart(new Label("Reconstruction Method") { Halign = Align.Start, WidthRequest = 170 }, false, false, 0);
            row.PackStart(recCombo, true, true, 0);
            area.PackStart(row, false, false, 0);

            area.PackStart(new Label("Workflow preset (manual mode):"), false, false, 0);
            var wfCombo = new ComboBoxText();
            wfCombo.AppendText($"Multi-View ({GetCurrentEngineName()})");
            wfCombo.AppendText("Feature Matching (SfM)");
            wfCombo.AppendText("TripoSR (Single Image)");
            wfCombo.AppendText("LGM (Single Image)");
            wfCombo.AppendText("Wonder3D (Single Image)");
            wfCombo.Active = _workflowCombo.Active;
            area.PackStart(wfCombo, false, false, 0);

            dlg.ShowAll();
            if (dlg.Run() == (int)ResponseType.Ok)
            {
                _autoWorkflowEnabled = autoCheck.Active;
                if (_autoWorkflowToggle != null)
                    _autoWorkflowToggle.Active = _autoWorkflowEnabled;

                if (recCombo.Active >= 0 && recCombo.Active < methods.Count)
                    IniSettings.Instance.ReconstructionMethod = methods[recCombo.Active];

                if (wfCombo.Active >= 0)
                    _workflowCombo.Active = wfCombo.Active;
            }
        }

        private void ConfigureReconstructionOptions() => ConfigureRunOptions();

        private void ConfigureMeshingOptions()
        {
            using var dlg = new Dialog("Meshing Options", this, DialogFlags.Modal);
            dlg.SetDefaultSize(380, 160);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Apply", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var combo = new ComboBoxText();
            var algos = Enum.GetValues(typeof(MeshingAlgorithm)).Cast<MeshingAlgorithm>().ToList();
            foreach (var algo in algos)
                combo.AppendText(algo.ToString());
            combo.Active = algos.IndexOf(IniSettings.Instance.MeshingAlgo);

            var row = new Box(Orientation.Horizontal, 8);
            row.PackStart(new Label("Meshing Algorithm") { Halign = Align.Start, WidthRequest = 150 }, false, false, 0);
            row.PackStart(combo, true, true, 0);
            area.PackStart(row, false, false, 0);

            dlg.ShowAll();
            if (dlg.Run() == (int)ResponseType.Ok && combo.Active >= 0 && combo.Active < algos.Count)
            {
                IniSettings.Instance.MeshingAlgo = algos[combo.Active];
            }
        }

        private void ConfigureRefinementOptions()
        {
            using var dlg = new Dialog("Refinement Options", this, DialogFlags.Modal);
            dlg.SetDefaultSize(380, 160);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("Apply", ResponseType.Ok);

            var area = dlg.ContentArea;
            area.BorderWidth = 10;
            area.Spacing = 8;

            var combo = new ComboBoxText();
            var methods = Enum.GetValues(typeof(MeshRefinementMethod)).Cast<MeshRefinementMethod>().ToList();
            foreach (var method in methods)
                combo.AppendText(method.ToString());
            combo.Active = methods.IndexOf(IniSettings.Instance.MeshRefinement);

            var row = new Box(Orientation.Horizontal, 8);
            row.PackStart(new Label("Refinement Method") { Halign = Align.Start, WidthRequest = 150 }, false, false, 0);
            row.PackStart(combo, true, true, 0);
            area.PackStart(row, false, false, 0);

            dlg.ShowAll();
            if (dlg.Run() == (int)ResponseType.Ok && combo.Active >= 0 && combo.Active < methods.Count)
            {
                IniSettings.Instance.MeshRefinement = methods[combo.Active];
            }
        }

        private void ConfigureAiModelOptions() => OnAIModelSettings(null, EventArgs.Empty);

        private Widget CreateToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 24;

            // Select Tool
            var selectBtn = new ToolButton(AppIconFactory.GenerateIcon("select", iconSize), "Select");
            SetTooltip(selectBtn, "Select Objects (Q)");
            selectBtn.Clicked += (s, e) => _viewport.SetGizmoMode(GizmoMode.Select);
            toolbar.Insert(selectBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Open Files
            var openBtn = new ToolButton(AppIconFactory.GenerateIcon("open", iconSize), "Open Images");
            SetTooltip(openBtn, "Load input images for reconstruction");
            openBtn.Clicked += OnAddImages;
            toolbar.Insert(openBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Settings
            var settingsBtn = new ToolButton(AppIconFactory.GenerateIcon("settings", iconSize), "Settings");
            SetTooltip(settingsBtn, "Configure Processing, Meshing, and GPU");
            settingsBtn.Clicked += OnOpenSettings;
            toolbar.Insert(settingsBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // View Toggles
            _meshToggle = new ToggleToolButton();
            _meshToggle.IconWidget = AppIconFactory.GenerateIcon("mesh", iconSize);
            _meshToggle.Label = "Mesh";
            SetTooltip(_meshToggle, "Show Solid Mesh");
            _meshToggle.Active = IniSettings.Instance.ShowMesh;
            _meshToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowMesh = _meshToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_meshToggle, -1);

            _pointsToggle = new ToggleToolButton();
            _pointsToggle.IconWidget = AppIconFactory.GenerateIcon("pointcloud", iconSize);
            _pointsToggle.Label = "Points";
            SetTooltip(_pointsToggle, "Show Point Cloud");
            _pointsToggle.Active = IniSettings.Instance.ShowPointCloud;
            _pointsToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowPointCloud = _pointsToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_pointsToggle, -1);

            _wireToggle = new ToggleToolButton();
            _wireToggle.IconWidget = AppIconFactory.GenerateIcon("wireframe", iconSize);
            _wireToggle.Label = "Wireframe";
            SetTooltip(_wireToggle, "Toggle Wireframe Overlay");
            _wireToggle.Active = IniSettings.Instance.ShowWireframe;
            _wireToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowWireframe = _wireToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_wireToggle, -1);

            _textureToggle = new ToggleToolButton();
            _textureToggle.IconWidget = AppIconFactory.GenerateIcon("texture", iconSize);
            _textureToggle.Label = "Texture";
            SetTooltip(_textureToggle, "Toggle Texture Display");
            _textureToggle.Active = IniSettings.Instance.ShowTexture;
            _textureToggle.Toggled += (s, e) => {
                IniSettings.Instance.ShowTexture = _textureToggle.Active;
                _viewport.QueueDraw();
            };
            toolbar.Insert(_textureToggle, -1);

            _camerasToggle = new ToggleToolButton();
            _camerasToggle.IconWidget = AppIconFactory.GenerateIcon("camera", iconSize);
            _camerasToggle.Label = "Cameras";
            SetTooltip(_camerasToggle, "Toggle Camera Frustums");
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
            SetTooltip(_rgbColorToggle, "Show original RGB colors");
            _rgbColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            _rgbColorToggle.Toggled += (s, e) => {
                if (_updatingPointCloudColorToggles) return;
                if (_rgbColorToggle.Active) SetPointCloudColorMode(PointCloudColorMode.RGB);
                else SyncPointCloudColorToggles();
            };
            toolbar.Insert(_rgbColorToggle, -1);

            _depthColorToggle = new ToggleToolButton();
            _depthColorToggle.IconWidget = AppIconFactory.GenerateIcon("depthmap", iconSize);
            _depthColorToggle.Label = "Depth";
            SetTooltip(_depthColorToggle, "Show distance map with colormap");
            _depthColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            _depthColorToggle.Toggled += (s, e) => {
                if (_updatingPointCloudColorToggles) return;
                if (_depthColorToggle.Active) SetPointCloudColorMode(PointCloudColorMode.DistanceMap);
                else SyncPointCloudColorToggles();
            };
            toolbar.Insert(_depthColorToggle, -1);

            _confidenceColorToggle = new ToggleToolButton();
            _confidenceColorToggle.IconWidget = AppIconFactory.GenerateIcon("confidence", iconSize);
            _confidenceColorToggle.Label = "Conf";
            SetTooltip(_confidenceColorToggle, "Show confidence map with colormap");
            _confidenceColorToggle.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.Confidence;
            _confidenceColorToggle.Toggled += (s, e) => {
                if (_updatingPointCloudColorToggles) return;
                if (_confidenceColorToggle.Active) SetPointCloudColorMode(PointCloudColorMode.Confidence);
                else SyncPointCloudColorToggles();
            };
            toolbar.Insert(_confidenceColorToggle, -1);
            SyncPointCloudColorToggles();

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Auto Workflow Toggle
            _autoWorkflowToggle = new ToggleToolButton();
            _autoWorkflowToggle.IconWidget = AppIconFactory.GenerateIcon("link", iconSize);
            _autoWorkflowToggle.Label = "Auto";
            SetTooltip(_autoWorkflowToggle, "Auto Workflow: When ON, Play runs full pipeline. When OFF, run each step manually.");
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
            SetTooltip(_workflowCombo, "Select the reconstruction pipeline to use");
            _workflowCombo.AppendText($"Multi-View ({GetCurrentEngineName()})"); // Uses Settings engine
            _workflowCombo.AppendText("Feature Matching (SfM)");
            _workflowCombo.AppendText("TripoSR (Single Image)");
            _workflowCombo.AppendText("LGM (Single Image)");
            _workflowCombo.AppendText("Wonder3D (Single Image)");
            _workflowCombo.Active = 0;
            wfBox.PackStart(_workflowCombo, false, false, 0);
            wfItem.Add(wfBox);
            toolbar.Insert(wfItem, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Run Point Cloud - uses engine from Settings
            var runPointsBtn = CreateActionButton(
                "pointcloud",
                iconSize,
                "Points",
                $"Generate Point Cloud with {GetCurrentEngineName()} (standalone)",
                OnGeneratePointCloud,
                ConfigureReconstructionOptions);
            toolbar.Insert(runPointsBtn, -1);

            var runMeshBtn = CreateActionButton(
                "mesh",
                iconSize,
                "Mesh",
                "Generate Mesh from existing Point Cloud (standalone)",
                OnGenerateMesh,
                ConfigureMeshingOptions);
            toolbar.Insert(runMeshBtn, -1);

            var runBtn = CreateActionButton(
                "run",
                iconSize,
                "Run",
                "Run (Auto: full workflow / Manual: selected step)",
                OnRunInference,
                ConfigureRunOptions);
            toolbar.Insert(runBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Standalone AI Model Operations
            var tripoSRBtn = CreateActionButton(
                "ai_single",
                iconSize,
                "TripoSR",
                "TripoSR: Single image to 3D (standalone)",
                (s, e) => OnRunSingleStep(AIModels.WorkflowStep.TripoSRGeneration),
                ConfigureAiModelOptions);
            toolbar.Insert(tripoSRBtn, -1);

            var lgmBtn = CreateActionButton(
                "ai_gauss",
                iconSize,
                "LGM",
                "LGM: Single-image Gaussian model (standalone)",
                (s, e) => OnRunSingleStep(AIModels.WorkflowStep.LGMGeneration),
                ConfigureAiModelOptions);
            toolbar.Insert(lgmBtn, -1);

            var wonder3DBtn = CreateActionButton(
                "ai_multi",
                iconSize,
                "Wonder3D",
                "Wonder3D: Single-image multi-view 3D generation (standalone)",
                (s, e) => OnRunSingleStep(AIModels.WorkflowStep.Wonder3DGeneration),
                ConfigureAiModelOptions);
            toolbar.Insert(wonder3DBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Refinement Operations
            var nerfBtn = CreateActionButton(
                "nerf",
                iconSize,
                "NeRF",
                "NeRF Refinement (standalone)",
                (s, e) => OnRunSingleStep(AIModels.WorkflowStep.NeRFRefinement),
                ConfigureRefinementOptions);
            toolbar.Insert(nerfBtn, -1);

            var aiRefineBtn = CreateActionButton(
                "refine",
                iconSize,
                "Refine",
                "Refine mesh with AI models (TripoSF/DeepMeshPrior/GaussianSDF)",
                OnAIRefine,
                ConfigureRefinementOptions);
            toolbar.Insert(aiRefineBtn, -1);

            var aiRigBtn = CreateActionButton(
                "rig",
                iconSize,
                "UniRig",
                "Auto-rig mesh with UniRig (standalone)",
                (s, e) => OnRunSingleStep(AIModels.WorkflowStep.UniRigAutoRig),
                ConfigureAiModelOptions);
            toolbar.Insert(aiRigBtn, -1);

            RegisterToolbarItems(toolbar);
            return toolbar;
        }

        private Widget CreateMeshEditorToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 20;

            // Primitive creation
            var planeBtn = CreateActionButton("plane", iconSize, "Plane", "Create Plane Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Plane),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Plane));
            toolbar.Insert(planeBtn, -1);

            var cubeBtn = CreateActionButton("cube", iconSize, "Cube", "Create Cube Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Cube),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Cube));
            toolbar.Insert(cubeBtn, -1);

            var sphereBtn = CreateActionButton("sphere", iconSize, "Sphere", "Create UV Sphere Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.UVSphere),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.UVSphere));
            toolbar.Insert(sphereBtn, -1);

            var cylinderBtn = CreateActionButton("cylinder", iconSize, "Cylinder", "Create Cylinder Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Cylinder),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Cylinder));
            toolbar.Insert(cylinderBtn, -1);

            var coneBtn = CreateActionButton("cone", iconSize, "Cone", "Create Cone Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Cone),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Cone));
            toolbar.Insert(coneBtn, -1);

            var torusBtn = CreateActionButton("torus", iconSize, "Torus", "Create Torus Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Torus),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Torus));
            toolbar.Insert(torusBtn, -1);

            var circleBtn = CreateActionButton("circle", iconSize, "Circle", "Create Circle Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Circle),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Circle));
            toolbar.Insert(circleBtn, -1);

            var polygonBtn = CreateActionButton("polygon", iconSize, "Polygon", "Create Polygon Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Polygon),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Polygon));
            toolbar.Insert(polygonBtn, -1);

            var gridBtn = CreateActionButton("grid_mesh", iconSize, "Grid", "Create Grid Primitive",
                (s, e) => OnCreatePrimitive(MeshPrimitiveType.Grid),
                () => ConfigurePrimitiveOptions(MeshPrimitiveType.Grid));
            toolbar.Insert(gridBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Mesh operations
            var decimateBtn = CreateActionButton("decimate", iconSize, "Decimate", "Decimate Selected Mesh",
                OnDecimateClicked, ConfigureDecimateOptions);
            toolbar.Insert(decimateBtn, -1);

            var smoothBtn = CreateActionButton("smooth", iconSize, "Smooth", "Smooth Selected Mesh",
                OnSmoothClicked, ConfigureSmoothOptions);
            toolbar.Insert(smoothBtn, -1);

            var optimizeBtn = CreateActionButton("optimize", iconSize, "Optimize", "Optimize Selected Mesh",
                OnOptimizeClicked, ConfigureOptimizeOptions);
            toolbar.Insert(optimizeBtn, -1);

            var fillHolesBtn = CreateActionButton("fill_holes", iconSize, "Fill Holes", "Fill Holes in Selected Mesh",
                OnFillHolesClicked, null);
            toolbar.Insert(fillHolesBtn, -1);

            var mergeMeshesBtn = CreateActionButton("merge_meshes", iconSize, "Merge", "Merge selected meshes (Merge / Align+Merge)",
                OnMergeMeshesToolbarClicked, null);
            toolbar.Insert(mergeMeshesBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            // Pen/face editing actions
            var moveVertBtn = CreateActionButton("vertex_move", iconSize, "Move Vtx", "Move Vertices in Selected Triangles",
                OnMoveSelectedVertices, ConfigureMoveSelectedVertices);
            toolbar.Insert(moveVertBtn, -1);

            var extrudeBtn = CreateActionButton("extrude", iconSize, "Extrude", "Extrude Selected Triangles",
                OnExtrudeSelectedTriangles, ConfigureExtrudeSelectedTriangles);
            toolbar.Insert(extrudeBtn, -1);

            var insetBtn = CreateActionButton("inset", iconSize, "Inset", "Inset Selected Triangles",
                OnInsetSelectedTriangles, ConfigureInsetSelectedTriangles);
            toolbar.Insert(insetBtn, -1);

            var bridgeBtn = CreateActionButton("bridge", iconSize, "Bridge", "Bridge Two Selected Triangles",
                OnBridgeSelectedTriangles, null);
            toolbar.Insert(bridgeBtn, -1);

            RegisterToolbarItems(toolbar);
            return toolbar;
        }

        private Widget CreatePointCloudToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 20;

            var voxelBtn = CreateActionButton("voxel_filter", iconSize, "Voxel", "Voxel Downsample Selected Point Cloud",
                OnPointCloudVoxelDownsampleClicked, ConfigurePointCloudVoxel);
            toolbar.Insert(voxelBtn, -1);

            var outlierBtn = CreateActionButton("outlier_filter", iconSize, "Outliers", "Remove Statistical Outliers",
                OnPointCloudRemoveOutliersClicked, ConfigurePointCloudOutliers);
            toolbar.Insert(outlierBtn, -1);

            var dupBtn = CreateActionButton("duplicate_filter", iconSize, "Duplicates", "Remove Duplicate Points",
                OnPointCloudRemoveDuplicatesClicked, ConfigurePointCloudDuplicates);
            toolbar.Insert(dupBtn, -1);

            var skyBtn = CreateActionButton("outlier_filter", iconSize, "Sky", "Remove Blue/Sky Points",
                OnPointCloudRemoveSkyBlueClicked, ConfigurePointCloudSkyBlue);
            toolbar.Insert(skyBtn, -1);

            var normalBtn = CreateActionButton("normals", iconSize, "Normals", "Estimate Point Cloud Normals",
                OnPointCloudEstimateNormalsClicked, ConfigurePointCloudNormals);
            toolbar.Insert(normalBtn, -1);

            var passBtn = CreateActionButton("axis_filter", iconSize, "Pass", "Pass-Through Axis Filter",
                OnPointCloudPassThroughClicked, ConfigurePointCloudPassThrough);
            toolbar.Insert(passBtn, -1);

            var cropBtn = CreateActionButton("radius_crop", iconSize, "Radius", "Radius Crop",
                OnPointCloudRadiusCropClicked, ConfigurePointCloudRadiusCrop);
            toolbar.Insert(cropBtn, -1);

            var denseBtn = CreateActionButton("dense_cloud", iconSize, "Dense", "Point Cloud to Dense Cloud",
                OnPointCloudDenseClicked, ConfigurePointCloudDense);
            toolbar.Insert(denseBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            var mergeBtn = CreateActionButton("merge_pointclouds", iconSize, "Merge", "Merge selected point clouds (Merge / Align+Merge)",
                OnMergePointCloudsToolbarClicked, null);
            toolbar.Insert(mergeBtn, -1);

            var alignBtn = CreateActionButton("align", iconSize, "Align", "Align Selected Point Clouds (ICP)",
                OnAlignClicked, null);
            toolbar.Insert(alignBtn, -1);

            toolbar.Insert(new SeparatorToolItem(), -1);

            var visibilityItem = new ToolItem();
            var visibilityBox = new Box(Orientation.Horizontal, 4);
            visibilityBox.PackStart(new Label("Visible"), false, false, 0);

            _pcVisibilityScale = new Scale(Orientation.Horizontal, 0, 100, 1)
            {
                DrawValue = false,
                Value = 100,
                WidthRequest = 140,
                Sensitive = false
            };
            SetTooltip(_pcVisibilityScale, "Adjust Point Cloud visibility (0-100%)");
            _pcVisibilityScale.ValueChanged += OnPointCloudVisibilitySliderChanged;
            visibilityBox.PackStart(_pcVisibilityScale, false, false, 0);

            _pcVisibilityLabel = new Label("--")
            {
                WidthRequest = 120
            };
            visibilityBox.PackStart(_pcVisibilityLabel, false, false, 0);

            visibilityItem.Add(visibilityBox);
            toolbar.Insert(visibilityItem, -1);

            RegisterToolbarItems(toolbar);
            return toolbar;
        }

        private Widget CreateGeoreferenceToolbar()
        {
            var toolbar = new Toolbar();
            toolbar.Style = ToolbarStyle.Icons;
            int iconSize = 20;

            var geoEditorBtn = new ToolButton(AppIconFactory.GenerateIcon("georef", iconSize), "GCP");
            SetTooltip(geoEditorBtn, "Apri editor GCP / Georeferenziazione");
            geoEditorBtn.Clicked += (s, e) => OnOpenGeoreferenceEditor();
            toolbar.Insert(geoEditorBtn, -1);

            var solveBtn = new ToolButton(AppIconFactory.GenerateIcon("residuals", iconSize), "Solve");
            SetTooltip(solveBtn, "Risolvi trasformazione Model -> CRS dai GCP");
            solveBtn.Clicked += (s, e) => OnSolveGeoreferenceFromCurrentGcps();
            toolbar.Insert(solveBtn, -1);

            var resolvePendingBtn = new ToolButton(AppIconFactory.GenerateIcon("residuals", iconSize), "Resolve Pending");
            SetTooltip(resolvePendingBtn, "Resolve Pending GCPs by sampling model points from scene");
            resolvePendingBtn.Clicked += (s, e) => OnResolvePendingGcpsFromCurrentScene();
            toolbar.Insert(resolvePendingBtn, -1);

            var demBtn = new ToolButton(AppIconFactory.GenerateIcon("dem", iconSize), "DEM");
            SetTooltip(demBtn, "Esporta DEM (GeoTIFF + ASCII Grid)");
            demBtn.Clicked += (s, e) => OnExportDem(null, EventArgs.Empty);
            toolbar.Insert(demBtn, -1);

            var geoExportBtn = new ToolButton(AppIconFactory.GenerateIcon("geo_export", iconSize), "Geo Export");
            SetTooltip(geoExportBtn, "Esporta mesh/point cloud georeferenziati");
            geoExportBtn.Clicked += (s, e) => OnExportGeoreferencedSelection();
            toolbar.Insert(geoExportBtn, -1);

            RegisterToolbarItems(toolbar);
            return toolbar;
        }
    }
}
