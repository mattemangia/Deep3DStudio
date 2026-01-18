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
        private Widget CreateMenuBar()
        {
            var menuBar = new MenuBar();
            var accelGroup = new AccelGroup();
            this.AddAccelGroup(accelGroup);

            // File Menu
            var fileMenu = new Menu();
            var fileMenuItem = new MenuItem("_File");
            fileMenuItem.Submenu = fileMenu;

            var newProjectItem = new MenuItem("_New Project");
            newProjectItem.Activated += OnNewProject;
            fileMenu.Append(newProjectItem);

            var openItem = new MenuItem("_Open Project...");
            openItem.Activated += OnLoadProject;
            fileMenu.Append(openItem);

            var saveItem = new MenuItem("_Save Project...");
            saveItem.Activated += OnSaveProject;
            fileMenu.Append(saveItem);

            fileMenu.Append(new SeparatorMenuItem());

            var openImagesItem = new MenuItem("_Open Pictures...");
            openImagesItem.Activated += OnAddImages;
            fileMenu.Append(openImagesItem);

            var importMeshItem = new MenuItem("_Import Mesh...");
            importMeshItem.Activated += OnImportMesh;
            fileMenu.Append(importMeshItem);

            var importPointsItem = new MenuItem("Import _Point Cloud...");
            importPointsItem.Activated += OnImportPointCloud;
            fileMenu.Append(importPointsItem);

            fileMenu.Append(new SeparatorMenuItem());

            var exportMeshItem = new MenuItem("_Export Mesh...");
            exportMeshItem.Activated += OnExportMesh;
            fileMenu.Append(exportMeshItem);

            var exportPointsItem = new MenuItem("Export _Point Cloud...");
            exportPointsItem.Activated += OnExportPointCloud;
            fileMenu.Append(exportPointsItem);

            var exportDepthItem = new MenuItem("Export _Depth Maps...");
            exportDepthItem.Activated += OnExportDepthMaps;
            fileMenu.Append(exportDepthItem);

            fileMenu.Append(new SeparatorMenuItem());

            var settingsItem = new MenuItem("_Settings...");
            settingsItem.Activated += OnOpenSettings;
            fileMenu.Append(settingsItem);

            fileMenu.Append(new SeparatorMenuItem());

            var exitItem = new MenuItem("E_xit");
            exitItem.Activated += (s, e) => Application.Quit();
            fileMenu.Append(exitItem);

            menuBar.Append(fileMenuItem);

            // Edit Menu
            var editMenu = new Menu();
            var editMenuItem = new MenuItem("_Edit");
            editMenuItem.Submenu = editMenu;

            var selectAllItem = new MenuItem("Select _All");
            selectAllItem.Activated += (s, e) => _sceneGraph.SelectAll();
            editMenu.Append(selectAllItem);

            var deselectAllItem = new MenuItem("_Deselect All");
            deselectAllItem.Activated += (s, e) => _sceneGraph.ClearSelection();
            editMenu.Append(deselectAllItem);

            editMenu.Append(new SeparatorMenuItem());

            var deleteItem = new MenuItem("_Delete");
            deleteItem.Activated += OnDeleteSelected;
            editMenu.Append(deleteItem);

            var duplicateItem = new MenuItem("D_uplicate");
            duplicateItem.Activated += OnDuplicateSelected;
            editMenu.Append(duplicateItem);

            editMenu.Append(new SeparatorMenuItem());

            var settingsItemEdit = new MenuItem("_Settings");
            settingsItemEdit.Activated += OnOpenSettings;
            settingsItemEdit.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.comma, Gdk.ModifierType.ControlMask, AccelFlags.Visible);
            editMenu.Append(settingsItemEdit);

            editMenu.Append(new SeparatorMenuItem());

            // Transform submenu
            var transformMenu = new Menu();
            var transformMenuItem = new MenuItem("_Transform");
            transformMenuItem.Submenu = transformMenu;

            _translateMenuItem = new RadioMenuItem("_Move (W)");
            _translateMenuItem.Active = true;
            _translateMenuItem.Toggled += (s, e) => {
                if (_translateMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Translate);
            };
            transformMenu.Append(_translateMenuItem);

            _rotateMenuItem = new RadioMenuItem(_translateMenuItem, "_Rotate (E)");
            _rotateMenuItem.Toggled += (s, e) => {
                if (_rotateMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Rotate);
            };
            transformMenu.Append(_rotateMenuItem);

            _scaleMenuItem = new RadioMenuItem(_translateMenuItem, "_Scale (R)");
            _scaleMenuItem.Toggled += (s, e) => {
                if (_scaleMenuItem.Active) _viewport.SetGizmoMode(GizmoMode.Scale);
            };
            transformMenu.Append(_scaleMenuItem);

            transformMenu.Append(new SeparatorMenuItem());

            var resetTransformItem = new MenuItem("Reset Transform");
            resetTransformItem.Activated += OnResetTransform;
            transformMenu.Append(resetTransformItem);

            editMenu.Append(transformMenuItem);

            // Mesh Operations submenu
            var meshOpsMenu = new Menu();
            var meshOpsMenuItem = new MenuItem("_Mesh Operations");
            meshOpsMenuItem.Submenu = meshOpsMenu;

            var decimateItem = new MenuItem("_Decimate (50%)");
            decimateItem.Activated += OnDecimateClicked;
            meshOpsMenu.Append(decimateItem);

            var smoothItem = new MenuItem("_Smooth");
            smoothItem.Activated += OnSmoothClicked;
            meshOpsMenu.Append(smoothItem);

            var optimizeItem = new MenuItem("_Optimize");
            optimizeItem.Activated += OnOptimizeClicked;
            meshOpsMenu.Append(optimizeItem);

            var splitItem = new MenuItem("Split by _Connectivity");
            splitItem.Activated += OnSplitClicked;
            meshOpsMenu.Append(splitItem);

            var flipNormalsItem = new MenuItem("_Flip Normals");
            flipNormalsItem.Activated += OnFlipNormals;
            meshOpsMenu.Append(flipNormalsItem);

            meshOpsMenu.Append(new SeparatorMenuItem());

            var mergeItem = new MenuItem("_Merge Selected");
            mergeItem.Activated += OnMergeClicked;
            meshOpsMenu.Append(mergeItem);

            var alignItem = new MenuItem("_Align (ICP)");
            alignItem.Activated += OnAlignClicked;
            meshOpsMenu.Append(alignItem);

            meshOpsMenu.Append(new SeparatorMenuItem());

            var scaleItem = new MenuItem("Set _Real Size...");
            scaleItem.Activated += OnSetRealSizeClicked;
            meshOpsMenu.Append(scaleItem);

            meshOpsMenu.Append(new SeparatorMenuItem());

            var cleanupItem = new MenuItem("_Cleanup Mesh...");
            cleanupItem.Activated += OnMeshCleanupClicked;
            meshOpsMenu.Append(cleanupItem);

            var bakeItem = new MenuItem("_Bake Textures...");
            bakeItem.Activated += OnBakeTexturesClicked;
            meshOpsMenu.Append(bakeItem);

            editMenu.Append(meshOpsMenuItem);

            // Triangle Editing submenu (Pen Tool operations)
            var triangleOpsMenu = new Menu();
            var triangleOpsMenuItem = new MenuItem("_Triangle Editing (Pen Tool)");
            triangleOpsMenuItem.Submenu = triangleOpsMenu;

            var deleteTrianglesItem = new MenuItem("_Delete Selected Triangles");
            deleteTrianglesItem.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.Delete, Gdk.ModifierType.None, AccelFlags.Visible);
            deleteTrianglesItem.Activated += OnDeleteSelectedTriangles;
            triangleOpsMenu.Append(deleteTrianglesItem);

            var flipTrianglesItem = new MenuItem("_Flip Selected Triangles");
            flipTrianglesItem.Activated += OnFlipSelectedTriangles;
            triangleOpsMenu.Append(flipTrianglesItem);

            var subdivideTrianglesItem = new MenuItem("_Subdivide Selected Triangles");
            subdivideTrianglesItem.Activated += OnSubdivideSelectedTriangles;
            triangleOpsMenu.Append(subdivideTrianglesItem);

            triangleOpsMenu.Append(new SeparatorMenuItem());

            var selectAllTrianglesItem = new MenuItem("Select _All Triangles");
            selectAllTrianglesItem.Activated += OnSelectAllTriangles;
            triangleOpsMenu.Append(selectAllTrianglesItem);

            var invertTriangleSelectionItem = new MenuItem("_Invert Selection");
            invertTriangleSelectionItem.Activated += OnInvertTriangleSelection;
            triangleOpsMenu.Append(invertTriangleSelectionItem);

            var growSelectionItem = new MenuItem("_Grow Selection");
            growSelectionItem.Activated += OnGrowTriangleSelection;
            triangleOpsMenu.Append(growSelectionItem);

            var clearTriangleSelectionItem = new MenuItem("_Clear Selection");
            clearTriangleSelectionItem.AddAccelerator("activate", accelGroup,
                (uint)Gdk.Key.Escape, Gdk.ModifierType.None, AccelFlags.Visible);
            clearTriangleSelectionItem.Activated += OnClearTriangleSelection;
            triangleOpsMenu.Append(clearTriangleSelectionItem);

            triangleOpsMenu.Append(new SeparatorMenuItem());

            var weldVerticesItem = new MenuItem("_Weld Vertices");
            weldVerticesItem.Activated += OnWeldSelectedVertices;
            triangleOpsMenu.Append(weldVerticesItem);

            editMenu.Append(triangleOpsMenuItem);

            menuBar.Append(editMenuItem);

            // View Menu
            var viewMenu = new Menu();
            var viewMenuItem = new MenuItem("_View");
            viewMenuItem.Submenu = viewMenu;

            // Display mode
            var meshModeItem = new RadioMenuItem("Show _Mesh");
            meshModeItem.Active = true;
            meshModeItem.Toggled += (s, e) => {
                if (meshModeItem.Active)
                {
                    IniSettings.Instance.ShowPointCloud = false;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(meshModeItem);

            var pointsModeItem = new RadioMenuItem(meshModeItem, "Show _Points");
            pointsModeItem.Toggled += (s, e) => {
                if (pointsModeItem.Active)
                {
                    IniSettings.Instance.ShowPointCloud = true;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(pointsModeItem);

            var wireframeItem = new CheckMenuItem("_Wireframe");
            wireframeItem.Active = IniSettings.Instance.ShowWireframe;
            wireframeItem.Toggled += (s, e) => {
                IniSettings.Instance.ShowWireframe = wireframeItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(wireframeItem);

            viewMenu.Append(new SeparatorMenuItem());

            // Color mode
            var rgbColorItem = new RadioMenuItem("_RGB Colors");
            rgbColorItem.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.RGB;
            rgbColorItem.Toggled += (s, e) => {
                if (rgbColorItem.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.RGB;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(rgbColorItem);

            var depthColorItem = new RadioMenuItem(rgbColorItem, "_Depth Colors");
            depthColorItem.Active = IniSettings.Instance.PointCloudColor == PointCloudColorMode.DistanceMap;
            depthColorItem.Toggled += (s, e) => {
                if (depthColorItem.Active)
                {
                    IniSettings.Instance.PointCloudColor = PointCloudColorMode.DistanceMap;
                    _viewport.QueueDraw();
                }
            };
            viewMenu.Append(depthColorItem);

            viewMenu.Append(new SeparatorMenuItem());

            // Viewport elements
            _showGridMenuItem = new CheckMenuItem("Show _Grid");
            _showGridMenuItem.Active = _viewport.ShowGrid;
            _showGridMenuItem.Toggled += (s, e) => {
                _viewport.ShowGrid = _showGridMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showGridMenuItem);

            _showAxesMenuItem = new CheckMenuItem("Show _Axes");
            _showAxesMenuItem.Active = _viewport.ShowAxes;
            _showAxesMenuItem.Toggled += (s, e) => {
                _viewport.ShowAxes = _showAxesMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showAxesMenuItem);

            _showCamerasMenuItem = new CheckMenuItem("Show _Cameras");
            _showCamerasMenuItem.Active = _viewport.ShowCameras;
            _showCamerasMenuItem.Toggled += (s, e) => {
                _viewport.ShowCameras = _showCamerasMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showCamerasMenuItem);

            _showInfoMenuItem = new CheckMenuItem("Show _Info Overlay");
            _showInfoMenuItem.Active = _viewport.ShowInfoText;
            _showInfoMenuItem.Toggled += (s, e) => {
                _viewport.ShowInfoText = _showInfoMenuItem.Active;
                _viewport.QueueDraw();
            };
            viewMenu.Append(_showInfoMenuItem);

            viewMenu.Append(new SeparatorMenuItem());

            var focusItem = new MenuItem("_Focus on Selection (F)");
            focusItem.Activated += (s, e) => _viewport.FocusOnSelection();
            viewMenu.Append(focusItem);

            var resetViewItem = new MenuItem("_Reset View");
            resetViewItem.Activated += (s, e) => _viewport.FocusOnSelection();
            viewMenu.Append(resetViewItem);

            menuBar.Append(viewMenuItem);

            // AI Models Menu
            var aiMenu = new Menu();
            var aiMenuItem = new MenuItem("_AI Models");
            aiMenuItem.Submenu = aiMenu;

            // Image to 3D submenu
            var imageTo3DMenu = new Menu();
            var imageTo3DMenuItem = new MenuItem("_Image to 3D");
            imageTo3DMenuItem.Submenu = imageTo3DMenu;

            var tripoSRItem = new MenuItem("_TripoSR (Fast)");
            tripoSRItem.Activated += OnTripoSRGenerate;
            imageTo3DMenu.Append(tripoSRItem);

            var lgmItem = new MenuItem("_LGM (High Quality)");
            lgmItem.Activated += OnLGMGenerate;
            imageTo3DMenu.Append(lgmItem);

            var wonder3DItem = new MenuItem("_Wonder3D (Multi-View)");
            wonder3DItem.Activated += OnWonder3DGenerate;
            imageTo3DMenu.Append(wonder3DItem);

            aiMenu.Append(imageTo3DMenuItem);

            // Mesh Processing submenu
            var meshProcessingMenu = new Menu();
            var meshProcessingMenuItem = new MenuItem("_Mesh Processing");
            meshProcessingMenuItem.Submenu = meshProcessingMenu;

            var deepMeshPriorItem = new MenuItem("_DeepMeshPrior Optimization");
            deepMeshPriorItem.Activated += OnDeepMeshPriorRefine;
            meshProcessingMenu.Append(deepMeshPriorItem);

            var tripoSFItem = new MenuItem("Tripo_SF Refinement");
            tripoSFItem.Activated += OnTripoSFRefine;
            meshProcessingMenu.Append(tripoSFItem);

            var gaussianSDFItem = new MenuItem("_GaussianSDF Refinement");
            gaussianSDFItem.Activated += OnGaussianSDFRefine;
            meshProcessingMenu.Append(gaussianSDFItem);

            meshProcessingMenu.Append(new SeparatorMenuItem());

            var aiDecimateItem = new MenuItem("_Decimate & Optimize");
            aiDecimateItem.Activated += (s, e) => OnDecimateClicked(null, EventArgs.Empty);
            meshProcessingMenu.Append(aiDecimateItem);

            aiMenu.Append(meshProcessingMenuItem);

            // Rigging submenu
            var riggingMenu = new Menu();
            var riggingMenuItem = new MenuItem("_Rigging");
            riggingMenuItem.Submenu = riggingMenu;

            var manualRigItem = new MenuItem("_Manual Rigging Tool");
            manualRigItem.Activated += OnShowRiggingPanel;
            riggingMenu.Append(manualRigItem);

            var createSkeletonItem = new MenuItem("_Create New Skeleton");
            createSkeletonItem.Activated += OnCreateNewSkeleton;
            riggingMenu.Append(createSkeletonItem);

            var humanoidTemplateItem = new MenuItem("_Humanoid Template");
            humanoidTemplateItem.Activated += OnCreateHumanoidSkeleton;
            riggingMenu.Append(humanoidTemplateItem);

            riggingMenu.Append(new SeparatorMenuItem());

            var uniRigItem = new MenuItem("_UniRig Auto Rig");
            uniRigItem.Activated += OnAutoRig;
            riggingMenu.Append(uniRigItem);

            riggingMenu.Append(new SeparatorMenuItem());

            var exportRiggedItem = new MenuItem("_Export Rigged Mesh...");
            exportRiggedItem.Activated += OnExportRiggedMesh;
            riggingMenu.Append(exportRiggedItem);

            aiMenu.Append(riggingMenuItem);

            aiMenu.Append(new SeparatorMenuItem());

            // Workflows submenu
            var workflowsMenu = new Menu();
            var workflowsMenuItem = new MenuItem("_Workflows");
            workflowsMenuItem.Submenu = workflowsMenu;

            var multiViewDmpItem = new MenuItem("_Multi-View → DeepMeshPrior (uses Settings engine)");
            multiViewDmpItem.Activated += OnMultiViewDeepMeshPriorWorkflow;
            workflowsMenu.Append(multiViewDmpItem);

            var multiViewNerfItem = new MenuItem("Multi-View → _NeRF → DeepMeshPrior (uses Settings engine)");
            multiViewNerfItem.Activated += OnMultiViewNeRFDeepMeshPriorWorkflow;
            workflowsMenu.Append(multiViewNerfItem);

            var fullPipelineItem = new MenuItem("_Full Pipeline (Mesh Only)");
            fullPipelineItem.Activated += OnFullPipelineWorkflow;
            workflowsMenu.Append(fullPipelineItem);

            workflowsMenu.Append(new SeparatorMenuItem());

            var sfmToAIItem = new MenuItem("_SfM → AI Refinement");
            sfmToAIItem.Activated += OnSfMToAIWorkflow;
            workflowsMenu.Append(sfmToAIItem);

            var pointCloudMergeItem = new MenuItem("_Point Cloud Merge & Refine");
            pointCloudMergeItem.Activated += OnPointCloudMergeWorkflow;
            workflowsMenu.Append(pointCloudMergeItem);

            aiMenu.Append(workflowsMenuItem);

            aiMenu.Append(new SeparatorMenuItem());

            var aiSettingsItem = new MenuItem("AI Model _Settings...");
            aiSettingsItem.Activated += OnAIModelSettings;
            aiMenu.Append(aiSettingsItem);

            menuBar.Append(aiMenuItem);

            // Window Menu
            var windowMenu = new Menu();
            var windowMenuItem = new MenuItem("_Window");
            windowMenuItem.Submenu = windowMenu;

            _showSceneTreeMenuItem = new CheckMenuItem("_Scene Tree");
            _showSceneTreeMenuItem.Active = true;
            _showSceneTreeMenuItem.Toggled += OnToggleSceneTree;
            windowMenu.Append(_showSceneTreeMenuItem);

            _showVerticalToolbarMenuItem = new CheckMenuItem("_Vertical Toolbar");
            _showVerticalToolbarMenuItem.Active = true;
            _showVerticalToolbarMenuItem.Toggled += OnToggleVerticalToolbar;
            windowMenu.Append(_showVerticalToolbarMenuItem);

            windowMenu.Append(new SeparatorMenuItem());

            var fullViewportItem = new MenuItem("_Full Viewport Mode");
            fullViewportItem.Activated += OnFullViewportMode;
            windowMenu.Append(fullViewportItem);

            var restorePanelsItem = new MenuItem("_Restore All Panels");
            restorePanelsItem.Activated += OnRestoreAllPanels;
            windowMenu.Append(restorePanelsItem);

            menuBar.Append(windowMenuItem);

            // Help Menu
            var helpMenu = new Menu();
            var helpMenuItem = new MenuItem("_Help");
            helpMenuItem.Submenu = helpMenu;

            var aiDiagItem = new MenuItem("AI _Diagnostic");
            aiDiagItem.Activated += (s, e) => new Deep3DStudio.UI.AIDiagnosticWindow().Show();
            helpMenu.Append(aiDiagItem);

            helpMenu.Append(new SeparatorMenuItem());

            var aboutItem = new MenuItem("_About");
            aboutItem.Activated += OnShowAbout;
            helpMenu.Append(aboutItem);

            menuBar.Append(helpMenuItem);

            return menuBar;
        }
    }
}
