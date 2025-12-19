using System;
using Gtk;
using Deep3DStudio.Configuration;

namespace Deep3DStudio
{
    /// <summary>
    /// Dialog for configuring AI model settings.
    /// Provides tabs for each AI model type with specific configuration options.
    /// </summary>
    public class AIModelSettingsDialog : Dialog
    {
        private readonly IniSettings _settings;

        // General
        private ComboBoxText _imageTo3DCombo;
        private ComboBoxText _reconstructionCombo;
        private ComboBoxText _meshingModelCombo;
        private ComboBoxText _meshRefinementCombo;
        private ComboBoxText _riggingCombo;

        // Compute Device (for all models)
        private ComboBoxText _computeDeviceCombo;

        // TripoSR
        private SpinButton _tripoSRResolution;
        private Entry _tripoSRModelPath;

        // LGM (Large Multi-View Gaussian Model)
        private SpinButton _lgmFlowSteps;
        private SpinButton _lgmResolution;
        private Entry _lgmModelPath;

        // Wonder3D
        private SpinButton _wonder3DSteps;
        private SpinButton _wonder3DGuidanceScale;
        private Entry _wonder3DModelPath;

        // TripoSF
        private SpinButton _tripoSFResolution;
        private Entry _tripoSFModelPath;

        // DeepMeshPrior
        private SpinButton _deepMeshPriorIterations;
        private SpinButton _deepMeshPriorLearningRate;
        private SpinButton _deepMeshPriorLaplacianWeight;

        // UniRig
        private SpinButton _uniRigMaxJoints;
        private Entry _uniRigModelPath;

        // Point Cloud Merger
        private SpinButton _mergerVoxelSize;
        private SpinButton _mergerMaxIterations;
        private SpinButton _mergerConvergenceThreshold;
        private SpinButton _mergerOutlierThreshold;

        public AIModelSettingsDialog(Window parent, IniSettings settings)
            : base("AI Model Settings", parent, DialogFlags.Modal | DialogFlags.DestroyWithParent)
        {
            _settings = settings;

            SetDefaultSize(600, 500);
            BorderWidth = 10;

            var notebook = new Notebook();
            notebook.TabPos = PositionType.Left;

            // Add tabs
            notebook.AppendPage(CreateGeneralTab(), new Label("General"));
            notebook.AppendPage(CreateTripoSRTab(), new Label("TripoSR"));
            notebook.AppendPage(CreateLGMTab(), new Label("LGM"));
            notebook.AppendPage(CreateWonder3DTab(), new Label("Wonder3D"));
            notebook.AppendPage(CreateTripoSFTab(), new Label("TripoSF"));
            notebook.AppendPage(CreateDeepMeshPriorTab(), new Label("DeepMeshPrior"));
            notebook.AppendPage(CreateUniRigTab(), new Label("UniRig"));
            notebook.AppendPage(CreateMergerTab(), new Label("Point Cloud"));

            ContentArea.PackStart(notebook, true, true, 0);

            // Buttons
            AddButton("Cancel", ResponseType.Cancel);
            AddButton("Apply", ResponseType.Ok);

            LoadSettings();
            ShowAll();
        }

        private Widget CreateGeneralTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            // Reconstruction model
            grid.Attach(new Label("Reconstruction Model:") { Halign = Align.Start }, 0, row, 1, 1);
            _reconstructionCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(ReconstructionMethod)))
                _reconstructionCombo.AppendText(name);
            grid.Attach(_reconstructionCombo, 1, row++, 1, 1);

            // Meshing / extraction model
            grid.Attach(new Label("Meshing Model:") { Halign = Align.Start }, 0, row, 1, 1);
            _meshingModelCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshingAlgorithm)))
                _meshingModelCombo.AppendText(name);
            grid.Attach(_meshingModelCombo, 1, row++, 1, 1);

            // Mesh refinement
            grid.Attach(new Label("Mesh Refinement:") { Halign = Align.Start }, 0, row, 1, 1);
            _meshRefinementCombo = new ComboBoxText();
            foreach (var name in Enum.GetNames(typeof(MeshRefinementMethod)))
                _meshRefinementCombo.AppendText(name);
            grid.Attach(_meshRefinementCombo, 1, row++, 1, 1);

            // Image to 3D model
            grid.Attach(new Label("Default Image→3D Model:") { Halign = Align.Start }, 0, row, 1, 1);
            _imageTo3DCombo = new ComboBoxText();
            _imageTo3DCombo.AppendText("None");
            _imageTo3DCombo.AppendText("TripoSR");
            _imageTo3DCombo.AppendText("LGM");
            _imageTo3DCombo.AppendText("Wonder3D");
            grid.Attach(_imageTo3DCombo, 1, row++, 1, 1);

            // Rigging method
            grid.Attach(new Label("Rigging Method:") { Halign = Align.Start }, 0, row, 1, 1);
            _riggingCombo = new ComboBoxText();
            _riggingCombo.AppendText("None");
            _riggingCombo.AppendText("UniRig");
            grid.Attach(_riggingCombo, 1, row++, 1, 1);

            row++;
            grid.Attach(new Separator(Orientation.Horizontal), 0, row++, 2, 1);

            // Compute device
            grid.Attach(new Label("AI Compute Device:") { Halign = Align.Start }, 0, row, 1, 1);
            _computeDeviceCombo = new ComboBoxText();
            _computeDeviceCombo.AppendText("CPU");
            _computeDeviceCombo.AppendText("CUDA (NVIDIA)");
            _computeDeviceCombo.AppendText("DirectML (AMD/Intel)");
            grid.Attach(_computeDeviceCombo, 1, row++, 1, 1);

            // Info text
            row++;
            var infoLabel = new Label(
                "Configure default AI models for each pipeline stage.\n\n" +
                "Image→3D: Converts single/multi images to 3D.\n" +
                "Mesh Extraction: Converts point clouds/SDFs to meshes.\n" +
                "Rigging: Auto-generates skeleton and weights.\n\n" +
                "Compute Device:\n" +
                "• CPU: Slowest but always available.\n" +
                "• CUDA: NVIDIA GPUs (fastest).\n" +
                "• DirectML: AMD/Intel GPUs on Windows.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(infoLabel, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateTripoSRTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            // Header
            var header = new Label("<b>TripoSR - Fast Single-Image 3D</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Resolution:") { Halign = Align.Start }, 0, row, 1, 1);
            _tripoSRResolution = new SpinButton(64, 512, 64);
            grid.Attach(_tripoSRResolution, 1, row++, 1, 1);

            grid.Attach(new Label("Model Path:") { Halign = Align.Start }, 0, row, 1, 1);
            var pathBox = new Box(Orientation.Horizontal, 5);
            _tripoSRModelPath = new Entry { WidthChars = 40 };
            var browseBtn = new Button("...");
            browseBtn.Clicked += (s, e) => BrowseFolder(_tripoSRModelPath);
            pathBox.PackStart(_tripoSRModelPath, true, true, 0);
            pathBox.PackStart(browseBtn, false, false, 0);
            grid.Attach(pathBox, 1, row++, 1, 1);

            // Info
            row++;
            var info = new Label(
                "TripoSR generates 3D models from a single image in ~0.5s.\n\n" +
                "Resolution: Higher values = more detail but slower.\n" +
                "Recommended: 256 for preview, 384-512 for final.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateLGMTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = new Label("<b>LGM - Large Multi-View Gaussian Model</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Flow Steps:") { Halign = Align.Start }, 0, row, 1, 1);
            _lgmFlowSteps = new SpinButton(10, 100, 5);
            grid.Attach(_lgmFlowSteps, 1, row++, 1, 1);

            grid.Attach(new Label("Resolution:") { Halign = Align.Start }, 0, row, 1, 1);
            _lgmResolution = new SpinButton(256, 1024, 128);
            grid.Attach(_lgmResolution, 1, row++, 1, 1);

            grid.Attach(new Label("Model Path:") { Halign = Align.Start }, 0, row, 1, 1);
            var pathBox = new Box(Orientation.Horizontal, 5);
            _lgmModelPath = new Entry { WidthChars = 40 };
            var browseBtn = new Button("...");
            browseBtn.Clicked += (s, e) => BrowseFolder(_lgmModelPath);
            pathBox.PackStart(_lgmModelPath, true, true, 0);
            pathBox.PackStart(browseBtn, false, false, 0);
            grid.Attach(pathBox, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "LGM uses Gaussian Splatting for high-quality\n" +
                "single-image 3D reconstruction.\n\n" +
                "Flow Steps: More steps = better quality, slower.\n" +
                "Recommended: 25-50 steps for good balance.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateWonder3DTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = new Label("<b>Wonder3D - Multi-View Generation</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Diffusion Steps:") { Halign = Align.Start }, 0, row, 1, 1);
            _wonder3DSteps = new SpinButton(10, 100, 5);
            grid.Attach(_wonder3DSteps, 1, row++, 1, 1);

            grid.Attach(new Label("Guidance Scale:") { Halign = Align.Start }, 0, row, 1, 1);
            _wonder3DGuidanceScale = new SpinButton(1.0, 20.0, 0.5);
            _wonder3DGuidanceScale.Digits = 1;
            grid.Attach(_wonder3DGuidanceScale, 1, row++, 1, 1);

            grid.Attach(new Label("Model Path:") { Halign = Align.Start }, 0, row, 1, 1);
            var pathBox = new Box(Orientation.Horizontal, 5);
            _wonder3DModelPath = new Entry { WidthChars = 40 };
            var browseBtn = new Button("...");
            browseBtn.Clicked += (s, e) => BrowseFolder(_wonder3DModelPath);
            pathBox.PackStart(_wonder3DModelPath, true, true, 0);
            pathBox.PackStart(browseBtn, false, false, 0);
            grid.Attach(pathBox, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "Wonder3D generates multi-view images using\n" +
                "cross-domain diffusion for consistent 3D.\n\n" +
                "Steps: 20-50 recommended.\n" +
                "Guidance Scale: Higher = more prompt adherence.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateTripoSFTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = new Label("<b>TripoSF - Ultra-High Resolution Mesh</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Resolution:") { Halign = Align.Start }, 0, row, 1, 1);
            _tripoSFResolution = new SpinButton(256, 1024, 128);
            grid.Attach(_tripoSFResolution, 1, row++, 1, 1);

            grid.Attach(new Label("Model Path:") { Halign = Align.Start }, 0, row, 1, 1);
            var pathBox = new Box(Orientation.Horizontal, 5);
            _tripoSFModelPath = new Entry { WidthChars = 40 };
            var browseBtn = new Button("...");
            browseBtn.Clicked += (s, e) => BrowseFolder(_tripoSFModelPath);
            pathBox.PackStart(_tripoSFModelPath, true, true, 0);
            pathBox.PackStart(browseBtn, false, false, 0);
            grid.Attach(pathBox, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "TripoSF uses SparseFlex for ultra-high resolution\n" +
                "mesh generation (up to 1024³).\n\n" +
                "Resolution: 256=4GB, 512=8GB, 1024=12GB VRAM.\n" +
                "Handles open surfaces (cloth, leaves) well.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateDeepMeshPriorTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = (new Label("<b>DeepMeshPrior - Mesh Denoising &amp; Optimization</b>") { UseMarkup = true, Halign = Align.Start });
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Iterations:") { Halign = Align.Start }, 0, row, 1, 1);
            _deepMeshPriorIterations = new SpinButton(100, 5000, 100);
            grid.Attach(_deepMeshPriorIterations, 1, row++, 1, 1);

            grid.Attach(new Label("Learning Rate:") { Halign = Align.Start }, 0, row, 1, 1);
            _deepMeshPriorLearningRate = new SpinButton(0.0001, 0.1, 0.001);
            _deepMeshPriorLearningRate.Digits = 4;
            grid.Attach(_deepMeshPriorLearningRate, 1, row++, 1, 1);

            grid.Attach(new Label("Laplacian Weight:") { Halign = Align.Start }, 0, row, 1, 1);
            _deepMeshPriorLaplacianWeight = new SpinButton(0.0, 10.0, 0.1);
            _deepMeshPriorLaplacianWeight.Digits = 2;
            grid.Attach(_deepMeshPriorLaplacianWeight, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "DeepMeshPrior uses a Graph Convolutional Network (GCN)\n" +
                "to optimize the mesh structure and remove noise.\n\n" +
                "Iterations: More = smoother but slower.\n" +
                "Laplacian Weight: Controls smoothness strength.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateUniRigTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = new Label("<b>UniRig - Automatic Rigging</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Max Joints:") { Halign = Align.Start }, 0, row, 1, 1);
            _uniRigMaxJoints = new SpinButton(10, 200, 10);
            grid.Attach(_uniRigMaxJoints, 1, row++, 1, 1);

            grid.Attach(new Label("Model Path:") { Halign = Align.Start }, 0, row, 1, 1);
            var pathBox = new Box(Orientation.Horizontal, 5);
            _uniRigModelPath = new Entry { WidthChars = 40 };
            var browseBtn = new Button("...");
            browseBtn.Clicked += (s, e) => BrowseFolder(_uniRigModelPath);
            pathBox.PackStart(_uniRigModelPath, true, true, 0);
            pathBox.PackStart(browseBtn, false, false, 0);
            grid.Attach(pathBox, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "UniRig auto-generates skeleton and skinning\n" +
                "weights for any 3D mesh.\n\n" +
                "Max Joints: Typical human=50-70, detailed=100+.\n" +
                "Works on arbitrary topology meshes.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private Widget CreateMergerTab()
        {
            var grid = new Grid
            {
                ColumnSpacing = 10,
                RowSpacing = 10,
                MarginStart = 10,
                MarginEnd = 10,
                MarginTop = 10
            };

            int row = 0;

            var header = new Label("<b>Point Cloud Merger Settings</b>") { UseMarkup = true, Halign = Align.Start };
            grid.Attach(header, 0, row++, 2, 1);

            grid.Attach(new Label("Voxel Size (m):") { Halign = Align.Start }, 0, row, 1, 1);
            _mergerVoxelSize = new SpinButton(0.001, 0.5, 0.005);
            _mergerVoxelSize.Digits = 3;
            grid.Attach(_mergerVoxelSize, 1, row++, 1, 1);

            grid.Attach(new Label("Max ICP Iterations:") { Halign = Align.Start }, 0, row, 1, 1);
            _mergerMaxIterations = new SpinButton(10, 200, 10);
            grid.Attach(_mergerMaxIterations, 1, row++, 1, 1);

            grid.Attach(new Label("Convergence Threshold:") { Halign = Align.Start }, 0, row, 1, 1);
            _mergerConvergenceThreshold = new SpinButton(1e-8, 1e-4, 1e-7);
            _mergerConvergenceThreshold.Digits = 8;
            grid.Attach(_mergerConvergenceThreshold, 1, row++, 1, 1);

            grid.Attach(new Label("Outlier Std Ratio:") { Halign = Align.Start }, 0, row, 1, 1);
            _mergerOutlierThreshold = new SpinButton(1.0, 5.0, 0.5);
            _mergerOutlierThreshold.Digits = 1;
            grid.Attach(_mergerOutlierThreshold, 1, row++, 1, 1);

            row++;
            var info = new Label(
                "Settings for merging multiple point clouds.\n\n" +
                "Voxel Size: Downsampling resolution.\n" +
                "ICP Iterations: More = better alignment, slower.\n" +
                "Outlier Ratio: Lower = more aggressive filtering.")
            {
                Halign = Align.Start,
                Wrap = true,
                MaxWidthChars = 50
            };
            grid.Attach(info, 0, row, 2, 1);

            return grid;
        }

        private void BrowseFolder(Entry entry)
        {
            var dialog = new FileChooserDialog(
                "Select Model Directory",
                this,
                FileChooserAction.SelectFolder,
                "Cancel", ResponseType.Cancel,
                "Select", ResponseType.Accept);

            if (dialog.Run() == (int)ResponseType.Accept)
            {
                entry.Text = dialog.Filename;
            }
            dialog.Destroy();
        }

        private void LoadSettings()
        {
            // General
            _imageTo3DCombo.Active = (int)_settings.ImageTo3D;
            _reconstructionCombo.Active = (int)_settings.ReconstructionMethod;
            _meshingModelCombo.Active = (int)_settings.MeshingAlgo;
            _meshRefinementCombo.Active = (int)_settings.MeshRefinement;
            _riggingCombo.Active = (int)_settings.RiggingModel;
            _computeDeviceCombo.Active = (int)_settings.AIDevice;

            // TripoSR
            _tripoSRResolution.Value = _settings.TripoSRResolution;
            _tripoSRModelPath.Text = _settings.TripoSRModelPath;

            // LGM
            _lgmFlowSteps.Value = _settings.LGMFlowSteps;
            _lgmResolution.Value = _settings.LGMResolution;
            _lgmModelPath.Text = _settings.LGMModelPath;

            // Wonder3D
            _wonder3DSteps.Value = _settings.Wonder3DSteps;
            _wonder3DGuidanceScale.Value = _settings.Wonder3DGuidanceScale;
            _wonder3DModelPath.Text = _settings.Wonder3DModelPath;

            // TripoSF
            _tripoSFResolution.Value = _settings.TripoSFResolution;
            _tripoSFModelPath.Text = _settings.TripoSFModelPath;

            // DeepMeshPrior
            _deepMeshPriorIterations.Value = _settings.DeepMeshPriorIterations;
            _deepMeshPriorLearningRate.Value = _settings.DeepMeshPriorLearningRate;
            _deepMeshPriorLaplacianWeight.Value = _settings.DeepMeshPriorLaplacianWeight;

            // UniRig
            _uniRigMaxJoints.Value = _settings.UniRigMaxJoints;
            _uniRigModelPath.Text = _settings.UniRigModelPath;

            // Merger
            _mergerVoxelSize.Value = _settings.MergerVoxelSize;
            _mergerMaxIterations.Value = _settings.MergerMaxIterations;
            _mergerConvergenceThreshold.Value = _settings.MergerConvergenceThreshold;
            _mergerOutlierThreshold.Value = _settings.MergerOutlierThreshold;
        }

        public void ApplySettings()
        {
            // General
            _settings.ImageTo3D = (ImageTo3DModel)_imageTo3DCombo.Active;
            _settings.ReconstructionMethod = (ReconstructionMethod)_reconstructionCombo.Active;
            _settings.MeshingAlgo = (MeshingAlgorithm)_meshingModelCombo.Active;
            _settings.MeshRefinement = (MeshRefinementMethod)_meshRefinementCombo.Active;
            _settings.MeshExtraction = _settings.MeshingAlgo switch
            {
                MeshingAlgorithm.DeepMeshPrior => MeshExtractionMethod.DeepMeshPrior,
                MeshingAlgorithm.TripoSF => MeshExtractionMethod.TripoSF,
                _ => MeshExtractionMethod.MarchingCubes
            };
            _settings.RiggingModel = (RiggingMethod)_riggingCombo.Active;
            _settings.AIDevice = (AIComputeDevice)_computeDeviceCombo.Active;

            // TripoSR
            _settings.TripoSRResolution = (int)_tripoSRResolution.Value;
            _settings.TripoSRModelPath = _tripoSRModelPath.Text;

            // LGM
            _settings.LGMFlowSteps = (int)_lgmFlowSteps.Value;
            _settings.LGMResolution = (int)_lgmResolution.Value;
            _settings.LGMModelPath = _lgmModelPath.Text;

            // Wonder3D
            _settings.Wonder3DSteps = (int)_wonder3DSteps.Value;
            _settings.Wonder3DGuidanceScale = (float)_wonder3DGuidanceScale.Value;
            _settings.Wonder3DModelPath = _wonder3DModelPath.Text;

            // TripoSF
            _settings.TripoSFResolution = (int)_tripoSFResolution.Value;
            _settings.TripoSFModelPath = _tripoSFModelPath.Text;

            // DeepMeshPrior
            _settings.DeepMeshPriorIterations = (int)_deepMeshPriorIterations.Value;
            _settings.DeepMeshPriorLearningRate = (float)_deepMeshPriorLearningRate.Value;
            _settings.DeepMeshPriorLaplacianWeight = (float)_deepMeshPriorLaplacianWeight.Value;

            // UniRig
            _settings.UniRigMaxJoints = (int)_uniRigMaxJoints.Value;
            _settings.UniRigModelPath = _uniRigModelPath.Text;

            // Merger
            _settings.MergerVoxelSize = (float)_mergerVoxelSize.Value;
            _settings.MergerMaxIterations = (int)_mergerMaxIterations.Value;
            _settings.MergerConvergenceThreshold = (float)_mergerConvergenceThreshold.Value;
            _settings.MergerOutlierThreshold = (float)_mergerOutlierThreshold.Value;
        }
    }
}
