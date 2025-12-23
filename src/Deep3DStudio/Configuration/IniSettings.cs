using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;

namespace Deep3DStudio.Configuration
{
    public enum ReconstructionMethod
    {
        Dust3r,
        FeatureMatching,
        TripoSR,
        Wonder3D
    }

    public enum RiggingMethod
    {
        None,
        UniRig
    }

    public enum MeshExtractionMethod
    {
        MarchingCubes,
        DeepMeshPrior,
        TripoSF
    }

    public enum MeshRefinementMethod
    {
        None,
        DeepMeshPrior,
        TripoSF,
        GaussianSDF
    }

    public enum ImageTo3DModel
    {
        None,
        TripoSR,
        LGM,
        Wonder3D
    }

    public enum AIComputeDevice
    {
        CPU,
        CUDA,
        DirectML  // For AMD/Intel GPUs on Windows
    }

    /// <summary>
    /// INI file based settings manager with platform-specific storage locations.
    /// - Windows: %APPDATA%/Deep3DStudio/settings.ini
    /// - macOS: ~/Library/Application Support/Deep3DStudio/settings.ini
    /// - Linux: ~/.config/Deep3DStudio/settings.ini
    /// </summary>
    public class IniSettings
    {
        private static IniSettings? _instance;
        private readonly string _settingsPath;
        private readonly Dictionary<string, Dictionary<string, string>> _sections;

        // Settings values with defaults
        public ComputeDevice Device { get; set; } = ComputeDevice.CPU;
        public MeshingAlgorithm MeshingAlgo { get; set; } = MeshingAlgorithm.MarchingCubes;
        public CoordinateSystem CoordSystem { get; set; } = CoordinateSystem.RightHanded_Y_Up;
        public BoundingBoxMode BoundingBoxStyle { get; set; } = BoundingBoxMode.Full;
        public ReconstructionMethod ReconstructionMethod { get; set; } = ReconstructionMethod.Dust3r;

        // Rendering Settings
        public bool ShowMesh { get; set; } = true;
        public bool ShowPointCloud { get; set; } = false;
        public bool ShowWireframe { get; set; } = false;
        public bool ShowTexture { get; set; } = true;
        public PointCloudColorMode PointCloudColor { get; set; } = PointCloudColorMode.RGB;

        // Viewport Colors (RGB floats 0-1)
        public float ViewportBgR { get; set; } = 0.12f;
        public float ViewportBgG { get; set; } = 0.12f;
        public float ViewportBgB { get; set; } = 0.14f;

        public float GridColorR { get; set; } = 0.35f;
        public float GridColorG { get; set; } = 0.35f;
        public float GridColorB { get; set; } = 0.35f;

        // NeRF Workflow Settings
        public int NeRFIterations { get; set; } = 50;
        public int VoxelGridSize { get; set; } = 128;
        public float NeRFLearningRate { get; set; } = 0.1f;

        // AI Model Settings
        public ImageTo3DModel ImageTo3D { get; set; } = ImageTo3DModel.None;
        public RiggingMethod RiggingModel { get; set; } = RiggingMethod.None;
        public MeshExtractionMethod MeshExtraction { get; set; } = MeshExtractionMethod.MarchingCubes;
        public MeshRefinementMethod MeshRefinement { get; set; } = MeshRefinementMethod.None;

        // TripoSR Settings
        public int TripoSRResolution { get; set; } = 256;
        public int TripoSRMarchingCubesRes { get; set; } = 128;

        // LGM Settings
        public int LGMFlowSteps { get; set; } = 25;
        public int LGMQueryResolution { get; set; } = 128;

        // Wonder3D Settings
        public int Wonder3DDiffusionSteps { get; set; } = 50;
        public float Wonder3DCFGScale { get; set; } = 3.0f;

        // UniRig Settings
        public int UniRigMaxJoints { get; set; } = 64;
        public int UniRigMaxBonesPerVertex { get; set; } = 4;

        // DeepMeshPrior Settings
        public int DeepMeshPriorIterations { get; set; } = 500;
        public float DeepMeshPriorLearningRate { get; set; } = 0.01f;
        public float DeepMeshPriorLaplacianWeight { get; set; } = 1.4f;

        // Gaussian SDF Refiner Settings
        public int GaussianSDFGridResolution { get; set; } = 128;
        public float GaussianSDFSigma { get; set; } = 1.0f;
        public int GaussianSDFIterations { get; set; } = 1;
        public float GaussianSDFIsoLevel { get; set; } = 0.0f;

        // TripoSF Settings
        public int TripoSFResolution { get; set; } = 512;
        public int TripoSFSparseDilation { get; set; } = 1;

        // AI Model Paths (relative to app directory or absolute)
        public string TripoSRModelPath { get; set; } = "models/triposr";
        public string LGMModelPath { get; set; } = "models/lgm";
        public string Wonder3DModelPath { get; set; } = "models/wonder3d";
        public string UniRigModelPath { get; set; } = "models/unirig";
        public string TripoSFModelPath { get; set; } = "models/triposf";

        // Additional AI Model Settings
        public AIComputeDevice AIDevice { get; set; } = AIComputeDevice.CUDA;
        public bool UseCudaForAI { get => AIDevice == AIComputeDevice.CUDA; set => AIDevice = value ? AIComputeDevice.CUDA : AIComputeDevice.CPU; }
        public int LGMResolution { get; set; } = 512;
        public int Wonder3DSteps { get; set; } = 50;
        public float Wonder3DGuidanceScale { get; set; } = 3.0f;

        // Point Cloud Merger Settings
        public float MergerVoxelSize { get; set; } = 0.02f;
        public int MergerMaxIterations { get; set; } = 50;
        public float MergerConvergenceThreshold { get; set; } = 1e-6f;
        public float MergerOutlierThreshold { get; set; } = 2.0f;

        // UI Settings
        public bool ShowGrid { get; set; } = true;
        public bool ShowAxes { get; set; } = true;
        public bool ShowCameras { get; set; } = true;
        public bool ShowGizmo { get; set; } = true;
        public bool ShowInfoOverlay { get; set; } = true;
        public bool ShowPointCloudBounds { get; set; } = true; // Show bounding box for point clouds
        public int LastWindowWidth { get; set; } = 1400;
        public int LastWindowHeight { get; set; } = 900;
        public int LastPanelWidth { get; set; } = 250;

        private IniSettings()
        {
            _settingsPath = GetSettingsPath();
            _sections = new Dictionary<string, Dictionary<string, string>>(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>
        /// Gets the singleton instance, loading from file if needed.
        /// </summary>
        public static IniSettings Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new IniSettings();
                    _instance.Load();
                }
                return _instance;
            }
        }

        /// <summary>
        /// Gets the platform-specific settings directory path.
        /// </summary>
        public static string GetSettingsDirectory()
        {
            string basePath;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Windows: %APPDATA%/Deep3DStudio
                basePath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // macOS: ~/Library/Application Support/Deep3DStudio
                basePath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                    "Library", "Application Support"
                );
            }
            else
            {
                // Linux/Other: ~/.config/Deep3DStudio
                string xdgConfig = Environment.GetEnvironmentVariable("XDG_CONFIG_HOME") ?? "";
                if (string.IsNullOrEmpty(xdgConfig))
                {
                    xdgConfig = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".config");
                }
                basePath = xdgConfig;
            }

            return Path.Combine(basePath, "Deep3DStudio");
        }

        /// <summary>
        /// Gets the full path to the settings.ini file.
        /// </summary>
        public static string GetSettingsPath()
        {
            return Path.Combine(GetSettingsDirectory(), "settings.ini");
        }

        /// <summary>
        /// Loads settings from the INI file.
        /// </summary>
        public void Load()
        {
            _sections.Clear();

            if (!File.Exists(_settingsPath))
            {
                Console.WriteLine($"Settings file not found at {_settingsPath}, using defaults.");
                return;
            }

            try
            {
                string currentSection = "General";
                _sections[currentSection] = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

                foreach (var line in File.ReadAllLines(_settingsPath))
                {
                    string trimmed = line.Trim();

                    // Skip empty lines and comments
                    if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith(";") || trimmed.StartsWith("#"))
                        continue;

                    // Section header
                    if (trimmed.StartsWith("[") && trimmed.EndsWith("]"))
                    {
                        currentSection = trimmed.Substring(1, trimmed.Length - 2).Trim();
                        if (!_sections.ContainsKey(currentSection))
                            _sections[currentSection] = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                        continue;
                    }

                    // Key=Value pair
                    int eqIndex = trimmed.IndexOf('=');
                    if (eqIndex > 0)
                    {
                        string key = trimmed.Substring(0, eqIndex).Trim();
                        string value = trimmed.Substring(eqIndex + 1).Trim();
                        _sections[currentSection][key] = value;
                    }
                }

                // Apply loaded values to properties
                ApplyLoadedSettings();
                Console.WriteLine($"Settings loaded from {_settingsPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading settings: {ex.Message}");
            }
        }

        /// <summary>
        /// Saves current settings to the INI file.
        /// </summary>
        public void Save()
        {
            try
            {
                string dir = Path.GetDirectoryName(_settingsPath)!;
                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                using (var writer = new StreamWriter(_settingsPath))
                {
                    writer.WriteLine("; Deep3D Studio Settings");
                    writer.WriteLine("; Auto-generated - modifications will be preserved");
                    writer.WriteLine($"; Last saved: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    writer.WriteLine();

                    // [General] section
                    writer.WriteLine("[General]");
                    writer.WriteLine($"Device={Device}");
                    writer.WriteLine($"MeshingAlgorithm={MeshingAlgo}");
                    writer.WriteLine($"CoordinateSystem={CoordSystem}");
                    writer.WriteLine($"BoundingBoxStyle={BoundingBoxStyle}");
                    writer.WriteLine($"ReconstructionMethod={ReconstructionMethod}");
                    writer.WriteLine();

                    // [Rendering] section
                    writer.WriteLine("[Rendering]");
                    writer.WriteLine($"ShowMesh={ShowMesh}");
                    writer.WriteLine($"ShowPointCloud={ShowPointCloud}");
                    writer.WriteLine($"ShowWireframe={ShowWireframe}");
                    writer.WriteLine($"ShowTexture={ShowTexture}");
                    writer.WriteLine($"PointCloudColorMode={PointCloudColor}");
                    writer.WriteLine();

                    // [Viewport] section
                    writer.WriteLine("[Viewport]");
                    writer.WriteLine($"BackgroundR={ViewportBgR.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"BackgroundG={ViewportBgG.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"BackgroundB={ViewportBgB.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"GridColorR={GridColorR.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"GridColorG={GridColorG.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"GridColorB={GridColorB.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"ShowGrid={ShowGrid}");
                    writer.WriteLine($"ShowAxes={ShowAxes}");
                    writer.WriteLine($"ShowGizmo={ShowGizmo}");
                    writer.WriteLine($"ShowCameras={ShowCameras}");
                    writer.WriteLine($"ShowInfoOverlay={ShowInfoOverlay}");
                    writer.WriteLine($"ShowPointCloudBounds={ShowPointCloudBounds}");
                    writer.WriteLine();

                    // [NeRF] section
                    writer.WriteLine("[NeRF]");
                    writer.WriteLine($"Iterations={NeRFIterations}");
                    writer.WriteLine($"VoxelGridSize={VoxelGridSize}");
                    writer.WriteLine($"LearningRate={NeRFLearningRate.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine();

                    // [AIModels] section
                    writer.WriteLine("[AIModels]");
                    writer.WriteLine($"ImageTo3D={ImageTo3D}");
                    writer.WriteLine($"RiggingModel={RiggingModel}");
                    writer.WriteLine($"MeshExtraction={MeshExtraction}");
                    writer.WriteLine($"MeshRefinement={MeshRefinement}");
                    writer.WriteLine($"ComputeDevice={AIDevice}");
                    writer.WriteLine();

                    // [TripoSR] section
                    writer.WriteLine("[TripoSR]");
                    writer.WriteLine($"Resolution={TripoSRResolution}");
                    writer.WriteLine($"MarchingCubesRes={TripoSRMarchingCubesRes}");
                    writer.WriteLine($"ModelPath={TripoSRModelPath}");
                    writer.WriteLine();

                    // [LGM] section
                    writer.WriteLine("[LGM]");
                    writer.WriteLine($"FlowSteps={LGMFlowSteps}");
                    writer.WriteLine($"QueryResolution={LGMQueryResolution}");
                    writer.WriteLine($"Resolution={LGMResolution}");
                    writer.WriteLine($"ModelPath={LGMModelPath}");
                    writer.WriteLine();

                    // [Wonder3D] section
                    writer.WriteLine("[Wonder3D]");
                    writer.WriteLine($"DiffusionSteps={Wonder3DDiffusionSteps}");
                    writer.WriteLine($"Steps={Wonder3DSteps}");
                    writer.WriteLine($"CFGScale={Wonder3DCFGScale.ToString("F2", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"GuidanceScale={Wonder3DGuidanceScale.ToString("F2", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"ModelPath={Wonder3DModelPath}");
                    writer.WriteLine();

                    // [UniRig] section
                    writer.WriteLine("[UniRig]");
                    writer.WriteLine($"MaxJoints={UniRigMaxJoints}");
                    writer.WriteLine($"MaxBonesPerVertex={UniRigMaxBonesPerVertex}");
                    writer.WriteLine($"ModelPath={UniRigModelPath}");
                    writer.WriteLine();

                    // [DeepMeshPrior] section
                    writer.WriteLine("[DeepMeshPrior]");
                    writer.WriteLine($"Iterations={DeepMeshPriorIterations}");
                    writer.WriteLine($"LearningRate={DeepMeshPriorLearningRate.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"LaplacianWeight={DeepMeshPriorLaplacianWeight.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine();

                    // [GaussianSDF] section
                    writer.WriteLine("[GaussianSDF]");
                    writer.WriteLine($"GridResolution={GaussianSDFGridResolution}");
                    writer.WriteLine($"Sigma={GaussianSDFSigma.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"Iterations={GaussianSDFIterations}");
                    writer.WriteLine($"IsoLevel={GaussianSDFIsoLevel.ToString("F3", CultureInfo.InvariantCulture)}");
                    writer.WriteLine();

                    // [TripoSF] section
                    writer.WriteLine("[TripoSF]");
                    writer.WriteLine($"Resolution={TripoSFResolution}");
                    writer.WriteLine($"SparseDilation={TripoSFSparseDilation}");
                    writer.WriteLine($"ModelPath={TripoSFModelPath}");
                    writer.WriteLine();

                    // [PointCloudMerger] section
                    writer.WriteLine("[PointCloudMerger]");
                    writer.WriteLine($"VoxelSize={MergerVoxelSize.ToString("F4", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"MaxIterations={MergerMaxIterations}");
                    writer.WriteLine($"ConvergenceThreshold={MergerConvergenceThreshold.ToString("E2", CultureInfo.InvariantCulture)}");
                    writer.WriteLine($"OutlierThreshold={MergerOutlierThreshold.ToString("F2", CultureInfo.InvariantCulture)}");
                    writer.WriteLine();

                    // [Window] section
                    writer.WriteLine("[Window]");
                    writer.WriteLine($"Width={LastWindowWidth}");
                    writer.WriteLine($"Height={LastWindowHeight}");
                    writer.WriteLine($"PanelWidth={LastPanelWidth}");
                }

                Console.WriteLine($"Settings saved to {_settingsPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving settings: {ex.Message}");
            }
        }

        /// <summary>
        /// Applies loaded INI values to the property fields.
        /// </summary>
        private void ApplyLoadedSettings()
        {
            // [General]
            if (TryGetValue("General", "Device", out string? deviceStr) && Enum.TryParse<ComputeDevice>(deviceStr, out var device))
                Device = device;
            if (TryGetValue("General", "MeshingAlgorithm", out string? meshStr) && Enum.TryParse<MeshingAlgorithm>(meshStr, out var mesh))
                MeshingAlgo = mesh;
            if (TryGetValue("General", "CoordinateSystem", out string? coordStr) && Enum.TryParse<CoordinateSystem>(coordStr, out var coord))
                CoordSystem = coord;
            if (TryGetValue("General", "BoundingBoxStyle", out string? bboxStr) && Enum.TryParse<BoundingBoxMode>(bboxStr, out var bbox))
                BoundingBoxStyle = bbox;
            if (TryGetValue("General", "ReconstructionMethod", out string? reconStr) && Enum.TryParse<ReconstructionMethod>(reconStr, out var recon))
                ReconstructionMethod = recon;

            // [Rendering]
            if (TryGetValue("Rendering", "ShowMesh", out string? smStr) && bool.TryParse(smStr, out var sm))
                ShowMesh = sm;
            if (TryGetValue("Rendering", "ShowPointCloud", out string? spcStr) && bool.TryParse(spcStr, out var spc))
                ShowPointCloud = spc;
            if (TryGetValue("Rendering", "ShowWireframe", out string? swStr) && bool.TryParse(swStr, out var sw))
                ShowWireframe = sw;
            if (TryGetValue("Rendering", "ShowTexture", out string? stStr) && bool.TryParse(stStr, out var st))
                ShowTexture = st;
            if (TryGetValue("Rendering", "PointCloudColorMode", out string? pcmStr) && Enum.TryParse<PointCloudColorMode>(pcmStr, out var pcm))
                PointCloudColor = pcm;

            // [Viewport]
            if (TryGetValue("Viewport", "BackgroundR", out string? bgRStr) && float.TryParse(bgRStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var bgR))
                ViewportBgR = bgR;
            if (TryGetValue("Viewport", "BackgroundG", out string? bgGStr) && float.TryParse(bgGStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var bgG))
                ViewportBgG = bgG;
            if (TryGetValue("Viewport", "BackgroundB", out string? bgBStr) && float.TryParse(bgBStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var bgB))
                ViewportBgB = bgB;
            if (TryGetValue("Viewport", "GridColorR", out string? gcRStr) && float.TryParse(gcRStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var gcR))
                GridColorR = gcR;
            if (TryGetValue("Viewport", "GridColorG", out string? gcGStr) && float.TryParse(gcGStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var gcG))
                GridColorG = gcG;
            if (TryGetValue("Viewport", "GridColorB", out string? gcBStr) && float.TryParse(gcBStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var gcB))
                GridColorB = gcB;
            if (TryGetValue("Viewport", "ShowGrid", out string? sgStr) && bool.TryParse(sgStr, out var sg))
                ShowGrid = sg;
            if (TryGetValue("Viewport", "ShowAxes", out string? saStr) && bool.TryParse(saStr, out var sa))
                ShowAxes = sa;
            if (TryGetValue("Viewport", "ShowCameras", out string? scStr) && bool.TryParse(scStr, out var sc))
                ShowCameras = sc;
            if (TryGetValue("Viewport", "ShowGizmo", out string? gizStr) && bool.TryParse(gizStr, out var giz))
                ShowGizmo = giz;
            if (TryGetValue("Viewport", "ShowInfoOverlay", out string? sioStr) && bool.TryParse(sioStr, out var sio))
                ShowInfoOverlay = sio;
            if (TryGetValue("Viewport", "ShowPointCloudBounds", out string? spcbStr) && bool.TryParse(spcbStr, out var spcb))
                ShowPointCloudBounds = spcb;

            // [NeRF]
            if (TryGetValue("NeRF", "Iterations", out string? itStr) && int.TryParse(itStr, out var it))
                NeRFIterations = Math.Clamp(it, 1, 500);
            if (TryGetValue("NeRF", "VoxelGridSize", out string? vgsStr) && int.TryParse(vgsStr, out var vgs))
                VoxelGridSize = Math.Clamp(vgs, 32, 512);
            if (TryGetValue("NeRF", "LearningRate", out string? lrStr) && float.TryParse(lrStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var lr))
                NeRFLearningRate = Math.Clamp(lr, 0.001f, 1.0f);

            // [Window]
            if (TryGetValue("Window", "Width", out string? wStr) && int.TryParse(wStr, out var w))
                LastWindowWidth = Math.Clamp(w, 800, 4000);
            if (TryGetValue("Window", "Height", out string? hStr) && int.TryParse(hStr, out var h))
                LastWindowHeight = Math.Clamp(h, 600, 3000);
            if (TryGetValue("Window", "PanelWidth", out string? pwStr) && int.TryParse(pwStr, out var pw))
                LastPanelWidth = Math.Clamp(pw, 100, 600);

            // [AIModels]
            if (TryGetValue("AIModels", "ImageTo3D", out string? img3dStr) && Enum.TryParse<ImageTo3DModel>(img3dStr, out var img3d))
                ImageTo3D = img3d;
            if (TryGetValue("AIModels", "RiggingModel", out string? rigStr) && Enum.TryParse<RiggingMethod>(rigStr, out var rig))
                RiggingModel = rig;
            if (TryGetValue("AIModels", "MeshExtraction", out string? meshExStr) && Enum.TryParse<MeshExtractionMethod>(meshExStr, out var meshEx))
                MeshExtraction = meshEx;
            if (TryGetValue("AIModels", "MeshRefinement", out string? meshRefineStr) && Enum.TryParse<MeshRefinementMethod>(meshRefineStr, out var meshRefine))
                MeshRefinement = meshRefine;

            // [TripoSR]
            if (TryGetValue("TripoSR", "Resolution", out string? tsrResStr) && int.TryParse(tsrResStr, out var tsrRes))
                TripoSRResolution = Math.Clamp(tsrRes, 128, 512);
            if (TryGetValue("TripoSR", "MarchingCubesRes", out string? tsrMcStr) && int.TryParse(tsrMcStr, out var tsrMc))
                TripoSRMarchingCubesRes = Math.Clamp(tsrMc, 64, 256);
            if (TryGetValue("TripoSR", "ModelPath", out string? tsrPath))
                TripoSRModelPath = tsrPath ?? TripoSRModelPath;

            // [LGM]
            if (TryGetValue("LGM", "FlowSteps", out string? lgmFlowStr) && int.TryParse(lgmFlowStr, out var lgmFlow))
                LGMFlowSteps = Math.Clamp(lgmFlow, 10, 100);
            if (TryGetValue("LGM", "QueryResolution", out string? lgmQryStr) && int.TryParse(lgmQryStr, out var lgmQry))
                LGMQueryResolution = Math.Clamp(lgmQry, 64, 256);
            if (TryGetValue("LGM", "ModelPath", out string? lgmPath))
                LGMModelPath = lgmPath ?? LGMModelPath;

            // [Wonder3D]
            if (TryGetValue("Wonder3D", "DiffusionSteps", out string? w3dStepsStr) && int.TryParse(w3dStepsStr, out var w3dSteps))
                Wonder3DDiffusionSteps = Math.Clamp(w3dSteps, 20, 100);
            if (TryGetValue("Wonder3D", "CFGScale", out string? w3dCfgStr) && float.TryParse(w3dCfgStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var w3dCfg))
                Wonder3DCFGScale = Math.Clamp(w3dCfg, 1.0f, 10.0f);
            if (TryGetValue("Wonder3D", "ModelPath", out string? w3dPath))
                Wonder3DModelPath = w3dPath ?? Wonder3DModelPath;

            // [UniRig]
            if (TryGetValue("UniRig", "MaxJoints", out string? urJointsStr) && int.TryParse(urJointsStr, out var urJoints))
                UniRigMaxJoints = Math.Clamp(urJoints, 16, 256);
            if (TryGetValue("UniRig", "MaxBonesPerVertex", out string? urBpvStr) && int.TryParse(urBpvStr, out var urBpv))
                UniRigMaxBonesPerVertex = Math.Clamp(urBpv, 1, 8);
            if (TryGetValue("UniRig", "ModelPath", out string? urPath))
                UniRigModelPath = urPath ?? UniRigModelPath;

            // [DeepMeshPrior]
            if (TryGetValue("DeepMeshPrior", "Iterations", out string? dmpIterStr) && int.TryParse(dmpIterStr, out var dmpIter))
                DeepMeshPriorIterations = Math.Clamp(dmpIter, 100, 5000);
            if (TryGetValue("DeepMeshPrior", "LearningRate", out string? dmpLrStr) && float.TryParse(dmpLrStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var dmpLr))
                DeepMeshPriorLearningRate = Math.Clamp(dmpLr, 0.0001f, 0.1f);
            if (TryGetValue("DeepMeshPrior", "LaplacianWeight", out string? dmpLapStr) && float.TryParse(dmpLapStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var dmpLap))
                DeepMeshPriorLaplacianWeight = Math.Clamp(dmpLap, 0.0f, 10.0f);

            // [GaussianSDF]
            if (TryGetValue("GaussianSDF", "GridResolution", out string? gsGridStr) && int.TryParse(gsGridStr, out var gsGrid))
                GaussianSDFGridResolution = Math.Clamp(gsGrid, 32, 512);
            if (TryGetValue("GaussianSDF", "Sigma", out string? gsSigmaStr) && float.TryParse(gsSigmaStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var gsSigma))
                GaussianSDFSigma = Math.Clamp(gsSigma, 0.1f, 10.0f);
            if (TryGetValue("GaussianSDF", "Iterations", out string? gsIterStr) && int.TryParse(gsIterStr, out var gsIter))
                GaussianSDFIterations = Math.Clamp(gsIter, 0, 10);
            if (TryGetValue("GaussianSDF", "IsoLevel", out string? gsIsoStr) && float.TryParse(gsIsoStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var gsIso))
                GaussianSDFIsoLevel = Math.Clamp(gsIso, -1.0f, 1.0f);

            // [TripoSF]
            if (TryGetValue("TripoSF", "Resolution", out string? tsfResStr) && int.TryParse(tsfResStr, out var tsfRes))
                TripoSFResolution = Math.Clamp(tsfRes, 256, 1024);
            if (TryGetValue("TripoSF", "SparseDilation", out string? tsfDilStr) && int.TryParse(tsfDilStr, out var tsfDil))
                TripoSFSparseDilation = Math.Clamp(tsfDil, 0, 3);
            if (TryGetValue("TripoSF", "ModelPath", out string? tsfPath))
                TripoSFModelPath = tsfPath ?? TripoSFModelPath;

            // [AIModels] additional settings
            if (TryGetValue("AIModels", "ComputeDevice", out string? aiDevStr) && Enum.TryParse<AIComputeDevice>(aiDevStr, out var aiDev))
                AIDevice = aiDev;
            else if (TryGetValue("AIModels", "UseCuda", out string? useCudaStr) && bool.TryParse(useCudaStr, out var useCuda))
                AIDevice = useCuda ? AIComputeDevice.CUDA : AIComputeDevice.CPU;

            // [LGM] additional settings
            if (TryGetValue("LGM", "Resolution", out string? lgmResStr) && int.TryParse(lgmResStr, out var lgmRes))
                LGMResolution = Math.Clamp(lgmRes, 256, 1024);

            // [Wonder3D] additional settings
            if (TryGetValue("Wonder3D", "Steps", out string? w3dSteps2Str) && int.TryParse(w3dSteps2Str, out var w3dSteps2))
                Wonder3DSteps = Math.Clamp(w3dSteps2, 10, 100);
            if (TryGetValue("Wonder3D", "GuidanceScale", out string? w3dGsStr) && float.TryParse(w3dGsStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var w3dGs))
                Wonder3DGuidanceScale = Math.Clamp(w3dGs, 1.0f, 20.0f);

            // [PointCloudMerger]
            if (TryGetValue("PointCloudMerger", "VoxelSize", out string? pcmVoxStr) && float.TryParse(pcmVoxStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var pcmVox))
                MergerVoxelSize = Math.Clamp(pcmVox, 0.001f, 0.5f);
            if (TryGetValue("PointCloudMerger", "MaxIterations", out string? pcmIterStr) && int.TryParse(pcmIterStr, out var pcmIter))
                MergerMaxIterations = Math.Clamp(pcmIter, 10, 200);
            if (TryGetValue("PointCloudMerger", "ConvergenceThreshold", out string? pcmConvStr) && float.TryParse(pcmConvStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var pcmConv))
                MergerConvergenceThreshold = Math.Clamp(pcmConv, 1e-8f, 1e-4f);
            if (TryGetValue("PointCloudMerger", "OutlierThreshold", out string? pcmOutStr) && float.TryParse(pcmOutStr, NumberStyles.Float, CultureInfo.InvariantCulture, out var pcmOut))
                MergerOutlierThreshold = Math.Clamp(pcmOut, 1.0f, 5.0f);
        }

        /// <summary>
        /// Tries to get a value from the loaded INI data.
        /// </summary>
        private bool TryGetValue(string section, string key, out string? value)
        {
            value = null;
            if (_sections.TryGetValue(section, out var sectionDict))
            {
                if (sectionDict.TryGetValue(key, out value))
                    return true;
            }
            return false;
        }

        /// <summary>
        /// Reloads settings from disk.
        /// </summary>
        public void Reload()
        {
            Load();
        }

        public void Reset()
        {
            ResetToDefaults();
        }

        /// <summary>
        /// Resets all settings to their default values.
        /// </summary>
        public void ResetToDefaults()
        {
            Device = ComputeDevice.CPU;
            MeshingAlgo = MeshingAlgorithm.MarchingCubes;
            CoordSystem = CoordinateSystem.RightHanded_Y_Up;
            BoundingBoxStyle = BoundingBoxMode.Full;
            ReconstructionMethod = ReconstructionMethod.Dust3r;
            ShowMesh = true;
            ShowPointCloud = false;
            ShowWireframe = false;
            ShowTexture = true;
            PointCloudColor = PointCloudColorMode.RGB;
            ViewportBgR = 0.12f;
            ViewportBgG = 0.12f;
            ViewportBgB = 0.14f;
            GridColorR = 0.35f;
            GridColorG = 0.35f;
            GridColorB = 0.35f;
            ShowGrid = true;
            ShowAxes = true;
            ShowGizmo = true;
            ShowCameras = true;
            ShowInfoOverlay = true;
            ShowPointCloudBounds = true;
            NeRFIterations = 50;
            VoxelGridSize = 128;
            NeRFLearningRate = 0.1f;
            LastWindowWidth = 1400;
            LastWindowHeight = 900;
            LastPanelWidth = 250;

            // AI Model Settings
            ImageTo3D = ImageTo3DModel.None;
            RiggingModel = RiggingMethod.None;
            MeshExtraction = MeshExtractionMethod.MarchingCubes;
            MeshRefinement = MeshRefinementMethod.None;

            // TripoSR
            TripoSRResolution = 256;
            TripoSRMarchingCubesRes = 128;
            TripoSRModelPath = "models/triposr";

            // LGM
            LGMFlowSteps = 25;
            LGMQueryResolution = 128;
            LGMModelPath = "models/lgm";

            // Wonder3D
            Wonder3DDiffusionSteps = 50;
            Wonder3DCFGScale = 3.0f;
            Wonder3DModelPath = "models/wonder3d";

            // UniRig
            UniRigMaxJoints = 64;
            UniRigMaxBonesPerVertex = 4;
            UniRigModelPath = "models/unirig";

            // DeepMeshPrior
            DeepMeshPriorIterations = 500;
            DeepMeshPriorLearningRate = 0.01f;
            DeepMeshPriorLaplacianWeight = 1.4f;

            // GaussianSDF
            GaussianSDFGridResolution = 128;
            GaussianSDFSigma = 1.0f;
            GaussianSDFIterations = 1;
            GaussianSDFIsoLevel = 0.0f;

            // TripoSF
            TripoSFResolution = 512;
            TripoSFSparseDilation = 1;
            TripoSFModelPath = "models/triposf";
        }
    }
}
