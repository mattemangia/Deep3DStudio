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
        FeatureMatching
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

        // UI Settings
        public bool ShowGrid { get; set; } = true;
        public bool ShowAxes { get; set; } = true;
        public bool ShowCameras { get; set; } = true;
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
            ShowCameras = true;
            ShowInfoOverlay = true;
            ShowPointCloudBounds = true;
            NeRFIterations = 50;
            VoxelGridSize = 128;
            NeRFLearningRate = 0.1f;
            LastWindowWidth = 1400;
            LastWindowHeight = 900;
            LastPanelWidth = 250;
        }
    }
}
