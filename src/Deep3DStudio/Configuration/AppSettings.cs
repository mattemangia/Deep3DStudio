using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Deep3DStudio.Configuration
{
    public enum ComputeDevice
    {
        CPU,
        CUDA,
        DirectML
    }

    public enum MeshingAlgorithm
    {
        MarchingCubes,
        Poisson,
        GreedyMeshing,
        SurfaceNets,
        Blocky
    }

    public enum CoordinateSystem
    {
        RightHanded_Y_Up,
        RightHanded_Z_Up,
        LeftHanded_Y_Up
    }

    public enum PointCloudColorMode
    {
        RGB,        // Original vertex colors from reconstruction
        DistanceMap // Colormap based on distance from camera/origin
    }

    public enum BoundingBoxMode
    {
        Corners,
        Full
    }

    public class AppSettings
    {
        private static AppSettings? _instance;
        private static readonly string ConfigPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "Deep3DStudio",
            "config.json"
        );

        public ComputeDevice Device { get; set; } = ComputeDevice.CPU;
        public MeshingAlgorithm MeshingAlgo { get; set; } = MeshingAlgorithm.MarchingCubes;
        public CoordinateSystem CoordSystem { get; set; } = CoordinateSystem.RightHanded_Y_Up;

        // Rendering State
        public bool ShowPointCloud { get; set; } = false;
        public bool ShowWireframe { get; set; } = false;
        public PointCloudColorMode PointCloudColor { get; set; } = PointCloudColorMode.RGB;

        // Bounding Box Settings
        public BoundingBoxMode BoundingBoxStyle { get; set; } = BoundingBoxMode.Full;

        // Viewport Colors (stored as RGB floats 0-1)
        public float ViewportBgR { get; set; } = 0.12f;
        public float ViewportBgG { get; set; } = 0.12f;
        public float ViewportBgB { get; set; } = 0.14f;

        public float GridColorR { get; set; } = 0.35f;
        public float GridColorG { get; set; } = 0.35f;
        public float GridColorB { get; set; } = 0.35f;

        private AppSettings() { }

        public static AppSettings Instance
        {
            get
            {
                if (_instance == null)
                    Load();
                return _instance!;
            }
        }

        public static void Load()
        {
            if (File.Exists(ConfigPath))
            {
                try
                {
                    string json = File.ReadAllText(ConfigPath);
                    _instance = JsonSerializer.Deserialize<AppSettings>(json);
                }
                catch
                {
                    _instance = new AppSettings();
                }
            }
            else
            {
                _instance = new AppSettings();
            }
        }

        public void Save()
        {
            try
            {
                string dir = Path.GetDirectoryName(ConfigPath);
                if (!Directory.Exists(dir))
                    Directory.CreateDirectory(dir);

                var options = new JsonSerializerOptions { WriteIndented = true };
                string json = JsonSerializer.Serialize(this, options);
                File.WriteAllText(ConfigPath, json);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to save settings: {ex.Message}");
            }
        }
    }
}
