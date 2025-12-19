using System;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model.AIModels
{
    /// <summary>
    /// Result from an AI model inference operation.
    /// </summary>
    public class AIModelResult
    {
        /// <summary>3D mesh vertices (world coordinates)</summary>
        public Vector3[]? Vertices { get; set; }

        /// <summary>Per-vertex colors (RGB, 0-1 range)</summary>
        public Vector3[]? Colors { get; set; }

        /// <summary>Triangle indices (triplets)</summary>
        public int[]? Triangles { get; set; }

        /// <summary>Per-vertex normals</summary>
        public Vector3[]? Normals { get; set; }

        /// <summary>Signed distance field grid (if applicable)</summary>
        public float[,,]? SDFGrid { get; set; }

        /// <summary>Triplane features (if applicable)</summary>
        public float[]? TriplaneTokens { get; set; }

        /// <summary>Latent representation (for downstream tasks)</summary>
        public float[]? Latent { get; set; }

        /// <summary>Status message for UI feedback</summary>
        public string? StatusMessage { get; set; }

        /// <summary>Whether the operation succeeded</summary>
        public bool Success { get; set; }

        /// <summary>Processing time in milliseconds</summary>
        public long ProcessingTimeMs { get; set; }
    }

    /// <summary>
    /// Result from a rigging operation (UniRig).
    /// </summary>
    public class RigResult
    {
        /// <summary>Joint positions in world space</summary>
        public Vector3[]? JointPositions { get; set; }

        /// <summary>Parent index for each joint (-1 for root)</summary>
        public int[]? ParentIndices { get; set; }

        /// <summary>Joint names (for FBX export)</summary>
        public string[]? JointNames { get; set; }

        /// <summary>Skinning weights [vertex, bone]</summary>
        public float[,]? SkinningWeights { get; set; }

        /// <summary>Bone local axes [bone, 3x3 matrix]</summary>
        public float[,,]? BoneAxes { get; set; }

        /// <summary>Status message</summary>
        public string? StatusMessage { get; set; }

        /// <summary>Whether the operation succeeded</summary>
        public bool Success { get; set; }
    }

    /// <summary>
    /// Base interface for AI model inference.
    /// </summary>
    public interface IAIModelInference : IDisposable
    {
        /// <summary>
        /// Whether the model is loaded and ready for inference.
        /// </summary>
        bool IsLoaded { get; }

        /// <summary>
        /// Model name for UI display.
        /// </summary>
        string ModelName { get; }

        /// <summary>
        /// Human-readable description.
        /// </summary>
        string Description { get; }

        /// <summary>
        /// Load the ONNX model(s) from the specified directory.
        /// </summary>
        /// <param name="modelPath">Directory containing ONNX files</param>
        /// <returns>True if loaded successfully</returns>
        bool LoadModel(string modelPath);

        /// <summary>
        /// Unload the model and free resources.
        /// </summary>
        void UnloadModel();
    }

    /// <summary>
    /// Interface for image-to-3D models (TripoSR, LGM, Wonder3D).
    /// </summary>
    public interface IImageTo3DModel : IAIModelInference
    {
        /// <summary>
        /// Generate 3D from a single image.
        /// </summary>
        /// <param name="imagePath">Path to input image</param>
        /// <returns>3D model result</returns>
        AIModelResult GenerateFromImage(string imagePath);

        /// <summary>
        /// Generate 3D from image bytes (for in-memory processing).
        /// </summary>
        /// <param name="imageData">RGBA image data</param>
        /// <param name="width">Image width</param>
        /// <param name="height">Image height</param>
        /// <returns>3D model result</returns>
        AIModelResult GenerateFromImageData(byte[] imageData, int width, int height);
    }

    /// <summary>
    /// Interface for mesh refinement models (TripoSF, DeepMeshPrior).
    /// </summary>
    public interface IMeshRefinementModel : IAIModelInference
    {
        /// <summary>
        /// Refine/reconstruct mesh from point cloud.
        /// </summary>
        /// <param name="points">Input point cloud</param>
        /// <param name="colors">Optional point colors</param>
        /// <returns>Refined mesh</returns>
        AIModelResult RefineFromPointCloud(Vector3[] points, Vector3[]? colors = null);

        /// <summary>
        /// Extract mesh from SDF grid.
        /// </summary>
        /// <param name="sdfGrid">3D SDF grid</param>
        /// <param name="resolution">Grid resolution</param>
        /// <returns>Extracted mesh</returns>
        AIModelResult ExtractFromSDF(float[,,] sdfGrid, int resolution);
    }

    /// <summary>
    /// Interface for rigging models (UniRig).
    /// </summary>
    public interface IRiggingModel : IAIModelInference
    {
        /// <summary>
        /// Rig a mesh with automatic skeleton and skinning weights.
        /// </summary>
        /// <param name="vertices">Mesh vertices</param>
        /// <param name="triangles">Triangle indices</param>
        /// <returns>Rigging result with skeleton and weights</returns>
        RigResult RigMesh(Vector3[] vertices, int[] triangles);
    }

    /// <summary>
    /// Interface for multi-view generation models (Wonder3D).
    /// </summary>
    public interface IMultiViewModel : IAIModelInference
    {
        /// <summary>
        /// Generate multiple views from a single image.
        /// </summary>
        /// <param name="imagePath">Path to input image</param>
        /// <param name="numViews">Number of views to generate</param>
        /// <returns>Generated view images (RGB) and normal maps</returns>
        (byte[][] rgbViews, byte[][] normalViews) GenerateViews(string imagePath, int numViews = 6);
    }
}
