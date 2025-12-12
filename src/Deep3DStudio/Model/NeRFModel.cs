using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenTK.Mathematics;
using Deep3DStudio.Model;
using Deep3DStudio.Meshing;
using SkiaSharp;

namespace Deep3DStudio.Model
{
    /// <summary>
    /// Implements a Voxel Grid based "NeRF" (Neural Radiance Field) approximation.
    /// It uses Dust3r points for initialization (Occupancy Grid) to skip empty space.
    /// It then optimizes the Color and Density of the voxels using the input images and poses.
    ///
    /// This is a CPU-based implementation optimized with Parallel.For.
    /// </summary>
    public class VoxelGridNeRF
    {
        private const int GridSize = 128; // Resolution of the voxel grid (128^3). Higher = More memory/time.
        private float[,,] _density;
        private Vector3[,,] _color;
        private bool[,,] _occupancy; // Coarse mask to skip raymarching

        private Vector3 _boundsMin;
        private Vector3 _boundsMax;
        private Vector3 _voxelSize;

        // Optimization parameters
        private float _learningRate = 0.1f;
        // private float _densityStep = 0.5f; // Removed unused variable

        public VoxelGridNeRF()
        {
            _density = new float[GridSize, GridSize, GridSize];
            _color = new Vector3[GridSize, GridSize, GridSize];
            _occupancy = new bool[GridSize, GridSize, GridSize];
        }

        /// <summary>
        /// Initializes the voxel grid from a coarse mesh (Dust3r output).
        /// Determines the bounding box and marks occupied voxels.
        /// </summary>
        public void InitializeFromMesh(List<MeshData> meshes)
        {
            // 1. Compute Bounds
            _boundsMin = new Vector3(float.PositiveInfinity);
            _boundsMax = new Vector3(float.NegativeInfinity);

            int vertexCount = 0;
            foreach (var mesh in meshes)
            {
                foreach (var v in mesh.Vertices)
                {
                    _boundsMin = Vector3.ComponentMin(_boundsMin, v);
                    _boundsMax = Vector3.ComponentMax(_boundsMax, v);
                    vertexCount++;
                }
            }

            if (vertexCount == 0) return;

            // Add padding
            Vector3 padding = (_boundsMax - _boundsMin) * 0.1f;
            _boundsMin -= padding;
            _boundsMax += padding;

            _voxelSize = (_boundsMax - _boundsMin) / GridSize;

            // 2. Mark Occupancy and Initialize Colors
            // We iterate over points and splat them into the grid
            foreach (var mesh in meshes)
            {
                for (int i = 0; i < mesh.Vertices.Count; i++)
                {
                    Vector3 p = mesh.Vertices[i];
                    Vector3 c = mesh.Colors[i];

                    Vector3 localPos = (p - _boundsMin);
                    int x = (int)(localPos.X / _voxelSize.X);
                    int y = (int)(localPos.Y / _voxelSize.Y);
                    int z = (int)(localPos.Z / _voxelSize.Z);

                    if (x >= 0 && x < GridSize && y >= 0 && y < GridSize && z >= 0 && z < GridSize)
                    {
                        _occupancy[x, y, z] = true;

                        // Initial guess: Average color, High density
                        // Simple running average for initialization
                        if (_density[x, y, z] == 0)
                        {
                            _color[x, y, z] = c;
                            _density[x, y, z] = 10.0f;
                        }
                        else
                        {
                            _color[x, y, z] = (_color[x, y, z] + c) * 0.5f;
                        }

                        // Dilate occupancy slightly to capture near surface details
                        DilateOccupancy(x, y, z);
                    }
                }
            }
        }

        private void DilateOccupancy(int x, int y, int z)
        {
            for (int dx = -1; dx <= 1; dx++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dz = -1; dz <= 1; dz++)
                    {
                        int nx = x + dx, ny = y + dy, nz = z + dz;
                        if (nx >= 0 && nx < GridSize && ny >= 0 && ny < GridSize && nz >= 0 && nz < GridSize)
                        {
                            if (!_occupancy[nx, ny, nz])
                            {
                                _occupancy[nx, ny, nz] = true;
                                _density[nx, ny, nz] = 0.1f; // Low initial density for expanded areas
                            }
                        }
                    }
        }

        /// <summary>
        /// Runs the training loop to refine Color and Density using the input images.
        /// </summary>
        public void Train(List<CameraPose> poses, int iterations = 5)
        {
            // Simple SGD-like optimization
            Console.WriteLine($"Starting NeRF Training ({iterations} iterations)...");

            for (int iter = 0; iter < iterations; iter++)
            {
                var densityGrad = new float[GridSize, GridSize, GridSize];
                var colorGrad = new Vector3[GridSize, GridSize, GridSize];
                var counts = new int[GridSize, GridSize, GridSize];

                foreach (var pose in poses)
                {
                    ProcessImageOptimized(pose, densityGrad, colorGrad, counts);
                }

                // Apply Updates
                ApplyGradients(densityGrad, colorGrad, counts);

                Console.WriteLine($"  Iteration {iter+1}/{iterations} complete.");
            }
        }

        private void ProcessImageOptimized(CameraPose pose, float[,,] densityGrad, Vector3[,,] colorGrad, int[,,] counts)
        {
            var (tensor, shape) = ImageUtils.LoadAndPreprocessImage(pose.ImagePath);
            int W = shape[1];
            int H = shape[0];
            var colors = ImageUtils.ExtractColors(pose.ImagePath, W, H); // This reloads/resizes to match W/H

            Vector3 camPos = pose.CameraToWorld.ExtractTranslation();

            // Assume FOV 60
            float fov = MathHelper.DegreesToRadians(60);
            float focal = (W * 0.5f) / (float)Math.Tan(fov * 0.5f);

            int stride = 4;
            int numRaysY = (H + stride - 1) / stride;

            Parallel.For(0, numRaysY,
                () => new Dictionary<int, (Vector3 colSum, float denSum, int count)>(), // Local Init
                (yIdx, loop, localState) => // Body
                {
                    int y = yIdx * stride;
                    if (y >= H) return localState;

                    for (int x = 0; x < W; x += stride)
                    {
                        // Ray Generation
                        float u = (x - W * 0.5f) / focal;
                        float v = (y - H * 0.5f) / focal;
                        Vector3 dirCam = new Vector3(u, v, 1.0f).Normalized();
                        Vector3 dirWorld = Vector3.TransformNormal(dirCam, pose.CameraToWorld).Normalized();

                        if (!GeometryUtils.RayBoxIntersection(camPos, dirWorld, _boundsMin, _boundsMax, out float tMin, out float tMax))
                            continue;

                        float t = Math.Max(0, tMin);
                        float stepSize = _voxelSize.X * 0.5f;
                        float transmittance = 1.0f;

                        var targetColP = colors[y * W + x];
                        var targetCol = new Vector3(targetColP.Red/255f, targetColP.Green/255f, targetColP.Blue/255f);

                        while (t < tMax && transmittance > 0.01f)
                        {
                            Vector3 pos = camPos + dirWorld * t;
                            Vector3 localPos = pos - _boundsMin;
                            int vx = (int)(localPos.X / _voxelSize.X);
                            int vy = (int)(localPos.Y / _voxelSize.Y);
                            int vz = (int)(localPos.Z / _voxelSize.Z);

                            if (vx >= 0 && vx < GridSize && vy >= 0 && vy < GridSize && vz >= 0 && vz < GridSize)
                            {
                                if (_occupancy[vx, vy, vz])
                                {
                                    float dens = _density[vx, vy, vz];
                                    Vector3 col = _color[vx, vy, vz];
                                    float alpha = 1.0f - (float)Math.Exp(-dens * stepSize);
                                    float weight = alpha * transmittance;

                                    // Accumulate into local state
                                    int key = vz * GridSize * GridSize + vy * GridSize + vx;

                                    float dGrad = 0;
                                    float error = (targetCol - col).Length;
                                    if (error < 0.2f) dGrad = 1.0f * weight;
                                    else dGrad = -0.5f * weight;

                                    if (localState.TryGetValue(key, out var val))
                                    {
                                        localState[key] = (val.colSum + targetCol * weight, val.denSum + dGrad, val.count + 1);
                                    }
                                    else
                                    {
                                        localState[key] = (targetCol * weight, dGrad, 1);
                                    }

                                    transmittance *= (1.0f - alpha);
                                }
                            }
                            t += stepSize;
                        }
                    }
                    return localState;
                },
                (localState) => // Finalizer
                {
                    lock (counts)
                    {
                        foreach (var kvp in localState)
                        {
                            int idx = kvp.Key;
                            int vz = idx / (GridSize * GridSize);
                            int rem = idx % (GridSize * GridSize);
                            int vy = rem / GridSize;
                            int vx = rem % GridSize;

                            colorGrad[vx, vy, vz] += kvp.Value.colSum;
                            densityGrad[vx, vy, vz] += kvp.Value.denSum;
                            counts[vx, vy, vz] += kvp.Value.count;
                        }
                    }
                }
            );
        }

        private void ApplyGradients(float[,,] densityGrad, Vector3[,,] colorGrad, int[,,] counts)
        {
            // Apply accumulated gradients
            Parallel.For(0, GridSize, x =>
            {
                for (int y = 0; y < GridSize; y++)
                {
                    for (int z = 0; z < GridSize; z++)
                    {
                        if (_occupancy[x, y, z] && counts[x,y,z] > 0)
                        {
                            // Average color seen from all cameras
                            Vector3 avgTargetColor = colorGrad[x, y, z] / counts[x, y, z];

                            // Blend current color with observed color
                            _color[x, y, z] = Vector3.Lerp(_color[x, y, z], avgTargetColor, _learningRate);

                            // Update density
                            _density[x, y, z] += densityGrad[x, y, z] * _learningRate;
                            _density[x, y, z] = Math.Clamp(_density[x, y, z], 0.0f, 100.0f);

                            // Carve empty space
                            if (_density[x, y, z] < 0.1f)
                                _occupancy[x, y, z] = false;
                        }
                    }
                }
            });
        }

        public MeshData GetMesh()
        {
            // Default legacy extraction
            return GetMesh(new MarchingCubesMesher());
        }

        public MeshData GetMesh(IMesher mesher)
        {
            // Use the provided mesher
            // Note: Most meshers need density, origin, voxelSize, isoLevel.
            // Some meshers might ignore color.

            // If the mesher supports color sampling internally, great.
            // Generate mesh using the selected meshing algorithm
            var mesh = mesher.GenerateMesh(_density, _boundsMin, _voxelSize.X, 5.0f);

            // Post-process: interpolate colors from voxel color grid for each vertex
            for(int i=0; i<mesh.Vertices.Count; i++)
            {
                 Vector3 v = mesh.Vertices[i];
                 Vector3 local = v - _boundsMin;
                 int x = (int)(local.X / _voxelSize.X);
                 int y = (int)(local.Y / _voxelSize.Y);
                 int z = (int)(local.Z / _voxelSize.Z);
                 x = Math.Clamp(x, 0, GridSize-1);
                 y = Math.Clamp(y, 0, GridSize-1);
                 z = Math.Clamp(z, 0, GridSize-1);

                 // If the mesher didn't set colors (defaulting to white/grey), overwrite them.
                 // MarchingCubes (GeometryUtils) sets colors if provided.
                 // My wrappers might not.
                 // Let's just sample nearest voxel color.
                 if(mesh.Colors.Count <= i) mesh.Colors.Add(_color[x,y,z]);
                 else mesh.Colors[i] = _color[x,y,z];
            }

            return mesh;
        }
    }
}
