using System;
using System.Collections.Generic;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class CameraPose
    {
        public Matrix4 WorldToCamera { get; set; } // View Matrix
        public Matrix4 CameraToWorld { get; set; } // Pose Matrix
        public int ImageIndex { get; set; }
        public string ImagePath { get; set; } = string.Empty;
        public int Width { get; set; }
        public int Height { get; set; }
        /// <summary>
        /// Estimated focal length in pixels. Default is max(Width, Height) * 0.85 if not set.
        /// </summary>
        public float FocalLength { get; set; } = 0;

        /// <summary>
        /// Gets the effective focal length (uses estimated value if FocalLength is not set)
        /// </summary>
        public float GetEffectiveFocalLength()
        {
            if (FocalLength > 0) return FocalLength;
            return Math.Max(Width, Height) * 0.85f;
        }
    }

    public class SceneResult
    {
        public List<MeshData> Meshes { get; set; } = new List<MeshData>();
        public List<CameraPose> Poses { get; set; } = new List<CameraPose>();
    }
}
