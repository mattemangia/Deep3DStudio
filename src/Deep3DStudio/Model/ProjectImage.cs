using System;
using System.Collections.Generic;

namespace Deep3DStudio.Model
{
    public class ProjectImage
    {
        public string FilePath { get; set; } = "";
        public string Alias { get; set; } = ""; // Display name

        // Depth map data (width x height)
        // Stored as flattened array for serialization compatibility if needed,
        // but typically runtime only or serialized separately.
        // For simplicity in JSON, we can ignore it or assume it's transient
        // unless we want to cache it. User asked to "export depth maps",
        // but also "images miss the depth view button".
        // If we want to persist depth maps, we might save them as sidecar files
        // or just re-generate them.
        // Re-generation is safer but slow.
        // Let's keep it runtime-only for now or simple serialization.
        [System.Text.Json.Serialization.JsonIgnore]
        public float[,]? DepthMap { get; set; }
    }
}
