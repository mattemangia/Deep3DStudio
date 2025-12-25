using System;
using System.Collections.Generic;

namespace Deep3DStudio.Model
{
    public class ProjectImage
    {
        public string FilePath { get; set; } = "";
        public string Alias { get; set; } = ""; // Display name

        [System.Text.Json.Serialization.JsonIgnore]
        public float[,]? DepthMap { get; set; }
    }
}
