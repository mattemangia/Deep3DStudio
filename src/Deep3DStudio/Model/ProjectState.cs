using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class ProjectState
    {
        public string Version { get; set; } = "1.0";
        public List<string> ImagePaths { get; set; } = new List<string>();
        public List<ProjectImage> Images { get; set; } = new List<ProjectImage>();
        public SceneGraphDTO Scene { get; set; } = new SceneGraphDTO();
        public DateTime Created { get; set; } = DateTime.Now;
        public DateTime LastModified { get; set; } = DateTime.Now;
    }

    public class SceneGraphDTO
    {
        public List<SceneObjectDTO> Objects { get; set; } = new List<SceneObjectDTO>();
    }

    [JsonPolymorphic(TypeDiscriminatorPropertyName = "$type")]
    [JsonDerivedType(typeof(MeshObjectDTO), "Mesh")]
    [JsonDerivedType(typeof(PointCloudObjectDTO), "PointCloud")]
    [JsonDerivedType(typeof(CameraObjectDTO), "Camera")]
    [JsonDerivedType(typeof(GroupObjectDTO), "Group")]
    public class SceneObjectDTO
    {
        public string Name { get; set; } = "Object";
        public bool Visible { get; set; } = true;

        [JsonConverter(typeof(Vector3Converter))]
        public Vector3 Position { get; set; }

        [JsonConverter(typeof(Vector3Converter))]
        public Vector3 Rotation { get; set; }

        [JsonConverter(typeof(Vector3Converter))]
        public Vector3 Scale { get; set; } = Vector3.One;

        public List<SceneObjectDTO> Children { get; set; } = new List<SceneObjectDTO>();
    }

    public class MeshObjectDTO : SceneObjectDTO
    {
        public MeshDataDTO MeshData { get; set; } = new MeshDataDTO();
        public bool ShowAsPointCloud { get; set; }
        public float PointSize { get; set; }
        public bool ShowWireframe { get; set; }
    }

    public class PointCloudObjectDTO : SceneObjectDTO
    {
        public List<float> Points { get; set; } = new List<float>(); // Flattened [x,y,z, x,y,z...]
        public List<float> Colors { get; set; } = new List<float>();
        public float PointSize { get; set; }
    }

    public class CameraObjectDTO : SceneObjectDTO
    {
        public string ImagePath { get; set; } = "";
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public float FieldOfView { get; set; }
        public float NearPlane { get; set; }
        public float FarPlane { get; set; }
    }

    public class GroupObjectDTO : SceneObjectDTO
    {
    }

    public class MeshDataDTO
    {
        public List<float> Vertices { get; set; } = new List<float>(); // Flattened
        public List<float> Colors { get; set; } = new List<float>();
        public List<int> Indices { get; set; } = new List<int>();
    }

    public class Vector3Converter : JsonConverter<Vector3>
    {
        public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            // Expect array [x, y, z]
            if (reader.TokenType != JsonTokenType.StartArray)
                throw new JsonException();

            reader.Read();
            float x = (float)reader.GetDouble();
            reader.Read();
            float y = (float)reader.GetDouble();
            reader.Read();
            float z = (float)reader.GetDouble();
            reader.Read(); // EndArray

            return new Vector3(x, y, z);
        }

        public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
        {
            writer.WriteStartArray();
            writer.WriteNumberValue(value.X);
            writer.WriteNumberValue(value.Y);
            writer.WriteNumberValue(value.Z);
            writer.WriteEndArray();
        }
    }
}
