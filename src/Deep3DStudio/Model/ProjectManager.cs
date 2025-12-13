using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using System.Collections.Generic;
using Deep3DStudio.Scene;
using OpenTK.Mathematics;

namespace Deep3DStudio.Model
{
    public class ProjectManager
    {
        private static readonly JsonSerializerOptions _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            IncludeFields = true
        };

        public static void SaveProject(string filePath, MainWindow window, SceneGraph sceneGraph, List<string> imagePaths)
        {
            var state = new ProjectState();
            state.ImagePaths = new List<string>(imagePaths);
            state.Scene = ConvertSceneToDTO(sceneGraph.Root);
            state.LastModified = DateTime.Now;

            string json = JsonSerializer.Serialize(state, _jsonOptions);
            File.WriteAllText(filePath, json);
        }

        public static ProjectState LoadProject(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Project file not found", filePath);

            string json = File.ReadAllText(filePath);
            var state = JsonSerializer.Deserialize<ProjectState>(json, _jsonOptions);
            return state ?? throw new Exception("Failed to deserialize project state");
        }

        private static SceneGraphDTO ConvertSceneToDTO(GroupObject root)
        {
            var dto = new SceneGraphDTO();
            foreach (var child in root.Children)
            {
                var objDTO = ConvertObjectToDTO(child);
                if (objDTO != null)
                    dto.Objects.Add(objDTO);
            }
            return dto;
        }

        private static SceneObjectDTO? ConvertObjectToDTO(SceneObject obj)
        {
            SceneObjectDTO? dto = null;

            if (obj is MeshObject mesh)
            {
                var mDto = new MeshObjectDTO
                {
                    ShowAsPointCloud = mesh.ShowAsPointCloud,
                    PointSize = mesh.PointSize,
                    ShowWireframe = mesh.ShowWireframe,
                    MeshData = new MeshDataDTO
                    {
                        Vertices = FlattenVector3(mesh.MeshData.Vertices),
                        Colors = FlattenVector3(mesh.MeshData.Colors),
                        Indices = new List<int>(mesh.MeshData.Indices)
                    }
                };
                dto = mDto;
            }
            else if (obj is PointCloudObject pc)
            {
                var pcDto = new PointCloudObjectDTO
                {
                    PointSize = pc.PointSize,
                    Points = FlattenVector3(pc.Points),
                    Colors = FlattenVector3(pc.Colors)
                };
                dto = pcDto;
            }
            else if (obj is CameraObject cam)
            {
                var cDto = new CameraObjectDTO
                {
                    ImagePath = cam.ImagePath,
                    ImageWidth = cam.ImageWidth,
                    ImageHeight = cam.ImageHeight,
                    FieldOfView = cam.FieldOfView,
                    NearPlane = cam.NearPlane,
                    FarPlane = cam.FarPlane
                };
                dto = cDto;
            }
            else if (obj is GroupObject)
            {
                dto = new GroupObjectDTO();
            }

            if (dto != null)
            {
                dto.Name = obj.Name;
                dto.Visible = obj.Visible;
                dto.Position = obj.Position;
                dto.Rotation = obj.Rotation;
                dto.Scale = obj.Scale;

                foreach (var child in obj.Children)
                {
                    var childDto = ConvertObjectToDTO(child);
                    if (childDto != null)
                        dto.Children.Add(childDto);
                }
            }

            return dto;
        }

        public static void RestoreSceneFromState(ProjectState state, SceneGraph sceneGraph)
        {
            sceneGraph.Clear();

            foreach (var objDto in state.Scene.Objects)
            {
                var obj = ConvertDTOToObject(objDto);
                if (obj != null)
                {
                    sceneGraph.AddObject(obj);
                }
            }
        }

        private static SceneObject? ConvertDTOToObject(SceneObjectDTO dto)
        {
            SceneObject? obj = null;

            if (dto is MeshObjectDTO mDto)
            {
                var meshData = new MeshData
                {
                    Vertices = UnflattenVector3(mDto.MeshData.Vertices),
                    Colors = UnflattenVector3(mDto.MeshData.Colors),
                    Indices = mDto.MeshData.Indices
                };
                var mesh = new MeshObject(dto.Name, meshData)
                {
                    ShowAsPointCloud = mDto.ShowAsPointCloud,
                    PointSize = mDto.PointSize,
                    ShowWireframe = mDto.ShowWireframe
                };
                obj = mesh;
            }
            else if (dto is PointCloudObjectDTO pcDto)
            {
                var pc = new PointCloudObject(dto.Name)
                {
                    Points = UnflattenVector3(pcDto.Points),
                    Colors = UnflattenVector3(pcDto.Colors),
                    PointSize = pcDto.PointSize
                };
                pc.UpdateBounds();
                obj = pc;
            }
            else if (dto is CameraObjectDTO cDto)
            {
                var cam = new CameraObject(dto.Name)
                {
                    ImagePath = cDto.ImagePath,
                    ImageWidth = cDto.ImageWidth,
                    ImageHeight = cDto.ImageHeight,
                    FieldOfView = cDto.FieldOfView,
                    NearPlane = cDto.NearPlane,
                    FarPlane = cDto.FarPlane
                };
                obj = cam;
            }
            else if (dto is GroupObjectDTO)
            {
                obj = new GroupObject(dto.Name);
            }

            if (obj != null)
            {
                obj.Visible = dto.Visible;
                obj.Position = dto.Position;
                obj.Rotation = dto.Rotation;
                obj.Scale = dto.Scale;

                foreach (var childDto in dto.Children)
                {
                    var childObj = ConvertDTOToObject(childDto);
                    if (childObj != null)
                    {
                        obj.AddChild(childObj);
                    }
                }

                // Ensure bounds are updated after children added or properties set
                obj.UpdateBounds();
            }

            return obj;
        }

        private static List<float> FlattenVector3(List<Vector3> vectors)
        {
            var list = new List<float>(vectors.Count * 3);
            foreach (var v in vectors)
            {
                list.Add(v.X);
                list.Add(v.Y);
                list.Add(v.Z);
            }
            return list;
        }

        private static List<Vector3> UnflattenVector3(List<float> floats)
        {
            var list = new List<Vector3>(floats.Count / 3);
            for (int i = 0; i < floats.Count; i += 3)
            {
                if (i + 2 < floats.Count)
                {
                    list.Add(new Vector3(floats[i], floats[i+1], floats[i+2]));
                }
            }
            return list;
        }
    }
}
