using System;
using System.IO;
using System.Text.Json;
using System.Linq;
using System.Collections.Generic;
using Deep3DStudio.Scene;
using OpenTK.Mathematics;
using SkiaSharp;

namespace Deep3DStudio.Model
{
    public class ProjectManager
    {
        private static readonly JsonSerializerOptions _jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            IncludeFields = true
        };

        public static void SaveProject(
            string filePath,
            MainWindow window,
            SceneGraph sceneGraph,
            List<string> imagePaths,
            List<ProjectImage>? images = null)
        {
            var state = new ProjectState();
            if (images != null && images.Count > 0)
            {
                state.Images = images
                    .Where(i => i != null && !string.IsNullOrWhiteSpace(i.FilePath))
                    .Select(i => new ProjectImage
                    {
                        FilePath = i.FilePath,
                        Alias = string.IsNullOrWhiteSpace(i.Alias) ? Path.GetFileName(i.FilePath) : i.Alias
                    })
                    .ToList();
                state.ImagePaths = state.Images.Select(i => i.FilePath).ToList();
            }
            else
            {
                state.ImagePaths = new List<string>(imagePaths);
                state.Images = imagePaths.Select(p => new ProjectImage
                {
                    FilePath = p,
                    Alias = Path.GetFileName(p)
                }).ToList();
            }

            state.Scene = ConvertSceneToDTO(sceneGraph.Root);
            ApplyGcpFlagsToImages(state.Images);
            GeoReferenceRuntime.ApplyToState(state);
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
            var result = state ?? throw new Exception("Failed to deserialize project state");
            GeoReferenceRuntime.LoadFromState(result);
            return result;
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
                        Normals = FlattenVector3(mesh.MeshData.Normals),
                        Colors = FlattenVector3(mesh.MeshData.Colors),
                        Confidence = new List<float>(mesh.MeshData.Confidence),
                        UVs = FlattenVector2(mesh.MeshData.UVs),
                        TexturePngBase64 = EncodeTexture(mesh.MeshData.Texture),
                        PoseMatrix = mesh.MeshData.Pose.HasValue ? FlattenMatrix4(mesh.MeshData.Pose.Value) : null,
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
                    VisibleFraction = pc.VisibleFraction,
                    Points = FlattenVector3(pc.Points),
                    Colors = FlattenVector3(pc.Colors),
                    Normals = FlattenVector3(pc.Normals),
                    Confidence = new List<float>(pc.Confidence)
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
                    FarPlane = cam.FarPlane,
                    AspectRatio = cam.AspectRatio,
                    FrustumScale = cam.FrustumScale,
                    FrustumColor = cam.FrustumColor,
                    ShowFrustum = cam.ShowFrustum,
                    ShowImagePlane = cam.ShowImagePlane,
                    Pose = cam.Pose != null ? ToPoseDto(cam.Pose) : null
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
                dto.RenderMode = obj.RenderMode;
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
                    Normals = UnflattenVector3(mDto.MeshData.Normals),
                    Colors = UnflattenVector3(mDto.MeshData.Colors),
                    Confidence = new List<float>(mDto.MeshData.Confidence),
                    UVs = UnflattenVector2(mDto.MeshData.UVs),
                    Indices = mDto.MeshData.Indices
                };
                meshData.Texture = DecodeTexture(mDto.MeshData.TexturePngBase64);
                meshData.Pose = UnflattenMatrix4Nullable(mDto.MeshData.PoseMatrix);
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
                    Normals = UnflattenVector3(pcDto.Normals),
                    Confidence = new List<float>(pcDto.Confidence),
                    PointSize = pcDto.PointSize,
                    VisibleFraction = pcDto.VisibleFraction
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
                    FarPlane = cDto.FarPlane,
                    AspectRatio = cDto.AspectRatio > 0 ? cDto.AspectRatio : 1.333f,
                    FrustumScale = cDto.FrustumScale,
                    FrustumColor = cDto.FrustumColor,
                    ShowFrustum = cDto.ShowFrustum,
                    ShowImagePlane = cDto.ShowImagePlane,
                    Pose = cDto.Pose != null ? FromPoseDto(cDto.Pose) : null
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
                obj.RenderMode = dto.RenderMode;
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

        private static List<float> FlattenVector2(List<Vector2> vectors)
        {
            var list = new List<float>(vectors.Count * 2);
            foreach (var v in vectors)
            {
                list.Add(v.X);
                list.Add(v.Y);
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

        private static List<Vector2> UnflattenVector2(List<float> floats)
        {
            var list = new List<Vector2>(floats.Count / 2);
            for (int i = 0; i < floats.Count; i += 2)
            {
                if (i + 1 < floats.Count)
                {
                    list.Add(new Vector2(floats[i], floats[i + 1]));
                }
            }
            return list;
        }

        private static List<float> FlattenMatrix4(Matrix4 m)
        {
            return new List<float>(16)
            {
                m.M11, m.M12, m.M13, m.M14,
                m.M21, m.M22, m.M23, m.M24,
                m.M31, m.M32, m.M33, m.M34,
                m.M41, m.M42, m.M43, m.M44
            };
        }

        private static Matrix4 UnflattenMatrix4(List<float> data)
        {
            if (data.Count < 16) return Matrix4.Identity;
            return new Matrix4(
                data[0], data[1], data[2], data[3],
                data[4], data[5], data[6], data[7],
                data[8], data[9], data[10], data[11],
                data[12], data[13], data[14], data[15]
            );
        }

        private static Matrix4? UnflattenMatrix4Nullable(List<float>? data)
        {
            if (data == null || data.Count < 16) return null;
            return UnflattenMatrix4(data);
        }

        private static CameraPoseDTO ToPoseDto(CameraPose pose)
        {
            return new CameraPoseDTO
            {
                WorldToCamera = FlattenMatrix4(pose.WorldToCamera),
                CameraToWorld = FlattenMatrix4(pose.CameraToWorld),
                ImageIndex = pose.ImageIndex,
                ImagePath = pose.ImagePath,
                Width = pose.Width,
                Height = pose.Height,
                FocalLength = pose.FocalLength
            };
        }

        private static CameraPose FromPoseDto(CameraPoseDTO dto)
        {
            return new CameraPose
            {
                WorldToCamera = UnflattenMatrix4(dto.WorldToCamera),
                CameraToWorld = UnflattenMatrix4(dto.CameraToWorld),
                ImageIndex = dto.ImageIndex,
                ImagePath = dto.ImagePath,
                Width = dto.Width,
                Height = dto.Height,
                FocalLength = dto.FocalLength
            };
        }

        private static string? EncodeTexture(SKBitmap? texture)
        {
            if (texture == null) return null;

            using var image = SKImage.FromBitmap(texture);
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            if (data == null) return null;
            return Convert.ToBase64String(data.ToArray());
        }

        private static SKBitmap? DecodeTexture(string? base64)
        {
            if (string.IsNullOrWhiteSpace(base64)) return null;
            try
            {
                var bytes = Convert.FromBase64String(base64);
                return SKBitmap.Decode(bytes);
            }
            catch
            {
                return null;
            }
        }

        private static void ApplyGcpFlagsToImages(List<ProjectImage> images)
        {
            var gcps = GeoReferenceRuntime.Gcps;
            var pendingGcps = GeoReferenceRuntime.PendingGcps;
            foreach (var img in images)
            {
                int gcpCount = gcps.Count(g =>
                    string.Equals(NormalizePath(g.ImagePath), NormalizePath(img.FilePath), StringComparison.OrdinalIgnoreCase));
                int pendingCount = pendingGcps.Count(g =>
                    string.Equals(NormalizePath(g.ImagePath), NormalizePath(img.FilePath), StringComparison.OrdinalIgnoreCase));
                img.GcpCount = gcpCount + pendingCount;
                img.HasGcps = img.GcpCount > 0;
            }
        }

        private static string NormalizePath(string path)
        {
            try
            {
                return Path.GetFullPath(path).Replace('\\', '/').ToLowerInvariant();
            }
            catch
            {
                return path.Replace('\\', '/').ToLowerInvariant();
            }
        }
    }
}
