using System;
using Gtk;
using Gdk;
using Deep3DStudio.Viewport;
using Deep3DStudio.Icons;
using Deep3DStudio.Model;
using Deep3DStudio.Configuration;
using Deep3DStudio.Meshing;
using Deep3DStudio.UI;
using Deep3DStudio.Scene;
using Deep3DStudio.IO;
using Deep3DStudio.Texturing;
using AIModels = Deep3DStudio.Model.AIModels;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Action = System.Action;
using Vector3 = OpenTK.Mathematics.Vector3;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnSceneObjectSelected(object? sender, SceneObject obj)
        {
            _statusLabel.Text = $"Selected: {obj.Name}";

            if (obj is SkeletonObject skeletonObj)
            {
                _activeSkeletonObject = skeletonObj;
                _riggingPanel?.SetSkeleton(skeletonObj);
            }

            if (obj is MeshObject mesh)
            {
                _statusLabel.Text += $" ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)";
            }
            else if (obj is CameraObject cam)
            {
                _statusLabel.Text += $" ({cam.ImageWidth}x{cam.ImageHeight})";
            }

            UpdatePointCloudVisibilityControls();
        }

        private void OnSceneObjectDoubleClicked(object? sender, SceneObject obj)
        {
            _sceneGraph.Select(obj);
            _viewport.FocusOnSelection();

            if (obj is CameraObject cam && !string.IsNullOrEmpty(cam.ImagePath))
            {
                // Could show image preview here
            }
        }

        private void OnSceneObjectAction(object? sender, (SceneObject obj, string action) args)
        {
            switch (args.action)
            {
                case "refresh_viewport":
                    _viewport.QueueDraw();
                    break;

                case "focus":
                    if (args.obj != null)
                    {
                        _sceneGraph.Select(args.obj);
                        _viewport.FocusOnSelection();
                    }
                    break;

                case "move":
                    ApplySceneTransformFromDialog(
                        "Move Objects",
                        Vector3.Zero,
                        "Delta X", "Delta Y", "Delta Z",
                        (obj, value) => obj.Position += value,
                        value => $"Moved {_sceneGraph.SelectedObjects.Count} object(s) by ({value.X:F3}, {value.Y:F3}, {value.Z:F3})");
                    break;

                case "rotate":
                    ApplySceneTransformFromDialog(
                        "Rotate Objects",
                        Vector3.Zero,
                        "Delta X (deg)", "Delta Y (deg)", "Delta Z (deg)",
                        (obj, value) => obj.Rotation += value,
                        value => $"Rotated {_sceneGraph.SelectedObjects.Count} object(s) by ({value.X:F2}, {value.Y:F2}, {value.Z:F2}) deg");
                    break;

                case "scale":
                    ApplySceneTransformFromDialog(
                        "Scale Objects",
                        Vector3.One,
                        "Factor X", "Factor Y", "Factor Z",
                        (obj, value) =>
                        {
                            var factor = ClampScale(value);
                            obj.Scale = ClampScale(new Vector3(
                                obj.Scale.X * factor.X,
                                obj.Scale.Y * factor.Y,
                                obj.Scale.Z * factor.Z));
                        },
                        value => $"Scaled {_sceneGraph.SelectedObjects.Count} object(s) by factors ({value.X:F3}, {value.Y:F3}, {value.Z:F3})",
                        0.001f,
                        1000.0f,
                        0.01f,
                        4);
                    break;

                case "decimate":
                    OnDecimateClicked(null, EventArgs.Empty);
                    break;

                case "optimize":
                    OnOptimizeClicked(null, EventArgs.Empty);
                    break;

                case "smooth":
                    OnSmoothClicked(null, EventArgs.Empty);
                    break;

                case "split_connectivity":
                    OnSplitClicked(null, EventArgs.Empty);
                    break;

                case "merge_meshes":
                    OnMergeClicked(null, EventArgs.Empty);
                    break;

                case "align_meshes":
                    OnAlignClicked(null, EventArgs.Empty);
                    break;

                case "downsample":
                    OnPointCloudVoxelDownsampleClicked(null, EventArgs.Empty);
                    break;

                case "remove_outliers":
                    OnPointCloudRemoveOutliersClicked(null, EventArgs.Empty);
                    break;

                case "remove_duplicates":
                    OnPointCloudRemoveDuplicatesClicked(null, EventArgs.Empty);
                    break;

                case "remove_sky_blue":
                    OnPointCloudRemoveSkyBlueClicked(null, EventArgs.Empty);
                    break;

                case "estimate_normals":
                    OnPointCloudEstimateNormalsClicked(null, EventArgs.Empty);
                    break;

                case "pass_through":
                    OnPointCloudPassThroughClicked(null, EventArgs.Empty);
                    break;

                case "radius_crop":
                    OnPointCloudRadiusCropClicked(null, EventArgs.Empty);
                    break;

                case "merge_pointclouds":
                    OnMergeClicked(null, EventArgs.Empty);
                    break;

                case "align_pointclouds":
                    OnAlignClicked(null, EventArgs.Empty);
                    break;

                case "flip_normals":
                    OnFlipNormals(null, EventArgs.Empty);
                    break;

                case "cleanup_mesh":
                    OnMeshCleanupClicked(null, EventArgs.Empty);
                    break;

                case "bake_textures":
                    OnBakeTexturesClicked(null, EventArgs.Empty);
                    break;

                case "view_from_camera":
                    if (args.obj is CameraObject cam)
                    {
                        _statusLabel.Text = $"View from {cam.Name}";
                    }
                    break;

                case "show_camera_image":
                    if (args.obj is CameraObject camImg && !string.IsNullOrEmpty(camImg.ImagePath))
                    {
                        var entry = new ImageEntry { FilePath = camImg.ImagePath };
                        var previewDialog = new ImagePreviewDialog(this, entry);
                        previewDialog.Run();
                        previewDialog.Destroy();
                    }
                    break;

                case "add_group":
                    var group = new GroupObject("New Group");
                    _sceneGraph.AddObject(group);
                    _sceneTreeView.RefreshTree();
                    break;
            }
        }

        private void OnViewportObjectPicked(object? sender, SceneObject? obj)
        {
            if (obj != null)
            {
                _sceneTreeView.SelectObject(obj);
            }

            UpdatePointCloudVisibilityControls();
        }

        private void ApplySceneTransformFromDialog(
            string title,
            Vector3 defaultValue,
            string xLabel,
            string yLabel,
            string zLabel,
            System.Action<SceneObject, Vector3> apply,
            Func<Vector3, string> statusText,
            float minValue = -100000.0f,
            float maxValue = 100000.0f,
            float step = 0.1f,
            int digits = 4)
        {
            var selected = _sceneGraph.SelectedObjects.ToList();
            if (selected.Count == 0)
            {
                _statusLabel.Text = "No objects selected.";
                return;
            }

            var dialog = new Vector3InputDialog(
                this,
                title,
                xLabel, yLabel, zLabel,
                defaultValue,
                minValue,
                maxValue,
                step,
                digits);

            if (dialog.Run() == (int)ResponseType.Ok)
            {
                var value = dialog.Value;
                foreach (var obj in selected)
                {
                    apply(obj, value);
                }

                _isDirty = true;
                UpdateTitle();
                _viewport.QueueDraw();
                _statusLabel.Text = statusText(value);
            }

            dialog.Destroy();
        }

        private static Vector3 ClampScale(Vector3 scale)
        {
            const float minScale = 0.001f;
            return new Vector3(
                Math.Max(minScale, scale.X),
                Math.Max(minScale, scale.Y),
                Math.Max(minScale, scale.Z));
        }
    }
}
