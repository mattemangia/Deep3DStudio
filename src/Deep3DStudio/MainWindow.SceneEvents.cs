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

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnSceneObjectSelected(object? sender, SceneObject obj)
        {
            _statusLabel.Text = $"Selected: {obj.Name}";

            if (obj is MeshObject mesh)
            {
                _statusLabel.Text += $" ({mesh.VertexCount:N0} vertices, {mesh.TriangleCount:N0} triangles)";
            }
            else if (obj is CameraObject cam)
            {
                _statusLabel.Text += $" ({cam.ImageWidth}x{cam.ImageHeight})";
            }
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
        }
    }
}
