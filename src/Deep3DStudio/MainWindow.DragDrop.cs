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
        private void OnDragDataReceived(object o, DragDataReceivedArgs args)
        {
            if (args.SelectionData.Length > 0 && args.SelectionData.Format == 8)
            {
                var uris = System.Text.Encoding.UTF8.GetString(args.SelectionData.Data).Split('\n');
                var imageFiles = new List<string>();
                int importedCount = 0;

                foreach (var uri in uris)
                {
                    var cleanUri = uri.Trim();
                    if (string.IsNullOrEmpty(cleanUri)) continue;

                    if (cleanUri.StartsWith("file://"))
                    {
                        // On Linux/Unix, file:///path/to/file.
                        // On Windows, file:///C:/path/to/file
                        string path = new Uri(cleanUri).LocalPath;

                        // LocalPath is already unescaped by the Uri class

                        if (System.IO.Directory.Exists(path)) continue; // Skip directories for now

                        string ext = System.IO.Path.GetExtension(path).ToLower();

                        if (ext == ".obj" || ext == ".stl" || (ext == ".ply" && IsMeshPly(path)))
                        {
                            try
                            {
                                var meshData = MeshImporter.Load(path);
                                var meshObj = new MeshObject(System.IO.Path.GetFileNameWithoutExtension(path), meshData);
                                _sceneGraph.AddObject(meshObj);
                                importedCount++;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error importing mesh {path}: {ex.Message}");
                            }
                        }
                        else if (ext == ".xyz" || ext == ".ply") // PLY defaults to point cloud if not clearly mesh or failed mesh check
                        {
                            try
                            {
                                var pcObj = PointCloudImporter.Load(path);
                                _sceneGraph.AddObject(pcObj);
                                importedCount++;
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error importing point cloud {path}: {ex.Message}");
                            }
                        }
                        else if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".tif" || ext == ".tiff" || ext == ".bmp")
                        {
                            imageFiles.Add(path);
                        }
                    }
                }

                if (imageFiles.Count > 0)
                {
                    foreach (var img in imageFiles)
                    {
                        if (!_imagePaths.Contains(img))
                        {
                            _imagePaths.Add(img);
                            _imageBrowser.AddImage(img);
                        }
                    }
                    _statusLabel.Text = $"{_imageBrowser.ImageCount} images loaded";
                }

                if (importedCount > 0)
                {
                    _statusLabel.Text = $"Imported {importedCount} 3D objects";
                    _viewport.QueueDraw();
                    _sceneTreeView.RefreshTree();
                }
            }
            Gtk.Drag.Finish(args.Context, true, false, args.Time);
        }

        private bool IsMeshPly(string path)
        {
            // Simple heuristic check if PLY contains "element face"
            try
            {
                using (var reader = new System.IO.StreamReader(path))
                {
                    for (int i = 0; i < 20; i++) // Check first 20 lines
                    {
                        var line = reader.ReadLine();
                        if (line == null) break;
                        if (line.Contains("element face") && !line.Contains("element face 0")) return true;
                        if (line == "end_header") break;
                    }
                }
            }
            catch
            {
            }
            return false;
        }
    }
}
