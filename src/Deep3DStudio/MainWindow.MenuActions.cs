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
using System.IO;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnNewProject(object? sender, EventArgs e)
        {
            if (!CheckSaveChanges()) return;

            _imagePaths.Clear();
            _imageBrowser.Clear();
            _sceneGraph.Clear();
            _lastSceneResult = null;

            _statusLabel.Text = "Project cleared. Ready.";
            _isDirty = false;
            UpdateTitle();

            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnSaveProject(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Save Project", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Deep3D Project";
            filter.AddPattern("*.d3d");
            fc.AddFilter(filter);
            fc.CurrentName = "MyProject.d3d";

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string path = fc.Filename;
                    if (!path.EndsWith(".d3d")) path += ".d3d";

                    ProjectManager.SaveProject(path, this, _sceneGraph, _imagePaths);
                    _statusLabel.Text = $"Project saved to {path}";
                    _isDirty = false;
                    UpdateTitle();
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error saving project: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private void OnLoadProject(object? sender, EventArgs e)
        {
            if (!CheckSaveChanges()) return;

            var fc = new FileChooserDialog("Open Project", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Deep3D Project";
            filter.AddPattern("*.d3d");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    _statusLabel.Text = "Loading project...";
                    while (Application.EventsPending()) Application.RunIteration();

                    string path = fc.Filename;
                    var state = ProjectManager.LoadProject(path);

                    // Clear current
                    _imagePaths.Clear();
                    _imageBrowser.Clear();
                    _sceneGraph.Clear();
                    _lastSceneResult = null;

                    // Restore images
                    foreach (var imgPath in state.ImagePaths)
                    {
                        if (System.IO.File.Exists(imgPath))
                        {
                            _imagePaths.Add(imgPath);
                            _imageBrowser.AddImage(imgPath);
                        }
                        else
                        {
                            Console.WriteLine($"Warning: Image not found at {imgPath}");
                        }
                    }

                    // Restore scene
                    ProjectManager.RestoreSceneFromState(state, _sceneGraph);

                    _statusLabel.Text = $"Project loaded from {path}";
                    _isDirty = false;
                    UpdateTitle();

                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error loading project: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private bool CheckSaveChanges()
        {
            if (!_isDirty) return true;

            var dialog = new MessageDialog(this, DialogFlags.Modal, MessageType.Question, ButtonsType.None,
                "Do you want to save changes to the current project?");
            dialog.AddButton("Yes", ResponseType.Yes);
            dialog.AddButton("No", ResponseType.No);
            dialog.AddButton("Cancel", ResponseType.Cancel);

            int response = dialog.Run();
            dialog.Destroy();

            if (response == (int)ResponseType.Yes)
            {
                OnSaveProject(null, EventArgs.Empty);
                // If the project is still marked dirty after the save prompt, treat it as the user cancelling the save dialog.
                return !_isDirty;
            }
            else if (response == (int)ResponseType.No)
            {
                return true; // Discard changes
            }

            return false; // Cancel action
        }

        private void OnImportMesh(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Import Mesh", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Mesh Files";
            filter.AddPattern("*.obj");
            filter.AddPattern("*.ply");
            filter.AddPattern("*.stl");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string path = fc.Filename;
                    var meshData = MeshImporter.Load(path);
                    var meshObj = new MeshObject(System.IO.Path.GetFileNameWithoutExtension(path), meshData);
                    _sceneGraph.AddObject(meshObj);
                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Imported mesh from {path}";
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error importing mesh: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private void OnImportPointCloud(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Import Point Cloud", this, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "Point Cloud Files";
            filter.AddPattern("*.ply");
            filter.AddPattern("*.xyz");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string path = fc.Filename;
                    var pcObj = PointCloudImporter.Load(path);
                    _sceneGraph.AddObject(pcObj);
                    _sceneTreeView.RefreshTree();
                    _viewport.QueueDraw();
                    _statusLabel.Text = $"Imported point cloud from {path}";
                }
                catch (Exception ex)
                {
                    ShowMessage($"Error importing point cloud: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private void OnExportMesh(object? sender, EventArgs e)
        {
            var selectedMeshes = _sceneGraph.SelectedObjects.OfType<MeshObject>().ToList();
            if (selectedMeshes.Count == 0)
            {
                ShowMessage("Please select a mesh to export.");
                return;
            }

            var fc = new FileChooserDialog("Export Mesh", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            var filter = new FileFilter();
            filter.Name = "OBJ Files";
            filter.AddPattern("*.obj");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                _statusLabel.Text = $"Export to {fc.Filename} - Not yet implemented";
            }
            fc.Destroy();
        }

        private void OnExportPointCloud(object? sender, EventArgs e)
        {
            var selectedObjects = _sceneGraph.SelectedObjects;
            if (selectedObjects.Count == 0)
            {
                ShowMessage("Please select objects (Mesh or Point Cloud) to export.");
                return;
            }

            var fc = new FileChooserDialog("Export Point Cloud", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);

            var plyFilter = new FileFilter { Name = "PLY Files" };
            plyFilter.AddPattern("*.ply");
            fc.AddFilter(plyFilter);

            var xyzFilter = new FileFilter { Name = "XYZ Files" };
            xyzFilter.AddPattern("*.xyz");
            fc.AddFilter(xyzFilter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                string filename = fc.Filename;
                if (!filename.EndsWith(".ply", StringComparison.OrdinalIgnoreCase) && !filename.EndsWith(".xyz", StringComparison.OrdinalIgnoreCase))
                {
                    // Default to PLY if no extension or unknown
                    filename += ".ply";
                }

                string ext = System.IO.Path.GetExtension(filename).ToLower();
                var format = ext == ".xyz" ? PointCloudExporter.ExportFormat.XYZ : PointCloudExporter.ExportFormat.PLY;

                // Ask for RGB
                bool includeColors = true;
                var colorMsg = new MessageDialog(this, DialogFlags.Modal, MessageType.Question, ButtonsType.YesNo, "Include RGB Colors?");
                if (colorMsg.Run() == (int)ResponseType.No) includeColors = false;
                colorMsg.Destroy();

                int count = 0;
                foreach (var obj in selectedObjects)
                {
                    // When exporting multiple objects, append an identifier to avoid overwriting files
                    // and process each object individually rather than merging them.

                    string currentPath = filename;
                    if (selectedObjects.Count > 1)
                    {
                        string dir = System.IO.Path.GetDirectoryName(filename) ?? "";
                        string name = System.IO.Path.GetFileNameWithoutExtension(filename);
                        currentPath = System.IO.Path.Combine(dir, $"{name}_{obj.Name}{ext}");
                    }

                    if (obj is MeshObject meshObj)
                    {
                        PointCloudExporter.Export(currentPath, meshObj.MeshData, format, includeColors);
                        count++;
                    }
                    else if (obj is PointCloudObject pcObj)
                    {
                        PointCloudExporter.Export(currentPath, pcObj, format, includeColors);
                        count++;
                    }
                }

                _statusLabel.Text = $"Exported {count} point cloud(s).";
            }
            fc.Destroy();
        }

        private void OnExportDepthMaps(object? sender, EventArgs e)
        {
            var images = _imageBrowser.GetImages();
            var imagesWithDepth = images.Where(i => i.DepthMap != null).ToList();

            if (imagesWithDepth.Count == 0)
            {
                ShowMessage("No depth maps available. Run reconstruction first.");
                return;
            }

            var fc = new FileChooserDialog("Select Output Folder for Depth Maps", this, FileChooserAction.SelectFolder,
                "Cancel", ResponseType.Cancel, "Select", ResponseType.Accept);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                string outputDir = fc.Filename;
                int exported = 0;

                _statusLabel.Text = "Exporting depth maps...";
                while (Application.EventsPending()) Application.RunIteration();

                foreach (var img in imagesWithDepth)
                {
                    if (img.DepthMap == null) continue;

                    string baseName = System.IO.Path.GetFileNameWithoutExtension(img.FileName);
                    string outPath = System.IO.Path.Combine(outputDir, $"{baseName}_depth.png");

                    try
                    {
                        var depthMap = img.DepthMap;
                        int width = depthMap.GetLength(0);
                        int height = depthMap.GetLength(1);

                        // Find min/max for normalization
                        float minDepth = float.MaxValue;
                        float maxDepth = float.MinValue;
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float d = depthMap[x, y];
                                if (d > 0)
                                {
                                    if (d < minDepth) minDepth = d;
                                    if (d > maxDepth) maxDepth = d;
                                }
                            }
                        }

                        float range = maxDepth - minDepth;
                        if (range < 0.0001f) range = 1.0f;

                        using var bitmap = new SkiaSharp.SKBitmap(width, height, SkiaSharp.SKColorType.Rgba8888, SkiaSharp.SKAlphaType.Premul);
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                float d = depthMap[x, y];
                                if (d <= 0)
                                {
                                    // Transparent background
                                    bitmap.SetPixel(x, y, new SkiaSharp.SKColor(0, 0, 0, 0));
                                }
                                else
                                {
                                    float t = (d - minDepth) / range;
                                    t = Math.Clamp(t, 0f, 1f);

                                    var (r, g, b) = ImageUtils.TurboColormap(t);
                                    bitmap.SetPixel(x, y, new SkiaSharp.SKColor((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255));
                                }
                            }
                        }

                        using var image = SkiaSharp.SKImage.FromBitmap(bitmap);
                        using var data = image.Encode(SkiaSharp.SKEncodedImageFormat.Png, 100);
                        using var stream = File.OpenWrite(outPath);
                        data.SaveTo(stream);

                        exported++;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Failed to export depth map for {img.FileName}: {ex.Message}");
                    }
                }

                _statusLabel.Text = $"Exported {exported} depth maps to {outputDir}";
            }
            fc.Destroy();
        }

        private void OnDeleteSelected(object? sender, EventArgs e)
        {
            foreach (var obj in _sceneGraph.SelectedObjects.ToList())
            {
                _sceneGraph.RemoveObject(obj);
            }
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnDuplicateSelected(object? sender, EventArgs e)
        {
            var toDuplicate = _sceneGraph.SelectedObjects.ToList();
            foreach (var obj in toDuplicate)
            {
                var clone = obj.Clone();
                clone.Position += new OpenTK.Mathematics.Vector3(0.5f, 0, 0);
                _sceneGraph.AddObject(clone, obj.Parent);
            }
            _sceneTreeView.RefreshTree();
            _viewport.QueueDraw();
        }

        private void OnResetTransform(object? sender, EventArgs e)
        {
            foreach (var obj in _sceneGraph.SelectedObjects)
            {
                obj.Position = OpenTK.Mathematics.Vector3.Zero;
                obj.Rotation = OpenTK.Mathematics.Vector3.Zero;
                obj.Scale = OpenTK.Mathematics.Vector3.One;
            }
            _viewport.QueueDraw();
        }

        private void OnFlipNormals(object? sender, EventArgs e)
        {
            foreach (var meshObj in _sceneGraph.SelectedObjects.OfType<MeshObject>())
            {
                meshObj.MeshData = MeshOperations.FlipNormals(meshObj.MeshData);
            }
            _viewport.QueueDraw();
            _statusLabel.Text = "Flipped normals";
        }

        private void OnShowAbout(object? sender, EventArgs e)
        {
            var aboutDialog = new Dialog("About Deep3D Studio", this, DialogFlags.Modal);
            aboutDialog.AddButton("Close", ResponseType.Close);
            aboutDialog.SetDefaultSize(400, 300);

            var contentArea = aboutDialog.ContentArea;
            // Set dark theme
            var black = new Gdk.Color(0, 0, 0);
            contentArea.ModifyBg(StateType.Normal, black);

            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 20;
            vbox.Halign = Align.Center;
            vbox.ModifyBg(StateType.Normal, black);

            // Logo
            try
            {
                var assembly = System.Reflection.Assembly.GetExecutingAssembly();
                using (var stream = assembly.GetManifestResourceStream("Deep3DStudio.logo.png"))
                {
                    if (stream != null)
                    {
                        var pixbuf = new Gdk.Pixbuf(stream);
                        // Scale if too big
                        if (pixbuf.Width > 128)
                        {
                            pixbuf = pixbuf.ScaleSimple(128, 128, Gdk.InterpType.Bilinear);
                        }
                        var image = new Image(pixbuf);
                        vbox.PackStart(image, false, false, 0);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Could not load logo: {ex.Message}");
            }

            var labelTitle = new Label();
            labelTitle.Markup = "<span size='x-large' weight='bold' foreground='#FFFFFF'>Deep3D Studio</span>";
            vbox.PackStart(labelTitle, false, false, 0);

            var labelDesc = new Label("A 3D reconstruction tool using Dust3r and NeRF.\n\nVersion 1.0");
            labelDesc.Justify = Justification.Center;
            labelDesc.ModifyFg(StateType.Normal, new Gdk.Color(220, 220, 220));
            vbox.PackStart(labelDesc, false, false, 0);

            var authorLabel1 = new Label("Matteo Mangiagalli - m.mangiagalli@campus.uniurb.it");
            authorLabel1.Justify = Justification.Center;
            authorLabel1.ModifyFg(StateType.Normal, new Gdk.Color(200, 200, 200));
            vbox.PackStart(authorLabel1, false, false, 0);

            var authorLabel2 = new Label("Università degli Studi di Urbino - Carlo Bo");
            authorLabel2.Justify = Justification.Center;
            authorLabel2.ModifyFg(StateType.Normal, new Gdk.Color(200, 200, 200));
            vbox.PackStart(authorLabel2, false, false, 0);

            var authorLabel3 = new Label("2026");
            authorLabel3.Justify = Justification.Center;
            authorLabel3.ModifyFg(StateType.Normal, new Gdk.Color(200, 200, 200));
            vbox.PackStart(authorLabel3, false, false, 0);

            contentArea.PackStart(vbox, true, true, 0);
            contentArea.ShowAll();

            aboutDialog.Run();
            aboutDialog.Destroy();
        }
    }
}
