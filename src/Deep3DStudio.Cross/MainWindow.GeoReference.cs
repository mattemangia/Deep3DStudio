using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Deep3DStudio.IO;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;
using Deep3DStudio.Viewport;
using ImGuiNET;
using NativeFileDialogs.Net;
using OpenTK.Mathematics;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private bool _showGeoreferenceWindow = false;
        private int _geoImageIndex = -1;
        private string _geoEpsg = "EPSG:4326";
        private bool _geoInputLatLon = false;
        private System.Numerics.Vector2 _geoPixel = System.Numerics.Vector2.Zero;
        private System.Numerics.Vector3 _geoModel = System.Numerics.Vector3.Zero;
        private System.Numerics.Vector3 _geoWorld = System.Numerics.Vector3.Zero;
        private string _geoSelectedGcpId = string.Empty;
        private string _geoStatus = "No solve yet.";
        private float _geoDemCellSize = 0.5f;
        private int _geoPreviewTexture = -1;
        private readonly Dictionary<string, (int w, int h)> _geoImageSizeCache = new Dictionary<string, (int w, int h)>();

        private void RenderGeoreferenceToolbar(float yPos)
        {
            bool shown = BeginStyledToolbarWindow(
                "##GeoreferenceToolbar",
                new System.Numerics.Vector2(_showVerticalToolbar ? _verticalToolbarWidth : 0, yPos),
                new System.Numerics.Vector2(ClientSize.X - (_showVerticalToolbar ? _verticalToolbarWidth : 0), _auxToolbarHeight));
            if (shown)
            {
                var size = ToolbarIconSize;
                DrawToolbarButton("##GeoOpen", IconType.Georef, false, () => _showGeoreferenceWindow = true, "GCP Editor", size);
                ImGui.SameLine();
                DrawToolbarButton("##GeoSolve", IconType.Residuals, false, SolveGeoFromRuntime, "Solve from GCP", size);
                ImGui.SameLine();
                DrawToolbarButton("##GeoDem", IconType.Dem, false, OnExportDemImGui, "Export DEM", size);
                ImGui.SameLine();
                DrawToolbarButton("##GeoExport", IconType.GeoExport, false, OnExportGeoreferencedSelectionImGui, "Export georeferenced selection", size);
                ImGui.SameLine();
                ImGui.TextUnformatted(GeoReferenceRuntime.HasActiveGeoreference
                    ? $"CRS: {GeoReferenceRuntime.GeoReference.ProjectCrsEpsg}"
                    : "CRS: disabled");
            }
            EndStyledToolbarWindow();
        }

        private void DrawGeoreferenceWindow()
        {
            ImGui.SetNextWindowSize(new System.Numerics.Vector2(980, 700), ImGuiCond.FirstUseEver);
            if (!ImGui.Begin("Georeferencing###Geo", ref _showGeoreferenceWindow))
            {
                ImGui.End();
                return;
            }

            if (_geoImageIndex < 0 && _loadedImages.Count > 0)
                _geoImageIndex = 0;

            if (string.IsNullOrWhiteSpace(_geoEpsg))
                _geoEpsg = GeoReferenceRuntime.GeoReference.ProjectCrsEpsg;

            ImGui.InputText("Project CRS (EPSG)", ref _geoEpsg, 64);
            ImGui.SameLine();
            ImGui.Checkbox("Input Lat/Lon", ref _geoInputLatLon);
            ImGui.SameLine();
            if (ImGui.Button("Solve GCP")) SolveGeoFromRuntime();
            ImGui.SameLine();
            if (ImGui.Button("Reset Georef"))
            {
                var geo = GeoReferenceRuntime.GeoReference;
                geo.Enabled = false;
                geo.ModelToWorldMatrix.Clear();
                GeoReferenceRuntime.SetGeoReference(geo);
                _geoStatus = "Georeference reset.";
            }

            ImGui.TextUnformatted(_geoStatus);
            ImGui.Separator();

            if (_loadedImages.Count == 0)
            {
                ImGui.TextUnformatted("No images loaded.");
            }
            else
            {
                string[] imgNames = _loadedImages
                    .Select(i => string.IsNullOrWhiteSpace(i.Alias) ? Path.GetFileName(i.FilePath) : i.Alias)
                    .ToArray();
                int idx = Math.Clamp(_geoImageIndex, 0, imgNames.Length - 1);
                if (ImGui.Combo("Image", ref idx, imgNames, imgNames.Length))
                {
                    _geoImageIndex = idx;
                    LoadGeoPreviewTexture();
                }
                _geoImageIndex = idx;

                if (ImGui.Button("Reload Image"))
                    LoadGeoPreviewTexture();
                ImGui.SameLine();
                if (ImGui.Button("Sample Model Point"))
                    SampleGeoModelPoint();

                if (_geoPreviewTexture <= 0)
                    LoadGeoPreviewTexture();

                if (_geoPreviewTexture > 0 && _geoImageIndex >= 0 && _geoImageIndex < _loadedImages.Count)
                {
                    string imagePath = _loadedImages[_geoImageIndex].FilePath;
                    var size = GetGeoImageSize(imagePath);
                    float maxW = MathF.Max(240, ImGui.GetContentRegionAvail().X);
                    float maxH = 320;
                    float sx = maxW / Math.Max(1, size.w);
                    float sy = maxH / Math.Max(1, size.h);
                    float s = MathF.Min(1.0f, MathF.Min(sx, sy));
                    var drawSize = new System.Numerics.Vector2(size.w * s, size.h * s);
                    ImGui.Image((IntPtr)_geoPreviewTexture, drawSize);

                    if (ImGui.IsItemHovered() && ImGui.IsMouseClicked(ImGuiMouseButton.Left))
                    {
                        var min = ImGui.GetItemRectMin();
                        var mouse = ImGui.GetMousePos();
                        float lx = Math.Clamp(mouse.X - min.X, 0, drawSize.X - 1);
                        float ly = Math.Clamp(mouse.Y - min.Y, 0, drawSize.Y - 1);
                        _geoPixel.X = lx * size.w / drawSize.X;
                        _geoPixel.Y = ly * size.h / drawSize.Y;
                    }
                }
            }

            ImGui.Separator();
            ImGui.InputFloat2("Pixel X/Y", ref _geoPixel);
            ImGui.InputFloat3("Model X/Y/Z", ref _geoModel);
            ImGui.InputFloat3(_geoInputLatLon ? "Lon/Lat/Z" : "World X/Y/Z", ref _geoWorld);
            if (ImGui.Button("Add/Update GCP"))
                AddOrUpdateGeoGcp();
            ImGui.SameLine();
            if (ImGui.Button("Remove Selected GCP"))
                RemoveSelectedGeoGcp();

            ImGui.Separator();
            if (ImGui.BeginTable("##GeoGcps", 8, ImGuiTableFlags.RowBg | ImGuiTableFlags.Borders | ImGuiTableFlags.Resizable | ImGuiTableFlags.ScrollY, new System.Numerics.Vector2(0, 220)))
            {
                ImGui.TableSetupColumn("ID");
                ImGui.TableSetupColumn("Image");
                ImGui.TableSetupColumn("Px");
                ImGui.TableSetupColumn("Py");
                ImGui.TableSetupColumn("Model");
                ImGui.TableSetupColumn("World");
                ImGui.TableSetupColumn("Res");
                ImGui.TableSetupColumn("On");
                ImGui.TableHeadersRow();

                var gcps = GeoReferenceRuntime.Gcps.ToList();
                foreach (var g in gcps)
                {
                    ImGui.TableNextRow();
                    ImGui.TableSetColumnIndex(0);
                    bool selected = _geoSelectedGcpId == g.Id;
                    string shortId = g.Id.Length > 8 ? g.Id.Substring(0, 8) : g.Id;
                    if (ImGui.Selectable(shortId, selected, ImGuiSelectableFlags.SpanAllColumns))
                    {
                        _geoSelectedGcpId = g.Id;
                        _geoPixel = new System.Numerics.Vector2(g.PixelX, g.PixelY);
                        _geoModel = new System.Numerics.Vector3(g.ModelPoint.X, g.ModelPoint.Y, g.ModelPoint.Z);
                        _geoWorld = new System.Numerics.Vector3((float)g.InputLonOrX, (float)g.InputLatOrY, (float)g.InputZ);
                        _geoInputLatLon = g.InputIsLatLon;
                    }

                    ImGui.TableSetColumnIndex(1); ImGui.TextUnformatted(Path.GetFileName(g.ImagePath));
                    ImGui.TableSetColumnIndex(2); ImGui.Text($"{g.PixelX:F1}");
                    ImGui.TableSetColumnIndex(3); ImGui.Text($"{g.PixelY:F1}");
                    ImGui.TableSetColumnIndex(4); ImGui.Text($"{g.ModelPoint.X:F2},{g.ModelPoint.Y:F2},{g.ModelPoint.Z:F2}");
                    ImGui.TableSetColumnIndex(5); ImGui.Text($"{g.WorldPoint.X:F2},{g.WorldPoint.Y:F2},{g.WorldPoint.Z:F2}");
                    ImGui.TableSetColumnIndex(6); ImGui.Text($"{g.Residual:F4}");
                    ImGui.TableSetColumnIndex(7);
                    bool enabled = g.Enabled;
                    if (ImGui.Checkbox($"##g{g.Id}", ref enabled))
                    {
                        g.Enabled = enabled;
                        GeoReferenceRuntime.AddOrUpdateGcp(g);
                    }
                }
                ImGui.EndTable();
            }

            ImGui.Separator();
            ImGui.InputFloat("DEM Cell Size", ref _geoDemCellSize);
            ImGui.SameLine();
            if (ImGui.Button("Export DEM")) OnExportDemImGui();
            ImGui.SameLine();
            if (ImGui.Button("Export Georeferenced")) OnExportGeoreferencedSelectionImGui();

            ImGui.End();

            if (!_showGeoreferenceWindow && _geoPreviewTexture > 0)
            {
                TextureLoader.DeleteTexture(_geoPreviewTexture);
                _geoPreviewTexture = -1;
            }
        }

        private void LoadGeoPreviewTexture()
        {
            if (_geoImageIndex < 0 || _geoImageIndex >= _loadedImages.Count)
                return;
            string path = _loadedImages[_geoImageIndex].FilePath;
            if (!File.Exists(path))
                return;

            if (_geoPreviewTexture > 0)
                TextureLoader.DeleteTexture(_geoPreviewTexture);
            _geoPreviewTexture = TextureLoader.LoadTextureFromFile(path);
        }

        private void SampleGeoModelPoint()
        {
            if (_geoImageIndex < 0 || _geoImageIndex >= _loadedImages.Count)
                return;
            string imagePath = _loadedImages[_geoImageIndex].FilePath;
            if (GeoReferenceService.TryPickModelPointFromImagePixel(_sceneGraph, imagePath, _geoPixel.X, _geoPixel.Y, out Vector3 modelPoint, out string error))
            {
                _geoModel = new System.Numerics.Vector3(modelPoint.X, modelPoint.Y, modelPoint.Z);
                _geoStatus = $"Sample OK: ({modelPoint.X:F3}, {modelPoint.Y:F3}, {modelPoint.Z:F3})";
            }
            else
            {
                _geoStatus = error;
            }
        }

        private void AddOrUpdateGeoGcp()
        {
            if (_geoImageIndex < 0 || _geoImageIndex >= _loadedImages.Count)
            {
                _geoStatus = "Select an image first.";
                return;
            }

            string imagePath = _loadedImages[_geoImageIndex].FilePath;
            if (!GeoReferenceService.TryNormalizeInputCoordinate(
                _geoEpsg,
                _geoInputLatLon,
                _geoWorld.X,
                _geoWorld.Y,
                _geoWorld.Z,
                out Vector3 worldPoint,
                out string error))
            {
                _geoStatus = error;
                return;
            }

            string id = string.IsNullOrWhiteSpace(_geoSelectedGcpId) ? Guid.NewGuid().ToString("N") : _geoSelectedGcpId;
            var gcp = new GcpEntryDTO
            {
                Id = id,
                ImagePath = imagePath,
                PixelX = _geoPixel.X,
                PixelY = _geoPixel.Y,
                InputIsLatLon = _geoInputLatLon,
                InputLonOrX = _geoWorld.X,
                InputLatOrY = _geoWorld.Y,
                InputZ = _geoWorld.Z,
                ModelPoint = new Vector3(_geoModel.X, _geoModel.Y, _geoModel.Z),
                WorldPoint = worldPoint,
                Enabled = true
            };

            GeoReferenceRuntime.AddOrUpdateGcp(gcp);
            var geo = GeoReferenceRuntime.GeoReference;
            geo.ProjectCrsEpsg = string.IsNullOrWhiteSpace(_geoEpsg) ? "EPSG:4326" : _geoEpsg.Trim();
            GeoReferenceRuntime.SetGeoReference(geo);
            _isDirty = true;
            UpdateTitle();
            _geoStatus = "GCP saved.";
        }

        private void RemoveSelectedGeoGcp()
        {
            if (string.IsNullOrWhiteSpace(_geoSelectedGcpId))
                return;
            GeoReferenceRuntime.RemoveGcp(_geoSelectedGcpId);
            _geoSelectedGcpId = string.Empty;
            _isDirty = true;
            UpdateTitle();
            _geoStatus = "GCP removed.";
        }

        private void SolveGeoFromRuntime()
        {
            var gcps = GeoReferenceRuntime.Gcps.ToList();
            if (!gcps.Any())
            {
                _geoStatus = "No GCP defined.";
                return;
            }

            var geo = GeoReferenceRuntime.GeoReference;
            geo.ProjectCrsEpsg = string.IsNullOrWhiteSpace(_geoEpsg) ? "EPSG:4326" : _geoEpsg.Trim();
            GeoReferenceRuntime.SetGeoReference(geo);

            if (!GeoReferenceService.TrySolveModelToWorldFromGcps(gcps, out Matrix4 m, out double rms, out string error))
            {
                _geoStatus = error;
                return;
            }

            GeoReferenceRuntime.SetModelToWorldMatrix(m);
            GeoReferenceRuntime.SetGcps(gcps);
            _isDirty = true;
            UpdateTitle();
            _geoStatus = $"Solved. RMS {rms:F6}";
        }

        private void OnExportDemImGui()
        {
            var target = _sceneGraph.SelectedObjects.FirstOrDefault(o => o is MeshObject || o is PointCloudObject);
            if (target == null)
            {
                _geoStatus = "Select a mesh or point cloud first.";
                return;
            }

            var saveResult = Nfd.SaveDialog(out string path, new Dictionary<string, string>
            {
                { "GeoTIFF", "tif,tiff" }
            });
            if (saveResult != NfdStatus.Ok || string.IsNullOrEmpty(path))
                return;

            if (!path.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) &&
                !path.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
            {
                path += ".tif";
            }

            try
            {
                if (!DemGeneratorService.TryCollectWorldPoints(target, applyGeoref: true, out var points, out string err))
                {
                    _geoStatus = err;
                    return;
                }

                double cell = Math.Max(1e-6, _geoDemCellSize);
                var dem = DemGeneratorService.GenerateDemFromPoints(points, cell);
                string ascPath = Path.ChangeExtension(path, ".asc");
                string epsg = GeoReferenceRuntime.HasActiveGeoreference
                    ? GeoReferenceRuntime.GeoReference.ProjectCrsEpsg
                    : "EPSG:4326";
                DemGeneratorService.SaveGeoTiff(path, dem, epsg);
                DemGeneratorService.SaveAsciiGrid(ascPath, dem);
                _geoStatus = $"DEM exported: {Path.GetFileName(path)} + {Path.GetFileName(ascPath)}";
                _logBuffer += $"DEM exported: {path}\n";
            }
            catch (Exception ex)
            {
                _geoStatus = $"DEM export failed: {ex.Message}";
            }
        }

        private void OnExportGeoreferencedSelectionImGui()
        {
            var selected = _sceneGraph.SelectedObjects.Where(o => o is MeshObject || o is PointCloudObject).ToList();
            if (selected.Count == 0)
            {
                _geoStatus = "Select at least one mesh/point cloud.";
                return;
            }

            var saveResult = Nfd.SaveDialog(out string path, new Dictionary<string, string>
            {
                { "OBJ Mesh", "obj" },
                { "PLY Point Cloud", "ply" }
            });
            if (saveResult != NfdStatus.Ok || string.IsNullOrEmpty(path))
                return;

            string dir = Path.GetDirectoryName(path) ?? "";
            string baseName = Path.GetFileNameWithoutExtension(path);
            int exported = 0;
            foreach (var obj in selected)
            {
                try
                {
                    if (obj is MeshObject meshObj)
                    {
                        var mesh = GeoExportService.PrepareMeshForExport(meshObj);
                        string outPath = Path.Combine(dir, $"{baseName}_{SanitizeFileName(meshObj.Name)}.obj");
                        MeshExporter.Save(outPath, mesh);
                        GeoExportService.TryWriteGeoSidecar(outPath, mesh.Vertices);
                        exported++;
                    }
                    else if (obj is PointCloudObject pcObj)
                    {
                        var pc = GeoExportService.PreparePointCloudForExport(pcObj);
                        string outPath = Path.Combine(dir, $"{baseName}_{SanitizeFileName(pcObj.Name)}.ply");
                        PointCloudExporter.Export(outPath, pc, PointCloudExporter.ExportFormat.PLY, true);
                        GeoExportService.TryWriteGeoSidecar(outPath, pc.Points);
                        exported++;
                    }
                }
                catch (Exception ex)
                {
                    _logBuffer += $"Geo export failed for {obj.Name}: {ex.Message}\n";
                }
            }
            _geoStatus = $"Georeferenced export completed ({exported}).";
        }

        private (int w, int h) GetGeoImageSize(string path)
        {
            if (_geoImageSizeCache.TryGetValue(path, out var size))
                return size;
            try
            {
                using var bmp = ImageDecoder.DecodeBitmap(path);
                size = (bmp.Width, bmp.Height);
            }
            catch
            {
                size = (1024, 768);
            }
            _geoImageSizeCache[path] = size;
            return size;
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return string.IsNullOrWhiteSpace(name) ? "object" : name;
        }
    }
}
