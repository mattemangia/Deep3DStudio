using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Deep3DStudio.IO;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;
using Deep3DStudio.UI;
using Gtk;

namespace Deep3DStudio
{
    public partial class MainWindow
    {
        private void OnOpenGeoreferenceEditor()
        {
            var images = _imageBrowser.GetImages()
                .Where(i => !string.IsNullOrWhiteSpace(i.FilePath))
                .Select(i => new ProjectImage
                {
                    FilePath = i.FilePath,
                    Alias = string.IsNullOrWhiteSpace(i.DisplayName) ? System.IO.Path.GetFileName(i.FilePath) : i.DisplayName
                })
                .ToList();

            using var dlg = new GeoReferenceDialog(this, _sceneGraph, images);
            dlg.Run();
            dlg.Hide();

            _isDirty = true;
            UpdateTitle();
            _statusLabel.Text = GeoReferenceRuntime.HasActiveGeoreference
                ? $"Georeferenziazione attiva ({GeoReferenceRuntime.GeoReference.ProjectCrsEpsg})"
                : "Georeferenziazione aggiornata";
        }

        private void OnSolveGeoreferenceFromCurrentGcps()
        {
            var geo = GeoReferenceRuntime.GeoReference;
            string epsg = string.IsNullOrWhiteSpace(geo.ProjectCrsEpsg) ? "EPSG:4326" : geo.ProjectCrsEpsg.Trim();
            var pending = GeoReferenceRuntime.PendingGcps.Select(ClonePendingGcp).ToList();
            var gcps = GeoReferenceRuntime.Gcps.Select(CloneGcp).ToList();
            var resolve = GeoReferenceService.ResolvePendingGcpsFromScene(_sceneGraph, pending, gcps, epsg);
            GeoReferenceRuntime.SetPendingGcps(pending);
            GeoReferenceRuntime.SetGcps(gcps);

            if (!gcps.Any())
            {
                ShowMessage("Nessun GCP disponibile. Apri l'editor GCP.");
                return;
            }

            if (!GeoReferenceService.TrySolveModelToWorldFromGcps(gcps, out var m, out double rms, out string error))
            {
                ShowMessage(error);
                return;
            }

            GeoReferenceRuntime.SetModelToWorldMatrix(m);
            geo.Enabled = true;
            geo.ProjectCrsEpsg = epsg;
            GeoReferenceRuntime.SetGeoReference(geo);
            GeoReferenceRuntime.SetGcps(gcps);
            _isDirty = true;
            UpdateTitle();
            _statusLabel.Text = $"Georeferenziazione aggiornata. RMS {rms:F6}. Pending risolti: {resolve.ResolvedCount}, falliti: {resolve.FailedCount}";
        }

        private void OnResolvePendingGcpsFromCurrentScene()
        {
            var geo = GeoReferenceRuntime.GeoReference;
            string epsg = string.IsNullOrWhiteSpace(geo.ProjectCrsEpsg) ? "EPSG:4326" : geo.ProjectCrsEpsg.Trim();
            var pending = GeoReferenceRuntime.PendingGcps.Select(ClonePendingGcp).ToList();
            var gcps = GeoReferenceRuntime.Gcps.Select(CloneGcp).ToList();
            var summary = GeoReferenceService.ResolvePendingGcpsFromScene(_sceneGraph, pending, gcps, epsg);
            GeoReferenceRuntime.SetPendingGcps(pending);
            GeoReferenceRuntime.SetGcps(gcps);
            _isDirty = true;
            UpdateTitle();
            _statusLabel.Text = $"Pending GCP risolti: {summary.ResolvedCount}, falliti: {summary.FailedCount}";
        }

        private void TryAutoRefineGeoreferenceFromScene(string reason)
        {
            bool hasContext = GeoReferenceRuntime.Gcps.Count > 0 || GeoReferenceRuntime.PendingGcps.Count > 0;
            if (!hasContext)
                return;

            var summary = GeoReferenceService.TryAutoRefineGeoreferenceAfterGeometryChange(_sceneGraph, keepPreviousTransformOnFailure: true);
            _isDirty = true;
            UpdateTitle();
            _statusLabel.Text = $"Auto-georef ({reason}): {summary.Message} Pending resolved={summary.PendingResolved}, failed={summary.PendingFailed}";
        }

        private void OnExportDem(object? sender, EventArgs e)
        {
            var target = _sceneGraph.SelectedObjects.FirstOrDefault(o => o is MeshObject || o is PointCloudObject);
            if (target == null)
            {
                ShowMessage("Seleziona una Mesh o Point Cloud per generare DEM.");
                return;
            }

            if (!TryPromptDouble("Risoluzione DEM", "Cell size (unita' CRS):", "0.5", out double cellSize))
                return;

            var fc = new FileChooserDialog("Export DEM (GeoTIFF)", this, FileChooserAction.Save,
                "Cancel", ResponseType.Cancel, "Save", ResponseType.Accept);
            var tifFilter = new FileFilter { Name = "GeoTIFF (*.tif)" };
            tifFilter.AddPattern("*.tif");
            tifFilter.AddPattern("*.tiff");
            fc.AddFilter(tifFilter);
            fc.CurrentName = "dem.tif";

            if (fc.Run() == (int)ResponseType.Accept)
            {
                try
                {
                    string tifPath = fc.Filename;
                    if (!tifPath.EndsWith(".tif", StringComparison.OrdinalIgnoreCase) &&
                        !tifPath.EndsWith(".tiff", StringComparison.OrdinalIgnoreCase))
                    {
                        tifPath += ".tif";
                    }

                    if (!DemGeneratorService.TryCollectWorldPoints(target, applyGeoref: true, out var points, out string err))
                    {
                        ShowMessage(err);
                        return;
                    }

                    var dem = DemGeneratorService.GenerateDemFromPoints(points, cellSize);
                    string ascPath = System.IO.Path.ChangeExtension(tifPath, ".asc");
                    string epsg = GeoReferenceRuntime.HasActiveGeoreference
                        ? GeoReferenceRuntime.GeoReference.ProjectCrsEpsg
                        : "EPSG:4326";

                    DemGeneratorService.SaveGeoTiff(tifPath, dem, epsg);
                    DemGeneratorService.SaveAsciiGrid(ascPath, dem);

                    _statusLabel.Text = $"DEM esportato: {System.IO.Path.GetFileName(tifPath)} + {System.IO.Path.GetFileName(ascPath)}";
                }
                catch (Exception ex)
                {
                    ShowMessage($"Errore export DEM: {ex.Message}");
                }
            }
            fc.Destroy();
        }

        private void OnExportGeoreferencedSelection()
        {
            var selected = _sceneGraph.SelectedObjects.Where(o => o is MeshObject || o is PointCloudObject).ToList();
            if (selected.Count == 0)
            {
                ShowMessage("Seleziona almeno una Mesh o Point Cloud.");
                return;
            }

            var fc = new FileChooserDialog("Scegli cartella export georeferenziato", this, FileChooserAction.SelectFolder,
                "Cancel", ResponseType.Cancel, "Select", ResponseType.Accept);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                string folder = fc.Filename;
                int count = 0;
                foreach (var obj in selected)
                {
                    try
                    {
                        if (obj is MeshObject meshObj)
                        {
                            var mesh = GeoExportService.PrepareMeshForExport(meshObj);
                            string outPath = System.IO.Path.Combine(folder, $"{SanitizeFileName(meshObj.Name)}_geo.obj");
                            MeshExporter.Save(outPath, mesh);
                            GeoExportService.TryWriteGeoSidecar(outPath, mesh.Vertices);
                            count++;
                        }
                        else if (obj is PointCloudObject pcObj)
                        {
                            var pc = GeoExportService.PreparePointCloudForExport(pcObj);
                            string outPath = System.IO.Path.Combine(folder, $"{SanitizeFileName(pcObj.Name)}_geo.ply");
                            PointCloudExporter.Export(outPath, pc, PointCloudExporter.ExportFormat.PLY, true);
                            GeoExportService.TryWriteGeoSidecar(outPath, pc.Points);
                            count++;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Geo-export failed for {obj.Name}: {ex.Message}");
                    }
                }
                _statusLabel.Text = $"Export georeferenziato completato ({count} oggetti).";
            }
            fc.Destroy();
        }

        private static string SanitizeFileName(string name)
        {
            foreach (char c in System.IO.Path.GetInvalidFileNameChars())
                name = name.Replace(c, '_');
            return string.IsNullOrWhiteSpace(name) ? "object" : name;
        }

        private static GcpEntryDTO CloneGcp(GcpEntryDTO g)
        {
            return new GcpEntryDTO
            {
                Id = g.Id,
                ImagePath = g.ImagePath,
                PixelX = g.PixelX,
                PixelY = g.PixelY,
                InputIsLatLon = g.InputIsLatLon,
                InputLonOrX = g.InputLonOrX,
                InputLatOrY = g.InputLatOrY,
                InputZ = g.InputZ,
                ModelPoint = g.ModelPoint,
                WorldPoint = g.WorldPoint,
                Residual = g.Residual,
                Enabled = g.Enabled
            };
        }

        private static PendingGcpEntryDTO ClonePendingGcp(PendingGcpEntryDTO g)
        {
            return new PendingGcpEntryDTO
            {
                Id = g.Id,
                ImagePath = g.ImagePath,
                PixelX = g.PixelX,
                PixelY = g.PixelY,
                InputIsLatLon = g.InputIsLatLon,
                InputLonOrX = g.InputLonOrX,
                InputLatOrY = g.InputLatOrY,
                InputZ = g.InputZ,
                WorldPoint = g.WorldPoint,
                Enabled = g.Enabled,
                Status = g.Status,
                LastError = g.LastError
            };
        }

        private bool TryPromptDouble(string title, string prompt, string defaultValue, out double value)
        {
            value = 0;
            using var dlg = new Dialog(title, this, DialogFlags.Modal);
            dlg.AddButton("Cancel", ResponseType.Cancel);
            dlg.AddButton("OK", ResponseType.Ok);
            dlg.SetDefaultSize(360, 120);

            var box = dlg.ContentArea;
            var grid = new Grid { ColumnSpacing = 8, RowSpacing = 8, Margin = 10 };
            var entry = new Entry(defaultValue);
            grid.Attach(new Label(prompt) { Halign = Align.Start }, 0, 0, 1, 1);
            grid.Attach(entry, 1, 0, 1, 1);
            box.PackStart(grid, true, true, 0);
            dlg.ShowAll();

            if (dlg.Run() == (int)ResponseType.Ok &&
                double.TryParse(entry.Text, NumberStyles.Float, CultureInfo.InvariantCulture, out value) &&
                value > 0)
            {
                return true;
            }
            return false;
        }
    }
}
