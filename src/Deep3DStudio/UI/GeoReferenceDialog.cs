using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Deep3DStudio.Model;
using Deep3DStudio.Scene;
using Gdk;
using Gtk;
using OpenTK.Mathematics;

namespace Deep3DStudio.UI
{
    public class GeoReferenceDialog : Dialog
    {
        private readonly SceneGraph _sceneGraph;
        private readonly List<ProjectImage> _images;
        private readonly List<GcpEntryDTO> _workingGcps = new List<GcpEntryDTO>();

        private readonly ComboBoxText _imageCombo = new ComboBoxText();
        private readonly Entry _epsgEntry = new Entry();
        private readonly CheckButton _inputLatLonCheck = new CheckButton("Input Lat/Lon");
        private readonly Entry _pixelXEntry = new Entry();
        private readonly Entry _pixelYEntry = new Entry();
        private readonly Entry _modelXEntry = new Entry();
        private readonly Entry _modelYEntry = new Entry();
        private readonly Entry _modelZEntry = new Entry();
        private readonly Entry _worldXEntry = new Entry();
        private readonly Entry _worldYEntry = new Entry();
        private readonly Entry _worldZEntry = new Entry();
        private readonly Label _statsLabel = new Label("Nessun solve eseguito.");

        private readonly Gtk.Image _imageWidget = new Gtk.Image();
        private readonly EventBox _imageClickBox = new EventBox();
        private readonly ListStore _gcpStore = new ListStore(
            typeof(string), typeof(string), typeof(string), typeof(string), typeof(string), typeof(string), typeof(string), typeof(bool));
        private readonly TreeView _gcpTree = new TreeView();

        private Pixbuf? _fullPixbuf;
        private Pixbuf? _displayPixbuf;
        private int _sourceWidth;
        private int _sourceHeight;
        private int _displayWidth;
        private int _displayHeight;
        private string? _selectedGcpId;

        public GeoReferenceDialog(Gtk.Window parent, SceneGraph sceneGraph, List<ProjectImage> images)
            : base("Georeferenziazione (GCP)", parent, DialogFlags.Modal | DialogFlags.DestroyWithParent)
        {
            _sceneGraph = sceneGraph;
            _images = images ?? new List<ProjectImage>();
            _workingGcps.AddRange(GeoReferenceRuntime.Gcps.Select(CloneGcp));

            SetDefaultSize(1100, 760);
            AddButton("Chiudi", ResponseType.Close);

            BuildUi();
            LoadFromRuntime();
            RefreshGcpTable();
            ShowAll();
        }

        private void BuildUi()
        {
            var root = new Box(Orientation.Vertical, 8) { Margin = 8 };
            ContentArea.PackStart(root, true, true, 0);

            var topRow = new Box(Orientation.Horizontal, 8);
            topRow.PackStart(new Label("CRS (EPSG):"), false, false, 0);
            _epsgEntry.WidthChars = 16;
            topRow.PackStart(_epsgEntry, false, false, 0);
            topRow.PackStart(_inputLatLonCheck, false, false, 0);
            var solveBtn = new Button("Solve GCP");
            solveBtn.Clicked += (s, e) => SolveAndApply();
            topRow.PackStart(solveBtn, false, false, 0);
            var clearBtn = new Button("Reset Georef");
            clearBtn.Clicked += (s, e) => ResetGeoreference();
            topRow.PackStart(clearBtn, false, false, 0);
            topRow.PackStart(_statsLabel, true, true, 0);
            root.PackStart(topRow, false, false, 0);

            var hSplit = new Paned(Orientation.Horizontal);
            root.PackStart(hSplit, true, true, 0);

            var left = new Box(Orientation.Vertical, 6) { Margin = 4 };
            hSplit.Pack1(left, true, false);

            var imageRow = new Box(Orientation.Horizontal, 6);
            imageRow.PackStart(new Label("Immagine:"), false, false, 0);
            foreach (var img in _images)
                _imageCombo.AppendText(string.IsNullOrWhiteSpace(img.Alias) ? System.IO.Path.GetFileName(img.FilePath) : img.Alias);
            _imageCombo.Changed += (s, e) => LoadSelectedImage();
            _imageCombo.Active = _images.Count > 0 ? 0 : -1;
            imageRow.PackStart(_imageCombo, true, true, 0);
            var pickBtn = new Button("Campiona punto 3D da click");
            pickBtn.Clicked += (s, e) => SampleModelPointFromCurrentPixel();
            imageRow.PackStart(pickBtn, false, false, 0);
            left.PackStart(imageRow, false, false, 0);

            _imageClickBox.Add(_imageWidget);
            _imageClickBox.ButtonPressEvent += OnImageClicked;
            var imageScroll = new ScrolledWindow();
            imageScroll.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            imageScroll.AddWithViewport(_imageClickBox);
            left.PackStart(imageScroll, true, true, 0);

            var form = new Grid
            {
                ColumnSpacing = 8,
                RowSpacing = 6,
                MarginTop = 6
            };
            left.PackStart(form, false, false, 0);

            int row = 0;
            AddFormRow(form, row++, "Pixel X", _pixelXEntry, "Pixel Y", _pixelYEntry);
            AddFormRow(form, row++, "Model X", _modelXEntry, "Model Y", _modelYEntry);
            AddFormRow(form, row++, "Model Z", _modelZEntry, "World X/Lon", _worldXEntry);
            AddFormRow(form, row++, "World Y/Lat", _worldYEntry, "World Z", _worldZEntry);

            var formButtons = new Box(Orientation.Horizontal, 6);
            var addBtn = new Button("Aggiungi/Update GCP");
            addBtn.Clicked += (s, e) => AddOrUpdateGcp();
            formButtons.PackStart(addBtn, false, false, 0);
            var removeBtn = new Button("Rimuovi selezionato");
            removeBtn.Clicked += (s, e) => RemoveSelectedGcp();
            formButtons.PackStart(removeBtn, false, false, 0);
            var clearFieldsBtn = new Button("Pulisci campi");
            clearFieldsBtn.Clicked += (s, e) => ClearInputFields();
            formButtons.PackStart(clearFieldsBtn, false, false, 0);
            left.PackStart(formButtons, false, false, 0);

            var right = new Box(Orientation.Vertical, 6) { Margin = 4 };
            hSplit.Pack2(right, true, false);
            right.PackStart(new Label("Ground Control Points"), false, false, 0);

            _gcpTree.Model = _gcpStore;
            _gcpTree.HeadersVisible = true;
            AddColumn(_gcpTree, "ID", 0, 90);
            AddColumn(_gcpTree, "Image", 1, 130);
            AddColumn(_gcpTree, "Px", 2, 75);
            AddColumn(_gcpTree, "Py", 3, 75);
            AddColumn(_gcpTree, "Model XYZ", 4, 170);
            AddColumn(_gcpTree, "World XYZ", 5, 190);
            AddColumn(_gcpTree, "Residual", 6, 75);
            var enabledCol = new TreeViewColumn { Title = "On" };
            var toggle = new CellRendererToggle();
            enabledCol.PackStart(toggle, true);
            enabledCol.AddAttribute(toggle, "active", 7);
            _gcpTree.AppendColumn(enabledCol);

            _gcpTree.Selection.Changed += (s, e) => LoadSelectedGcpToFields();
            toggle.Toggled += OnGcpToggleEnabled;

            var gcpScroll = new ScrolledWindow();
            gcpScroll.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            gcpScroll.Add(_gcpTree);
            right.PackStart(gcpScroll, true, true, 0);

            hSplit.Position = 680;
        }

        private void LoadFromRuntime()
        {
            _epsgEntry.Text = string.IsNullOrWhiteSpace(GeoReferenceRuntime.GeoReference.ProjectCrsEpsg)
                ? "EPSG:4326"
                : GeoReferenceRuntime.GeoReference.ProjectCrsEpsg;
            _inputLatLonCheck.Active = false;
            _statsLabel.Text = GeoReferenceService.FormatResidualStats(_workingGcps);
            LoadSelectedImage();
        }

        private void LoadSelectedImage()
        {
            DisposePixbufs();
            int idx = _imageCombo.Active;
            if (idx < 0 || idx >= _images.Count)
                return;

            string path = _images[idx].FilePath;
            if (!System.IO.File.Exists(path))
                return;

            try
            {
                _fullPixbuf = new Pixbuf(path);
                _sourceWidth = _fullPixbuf.Width;
                _sourceHeight = _fullPixbuf.Height;

                const int maxW = 700;
                const int maxH = 460;
                double sx = maxW / (double)_sourceWidth;
                double sy = maxH / (double)_sourceHeight;
                double s = Math.Min(1.0, Math.Min(sx, sy));
                _displayWidth = Math.Max(1, (int)Math.Round(_sourceWidth * s));
                _displayHeight = Math.Max(1, (int)Math.Round(_sourceHeight * s));
                _displayPixbuf = _fullPixbuf.ScaleSimple(_displayWidth, _displayHeight, InterpType.Bilinear);
                _imageWidget.Pixbuf = _displayPixbuf;
                _imageWidget.SetSizeRequest(_displayWidth, _displayHeight);
            }
            catch
            {
                DisposePixbufs();
            }
        }

        private void OnImageClicked(object o, ButtonPressEventArgs args)
        {
            if (_displayWidth <= 0 || _displayHeight <= 0 || _sourceWidth <= 0 || _sourceHeight <= 0)
                return;

            float localX = (float)Math.Clamp(args.Event.X, 0, _displayWidth - 1);
            float localY = (float)Math.Clamp(args.Event.Y, 0, _displayHeight - 1);
            float srcX = localX * _sourceWidth / _displayWidth;
            float srcY = localY * _sourceHeight / _displayHeight;
            _pixelXEntry.Text = srcX.ToString("F2", CultureInfo.InvariantCulture);
            _pixelYEntry.Text = srcY.ToString("F2", CultureInfo.InvariantCulture);
        }

        private void SampleModelPointFromCurrentPixel()
        {
            if (!TryReadFloat(_pixelXEntry.Text, out float px) || !TryReadFloat(_pixelYEntry.Text, out float py))
            {
                _statsLabel.Text = "Pixel non validi.";
                return;
            }

            int idx = _imageCombo.Active;
            if (idx < 0 || idx >= _images.Count)
            {
                _statsLabel.Text = "Seleziona un'immagine.";
                return;
            }

            string imagePath = _images[idx].FilePath;
            if (GeoReferenceService.TryPickModelPointFromImagePixel(_sceneGraph, imagePath, px, py, out Vector3 modelPoint, out string error))
            {
                _modelXEntry.Text = modelPoint.X.ToString("F6", CultureInfo.InvariantCulture);
                _modelYEntry.Text = modelPoint.Y.ToString("F6", CultureInfo.InvariantCulture);
                _modelZEntry.Text = modelPoint.Z.ToString("F6", CultureInfo.InvariantCulture);
                _statsLabel.Text = $"Punto modello campionato: ({modelPoint.X:F3}, {modelPoint.Y:F3}, {modelPoint.Z:F3})";
            }
            else
            {
                _statsLabel.Text = error;
            }
        }

        private void AddOrUpdateGcp()
        {
            int imageIdx = _imageCombo.Active;
            if (imageIdx < 0 || imageIdx >= _images.Count)
            {
                _statsLabel.Text = "Seleziona un'immagine.";
                return;
            }

            if (!TryReadFloat(_pixelXEntry.Text, out float px) ||
                !TryReadFloat(_pixelYEntry.Text, out float py) ||
                !TryReadFloat(_modelXEntry.Text, out float mx) ||
                !TryReadFloat(_modelYEntry.Text, out float my) ||
                !TryReadFloat(_modelZEntry.Text, out float mz) ||
                !TryReadDouble(_worldXEntry.Text, out double wxInput) ||
                !TryReadDouble(_worldYEntry.Text, out double wyInput) ||
                !TryReadDouble(_worldZEntry.Text, out double wzInput))
            {
                _statsLabel.Text = "Compila tutti i campi numerici.";
                return;
            }

            string epsg = string.IsNullOrWhiteSpace(_epsgEntry.Text) ? "EPSG:4326" : _epsgEntry.Text.Trim();
            bool latLon = _inputLatLonCheck.Active;
            if (!GeoReferenceService.TryNormalizeInputCoordinate(epsg, latLon, wxInput, wyInput, wzInput, out Vector3 worldPoint, out string err))
            {
                _statsLabel.Text = err;
                return;
            }

            var gcp = new GcpEntryDTO
            {
                Id = _selectedGcpId ?? Guid.NewGuid().ToString("N"),
                ImagePath = _images[imageIdx].FilePath,
                PixelX = px,
                PixelY = py,
                InputIsLatLon = latLon,
                InputLonOrX = wxInput,
                InputLatOrY = wyInput,
                InputZ = wzInput,
                ModelPoint = new Vector3(mx, my, mz),
                WorldPoint = worldPoint,
                Enabled = true
            };

            int existing = _workingGcps.FindIndex(g => g.Id == gcp.Id);
            if (existing >= 0)
                _workingGcps[existing] = gcp;
            else
                _workingGcps.Add(gcp);

            RefreshGcpTable();
            SyncRuntime();
            _statsLabel.Text = $"GCP {(existing >= 0 ? "aggiornato" : "aggiunto")}.";
            ClearInputFields();
        }

        private void RemoveSelectedGcp()
        {
            if (string.IsNullOrWhiteSpace(_selectedGcpId))
                return;
            _workingGcps.RemoveAll(g => g.Id == _selectedGcpId);
            _selectedGcpId = null;
            RefreshGcpTable();
            SyncRuntime();
        }

        private void SolveAndApply()
        {
            string epsg = string.IsNullOrWhiteSpace(_epsgEntry.Text) ? "EPSG:4326" : _epsgEntry.Text.Trim();
            var geo = GeoReferenceRuntime.GeoReference;
            geo.ProjectCrsEpsg = epsg;
            GeoReferenceRuntime.SetGeoReference(geo);
            GeoReferenceRuntime.SetGcps(_workingGcps);

            if (!GeoReferenceService.TrySolveModelToWorldFromGcps(_workingGcps, out Matrix4 m, out double rms, out string error))
            {
                _statsLabel.Text = error;
                return;
            }

            GeoReferenceRuntime.SetModelToWorldMatrix(m);
            var newGeo = GeoReferenceRuntime.GeoReference;
            newGeo.Enabled = true;
            newGeo.ProjectCrsEpsg = epsg;
            GeoReferenceRuntime.SetGeoReference(newGeo);
            GeoReferenceService.UpdateResiduals(_workingGcps, m, out _);
            SyncRuntime();
            RefreshGcpTable();
            _statsLabel.Text = $"Solve completato. RMS: {rms:F6}";
        }

        private void ResetGeoreference()
        {
            var geo = GeoReferenceRuntime.GeoReference;
            geo.Enabled = false;
            geo.ModelToWorldMatrix.Clear();
            GeoReferenceRuntime.SetGeoReference(geo);
            foreach (var g in _workingGcps) g.Residual = 0;
            SyncRuntime();
            RefreshGcpTable();
            _statsLabel.Text = "Georeferenziazione disattivata.";
        }

        private void OnGcpToggleEnabled(object o, ToggledArgs args)
        {
            if (_gcpStore.GetIterFromString(out TreeIter iter, args.Path))
            {
                string id = (string)_gcpStore.GetValue(iter, 0);
                var g = _workingGcps.FirstOrDefault(x => x.Id == id);
                if (g != null)
                {
                    g.Enabled = !g.Enabled;
                    RefreshGcpTable();
                    SyncRuntime();
                }
            }
        }

        private void LoadSelectedGcpToFields()
        {
            if (!_gcpTree.Selection.GetSelected(out TreeIter iter))
                return;

            string id = (string)_gcpStore.GetValue(iter, 0);
            var g = _workingGcps.FirstOrDefault(x => x.Id == id);
            if (g == null)
                return;

            _selectedGcpId = g.Id;
            int imgIndex = _images.FindIndex(i => SamePath(i.FilePath, g.ImagePath));
            if (imgIndex >= 0)
                _imageCombo.Active = imgIndex;

            _pixelXEntry.Text = g.PixelX.ToString("F2", CultureInfo.InvariantCulture);
            _pixelYEntry.Text = g.PixelY.ToString("F2", CultureInfo.InvariantCulture);
            _modelXEntry.Text = g.ModelPoint.X.ToString("F6", CultureInfo.InvariantCulture);
            _modelYEntry.Text = g.ModelPoint.Y.ToString("F6", CultureInfo.InvariantCulture);
            _modelZEntry.Text = g.ModelPoint.Z.ToString("F6", CultureInfo.InvariantCulture);
            _worldXEntry.Text = g.InputLonOrX.ToString("G17", CultureInfo.InvariantCulture);
            _worldYEntry.Text = g.InputLatOrY.ToString("G17", CultureInfo.InvariantCulture);
            _worldZEntry.Text = g.InputZ.ToString("G17", CultureInfo.InvariantCulture);
            _inputLatLonCheck.Active = g.InputIsLatLon;
        }

        private void RefreshGcpTable()
        {
            _gcpStore.Clear();
            foreach (var g in _workingGcps)
            {
                string img = System.IO.Path.GetFileName(g.ImagePath);
                string model = $"{g.ModelPoint.X:F3},{g.ModelPoint.Y:F3},{g.ModelPoint.Z:F3}";
                string world = $"{g.WorldPoint.X:F3},{g.WorldPoint.Y:F3},{g.WorldPoint.Z:F3}";
                _gcpStore.AppendValues(g.Id[..Math.Min(8, g.Id.Length)], img, g.PixelX.ToString("F1"), g.PixelY.ToString("F1"), model, world, g.Residual.ToString("F4"), g.Enabled);
            }
            _statsLabel.Text = GeoReferenceService.FormatResidualStats(_workingGcps);
        }

        private void ClearInputFields()
        {
            _selectedGcpId = null;
            _pixelXEntry.Text = "";
            _pixelYEntry.Text = "";
            _modelXEntry.Text = "";
            _modelYEntry.Text = "";
            _modelZEntry.Text = "";
            _worldXEntry.Text = "";
            _worldYEntry.Text = "";
            _worldZEntry.Text = "";
        }

        private void SyncRuntime()
        {
            var geo = GeoReferenceRuntime.GeoReference;
            geo.ProjectCrsEpsg = string.IsNullOrWhiteSpace(_epsgEntry.Text) ? "EPSG:4326" : _epsgEntry.Text.Trim();
            GeoReferenceRuntime.SetGeoReference(geo);
            GeoReferenceRuntime.SetGcps(_workingGcps.Select(CloneGcp));
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

        private static void AddColumn(TreeView tree, string title, int modelColumn, int width)
        {
            var col = new TreeViewColumn { Title = title, Sizing = TreeViewColumnSizing.Fixed, FixedWidth = width };
            var cell = new CellRendererText();
            col.PackStart(cell, true);
            col.AddAttribute(cell, "text", modelColumn);
            tree.AppendColumn(col);
        }

        private static void AddFormRow(Grid grid, int row, string label1, Widget w1, string label2, Widget w2)
        {
            grid.Attach(new Label(label1) { Halign = Align.Start }, 0, row, 1, 1);
            grid.Attach(w1, 1, row, 1, 1);
            grid.Attach(new Label(label2) { Halign = Align.Start }, 2, row, 1, 1);
            grid.Attach(w2, 3, row, 1, 1);
        }

        private static bool TryReadFloat(string text, out float value)
            => float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);

        private static bool TryReadDouble(string text, out double value)
            => double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value);

        private static bool SamePath(string a, string b)
        {
            try
            {
                return string.Equals(System.IO.Path.GetFullPath(a), System.IO.Path.GetFullPath(b), StringComparison.OrdinalIgnoreCase);
            }
            catch
            {
                return string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
            }
        }

        private void DisposePixbufs()
        {
            _displayPixbuf?.Dispose();
            _displayPixbuf = null;
            _fullPixbuf?.Dispose();
            _fullPixbuf = null;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                DisposePixbufs();
            base.Dispose(disposing);
        }
    }
}
