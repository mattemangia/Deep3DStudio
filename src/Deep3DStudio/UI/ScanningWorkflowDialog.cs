using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gtk;
using SkiaSharp;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// Scanning capture pattern for small objects
    /// </summary>
    public enum ScanPattern
    {
        Turntable,      // Object rotates on turntable, camera fixed
        Orbital,        // Camera moves around stationary object
        Dome,           // Multiple elevation levels
        Handheld,       // Freeform capture
        Photogrammetry  // Full photogrammetry coverage
    }

    /// <summary>
    /// Quality preset for scanning workflow
    /// </summary>
    public enum ScanQuality
    {
        Draft,      // 12-16 images
        Standard,   // 24-36 images
        High,       // 48-72 images
        UltraHigh   // 100+ images
    }

    /// <summary>
    /// Dialog for small object scanning workflow
    /// Provides guided capture and image management for Deep3DStudio reconstruction
    /// </summary>
    public class ScanningWorkflowDialog : Dialog
    {
        // UI Components
        private ComboBoxText _patternCombo;
        private ComboBoxText _qualityCombo;
        private SpinButton _numImagesSpin;
        private SpinButton _elevationLevelsSpin;
        private Entry _objectNameEntry;
        private TextView _notesView;
        private FlowBox _imageFlowBox;
        private Label _progressLabel;
        private ProgressBar _progressBar;
        private Label _captureGuideLabel;
        private DrawingArea _capturePreview;
        private Button _addImagesBtn;
        private Button _captureBtn;
        private Button _removeSelectedBtn;
        private Button _clearAllBtn;

        // State
        private List<CapturedImage> _capturedImages = new List<CapturedImage>();
        private int _currentCaptureIndex = 0;
        private ScanPattern _currentPattern = ScanPattern.Turntable;
        private ScanQuality _currentQuality = ScanQuality.Standard;
        private int _expectedImageCount = 24;

        // Results
        public List<string> CapturedImagePaths => _capturedImages.Select(i => i.FilePath).ToList();
        public string ObjectName => _objectNameEntry.Text;
        public string Notes => _notesView.Buffer.Text;
        public ScanPattern Pattern => _currentPattern;
        public bool StartReconstruction { get; private set; } = false;

        public ScanningWorkflowDialog(Window parent) : base("Small Object Scanning", parent, DialogFlags.Modal)
        {
            SetDefaultSize(900, 700);
            BuildUI();
            UpdateCaptureGuide();
        }

        private void BuildUI()
        {
            var contentArea = ContentArea;
            contentArea.Margin = 10;
            contentArea.Spacing = 10;

            // Main horizontal paned
            var mainPaned = new Paned(Orientation.Horizontal);
            contentArea.PackStart(mainPaned, true, true, 0);

            // Left panel - Settings and Guide
            var leftPanel = CreateSettingsPanel();
            mainPaned.Pack1(leftPanel, false, false);

            // Right panel - Image Gallery
            var rightPanel = CreateImageGalleryPanel();
            mainPaned.Pack2(rightPanel, true, false);

            mainPaned.Position = 350;

            // Bottom actions
            var actionBar = CreateActionBar();
            contentArea.PackStart(actionBar, false, false, 0);

            // Dialog buttons
            AddButton("Cancel", ResponseType.Cancel);
            var reconstructBtn = AddButton("Start Reconstruction", ResponseType.Ok);
            reconstructBtn.Sensitive = false;
            reconstructBtn.Name = "reconstruct_btn";

            ShowAll();
        }

        private Widget CreateSettingsPanel()
        {
            var vbox = new Box(Orientation.Vertical, 10);
            vbox.Margin = 5;
            vbox.SetSizeRequest(340, -1);

            // Object Info Frame
            var infoFrame = new Frame("Object Information");
            var infoBox = new Box(Orientation.Vertical, 5);
            infoBox.Margin = 10;

            var nameBox = new Box(Orientation.Horizontal, 5);
            nameBox.PackStart(new Label("Object Name:"), false, false, 0);
            _objectNameEntry = new Entry();
            _objectNameEntry.PlaceholderText = "e.g., Ancient Coin, Small Figurine";
            nameBox.PackStart(_objectNameEntry, true, true, 0);
            infoBox.PackStart(nameBox, false, false, 0);

            var notesLabel = new Label("Notes:");
            notesLabel.Halign = Align.Start;
            infoBox.PackStart(notesLabel, false, false, 0);

            var notesScroll = new ScrolledWindow();
            notesScroll.SetSizeRequest(-1, 60);
            _notesView = new TextView();
            _notesView.WrapMode = WrapMode.Word;
            notesScroll.Add(_notesView);
            infoBox.PackStart(notesScroll, false, false, 0);

            infoFrame.Add(infoBox);
            vbox.PackStart(infoFrame, false, false, 0);

            // Capture Settings Frame
            var settingsFrame = new Frame("Capture Settings");
            var settingsGrid = new Grid();
            settingsGrid.RowSpacing = 8;
            settingsGrid.ColumnSpacing = 10;
            settingsGrid.Margin = 10;

            int row = 0;

            // Pattern
            settingsGrid.Attach(new Label("Capture Pattern:") { Halign = Align.Start }, 0, row, 1, 1);
            _patternCombo = new ComboBoxText();
            _patternCombo.AppendText("Turntable (Object Rotates)");
            _patternCombo.AppendText("Orbital (Camera Moves)");
            _patternCombo.AppendText("Dome (Multi-Level)");
            _patternCombo.AppendText("Handheld (Freeform)");
            _patternCombo.AppendText("Full Photogrammetry");
            _patternCombo.Active = 0;
            _patternCombo.Changed += OnPatternChanged;
            settingsGrid.Attach(_patternCombo, 1, row++, 1, 1);

            // Quality
            settingsGrid.Attach(new Label("Quality Preset:") { Halign = Align.Start }, 0, row, 1, 1);
            _qualityCombo = new ComboBoxText();
            _qualityCombo.AppendText("Draft (12-16 images)");
            _qualityCombo.AppendText("Standard (24-36 images)");
            _qualityCombo.AppendText("High (48-72 images)");
            _qualityCombo.AppendText("Ultra High (100+ images)");
            _qualityCombo.Active = 1;
            _qualityCombo.Changed += OnQualityChanged;
            settingsGrid.Attach(_qualityCombo, 1, row++, 1, 1);

            // Number of Images
            settingsGrid.Attach(new Label("Target Images:") { Halign = Align.Start }, 0, row, 1, 1);
            _numImagesSpin = new SpinButton(8, 200, 4);
            _numImagesSpin.Value = 24;
            _numImagesSpin.ValueChanged += OnNumImagesChanged;
            settingsGrid.Attach(_numImagesSpin, 1, row++, 1, 1);

            // Elevation Levels (for Dome pattern)
            settingsGrid.Attach(new Label("Elevation Levels:") { Halign = Align.Start }, 0, row, 1, 1);
            _elevationLevelsSpin = new SpinButton(1, 5, 1);
            _elevationLevelsSpin.Value = 3;
            _elevationLevelsSpin.Sensitive = false;
            settingsGrid.Attach(_elevationLevelsSpin, 1, row++, 1, 1);

            settingsFrame.Add(settingsGrid);
            vbox.PackStart(settingsFrame, false, false, 0);

            // Capture Guide Frame
            var guideFrame = new Frame("Capture Guide");
            var guideBox = new Box(Orientation.Vertical, 5);
            guideBox.Margin = 10;

            // Visual preview of capture pattern
            _capturePreview = new DrawingArea();
            _capturePreview.SetSizeRequest(300, 200);
            _capturePreview.Drawn += OnCapturePreviewDrawn;
            guideBox.PackStart(_capturePreview, false, false, 0);

            _captureGuideLabel = new Label();
            _captureGuideLabel.Wrap = true;
            _captureGuideLabel.Halign = Align.Start;
            _captureGuideLabel.UseMarkup = true;
            guideBox.PackStart(_captureGuideLabel, false, false, 5);

            guideFrame.Add(guideBox);
            vbox.PackStart(guideFrame, true, true, 0);

            return vbox;
        }

        private Widget CreateImageGalleryPanel()
        {
            var vbox = new Box(Orientation.Vertical, 5);
            vbox.Margin = 5;

            // Header with progress
            var headerBox = new Box(Orientation.Horizontal, 10);
            var titleLabel = new Label("<b>Captured Images</b>");
            titleLabel.UseMarkup = true;
            headerBox.PackStart(titleLabel, false, false, 0);

            _progressLabel = new Label("0 / 24 images");
            headerBox.PackEnd(_progressLabel, false, false, 0);
            vbox.PackStart(headerBox, false, false, 0);

            _progressBar = new ProgressBar();
            vbox.PackStart(_progressBar, false, false, 0);

            // Image toolbar
            var toolbar = new Box(Orientation.Horizontal, 5);

            _addImagesBtn = new Button("Add Images...");
            _addImagesBtn.Clicked += OnAddImagesClicked;
            toolbar.PackStart(_addImagesBtn, false, false, 0);

            _captureBtn = new Button("Capture from Camera");
            _captureBtn.Clicked += OnCaptureClicked;
            _captureBtn.Sensitive = false; // Disabled until camera integration
            toolbar.PackStart(_captureBtn, false, false, 0);

            toolbar.PackStart(new Separator(Orientation.Vertical), false, false, 5);

            _removeSelectedBtn = new Button("Remove Selected");
            _removeSelectedBtn.Clicked += OnRemoveSelectedClicked;
            _removeSelectedBtn.Sensitive = false;
            toolbar.PackStart(_removeSelectedBtn, false, false, 0);

            _clearAllBtn = new Button("Clear All");
            _clearAllBtn.Clicked += OnClearAllClicked;
            _clearAllBtn.Sensitive = false;
            toolbar.PackStart(_clearAllBtn, false, false, 0);

            vbox.PackStart(toolbar, false, false, 0);

            // Image flow box in scrolled window
            var scrolled = new ScrolledWindow();
            scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);
            scrolled.ShadowType = ShadowType.In;

            _imageFlowBox = new FlowBox();
            _imageFlowBox.Homogeneous = true;
            _imageFlowBox.MinChildrenPerLine = 3;
            _imageFlowBox.MaxChildrenPerLine = 6;
            _imageFlowBox.ColumnSpacing = 5;
            _imageFlowBox.RowSpacing = 5;
            _imageFlowBox.SelectionMode = SelectionMode.Multiple;
            _imageFlowBox.SelectedChildrenChanged += OnImageSelectionChanged;

            scrolled.Add(_imageFlowBox);
            vbox.PackStart(scrolled, true, true, 0);

            // Coverage analysis
            var analysisFrame = new Frame("Coverage Analysis");
            var analysisBox = new Box(Orientation.Vertical, 5);
            analysisBox.Margin = 5;

            var coverageLabel = new Label("Add images to see coverage analysis");
            coverageLabel.Name = "coverage_label";
            analysisBox.PackStart(coverageLabel, false, false, 0);

            analysisFrame.Add(analysisBox);
            vbox.PackStart(analysisFrame, false, false, 0);

            return vbox;
        }

        private Widget CreateActionBar()
        {
            var hbox = new Box(Orientation.Horizontal, 10);

            // Auto-sort button
            var sortBtn = new Button("Sort by Angle");
            sortBtn.Clicked += OnSortByAngleClicked;
            hbox.PackStart(sortBtn, false, false, 0);

            // Validate coverage button
            var validateBtn = new Button("Validate Coverage");
            validateBtn.Clicked += OnValidateCoverageClicked;
            hbox.PackStart(validateBtn, false, false, 0);

            return hbox;
        }

        #region Event Handlers

        private void OnPatternChanged(object? sender, EventArgs e)
        {
            _currentPattern = (ScanPattern)_patternCombo.Active;
            _elevationLevelsSpin.Sensitive = _currentPattern == ScanPattern.Dome;

            UpdateExpectedImageCount();
            UpdateCaptureGuide();
            _capturePreview.QueueDraw();
        }

        private void OnQualityChanged(object? sender, EventArgs e)
        {
            _currentQuality = (ScanQuality)_qualityCombo.Active;
            UpdateExpectedImageCount();
            UpdateCaptureGuide();
        }

        private void OnNumImagesChanged(object? sender, EventArgs e)
        {
            _expectedImageCount = (int)_numImagesSpin.Value;
            UpdateProgress();
        }

        private void UpdateExpectedImageCount()
        {
            int baseCount = _currentQuality switch
            {
                ScanQuality.Draft => 12,
                ScanQuality.Standard => 24,
                ScanQuality.High => 48,
                ScanQuality.UltraHigh => 100,
                _ => 24
            };

            if (_currentPattern == ScanPattern.Dome)
            {
                baseCount = baseCount * (int)_elevationLevelsSpin.Value / 2;
            }
            else if (_currentPattern == ScanPattern.Photogrammetry)
            {
                baseCount = (int)(baseCount * 1.5);
            }

            _numImagesSpin.Value = baseCount;
            _expectedImageCount = baseCount;
            UpdateProgress();
        }

        private void UpdateCaptureGuide()
        {
            string guide = _currentPattern switch
            {
                ScanPattern.Turntable => GetTurntableGuide(),
                ScanPattern.Orbital => GetOrbitalGuide(),
                ScanPattern.Dome => GetDomeGuide(),
                ScanPattern.Handheld => GetHandheldGuide(),
                ScanPattern.Photogrammetry => GetPhotogrammetryGuide(),
                _ => ""
            };

            _captureGuideLabel.Markup = guide;
        }

        private string GetTurntableGuide()
        {
            int angleStep = 360 / _expectedImageCount;
            return $"<b>Turntable Capture:</b>\n" +
                   $"1. Place object on turntable\n" +
                   $"2. Position camera at eye level\n" +
                   $"3. Rotate turntable {angleStep} between shots\n" +
                   $"4. Keep lighting consistent\n\n" +
                   $"<i>Tip: Use diffuse lighting to avoid harsh shadows</i>";
        }

        private string GetOrbitalGuide()
        {
            int angleStep = 360 / _expectedImageCount;
            return $"<b>Orbital Capture:</b>\n" +
                   $"1. Place object on stable surface\n" +
                   $"2. Move camera around object\n" +
                   $"3. Take photo every {angleStep}\n" +
                   $"4. Maintain consistent distance\n\n" +
                   $"<i>Tip: Keep object centered in frame</i>";
        }

        private string GetDomeGuide()
        {
            int levels = (int)_elevationLevelsSpin.Value;
            int imagesPerLevel = _expectedImageCount / levels;
            return $"<b>Dome Capture ({levels} levels):</b>\n" +
                   $"1. Capture {imagesPerLevel} images at each level\n" +
                   $"2. Start at eye level (0)\n" +
                   $"3. Move up in 30 increments\n" +
                   $"4. Include top-down shot\n\n" +
                   $"<i>Tip: This provides best coverage for complex objects</i>";
        }

        private string GetHandheldGuide()
        {
            return $"<b>Handheld Capture:</b>\n" +
                   $"1. Hold camera steady\n" +
                   $"2. Move around object smoothly\n" +
                   $"3. Overlap each shot by 60-70%\n" +
                   $"4. Capture all angles and details\n\n" +
                   $"<i>Tip: Include close-ups of details</i>";
        }

        private string GetPhotogrammetryGuide()
        {
            return $"<b>Full Photogrammetry:</b>\n" +
                   $"1. Capture all surfaces thoroughly\n" +
                   $"2. Include multiple elevation levels\n" +
                   $"3. Add detail shots for fine features\n" +
                   $"4. Ensure 70%+ overlap between shots\n\n" +
                   $"<i>Tip: More images = better quality</i>";
        }

        private void OnCapturePreviewDrawn(object o, DrawnArgs args)
        {
            var cr = args.Cr;
            int width = _capturePreview.AllocatedWidth;
            int height = _capturePreview.AllocatedHeight;

            // Background
            cr.SetSourceRGB(0.15, 0.15, 0.18);
            cr.Rectangle(0, 0, width, height);
            cr.Fill();

            // Draw capture pattern visualization
            double cx = width / 2.0;
            double cy = height / 2.0;
            double radius = Math.Min(width, height) * 0.35;

            // Object in center
            cr.SetSourceRGB(0.4, 0.6, 0.8);
            cr.Arc(cx, cy, 20, 0, Math.PI * 2);
            cr.Fill();

            // Camera positions based on pattern
            int numPositions = _expectedImageCount;
            if (_currentPattern == ScanPattern.Dome)
                numPositions = _expectedImageCount / (int)_elevationLevelsSpin.Value;

            cr.SetSourceRGB(0.9, 0.7, 0.2);
            for (int i = 0; i < Math.Min(numPositions, 36); i++)
            {
                double angle = (2 * Math.PI * i) / numPositions;
                double px = cx + Math.Cos(angle) * radius;
                double py = cy + Math.Sin(angle) * radius;

                // Camera icon
                cr.Arc(px, py, 8, 0, Math.PI * 2);
                cr.Fill();

                // Line to object
                cr.SetSourceRGBA(0.9, 0.7, 0.2, 0.3);
                cr.MoveTo(px, py);
                cr.LineTo(cx, cy);
                cr.Stroke();
                cr.SetSourceRGB(0.9, 0.7, 0.2);
            }

            // Highlight current capture position
            if (_capturedImages.Count < numPositions)
            {
                double angle = (2 * Math.PI * _capturedImages.Count) / numPositions;
                double px = cx + Math.Cos(angle) * radius;
                double py = cy + Math.Sin(angle) * radius;

                cr.SetSourceRGB(0.2, 0.9, 0.3);
                cr.Arc(px, py, 12, 0, Math.PI * 2);
                cr.Stroke();
            }

            // Pattern label
            cr.SetSourceRGB(0.8, 0.8, 0.8);
            cr.SelectFontFace("Sans", Cairo.FontSlant.Normal, Cairo.FontWeight.Bold);
            cr.SetFontSize(12);
            cr.MoveTo(10, 20);
            cr.ShowText(_currentPattern.ToString());
        }

        private void OnAddImagesClicked(object? sender, EventArgs e)
        {
            var fc = new FileChooserDialog("Add Images", (Window)Toplevel, FileChooserAction.Open,
                "Cancel", ResponseType.Cancel, "Open", ResponseType.Accept);
            fc.SelectMultiple = true;

            var filter = new FileFilter();
            filter.Name = "Image Files";
            filter.AddPattern("*.jpg");
            filter.AddPattern("*.jpeg");
            filter.AddPattern("*.png");
            filter.AddPattern("*.tif");
            filter.AddPattern("*.tiff");
            filter.AddPattern("*.bmp");
            fc.AddFilter(filter);

            if (fc.Run() == (int)ResponseType.Accept)
            {
                foreach (var path in fc.Filenames.OrderBy(f => f))
                {
                    AddImage(path);
                }
            }
            fc.Destroy();
        }

        private void OnCaptureClicked(object? sender, EventArgs e)
        {
            // Future: Direct camera capture integration
            ShowMessage("Camera capture not yet implemented.\nPlease use 'Add Images' to import photos.");
        }

        private void OnRemoveSelectedClicked(object? sender, EventArgs e)
        {
            var selectedChildren = _imageFlowBox.SelectedChildren.ToList();
            foreach (var child in selectedChildren)
            {
                int index = child.Index;
                if (index >= 0 && index < _capturedImages.Count)
                {
                    _capturedImages.RemoveAt(index);
                }
                _imageFlowBox.Remove(child);
            }
            UpdateProgress();
            UpdateCoverageAnalysis();
        }

        private void OnClearAllClicked(object? sender, EventArgs e)
        {
            _capturedImages.Clear();
            foreach (var child in _imageFlowBox.Children.ToList())
            {
                _imageFlowBox.Remove(child);
            }
            UpdateProgress();
            UpdateCoverageAnalysis();
        }

        private void OnImageSelectionChanged(object? sender, EventArgs e)
        {
            var selectedCount = _imageFlowBox.SelectedChildren.Count();
            _removeSelectedBtn.Sensitive = selectedCount > 0;
        }

        private void OnSortByAngleClicked(object? sender, EventArgs e)
        {
            // Sort images by filename (assuming numbered sequence)
            _capturedImages = _capturedImages.OrderBy(i => i.FilePath).ToList();
            RefreshImageGallery();
        }

        private void OnValidateCoverageClicked(object? sender, EventArgs e)
        {
            if (_capturedImages.Count < 2)
            {
                ShowMessage("Please add at least 2 images to validate coverage.");
                return;
            }

            var issues = new List<string>();

            // Check minimum count
            if (_capturedImages.Count < _expectedImageCount * 0.5)
            {
                issues.Add($"Image count ({_capturedImages.Count}) is below 50% of target ({_expectedImageCount})");
            }

            // Check for resolution consistency
            var resolutions = _capturedImages.Select(i => $"{i.Width}x{i.Height}").Distinct().ToList();
            if (resolutions.Count > 1)
            {
                issues.Add($"Mixed resolutions detected: {string.Join(", ", resolutions)}");
            }

            // Check for very low resolution
            if (_capturedImages.Any(i => i.Width < 1000 || i.Height < 1000))
            {
                issues.Add("Some images have low resolution (<1000px). Higher resolution recommended.");
            }

            string message;
            if (issues.Count == 0)
            {
                message = "Coverage validation passed!\n\n" +
                          $"Images: {_capturedImages.Count} / {_expectedImageCount}\n" +
                          $"Resolution: {_capturedImages[0].Width}x{_capturedImages[0].Height}\n" +
                          "Ready for reconstruction.";
            }
            else
            {
                message = "Coverage issues detected:\n\n" + string.Join("\n", issues.Select(i => $"- {i}"));
            }

            ShowMessage(message);
        }

        #endregion

        #region Helper Methods

        private void AddImage(string filePath)
        {
            if (_capturedImages.Any(i => i.FilePath == filePath))
                return;

            try
            {
                var img = new CapturedImage(filePath);
                _capturedImages.Add(img);

                // Create thumbnail widget
                var thumbWidget = CreateThumbnailWidget(img, _capturedImages.Count - 1);
                _imageFlowBox.Add(thumbWidget);
                _imageFlowBox.ShowAll();

                UpdateProgress();
                UpdateCoverageAnalysis();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to add image {filePath}: {ex.Message}");
            }
        }

        private Widget CreateThumbnailWidget(CapturedImage img, int index)
        {
            var vbox = new Box(Orientation.Vertical, 2);
            vbox.SetSizeRequest(100, 120);

            // Thumbnail image
            try
            {
                var pixbuf = new Gdk.Pixbuf(img.FilePath);
                var scaled = pixbuf.ScaleSimple(90, 70, Gdk.InterpType.Bilinear);

                var image = new Image(scaled);
                vbox.PackStart(image, false, false, 0);
            }
            catch
            {
                var placeholder = new Label("[Image]");
                placeholder.SetSizeRequest(90, 70);
                vbox.PackStart(placeholder, false, false, 0);
            }

            // Index label
            var label = new Label($"#{index + 1}");
            label.SetSizeRequest(-1, 20);
            vbox.PackStart(label, false, false, 0);

            // Size info
            var sizeLabel = new Label($"{img.Width}x{img.Height}");
            sizeLabel.SetSizeRequest(-1, 15);
            var attrs = new Pango.AttrList();
            attrs.Insert(new Pango.AttrScale(0.8));
            sizeLabel.Attributes = attrs;
            vbox.PackStart(sizeLabel, false, false, 0);

            return vbox;
        }

        private void RefreshImageGallery()
        {
            foreach (var child in _imageFlowBox.Children.ToList())
            {
                _imageFlowBox.Remove(child);
            }

            for (int i = 0; i < _capturedImages.Count; i++)
            {
                var thumbWidget = CreateThumbnailWidget(_capturedImages[i], i);
                _imageFlowBox.Add(thumbWidget);
            }

            _imageFlowBox.ShowAll();
        }

        private void UpdateProgress()
        {
            int count = _capturedImages.Count;
            _progressLabel.Text = $"{count} / {_expectedImageCount} images";
            _progressBar.Fraction = Math.Min(1.0, (double)count / _expectedImageCount);

            _clearAllBtn.Sensitive = count > 0;

            // Enable reconstruct button if we have enough images
            var reconstructBtn = Children.OfType<Box>()
                .SelectMany(b => b.Children)
                .OfType<Button>()
                .FirstOrDefault(b => b.Name == "reconstruct_btn");

            // Find in action area
            foreach (var widget in ActionArea.Children)
            {
                if (widget is Button btn && btn.Label == "Start Reconstruction")
                {
                    btn.Sensitive = count >= 2;
                    break;
                }
            }

            _capturePreview.QueueDraw();
        }

        private void UpdateCoverageAnalysis()
        {
            // Find coverage label and update
            // This would analyze camera angles and coverage
            string analysis = "";

            if (_capturedImages.Count == 0)
            {
                analysis = "Add images to see coverage analysis";
            }
            else if (_capturedImages.Count < _expectedImageCount * 0.5)
            {
                analysis = $"Coverage: Low ({_capturedImages.Count * 100 / _expectedImageCount}%)\nRecommend adding more images.";
            }
            else if (_capturedImages.Count < _expectedImageCount)
            {
                analysis = $"Coverage: Good ({_capturedImages.Count * 100 / _expectedImageCount}%)\nAdd more for better quality.";
            }
            else
            {
                analysis = $"Coverage: Excellent ({_capturedImages.Count * 100 / _expectedImageCount}%)\nReady for reconstruction!";
            }

            // Update label (would need to find it in the hierarchy)
        }

        private void ShowMessage(string message)
        {
            var md = new MessageDialog((Window)Toplevel, DialogFlags.Modal,
                MessageType.Info, ButtonsType.Ok, message);
            md.Run();
            md.Destroy();
        }

        #endregion

        protected override void OnResponse(ResponseType response_id)
        {
            if (response_id == ResponseType.Ok)
            {
                StartReconstruction = true;
            }
            base.OnResponse(response_id);
        }
    }

    /// <summary>
    /// Represents a captured image with metadata
    /// </summary>
    public class CapturedImage
    {
        public string FilePath { get; }
        public int Width { get; }
        public int Height { get; }
        public DateTime CaptureTime { get; }
        public float EstimatedAngle { get; set; } // Estimated rotation angle
        public float EstimatedElevation { get; set; } // Estimated elevation

        public CapturedImage(string filePath)
        {
            FilePath = filePath;
            CaptureTime = File.GetCreationTime(filePath);

            // Get image dimensions
            try
            {
                using var bitmap = SKBitmap.Decode(filePath);
                if (bitmap != null)
                {
                    Width = bitmap.Width;
                    Height = bitmap.Height;
                }
                else
                {
                    Width = Height = 0;
                }
            }
            catch
            {
                Width = Height = 0;
            }
        }
    }
}
