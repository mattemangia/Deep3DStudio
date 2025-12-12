using System;
using System.Runtime.InteropServices;
using Deep3DStudio.Model;
using Gdk;
using Gtk;
using SkiaSharp;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// Dialog for viewing images in full size with RGB/Depth toggle.
    /// </summary>
    public class ImagePreviewDialog : Dialog
    {
        private ImageEntry _entry;
        private Gtk.Image _imageWidget;
        private ToggleButton _rgbBtn;
        private ToggleButton _depthBtn;
        private Label _infoLabel;
        private ScrolledWindow _scrolled;

        private Pixbuf? _fullRgb;
        private Pixbuf? _fullDepth;
        private ImageViewMode _currentMode = ImageViewMode.RGB;

        private const int MaxDisplaySize = 800;

        public ImagePreviewDialog(Gtk.Window parent, ImageEntry entry) : base(
            entry.FileName,
            parent,
            DialogFlags.Modal | DialogFlags.DestroyWithParent,
            "Close", ResponseType.Close)
        {
            _entry = entry;
            this.SetDefaultSize(850, 700);
            this.WindowPosition = WindowPosition.CenterOnParent;

            var contentBox = new Box(Orientation.Vertical, 10);
            contentBox.Margin = 10;

            // Toolbar with mode toggles
            var toolbar = new Box(Orientation.Horizontal, 10);

            _rgbBtn = new ToggleButton("RGB View");
            _rgbBtn.Active = true;
            _rgbBtn.Toggled += OnRgbToggled;
            toolbar.PackStart(_rgbBtn, false, false, 0);

            _depthBtn = new ToggleButton("Depth View");
            _depthBtn.Active = false;
            _depthBtn.Sensitive = entry.DepthMap != null;
            _depthBtn.TooltipText = entry.DepthMap != null
                ? "View depth map with colormap"
                : "Depth data not available (run reconstruction first)";
            _depthBtn.Toggled += OnDepthToggled;
            toolbar.PackStart(_depthBtn, false, false, 0);

            toolbar.PackStart(new Label(""), true, true, 0); // Spacer

            _infoLabel = new Label($"{entry.Width} x {entry.Height}");
            toolbar.PackStart(_infoLabel, false, false, 0);

            contentBox.PackStart(toolbar, false, false, 0);

            // Scrollable image area
            _scrolled = new ScrolledWindow();
            _scrolled.SetPolicy(PolicyType.Automatic, PolicyType.Automatic);

            _imageWidget = new Gtk.Image();
            _scrolled.Add(_imageWidget);

            contentBox.PackStart(_scrolled, true, true, 0);

            // Color legend for depth mode
            var legendBox = CreateDepthLegend();
            contentBox.PackStart(legendBox, false, false, 5);

            this.ContentArea.PackStart(contentBox, true, true, 0);

            // Load full-size images
            LoadImages();

            this.ShowAll();
            legendBox.Visible = false; // Hide legend initially
        }

        private void LoadImages()
        {
            // Load full RGB image
            try
            {
                using var bitmap = ImageDecoder.DecodeBitmap(_entry.FilePath);
                using var rgba = bitmap.Copy(SKColorType.Rgba8888)
                    ?? throw new InvalidOperationException("Unable to convert image to RGBA8888.");

                int displayWidth = rgba.Width;
                int displayHeight = rgba.Height;

                if (displayWidth > MaxDisplaySize || displayHeight > MaxDisplaySize)
                {
                    float scale = Math.Min((float)MaxDisplaySize / displayWidth, (float)MaxDisplaySize / displayHeight);
                    displayWidth = (int)(displayWidth * scale);
                    displayHeight = (int)(displayHeight * scale);
                }

                using var resized = ResizeBitmap(rgba, displayWidth, displayHeight);

                _fullRgb = PixbufFromBitmap(resized);

                _imageWidget.Pixbuf = _fullRgb;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load image: {ex.Message}");
            }

            // Create depth image if available
            if (_entry.DepthMap != null)
            {
                _fullDepth = CreateDepthImage(_entry.DepthMap, MaxDisplaySize);
            }
        }

        private void OnRgbToggled(object? sender, EventArgs e)
        {
            if (_rgbBtn.Active)
            {
                _currentMode = ImageViewMode.RGB;
                _depthBtn.Active = false;
                UpdateDisplay();
            }
            else if (!_depthBtn.Active)
            {
                _rgbBtn.Active = true;
            }
        }

        private void OnDepthToggled(object? sender, EventArgs e)
        {
            if (_depthBtn.Active)
            {
                _currentMode = ImageViewMode.DepthMap;
                _rgbBtn.Active = false;
                UpdateDisplay();
            }
            else if (!_rgbBtn.Active)
            {
                _depthBtn.Active = true;
            }
        }

        private void UpdateDisplay()
        {
            if (_currentMode == ImageViewMode.RGB && _fullRgb != null)
            {
                _imageWidget.Pixbuf = _fullRgb;
                _infoLabel.Text = $"{_entry.Width} x {_entry.Height} - RGB";

                // Hide legend
                var legendBox = this.ContentArea.Children[0] as Box;
                if (legendBox != null && legendBox.Children.Length > 2)
                {
                    legendBox.Children[2].Visible = false;
                }
            }
            else if (_currentMode == ImageViewMode.DepthMap && _fullDepth != null)
            {
                _imageWidget.Pixbuf = _fullDepth;
                _infoLabel.Text = $"{_entry.Width} x {_entry.Height} - Depth Map";

                // Show legend
                var legendBox = this.ContentArea.Children[0] as Box;
                if (legendBox != null && legendBox.Children.Length > 2)
                {
                    legendBox.Children[2].Visible = true;
                }
            }
        }

        private Pixbuf CreateDepthImage(float[,] depthMap, int maxSize)
        {
            int origWidth = depthMap.GetLength(0);
            int origHeight = depthMap.GetLength(1);

            // Find min/max for normalization
            float minDepth = float.MaxValue;
            float maxDepth = float.MinValue;
            for (int y = 0; y < origHeight; y++)
            {
                for (int x = 0; x < origWidth; x++)
                {
                    float d = depthMap[x, y];
                    if (d > 0 && d < float.MaxValue)
                    {
                        if (d < minDepth) minDepth = d;
                        if (d > maxDepth) maxDepth = d;
                    }
                }
            }

            float range = maxDepth - minDepth;
            if (range < 0.0001f) range = 1.0f;

            // Calculate display dimensions
            int displayWidth = origWidth;
            int displayHeight = origHeight;

            if (displayWidth > maxSize || displayHeight > maxSize)
            {
                float scale = Math.Min((float)maxSize / displayWidth, (float)maxSize / displayHeight);
                displayWidth = (int)(displayWidth * scale);
                displayHeight = (int)(displayHeight * scale);
            }

            byte[] pixels = new byte[displayWidth * displayHeight * 4];

            for (int y = 0; y < displayHeight; y++)
            {
                for (int x = 0; x < displayWidth; x++)
                {
                    // Sample from original
                    int srcX = (int)(x * (float)origWidth / displayWidth);
                    int srcY = (int)(y * (float)origHeight / displayHeight);
                    srcX = Math.Clamp(srcX, 0, origWidth - 1);
                    srcY = Math.Clamp(srcY, 0, origHeight - 1);

                    float d = depthMap[srcX, srcY];
                    float t = (d - minDepth) / range;
                    t = Math.Clamp(t, 0f, 1f);

                    // Turbo colormap
                    var (r, g, b) = ImageBrowserPanel.TurboColormap(t);

                    int idx = (y * displayWidth + x) * 4;
                    pixels[idx] = (byte)(r * 255);
                    pixels[idx + 1] = (byte)(g * 255);
                    pixels[idx + 2] = (byte)(b * 255);
                    pixels[idx + 3] = 255;
                }
            }

            return new Pixbuf(pixels, Colorspace.Rgb, true, 8, displayWidth, displayHeight, displayWidth * 4);
        }

        private Box CreateDepthLegend()
        {
            var legendBox = new Box(Orientation.Horizontal, 10);
            legendBox.Halign = Align.Center;

            var nearLabel = new Label("Near (Blue)");
            legendBox.PackStart(nearLabel, false, false, 5);

            // Gradient preview
            var gradientArea = new DrawingArea();
            gradientArea.SetSizeRequest(200, 20);
            gradientArea.Drawn += (sender, args) =>
            {
                var cr = args.Cr;
                int width = gradientArea.AllocatedWidth;
                int height = gradientArea.AllocatedHeight;

                for (int x = 0; x < width; x++)
                {
                    float t = (float)x / width;
                    var (r, g, b) = ImageBrowserPanel.TurboColormap(t);
                    cr.SetSourceRGB(r, g, b);
                    cr.Rectangle(x, 0, 1, height);
                    cr.Fill();
                }

                // Border
                cr.SetSourceRGB(0.3, 0.3, 0.3);
                cr.LineWidth = 1;
                cr.Rectangle(0, 0, width, height);
                cr.Stroke();
            };
            legendBox.PackStart(gradientArea, false, false, 0);

            var farLabel = new Label("Far (Red)");
            legendBox.PackStart(farLabel, false, false, 5);

            return legendBox;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _fullRgb?.Dispose();
                _fullDepth?.Dispose();
            }
            base.Dispose(disposing);
        }

        private static Pixbuf PixbufFromBitmap(SKBitmap bitmap)
        {
            using var pixmap = bitmap.PeekPixels();
            if (pixmap == null)
                throw new InvalidOperationException("Unable to access bitmap pixels.");

            int byteCount = pixmap.RowBytes * pixmap.Height;
            byte[] pixels = new byte[byteCount];
            Marshal.Copy(pixmap.GetPixels(), pixels, 0, byteCount);

            return new Pixbuf(pixels, Colorspace.Rgb, true, 8, pixmap.Width, pixmap.Height, pixmap.RowBytes);
        }

        private static SKBitmap ResizeBitmap(SKBitmap source, int width, int height)
        {
            var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            var result = new SKBitmap(info);

            using var canvas = new SKCanvas(result);
            using var image = SKImage.FromBitmap(source);
            canvas.Clear(SKColors.Transparent);
            canvas.DrawImage(image, new SKRect(0, 0, width, height));
            canvas.Flush();

            return result;
        }
    }
}
