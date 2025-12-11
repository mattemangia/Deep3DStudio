using System;
using System.Collections.Generic;
using Gtk;
using Gdk;
using Cairo;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Deep3DStudio.UI
{
    /// <summary>
    /// Stores image data along with optional depth information for visualization.
    /// </summary>
    public class ImageEntry
    {
        public string FilePath { get; set; } = string.Empty;
        public string FileName { get; set; } = string.Empty;
        public Pixbuf? Thumbnail { get; set; }
        public Pixbuf? ThumbnailDepth { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public float[,]? DepthMap { get; set; } // Optional depth data from reconstruction
    }

    public enum ImageViewMode
    {
        RGB,
        DepthMap
    }

    /// <summary>
    /// Panel that displays image thumbnails with RGB/Depth toggle.
    /// </summary>
    public class ImageBrowserPanel : Box
    {
        private FlowBox _flowBox;
        private ToggleButton _rgbModeBtn;
        private ToggleButton _depthModeBtn;
        private Label _countLabel;
        private List<ImageEntry> _images = new List<ImageEntry>();
        private ImageViewMode _viewMode = ImageViewMode.RGB;

        private const int ThumbnailSize = 80;

        public event EventHandler<ImageEntry>? ImageDoubleClicked;
        public event EventHandler<ImageEntry>? ImageSelected;

        public ImageBrowserPanel() : base(Orientation.Vertical, 5)
        {
            this.MarginStart = 5;
            this.MarginEnd = 5;

            // Header with mode toggles
            var headerBox = new Box(Orientation.Horizontal, 5);

            var titleLabel = new Label("Images");
            titleLabel.Attributes = new Pango.AttrList();
            titleLabel.Attributes.Insert(new Pango.AttrWeight(Pango.Weight.Bold));
            headerBox.PackStart(titleLabel, false, false, 0);

            _countLabel = new Label("(0)");
            headerBox.PackStart(_countLabel, false, false, 0);

            headerBox.PackStart(new Label(""), true, true, 0); // Spacer

            _rgbModeBtn = new ToggleButton("RGB");
            _rgbModeBtn.Active = true;
            _rgbModeBtn.TooltipText = "Show RGB thumbnails";
            _rgbModeBtn.Toggled += OnRgbModeToggled;
            headerBox.PackStart(_rgbModeBtn, false, false, 0);

            _depthModeBtn = new ToggleButton("Depth");
            _depthModeBtn.Active = false;
            _depthModeBtn.TooltipText = "Show depth map thumbnails";
            _depthModeBtn.Toggled += OnDepthModeToggled;
            headerBox.PackStart(_depthModeBtn, false, false, 0);

            this.PackStart(headerBox, false, false, 5);

            // Scrollable flow box for thumbnails
            var scrolled = new ScrolledWindow();
            scrolled.SetPolicy(PolicyType.Never, PolicyType.Automatic);
            scrolled.SetSizeRequest(-1, 200);

            _flowBox = new FlowBox();
            _flowBox.SelectionMode = SelectionMode.Single;
            _flowBox.Homogeneous = true;
            _flowBox.MaxChildrenPerLine = 4;
            _flowBox.MinChildrenPerLine = 2;
            _flowBox.RowSpacing = 5;
            _flowBox.ColumnSpacing = 5;
            _flowBox.ChildActivated += OnChildActivated;

            scrolled.Add(_flowBox);
            this.PackStart(scrolled, true, true, 0);
        }

        private void OnRgbModeToggled(object? sender, EventArgs e)
        {
            if (_rgbModeBtn.Active)
            {
                _viewMode = ImageViewMode.RGB;
                _depthModeBtn.Active = false;
                RefreshThumbnails();
            }
            else if (!_depthModeBtn.Active)
            {
                _rgbModeBtn.Active = true; // Keep one active
            }
        }

        private void OnDepthModeToggled(object? sender, EventArgs e)
        {
            if (_depthModeBtn.Active)
            {
                _viewMode = ImageViewMode.DepthMap;
                _rgbModeBtn.Active = false;
                RefreshThumbnails();
            }
            else if (!_rgbModeBtn.Active)
            {
                _depthModeBtn.Active = true; // Keep one active
            }
        }

        private void OnChildActivated(object o, ChildActivatedArgs args)
        {
            var child = args.Child;
            int index = child.Index;
            if (index >= 0 && index < _images.Count)
            {
                ImageDoubleClicked?.Invoke(this, _images[index]);
            }
        }

        /// <summary>
        /// Add an image to the browser.
        /// </summary>
        public void AddImage(string filePath)
        {
            var entry = new ImageEntry
            {
                FilePath = filePath,
                FileName = System.IO.Path.GetFileName(filePath)
            };

            // Load and create thumbnail
            try
            {
                using (var img = SixLabors.ImageSharp.Image.Load<Rgba32>(filePath))
                {
                    entry.Width = img.Width;
                    entry.Height = img.Height;

                    // Create RGB thumbnail
                    entry.Thumbnail = CreateThumbnailFromImage(img, ThumbnailSize);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load image {filePath}: {ex.Message}");
                return;
            }

            _images.Add(entry);
            AddThumbnailWidget(entry);
            UpdateCountLabel();
        }

        /// <summary>
        /// Set depth data for an image (called after reconstruction).
        /// </summary>
        public void SetDepthData(int index, float[,] depthMap)
        {
            if (index < 0 || index >= _images.Count) return;

            var entry = _images[index];
            entry.DepthMap = depthMap;
            entry.ThumbnailDepth = CreateDepthThumbnail(depthMap, ThumbnailSize);

            // Refresh if in depth mode
            if (_viewMode == ImageViewMode.DepthMap)
            {
                RefreshThumbnails();
            }
        }

        /// <summary>
        /// Clear all images.
        /// </summary>
        public void Clear()
        {
            foreach (var entry in _images)
            {
                entry.Thumbnail?.Dispose();
                entry.ThumbnailDepth?.Dispose();
            }
            _images.Clear();

            foreach (var child in _flowBox.Children)
            {
                _flowBox.Remove(child);
                child.Destroy();
            }

            UpdateCountLabel();
        }

        public List<ImageEntry> GetImages() => _images;

        public int ImageCount => _images.Count;

        private void AddThumbnailWidget(ImageEntry entry)
        {
            var box = new Box(Orientation.Vertical, 2);

            var eventBox = new EventBox();
            var image = new Gtk.Image();

            var pixbuf = _viewMode == ImageViewMode.DepthMap && entry.ThumbnailDepth != null
                ? entry.ThumbnailDepth
                : entry.Thumbnail;

            if (pixbuf != null)
            {
                image.Pixbuf = pixbuf;
            }

            eventBox.Add(image);
            eventBox.TooltipText = entry.FileName;

            box.PackStart(eventBox, false, false, 0);

            // Truncate filename if too long
            string displayName = entry.FileName.Length > 12
                ? entry.FileName.Substring(0, 9) + "..."
                : entry.FileName;

            var label = new Label(displayName);
            label.SetSizeRequest(ThumbnailSize, -1);
            label.Ellipsize = Pango.EllipsizeMode.End;
            box.PackStart(label, false, false, 0);

            box.ShowAll();
            _flowBox.Add(box);
        }

        private void RefreshThumbnails()
        {
            // Remove all children
            foreach (var child in _flowBox.Children)
            {
                _flowBox.Remove(child);
                child.Destroy();
            }

            // Re-add with current mode
            foreach (var entry in _images)
            {
                AddThumbnailWidget(entry);
            }
        }

        private void UpdateCountLabel()
        {
            _countLabel.Text = $"({_images.Count})";
        }

        private Pixbuf CreateThumbnailFromImage(Image<Rgba32> img, int size)
        {
            // Resize maintaining aspect ratio
            int newWidth, newHeight;
            if (img.Width > img.Height)
            {
                newWidth = size;
                newHeight = (int)(size * (float)img.Height / img.Width);
            }
            else
            {
                newHeight = size;
                newWidth = (int)(size * (float)img.Width / img.Height);
            }

            var resized = img.Clone(ctx => ctx.Resize(newWidth, newHeight));

            // Convert to Pixbuf
            byte[] pixels = new byte[newWidth * newHeight * 4];
            int idx = 0;
            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    var pixel = resized[x, y];
                    pixels[idx++] = pixel.R;
                    pixels[idx++] = pixel.G;
                    pixels[idx++] = pixel.B;
                    pixels[idx++] = pixel.A;
                }
            }

            resized.Dispose();
            return new Pixbuf(pixels, Colorspace.Rgb, true, 8, newWidth, newHeight, newWidth * 4);
        }

        private Pixbuf CreateDepthThumbnail(float[,] depthMap, int size)
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

            // Calculate thumbnail dimensions
            int newWidth, newHeight;
            if (origWidth > origHeight)
            {
                newWidth = size;
                newHeight = (int)(size * (float)origHeight / origWidth);
            }
            else
            {
                newHeight = size;
                newWidth = (int)(size * (float)origWidth / origHeight);
            }

            byte[] pixels = new byte[newWidth * newHeight * 4];

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    // Sample from original
                    int srcX = (int)(x * (float)origWidth / newWidth);
                    int srcY = (int)(y * (float)origHeight / newHeight);
                    srcX = Math.Clamp(srcX, 0, origWidth - 1);
                    srcY = Math.Clamp(srcY, 0, origHeight - 1);

                    float d = depthMap[srcX, srcY];
                    float t = (d - minDepth) / range;
                    t = Math.Clamp(t, 0f, 1f);

                    // Turbo colormap
                    var (r, g, b) = TurboColormap(t);

                    int idx = (y * newWidth + x) * 4;
                    pixels[idx] = (byte)(r * 255);
                    pixels[idx + 1] = (byte)(g * 255);
                    pixels[idx + 2] = (byte)(b * 255);
                    pixels[idx + 3] = 255;
                }
            }

            return new Pixbuf(pixels, Colorspace.Rgb, true, 8, newWidth, newHeight, newWidth * 4);
        }

        public static (float r, float g, float b) TurboColormap(float t)
        {
            t = Math.Clamp(t, 0f, 1f);

            float r, g, b;

            if (t < 0.25f)
            {
                float s = t / 0.25f;
                r = 0.0f; g = s; b = 1.0f;
            }
            else if (t < 0.5f)
            {
                float s = (t - 0.25f) / 0.25f;
                r = 0.0f; g = 1.0f; b = 1.0f - s;
            }
            else if (t < 0.75f)
            {
                float s = (t - 0.5f) / 0.25f;
                r = s; g = 1.0f; b = 0.0f;
            }
            else
            {
                float s = (t - 0.75f) / 0.25f;
                r = 1.0f; g = 1.0f - s; b = 0.0f;
            }

            return (r, g, b);
        }
    }
}
