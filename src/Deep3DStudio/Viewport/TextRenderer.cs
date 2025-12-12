using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using OpenTK.Graphics.OpenGL;
using Cairo;
using Pango;

namespace Deep3DStudio.Viewport
{
    /// <summary>
    /// OpenGL text rendering using Cairo/Pango for font rendering
    /// Creates textures from rendered text that can be displayed in the GL viewport
    /// </summary>
    public class TextRenderer : IDisposable
    {
        private readonly Dictionary<string, CachedTextTexture> _textureCache = new Dictionary<string, CachedTextTexture>();
        private int _maxCacheSize = 100;
        private bool _disposed = false;

        // Default font settings
        public string FontFamily { get; set; } = "Sans";
        public int FontSize { get; set; } = 12;
        public FontWeight FontWeight { get; set; } = FontWeight.Normal;

        /// <summary>
        /// Renders text at the specified screen position
        /// </summary>
        /// <param name="text">Text to render</param>
        /// <param name="x">Screen X position</param>
        /// <param name="y">Screen Y position</param>
        /// <param name="color">Text color (RGBA, 0-1 range)</param>
        /// <param name="shadowColor">Optional shadow color (null for no shadow)</param>
        public void DrawText(string text, float x, float y,
            (float r, float g, float b, float a) color,
            (float r, float g, float b, float a)? shadowColor = null)
        {
            if (string.IsNullOrEmpty(text)) return;

            var texture = GetOrCreateTexture(text, color, shadowColor);
            if (texture == null) return;

            RenderTexturedQuad(texture, x, y);
        }

        /// <summary>
        /// Renders text with a background box
        /// </summary>
        public void DrawTextWithBackground(string text, float x, float y,
            (float r, float g, float b, float a) textColor,
            (float r, float g, float b, float a) bgColor,
            float padding = 4)
        {
            if (string.IsNullOrEmpty(text)) return;

            var texture = GetOrCreateTexture(text, textColor, null);
            if (texture == null) return;

            // Draw background
            GL.Color4(bgColor.r, bgColor.g, bgColor.b, bgColor.a);
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(x - padding, y - padding);
            GL.Vertex2(x + texture.Width + padding, y - padding);
            GL.Vertex2(x + texture.Width + padding, y + texture.Height + padding);
            GL.Vertex2(x - padding, y + texture.Height + padding);
            GL.End();

            RenderTexturedQuad(texture, x, y);
        }

        /// <summary>
        /// Renders multi-line text
        /// </summary>
        public void DrawMultilineText(string[] lines, float x, float y,
            (float r, float g, float b, float a) color, float lineSpacing = 1.2f)
        {
            float currentY = y;
            int lineHeight = (int)(FontSize * lineSpacing);

            foreach (var line in lines)
            {
                DrawText(line, x, currentY, color);
                currentY += lineHeight;
            }
        }

        /// <summary>
        /// Measures text dimensions without rendering
        /// </summary>
        public (int width, int height) MeasureText(string text)
        {
            if (string.IsNullOrEmpty(text)) return (0, 0);

            using var surface = new ImageSurface(Format.ARGB32, 1, 1);
            using var context = new Cairo.Context(surface);
            using var layout = Pango.CairoHelper.CreateLayout(context);

            var fontDesc = new FontDescription
            {
                Family = FontFamily,
                Size = FontSize * Pango.Scale.PangoScale,
                Weight = FontWeight
            };
            layout.FontDescription = fontDesc;
            layout.SetText(text);
            layout.GetPixelSize(out int width, out int height);

            return (width, height);
        }

        /// <summary>
        /// Clears the texture cache
        /// </summary>
        public void ClearCache()
        {
            foreach (var texture in _textureCache.Values)
            {
                if (texture.TextureId != 0)
                    GL.DeleteTexture(texture.TextureId);
            }
            _textureCache.Clear();
        }

        private CachedTextTexture? GetOrCreateTexture(string text,
            (float r, float g, float b, float a) color,
            (float r, float g, float b, float a)? shadowColor)
        {
            string key = $"{text}_{FontFamily}_{FontSize}_{FontWeight}_{color}_{shadowColor}";

            if (_textureCache.TryGetValue(key, out var cached))
            {
                cached.LastUsed = DateTime.Now;
                return cached;
            }

            // Evict old entries if cache is full
            if (_textureCache.Count >= _maxCacheSize)
            {
                EvictOldestEntry();
            }

            var texture = CreateTextTexture(text, color, shadowColor);
            if (texture != null)
            {
                _textureCache[key] = texture;
            }

            return texture;
        }

        private CachedTextTexture? CreateTextTexture(string text,
            (float r, float g, float b, float a) color,
            (float r, float g, float b, float a)? shadowColor)
        {
            try
            {
                // First, measure the text
                var (width, height) = MeasureText(text);
                if (width <= 0 || height <= 0) return null;

                // Add padding for shadow
                int padding = shadowColor.HasValue ? 2 : 0;
                int texWidth = width + padding * 2;
                int texHeight = height + padding * 2;

                // Power of 2 sizes for better compatibility
                texWidth = NextPowerOfTwo(texWidth);
                texHeight = NextPowerOfTwo(texHeight);

                // Create Cairo surface
                using var surface = new ImageSurface(Format.ARGB32, texWidth, texHeight);
                using var context = new Cairo.Context(surface);
                using var layout = Pango.CairoHelper.CreateLayout(context);

                // Clear with transparent
                context.SetSourceRGBA(0, 0, 0, 0);
                context.Paint();

                // Setup font
                var fontDesc = new FontDescription
                {
                    Family = FontFamily,
                    Size = FontSize * Pango.Scale.PangoScale,
                    Weight = FontWeight
                };
                layout.FontDescription = fontDesc;
                layout.SetText(text);

                // Draw shadow if specified
                if (shadowColor.HasValue)
                {
                    var sc = shadowColor.Value;
                    context.SetSourceRGBA(sc.r, sc.g, sc.b, sc.a);
                    context.MoveTo(padding + 1, padding + 1);
                    Pango.CairoHelper.ShowLayout(context, layout);
                }

                // Draw main text
                context.SetSourceRGBA(color.r, color.g, color.b, color.a);
                context.MoveTo(padding, padding);
                Pango.CairoHelper.ShowLayout(context, layout);

                surface.Flush();

                // Create OpenGL texture
                int textureId = GL.GenTexture();
                GL.BindTexture(TextureTarget.Texture2D, textureId);

                // Get pixel data
                byte[] pixels = new byte[texWidth * texHeight * 4];
                Marshal.Copy(surface.DataPtr, pixels, 0, pixels.Length);

                // Convert BGRA to RGBA (Cairo uses BGRA, OpenGL uses RGBA)
                for (int i = 0; i < pixels.Length; i += 4)
                {
                    byte b = pixels[i];
                    byte r = pixels[i + 2];
                    pixels[i] = r;
                    pixels[i + 2] = b;
                }

                GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba,
                    texWidth, texHeight, 0, PixelFormat.Rgba, PixelType.UnsignedByte, pixels);

                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToEdge);
                GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToEdge);

                GL.BindTexture(TextureTarget.Texture2D, 0);

                return new CachedTextTexture
                {
                    TextureId = textureId,
                    Width = width + padding,
                    Height = height + padding,
                    TextureWidth = texWidth,
                    TextureHeight = texHeight,
                    LastUsed = DateTime.Now
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to create text texture: {ex.Message}");
                return null;
            }
        }

        private void RenderTexturedQuad(CachedTextTexture texture, float x, float y)
        {
            GL.Enable(EnableCap.Texture2D);
            GL.BindTexture(TextureTarget.Texture2D, texture.TextureId);

            // Use white color to let texture color through
            GL.Color4(1.0f, 1.0f, 1.0f, 1.0f);

            // Calculate texture coordinates to only use the actual text portion
            float u = (float)texture.Width / texture.TextureWidth;
            float v = (float)texture.Height / texture.TextureHeight;

            GL.Begin(PrimitiveType.Quads);

            GL.TexCoord2(0, 0);
            GL.Vertex2(x, y);

            GL.TexCoord2(u, 0);
            GL.Vertex2(x + texture.Width, y);

            GL.TexCoord2(u, v);
            GL.Vertex2(x + texture.Width, y + texture.Height);

            GL.TexCoord2(0, v);
            GL.Vertex2(x, y + texture.Height);

            GL.End();

            GL.Disable(EnableCap.Texture2D);
            GL.BindTexture(TextureTarget.Texture2D, 0);
        }

        private void EvictOldestEntry()
        {
            string? oldestKey = null;
            DateTime oldestTime = DateTime.MaxValue;

            foreach (var kvp in _textureCache)
            {
                if (kvp.Value.LastUsed < oldestTime)
                {
                    oldestTime = kvp.Value.LastUsed;
                    oldestKey = kvp.Key;
                }
            }

            if (oldestKey != null)
            {
                if (_textureCache[oldestKey].TextureId != 0)
                    GL.DeleteTexture(_textureCache[oldestKey].TextureId);
                _textureCache.Remove(oldestKey);
            }
        }

        private static int NextPowerOfTwo(int value)
        {
            int result = 1;
            while (result < value)
                result *= 2;
            return result;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                ClearCache();
                _disposed = true;
            }
        }

        private class CachedTextTexture
        {
            public int TextureId { get; set; }
            public int Width { get; set; }      // Actual text width
            public int Height { get; set; }     // Actual text height
            public int TextureWidth { get; set; }   // Texture width (power of 2)
            public int TextureHeight { get; set; }  // Texture height (power of 2)
            public DateTime LastUsed { get; set; }
        }
    }

    /// <summary>
    /// Overlay panel for displaying viewport information
    /// </summary>
    public class ViewportOverlay
    {
        private TextRenderer _textRenderer;
        private bool _initialized = false;

        // Display settings
        public bool ShowFPS { get; set; } = true;
        public bool ShowObjectCount { get; set; } = true;
        public bool ShowSelectionInfo { get; set; } = true;
        public bool ShowGizmoMode { get; set; } = true;
        public bool ShowCameraInfo { get; set; } = true;

        // Colors
        public (float r, float g, float b, float a) BackgroundColor { get; set; } = (0.0f, 0.0f, 0.0f, 0.6f);
        public (float r, float g, float b, float a) TextColor { get; set; } = (0.9f, 0.9f, 0.9f, 1.0f);
        public (float r, float g, float b, float a) HighlightColor { get; set; } = (1.0f, 0.8f, 0.2f, 1.0f);
        public (float r, float g, float b, float a) WarningColor { get; set; } = (1.0f, 0.4f, 0.2f, 1.0f);

        public ViewportOverlay()
        {
            _textRenderer = new TextRenderer
            {
                FontFamily = "Monospace",
                FontSize = 11,
                FontWeight = FontWeight.Normal
            };
        }

        public void Initialize()
        {
            _initialized = true;
        }

        /// <summary>
        /// Draws the complete viewport overlay with all information
        /// </summary>
        public void Draw(int viewportWidth, int viewportHeight, ViewportStats stats)
        {
            if (!_initialized)
            {
                Initialize();
            }

            // Setup 2D orthographic projection
            GL.MatrixMode(MatrixMode.Projection);
            GL.PushMatrix();
            GL.LoadIdentity();
            GL.Ortho(0, viewportWidth, viewportHeight, 0, -1, 1);

            GL.MatrixMode(MatrixMode.Modelview);
            GL.PushMatrix();
            GL.LoadIdentity();

            GL.Disable(EnableCap.DepthTest);
            GL.Enable(EnableCap.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);

            // Draw info panel in top-left
            DrawInfoPanel(10, 10, stats);

            // Draw gizmo mode indicator in bottom-left
            if (ShowGizmoMode)
            {
                DrawGizmoModeIndicator(10, viewportHeight - 30, stats.GizmoMode);
            }

            // Draw selection info in bottom-right
            if (ShowSelectionInfo && stats.SelectedCount > 0)
            {
                DrawSelectionInfo(viewportWidth - 10, viewportHeight - 10, stats);
            }

            // Draw help hint in top-right
            DrawHelpHint(viewportWidth - 10, 10);

            GL.Enable(EnableCap.DepthTest);

            // Restore matrices
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Projection);
            GL.PopMatrix();
            GL.MatrixMode(MatrixMode.Modelview);
        }

        private void DrawInfoPanel(float x, float y, ViewportStats stats)
        {
            var lines = new List<string>();

            if (ShowFPS)
            {
                lines.Add($"FPS: {stats.FPS:F1}");
            }

            if (ShowObjectCount)
            {
                lines.Add($"Objects: {stats.VisibleObjectCount}");
                if (stats.TotalVertexCount > 0)
                {
                    lines.Add($"Vertices: {FormatNumber(stats.TotalVertexCount)}");
                }
                if (stats.TotalTriangleCount > 0)
                {
                    lines.Add($"Triangles: {FormatNumber(stats.TotalTriangleCount)}");
                }
            }

            if (ShowCameraInfo)
            {
                lines.Add($"Zoom: {Math.Abs(stats.CameraZoom):F2}");
            }

            if (lines.Count == 0) return;

            // Calculate panel size
            int maxWidth = 0;
            foreach (var line in lines)
            {
                var (w, _) = _textRenderer.MeasureText(line);
                maxWidth = Math.Max(maxWidth, w);
            }

            int lineHeight = (int)(_textRenderer.FontSize * 1.4f);
            int panelHeight = lines.Count * lineHeight + 10;
            int panelWidth = maxWidth + 20;

            // Draw background
            GL.Color4(BackgroundColor.r, BackgroundColor.g, BackgroundColor.b, BackgroundColor.a);
            DrawRoundedRect(x, y, panelWidth, panelHeight, 4);

            // Draw lines
            float textY = y + 6;
            foreach (var line in lines)
            {
                var color = TextColor;

                // Highlight FPS based on performance
                if (line.StartsWith("FPS:"))
                {
                    if (stats.FPS >= 30) color = (0.4f, 1.0f, 0.4f, 1.0f); // Green
                    else if (stats.FPS >= 15) color = HighlightColor;
                    else color = WarningColor;
                }

                _textRenderer.DrawText(line, x + 8, textY, color, (0, 0, 0, 0.5f));
                textY += lineHeight;
            }
        }

        private void DrawGizmoModeIndicator(float x, float y, string gizmoMode)
        {
            string text = $"Mode: {gizmoMode}";
            var (width, height) = _textRenderer.MeasureText(text);

            // Background
            GL.Color4(BackgroundColor.r, BackgroundColor.g, BackgroundColor.b, BackgroundColor.a);
            DrawRoundedRect(x, y, width + 16, height + 8, 4);

            // Text with appropriate color
            var color = gizmoMode switch
            {
                "Translate" => (1.0f, 0.5f, 0.2f, 1.0f),
                "Rotate" => (0.2f, 0.8f, 0.2f, 1.0f),
                "Scale" => (0.4f, 0.6f, 1.0f, 1.0f),
                _ => TextColor
            };

            _textRenderer.DrawText(text, x + 8, y + 4, color);
        }

        private void DrawSelectionInfo(float x, float y, ViewportStats stats)
        {
            var lines = new List<string>();

            if (stats.SelectedCount == 1 && !string.IsNullOrEmpty(stats.SelectedObjectName))
            {
                lines.Add(stats.SelectedObjectName);
                lines.Add(stats.SelectedObjectType);
                if (stats.SelectedVertexCount > 0)
                {
                    lines.Add($"{FormatNumber(stats.SelectedVertexCount)} verts");
                }
            }
            else
            {
                lines.Add($"{stats.SelectedCount} selected");
            }

            // Calculate panel size
            int maxWidth = 0;
            foreach (var line in lines)
            {
                var (w, _) = _textRenderer.MeasureText(line);
                maxWidth = Math.Max(maxWidth, w);
            }

            int lineHeight = (int)(_textRenderer.FontSize * 1.4f);
            int panelHeight = lines.Count * lineHeight + 10;
            int panelWidth = maxWidth + 20;

            // Position from right edge
            float panelX = x - panelWidth;
            float panelY = y - panelHeight;

            // Draw background
            GL.Color4(BackgroundColor.r, BackgroundColor.g, BackgroundColor.b, BackgroundColor.a);
            DrawRoundedRect(panelX, panelY, panelWidth, panelHeight, 4);

            // Draw lines
            float textY = panelY + 6;
            for (int i = 0; i < lines.Count; i++)
            {
                var color = i == 0 ? HighlightColor : TextColor;
                _textRenderer.DrawText(lines[i], panelX + 8, textY, color);
                textY += lineHeight;
            }
        }

        private void DrawHelpHint(float x, float y)
        {
            string text = "? Help";
            var (width, height) = _textRenderer.MeasureText(text);
            float panelX = x - width - 16;

            GL.Color4(BackgroundColor.r, BackgroundColor.g, BackgroundColor.b, BackgroundColor.a * 0.5f);
            DrawRoundedRect(panelX, y, width + 16, height + 8, 4);

            _textRenderer.DrawText(text, panelX + 8, y + 4, (0.6f, 0.6f, 0.6f, 1.0f));
        }

        private void DrawRoundedRect(float x, float y, float width, float height, float radius)
        {
            // Simple rectangle for now (rounded corners would require more vertices)
            GL.Begin(PrimitiveType.Quads);
            GL.Vertex2(x, y);
            GL.Vertex2(x + width, y);
            GL.Vertex2(x + width, y + height);
            GL.Vertex2(x, y + height);
            GL.End();
        }

        private static string FormatNumber(int number)
        {
            if (number >= 1000000)
                return $"{number / 1000000.0:F1}M";
            if (number >= 1000)
                return $"{number / 1000.0:F1}K";
            return number.ToString();
        }

        public void Dispose()
        {
            _textRenderer?.Dispose();
        }
    }

    /// <summary>
    /// Statistics for the viewport overlay
    /// </summary>
    public class ViewportStats
    {
        public float FPS { get; set; }
        public int VisibleObjectCount { get; set; }
        public int TotalVertexCount { get; set; }
        public int TotalTriangleCount { get; set; }
        public int SelectedCount { get; set; }
        public string SelectedObjectName { get; set; } = "";
        public string SelectedObjectType { get; set; } = "";
        public int SelectedVertexCount { get; set; }
        public string GizmoMode { get; set; } = "Select";
        public float CameraZoom { get; set; }
    }
}
