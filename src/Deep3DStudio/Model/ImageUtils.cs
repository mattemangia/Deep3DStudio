using System;
using System.IO;
using SkiaSharp;

namespace Deep3DStudio.Model
{
    public static class ImageUtils
    {
        public static (float[] tensor, int[] shape) LoadAndPreprocessImage(string filePath, int size = 512)
        {
            using var decoded = ImageDecoder.DecodeBitmap(filePath);

            int newWidth = decoded.Width;
            int newHeight = decoded.Height;

            float scale = (float)size / Math.Max(newWidth, newHeight);
            if (scale < 1.0f)
            {
                newWidth = (int)(newWidth * scale);
                newHeight = (int)(newHeight * scale);
            }

            newWidth = Math.Max(16, (newWidth / 16) * 16);
            newHeight = Math.Max(16, (newHeight / 16) * 16);

            using var resized = ResizeBitmap(decoded, newWidth, newHeight);

            // Create float array [1, 3, H, W]
            float[] tensor = new float[3 * newHeight * newWidth];

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    var pixel = resized.GetPixel(x, y);

                    // Normalize to [-1, 1]
                    tensor[0 * newHeight * newWidth + y * newWidth + x] = (pixel.Red / 255.0f - 0.5f) / 0.5f;
                    tensor[1 * newHeight * newWidth + y * newWidth + x] = (pixel.Green / 255.0f - 0.5f) / 0.5f;
                    tensor[2 * newHeight * newWidth + y * newWidth + x] = (pixel.Blue / 255.0f - 0.5f) / 0.5f;
                }
            }

            return (tensor, new int[] { 1, 3, newHeight, newWidth });
        }

        public static (int width, int height) GetImageDimensions(string filePath)
        {
            using var decoded = ImageDecoder.DecodeBitmap(filePath);
            return (decoded.Width, decoded.Height);
        }

        public static SKColor[] ExtractColors(string filePath, int width, int height)
        {
            using var decoded = ImageDecoder.DecodeBitmap(filePath);
            using var resized = ResizeBitmap(decoded, width, height);

            var colors = new SKColor[width * height];
            int idx = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    colors[idx++] = resized.GetPixel(x, y);
                }
            }

            return colors;
        }
        
        private static SKBitmap ResizeBitmap(SKBitmap source, int width, int height)
        {
            var info = new SKImageInfo(width, height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            var result = new SKBitmap(info);

            // Use high quality sampling
            source.ScalePixels(result, SKFilterQuality.High);

            return result;
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

        public static SKBitmap ColorizeDepthMap(float[,] depthMap)
        {
            if (depthMap == null) return null;

            int width = depthMap.GetLength(0);
            int height = depthMap.GetLength(1);

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

            var bitmap = new SKBitmap(width, height, SKColorType.Rgba8888, SKAlphaType.Premul);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float d = depthMap[x, y];
                    if (d <= 0)
                    {
                        // Transparent background
                        bitmap.SetPixel(x, y, new SKColor(0, 0, 0, 0));
                    }
                    else
                    {
                        float t = (d - minDepth) / range;
                        // t = Math.Clamp(t, 0f, 1f);

                        var (r, g, b) = TurboColormap(t);
                        bitmap.SetPixel(x, y, new SKColor((byte)(r * 255), (byte)(g * 255), (byte)(b * 255), 255));
                    }
                }
            }
            return bitmap;
        }
    }
}
