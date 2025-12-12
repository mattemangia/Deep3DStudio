using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace Deep3DStudio.Model
{
    public static class ImageUtils
    {
        public static (DenseTensor<float> tensor, int[] shape) LoadAndPreprocessImage(string filePath, int size = 512)
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

            using var rgba = decoded.Copy(SKColorType.Rgba8888)
                ?? throw new InvalidOperationException("Unable to convert image to RGBA8888.");

            using var resized = ResizeBitmap(rgba, newWidth, newHeight);

            var tensor = new DenseTensor<float>(new[] { 1, 3, newHeight, newWidth });

            for (int y = 0; y < newHeight; y++)
            {
                for (int x = 0; x < newWidth; x++)
                {
                    var pixel = resized.GetPixel(x, y);

                    tensor[0, 0, y, x] = (pixel.Red / 255.0f - 0.5f) / 0.5f;
                    tensor[0, 1, y, x] = (pixel.Green / 255.0f - 0.5f) / 0.5f;
                    tensor[0, 2, y, x] = (pixel.Blue / 255.0f - 0.5f) / 0.5f;
                }
            }

            return (tensor, new int[] { resized.Height, resized.Width });
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

            using var canvas = new SKCanvas(result);
            using var image = SKImage.FromBitmap(source);
            canvas.Clear(SKColors.Transparent);
            canvas.DrawImage(image, new SKRect(0, 0, width, height));
            canvas.Flush();

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
    }
}
