using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Deep3DStudio.Model
{
    public static class ImageUtils
    {
        public static (DenseTensor<float> tensor, int[] shape) LoadAndPreprocessImage(string filePath, int size = 512)
        {
            using (var image = Image.Load<Rgb24>(filePath))
            {
                // Resize to multiple of 16
                // For simplicity, we can resize to a fixed size or nearest multiple of 16.
                // Dust3r handles dynamic sizes, but it's good to keep it manageable.
                // Let's resize ensuring max dimension is 'size' and keeping aspect ratio,
                // but ensuring dimensions are multiples of 16.

                var newWidth = image.Width;
                var newHeight = image.Height;

                // Simple logic: resize so largest edge is 512, then round to 16
                float scale = (float)size / Math.Max(newWidth, newHeight);
                if (scale < 1.0f)
                {
                    newWidth = (int)(newWidth * scale);
                    newHeight = (int)(newHeight * scale);
                }

                // Align to 16
                newWidth = (newWidth / 16) * 16;
                newHeight = (newHeight / 16) * 16;
                if (newWidth == 0) newWidth = 16;
                if (newHeight == 0) newHeight = 16;

                image.Mutate(x => x.Resize(newWidth, newHeight));

                var tensor = new DenseTensor<float>(new[] { 1, 3, newHeight, newWidth });

                // Normalization constants for Dust3r (mean=0.5, std=0.5)
                // Pixel values 0..255 -> 0..1 -> (val - 0.5)/0.5 = 2*val - 1

                image.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        var pixelRow = accessor.GetRowSpan(y);
                        for (int x = 0; x < accessor.Width; x++)
                        {
                            var pixel = pixelRow[x];

                            // Batch 0
                            tensor[0, 0, y, x] = (pixel.R / 255.0f - 0.5f) / 0.5f;
                            tensor[0, 1, y, x] = (pixel.G / 255.0f - 0.5f) / 0.5f;
                            tensor[0, 2, y, x] = (pixel.B / 255.0f - 0.5f) / 0.5f;
                        }
                    }
                });

                return (tensor, new int[] { image.Height, image.Width });
            }
        }

        public static SixLabors.ImageSharp.Color[] ExtractColors(string filePath, int width, int height)
        {
             // Helper to reload original image and extract colors matching the tensor shape
             // This assumes the file on disk hasn't changed.
             // Ideally we should do this during the first pass but for now we reload.
             using (var image = Image.Load<Rgb24>(filePath))
             {
                 image.Mutate(x => x.Resize(width, height));
                 var colors = new SixLabors.ImageSharp.Color[width * height];
                 int idx = 0;
                 for (int y = 0; y < height; y++)
                 {
                     for (int x = 0; x < width; x++)
                     {
                         colors[idx++] = image[x, y];
                     }
                 }
                 return colors;
             }
        }
    }
}
