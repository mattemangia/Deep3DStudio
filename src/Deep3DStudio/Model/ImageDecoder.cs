using System;
using System.IO;
using OpenCvSharp;
using SkiaSharp;

namespace Deep3DStudio.Model
{
    /// <summary>
    /// Provides resilient image decoding that falls back to OpenCV when Skia cannot decode the file.
    /// </summary>
    public static class ImageDecoder
    {
        /// <summary>
        /// Decode an image into an SKBitmap, attempting Skia first and falling back to OpenCV
        /// to support a wider range of formats (e.g., TIFF).
        /// </summary>
        public static SKBitmap DecodeBitmap(string filePath)
        {
            try
            {
                var bitmap = SKBitmap.Decode(filePath);
                if (bitmap != null)
                {
                    return bitmap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Skia decode failed for {filePath}: {ex.Message}");
            }

            try
            {
                using var mat = Cv2.ImRead(filePath, ImreadModes.Unchanged);

                if (mat.Empty())
                {
                    throw new InvalidOperationException("OpenCV returned an empty image.");
                }

                using var bgra = ToBgra(mat);

                var info = new SKImageInfo(bgra.Width, bgra.Height, SKColorType.Bgra8888, SKAlphaType.Unpremul);
                var bitmap = new SKBitmap(info);

                bitmap.WritePixels(info, bgra.Data, bgra.Step());
                return bitmap;
            }
            catch (Exception ex)
            {
                throw new FileNotFoundException($"Unable to decode image: {filePath}", ex);
            }
        }

        private static Mat ToBgra(Mat source)
        {
            if (source.Channels() == 4)
            {
                return source.Clone();
            }

            var destination = new Mat();

            if (source.Channels() == 3)
            {
                Cv2.CvtColor(source, destination, ColorConversionCodes.BGR2BGRA);
            }
            else
            {
                Cv2.CvtColor(source, destination, ColorConversionCodes.GRAY2BGRA);
            }

            return destination;
        }
    }
}
