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
                // Try reading with codec first to handle EXIF orientation
                using var codec = SKCodec.Create(filePath);
                if (codec != null)
                {
                    var bitmap = SKBitmap.Decode(codec);
                    if (bitmap != null)
                    {
                        return ApplyOrientation(bitmap, codec.EncodedOrigin);
                    }
                }

                // Fallback to direct decode if codec failed but might still work
                var directBitmap = SKBitmap.Decode(filePath);
                if (directBitmap != null)
                {
                    return directBitmap;
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

                unsafe
                {
                    byte* srcPtr = (byte*)bgra.Data;
                    byte* dstPtr = (byte*)bitmap.GetPixels();
                    long height = bgra.Height;
                    long widthInBytes = bgra.Width * 4; // 4 bytes per pixel for BGRA8888
                    long srcStep = bgra.Step();
                    long dstStep = bitmap.RowBytes;

                    for (int y = 0; y < height; y++)
                    {
                        Buffer.MemoryCopy(srcPtr, dstPtr, widthInBytes, widthInBytes);
                        srcPtr += srcStep;
                        dstPtr += dstStep;
                    }
                }
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

        /// <summary>
        /// Decodes an image to an OpenCV Mat (BGR), handling EXIF orientation via SkiaSharp.
        /// </summary>
        public static Mat DecodeToMat(string filePath)
        {
            var bitmap = DecodeBitmap(filePath);
            if (bitmap == null)
                throw new FileNotFoundException($"Unable to decode image: {filePath}");

            // Ensure BGRA8888 for consistent memory layout
            SKBitmap source = bitmap;
            bool disposeSource = false;

            // Note: DecodeBitmap returns a bitmap we own (using var in caller would dispose it).
            // Here we assigned it to 'bitmap' variable which we must dispose.

            try
            {
                if (bitmap.ColorType != SKColorType.Bgra8888)
                {
                    source = bitmap.Copy(SKColorType.Bgra8888);
                    disposeSource = true; // We created a new copy, we must dispose it
                    // The original 'bitmap' will be disposed in finally block
                }

                if (source == null)
                    throw new InvalidOperationException("Failed to convert bitmap to BGRA8888");

                var mat = new Mat(source.Height, source.Width, MatType.CV_8UC4);

                unsafe
                {
                    // Copy row by row to handle stride (padding) differences safely
                    byte* srcPtr = (byte*)source.GetPixels();
                    byte* dstPtr = (byte*)mat.Data;
                    long height = source.Height;
                    long widthInBytes = source.Width * 4; // 4 bytes per pixel for BGRA8888
                    long srcStep = source.RowBytes;
                    long dstStep = mat.Step(); // OpenCV Mat step (stride)

                    for (int y = 0; y < height; y++)
                    {
                        Buffer.MemoryCopy(srcPtr, dstPtr, widthInBytes, widthInBytes);
                        srcPtr += srcStep;
                        dstPtr += dstStep;
                    }
                }

                var result = new Mat();
                Cv2.CvtColor(mat, result, ColorConversionCodes.BGRA2BGR);
                mat.Dispose();

                return result;
            }
            finally
            {
                if (disposeSource && source != null) source.Dispose();
                bitmap.Dispose();
            }
        }

        private static SKBitmap ApplyOrientation(SKBitmap bitmap, SKEncodedOrigin origin)
        {
            if (origin == SKEncodedOrigin.TopLeft)
                return bitmap;

            SKBitmap newBitmap;

            // For simple rotations/flips, we can use Canvas while ensuring dimensions remain correct.

            int width = bitmap.Width;
            int height = bitmap.Height;

            // Use a minimal set of transformations to handle the supported orientations.
            switch (origin)
            {
                case SKEncodedOrigin.BottomRight: // Rotate 180
                    newBitmap = new SKBitmap(width, height);
                    using (var canvas = new SKCanvas(newBitmap))
                    {
                        canvas.RotateDegrees(180, width / 2f, height / 2f);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                    break;

                case SKEncodedOrigin.RightTop: // Rotate 90 CW
                    newBitmap = new SKBitmap(height, width);
                    using (var canvas = new SKCanvas(newBitmap))
                    {
                        canvas.Translate(newBitmap.Width, 0);
                        canvas.RotateDegrees(90);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                    break;

                case SKEncodedOrigin.LeftBottom: // Rotate 270 CW
                    newBitmap = new SKBitmap(height, width);
                    using (var canvas = new SKCanvas(newBitmap))
                    {
                        canvas.Translate(0, newBitmap.Height);
                        canvas.RotateDegrees(270);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                    break;

                case SKEncodedOrigin.TopRight: // Flip Horizontal
                    newBitmap = new SKBitmap(width, height);
                    using (var canvas = new SKCanvas(newBitmap))
                    {
                        canvas.Scale(-1, 1, width / 2f, height / 2f);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                    break;

                 case SKEncodedOrigin.BottomLeft: // Flip Vertical
                    newBitmap = new SKBitmap(width, height);
                    using (var canvas = new SKCanvas(newBitmap))
                    {
                        canvas.Scale(1, -1, width / 2f, height / 2f);
                        canvas.DrawBitmap(bitmap, 0, 0);
                    }
                    break;

                 // For the rarer transpose/transverse, we can combine operations or skip for now if unsure,
                 // but phone cameras mostly use TopLeft, RightTop (90), BottomRight (180), LeftBottom (270).
                 // The others are rare.

                 default:
                     return bitmap;
            }

            bitmap.Dispose();
            return newBitmap;
        }
    }
}
