using System;
using System.Reflection;
using System.Text;
using SkiaSharp;

namespace Deep3DStudio.CLI
{
    public static class ConsoleLogo
    {
        public static string GenerateAsciiLogo(int targetWidth = 80)
        {
            try
            {
                var assembly = Assembly.GetExecutingAssembly();
                string? resourceName = null;
                foreach (var name in assembly.GetManifestResourceNames())
                {
                    if (name.EndsWith("logo.png"))
                    {
                        resourceName = name;
                        break;
                    }
                }

                if (resourceName == null) return "Deep3DStudio (Logo Resource Not Found)";

                using var stream = assembly.GetManifestResourceStream(resourceName);
                if (stream == null) return "Deep3DStudio";

                using var bitmap = SKBitmap.Decode(stream);
                if (bitmap == null) return "Deep3DStudio";

                // Console characters are roughly 1:2 width:height, so we halve the height to maintain aspect ratio
                int height = (int)(bitmap.Height * ((float)targetWidth / bitmap.Width) * 0.5f);
                
                using var resized = bitmap.Resize(new SKImageInfo(targetWidth, height), SKFilterQuality.High);
                if (resized == null) return "Deep3DStudio";

                var sb = new StringBuilder();
                // Unicode shading ramp: Space, Light, Medium, Dark, Full
                string ramp = " ░▒▓█";

                for (int y = 0; y < resized.Height; y++)
                {
                    for (int x = 0; x < resized.Width; x++)
                    {
                        var color = resized.GetPixel(x, y);
                        // Simple luminance: 0.299R + 0.587G + 0.114B
                        float luminance = (0.299f * color.Red + 0.587f * color.Green + 0.114f * color.Blue) / 255f;
                        
                        if (color.Alpha < 64) // Threshold for transparency
                        {
                            sb.Append(' ');
                        }
                        else
                        {
                            // Map luminance 0..1 to ramp index
                            int index = (int)(luminance * (ramp.Length - 1));
                            index = Math.Clamp(index, 0, ramp.Length - 1);
                            sb.Append(ramp[index]);
                        }
                    }
                    sb.AppendLine();
                }

                return sb.ToString();
            }
            catch (Exception ex)
            {
                return $"Deep3DStudio (Logo Error: {ex.Message})";
            }
        }
    }
}
