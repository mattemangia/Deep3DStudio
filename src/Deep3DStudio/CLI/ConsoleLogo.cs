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
                        float max = Math.Max(color.Red, Math.Max(color.Green, color.Blue)) / 255f;
                        float min = Math.Min(color.Red, Math.Min(color.Green, color.Blue)) / 255f;
                        float saturation = max <= 0f ? 0f : (max - min) / max;

                        // Treat dark, low-saturation pixels as background to emphasize the cube/glow.
                        bool isBackground = luminance < 0.12f && saturation < 0.12f;
                        if (color.Alpha < 64 || isBackground)
                        {
                            sb.Append(' ');
                            continue;
                        }

                        float boosted = Math.Clamp((luminance - 0.08f) / 0.92f, 0f, 1f);
                        boosted = Math.Clamp(boosted + saturation * 0.25f, 0f, 1f);

                        // Map luminance 0..1 to ramp index
                        int index = (int)(boosted * (ramp.Length - 1));
                        index = Math.Clamp(index, 0, ramp.Length - 1);
                        sb.Append(ramp[index]);
                    }
                    sb.AppendLine();
                }

                AppendTitle(sb, targetWidth);
                return sb.ToString();
            }
            catch (Exception ex)
            {
                return $"Deep3DStudio (Logo Error: {ex.Message})";
            }
        }

        private static void AppendTitle(StringBuilder sb, int targetWidth)
        {
            const string title = "Deep3D Studio";
            int leftPad = Math.Max(0, (targetWidth - title.Length) / 2);
            sb.AppendLine(new string(' ', leftPad) + title);
        }

    }
}
