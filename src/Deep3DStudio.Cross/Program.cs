using System;
using OpenTK.Windowing.Desktop;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using ImGuiNET;
using Deep3DStudio.Configuration;
using Deep3DStudio.Python;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Deep3DStudio
{
    class Program
    {
        [STAThread]
        public static void Main(string[] args)
        {
            // Force legacy OpenGL support for ThreeDView
            Environment.SetEnvironmentVariable("MESA_GL_VERSION_OVERRIDE", "3.3COMPAT");

            var nativeWindowSettings = new NativeWindowSettings()
            {
                Size = new Vector2i(1280, 720),
                Title = "Deep3DStudio (Cross-Platform / ImGui)",
                Flags = ContextFlags.Default
            };

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                // macOS only supports legacy OpenGL up to 2.1.
                // Requesting 3.3 Compat causes a crash.
                nativeWindowSettings.APIVersion = new Version(2, 1);
                nativeWindowSettings.Profile = ContextProfile.Any;
            }
            else
            {
                // Windows/Linux support 3.3 Compatibility Profile
                nativeWindowSettings.APIVersion = new Version(3, 3);
                nativeWindowSettings.Profile = ContextProfile.Compatability;
            }

            using (var window = new MainWindow(GameWindowSettings.Default, nativeWindowSettings))
            {
                window.Run();
            }
        }
    }
}
