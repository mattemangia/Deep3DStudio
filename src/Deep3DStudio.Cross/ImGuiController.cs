using System;
using ImGuiNET;
using OpenTK.Mathematics;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;
using OpenTK.Graphics.OpenGL;
using System.Collections.Generic;

namespace Deep3DStudio
{
    public class ImGuiController : IDisposable
    {
        private bool _frameBegun;
        private int _vertexArray;
        private int _vertexBuffer;
        private int _vertexBufferSize;
        private int _indexBuffer;
        private int _indexBufferSize;

        private int _fontTexture;
        private int _shader;
        private bool _shaderValid = false;
        private int _shaderFontTextureLocation;
        private int _shaderProjectionMatrixLocation;

        private int _windowWidth;
        private int _windowHeight;
        private System.Numerics.Vector2 _scaleFactor = System.Numerics.Vector2.One;

        public ImGuiController(int width, int height)
        {
            _windowWidth = width;
            _windowHeight = height;

            IntPtr context = ImGui.CreateContext();
            ImGui.SetCurrentContext(context);
            var io = ImGui.GetIO();
            io.BackendFlags |= ImGuiBackendFlags.RendererHasVtxOffset;

            // Fonts
            io.Fonts.AddFontDefault();

            CreateDeviceObjects();
            SetPerFrameImGuiData(1f / 60f);

            ImGui.NewFrame();
            _frameBegun = true;
        }

        public void WindowResized(int width, int height)
        {
            _windowWidth = width;
            _windowHeight = height;
        }

        public void DestroyDeviceObjects()
        {
            Dispose();
        }

        public void CreateDeviceObjects()
        {
            // Vertex buffer
            _vertexArray = GL.GenVertexArray();
            _vertexBuffer = GL.GenBuffer();
            _indexBuffer = GL.GenBuffer();

            _vertexBufferSize = 10000;
            _indexBufferSize = 2000;

            GL.BindVertexArray(_vertexArray);
            GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBuffer);
            GL.BufferData(BufferTarget.ArrayBuffer, _vertexBufferSize, IntPtr.Zero, BufferUsageHint.DynamicDraw);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, _indexBuffer);
            GL.BufferData(BufferTarget.ElementArrayBuffer, _indexBufferSize, IntPtr.Zero, BufferUsageHint.DynamicDraw);

            // Shader (Compatibility Profile / Legacy GL 2.1 friendly)
            var vertexSource = @"#version 120
attribute vec2 Position;
attribute vec2 UV;
attribute vec4 Color;
uniform mat4 ProjMtx;
varying vec2 Frag_UV;
varying vec4 Frag_Color;
void main()
{
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy,0,1);
}";
            var fragmentSource = @"#version 120
varying vec2 Frag_UV;
varying vec4 Frag_Color;
uniform sampler2D Texture;
void main()
{
    gl_FragColor = Frag_Color * texture2D(Texture, Frag_UV.st);
}";
            _shader = CreateProgram("ImGui", vertexSource, fragmentSource);
            if (_shader != 0)
            {
                _shaderValid = true;
                _shaderFontTextureLocation = GL.GetUniformLocation(_shader, "Texture");
                _shaderProjectionMatrixLocation = GL.GetUniformLocation(_shader, "ProjMtx");
            }

            int attribLocationPosition = GL.GetAttribLocation(_shader, "Position");
            int attribLocationUV = GL.GetAttribLocation(_shader, "UV");
            int attribLocationColor = GL.GetAttribLocation(_shader, "Color");

            GL.EnableVertexAttribArray(attribLocationPosition);
            GL.VertexAttribPointer(attribLocationPosition, 2, VertexAttribPointerType.Float, false, 20, 0);

            GL.EnableVertexAttribArray(attribLocationUV);
            GL.VertexAttribPointer(attribLocationUV, 2, VertexAttribPointerType.Float, false, 20, 8);

            GL.EnableVertexAttribArray(attribLocationColor);
            GL.VertexAttribPointer(attribLocationColor, 4, VertexAttribPointerType.UnsignedByte, true, 20, 16);

            RecreateFontDeviceTexture();

            GL.BindVertexArray(0);
        }

        public void RecreateFontDeviceTexture()
        {
            ImGuiIOPtr io = ImGui.GetIO();
            IntPtr pixels;
            int width, height, bytesPerPixel;
            io.Fonts.GetTexDataAsRGBA32(out pixels, out width, out height, out bytesPerPixel);

            int mips = (int)Math.Floor(Math.Log(Math.Max(width, height), 2));
            _fontTexture = GL.GenTexture();
            GL.BindTexture(TextureTarget.Texture2D, _fontTexture);
            GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, width, height, 0, PixelFormat.Bgra, PixelType.UnsignedByte, pixels);

            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
            GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);

            GL.BindTexture(TextureTarget.Texture2D, 0);

            io.Fonts.SetTexID((IntPtr)_fontTexture);
            io.Fonts.ClearTexData();
        }

        public void Update(GameWindow wnd, float dt)
        {
            if (_frameBegun) ImGui.Render();
            SetPerFrameImGuiData(dt);
            UpdateImGuiInput(wnd);
            _frameBegun = true;
            ImGui.NewFrame();
        }

        private void SetPerFrameImGuiData(float dt)
        {
            ImGuiIOPtr io = ImGui.GetIO();
            io.DisplaySize = new System.Numerics.Vector2(_windowWidth, _windowHeight);
            io.DisplayFramebufferScale = _scaleFactor;
            io.DeltaTime = dt;
        }

        private void UpdateImGuiInput(GameWindow wnd)
        {
            ImGuiIOPtr io = ImGui.GetIO();
            var mouseState = wnd.MouseState;
            var keyboardState = wnd.KeyboardState;

            io.MouseDown[0] = mouseState[MouseButton.Left];
            io.MouseDown[1] = mouseState[MouseButton.Right];
            io.MouseDown[2] = mouseState[MouseButton.Middle];
            io.MousePos = new System.Numerics.Vector2(mouseState.X, mouseState.Y);

            io.KeyCtrl = keyboardState.IsKeyDown(Keys.LeftControl) || keyboardState.IsKeyDown(Keys.RightControl);
            io.KeyAlt = keyboardState.IsKeyDown(Keys.LeftAlt) || keyboardState.IsKeyDown(Keys.RightAlt);
            io.KeyShift = keyboardState.IsKeyDown(Keys.LeftShift) || keyboardState.IsKeyDown(Keys.RightShift);
            io.KeySuper = keyboardState.IsKeyDown(Keys.LeftSuper) || keyboardState.IsKeyDown(Keys.RightSuper);
        }

        public void PressChar(char keyChar)
        {
            ImGui.GetIO().AddInputCharacter(keyChar);
        }

        public void MouseScroll(System.Numerics.Vector2 offset)
        {
            ImGuiIOPtr io = ImGui.GetIO();
            io.MouseWheel = offset.Y;
            io.MouseWheelH = offset.X;
        }

        public void Render()
        {
            if (_frameBegun)
            {
                _frameBegun = false;
                ImGui.Render();
                RenderImDrawData(ImGui.GetDrawData());
            }
        }

        private void RenderImDrawData(ImDrawDataPtr draw_data)
        {
            if (draw_data.CmdListsCount == 0) return;
            if (!_shaderValid) return;

            GL.Viewport(0, 0, _windowWidth, _windowHeight);

            GL.Enable(EnableCap.Blend);
            GL.Enable(EnableCap.ScissorTest);
            GL.BlendEquation(BlendEquationMode.FuncAdd);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            GL.Disable(EnableCap.CullFace);
            GL.Disable(EnableCap.DepthTest);

            GL.UseProgram(_shader);
            GL.BindVertexArray(_vertexArray);
            CheckError("ImGui Setup");

            float L = draw_data.DisplayPos.X;
            float R = draw_data.DisplayPos.X + draw_data.DisplaySize.X;
            float T = draw_data.DisplayPos.Y;
            float B = draw_data.DisplayPos.Y + draw_data.DisplaySize.Y;

            Matrix4 orthoProjection = Matrix4.CreateOrthographicOffCenter(L, R, B, T, -1.0f, 1.0f);
            GL.UniformMatrix4(_shaderProjectionMatrixLocation, false, ref orthoProjection);
            GL.Uniform1(_shaderFontTextureLocation, 0);

            for (int n = 0; n < draw_data.CmdListsCount; n++)
            {
                ImDrawListPtr cmd_list = draw_data.CmdLists[n];

                GL.BindBuffer(BufferTarget.ArrayBuffer, _vertexBuffer);
                GL.BufferData(BufferTarget.ArrayBuffer, cmd_list.VtxBuffer.Size * 20, cmd_list.VtxBuffer.Data, BufferUsageHint.StreamDraw);

                GL.BindBuffer(BufferTarget.ElementArrayBuffer, _indexBuffer);
                GL.BufferData(BufferTarget.ElementArrayBuffer, cmd_list.IdxBuffer.Size * 2, cmd_list.IdxBuffer.Data, BufferUsageHint.StreamDraw);

                int vtx_offset = 0;
                int idx_offset = 0;

                for (int cmd_i = 0; cmd_i < cmd_list.CmdBuffer.Size; cmd_i++)
                {
                    ImDrawCmdPtr pcmd = cmd_list.CmdBuffer[cmd_i];
                    if (pcmd.UserCallback != IntPtr.Zero)
                    {
                        Console.WriteLine("ImGui UserCallback not implemented");
                    }
                    else
                    {
                        GL.ActiveTexture(TextureUnit.Texture0);
                        if (pcmd.TextureId != IntPtr.Zero)
                        {
                            GL.BindTexture(TextureTarget.Texture2D, (int)pcmd.TextureId);
                        }

                        GL.Scissor((int)pcmd.ClipRect.X, _windowHeight - (int)pcmd.ClipRect.W, (int)(pcmd.ClipRect.Z - pcmd.ClipRect.X), (int)(pcmd.ClipRect.W - pcmd.ClipRect.Y));

                        GL.DrawElements(BeginMode.Triangles, (int)pcmd.ElemCount, DrawElementsType.UnsignedShort, idx_offset * 2);
                        CheckError("ImGui DrawElements");
                    }
                    idx_offset += (int)pcmd.ElemCount;
                }
            }

            // Cleanup state to prevent leakage to ThreeDView
            GL.BindVertexArray(0);
            GL.BindBuffer(BufferTarget.ArrayBuffer, 0);
            GL.BindBuffer(BufferTarget.ElementArrayBuffer, 0);
            GL.Disable(EnableCap.Blend);
            GL.Disable(EnableCap.ScissorTest);
        }

        private void CheckError(string stage)
        {
            var err = GL.GetError();
            if (err != OpenTK.Graphics.OpenGL.ErrorCode.NoError && err != OpenTK.Graphics.OpenGL.ErrorCode.InvalidFramebufferOperation)
            {
                Console.WriteLine($"OpenGL Error at ImGui {stage}: {err}");
            }
        }

        private int CreateProgram(string name, string vertexSource, string fragmentSource)
        {
            int program = GL.CreateProgram();
            int vertex = CompileShader(name, ShaderType.VertexShader, vertexSource);
            int fragment = CompileShader(name, ShaderType.FragmentShader, fragmentSource);

            GL.AttachShader(program, vertex);
            GL.AttachShader(program, fragment);
            GL.LinkProgram(program);

            GL.GetProgram(program, GetProgramParameterName.LinkStatus, out int success);
            if (success == 0)
            {
                string info = GL.GetProgramInfoLog(program);
                Console.WriteLine($"GL.LinkProgram had info log [{name}]:\n{info}");
                // Clean up if link failed
                GL.DeleteProgram(program);
                program = 0;
            }

            GL.DetachShader(program, vertex);
            GL.DetachShader(program, fragment);
            GL.DeleteShader(vertex);
            GL.DeleteShader(fragment);

            return program;
        }

        private int CompileShader(string name, ShaderType type, string source)
        {
            int shader = GL.CreateShader(type);
            GL.ShaderSource(shader, source);
            GL.CompileShader(shader);

            GL.GetShader(shader, ShaderParameter.CompileStatus, out int success);
            if (success == 0)
            {
                string info = GL.GetShaderInfoLog(shader);
                Console.WriteLine($"GL.CompileShader had info log [{name}]:\n{info}");
            }

            return shader;
        }

        public void Dispose()
        {
            GL.DeleteVertexArray(_vertexArray);
            GL.DeleteBuffer(_vertexBuffer);
            GL.DeleteBuffer(_indexBuffer);
            GL.DeleteTexture(_fontTexture);
            GL.DeleteProgram(_shader);
        }
    }
}
