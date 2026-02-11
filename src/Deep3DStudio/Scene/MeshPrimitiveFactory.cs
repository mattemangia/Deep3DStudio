using System;
using System.Collections.Generic;
using OpenTK.Mathematics;
using Deep3DStudio.Model;

namespace Deep3DStudio.Scene
{
    public enum MeshPrimitiveType
    {
        Plane,
        Cube,
        UVSphere,
        Cylinder,
        Cone,
        Torus,
        Circle,
        Polygon,
        Grid
    }

    /// <summary>
    /// Generates common editable polygonal primitives.
    /// </summary>
    public static class MeshPrimitiveFactory
    {
        public static MeshData CreatePrimitive(MeshPrimitiveType type)
        {
            return type switch
            {
                MeshPrimitiveType.Plane => CreatePlane(1.0f, 1.0f),
                MeshPrimitiveType.Cube => CreateCube(1.0f),
                MeshPrimitiveType.UVSphere => CreateUVSphere(0.5f, 24, 16),
                MeshPrimitiveType.Cylinder => CreateCylinder(0.5f, 1.0f, 24, true),
                MeshPrimitiveType.Cone => CreateCone(0.5f, 1.0f, 24, true),
                MeshPrimitiveType.Torus => CreateTorus(0.65f, 0.2f, 28, 16),
                MeshPrimitiveType.Circle => CreateCircle(0.6f, 40),
                MeshPrimitiveType.Polygon => CreatePolygon(6, 0.6f),
                MeshPrimitiveType.Grid => CreateGrid(12, 0.1f),
                _ => CreateCube(1.0f)
            };
        }

        public static MeshData CreatePlane(float width, float depth)
        {
            var hw = width * 0.5f;
            var hd = depth * 0.5f;
            var mesh = new MeshData
            {
                Vertices = new List<Vector3>
                {
                    new Vector3(-hw, 0, -hd),
                    new Vector3(hw, 0, -hd),
                    new Vector3(hw, 0, hd),
                    new Vector3(-hw, 0, hd)
                },
                Indices = new List<int>
                {
                    0, 1, 2,
                    0, 2, 3
                },
                Colors = CreateColorList(4, new Vector3(0.8f, 0.82f, 0.86f))
            };

            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateCube(float size)
        {
            float h = size * 0.5f;
            var mesh = new MeshData
            {
                Vertices = new List<Vector3>
                {
                    new Vector3(-h, -h, -h), new Vector3(h, -h, -h), new Vector3(h, h, -h), new Vector3(-h, h, -h),
                    new Vector3(-h, -h, h),  new Vector3(h, -h, h),  new Vector3(h, h, h),  new Vector3(-h, h, h)
                },
                Indices = new List<int>
                {
                    0, 1, 2, 0, 2, 3, // back
                    4, 6, 5, 4, 7, 6, // front
                    0, 4, 5, 0, 5, 1, // bottom
                    3, 2, 6, 3, 6, 7, // top
                    1, 5, 6, 1, 6, 2, // right
                    0, 3, 7, 0, 7, 4  // left
                },
                Colors = CreateColorList(8, new Vector3(0.86f, 0.76f, 0.62f))
            };

            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateUVSphere(float radius, int segments, int rings)
        {
            segments = Math.Max(6, segments);
            rings = Math.Max(4, rings);

            var vertices = new List<Vector3>();
            var indices = new List<int>();

            for (int r = 0; r <= rings; r++)
            {
                float v = r / (float)rings;
                float phi = v * MathF.PI;
                float y = MathF.Cos(phi);
                float sinPhi = MathF.Sin(phi);

                for (int s = 0; s <= segments; s++)
                {
                    float u = s / (float)segments;
                    float theta = u * MathF.PI * 2f;
                    float x = MathF.Cos(theta) * sinPhi;
                    float z = MathF.Sin(theta) * sinPhi;
                    vertices.Add(new Vector3(x, y, z) * radius);
                }
            }

            int stride = segments + 1;
            for (int r = 0; r < rings; r++)
            {
                for (int s = 0; s < segments; s++)
                {
                    int i0 = r * stride + s;
                    int i1 = i0 + 1;
                    int i2 = i0 + stride;
                    int i3 = i2 + 1;

                    indices.Add(i0); indices.Add(i2); indices.Add(i1);
                    indices.Add(i1); indices.Add(i2); indices.Add(i3);
                }
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.72f, 0.83f, 0.96f))
            };

            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateCylinder(float radius, float height, int segments, bool capEnds)
        {
            segments = Math.Max(6, segments);
            var vertices = new List<Vector3>();
            var indices = new List<int>();
            float h = height * 0.5f;

            for (int i = 0; i <= segments; i++)
            {
                float t = i / (float)segments;
                float angle = t * MathF.PI * 2f;
                float x = MathF.Cos(angle) * radius;
                float z = MathF.Sin(angle) * radius;
                vertices.Add(new Vector3(x, -h, z));
                vertices.Add(new Vector3(x, h, z));
            }

            for (int i = 0; i < segments; i++)
            {
                int i0 = i * 2;
                int i1 = i0 + 1;
                int i2 = i0 + 2;
                int i3 = i0 + 3;

                indices.Add(i0); indices.Add(i1); indices.Add(i2);
                indices.Add(i2); indices.Add(i1); indices.Add(i3);
            }

            if (capEnds)
            {
                int bottomCenter = vertices.Count;
                vertices.Add(new Vector3(0, -h, 0));
                int topCenter = vertices.Count;
                vertices.Add(new Vector3(0, h, 0));

                for (int i = 0; i < segments; i++)
                {
                    int b0 = i * 2;
                    int b1 = ((i + 1) % segments) * 2;
                    int t0 = b0 + 1;
                    int t1 = b1 + 1;

                    indices.Add(bottomCenter); indices.Add(b1); indices.Add(b0);
                    indices.Add(topCenter); indices.Add(t0); indices.Add(t1);
                }
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.82f, 0.84f, 0.67f))
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateCone(float radius, float height, int segments, bool capBase)
        {
            segments = Math.Max(6, segments);
            var vertices = new List<Vector3>();
            var indices = new List<int>();
            float h = height * 0.5f;

            int apex = 0;
            vertices.Add(new Vector3(0, h, 0));

            for (int i = 0; i < segments; i++)
            {
                float t = i / (float)segments;
                float angle = t * MathF.PI * 2f;
                vertices.Add(new Vector3(MathF.Cos(angle) * radius, -h, MathF.Sin(angle) * radius));
            }

            for (int i = 0; i < segments; i++)
            {
                int b0 = 1 + i;
                int b1 = 1 + ((i + 1) % segments);
                indices.Add(apex); indices.Add(b0); indices.Add(b1);
            }

            if (capBase)
            {
                int center = vertices.Count;
                vertices.Add(new Vector3(0, -h, 0));
                for (int i = 0; i < segments; i++)
                {
                    int b0 = 1 + i;
                    int b1 = 1 + ((i + 1) % segments);
                    indices.Add(center); indices.Add(b1); indices.Add(b0);
                }
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.9f, 0.72f, 0.56f))
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateTorus(float majorRadius, float minorRadius, int majorSegments, int minorSegments)
        {
            majorSegments = Math.Max(8, majorSegments);
            minorSegments = Math.Max(6, minorSegments);

            var vertices = new List<Vector3>();
            var indices = new List<int>();

            for (int i = 0; i <= majorSegments; i++)
            {
                float u = i / (float)majorSegments * MathF.PI * 2f;
                float cu = MathF.Cos(u);
                float su = MathF.Sin(u);

                for (int j = 0; j <= minorSegments; j++)
                {
                    float v = j / (float)minorSegments * MathF.PI * 2f;
                    float cv = MathF.Cos(v);
                    float sv = MathF.Sin(v);

                    float x = (majorRadius + minorRadius * cv) * cu;
                    float y = minorRadius * sv;
                    float z = (majorRadius + minorRadius * cv) * su;
                    vertices.Add(new Vector3(x, y, z));
                }
            }

            int stride = minorSegments + 1;
            for (int i = 0; i < majorSegments; i++)
            {
                for (int j = 0; j < minorSegments; j++)
                {
                    int i0 = i * stride + j;
                    int i1 = i0 + 1;
                    int i2 = i0 + stride;
                    int i3 = i2 + 1;

                    indices.Add(i0); indices.Add(i2); indices.Add(i1);
                    indices.Add(i1); indices.Add(i2); indices.Add(i3);
                }
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.74f, 0.9f, 0.77f))
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateCircle(float radius, int segments)
        {
            return CreatePolygon(Math.Max(3, segments), radius);
        }

        public static MeshData CreatePolygon(int sides, float radius)
        {
            sides = Math.Max(3, sides);
            var vertices = new List<Vector3> { Vector3.Zero };
            var indices = new List<int>();

            for (int i = 0; i < sides; i++)
            {
                float angle = i / (float)sides * MathF.PI * 2f;
                vertices.Add(new Vector3(MathF.Cos(angle) * radius, 0, MathF.Sin(angle) * radius));
            }

            for (int i = 1; i <= sides; i++)
            {
                int next = i == sides ? 1 : i + 1;
                indices.Add(0);
                indices.Add(i);
                indices.Add(next);
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.76f, 0.78f, 0.92f))
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        public static MeshData CreateGrid(int cellsPerSide, float cellSize)
        {
            cellsPerSide = Math.Max(1, cellsPerSide);
            cellSize = Math.Max(0.001f, cellSize);

            float size = cellsPerSide * cellSize;
            float half = size * 0.5f;

            var vertices = new List<Vector3>();
            var indices = new List<int>();

            for (int z = 0; z <= cellsPerSide; z++)
            {
                for (int x = 0; x <= cellsPerSide; x++)
                {
                    vertices.Add(new Vector3(-half + x * cellSize, 0, -half + z * cellSize));
                }
            }

            int row = cellsPerSide + 1;
            for (int z = 0; z < cellsPerSide; z++)
            {
                for (int x = 0; x < cellsPerSide; x++)
                {
                    int i0 = z * row + x;
                    int i1 = i0 + 1;
                    int i2 = i0 + row;
                    int i3 = i2 + 1;

                    indices.Add(i0); indices.Add(i2); indices.Add(i1);
                    indices.Add(i1); indices.Add(i2); indices.Add(i3);
                }
            }

            var mesh = new MeshData
            {
                Vertices = vertices,
                Indices = indices,
                Colors = CreateColorList(vertices.Count, new Vector3(0.68f, 0.74f, 0.84f))
            };
            mesh.RecalculateNormals();
            return mesh;
        }

        private static List<Vector3> CreateColorList(int count, Vector3 color)
        {
            var colors = new List<Vector3>(count);
            for (int i = 0; i < count; i++)
                colors.Add(color);
            return colors;
        }
    }
}
