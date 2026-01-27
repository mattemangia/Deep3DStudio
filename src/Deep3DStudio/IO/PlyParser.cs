using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using OpenTK.Mathematics;

namespace Deep3DStudio.IO
{
    public class PlyData
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Vector3> Colors { get; set; } = new List<Vector3>();
        public List<Vector2> UVs { get; set; } = new List<Vector2>();
        public List<int> Indices { get; set; } = new List<int>();
    }

    public static class PlyParser
    {
        private enum PlyPropertyType { Char, UChar, Short, UShort, Int, UInt, Float, Double, List }

        private class PlyProperty
        {
            public string Name;
            public PlyPropertyType Type;
            public PlyPropertyType ListCountType; // Only for list
            public PlyPropertyType ListItemType;  // Only for list
        }

        private class PlyElement
        {
            public string Name;
            public int Count;
            public List<PlyProperty> Properties = new List<PlyProperty>();
        }

        public static PlyData Parse(string filepath)
        {
            var plyData = new PlyData();

            // 1. Read Header
            using (var fs = File.OpenRead(filepath))
            {
                var headerLines = new List<string>();
                int readByte = 0;
                var sb = new StringBuilder();

                while ((readByte = fs.ReadByte()) != -1)
                {
                    if (readByte == '\n')
                    {
                        var line = sb.ToString().Trim();
                        headerLines.Add(line);
                        sb.Clear();
                        if (line == "end_header") 
                        {
                            break;
                        }
                    }
                    else
                    {
                        sb.Append((char)readByte);
                    }
                }

                bool isBinary = false;
                bool isBigEndian = false;
                var elements = new List<PlyElement>();
                PlyElement currentElement = null;

                foreach(var line in headerLines)
                {
                    var parts = line.Split(new[]{' '}, StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0) continue;

                    if (parts[0] == "format")
                    {
                        if (parts.Length < 2) continue;
                        if (parts[1].Contains("binary")) isBinary = true;
                        if (parts[1] == "binary_big_endian") { isBinary = true; isBigEndian = true; }
                    }
                    else if (parts[0] == "element")
                    {
                        if (parts.Length < 3) continue;
                        if (!TryParseInt(parts[2], out int count)) continue;
                        currentElement = new PlyElement { Name = parts[1], Count = count };
                        elements.Add(currentElement);
                    }
                    else if (parts[0] == "property")
                    {
                        if (currentElement != null)
                        {
                            var prop = new PlyProperty();
                            if (parts[1] == "list")
                            {
                                if (parts.Length < 5) continue;
                                prop.Name = parts[4];
                                prop.Type = PlyPropertyType.List;
                                prop.ListCountType = ParseType(parts[2]);
                                prop.ListItemType = ParseType(parts[3]);
                            }
                            else
                            {
                                if (parts.Length < 3) continue;
                                prop.Name = parts[2];
                                prop.Type = ParseType(parts[1]);
                            }
                            currentElement.Properties.Add(prop);
                        }
                    }
                }

                // Pre-allocate
                foreach(var el in elements) {
                    if (el.Name == "vertex") {
                        plyData.Vertices.Capacity = el.Count;
                        plyData.Colors.Capacity = el.Count;
                        plyData.UVs.Capacity = el.Count;
                    }
                    else if (el.Name == "face") {
                        plyData.Indices.Capacity = el.Count * 3;
                    }
                }

                // 2. Read Body
                if (isBinary)
                {
                    // For binary, continue using the stream.
                    // BinaryReader takes ownership of stream, so we use leaveOpen=true to let the outer using dispose fs.
                    using (var reader = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true))
                    {
                        foreach(var element in elements)
                        {
                            if (element.Name == "vertex")
                            {
                                for(int i=0; i<element.Count; i++)
                                {
                                    float x=0, y=0, z=0;
                                    float r=0.8f, g=0.8f, b=0.8f;
                                    float u=0, v=0;

                                    ReadBinaryVertex(reader, element.Properties, isBigEndian, ref x, ref y, ref z, ref r, ref g, ref b, ref u, ref v);
                                    
                                    plyData.Vertices.Add(new Vector3(x, y, z));
                                    plyData.Colors.Add(new Vector3(r, g, b));
                                    plyData.UVs.Add(new Vector2(u, v));
                                }
                            }
                            else if (element.Name == "face")
                            {
                                for(int i=0; i<element.Count; i++)
                                {
                                    ReadBinaryFace(reader, element.Properties, isBigEndian, plyData.Indices);
                                }
                            }
                            else
                            {
                                // Skip other elements
                                for(int i=0; i<element.Count; i++)
                                {
                                    SkipBinaryElement(reader, element.Properties, isBigEndian);
                                }
                            }
                        }
                    }
                }
                else
                {
                    // For ASCII, verify stream position is correct (StreamReader might have buffering issues if we mix ReadByte and StreamReader)
                    // But since we read until end_header\n, the stream position should be correct for the start of body.
                    using (var textReader = new StreamReader(fs, Encoding.ASCII, detectEncodingFromByteOrderMarks: false, bufferSize: 4096, leaveOpen: true))
                    {
                        foreach(var element in elements)
                        {
                            if (element.Name == "vertex")
                            {
                                for(int i=0; i<element.Count; i++)
                                {
                                    float x=0, y=0, z=0;
                                    float r=0.8f, g=0.8f, b=0.8f;
                                    float u=0, v=0;

                                    if (!ReadAsciiVertex(textReader, element.Properties, ref x, ref y, ref z, ref r, ref g, ref b, ref u, ref v))
                                        break;

                                    plyData.Vertices.Add(new Vector3(x, y, z));
                                    plyData.Colors.Add(new Vector3(r, g, b));
                                    plyData.UVs.Add(new Vector2(u, v));
                                }
                            }
                            else if (element.Name == "face")
                            {
                                for(int i=0; i<element.Count; i++)
                                {
                                    if (!ReadAsciiFace(textReader, element.Properties, plyData.Indices))
                                        break;
                                }
                            }
                            else
                            {
                                // Skip other elements
                                for(int i=0; i<element.Count; i++)
                                {
                                    textReader.ReadLine();
                                }
                            }
                        }
                    }
                }
            }

            return plyData;
        }

        private static PlyPropertyType ParseType(string typeStr)
        {
            switch (typeStr)
            {
                case "char": case "int8": return PlyPropertyType.Char;
                case "uchar": case "uint8": return PlyPropertyType.UChar;
                case "short": case "int16": return PlyPropertyType.Short;
                case "ushort": case "uint16": return PlyPropertyType.UShort;
                case "int": case "int32": return PlyPropertyType.Int;
                case "uint": case "uint32": return PlyPropertyType.UInt;
                case "float": case "float32": return PlyPropertyType.Float;
                case "double": case "float64": return PlyPropertyType.Double;
                default: return PlyPropertyType.Float; // Fallback
            }
        }

        private static bool ReadAsciiVertex(StreamReader reader, List<PlyProperty> props, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b, ref float u, ref float v)
        {
            var line = reader.ReadLine();
            if (line == null) return false;
            var parts = line.Split(new[] { ' ', '\t', ',' }, StringSplitOptions.RemoveEmptyEntries);

            int partIdx = 0;
            for (int p = 0; p < props.Count && partIdx < parts.Length; p++)
            {
                var prop = props[p];
                if (prop.Type == PlyPropertyType.List)
                {
                    if (!TryParseInt(parts[partIdx++], out int count)) return true;
                    partIdx = Math.Min(parts.Length, partIdx + count);
                    continue;
                }

                if (TryParseDouble(parts[partIdx++], out double val))
                {
                    ApplyProperty(prop.Name, val, ref x, ref y, ref z, ref r, ref g, ref b, ref u, ref v);
                }
            }

            return true;
        }

        private static void ReadBinaryVertex(BinaryReader reader, List<PlyProperty> props, bool be, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b, ref float u, ref float v)
        {
            foreach (var prop in props)
            {
                if (prop.Type == PlyPropertyType.List)
                {
                    int count = (int)ReadValue(reader, prop.ListCountType, be);
                    for (int i = 0; i < count; i++) ReadValue(reader, prop.ListItemType, be);
                    continue;
                }

                double val = ReadValue(reader, prop.Type, be);
                ApplyProperty(prop.Name, val, ref x, ref y, ref z, ref r, ref g, ref b, ref u, ref v);
            }
        }

        private static void ApplyProperty(string name, double val, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b, ref float u, ref float v)
        {
            if (name == "x") x = (float)val;
            else if (name == "y") y = (float)val;
            else if (name == "z") z = (float)val;
            else if (name == "red" || name == "r" || name == "diffuse_red")
            {
                r = (float)val;
                if (r > 1.0f) r /= 255f;
            }
            else if (name == "green" || name == "g" || name == "diffuse_green")
            {
                g = (float)val;
                if (g > 1.0f) g /= 255f;
            }
            else if (name == "blue" || name == "b" || name == "diffuse_blue")
            {
                b = (float)val;
                if (b > 1.0f) b /= 255f;
            }
            else if (name == "s" || name == "u" || name == "texture_u") u = (float)val;
            else if (name == "t" || name == "v" || name == "texture_v") v = (float)val;
        }

        private static bool ReadAsciiFace(StreamReader reader, List<PlyProperty> props, List<int> indices)
        {
             var line = reader.ReadLine();
             if (line == null) return false;
             var parts = line.Split(new[]{' ', '\t', ','}, StringSplitOptions.RemoveEmptyEntries);

             int partIdx = 0;
             foreach(var prop in props)
             {
                 if (prop.Type == PlyPropertyType.List)
                 {
                     if (partIdx >= parts.Length || !TryParseInt(parts[partIdx++], out int count)) return true;
                     if (prop.Name == "vertex_indices" || prop.Name == "vertex_index")
                     {
                         var faceIndices = new List<int>();
                         for(int c=0; c<count && partIdx < parts.Length; c++)
                         {
                             if (TryParseInt(parts[partIdx++], out int idx))
                                 faceIndices.Add(idx);
                         }

                         // Ignore faces with too many vertices (likely a point cloud saved as a single polygon)
                         if (faceIndices.Count > 32) return true;

                         // Triangulate
                         for(int i=1; i<faceIndices.Count-1; i++)
                         {
                             indices.Add(faceIndices[0]);
                             indices.Add(faceIndices[i]);
                             indices.Add(faceIndices[i+1]);
                         }
                     }
                     else
                     {
                         partIdx += count; // Skip
                     }
                 }
                 else
                 {
                     partIdx++;
                 }
             }
             return true;
        }

        private static void ReadBinaryFace(BinaryReader reader, List<PlyProperty> props, bool be, List<int> indices)
        {
            foreach(var prop in props)
            {
                if (prop.Type == PlyPropertyType.List)
                {
                    int count = (int)ReadValue(reader, prop.ListCountType, be);
                    if (prop.Name == "vertex_indices" || prop.Name == "vertex_index")
                    {
                        var faceIndices = new List<int>();
                        for(int c=0; c<count; c++)
                            faceIndices.Add((int)ReadValue(reader, prop.ListItemType, be));

                        // Ignore faces with too many vertices (likely a point cloud saved as a single polygon)
                        if (faceIndices.Count > 32) continue;

                        // Triangulate
                        for(int i=1; i<faceIndices.Count-1; i++)
                        {
                            indices.Add(faceIndices[0]);
                            indices.Add(faceIndices[i]);
                            indices.Add(faceIndices[i+1]);
                        }
                    }
                    else
                    {
                        for(int c=0; c<count; c++) ReadValue(reader, prop.ListItemType, be); // Skip
                    }
                }
                else
                {
                    ReadValue(reader, prop.Type, be);
                }
            }
        }

        private static void SkipBinaryElement(BinaryReader reader, List<PlyProperty> props, bool be)
        {
            foreach(var prop in props)
            {
                if (prop.Type == PlyPropertyType.List)
                {
                    int count = (int)ReadValue(reader, prop.ListCountType, be);
                    for(int c=0; c<count; c++) ReadValue(reader, prop.ListItemType, be);
                }
                else
                {
                    ReadValue(reader, prop.Type, be);
                }
            }
        }

        private static double ReadValue(BinaryReader br, PlyPropertyType type, bool be)
        {
            switch(type)
            {
                case PlyPropertyType.Char: return br.ReadSByte();
                case PlyPropertyType.UChar: return br.ReadByte();
                case PlyPropertyType.Short: return be ? ReadShortBE(br) : br.ReadInt16();
                case PlyPropertyType.UShort: return be ? ReadUShortBE(br) : br.ReadUInt16();
                case PlyPropertyType.Int: return be ? ReadIntBE(br) : br.ReadInt32();
                case PlyPropertyType.UInt: return be ? ReadUIntBE(br) : br.ReadUInt32();
                case PlyPropertyType.Float: return be ? ReadFloatBE(br) : br.ReadSingle();
                case PlyPropertyType.Double: return be ? ReadDoubleBE(br) : br.ReadDouble();
            }
            return 0;
        }

        private static bool TryParseDouble(string value, out double result)
        {
            if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result))
                return true;
            return double.TryParse(value, NumberStyles.Float, CultureInfo.CurrentCulture, out result);
        }

        private static bool TryParseInt(string value, out int result)
        {
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result))
                return true;
            return int.TryParse(value, NumberStyles.Integer, CultureInfo.CurrentCulture, out result);
        }

        private static short ReadShortBE(BinaryReader br) { var b = br.ReadBytes(2); if (b.Length < 2) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToInt16(b, 0); }
        private static ushort ReadUShortBE(BinaryReader br) { var b = br.ReadBytes(2); if (b.Length < 2) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToUInt16(b, 0); }
        private static int ReadIntBE(BinaryReader br) { var b = br.ReadBytes(4); if (b.Length < 4) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToInt32(b, 0); }
        private static uint ReadUIntBE(BinaryReader br) { var b = br.ReadBytes(4); if (b.Length < 4) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToUInt32(b, 0); }
        private static float ReadFloatBE(BinaryReader br) { var b = br.ReadBytes(4); if (b.Length < 4) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToSingle(b, 0); }
        private static double ReadDoubleBE(BinaryReader br) { var b = br.ReadBytes(8); if (b.Length < 8) throw new EndOfStreamException(); Array.Reverse(b); return BitConverter.ToDouble(b, 0); }
    }
}