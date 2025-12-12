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

            using (var fs = File.OpenRead(filepath))
            {
                // 1. Read Header
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
                        if (line == "end_header") break;
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
                        if (parts[1].Contains("binary")) isBinary = true;
                        if (parts[1] == "binary_big_endian") { isBinary = true; isBigEndian = true; }
                    }
                    else if (parts[0] == "element")
                    {
                        currentElement = new PlyElement { Name = parts[1], Count = int.Parse(parts[2]) };
                        elements.Add(currentElement);
                    }
                    else if (parts[0] == "property")
                    {
                        if (currentElement != null)
                        {
                            var prop = new PlyProperty();
                            if (parts[1] == "list")
                            {
                                prop.Name = parts[4];
                                prop.Type = PlyPropertyType.List;
                                prop.ListCountType = ParseType(parts[2]);
                                prop.ListItemType = ParseType(parts[3]);
                            }
                            else
                            {
                                prop.Name = parts[2];
                                prop.Type = ParseType(parts[1]);
                            }
                            currentElement.Properties.Add(prop);
                        }
                    }
                }

                // 2. Read Body
                // We assume sequential elements as defined in header

                using (var reader = isBinary ? (BinaryReader)new BinaryReader(fs) : null)
                using (var textReader = !isBinary ? new StreamReader(fs, Encoding.ASCII, false, 1024, true) : null)
                {
                    // Pre-allocate if possible
                    foreach(var el in elements) {
                        if (el.Name == "vertex") {
                            plyData.Vertices.Capacity = el.Count;
                            plyData.Colors.Capacity = el.Count;
                        }
                        else if (el.Name == "face") {
                            plyData.Indices.Capacity = el.Count * 3;
                        }
                    }

                    foreach(var element in elements)
                    {
                        if (element.Name == "vertex")
                        {
                            for(int i=0; i<element.Count; i++)
                            {
                                float x=0, y=0, z=0;
                                float r=0.8f, g=0.8f, b=0.8f;

                                if (isBinary)
                                {
                                    ReadBinaryVertex(reader, element.Properties, isBigEndian, ref x, ref y, ref z, ref r, ref g, ref b);
                                }
                                else
                                {
                                    ReadAsciiVertex(textReader, element.Properties, ref x, ref y, ref z, ref r, ref g, ref b);
                                }

                                plyData.Vertices.Add(new Vector3(x, y, z));
                                plyData.Colors.Add(new Vector3(r, g, b));
                            }
                        }
                        else if (element.Name == "face")
                        {
                            for(int i=0; i<element.Count; i++)
                            {
                                if (isBinary)
                                {
                                    ReadBinaryFace(reader, element.Properties, isBigEndian, plyData.Indices);
                                }
                                else
                                {
                                    ReadAsciiFace(textReader, element.Properties, plyData.Indices);
                                }
                            }
                        }
                        else
                        {
                            // Skip other elements
                            for(int i=0; i<element.Count; i++)
                            {
                                if (isBinary) SkipBinaryElement(reader, element.Properties, isBigEndian);
                                else if (textReader != null) textReader.ReadLine();
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

        private static void ReadAsciiVertex(StreamReader reader, List<PlyProperty> props, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b)
        {
            var line = reader.ReadLine();
            if (line == null) return;
            var parts = line.Split(new[]{' '}, StringSplitOptions.RemoveEmptyEntries);

            for(int p=0; p<props.Count && p<parts.Length; p++)
            {
                var prop = props[p];
                double val = double.Parse(parts[p], CultureInfo.InvariantCulture); // Read as double first

                ApplyProperty(prop.Name, val, ref x, ref y, ref z, ref r, ref g, ref b);
            }
        }

        private static void ReadBinaryVertex(BinaryReader reader, List<PlyProperty> props, bool be, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b)
        {
             foreach(var prop in props)
             {
                 double val = ReadValue(reader, prop.Type, be);
                 ApplyProperty(prop.Name, val, ref x, ref y, ref z, ref r, ref g, ref b);
             }
        }

        private static void ApplyProperty(string name, double val, ref float x, ref float y, ref float z, ref float r, ref float g, ref float b)
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
        }

        private static void ReadAsciiFace(StreamReader reader, List<PlyProperty> props, List<int> indices)
        {
             var line = reader.ReadLine();
             if (line == null) return;
             var parts = line.Split(new[]{' '}, StringSplitOptions.RemoveEmptyEntries);

             int partIdx = 0;
             foreach(var prop in props)
             {
                 if (prop.Type == PlyPropertyType.List)
                 {
                     int count = int.Parse(parts[partIdx++]);
                     if (prop.Name == "vertex_indices" || prop.Name == "vertex_index")
                     {
                         var faceIndices = new List<int>();
                         for(int c=0; c<count; c++)
                             faceIndices.Add(int.Parse(parts[partIdx++]));

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

        private static short ReadShortBE(BinaryReader br) { var b = br.ReadBytes(2); Array.Reverse(b); return BitConverter.ToInt16(b, 0); }
        private static ushort ReadUShortBE(BinaryReader br) { var b = br.ReadBytes(2); Array.Reverse(b); return BitConverter.ToUInt16(b, 0); }
        private static int ReadIntBE(BinaryReader br) { var b = br.ReadBytes(4); Array.Reverse(b); return BitConverter.ToInt32(b, 0); }
        private static uint ReadUIntBE(BinaryReader br) { var b = br.ReadBytes(4); Array.Reverse(b); return BitConverter.ToUInt32(b, 0); }
        private static float ReadFloatBE(BinaryReader br) { var b = br.ReadBytes(4); Array.Reverse(b); return BitConverter.ToSingle(b, 0); }
        private static double ReadDoubleBE(BinaryReader br) { var b = br.ReadBytes(8); Array.Reverse(b); return BitConverter.ToDouble(b, 0); }
    }
}
