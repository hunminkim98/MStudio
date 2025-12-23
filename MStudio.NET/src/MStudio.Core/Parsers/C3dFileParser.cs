using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Core.Parsers
{
    public class C3dFileParser : IFileParser
    {
        public bool CanParse(string filePath)
        {
            return Path.GetExtension(filePath).Equals(".c3d", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<MotionData> ParseAsync(string filePath)
        {
            return await Task.Run(() =>
            {
                using var fs = File.OpenRead(filePath);
                using var reader = new BinaryReader(fs);

                // --- 1. Read Header (512 bytes) ---
                byte paramBlock = reader.ReadByte();
                byte magic = reader.ReadByte();
                if (magic != 0x50) throw new InvalidDataException("Not a valid C3D file.");

                ushort numPoints = reader.ReadUInt16();
                ushort numAnalog = reader.ReadUInt16();
                ushort firstFrame = reader.ReadUInt16();
                ushort lastFrame = reader.ReadUInt16();
                reader.BaseStream.Seek(12, SeekOrigin.Begin);
                float scaleFactor = reader.ReadSingle();
                ushort dataBlock = reader.ReadUInt16();
                reader.BaseStream.Seek(20, SeekOrigin.Begin);
                float frameRate = reader.ReadSingle();

                int totalFrames = lastFrame - firstFrame + 1;

                // --- 2. Read Marker Names from Parameter Block ---
                var markerNames = ReadMarkerLabels(fs, reader, paramBlock, numPoints);

                var container = new MarkerDataContainer(numPoints, totalFrames);

                // --- 3. Read Data ---
                // Jump to data block (1-based block index, each block is 512 bytes)
                fs.Seek((dataBlock - 1) * 512, SeekOrigin.Begin);

                bool isFloat = scaleFactor < 0;
                float absoluteScale = Math.Abs(scaleFactor);

                for (int f = 0; f < totalFrames; f++)
                {
                    // For each frame:
                    // Points data: numPoints * 4 items (X, Y, Z, Residual/CameraMask)
                    for (int m = 0; m < numPoints; m++)
                    {
                        float x, y, z;
                        if (isFloat)
                        {
                            x = reader.ReadSingle();
                            y = reader.ReadSingle();
                            z = reader.ReadSingle();
                            reader.ReadSingle(); // Residual
                        }
                        else
                        {
                            x = reader.ReadInt16() * absoluteScale;
                            y = reader.ReadInt16() * absoluteScale;
                            z = reader.ReadInt16() * absoluteScale;
                            reader.ReadInt16(); // Residual
                        }

                        // Convert mm to meters as per Python app
                        container.SetPosition(m, f, x / 1000.0f, y / 1000.0f, z / 1000.0f);
                    }

                    // Skip analog data if present
                    if (numAnalog > 0)
                    {
                        // Simplified - skip analog data
                        // Full implementation would read ANALOG:RATE parameter
                    }
                }

                return new MotionData
                {
                    Metadata = new MotionMetadata
                    {
                        FilePath = filePath,
                        FrameRate = frameRate,
                        TotalFrames = totalFrames,
                        MarkerNames = markerNames,
                        Units = "m"
                    },
                    Markers = container
                };
            });
        }

        private List<string> ReadMarkerLabels(FileStream fs, BinaryReader reader, byte paramBlockStart, int numPoints)
        {
            var labels = new List<string>();
            
            try
            {
                // Navigate to parameter block
                fs.Seek((paramBlockStart - 1) * 512, SeekOrigin.Begin);
                
                // Skip first 4 bytes of parameter section header
                reader.ReadBytes(4);
                
                // Read parameter groups and parameters
                while (fs.Position < fs.Length)
                {
                    sbyte nameLen = reader.ReadSByte();
                    if (nameLen == 0) break; // End of parameters
                    
                    sbyte groupId = reader.ReadSByte();
                    bool isGroup = groupId < 0;
                    int absNameLen = Math.Abs(nameLen);
                    
                    if (absNameLen == 0 || absNameLen > 127) break;
                    
                    string name = Encoding.ASCII.GetString(reader.ReadBytes(absNameLen)).Trim();
                    short offset = reader.ReadInt16(); // Offset to next item
                    
                    if (isGroup)
                    {
                        // Skip group description
                        if (offset > 0)
                        {
                            byte descLen = reader.ReadByte();
                            reader.ReadBytes(descLen);
                        }
                    }
                    else
                    {
                        // Read parameter data
                        sbyte dataType = reader.ReadSByte();
                        byte numDims = reader.ReadByte();
                        
                        int[] dims = new int[numDims];
                        int totalElements = 1;
                        for (int i = 0; i < numDims; i++)
                        {
                            dims[i] = reader.ReadByte();
                            totalElements *= dims[i];
                        }
                        
                        // Check if this is POINT:LABELS (or just LABELS in group POINT)
                        if (name.Equals("LABELS", StringComparison.OrdinalIgnoreCase) && dataType == -1)
                        {
                            // Character array
                            int stringLength = dims.Length > 0 ? dims[0] : 0;
                            int numStrings = dims.Length > 1 ? dims[1] : 1;
                            
                            for (int i = 0; i < numStrings; i++)
                            {
                                string label = Encoding.ASCII.GetString(reader.ReadBytes(stringLength)).Trim();
                                if (!string.IsNullOrWhiteSpace(label))
                                {
                                    labels.Add(label);
                                }
                            }
                            
                            // Found labels, return them
                            if (labels.Count > 0)
                            {
                                // Pad with generic names if needed
                                while (labels.Count < numPoints)
                                {
                                    labels.Add($"Marker_{labels.Count + 1}");
                                }
                                return labels;
                            }
                        }
                        else
                        {
                            // Skip this parameter's data
                            int dataSize = dataType switch
                            {
                                -1 => 1, // char
                                1 => 1,  // byte
                                2 => 2,  // int16
                                4 => 4,  // float
                                _ => 1
                            };
                            reader.ReadBytes(Math.Abs(dataSize) * totalElements);
                        }
                        
                        // Skip description
                        byte descLen = reader.ReadByte();
                        if (descLen > 0)
                        {
                            reader.ReadBytes(descLen);
                        }
                    }
                }
            }
            catch
            {
                // If parsing fails, fall back to generic names
            }
            
            // Fallback: generate generic marker names
            if (labels.Count == 0)
            {
                labels = Enumerable.Range(1, numPoints).Select(i => $"Marker_{i}").ToList();
            }
            
            return labels;
        }
    }
}

