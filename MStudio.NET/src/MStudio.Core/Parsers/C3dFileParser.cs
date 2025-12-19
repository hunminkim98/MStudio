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

                // --- 2. Simplified Parameter Extraction (Skip to marker labels if possible) ---
                // For now, we'll use generic names if we can't easily find labels in binary
                // In a full implementation, we'd traverse the parameter blocks.
                var markerNames = Enumerable.Range(1, numPoints).Select(i => $"Marker_{i}").ToList();

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
                    int analogSize = numAnalog * 4; // Simplified, usually analog samples per frame
                    // C3D spec says analog data follows points in each frame.
                    // This is simplified and might need adjustment for specific C3D variants.
                    if (numAnalog > 0)
                    {
                        // Each frame has numAnalog items. If analog rate is higher than frame rate, 
                        // there are multiple samples per frame. 
                        // For MStudio, we focus on markers.
                        // We need to skip: (total analog channels) * (analog samples per point frame)
                        // This usually needs the ANALOG:RATE parameter.
                        // For a basic parser, we might need to look closer at the file.
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
    }
}
