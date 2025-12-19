using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Core.Parsers
{
    public interface IFileParser
    {
        bool CanParse(string filePath);
        Task<MotionData> ParseAsync(string filePath);
    }

    public class TrcFileParser : IFileParser
    {
        public bool CanParse(string filePath)
        {
            return Path.GetExtension(filePath).Equals(".trc", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<MotionData> ParseAsync(string filePath)
        {
            var lines = await File.ReadAllLinesAsync(filePath);
            if (lines.Length < 7) throw new InvalidDataException("TRC file is too short.");

            // Line 3 (Index 2): DataRate, CameraRate, NumFrames, NumMarkers, Units...
            var metaParts = lines[2].Split('\t', StringSplitOptions.RemoveEmptyEntries);
            float frameRate = float.Parse(metaParts[0], CultureInfo.InvariantCulture);
            int totalFrames = int.Parse(metaParts[2]);
            int markerCount = int.Parse(metaParts[3]);
            string units = metaParts.Length > 4 ? metaParts[4] : "m";

            // Line 4 (Index 3): Frame#, Time, Marker1, Marker2...
            var nameParts = lines[3].Split('\t', StringSplitOptions.RemoveEmptyEntries);
            var markerNames = nameParts.Skip(2).Take(markerCount).ToList();

            var container = new MarkerDataContainer(markerCount, totalFrames);

            // Data starts from Line 7 (Index 6)
            for (int f = 0; f < totalFrames; f++)
            {
                int lineIdx = f + 6;
                if (lineIdx >= lines.Length) break;

                var dataParts = lines[lineIdx].Split('\t');
                // dataParts[0] = Frame#, dataParts[1] = Time
                // coordinates start from index 2
                for (int m = 0; m < markerCount; m++)
                {
                    int baseIdx = 2 + (m * 3);
                    if (baseIdx + 2 >= dataParts.Length) continue;

                    float x = ParseFloat(dataParts[baseIdx]);
                    float y = ParseFloat(dataParts[baseIdx + 1]);
                    float z = ParseFloat(dataParts[baseIdx + 2]);
                    container.SetPosition(m, f, x, y, z);
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
                    Units = units
                },
                Markers = container
            };
        }

        private static float ParseFloat(string s)
        {
            return string.IsNullOrWhiteSpace(s) ? 0.0f : float.Parse(s, CultureInfo.InvariantCulture);
        }
    }
}
