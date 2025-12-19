using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Core.Parsers
{
    public class JsonPoseParser : IFileParser
    {
        public bool CanParse(string filePath)
        {
            return Path.GetExtension(filePath).Equals(".json", StringComparison.OrdinalIgnoreCase);
        }

        public async Task<MotionData> ParseAsync(string filePath)
        {
            string folderPath = Path.GetDirectoryName(filePath) ?? throw new InvalidOperationException("Invalid file path.");
            
            // Look for all JSON files in the same folder, matching frame pattern [digits].json
            var jsonFiles = Directory.GetFiles(folderPath, "*.json")
                .Select(f => new { Path = f, Frame = GetFrameNumber(f) })
                .OrderBy(f => f.Frame)
                .ToList();

            if (!jsonFiles.Any())
                throw new FileNotFoundException("No JSON files found in the sequence.");

            // Read first file to determine marker count
            using var firstStream = File.OpenRead(jsonFiles[0].Path);
            using var firstDoc = await JsonDocument.ParseAsync(firstStream);
            
            int markerCount = GetMarkerCount(firstDoc);
            int totalFrames = jsonFiles.Count;
            float frameRate = 30.0f; // Default OpenPose frame rate

            var container = new MarkerDataContainer(markerCount, totalFrames);
            var markerNames = Enumerable.Range(0, markerCount).Select(i => $"Keypoint_{i}").ToList();

            for (int f = 0; f < totalFrames; f++)
            {
                using var stream = File.OpenRead(jsonFiles[f].Path);
                using var doc = await JsonDocument.ParseAsync(stream);
                
                FillFrameData(container, f, doc, markerCount);
            }

            return new MotionData
            {
                Metadata = new MotionMetadata
                {
                    FilePath = folderPath,
                    FrameRate = frameRate,
                    TotalFrames = totalFrames,
                    MarkerNames = markerNames,
                    Units = "m"
                },
                Markers = container
            };
        }

        private int GetFrameNumber(string path)
        {
            var match = Regex.Match(Path.GetFileName(path), @"(\d+)");
            return match.Success ? int.Parse(match.Groups[1].Value) : 0;
        }

        private int GetMarkerCount(JsonDocument doc)
        {
            if (doc.RootElement.TryGetProperty("people", out var people) && people.GetArrayLength() > 0)
            {
                var firstPerson = people[0];
                if (firstPerson.TryGetProperty("pose_keypoints_2d", out var points))
                {
                    return points.GetArrayLength() / 3;
                }
            }
            return 0;
        }

        private void FillFrameData(MarkerDataContainer container, int frameIdx, JsonDocument doc, int markerCount)
        {
            if (doc.RootElement.TryGetProperty("people", out var people) && people.GetArrayLength() > 0)
            {
                var firstPerson = people[0];
                if (firstPerson.TryGetProperty("pose_keypoints_2d", out var points))
                {
                    int i = 0;
                    foreach (var p in points.EnumerateArray())
                    {
                        int markerIdx = i / 3;
                        if (markerIdx >= markerCount) break;

                        float x = (float)points[markerIdx * 3].GetDouble() / 1000.0f;
                        float y = (float)points[markerIdx * 3 + 1].GetDouble() / 1000.0f;
                        float confidence = (float)points[markerIdx * 3 + 2].GetDouble();

                        if (confidence > 0)
                        {
                            // OpenPose coordinate system: Y is down, usually. 
                            // Python code flipped Y: -y / 1000.0
                            container.SetPosition(markerIdx, frameIdx, x, -y, 0);
                        }
                        else
                        {
                            container.SetPosition(markerIdx, frameIdx, float.NaN, float.NaN, float.NaN);
                        }
                        
                        i += 3;
                        if (i >= points.GetArrayLength()) break;
                    }
                }
            }
        }
    }
}
