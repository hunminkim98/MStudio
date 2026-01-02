using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using MStudio.Core.Models;

namespace MStudio.Core.Writers
{
    /// <summary>
    /// Writes motion data to TRC (Track Row Column) file format.
    /// Ported from Python MStudio dataSaver.py
    /// </summary>
    public static class TrcFileWriter
    {
        /// <summary>
        /// Writes motion data to a TRC file asynchronously.
        /// </summary>
        /// <param name="filePath">Path to save the TRC file</param>
        /// <param name="motionData">Motion data to save</param>
        public static async Task WriteAsync(string filePath, MotionData motionData)
        {
            var metadata = motionData.Metadata;
            var markers = motionData.Markers;
            var markerNames = metadata.MarkerNames;
            
            var sb = new StringBuilder();
            
            // Line 1: PathFileType header
            sb.AppendLine($"PathFileType\t4\t(X/Y/Z)\t{Path.GetFileName(filePath)}");
            
            // Line 2: Column headers for metadata
            sb.AppendLine("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames");
            
            // Line 3: Metadata values
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0}\t{0}\t{1}\t{2}\t{3}\t{0}\t1\t{1}",
                metadata.FrameRate,
                metadata.TotalFrames,
                markerNames.Count,
                metadata.Units));
            
            // Line 4: Frame#, Time, and marker names (each marker name repeated with tabs for X, Y, Z)
            var markerNamesLine = new StringBuilder("Frame#\tTime");
            foreach (var name in markerNames)
            {
                markerNamesLine.Append($"\t{name}\t\t");
            }
            sb.AppendLine(markerNamesLine.ToString().TrimEnd());
            
            // Line 5: Empty columns for Frame# and Time, then X1, Y1, Z1, X2, Y2, Z2... labels for each marker
            // OpenSim TRCFileAdapter requires numbered format (X1, Y1, Z1, X2, Y2, Z2...)
            var xyzLine = new StringBuilder("\t");
            for (int i = 0; i < markerNames.Count; i++)
            {
                int num = i + 1;
                xyzLine.Append($"\tX{num}\tY{num}\tZ{num}");
            }
            sb.AppendLine(xyzLine.ToString());
            
            // Data lines (starting from line 6)
            for (int frame = 0; frame < metadata.TotalFrames; frame++)
            {
                float time = frame / metadata.FrameRate;
                var dataLine = new StringBuilder();
                dataLine.Append(string.Format(CultureInfo.InvariantCulture, "{0}\t{1:F6}", frame + 1, time));
                
                for (int m = 0; m < markerNames.Count; m++)
                {
                    var pos = markers.GetPosition(m, frame);
                    
                    if (float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        dataLine.Append("\t\t\t");
                    }
                    else
                    {
                        dataLine.Append(string.Format(CultureInfo.InvariantCulture, 
                            "\t{0:F7}\t{1:F7}\t{2:F7}", pos.X, pos.Y, pos.Z));
                    }
                }
                
                sb.AppendLine(dataLine.ToString());
            }
            
            await File.WriteAllTextAsync(filePath, sb.ToString());
        }
    }
}
