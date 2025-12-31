using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Globalization;
using System.Threading.Tasks;
using Xunit;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;
using MStudio.Services.Implementations;
using System.Numerics;

namespace MStudio.Tests
{
    public class CMJAnalysisDebug
    {
        [Fact]
        public async Task DebugCMJAnalysis()
        {
            // 1. Load TRC File
            string trcPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../../../tests/CMJ.trc");
            if (!File.Exists(trcPath))
            {
                trcPath = @"c:\Users\BB1\Desktop\MStudio\tests\CMJ.trc";
            }
            
            Assert.True(File.Exists(trcPath), $"File not found: {trcPath}");
            
            var motionData = ParseTrc(trcPath);
            var service = new CMJAnalysisService();

            // 2. Run Analysis
            Console.WriteLine("=== CMJ ANALYSIS DEBUG START ===");
            
            // Check Lowest CoM
            int lowestFrame = service.FindLowestCoMFrameUsingSegmentModel(motionData, Gender.Male);
            Console.WriteLine($"Lowest CoM Frame: {lowestFrame}");
            
            // Debug Coordinates at Lowest Frame
            DebugMarkerCoords(motionData, lowestFrame, "RHip", "RKnee", "RAnkle");
            
            // Manual Valgus Logic Check
            float manualValgus = CalculateValgusManual(motionData, lowestFrame, "RHip", "RKnee", "RAnkle");
            Console.WriteLine($"Manual Calculated Valgus: {manualValgus:F2} degrees");

            // Service Valgus Check
            var (leftValgus, rightValgus) = service.CalculateKneeValgus(motionData, lowestFrame, Gender.Male);
            Console.WriteLine($"Service Left Knee Valgus: {leftValgus.AngleDegrees:F2} degrees");
            Console.WriteLine($"Service Right Knee Valgus: {rightValgus.AngleDegrees:F2} degrees");

            // Phase Detection
            var phases = service.DetectPhases(motionData, lowestFrame);
            var takeoffPhase = phases.FirstOrDefault(p => p.Phase == CMJPhase.Propulsion);
            int takeoffFrame = takeoffPhase?.EndFrame ?? 0;
            
            var landingPhase = phases.FirstOrDefault(p => p.Phase == CMJPhase.LandingAbsorption);
            int landingFrame = landingPhase?.StartFrame ?? 0;
            
            Console.WriteLine($"Take-off Frame: {takeoffFrame}");
            Console.WriteLine($"Landing Frame: {landingFrame}");

            // Debug Toe Heights
            DebugToeHeight(motionData, lowestFrame, takeoffFrame);

            Console.WriteLine("=== CMJ ANALYSIS DEBUG END ===");
        }

        private void DebugMarkerCoords(MotionData data, int frame, string h, string k, string a)
        {
            var hip = GetPos(data, h, frame);
            var knee = GetPos(data, k, frame);
            var ankle = GetPos(data, a, frame);
            Console.WriteLine($"Coords Frame {frame}:");
            Console.WriteLine($"  {h}: {hip}");
            Console.WriteLine($"  {k}: {knee}");
            Console.WriteLine($"  {a}: {ankle}");
        }

        private Vector3 GetPos(MotionData data, string name, int frame)
        {
            var names = data.Metadata.MarkerNames;
            int idx = -1;
            for(int i=0; i<names.Count; i++)
            {
                if(names[i] == name)
                {
                    idx = i;
                    break;
                }
            }
            if (idx == -1) return Vector3.Zero;
            return data.Markers.GetPosition(idx, frame);
        }

        private float CalculateValgusManual(MotionData data, int frame, string h, string k, string a)
        {
            var hip = GetPos(data, h, frame);
            var knee = GetPos(data, k, frame);
            var ankle = GetPos(data, a, frame);

            float dzThigh = knee.Z - hip.Z; // Change X to Z
            float dyThigh = knee.Y - hip.Y;
            float thighAngle = MathF.Atan2(dzThigh, -dyThigh);

            float dzShank = ankle.Z - knee.Z; // Change X to Z
            float dyShank = ankle.Y - knee.Y;
            float shankAngle = MathF.Atan2(dzShank, -dyShank);

            float valgusDeg = (thighAngle - shankAngle) * (180f / MathF.PI);
            return Math.Abs(valgusDeg);
        }

        private void DebugToeHeight(MotionData data, int lowest, int takeoff)
        {
            var rToeL = GetPos(data, "RBigToe", lowest);
            var lToeL = GetPos(data, "LBigToe", lowest);
            float baseH = (rToeL.Y + lToeL.Y) / 2f;
            
            var rToeT = GetPos(data, "RBigToe", takeoff);
            var lToeT = GetPos(data, "LBigToe", takeoff);
            float takeH = (rToeT.Y + lToeT.Y) / 2f;

            Console.WriteLine($"Toe Height Debug:");
            Console.WriteLine($"  LowestFrame({lowest}) Height: {baseH:F3} m");
            Console.WriteLine($"  TakeoffFrame({takeoff}) Height: {takeH:F3} m");
            Console.WriteLine($"  Diff: {(takeH - baseH):F3} m");
        }

        private MotionData ParseTrc(string filePath)
        {
            var lines = File.ReadAllLines(filePath);
            
            var metaLine = lines[2].Split('\t');
            float frameRate = float.Parse(metaLine[0]);
            int numFrames = int.Parse(metaLine[2]);

            var nameLine = lines[3].Split('\t');
            var markerNames = new List<string>();
            for (int i = 2; i < nameLine.Length; i++) 
            {
                if (!string.IsNullOrWhiteSpace(nameLine[i]))
                    markerNames.Add(nameLine[i].Trim());
            }

            var container = new MarkerDataContainer(markerNames.Count, numFrames);
            
            for (int i = 0; i < numFrames; i++)
            {
                if (5 + i >= lines.Length) break;
                var line = lines[5 + i].Split('\t');
                
                for (int m = 0; m < markerNames.Count; m++)
                {
                    int colBase = 2 + (m * 3); 
                    if (colBase + 2 < line.Length)
                    {
                        if (float.TryParse(line[colBase], NumberStyles.Any, CultureInfo.InvariantCulture, out float x) &&
                            float.TryParse(line[colBase + 1], NumberStyles.Any, CultureInfo.InvariantCulture, out float y) &&
                            float.TryParse(line[colBase + 2], NumberStyles.Any, CultureInfo.InvariantCulture, out float z))
                        {
                            container.SetPosition(m, i, x, y, z);
                        }
                    }
                }
            }

            return new MotionData
            {
                Metadata = new MotionMetadata 
                { 
                    FilePath = filePath,
                    FrameRate = frameRate,
                    TotalFrames = numFrames,
                    MarkerNames = markerNames 
                },
                Markers = container
            };
        }
    }
}
