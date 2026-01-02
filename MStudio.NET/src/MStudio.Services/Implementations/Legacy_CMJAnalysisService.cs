using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using MStudio.Core.Interfaces;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// Implementation of CMJ analysis service.
    /// </summary>
    /// <summary>
    /// [LEGACY] Original CMJ analysis service using De Leva (1996) quasi-static moment analysis.
    /// Superseded by OpenSim-based analysis. Kept for reference and fallback.
    /// </summary>
    public class Legacy_CMJAnalysisService : ICMJAnalysisService
    {
        // Phase detection service (논문 기반 9-point 알고리즘)
        private readonly ICMJPhaseDetectionService _phaseDetectionService;

        // Required marker names for CMJ analysis
        private static readonly string[] RequiredMarkers = new[]
        {
            "Hip", "RHip", "LHip", 
            "RKnee", "LKnee", 
            "RAnkle", "LAnkle"
        };

        public Legacy_CMJAnalysisService()
        {
            _phaseDetectionService = new CMJPhaseDetectionService();
        }

        public Legacy_CMJAnalysisService(ICMJPhaseDetectionService phaseDetectionService)
        {
            _phaseDetectionService = phaseDetectionService;
        }

        // Valgus normal ranges by gender (degrees)
        private const float MaleValgusMin = 3f;
        private const float MaleValgusMax = 8f;
        private const float FemaleValgusMin = 7f;
        private const float FemaleValgusMax = 13f;

        public async Task<CMJAnalysisResult> AnalyzeAsync(MotionData data, Gender gender, float bodyMassKg)
        {
            return await Task.Run(() =>
            {
                // Find key frames using whole-body CoM calculation
                int lowestCoMFrame = FindLowestCoMFrameUsingSegmentModel(data, gender);
                
                // Calculate Hip/Knee ratio at lowest point
                var (ratio, dominance) = CalculateHipKneeRatio(data, lowestCoMFrame, gender, bodyMassKg);
                
                // Calculate Knee Valgus at lowest point
                var (leftValgus, rightValgus) = CalculateKneeValgus(data, lowestCoMFrame, gender);
                
                // Detect phases
                var phases = DetectPhases(data, lowestCoMFrame);
                
                // Generate time series data
                var timeSeries = GenerateTimeSeries(data, gender);
                
                // Calculate CoM positions for all frames
                var comPositions = CalculateAllCoMPositions(data, gender);
                
                // Calculate jump metrics
                var jumpMetrics = CalculateJumpMetrics(data, phases.ToList(), comPositions);

                return new CMJAnalysisResult
                {
                    Type = AnalysisType.CounterMovementJump,
                    IsSuccess = true,
                    AnalyzedAt = DateTime.Now,
                    Summary = $"CMJ Analysis Complete - {dominance}",
                    
                    SubjectGender = gender,
                    SubjectMassKg = bodyMassKg,
                    
                    LowestCoMFrame = lowestCoMFrame,
                    TakeoffFrame = jumpMetrics.takeoffFrame,
                    LandingFrame = jumpMetrics.landingFrame,
                    PeakFlightFrame = jumpMetrics.peakFrame,
                    
                    HipKneeRatio = ratio,
                    Dominance = dominance,
                    HipMomentEstimate = ratio > 1 ? ratio : 1f,
                    KneeMomentEstimate = ratio < 1 ? 1f / ratio : 1f,
                    
                    LeftKneeValgus = leftValgus,
                    RightKneeValgus = rightValgus,
                    
                    Phases = phases,
                    TimeSeries = timeSeries,
                    CoMPositions = comPositions,
                    
                    JumpHeightMeters = jumpMetrics.height,
                    FlightTimeSeconds = jumpMetrics.flightTime,
                    ContactTimeSeconds = jumpMetrics.contactTime
                };
            });
        }

        /// <summary>
        /// Runs CMJ analysis with optional OpenSim GRF analysis.
        /// </summary>
        public async Task<CMJAnalysisResult> AnalyzeAsync(
            MotionData data,
            Gender gender,
            float bodyMassKg,
            float heightM,
            bool useOpenSimGRF,
            string? trcFilePath)
        {
            // First, run the standard CMJ analysis
            var result = await AnalyzeAsync(data, gender, bodyMassKg);

            // If OpenSim GRF analysis is requested and we have a TRC file
            if (useOpenSimGRF && !string.IsNullOrEmpty(trcFilePath))
            {
                try
                {
                    // Pass kinematics-derived events to refine GRF analysis
                    int? toFrame = result.TakeoffFrame > 0 ? result.TakeoffFrame : null;
                    int? landFrame = result.LandingFrame > 0 ? result.LandingFrame : null;

                    var grfData = await RunOpenSimGRFAnalysisAsync(trcFilePath, heightM, bodyMassKg, toFrame, landFrame);
                    
                    if (grfData != null)
                    {
                        // Create a new result with GRF data
                        return new CMJAnalysisResult
                        {
                            Type = result.Type,
                            IsSuccess = result.IsSuccess,
                            AnalyzedAt = result.AnalyzedAt,
                            Summary = result.Summary + " (with OpenSim GRF)",
                            
                            SubjectGender = result.SubjectGender,
                            SubjectMassKg = result.SubjectMassKg,
                            
                            LowestCoMFrame = result.LowestCoMFrame,
                            TakeoffFrame = grfData.TakeoffFrame ?? result.TakeoffFrame,
                            LandingFrame = grfData.LandingFrame ?? result.LandingFrame,
                            PeakFlightFrame = result.PeakFlightFrame,
                            
                            HipKneeRatio = result.HipKneeRatio,
                            Dominance = result.Dominance,
                            HipMomentEstimate = result.HipMomentEstimate,
                            KneeMomentEstimate = result.KneeMomentEstimate,
                            
                            LeftKneeValgus = result.LeftKneeValgus,
                            RightKneeValgus = result.RightKneeValgus,
                            
                            Phases = result.Phases,
                            TimeSeries = result.TimeSeries,
                            CoMPositions = result.CoMPositions,
                            
                            JumpHeightMeters = result.JumpHeightMeters,
                            FlightTimeSeconds = result.FlightTimeSeconds,
                            ContactTimeSeconds = result.ContactTimeSeconds,

                            // OpenSim GRF data
                            HasGRFData = true,
                            PeakVerticalGRF_N = grfData.PeakGRF,
                            NetVerticalImpulse_Ns = grfData.NetImpulse,
                            RFD_NPerS = grfData.RFD,
                            GRFTimeSeries = grfData.GRFTimeSeries,
                            GRFTimeValues = grfData.TimeValues
                        };
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"OpenSim GRF analysis failed: {ex.Message}");
                    // Fall back to standard result without GRF
                }
            }

            return result;
        }

        /// <summary>
        /// Runs OpenSim-based GRF analysis using Pose2Sim wrapper.
        /// </summary>
        private async Task<OpenSimGRFData?> RunOpenSimGRFAnalysisAsync(
            string trcFilePath, 
            float heightM, 
            float massKg,
            int? takeoffFrame = null,
            int? landingFrame = null)
        {
            var pose2sim = Pose2SimWrapperService.CreateFromConfig();

            // Check availability
            var (available, _, error) = await pose2sim.CheckAvailabilityAsync();
            if (!available)
            {
                System.Diagnostics.Debug.WriteLine($"Pose2Sim not available: {error}");
                return null;
            }

            // Ensure BodyKinematics CSV exists
            // The trcFilePath is used as a base to find the _bodykin.csv
            // Expected format: {filename}.trc -> {filename}_bodykin.csv in the same folder or opensim_output folder
            
            string csvPath = trcFilePath;
            
            // If input is not a CSV, look for the _bodykin.csv
            if (!csvPath.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
            {
                string dir = System.IO.Path.GetDirectoryName(trcFilePath) ?? "";
                string name = System.IO.Path.GetFileNameWithoutExtension(trcFilePath);
                
                // Strategy 1: Check in "opensim_output" subdirectory (standard workflow)
                string candidate1 = System.IO.Path.Combine(dir, "opensim_output", name + "_bodykin.csv");
                
                // Strategy 2: Check in same directory
                string candidate2 = System.IO.Path.Combine(dir, name + "_bodykin.csv");

                if (System.IO.File.Exists(candidate1)) csvPath = candidate1;
                else if (System.IO.File.Exists(candidate2)) csvPath = candidate2;
                else
                {
                    System.Diagnostics.Debug.WriteLine($"BodyKinematics CSV not found. Expected at: {candidate1} or {candidate2}");
                    return null;
                }
            }

            if (!System.IO.File.Exists(csvPath))
            {
                System.Diagnostics.Debug.WriteLine($"GRF Analysis aborted: Input CSV file does not exist: {csvPath}");
                return null;
            }

            // Step 4: Estimate GRF (Directly)
            var grfResult = await pose2sim.EstimateGRFAsync(csvPath, massKg, takeoffFrame, landingFrame);
            if (!grfResult.Success || grfResult.Metrics == null)
            {
                System.Diagnostics.Debug.WriteLine($"GRF estimation failed: {grfResult.Error}");
                return null;
            }

            return new OpenSimGRFData
            {
                PeakGRF = grfResult.Metrics.PeakVerticalGrfN,
                NetImpulse = grfResult.Metrics.NetVerticalImpulseNs,
                RFD = grfResult.Metrics.RfdNPerS,
                TakeoffFrame = grfResult.Metrics.TakeoffFrame,
                LandingFrame = null, // TODO: Parse from metrics if available
                GRFTimeSeries = grfResult.GrfTimeseries?.GrfVerticalN?.ToList() ?? new List<float>(),
                TimeValues = grfResult.GrfTimeseries?.TimeS?.ToList() ?? new List<float>()
            };
        }

        /// <summary>
        /// Internal data class for OpenSim GRF results.
        /// </summary>
        private class OpenSimGRFData
        {
            public float PeakGRF { get; set; }
            public float NetImpulse { get; set; }
            public float RFD { get; set; }
            public int? TakeoffFrame { get; set; }
            public int? LandingFrame { get; set; }
            public List<float> GRFTimeSeries { get; set; } = new();
            public List<float> TimeValues { get; set; } = new();
        }

        public int FindLowestCoMFrame(MotionData data)
        {
            int hipIndex = GetMarkerIndex(data, "Hip");
            if (hipIndex < 0) 
            {
                // Fallback: try to use average of RHip and LHip
                int rHipIdx = GetMarkerIndex(data, "RHip");
                int lHipIdx = GetMarkerIndex(data, "LHip");
                if (rHipIdx >= 0 && lHipIdx >= 0)
                {
                    return FindLowestCoMFrameFromBilateralHip(data, rHipIdx, lHipIdx);
                }
                return 0;
            }

            float lowestY = float.MaxValue;
            int lowestFrame = 0;

            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                var pos = data.Markers.GetPosition(hipIndex, frame);
                if (!float.IsNaN(pos.Y) && pos.Y > 0.01f && pos.Y < lowestY)
                {
                    lowestY = pos.Y;
                    lowestFrame = frame;
                }
            }

            return lowestFrame;
        }

        private int FindLowestCoMFrameFromBilateralHip(MotionData data, int rHipIdx, int lHipIdx)
        {
            float lowestY = float.MaxValue;
            int lowestFrame = 0;

            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                var rPos = data.Markers.GetPosition(rHipIdx, frame);
                var lPos = data.Markers.GetPosition(lHipIdx, frame);
                float avgY = (rPos.Y + lPos.Y) / 2f;
                if (!float.IsNaN(avgY) && avgY > 0.01f && avgY < lowestY)
                {
                    lowestY = avgY;
                    lowestFrame = frame;
                }
            }

            return lowestFrame;
        }

        /// <summary>
        /// Calculates the whole-body Center of Mass using De Leva segment data.
        /// </summary>
        private Vector3 CalculateWholeBodyCoM(MotionData data, int frame, Gender gender)
        {
            // Get all marker positions
            var hip = GetMarkerPosition(data, "Hip", frame);
            var neck = GetMarkerPosition(data, "Neck", frame);
            var head = GetMarkerPosition(data, "Head", frame);
            var rHip = GetMarkerPosition(data, "RHip", frame);
            var lHip = GetMarkerPosition(data, "LHip", frame);
            var rKnee = GetMarkerPosition(data, "RKnee", frame);
            var lKnee = GetMarkerPosition(data, "LKnee", frame);
            var rAnkle = GetMarkerPosition(data, "RAnkle", frame);
            var lAnkle = GetMarkerPosition(data, "LAnkle", frame);
            var rHeel = GetMarkerPosition(data, "RHeel", frame);
            var lHeel = GetMarkerPosition(data, "LHeel", frame);
            var rToe = GetMarkerPosition(data, "RBigToe", frame);
            var lToe = GetMarkerPosition(data, "LBigToe", frame);
            var rShoulder = GetMarkerPosition(data, "RShoulder", frame);
            var lShoulder = GetMarkerPosition(data, "LShoulder", frame);
            var rElbow = GetMarkerPosition(data, "RElbow", frame);
            var lElbow = GetMarkerPosition(data, "LElbow", frame);
            var rWrist = GetMarkerPosition(data, "RWrist", frame);
            var lWrist = GetMarkerPosition(data, "LWrist", frame);

            // Use Hip as fallback if Neck is missing
            if (neck == Vector3.Zero && hip != Vector3.Zero)
                neck = hip + new Vector3(0, 0.5f, 0); // Approximate

            // Calculate segment CoM positions
            var segmentCoMs = new List<(Vector3 position, float massPercent)>();

            // Head: from Neck to Head
            if (neck != Vector3.Zero && head != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Head);
                var headCoM = neck + (head - neck) * ratio;
                segmentCoMs.Add((headCoM, Legacy_BodySegmentMassModel.MassPercentage.Head(gender)));
            }

            // Trunk: from Hip to Neck (simplified as whole trunk)
            if (hip != Vector3.Zero && neck != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Trunk);
                // Trunk proximal is at cervicale (neck), so we go from neck toward hip
                var trunkCoM = neck + (hip - neck) * ratio;
                segmentCoMs.Add((trunkCoM, Legacy_BodySegmentMassModel.MassPercentage.Trunk(gender)));
            }

            // Right Thigh: from RHip to RKnee
            AddSegmentCoM(segmentCoMs, rHip, rKnee, gender, BodySegment.Thigh);
            // Left Thigh
            AddSegmentCoM(segmentCoMs, lHip, lKnee, gender, BodySegment.Thigh);

            // Right Shank: from RKnee to RAnkle
            AddSegmentCoM(segmentCoMs, rKnee, rAnkle, gender, BodySegment.Shank);
            // Left Shank
            AddSegmentCoM(segmentCoMs, lKnee, lAnkle, gender, BodySegment.Shank);

            // Right Foot: from RHeel to RToe
            if (rHeel != Vector3.Zero && rToe != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Foot);
                var footCoM = rHeel + (rToe - rHeel) * ratio;
                segmentCoMs.Add((footCoM, Legacy_BodySegmentMassModel.MassPercentage.Foot(gender)));
            }
            // Left Foot
            if (lHeel != Vector3.Zero && lToe != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Foot);
                var footCoM = lHeel + (lToe - lHeel) * ratio;
                segmentCoMs.Add((footCoM, Legacy_BodySegmentMassModel.MassPercentage.Foot(gender)));
            }

            // Right Upper Arm: from RShoulder to RElbow
            AddSegmentCoM(segmentCoMs, rShoulder, rElbow, gender, BodySegment.UpperArm);
            // Left Upper Arm
            AddSegmentCoM(segmentCoMs, lShoulder, lElbow, gender, BodySegment.UpperArm);

            // Right Forearm: from RElbow to RWrist
            AddSegmentCoM(segmentCoMs, rElbow, rWrist, gender, BodySegment.Forearm);
            // Left Forearm
            AddSegmentCoM(segmentCoMs, lElbow, lWrist, gender, BodySegment.Forearm);

            // Calculate weighted average
            if (segmentCoMs.Count == 0)
            {
                // Fallback to Hip if no segments available
                return hip != Vector3.Zero ? hip : (rHip + lHip) / 2f;
            }

            Vector3 weightedSum = Vector3.Zero;
            float totalMass = 0f;

            foreach (var (position, massPercent) in segmentCoMs)
            {
                if (position != Vector3.Zero && !float.IsNaN(position.Y))
                {
                    weightedSum += position * massPercent;
                    totalMass += massPercent;
                }
            }

            return totalMass > 0 ? weightedSum / totalMass : Vector3.Zero;
        }

        private void AddSegmentCoM(List<(Vector3 position, float massPercent)> list, 
            Vector3 proximal, Vector3 distal, Gender gender, BodySegment segment)
        {
            if (proximal != Vector3.Zero && distal != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, segment);
                var segmentCoM = proximal + (distal - proximal) * ratio;
                float massPercent = segment switch
                {
                    BodySegment.Thigh => Legacy_BodySegmentMassModel.MassPercentage.Thigh(gender),
                    BodySegment.Shank => Legacy_BodySegmentMassModel.MassPercentage.Shank(gender),
                    BodySegment.UpperArm => Legacy_BodySegmentMassModel.MassPercentage.UpperArm(gender),
                    BodySegment.Forearm => Legacy_BodySegmentMassModel.MassPercentage.Forearm(gender),
                    _ => 0f
                };
                list.Add((segmentCoM, massPercent));
            }
        }

        /// <summary>
        /// Finds the frame with lowest whole-body CoM using De Leva segment model.
        /// Searches for the first major dip (countermovement) to avoid capturing the landing phase.
        /// </summary>
        public int FindLowestCoMFrameUsingSegmentModel(MotionData data, Gender gender)
        {
            // Calculate initial standing height (average of first 5 valid frames)
            float standingHeight = 0;
            int validFrames = 0;
            for (int i = 0; i < Math.Min(10, data.Metadata.TotalFrames); i++)
            {
                var com = CalculateWholeBodyCoM(data, i, gender);
                if (!float.IsNaN(com.Y) && com.Y > 0.01f)
                {
                    standingHeight += com.Y;
                    validFrames++;
                }
            }
            
            if (validFrames > 0) standingHeight /= validFrames;
            else return 0; // Data invalid

            float lowestY = float.MaxValue;
            int lowestFrame = 0;
            bool descentStarted = false;

            // Scan frames
            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                var com = CalculateWholeBodyCoM(data, frame, gender);
                if (float.IsNaN(com.Y) || com.Y <= 0.01f) continue;

                // Threshold to start detecting descent (e.g., 3cm drop from standing)
                if (!descentStarted && com.Y < standingHeight - 0.03f)
                {
                    descentStarted = true;
                    lowestY = com.Y;
                    lowestFrame = frame;
                }

                if (descentStarted)
                {
                    // Track the lowest point
                    if (com.Y < lowestY)
                    {
                        lowestY = com.Y;
                        lowestFrame = frame;
                    }

                    // Stop condition: If we have risen significantly from the lowest point (e.g., 10cm)
                    // This indicates we are in the propulsion/flight phase, so we stop before finding landing.
                    if (frame > lowestFrame && com.Y > lowestY + 0.10f)
                    {
                        break; 
                    }
                }
            }

            // Fallback: If no descent detected, or just minimal movement, force global search if result is 0
            if (lowestFrame == 0)
            {
                for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
                {
                    var com = CalculateWholeBodyCoM(data, frame, gender);
                    if (!float.IsNaN(com.Y) && com.Y > 0.01f && com.Y < lowestY)
                    {
                        lowestY = com.Y;
                        lowestFrame = frame;
                    }
                }
            }

            return lowestFrame;
        }

        public (float ratio, DominanceType dominance) CalculateHipKneeRatio(
            MotionData data, int frame, Gender gender, float bodyMassKg)
        {
            // === Quasi-static Moment Analysis ===
            // At lowest CoM point, velocity ≈ 0, so we assume F = M × g
            // Moment = M × g × d (g cancels out in ratio)
            
            // 1. Calculate upper body CoM and mass (segments above hip)
            var (upperCoM, upperMassPercent) = CalculateUpperBodyCoMAndMass(data, frame, gender);
            float upperMass = bodyMassKg * upperMassPercent / 100f;
            
            // 2. Calculate thigh CoM and mass
            var (thighCoM, thighMassPercent) = CalculateThighCoMAndMass(data, frame, gender);
            float thighMass = bodyMassKg * thighMassPercent / 100f;
            
            // 3. Calculate above-knee composite CoM and total mass
            float aboveKneeMass = upperMass + thighMass;
            Vector3 aboveKneeCoM = aboveKneeMass > 0.001f 
                ? (upperCoM * upperMass + thighCoM * thighMass) / aboveKneeMass
                : Vector3.Zero;
            
            // 4. Get joint centers
            var rHip = GetMarkerPosition(data, "RHip", frame);
            var lHip = GetMarkerPosition(data, "LHip", frame);
            var rKnee = GetMarkerPosition(data, "RKnee", frame);
            var lKnee = GetMarkerPosition(data, "LKnee", frame);
            
            Vector3 hipCenter = (rHip + lHip) / 2f;
            Vector3 kneeCenter = (rKnee + lKnee) / 2f;
            
            // 5. Calculate Moment Arms (horizontal distance in X direction = Forward)
            // Hip Moment Arm: distance from upper body CoM to hip joint
            float hipMomentArm = Math.Abs(upperCoM.X - hipCenter.X);
            // Knee Moment Arm: distance from above-knee CoM to knee joint
            float kneeMomentArm = Math.Abs(aboveKneeCoM.X - kneeCenter.X);
            
            // 6. Calculate Moments (g cancels out in ratio, so omitted)
            // Hip Moment = M_upper × d_hip
            // Knee Moment = M_above_knee × d_knee
            float hipMoment = upperMass * hipMomentArm;
            float kneeMoment = aboveKneeMass * kneeMomentArm;
            
            // 7. Calculate Ratio
            float ratio = kneeMoment > 0.001f ? hipMoment / kneeMoment : 1f;
            
            // 8. Classify dominance (user-defined thresholds)
            DominanceType dominance;
            if (ratio > 1.1f)
                dominance = DominanceType.HipDominant;
            else if (ratio < 0.9f)
                dominance = DominanceType.KneeDominant;
            else
                dominance = DominanceType.Balanced;

            return (ratio, dominance);
        }
        
        /// <summary>
        /// Calculates the composite CoM and total mass percentage of upper body segments (above hip).
        /// Includes: Head, Trunk, Upper Arms, Forearms
        /// </summary>
        private (Vector3 CoM, float massPercent) CalculateUpperBodyCoMAndMass(MotionData data, int frame, Gender gender)
        {
            var segmentCoMs = new List<(Vector3 position, float massPercent)>();
            
            // Get marker positions
            var hip = GetMarkerPosition(data, "Hip", frame);
            var neck = GetMarkerPosition(data, "Neck", frame);
            var head = GetMarkerPosition(data, "Head", frame);
            var rShoulder = GetMarkerPosition(data, "RShoulder", frame);
            var lShoulder = GetMarkerPosition(data, "LShoulder", frame);
            var rElbow = GetMarkerPosition(data, "RElbow", frame);
            var lElbow = GetMarkerPosition(data, "LElbow", frame);
            var rWrist = GetMarkerPosition(data, "RWrist", frame);
            var lWrist = GetMarkerPosition(data, "LWrist", frame);
            
            // Fallback for missing neck
            if (neck == Vector3.Zero && hip != Vector3.Zero)
                neck = hip + new Vector3(0, 0.5f, 0);
            
            // Head
            if (neck != Vector3.Zero && head != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Head);
                var headCoM = neck + (head - neck) * ratio;
                segmentCoMs.Add((headCoM, Legacy_BodySegmentMassModel.MassPercentage.Head(gender)));
            }
            
            // Trunk
            if (hip != Vector3.Zero && neck != Vector3.Zero)
            {
                float ratio = Legacy_BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Trunk);
                var trunkCoM = neck + (hip - neck) * ratio;
                segmentCoMs.Add((trunkCoM, Legacy_BodySegmentMassModel.MassPercentage.Trunk(gender)));
            }
            
            // Upper Arms
            AddSegmentCoM(segmentCoMs, rShoulder, rElbow, gender, BodySegment.UpperArm);
            AddSegmentCoM(segmentCoMs, lShoulder, lElbow, gender, BodySegment.UpperArm);
            
            // Forearms
            AddSegmentCoM(segmentCoMs, rElbow, rWrist, gender, BodySegment.Forearm);
            AddSegmentCoM(segmentCoMs, lElbow, lWrist, gender, BodySegment.Forearm);
            
            // Calculate weighted average
            Vector3 weightedSum = Vector3.Zero;
            float totalMassPercent = 0f;
            
            foreach (var (position, massPercent) in segmentCoMs)
            {
                if (position != Vector3.Zero && !float.IsNaN(position.Y))
                {
                    weightedSum += position * massPercent;
                    totalMassPercent += massPercent;
                }
            }
            
            Vector3 compositeCoM = totalMassPercent > 0 ? weightedSum / totalMassPercent : Vector3.Zero;
            return (compositeCoM, totalMassPercent);
        }
        
        /// <summary>
        /// Calculates the composite CoM and total mass percentage of both thighs.
        /// </summary>
        private (Vector3 CoM, float massPercent) CalculateThighCoMAndMass(MotionData data, int frame, Gender gender)
        {
            var segmentCoMs = new List<(Vector3 position, float massPercent)>();
            
            var rHip = GetMarkerPosition(data, "RHip", frame);
            var lHip = GetMarkerPosition(data, "LHip", frame);
            var rKnee = GetMarkerPosition(data, "RKnee", frame);
            var lKnee = GetMarkerPosition(data, "LKnee", frame);
            
            // Right Thigh
            AddSegmentCoM(segmentCoMs, rHip, rKnee, gender, BodySegment.Thigh);
            // Left Thigh
            AddSegmentCoM(segmentCoMs, lHip, lKnee, gender, BodySegment.Thigh);
            
            // Calculate weighted average
            Vector3 weightedSum = Vector3.Zero;
            float totalMassPercent = 0f;
            
            foreach (var (position, massPercent) in segmentCoMs)
            {
                if (position != Vector3.Zero && !float.IsNaN(position.Y))
                {
                    weightedSum += position * massPercent;
                    totalMassPercent += massPercent;
                }
            }
            
            Vector3 compositeCoM = totalMassPercent > 0 ? weightedSum / totalMassPercent : Vector3.Zero;
            return (compositeCoM, totalMassPercent);
        }

        public (KneeValgusResult left, KneeValgusResult right) CalculateKneeValgus(
            MotionData data, int frame, Gender gender)
        {
            // Get marker positions
            var rHip = GetMarkerPosition(data, "RHip", frame);
            var lHip = GetMarkerPosition(data, "LHip", frame);
            var rKnee = GetMarkerPosition(data, "RKnee", frame);
            var lKnee = GetMarkerPosition(data, "LKnee", frame);
            var rAnkle = GetMarkerPosition(data, "RAnkle", frame);
            var lAnkle = GetMarkerPosition(data, "LAnkle", frame);

            // Calculate valgus angle in frontal plane (Z-Y projection)
            // Pass isRightLeg to handle sign correctly (Valgus +, Varus -)
            float rightValgus = CalculateFrontalPlaneAngle(rHip, rKnee, rAnkle, lHip, rHip, isRightLeg: true);
            float leftValgus = CalculateFrontalPlaneAngle(lHip, lKnee, lAnkle, lHip, rHip, isRightLeg: false);

            // Get normal ranges based on gender
            float minRange = gender == Gender.Male ? MaleValgusMin : FemaleValgusMin;
            float maxRange = gender == Gender.Male ? MaleValgusMax : FemaleValgusMax;

            // Classify risk
            var rightResult = new KneeValgusResult(
                rightValgus,
                ClassifyValgusRisk(rightValgus, minRange, maxRange),
                minRange,
                maxRange);

            var leftResult = new KneeValgusResult(
                leftValgus,
                ClassifyValgusRisk(leftValgus, minRange, maxRange),
                minRange,
                maxRange);

            return (leftResult, rightResult);
        }

        /// <summary>
        /// 논문 기반 CMJPhaseDetectionService를 사용하여 단계를 감지합니다.
        /// OpenSim CoM 데이터가 있는 경우 해당 데이터 사용, 없으면 마커 기반 CoM 계산.
        /// </summary>
        public IReadOnlyList<CMJPhaseInfo> DetectPhases(MotionData data, int lowestFrame)
        {
            // Calculate CoM positions using segment model
            var comPositions = CalculateAllCoMPositions(data, Gender.Male); // Gender fallback
            
            // Use new phase detection service
            var result = _phaseDetectionService.DetectPhases(comPositions, data, data.Metadata.FrameRate);
            
            if (result.IsSuccess)
            {
                return result.Phases;
            }
            
            // Fallback to minimal phase list if detection fails
            return new List<CMJPhaseInfo>
            {
                new CMJPhaseInfo(CMJPhase.LowestPoint, lowestFrame, lowestFrame, "Maximum knee flexion")
            };
        }

        /// <summary>
        /// OpenSim CoM 데이터를 사용하여 단계를 감지합니다 (권장).
        /// </summary>
        public CMJPhaseDetectionResult DetectPhasesWithOpenSimCoM(
            IReadOnlyList<Vector3> openSimComPositions, 
            MotionData data, 
            float frameRate)
        {
            return _phaseDetectionService.DetectPhases(openSimComPositions, data, frameRate);
        }

        public bool HasRequiredMarkers(MotionData data)
        {
            var markerNames = data.Metadata.MarkerNames;
            return RequiredMarkers.All(required => 
                markerNames.Any(m => m.Equals(required, StringComparison.OrdinalIgnoreCase)));
        }

        #region Private Helper Methods

        private int GetMarkerIndex(MotionData data, string markerName)
        {
            var names = data.Metadata.MarkerNames;
            for (int i = 0; i < names.Count; i++)
            {
                if (names[i].Equals(markerName, StringComparison.OrdinalIgnoreCase))
                    return i;
            }
            return -1;
        }

        private Vector3 GetMarkerPosition(MotionData data, string markerName, int frame)
        {
            int index = GetMarkerIndex(data, markerName);
            if (index < 0) return Vector3.Zero;
            return data.Markers.GetPosition(index, frame);
        }

        private float CalculateJointAngle(Vector3 p1, Vector3 p2, Vector3 p3)
        {
            var v1 = Vector3.Normalize(p1 - p2);
            var v2 = Vector3.Normalize(p3 - p2);
            float dot = Vector3.Dot(v1, v2);
            dot = Math.Clamp(dot, -1f, 1f);
            return MathF.Acos(dot) * (180f / MathF.PI);
        }

        /// <summary>
        /// Calculates knee valgus angle in the local frontal plane defined by pelvis orientation.
        /// </summary>
        private float CalculateFrontalPlaneAngle(Vector3 hip, Vector3 knee, Vector3 ankle, 
            Vector3 lHip, Vector3 rHip, bool isRightLeg)
        {
            // Check for invalid positions (zero or NaN)
            if (hip == Vector3.Zero || knee == Vector3.Zero || ankle == Vector3.Zero)
                return float.NaN;
            if (lHip == Vector3.Zero || rHip == Vector3.Zero)
                return float.NaN;

            // === Define Local Coordinate System based on Pelvis ===
            // Local Z (Right) = RHip - LHip direction
            Vector3 pelvisRight = rHip - lHip;
            if (pelvisRight.LengthSquared() < 0.001f)
                return float.NaN;
            Vector3 localZ = Vector3.Normalize(pelvisRight);
            
            // Local Y (Up) = Global Y axis (vertical)
            Vector3 localY = new Vector3(0, 1, 0);
            
            // Local X (Forward) = Y cross Z (right-hand rule)
            Vector3 localX = Vector3.Cross(localY, localZ);
            if (localX.LengthSquared() < 0.001f)
                return float.NaN;
            localX = Vector3.Normalize(localX);
            
            // Re-orthogonalize Y to ensure perfect orthogonality
            localY = Vector3.Cross(localZ, localX);
            localY = Vector3.Normalize(localY);

            // === Calculate vectors ===
            // Thigh extended direction (Knee - Hip) = direction from Hip toward Knee
            Vector3 thighExtended = knee - hip;
            
            // Shank direction (Ankle - Knee)
            Vector3 shank = ankle - knee;
            
            // === Project to Local Frontal Plane (Local Z - Local Y) ===
            // Project by taking dot products with local axes
            float thighLocalZ = Vector3.Dot(thighExtended, localZ);
            float thighLocalY = Vector3.Dot(thighExtended, localY);
            float shankLocalZ = Vector3.Dot(shank, localZ);
            float shankLocalY = Vector3.Dot(shank, localY);
            
            // Calculate magnitudes in local Z-Y plane
            float thighLen = MathF.Sqrt(thighLocalZ * thighLocalZ + thighLocalY * thighLocalY);
            float shankLen = MathF.Sqrt(shankLocalZ * shankLocalZ + shankLocalY * shankLocalY);
            
            if (thighLen < 0.001f || shankLen < 0.001f)
                return float.NaN;
            
            // Normalize
            float thighZn = thighLocalZ / thighLen;
            float thighYn = thighLocalY / thighLen;
            float shankZn = shankLocalZ / shankLen;
            float shankYn = shankLocalY / shankLen;
            
            // Dot product for angle
            float dot = thighZn * shankZn + thighYn * shankYn;
            dot = Math.Clamp(dot, -1f, 1f);
            float angleRad = MathF.Acos(dot);
            
            // Cross product (2D in local Z-Y) to determine direction
            float cross = thighZn * shankYn - thighYn * shankZn;
            
            // Convert to degrees
            float valgusDeg = angleRad * (180f / MathF.PI);
            
            // Apply sign based on cross product
            // Negative = Valgus (inward), Positive = Varus (outward)
            if (cross < 0)
                valgusDeg = -valgusDeg;
            
            return isRightLeg ? -valgusDeg : valgusDeg;
        }

        private ValgusRisk ClassifyValgusRisk(float angle, float minNormal, float maxNormal)
        {
            if (angle < minNormal)
                return ValgusRisk.BelowNormal;
            if (angle > maxNormal)
                return ValgusRisk.AboveNormal;
            return ValgusRisk.Normal;
        }

        private IReadOnlyList<CMJTimeSeriesPoint> GenerateTimeSeries(MotionData data, Gender gender)
        {
            var series = new List<CMJTimeSeriesPoint>();
            float frameRate = data.Metadata.FrameRate;

            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                var hip = GetMarkerPosition(data, "Hip", frame);
                var rHip = GetMarkerPosition(data, "RHip", frame);
                var lHip = GetMarkerPosition(data, "LHip", frame);
                var rKnee = GetMarkerPosition(data, "RKnee", frame);
                var lKnee = GetMarkerPosition(data, "LKnee", frame);
                var rAnkle = GetMarkerPosition(data, "RAnkle", frame);
                var lAnkle = GetMarkerPosition(data, "LAnkle", frame);

                float hipAngle = (CalculateJointAngle(hip, rHip, rKnee) + CalculateJointAngle(hip, lHip, lKnee)) / 2f;
                float kneeAngle = (CalculateJointAngle(rHip, rKnee, rAnkle) + CalculateJointAngle(lHip, lKnee, lAnkle)) / 2f;
                
                float rValgus = CalculateFrontalPlaneAngle(rHip, rKnee, rAnkle, lHip, rHip, true);
                float lValgus = CalculateFrontalPlaneAngle(lHip, lKnee, lAnkle, lHip, rHip, false);

                series.Add(new CMJTimeSeriesPoint(
                    frame,
                    frame / frameRate,
                    hip.Y,
                    hipAngle,
                    kneeAngle,
                    0f, // Ankle angle simplified
                    lValgus,
                    rValgus));
            }

            return series;
        }

        private IReadOnlyList<Vector3> CalculateAllCoMPositions(MotionData data, Gender gender)
        {
            var positions = new List<Vector3>(data.Metadata.TotalFrames);

            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                var com = CalculateWholeBodyCoM(data, frame, gender);
                
                // If CoM calculation returns Zero (failed), try fallback to Hip
                if (com == Vector3.Zero)
                {
                    var hipPos = GetMarkerPosition(data, "Hip", frame);
                    if (hipPos == Vector3.Zero)
                    {
                        // If even Hip is missing, try average of RHip/LHip
                        var rHip = GetMarkerPosition(data, "RHip", frame);
                        var lHip = GetMarkerPosition(data, "LHip", frame);
                        if (rHip != Vector3.Zero && lHip != Vector3.Zero)
                        {
                            hipPos = (rHip + lHip) * 0.5f;
                        }
                    }
                    com = hipPos;
                }
                
                positions.Add(com);
            }

            return positions;
        }

        private (int takeoffFrame, int landingFrame, int peakFrame, float height, float flightTime, float contactTime) 
            CalculateJumpMetrics(MotionData data, List<CMJPhaseInfo> phases, IReadOnlyList<Vector3> comPositions)
        {
            // Simplified: use phase detection results
            var propulsion = phases.FirstOrDefault(p => p.Phase == CMJPhase.Propulsion);
            var lowest = phases.FirstOrDefault(p => p.Phase == CMJPhase.LowestPoint);
            
            int takeoffFrame = propulsion?.EndFrame ?? 0;
            
            var landing = phases.FirstOrDefault(p => p.Phase == CMJPhase.LandingAbsorption);
            int landingFrame = landing != null ? landing.StartFrame : data.Metadata.TotalFrames - 1;
            
            // Peak Flight: between takeoff and landing
            int peakFrame = takeoffFrame > 0 && landingFrame > takeoffFrame 
                ? (takeoffFrame + landingFrame) / 2 
                : 0;

            // --- Jump Height Calculation (CoM Displacement Method) ---
            // 1. Get Standing Height (from initial 'Standing' phase or first few frames)
            float standingHeight = 0f;
            var standingPhase = phases.FirstOrDefault(p => p.Phase == CMJPhase.Standing);
            
            if (standingPhase != null)
            {
                // Average CoM Y during standing phase
                int count = 0;
                for (int i = standingPhase.StartFrame; i <= standingPhase.EndFrame; i++)
                {
                    if (i < comPositions.Count)
                    {
                        standingHeight += comPositions[i].Y;
                        count++;
                    }
                }
                if (count > 0) standingHeight /= count;
            }
            
            // Fallback if no standing phase detected or invalid
            if (standingHeight <= 0.01f && comPositions.Count > 0)
            {
                // Use first 10 frames
                int count = 0;
                for (int i = 0; i < Math.Min(10, comPositions.Count); i++)
                {
                    standingHeight += comPositions[i].Y;
                    count++;
                }
                if (count > 0) standingHeight /= count;
            }

            // 2. Get Peak Height (Max CoM Y between takeoff and landing)
            float peakHeight = 0f;
            if (takeoffFrame < landingFrame && takeoffFrame < comPositions.Count)
            {
                for (int i = takeoffFrame; i < Math.Min(landingFrame, comPositions.Count); i++)
                {
                    if (comPositions[i].Y > peakHeight)
                    {
                        peakHeight = comPositions[i].Y;
                        peakFrame = i; // Refine peak frame
                    }
                }
            }

            // 3. Calculate Jump Height (Displacement)
            // Ensure we subtract standing height, result can't be negative
            float height = Math.Max(0, peakHeight - standingHeight);
            
            float frameRate = data.Metadata.FrameRate;
            float flightTime = (landingFrame - takeoffFrame) / frameRate;
            
            // Contact Time: From movement start (Unweighting start) to Take-off
            var unweighting = phases.FirstOrDefault(p => p.Phase == CMJPhase.Unweighting);
            int movementStartFrame = unweighting?.StartFrame ?? 0;
            float contactTime = (takeoffFrame - movementStartFrame) / frameRate;

            return (takeoffFrame, landingFrame, peakFrame, height, flightTime, contactTime);
        }

        #endregion
    }
}
