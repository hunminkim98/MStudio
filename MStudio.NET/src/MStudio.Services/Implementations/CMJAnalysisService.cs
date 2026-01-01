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
    public class CMJAnalysisService : ICMJAnalysisService
    {
        // Required marker names for CMJ analysis
        private static readonly string[] RequiredMarkers = new[]
        {
            "Hip", "RHip", "LHip", 
            "RKnee", "LKnee", 
            "RAnkle", "LAnkle"
        };

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
                var jumpMetrics = CalculateJumpMetrics(data, phases.ToList());

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
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Head);
                var headCoM = neck + (head - neck) * ratio;
                segmentCoMs.Add((headCoM, BodySegmentMassModel.MassPercentage.Head(gender)));
            }

            // Trunk: from Hip to Neck (simplified as whole trunk)
            if (hip != Vector3.Zero && neck != Vector3.Zero)
            {
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Trunk);
                // Trunk proximal is at cervicale (neck), so we go from neck toward hip
                var trunkCoM = neck + (hip - neck) * ratio;
                segmentCoMs.Add((trunkCoM, BodySegmentMassModel.MassPercentage.Trunk(gender)));
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
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Foot);
                var footCoM = rHeel + (rToe - rHeel) * ratio;
                segmentCoMs.Add((footCoM, BodySegmentMassModel.MassPercentage.Foot(gender)));
            }
            // Left Foot
            if (lHeel != Vector3.Zero && lToe != Vector3.Zero)
            {
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Foot);
                var footCoM = lHeel + (lToe - lHeel) * ratio;
                segmentCoMs.Add((footCoM, BodySegmentMassModel.MassPercentage.Foot(gender)));
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
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, segment);
                var segmentCoM = proximal + (distal - proximal) * ratio;
                float massPercent = segment switch
                {
                    BodySegment.Thigh => BodySegmentMassModel.MassPercentage.Thigh(gender),
                    BodySegment.Shank => BodySegmentMassModel.MassPercentage.Shank(gender),
                    BodySegment.UpperArm => BodySegmentMassModel.MassPercentage.UpperArm(gender),
                    BodySegment.Forearm => BodySegmentMassModel.MassPercentage.Forearm(gender),
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
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Head);
                var headCoM = neck + (head - neck) * ratio;
                segmentCoMs.Add((headCoM, BodySegmentMassModel.MassPercentage.Head(gender)));
            }
            
            // Trunk
            if (hip != Vector3.Zero && neck != Vector3.Zero)
            {
                float ratio = BodySegmentMassModel.GetCoMProximalRatio(gender, BodySegment.Trunk);
                var trunkCoM = neck + (hip - neck) * ratio;
                segmentCoMs.Add((trunkCoM, BodySegmentMassModel.MassPercentage.Trunk(gender)));
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
            float rightValgus = CalculateFrontalPlaneAngle(rHip, rKnee, rAnkle, isRightLeg: true);
            float leftValgus = CalculateFrontalPlaneAngle(lHip, lKnee, lAnkle, isRightLeg: false);

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

        public IReadOnlyList<CMJPhaseInfo> DetectPhases(MotionData data, int lowestFrame)
        {
            var phases = new List<CMJPhaseInfo>();
            int totalFrames = data.Metadata.TotalFrames;
            
            // Simple phase detection based on CoM position
            int hipIndex = GetMarkerIndex(data, "Hip");
            if (hipIndex < 0) return phases;

            // Get initial standing height
            var initialPos = data.Markers.GetPosition(hipIndex, 0);
            float standingHeight = initialPos.Y;

            // Detect descent start (when CoM drops below 99% of standing)
            int descentStart = 0;
            for (int f = 1; f < lowestFrame; f++)
            {
                var pos = data.Markers.GetPosition(hipIndex, f);
                if (pos.Y < standingHeight * 0.99f)
                {
                    descentStart = f;
                    break;
                }
            }

            // Add phases
            if (descentStart > 0)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Standing, 0, descentStart - 1, "Initial standing position"));
            }

            phases.Add(new CMJPhaseInfo(CMJPhase.Unweighting, descentStart, lowestFrame - 1, "Countermovement descent"));
            phases.Add(new CMJPhaseInfo(CMJPhase.LowestPoint, lowestFrame, lowestFrame, "Maximum knee flexion"));

            // Detect take-off (Propulsion End)
            // Logic: Toe height rises > 1cm from its position at LowestCoM
            int propulsionEnd = lowestFrame;
            
            // Get toe indices
            int rToeIdx = GetMarkerIndex(data, "RBigToe");
            int lToeIdx = GetMarkerIndex(data, "LBigToe");
            // Fallback
            if (rToeIdx < 0) rToeIdx = GetMarkerIndex(data, "RToe");
            if (lToeIdx < 0) lToeIdx = GetMarkerIndex(data, "LToe");

            if (rToeIdx >= 0 && lToeIdx >= 0)
            {
                // Get toe height at lowest frame
                var rToePos = data.Markers.GetPosition(rToeIdx, lowestFrame);
                var lToePos = data.Markers.GetPosition(lToeIdx, lowestFrame);
                float baseToeH = (rToePos.Y + lToePos.Y) / 2f;

                // Scan for 1cm rise
                for (int f = lowestFrame + 1; f < totalFrames; f++)
                {
                    var rP = data.Markers.GetPosition(rToeIdx, f);
                    var lP = data.Markers.GetPosition(lToeIdx, f);
                    float currentToeH = (rP.Y + lP.Y) / 2f;

                    if (currentToeH > baseToeH + 0.01f) // 1cm threshold
                    {
                        propulsionEnd = f;
                        break;
                    }
                }
            }
            else
            {
                // Fallback: use CoM return to standing height if no toes
                for (int f = lowestFrame + 1; f < totalFrames; f++)
                {
                    var pos = data.Markers.GetPosition(hipIndex, f);
                    if (pos.Y >= standingHeight)
                    {
                        propulsionEnd = f;
                        break;
                    }
                }
            }

            if (propulsionEnd > lowestFrame)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Propulsion, lowestFrame + 1, propulsionEnd, "Push-off phase"));
            }

            // Remaining frames as recovery
            if (propulsionEnd < totalFrames - 1)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Flight, propulsionEnd + 1, totalFrames - 1, "Flight phase")); 
                // Note: Simplified logic assumes remaining is flight/landing. 
                // Can refine later with landing detection using toe contact.
                
                // Detect landing based on Take-off toe height
                // User requirement: Landing is when toe height drops below Take-off toe height
                
                // Get toe height at Take-off frame (propulsionEnd)
                var rToePosTO = data.Markers.GetPosition(rToeIdx, propulsionEnd);
                var lToePosTO = data.Markers.GetPosition(lToeIdx, propulsionEnd);
                float toeHeightAtTakeoff = (rToePosTO.Y + lToePosTO.Y) / 2f;

                // Scan for landing
                // Start a bit after take-off to avoid immediate noise (e.g., 5 frames)
                 for (int f = propulsionEnd + 5; f < totalFrames; f++) 
                {
                    if (rToeIdx >= 0 && lToeIdx >= 0)
                    {
                         var rP = data.Markers.GetPosition(rToeIdx, f);
                         var lP = data.Markers.GetPosition(lToeIdx, f);
                         float currentToeH = (rP.Y + lP.Y) / 2f;
                         
                         // Landing if toe drops back to take-off height or lower
                         if (currentToeH <= toeHeightAtTakeoff + 0.005f) // Add small tolerance 5mm
                         {
                             // Split flight and landing
                             int flightEnd = f;
                             if (phases.Count > 0 && phases[^1].Phase == CMJPhase.Flight)
                             {
                                 phases[^1] = new CMJPhaseInfo(CMJPhase.Flight, propulsionEnd + 1, flightEnd - 1, "Flight phase");
                             }
                             else
                             {
                                 // Should be flight phase
                                 phases.Add(new CMJPhaseInfo(CMJPhase.Flight, propulsionEnd + 1, flightEnd - 1, "Flight phase"));
                             }
                             
                             phases.Add(new CMJPhaseInfo(CMJPhase.LandingAbsorption, flightEnd, totalFrames - 1, "Landing"));
                             break;
                         }
                    }
                }
            }

            return phases;
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

        private float CalculateFrontalPlaneAngle(Vector3 hip, Vector3 knee, Vector3 ankle, bool isRightLeg)
        {
            // Check for invalid positions (zero or NaN)
            if (hip == Vector3.Zero || knee == Vector3.Zero || ankle == Vector3.Zero)
                return float.NaN;

            // Calculate angles relative to vertical axis in Frontal Plane (Z-Y)
            // Based on user feedback/image: X is Forward(Depth), Z is Right(Width), Y is Up.
            // So Frontal Plane is Z-Y.

            // Thigh angle: Deviation of Hip-Knee vector from vertical
            float dzThigh = knee.Z - hip.Z;
            float dyThigh = knee.Y - hip.Y; // Negative value (Hip is higher)
            
            if (Math.Abs(dyThigh) < 0.001f) return float.NaN; 
            
            float thighAngleRad = MathF.Atan2(dzThigh, -dyThigh); 

            // Shank angle: Deviation of Knee-Ankle vector from vertical
            float dzShank = ankle.Z - knee.Z;
            float dyShank = ankle.Y - knee.Y; 
            
            if (Math.Abs(dyShank) < 0.001f) return float.NaN;

            float shankAngleRad = MathF.Atan2(dzShank, -dyShank);

            // Valgus = difference between thigh and shank angles
            float valgusRad = thighAngleRad - shankAngleRad;
            float valgusDeg = valgusRad * (180f / MathF.PI);

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
                
                float rValgus = CalculateFrontalPlaneAngle(rHip, rKnee, rAnkle, true);
                float lValgus = CalculateFrontalPlaneAngle(lHip, lKnee, lAnkle, false);

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
            CalculateJumpMetrics(MotionData data, List<CMJPhaseInfo> phases)
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
            
            // Estimate jump height from CoM displacement
            int hipIndex = GetMarkerIndex(data, "Hip");
            float standingHeight = hipIndex >= 0 ? data.Markers.GetPosition(hipIndex, 0).Y : 0;
            float lowestHeight = lowest != null && hipIndex >= 0 
                ? data.Markers.GetPosition(hipIndex, lowest.StartFrame).Y 
                : standingHeight;
            
            float frameRate = data.Metadata.FrameRate;
            float flightTime = (landingFrame - takeoffFrame) / frameRate;
            
            // Contact Time: From movement start (Unweighting start) to Take-off
            var unweighting = phases.FirstOrDefault(p => p.Phase == CMJPhase.Unweighting);
            int movementStartFrame = unweighting?.StartFrame ?? 0;
            float contactTime = (takeoffFrame - movementStartFrame) / frameRate;
            
            // Simple flight time method: h = 0.5 * g * (t/2)^2
            float height = 0.5f * 9.81f * MathF.Pow(flightTime / 2f, 2);

            return (takeoffFrame, landingFrame, peakFrame, height, flightTime, contactTime);
        }

        #endregion
    }
}
