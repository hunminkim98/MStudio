using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using MStudio.Core.Interfaces;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// 논문 기반 CMJ Phase Detection 서비스.
    /// CoM velocity와 Toe height를 사용하여 9개의 핵심 포인트를 감지합니다.
    /// 
    /// Reference: Sensors 2024, 24, 6624
    /// "Markerless vs. Marker-Based Gait Analysis: A Proof of Concept Study"
    /// </summary>
    public class CMJPhaseDetectionService : ICMJPhaseDetectionService
    {
        // ===== Detection Thresholds (논문 기준) =====
        
        /// <summary>Movement start: CoM velocity > 5% of max velocity</summary>
        private const float VelocityThresholdPercent = 0.05f;
        
        /// <summary>Take-off: Toe height > initial + 1cm</summary>
        private const float ToeHeightThreshold = 0.01f;
        
        /// <summary>Landing: Toe height <= Take-off height + 5mm tolerance</summary>
        private const float LandingTolerance = 0.005f;
        
        /// <summary>Minimum frames after take-off before searching for landing</summary>
        private const int MinFlightFrames = 5;

        /// <summary>
        /// CoM 및 Toe 위치 데이터를 기반으로 CMJ 단계를 감지합니다.
        /// </summary>
        public CMJPhaseDetectionResult DetectPhases(
            IReadOnlyList<Vector3> comPositions,
            IReadOnlyList<Vector3> toePositions,
            float frameRate)
        {
            // Validation
            if (comPositions == null || comPositions.Count < 10)
            {
                return new CMJPhaseDetectionResult
                {
                    IsSuccess = false,
                    ErrorMessage = "CoM positions data is insufficient (min 10 frames required)"
                };
            }

            if (toePositions == null || toePositions.Count < 10)
            {
                return new CMJPhaseDetectionResult
                {
                    IsSuccess = false,
                    ErrorMessage = "Toe positions data is insufficient (min 10 frames required)"
                };
            }

            if (comPositions.Count != toePositions.Count)
            {
                return new CMJPhaseDetectionResult
                {
                    IsSuccess = false,
                    ErrorMessage = "CoM and Toe position counts must match"
                };
            }

            try
            {
                int totalFrames = comPositions.Count;
                float dt = 1.0f / frameRate;

                // Step 1: Calculate CoM vertical velocity (Vz)
                var velocityZ = CalculateVelocity(comPositions, dt);

                // Step 2: Detect 9 key points in sequence
                int a = FindMovementStart(velocityZ);
                int c = FindBrakingStart(velocityZ, a);
                int d = FindLowestCoM(comPositions, a, totalFrames);
                int e = FindMaxVelocity(velocityZ, d, totalFrames);
                int f = FindTakeoff(toePositions, d, totalFrames);
                int g = FindPeakHeight(comPositions, f, totalFrames);
                int h = FindLanding(toePositions, f, g, totalFrames);
                int i = FindLandingDepth(comPositions, h, totalFrames);

                // Get toe heights at key frames
                float toeHeightAtTakeoff = f < toePositions.Count ? toePositions[f].Y : 0;
                float toeHeightAtLanding = h < toePositions.Count ? toePositions[h].Y : 0;

                // Step 3: Generate phase list
                var phases = GeneratePhases(a, c, d, e, f, g, h, i, totalFrames);

                return new CMJPhaseDetectionResult
                {
                    IsSuccess = true,
                    MovementStartFrame = a,
                    BrakingStartFrame = c,
                    LowestCoMFrame = d,
                    MaxVelocityFrame = e,
                    TakeoffFrame = f,
                    PeakHeightFrame = g,
                    LandingFrame = h,
                    LandingDepthFrame = i,
                    Phases = phases,
                    CoMVelocityZ = velocityZ,
                    ToeHeightAtTakeoff = toeHeightAtTakeoff,
                    ToeHeightAtLanding = toeHeightAtLanding
                };
            }
            catch (Exception ex)
            {
                return new CMJPhaseDetectionResult
                {
                    IsSuccess = false,
                    ErrorMessage = $"Phase detection failed: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// 마커 데이터에서 Toe 위치를 추출하여 단계를 감지합니다.
        /// </summary>
        public CMJPhaseDetectionResult DetectPhases(
            IReadOnlyList<Vector3> comPositions,
            MotionData motionData,
            float frameRate)
        {
            // Extract toe positions from marker data
            var toePositions = ExtractToePositions(motionData);
            return DetectPhases(comPositions, toePositions, frameRate);
        }

        #region Private Methods - Velocity Calculation

        /// <summary>
        /// CoM 위치에서 수직 속도를 계산합니다 (중앙 차분법).
        /// </summary>
        private List<float> CalculateVelocity(IReadOnlyList<Vector3> positions, float dt)
        {
            var velocity = new List<float>(positions.Count);

            for (int i = 0; i < positions.Count; i++)
            {
                float vz;
                if (i == 0)
                {
                    // Forward difference for first frame
                    vz = (positions[1].Y - positions[0].Y) / dt;
                }
                else if (i == positions.Count - 1)
                {
                    // Backward difference for last frame
                    vz = (positions[i].Y - positions[i - 1].Y) / dt;
                }
                else
                {
                    // Central difference for middle frames
                    vz = (positions[i + 1].Y - positions[i - 1].Y) / (2 * dt);
                }

                velocity.Add(vz);
            }

            return velocity;
        }

        #endregion

        #region Private Methods - Key Point Detection

        /// <summary>
        /// (a) Movement Start: CoM 속도가 최대 속도의 5%를 초과하는 첫 프레임 (하강 방향).
        /// </summary>
        private int FindMovementStart(List<float> velocityZ)
        {
            // Find minimum velocity (max negative = fastest descent)
            float minVelocity = velocityZ.Min();
            float threshold = minVelocity * VelocityThresholdPercent;

            // Find first frame where velocity exceeds 5% of min (negative direction)
            for (int i = 0; i < velocityZ.Count; i++)
            {
                if (velocityZ[i] < threshold)
                {
                    return i;
                }
            }

            return 0;
        }

        /// <summary>
        /// (c) Braking Start: 첫 번째 속도 최소값 (가장 빠른 하강 속도).
        /// </summary>
        private int FindBrakingStart(List<float> velocityZ, int startFrame)
        {
            int minFrame = startFrame;
            float minVelocity = velocityZ[startFrame];

            // Search for first local minimum after movement start
            for (int i = startFrame + 1; i < velocityZ.Count; i++)
            {
                if (velocityZ[i] < minVelocity)
                {
                    minVelocity = velocityZ[i];
                    minFrame = i;
                }
                // Stop if velocity starts increasing (moving toward zero/positive)
                else if (velocityZ[i] > minVelocity + 0.1f)
                {
                    break;
                }
            }

            return minFrame;
        }

        /// <summary>
        /// (d) Lowest CoM: Take-off 전 CoM 높이 최소값.
        /// </summary>
        private int FindLowestCoM(IReadOnlyList<Vector3> comPositions, int startFrame, int totalFrames)
        {
            int minFrame = startFrame;
            float minHeight = comPositions[startFrame].Y;

            // Calculate initial standing height (first 10 frames average)
            float standingHeight = 0;
            int validCount = 0;
            for (int i = 0; i < Math.Min(10, totalFrames); i++)
            {
                if (!float.IsNaN(comPositions[i].Y) && comPositions[i].Y > 0.01f)
                {
                    standingHeight += comPositions[i].Y;
                    validCount++;
                }
            }
            standingHeight = validCount > 0 ? standingHeight / validCount : 0;

            // Search for lowest point
            for (int i = startFrame; i < totalFrames; i++)
            {
                float height = comPositions[i].Y;
                if (!float.IsNaN(height) && height > 0.01f && height < minHeight)
                {
                    minHeight = height;
                    minFrame = i;
                }
                // Stop if height rises significantly (entering propulsion phase)
                else if (height > minHeight + 0.10f)
                {
                    break;
                }
            }

            return minFrame;
        }

        /// <summary>
        /// (e) Max Velocity: CoM 수직 속도 최대값 (양수 피크).
        /// </summary>
        private int FindMaxVelocity(List<float> velocityZ, int afterFrame, int totalFrames)
        {
            int maxFrame = afterFrame;
            float maxVelocity = velocityZ[afterFrame];

            for (int i = afterFrame; i < totalFrames; i++)
            {
                if (velocityZ[i] > maxVelocity)
                {
                    maxVelocity = velocityZ[i];
                    maxFrame = i;
                }
            }

            return maxFrame;
        }

        /// <summary>
        /// (f) Take-off: Toe가 지면에서 떨어지는 첫 프레임.
        /// 기준: Toe 높이 > Lowest CoM 시점 높이 + 1cm threshold
        /// </summary>
        private int FindTakeoff(IReadOnlyList<Vector3> toePositions, int afterFrame, int totalFrames)
        {
            // Get baseline toe height at lowest CoM frame
            float baseToeHeight = toePositions[afterFrame].Y;

            // Search for take-off (toe rises above threshold)
            for (int i = afterFrame + 1; i < totalFrames; i++)
            {
                float currentToeHeight = toePositions[i].Y;
                
                if (currentToeHeight > baseToeHeight + ToeHeightThreshold)
                {
                    return i;
                }
            }

            // Fallback: if no clear take-off detected, estimate from velocity peak
            return Math.Min(afterFrame + 10, totalFrames - 1);
        }

        /// <summary>
        /// (g) Peak Height: CoM 높이 최대값 (비행 정점).
        /// </summary>
        private int FindPeakHeight(IReadOnlyList<Vector3> comPositions, int afterFrame, int totalFrames)
        {
            int maxFrame = afterFrame;
            float maxHeight = comPositions[afterFrame].Y;

            for (int i = afterFrame; i < totalFrames; i++)
            {
                if (comPositions[i].Y > maxHeight)
                {
                    maxHeight = comPositions[i].Y;
                    maxFrame = i;
                }
            }

            return maxFrame;
        }

        /// <summary>
        /// (h) Landing: Toe가 지면에 착지하는 첫 프레임.
        /// 기준: Toe 높이 <= Take-off 높이 + 5mm tolerance
        /// </summary>
        private int FindLanding(IReadOnlyList<Vector3> toePositions, int takeoffFrame, int peakFrame, int totalFrames)
        {
            // Get toe height at take-off
            float toeHeightAtTakeoff = toePositions[takeoffFrame].Y;

            // Start searching after peak height + minimum flight frames
            int searchStart = Math.Max(peakFrame, takeoffFrame + MinFlightFrames);

            for (int i = searchStart; i < totalFrames; i++)
            {
                float currentToeHeight = toePositions[i].Y;

                // Landing: toe drops back to take-off height or lower
                if (currentToeHeight <= toeHeightAtTakeoff + LandingTolerance)
                {
                    return i;
                }
            }

            // Fallback: use last frame
            return totalFrames - 1;
        }

        /// <summary>
        /// (i) Landing Depth: 착지 후 두 번째 CoM 높이 최소값.
        /// </summary>
        private int FindLandingDepth(IReadOnlyList<Vector3> comPositions, int afterFrame, int totalFrames)
        {
            int minFrame = afterFrame;
            float minHeight = comPositions[afterFrame].Y;

            for (int i = afterFrame; i < totalFrames; i++)
            {
                if (comPositions[i].Y < minHeight)
                {
                    minHeight = comPositions[i].Y;
                    minFrame = i;
                }
            }

            return minFrame;
        }

        #endregion

        #region Private Methods - Phase Generation

        /// <summary>
        /// 감지된 키 프레임을 기반으로 단계 목록을 생성합니다.
        /// </summary>
        private List<CMJPhaseInfo> GeneratePhases(
            int a, int c, int d, int e, int f, int g, int h, int i, int totalFrames)
        {
            var phases = new List<CMJPhaseInfo>();

            // Standing phase (0 to a-1)
            if (a > 0)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Standing, 0, a - 1, "Initial standing position"));
            }

            // Unweighting phase (a to c-1)
            if (c > a)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Unweighting, a, c - 1, "Countermovement descent"));
            }

            // Braking phase (c to d-1)
            if (d > c)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Braking, c, d - 1, "Deceleration before lowest point"));
            }

            // Lowest point (d)
            phases.Add(new CMJPhaseInfo(CMJPhase.LowestPoint, d, d, "Maximum knee flexion"));

            // Propulsion phase (d+1 to f)
            if (f > d)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Propulsion, d + 1, f, "Push-off phase"));
            }

            // Flight phase (f+1 to h-1)
            if (h > f + 1)
            {
                phases.Add(new CMJPhaseInfo(CMJPhase.Flight, f + 1, h - 1, "Airborne phase"));
            }

            // Landing absorption (h to i or end)
            int landingEnd = i > h ? i : totalFrames - 1;
            phases.Add(new CMJPhaseInfo(CMJPhase.LandingAbsorption, h, landingEnd, "Landing absorption"));

            return phases;
        }

        #endregion

        #region Private Methods - Helper

        /// <summary>
        /// 마커 데이터에서 평균 Toe 위치를 추출합니다.
        /// </summary>
        private List<Vector3> ExtractToePositions(MotionData data)
        {
            var positions = new List<Vector3>(data.Metadata.TotalFrames);

            // Find toe marker indices
            int rToeIdx = GetMarkerIndex(data, "RBigToe");
            int lToeIdx = GetMarkerIndex(data, "LBigToe");

            // Fallback to alternative names
            if (rToeIdx < 0) rToeIdx = GetMarkerIndex(data, "RToe");
            if (lToeIdx < 0) lToeIdx = GetMarkerIndex(data, "LToe");

            bool hasRToe = rToeIdx >= 0;
            bool hasLToe = lToeIdx >= 0;

            for (int frame = 0; frame < data.Metadata.TotalFrames; frame++)
            {
                Vector3 rToePos = hasRToe ? data.Markers.GetPosition(rToeIdx, frame) : Vector3.Zero;
                Vector3 lToePos = hasLToe ? data.Markers.GetPosition(lToeIdx, frame) : Vector3.Zero;

                // Calculate average
                Vector3 avgPos;
                if (hasRToe && hasLToe)
                {
                    avgPos = (rToePos + lToePos) / 2f;
                }
                else if (hasRToe)
                {
                    avgPos = rToePos;
                }
                else if (hasLToe)
                {
                    avgPos = lToePos;
                }
                else
                {
                    avgPos = Vector3.Zero;
                }

                positions.Add(avgPos);
            }

            return positions;
        }

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

        #endregion
    }
}
