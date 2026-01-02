using System;
using System.Collections.Generic;

namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// CMJ Phase Detection 결과.
    /// 논문 기반 9개의 핵심 프레임과 단계 정보를 포함합니다.
    /// 
    /// Reference: Sensors 2024, 24, 6624 - Figure 3
    /// </summary>
    public record CMJPhaseDetectionResult
    {
        /// <summary>감지 성공 여부</summary>
        public bool IsSuccess { get; init; } = true;

        /// <summary>오류 메시지 (실패 시)</summary>
        public string? ErrorMessage { get; init; }

        // ========== 9 Key Frames (논문 기준) ==========

        /// <summary>
        /// (a) Movement Start Frame.
        /// CoM 속도가 최대 속도의 5%를 초과하는 첫 프레임 (하강 방향).
        /// </summary>
        public int MovementStartFrame { get; init; }

        /// <summary>
        /// (c) Braking Start Frame.
        /// 첫 번째 속도 최소값 (음수 피크) - 상승 전환점.
        /// </summary>
        public int BrakingStartFrame { get; init; }

        /// <summary>
        /// (d) Lowest CoM Frame.
        /// Take-off 전 CoM 높이 최소값 - Eccentric phase 끝.
        /// </summary>
        public int LowestCoMFrame { get; init; }

        /// <summary>
        /// (e) Max Velocity Frame.
        /// CoM 수직 속도 최대값 (양수 피크).
        /// </summary>
        public int MaxVelocityFrame { get; init; }

        /// <summary>
        /// (f) Take-off Frame ⭐
        /// Toe가 지면에서 떨어지는 첫 프레임.
        /// 기준: Toe 높이 > 초기 높이 + threshold (1cm)
        /// </summary>
        public int TakeoffFrame { get; init; }

        /// <summary>
        /// (g) Peak Height Frame.
        /// CoM 높이 최대값 - 비행 정점.
        /// </summary>
        public int PeakHeightFrame { get; init; }

        /// <summary>
        /// (h) Landing Frame ⭐
        /// Toe가 지면에 착지하는 첫 프레임.
        /// 기준: Toe 높이 ≤ Take-off 높이 + tolerance (5mm)
        /// </summary>
        public int LandingFrame { get; init; }

        /// <summary>
        /// (i) Landing Depth Frame.
        /// 착지 후 두 번째 CoM 높이 최소값 - Landing absorption 끝.
        /// </summary>
        public int LandingDepthFrame { get; init; }

        // ========== Derived Data ==========

        /// <summary>감지된 CMJ 단계 목록</summary>
        public IReadOnlyList<CMJPhaseInfo> Phases { get; init; } = Array.Empty<CMJPhaseInfo>();

        /// <summary>CoM 수직 속도 (Vz) 시계열 데이터 (m/s)</summary>
        public IReadOnlyList<float> CoMVelocityZ { get; init; } = Array.Empty<float>();

        /// <summary>Take-off 시점 Toe 높이 (m)</summary>
        public float ToeHeightAtTakeoff { get; init; }

        /// <summary>착지 시점 Toe 높이 (m)</summary>
        public float ToeHeightAtLanding { get; init; }

        // ========== Computed Properties (논문 Table 1 기준) ==========

        /// <summary>Unweighting phase 시간 (초) - a ~ c</summary>
        public float UnweightingPhaseSeconds(float frameRate) => 
            frameRate > 0 ? (BrakingStartFrame - MovementStartFrame) / frameRate : 0;

        /// <summary>Braking phase 시간 (초) - c ~ d</summary>
        public float BrakingPhaseSeconds(float frameRate) => 
            frameRate > 0 ? (LowestCoMFrame - BrakingStartFrame) / frameRate : 0;

        /// <summary>Eccentric phase 시간 (초) - a ~ d</summary>
        public float EccentricPhaseSeconds(float frameRate) => 
            frameRate > 0 ? (LowestCoMFrame - MovementStartFrame) / frameRate : 0;

        /// <summary>Propulsive phase 시간 (초) - d ~ f</summary>
        public float PropulsivePhaseSeconds(float frameRate) => 
            frameRate > 0 ? (TakeoffFrame - LowestCoMFrame) / frameRate : 0;

        /// <summary>Take-off phase 시간 (초) - a ~ f (움직임 시작부터 Take-off까지)</summary>
        public float TakeoffPhaseSeconds(float frameRate) => 
            frameRate > 0 ? (TakeoffFrame - MovementStartFrame) / frameRate : 0;

        /// <summary>Landing eccentric phase 시간 (초) - h ~ i</summary>
        public float LandingEccentricPhaseSeconds(float frameRate) => 
            frameRate > 0 ? (LandingDepthFrame - LandingFrame) / frameRate : 0;

        /// <summary>비행 시간 (초) - f ~ h</summary>
        public float FlightTimeSeconds(float frameRate) => 
            frameRate > 0 ? (LandingFrame - TakeoffFrame) / frameRate : 0;

        /// <summary>접촉 시간 (초) - 움직임 시작부터 Take-off까지 (= TakeoffPhaseSeconds)</summary>
        public float ContactTimeSeconds(float frameRate) => 
            TakeoffPhaseSeconds(frameRate);
    }
}
