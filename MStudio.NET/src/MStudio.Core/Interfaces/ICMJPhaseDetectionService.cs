using System.Collections.Generic;
using System.Numerics;
using MStudio.Core.Models.Analysis;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// CMJ Phase Detection 서비스 인터페이스.
    /// 논문 기반 9-point 알고리즘을 사용하여 CMJ의 주요 단계를 감지합니다.
    /// 
    /// Reference: Sensors 2024, 24, 6624
    /// Key points:
    /// (a) Movement Start - CoM velocity > 5% of max
    /// (c) Braking Start - First velocity minimum
    /// (d) Lowest CoM - Minimum CoM height before take-off
    /// (e) Max Velocity - Peak vertical velocity
    /// (f) Take-off - Toe lifts off ground
    /// (g) Peak Height - Maximum CoM height
    /// (h) Landing - Toe contacts ground
    /// (i) Landing Depth - Second CoM minimum after landing
    /// </summary>
    public interface ICMJPhaseDetectionService
    {
        /// <summary>
        /// CoM 및 Toe 위치 데이터를 기반으로 CMJ 단계를 감지합니다.
        /// </summary>
        /// <param name="comPositions">OpenSim에서 계산된 CoM 위치 (프레임별)</param>
        /// <param name="toePositions">평균 Toe 위치 (RBigToe + LBigToe) / 2 (프레임별)</param>
        /// <param name="frameRate">프레임 레이트 (Hz)</param>
        /// <returns>감지된 단계 정보</returns>
        CMJPhaseDetectionResult DetectPhases(
            IReadOnlyList<Vector3> comPositions,
            IReadOnlyList<Vector3> toePositions,
            float frameRate);

        /// <summary>
        /// 마커 데이터에서 직접 Toe 위치를 추출하여 단계를 감지합니다.
        /// </summary>
        /// <param name="comPositions">OpenSim에서 계산된 CoM 위치 (프레임별)</param>
        /// <param name="motionData">마커 데이터 (RBigToe, LBigToe 추출용)</param>
        /// <param name="frameRate">프레임 레이트 (Hz)</param>
        /// <returns>감지된 단계 정보</returns>
        CMJPhaseDetectionResult DetectPhases(
            IReadOnlyList<Vector3> comPositions,
            MStudio.Core.Models.MotionData motionData,
            float frameRate);
    }
}
