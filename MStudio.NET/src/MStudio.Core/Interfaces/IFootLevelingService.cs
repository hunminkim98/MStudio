using System.Collections.Generic;
using MStudio.Core.Models;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// 발 레벨링 서비스 인터페이스.
    /// Markerless data에서 발이 지면(Y=0)에 닿도록 모든 마커의 Y값을 조정합니다.
    /// </summary>
    public interface IFootLevelingService
    {
        /// <summary>
        /// Foot leveling이 현재 적용되어 있는지 여부
        /// </summary>
        bool IsApplied { get; }
        
        /// <summary>
        /// 적용된 Y 오프셋 값 (Undo용)
        /// </summary>
        float AppliedOffset { get; }
        
        /// <summary>
        /// 모든 마커의 Y값에서 오프셋을 차감하여 발을 지면에 닿게 합니다.
        /// 발 마커의 평균 Y값이 가장 낮은 프레임을 기준으로 오프셋을 계산합니다.
        /// </summary>
        /// <returns>성공 시 true, 발 마커가 없으면 false</returns>
        bool ApplyFootLeveling(MotionData data);
        
        /// <summary>
        /// 적용된 Foot leveling을 취소합니다.
        /// </summary>
        void UndoFootLeveling(MotionData data);
        
        /// <summary>
        /// 발 마커 인덱스 목록을 반환합니다.
        /// </summary>
        IReadOnlyList<int> GetFootMarkerIndices(IReadOnlyList<string> markerNames);
        
        /// <summary>
        /// 마커 목록에 발 마커가 있는지 확인합니다.
        /// </summary>
        bool HasFootMarkers(IReadOnlyList<string> markerNames);
    }
}
