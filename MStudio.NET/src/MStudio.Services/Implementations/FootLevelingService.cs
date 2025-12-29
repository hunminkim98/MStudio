using System;
using System.Collections.Generic;
using System.Linq;
using MStudio.Core.Interfaces;
using MStudio.Core.Models;

namespace MStudio.Services.Implementations
{
    /// <summary>
    /// 발 레벨링 서비스 구현.
    /// 발 마커의 평균 Y값이 가장 낮은 프레임을 찾아 모든 마커의 Y값을 조정합니다.
    /// </summary>
    public class FootLevelingService : IFootLevelingService
    {
        // 발 마커 이름 (Toe, Heel 마커들)
        private static readonly string[] FootMarkerNames =
        {
            "LBigToe", "RBigToe",
            "LSmallToe", "RSmallToe",
            "LHeel", "RHeel"
        };

        /// <inheritdoc/>
        public bool IsApplied { get; private set; }

        /// <inheritdoc/>
        public float AppliedOffset { get; private set; }

        /// <inheritdoc/>
        public bool ApplyFootLeveling(MotionData data)
        {
            if (IsApplied) return true; // 이미 적용됨
            if (data?.Markers == null) return false;

            var footIndices = GetFootMarkerIndices(data.Metadata.MarkerNames);
            if (footIndices.Count == 0) return false; // 발 마커 없음 - 호출자에게 알림

            // 발 마커의 평균 Y값이 가장 낮은 프레임 찾기
            float minAverageY = float.MaxValue;
            int groundFrame = 0;

            for (int frame = 0; frame < data.Markers.FrameCount; frame++)
            {
                float sumY = 0;
                int validCount = 0;
                bool hasNegativeY = false;

                foreach (int markerIdx in footIndices)
                {
                    var pos = data.Markers.GetPosition(markerIdx, frame);
                    if (!float.IsNaN(pos.Y) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        // 음수 Y값이 있으면 이 프레임 제외
                        if (pos.Y < 0)
                        {
                            hasNegativeY = true;
                            break;
                        }
                        sumY += pos.Y;
                        validCount++;
                    }
                }

                // 음수 Y가 포함된 프레임은 건너뛰기
                if (hasNegativeY) continue;

                if (validCount > 0)
                {
                    float avgY = sumY / validCount;
                    if (avgY < minAverageY)
                    {
                        minAverageY = avgY;
                        groundFrame = frame;
                    }
                }
            }

            if (minAverageY == float.MaxValue) return false; // 유효한 발 마커 없음

            // 오프셋 적용
            SubtractYOffset(data, minAverageY);
            AppliedOffset = minAverageY;
            IsApplied = true;
            return true;
        }

        /// <inheritdoc/>
        public void UndoFootLeveling(MotionData data)
        {
            if (!IsApplied) return; // 적용된 적 없음
            if (data?.Markers == null) return;

            // 역방향 적용 (오프셋 더하기)
            SubtractYOffset(data, -AppliedOffset);
            AppliedOffset = 0;
            IsApplied = false;
        }

        /// <inheritdoc/>
        public IReadOnlyList<int> GetFootMarkerIndices(IReadOnlyList<string> markerNames)
        {
            var indices = new List<int>();

            for (int i = 0; i < markerNames.Count; i++)
            {
                if (FootMarkerNames.Any(name =>
                    string.Equals(markerNames[i], name, StringComparison.OrdinalIgnoreCase)))
                {
                    indices.Add(i);
                }
            }

            return indices;
        }

        /// <inheritdoc/>
        public bool HasFootMarkers(IReadOnlyList<string> markerNames)
        {
            return GetFootMarkerIndices(markerNames).Count > 0;
        }

        /// <summary>
        /// 모든 마커의 Y값에서 오프셋을 차감합니다.
        /// </summary>
        private void SubtractYOffset(MotionData data, float offset)
        {
            var markers = data.Markers;

            for (int frame = 0; frame < markers.FrameCount; frame++)
            {
                for (int marker = 0; marker < markers.MarkerCount; marker++)
                {
                    var pos = markers.GetPosition(marker, frame);
                    if (!float.IsNaN(pos.X)) // 유효한 데이터만 처리
                    {
                        markers.SetPosition(marker, frame, pos.X, pos.Y - offset, pos.Z);
                    }
                }
            }
        }
    }
}
