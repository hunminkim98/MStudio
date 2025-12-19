using System;
using System.Collections.Generic;
using System.Numerics;

namespace MStudio.Core.Models
{
    public record MotionMetadata
    {
        public required string FilePath { get; init; }
        public required float FrameRate { get; init; }
        public required int TotalFrames { get; init; }
        public required IReadOnlyList<string> MarkerNames { get; init; }
        public string Units { get; init; } = "m";
    }

    public sealed class MarkerDataContainer : IDisposable
    {
        private float[] _x;
        private float[] _y;
        private float[] _z;
        private bool _disposed;

        public int MarkerCount { get; }
        public int FrameCount { get; }

        public MarkerDataContainer(int markerCount, int frameCount)
        {
            MarkerCount = markerCount;
            FrameCount = frameCount;
            int size = markerCount * frameCount;
            _x = new float[size];
            _y = new float[size];
            _z = new float[size];
        }

        public void SetPosition(int markerIndex, int frameIndex, float x, float y, float z)
        {
            int idx = frameIndex * MarkerCount + markerIndex;
            _x[idx] = x;
            _y[idx] = y;
            _z[idx] = z;
        }

        public Vector3 GetPosition(int markerIndex, int frameIndex)
        {
            int idx = frameIndex * MarkerCount + markerIndex;
            return new Vector3(_x[idx], _y[idx], _z[idx]);
        }

        public void FillGaps(int markerIndex, int maxGapSize)
        {
            int start = -1;
            for (int f = 0; f < FrameCount; f++)
            {
                var pos = GetPosition(markerIndex, f);
                bool isMissing = float.IsNaN(pos.X) || (pos.X == 0 && pos.Y == 0 && pos.Z == 0);

                if (isMissing)
                {
                    if (start == -1) start = f;
                }
                else
                {
                    if (start != -1)
                    {
                        int end = f;
                        int gapSize = end - start;

                        if (gapSize <= maxGapSize && start > 0)
                        {
                            var pStart = GetPosition(markerIndex, start - 1);
                            var pEnd = GetPosition(markerIndex, end);

                            for (int i = start; i < end; i++)
                            {
                                float t = (float)(i - (start - 1)) / (gapSize + 1);
                                var pInterp = Vector3.Lerp(pStart, pEnd, t);
                                SetPosition(markerIndex, i, pInterp.X, pInterp.Y, pInterp.Z);
                            }
                        }
                        start = -1;
                    }
                }
            }
        }

        public void SmoothData(int markerIndex, int windowSize)
        {
            if (windowSize <= 1) return;

            var newX = new float[FrameCount];
            var newY = new float[FrameCount];
            var newZ = new float[FrameCount];

            for (int f = 0; f < FrameCount; f++)
            {
                int count = 0;
                float sumX = 0, sumY = 0, sumZ = 0;

                int start = Math.Max(0, f - windowSize / 2);
                int end = Math.Min(FrameCount - 1, f + windowSize / 2);

                for (int i = start; i <= end; i++)
                {
                    var pos = GetPosition(markerIndex, i);
                    if (!float.IsNaN(pos.X) && !(pos.X == 0 && pos.Y == 0 && pos.Z == 0))
                    {
                        sumX += pos.X;
                        sumY += pos.Y;
                        sumZ += pos.Z;
                        count++;
                    }
                }

                if (count > 0)
                {
                    newX[f] = sumX / count;
                    newY[f] = sumY / count;
                    newZ[f] = sumZ / count;
                }
                else
                {
                    newX[f] = float.NaN;
                    newY[f] = float.NaN;
                    newZ[f] = float.NaN;
                }
            }

            // Apply back
            for (int f = 0; f < FrameCount; f++)
            {
                SetPosition(markerIndex, f, newX[f], newY[f], newZ[f]);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _x = null!;
                _y = null!;
                _z = null!;
                _disposed = true;
            }
        }
    }

    public record MotionData
    {
        public required MotionMetadata Metadata { get; init; }
        public required MarkerDataContainer Markers { get; init; }
    }
}
