using System;
using System.Collections.Generic;
using System.Numerics;

namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Classification of movement strategy based on Hip/Knee moment ratio.
    /// </summary>
    public enum DominanceType
    {
        /// <summary>Ratio > 1.1 - more hip contribution.</summary>
        HipDominant,

        /// <summary>Ratio < 0.9 - more knee contribution.</summary>
        KneeDominant,

        /// <summary>0.9 ≤ Ratio ≤ 1.1 - balanced contribution.</summary>
        Balanced
    }

    /// <summary>
    /// Knee valgus risk classification.
    /// </summary>
    public enum ValgusRisk
    {
        /// <summary>Within normal range for gender.</summary>
        Normal,

        /// <summary>Below normal range - excessive varus.</summary>
        BelowNormal,

        /// <summary>Above normal range - excessive valgus (higher injury risk).</summary>
        AboveNormal
    }

    /// <summary>
    /// Result of knee valgus analysis for one leg.
    /// </summary>
    public record KneeValgusResult(
        float AngleDegrees,
        ValgusRisk Risk,
        float NormalRangeMin,
        float NormalRangeMax);

    /// <summary>
    /// Time-series data point for CMJ analysis.
    /// </summary>
    public record CMJTimeSeriesPoint(
        int Frame,
        float Time,
        float CoMHeight,
        float HipAngle,
        float KneeAngle,
        float AnkleAngle,
        float LeftValgus,
        float RightValgus);

    /// <summary>
    /// Complete CMJ analysis result.
    /// </summary>
    public class CMJAnalysisResult : AnalysisResult
    {
        /// <summary>Subject information.</summary>
        public Gender SubjectGender { get; init; }
        public float SubjectMassKg { get; init; }

        /// <summary>Key frame indices.</summary>
        public int LowestCoMFrame { get; init; }
        public int TakeoffFrame { get; init; }
        public int LandingFrame { get; init; }
        public int PeakFlightFrame { get; init; }

        /// <summary>Hip/Knee Moment Ratio analysis at lowest CoM.</summary>
        public float HipKneeRatio { get; init; }
        public DominanceType Dominance { get; init; }
        public float HipMomentEstimate { get; init; }
        public float KneeMomentEstimate { get; init; }

        /// <summary>Knee Valgus analysis at lowest CoM.</summary>
        public KneeValgusResult? LeftKneeValgus { get; init; }
        public KneeValgusResult? RightKneeValgus { get; init; }

        /// <summary>Detected phases.</summary>
        public IReadOnlyList<CMJPhaseInfo> Phases { get; init; } = Array.Empty<CMJPhaseInfo>();

        /// <summary>Time-series data for graphing.</summary>
        public IReadOnlyList<CMJTimeSeriesPoint> TimeSeries { get; init; } = Array.Empty<CMJTimeSeriesPoint>();

        /// <summary>Jump performance metrics.</summary>
        public float JumpHeightMeters { get; init; }
        public float FlightTimeSeconds { get; init; }
        public float ContactTimeSeconds { get; init; }

        /// <summary>Center of Mass positions per frame.</summary>
        public IReadOnlyList<Vector3> CoMPositions { get; init; } = Array.Empty<Vector3>();
    }
}
