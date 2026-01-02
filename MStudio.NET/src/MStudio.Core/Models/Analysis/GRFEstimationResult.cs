namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Result of GRF estimation from BCOM double differentiation.
    /// Based on: "Estimation of Ground Reaction Forces from Markerless Kinematics" (Colyer et al., 2023)
    /// </summary>
    public class GRFEstimationResult
    {
        public bool Success { get; set; }
        public string? Error { get; set; }
        public string? TrcPath { get; set; }
        public float MassKg { get; set; }
        public float FrameRate { get; set; }
        public int TotalFrames { get; set; }
        public GRFMetrics? Metrics { get; set; }
        public GRFTimeSeries? GrfTimeseries { get; set; }
    }

    /// <summary>
    /// GRF discrete metrics (peak force, impulse, RFD).
    /// </summary>
    public class GRFMetrics
    {
        /// <summary>Peak vertical ground reaction force in Newtons.</summary>
        public float PeakVerticalGrfN { get; set; }
        
        /// <summary>Net vertical impulse in Newton-seconds.</summary>
        public float NetVerticalImpulseNs { get; set; }
        
        /// <summary>Rate of Force Development in N/s.</summary>
        public float RfdNPerS { get; set; }
        
        /// <summary>Takeoff frame index (when GRF drops to 0).</summary>
        public int? TakeoffFrame { get; set; }
    }

    /// <summary>
    /// GRF time series data for visualization.
    /// </summary>
    public class GRFTimeSeries
    {
        /// <summary>Time values in seconds.</summary>
        public float[]? TimeS { get; set; }
        
        /// <summary>Vertical GRF values in Newtons.</summary>
        public float[]? GrfVerticalN { get; set; }
    }
}
