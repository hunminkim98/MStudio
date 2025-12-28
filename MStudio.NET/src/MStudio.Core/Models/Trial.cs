using System;

namespace MStudio.Core.Models
{
    /// <summary>
    /// Represents a single motion capture trial.
    /// Wraps MotionData with additional metadata for multi-trial management.
    /// </summary>
    public record Trial
    {
        /// <summary>
        /// Unique identifier for this trial.
        /// </summary>
        public required string Id { get; init; }
        
        /// <summary>
        /// Display name for this trial (typically derived from filename).
        /// </summary>
        public required string Name { get; init; }
        
        /// <summary>
        /// The underlying motion data for this trial.
        /// </summary>
        public required MotionData MotionData { get; init; }
        
        /// <summary>
        /// Color index for visual distinction in multi-trial view.
        /// Used to assign unique colors to each trial.
        /// </summary>
        public int ColorIndex { get; init; }
        
        // Convenience properties
        
        /// <summary>
        /// Total number of frames in this trial.
        /// </summary>
        public int FrameCount => MotionData.Metadata.TotalFrames;
        
        /// <summary>
        /// Frame rate of this trial in Hz.
        /// </summary>
        public float FrameRate => MotionData.Metadata.FrameRate;
        
        /// <summary>
        /// File path from which this trial was loaded.
        /// </summary>
        public string FilePath => MotionData.Metadata.FilePath;
        
        /// <summary>
        /// Number of markers in this trial.
        /// </summary>
        public int MarkerCount => MotionData.Markers.MarkerCount;
    }
}
