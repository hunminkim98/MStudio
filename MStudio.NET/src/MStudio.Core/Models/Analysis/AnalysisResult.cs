using System;

namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Base class for analysis results.
    /// Can be extended for specific analysis types.
    /// </summary>
    public class AnalysisResult
    {
        public AnalysisType Type { get; init; }
        public DateTime AnalyzedAt { get; init; } = DateTime.Now;
        public string Summary { get; init; } = string.Empty;
        public bool IsSuccess { get; init; }
        public string? ErrorMessage { get; init; }
    }
}
