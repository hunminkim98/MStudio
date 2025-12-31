using System.Collections.Generic;
using System.Threading.Tasks;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// Service interface for movement-based analysis operations.
    /// Separate from IAnalysisService which handles mathematical calculations.
    /// </summary>
    public interface IMovementAnalysisService
    {
        /// <summary>
        /// Gets the list of available movement analysis types with their metadata.
        /// </summary>
        IReadOnlyList<AnalysisTypeInfo> GetAvailableAnalysisTypes();

        /// <summary>
        /// Runs the specified analysis on the provided motion data.
        /// </summary>
        Task<AnalysisResult> RunAnalysisAsync(AnalysisType type, MotionData? data);
    }
}
