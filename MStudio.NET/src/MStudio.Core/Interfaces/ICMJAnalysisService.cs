using System.Threading.Tasks;
using MStudio.Core.Models;
using MStudio.Core.Models.Analysis;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// Service for Counter Movement Jump analysis.
    /// </summary>
    public interface ICMJAnalysisService
    {
        /// <summary>
        /// Runs a complete CMJ analysis on the motion data.
        /// </summary>
        /// <param name="data">Motion data containing marker positions.</param>
        /// <param name="gender">Subject gender for mass distribution calculations.</param>
        /// <param name="bodyMassKg">Subject body mass in kilograms.</param>
        /// <returns>Complete CMJ analysis result.</returns>
        Task<CMJAnalysisResult> AnalyzeAsync(MotionData data, Gender gender, float bodyMassKg);

        /// <summary>
        /// Finds the frame where Center of Mass is at its lowest point.
        /// </summary>
        int FindLowestCoMFrame(MotionData data);

        /// <summary>
        /// Calculates Hip/Knee moment ratio at a specific frame.
        /// </summary>
        (float ratio, DominanceType dominance) CalculateHipKneeRatio(
            MotionData data, 
            int frame, 
            Gender gender, 
            float bodyMassKg);

        /// <summary>
        /// Calculates knee valgus angles at a specific frame.
        /// </summary>
        (KneeValgusResult left, KneeValgusResult right) CalculateKneeValgus(
            MotionData data, 
            int frame, 
            Gender gender);

        /// <summary>
        /// Detects CMJ phases from motion data.
        /// </summary>
        System.Collections.Generic.IReadOnlyList<CMJPhaseInfo> DetectPhases(MotionData data, int lowestFrame);

        /// <summary>
        /// Checks if the motion data has the required markers for CMJ analysis.
        /// </summary>
        bool HasRequiredMarkers(MotionData data);
    }
}
