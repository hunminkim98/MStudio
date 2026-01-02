using System.Threading.Tasks;
using MStudio.Core.Models.Analysis;

namespace MStudio.Core.Interfaces
{
    /// <summary>
    /// Interface for Pose2Sim/OpenSim integration service.
    /// Provides GRF estimation and OpenSim workflow functionality.
    /// </summary>
    public interface IPose2SimService
    {
        /// <summary>
        /// Checks if Pose2Sim is available on the system.
        /// </summary>
        /// <returns>Availability status, version if available, and error message if not</returns>
        Task<(bool Available, string? Version, string? Error)> CheckAvailabilityAsync();

        /// <summary>
        /// Estimates Ground Reaction Force from a TRC file using BCOM double differentiation.
        /// Based on: "Estimation of Ground Reaction Forces from Markerless Kinematics" (Colyer et al., 2023)
        /// Formula: GRF_y = m * a_y + m * g
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="massKg">Subject body mass in kg</param>
        /// <returns>GRF estimation result with peak force, impulse, RFD, and time series</returns>
        Task<GRFEstimationResult> EstimateGRFAsync(string trcPath, float massKg);

        /// <summary>
        /// Runs OpenSim model scaling using Pose2Sim.
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="outputDir">Output directory for scaled model</param>
        /// <param name="heightM">Subject height in meters</param>
        /// <param name="massKg">Subject mass in kg</param>
        /// <param name="poseModel">Pose model name (default: COCO_133)</param>
        /// <returns>Path to scaled .osim model file</returns>
        Task<(bool Success, string? ScaledModelPath, string? Error)> ScaleModelAsync(
            string trcPath, string outputDir, float heightM, float massKg, string poseModel = "COCO_133");

        /// <summary>
        /// Runs OpenSim Inverse Kinematics using Pose2Sim.
        /// </summary>
        /// <param name="trcPath">Path to TRC file</param>
        /// <param name="outputDir">Output directory (must contain scaled .osim model)</param>
        /// <param name="poseModel">Pose model name (default: COCO_133)</param>
        /// <returns>Path to .mot motion file</returns>
        Task<(bool Success, string? MotionFilePath, string? Error)> RunInverseKinematicsAsync(
            string trcPath, string outputDir, string poseModel = "COCO_133");

        /// <summary>
        /// Runs BodyKinematics analysis to calculate Center of Mass (CoM) positions.
        /// </summary>
        Task<(bool Success, string? OutputCsvPath, float FrameRate, string? Error)> RunBodyKinematicsAsync(
            string motPath, string osimPath, string outputCsvPath, string direction = "yup");

        /// <summary>
        /// Reads BodyKinematics CSV file and returns CoM positions.
        /// </summary>
        System.Collections.Generic.List<System.Numerics.Vector3> LoadBodyKinematicsData(string csvPath);
    }
}
