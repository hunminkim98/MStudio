namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Defines the types of movement analysis available in the application.
    /// Designed for extensibility - new analysis types can be easily added.
    /// </summary>
    public enum AnalysisType
    {
        Gait,
        Running,
        Sprint,
        CounterMovementJump,
        BroadJump,
        ChangeOfDirection,
        BodyKinematics  // OpenSim-based body kinematics analysis
    }
}
