namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Represents the phases of a Counter Movement Jump.
    /// </summary>
    public enum CMJPhase
    {
        /// <summary>Standing position before movement starts.</summary>
        Standing,

        /// <summary>Unweighting/countermovement phase - CoM descending.</summary>
        Unweighting,

        /// <summary>Braking phase - deceleration before lowest point.</summary>
        Braking,

        /// <summary>Lowest point - maximum knee/hip flexion.</summary>
        LowestPoint,

        /// <summary>Propulsion phase - pushing off the ground.</summary>
        Propulsion,

        /// <summary>Flight phase - airborne.</summary>
        Flight,

        /// <summary>Initial contact during landing.</summary>
        InitialContact,

        /// <summary>Landing absorption phase.</summary>
        LandingAbsorption,

        /// <summary>Recovery to standing position.</summary>
        Recovery
    }

    /// <summary>
    /// Represents a detected phase with frame range.
    /// </summary>
    public record CMJPhaseInfo(
        CMJPhase Phase,
        int StartFrame,
        int EndFrame,
        string Description);
}
