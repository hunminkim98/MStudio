namespace MStudio.Core.Models.Analysis
{
    /// <summary>
    /// Gender enumeration for body segment mass calculations.
    /// </summary>
    public enum Gender
    {
        Male,
        Female
    }

    /// <summary>
    /// [LEGACY] Body segment mass ratios based on De Leva (1996).
    /// Percentages represent body mass distribution for each segment.
    /// Superseded by OpenSim model-based calculations. Kept for reference.
    /// </summary>
    public static class Legacy_BodySegmentMassModel
    {
        /// <summary>
        /// Body segment mass percentages by gender.
        /// Values are percentage of total body mass.
        /// </summary>
        public static class MassPercentage
        {
            // Head
            public static float Head(Gender gender) => gender == Gender.Male ? 6.94f : 6.68f;

            // Trunk (Upper + Middle + Lower combined)
            public static float Trunk(Gender gender) => gender == Gender.Male ? 43.46f : 42.57f;

            // Upper Arm (single arm)
            public static float UpperArm(Gender gender) => gender == Gender.Male ? 2.71f : 2.55f;

            // Forearm (single arm)
            public static float Forearm(Gender gender) => gender == Gender.Male ? 1.62f : 1.38f;

            // Hand (single hand)
            public static float Hand(Gender gender) => gender == Gender.Male ? 0.61f : 0.56f;

            // Thigh (single leg)
            public static float Thigh(Gender gender) => gender == Gender.Male ? 14.16f : 14.78f;

            // Shank (single leg)
            public static float Shank(Gender gender) => gender == Gender.Male ? 4.33f : 4.81f;

            // Foot (single foot)
            public static float Foot(Gender gender) => gender == Gender.Male ? 1.37f : 1.29f;
        }

        /// <summary>
        /// Calculates segment mass in kg based on total body mass.
        /// </summary>
        public static float GetSegmentMass(float totalBodyMass, Gender gender, BodySegment segment)
        {
            float percentage = segment switch
            {
                BodySegment.Head => MassPercentage.Head(gender),
                BodySegment.Trunk => MassPercentage.Trunk(gender),
                BodySegment.UpperArm => MassPercentage.UpperArm(gender),
                BodySegment.Forearm => MassPercentage.Forearm(gender),
                BodySegment.Hand => MassPercentage.Hand(gender),
                BodySegment.Thigh => MassPercentage.Thigh(gender),
                BodySegment.Shank => MassPercentage.Shank(gender),
                BodySegment.Foot => MassPercentage.Foot(gender),
                _ => 0f
            };

            return totalBodyMass * (percentage / 100f);
        }

        /// <summary>
        /// Gets the center of mass position ratio along the segment.
        /// Values from De Leva (1996) - proximal ratio (% from proximal joint).
        /// Reference points vary by segment:
        /// - Head: from Vertex
        /// - Trunk: from Cervicale (C7)
        /// - Upper Arm: from Shoulder JC
        /// - Forearm: from Elbow JC
        /// - Hand: from Wrist JC
        /// - Thigh: from Hip JC
        /// - Shank: from Knee JC
        /// - Foot: from Heel
        /// </summary>
        public static float GetCoMProximalRatio(Gender gender, BodySegment segment)
        {
            return segment switch
            {
                BodySegment.Head => gender == Gender.Male ? 0.5002f : 0.4841f,
                BodySegment.Trunk => gender == Gender.Male ? 0.4413f : 0.4423f,  // Whole Trunk from Cervicale
                BodySegment.UpperArm => gender == Gender.Male ? 0.5772f : 0.5754f,
                BodySegment.Forearm => gender == Gender.Male ? 0.4574f : 0.4559f,
                BodySegment.Hand => gender == Gender.Male ? 0.7900f : 0.7474f,
                BodySegment.Thigh => gender == Gender.Male ? 0.4095f : 0.3612f,
                BodySegment.Shank => gender == Gender.Male ? 0.4459f : 0.4416f,
                BodySegment.Foot => gender == Gender.Male ? 0.4415f : 0.4014f,  // from Heel
                _ => 0.5f
            };
        }
    }

    /// <summary>
    /// Body segment enumeration for mass calculations.
    /// </summary>
    public enum BodySegment
    {
        Head,
        Trunk,
        UpperArm,
        Forearm,
        Hand,
        Thigh,
        Shank,
        Foot
    }
}
