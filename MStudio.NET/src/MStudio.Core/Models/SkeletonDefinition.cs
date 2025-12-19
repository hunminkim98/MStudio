using System.Collections.Generic;

namespace MStudio.Core.Models
{
    public record SkeletonDefinition
    {
        public required string Name { get; init; }
        public required IReadOnlyDictionary<int, string> JointMap { get; init; }
        public required IReadOnlyList<(int Parent, int Child)> Bones { get; init; }
    }

    public static class PredefinedSkeletons
    {
        public static readonly SkeletonDefinition Halpe26 = new SkeletonDefinition
        {
            Name = "HALPE_26",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "LEye"}, {2, "REye"}, {3, "LEar"}, {4, "REar"},
                {5, "LShoulder"}, {6, "RShoulder"}, {7, "LElbow"}, {8, "RElbow"},
                {9, "LWrist"}, {10, "RWrist"}, {11, "LHip"}, {12, "RHip"},
                {13, "LKnee"}, {14, "RKnee"}, {15, "LAnkle"}, {16, "RAnkle"},
                {17, "Head"}, {18, "Neck"}, {19, "Hip"}, {20, "LBigToe"}, {21, "RBigToe"},
                {22, "LSmallToe"}, {23, "RSmallToe"}, {24, "LHeel"}, {25, "RHeel"}
            },
            Bones = new[]
            {
                (19, 12), (19, 11), (19, 18),
                (12, 14), (14, 16), (16, 21), (21, 23), (16, 25),
                (11, 13), (13, 15), (15, 20), (20, 22), (15, 24),
                (18, 17), (17, 0), (18, 5), (18, 6),
                (5, 7), (7, 9), (6, 8), (8, 10)
            }
        };
    }
}
