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
        // HALPE_26 (full-body without hands, from AlphaPose, MMPose, etc.)
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
                // Hip to legs
                (19, 12), (19, 11), (19, 18),
                // Right leg
                (12, 14), (14, 16), (16, 21), (21, 23), (16, 25),
                // Left leg
                (11, 13), (13, 15), (15, 20), (20, 22), (15, 24),
                // Upper body
                (18, 17), (17, 0), (18, 5), (18, 6),
                // Arms
                (5, 7), (7, 9), (6, 8), (8, 10)
            }
        };

        // COCO_17 (full-body without hands and feet)
        public static readonly SkeletonDefinition Coco17 = new SkeletonDefinition
        {
            Name = "COCO_17",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "LEye"}, {2, "REye"}, {3, "LEar"}, {4, "REar"},
                {5, "LShoulder"}, {6, "RShoulder"}, {7, "LElbow"}, {8, "RElbow"},
                {9, "LWrist"}, {10, "RWrist"}, {11, "LHip"}, {12, "RHip"},
                {13, "LKnee"}, {14, "RKnee"}, {15, "LAnkle"}, {16, "RAnkle"}
            },
            Bones = new[]
            {
                // Torso
                (5, 6), (5, 11), (6, 12), (11, 12),
                // Right leg
                (12, 14), (14, 16),
                // Left leg
                (11, 13), (13, 15),
                // Right arm
                (6, 8), (8, 10),
                // Left arm
                (5, 7), (7, 9),
                // Head
                (0, 1), (0, 2), (1, 3), (2, 4)
            }
        };

        // BODY_25 (OpenPose standard)
        public static readonly SkeletonDefinition Body25 = new SkeletonDefinition
        {
            Name = "BODY_25",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "Neck"}, {2, "RShoulder"}, {3, "RElbow"}, {4, "RWrist"},
                {5, "LShoulder"}, {6, "LElbow"}, {7, "LWrist"}, {8, "CHip"},
                {9, "RHip"}, {10, "RKnee"}, {11, "RAnkle"}, {12, "LHip"},
                {13, "LKnee"}, {14, "LAnkle"}, {15, "REye"}, {16, "LEye"},
                {17, "REar"}, {18, "LEar"}, {19, "LBigToe"}, {20, "LSmallToe"},
                {21, "LHeel"}, {22, "RBigToe"}, {23, "RSmallToe"}, {24, "RHeel"}
            },
            Bones = new[]
            {
                // Head
                (0, 15), (0, 16), (15, 17), (16, 18),
                // Torso
                (0, 1), (1, 8),
                // Right arm
                (1, 2), (2, 3), (3, 4),
                // Left arm
                (1, 5), (5, 6), (6, 7),
                // Right leg
                (8, 9), (9, 10), (10, 11), (11, 22), (22, 23), (11, 24),
                // Left leg
                (8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21)
            }
        };

        // BODY_25B (OpenPose experimental)
        public static readonly SkeletonDefinition Body25B = new SkeletonDefinition
        {
            Name = "BODY_25B",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "LEye"}, {2, "REye"}, {3, "LEar"}, {4, "REar"},
                {5, "LShoulder"}, {6, "RShoulder"}, {7, "LElbow"}, {8, "RElbow"},
                {9, "LWrist"}, {10, "RWrist"}, {11, "LHip"}, {12, "RHip"},
                {13, "LKnee"}, {14, "RKnee"}, {15, "LAnkle"}, {16, "RAnkle"},
                {17, "Neck"}, {18, "Head"}, {19, "LBigToe"}, {20, "LSmallToe"},
                {21, "LHeel"}, {22, "RBigToe"}, {23, "RSmallToe"}, {24, "RHeel"}
            },
            Bones = new[]
            {
                // Head
                (17, 18), (18, 0),
                // Torso
                (17, 5), (17, 6),
                // Right arm
                (6, 8), (8, 10),
                // Left arm
                (5, 7), (7, 9),
                // Right leg
                (12, 14), (14, 16), (16, 22), (22, 23), (16, 24),
                // Left leg
                (11, 13), (13, 15), (15, 19), (19, 20), (15, 21)
            }
        };

        // COCO (OpenPose COCO format)
        public static readonly SkeletonDefinition Coco = new SkeletonDefinition
        {
            Name = "COCO",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "Neck"}, {2, "RShoulder"}, {3, "RElbow"}, {4, "RWrist"},
                {5, "LShoulder"}, {6, "LElbow"}, {7, "LWrist"}, {8, "RHip"},
                {9, "RKnee"}, {10, "RAnkle"}, {11, "LHip"}, {12, "LKnee"},
                {13, "LAnkle"}, {14, "REye"}, {15, "LEye"}, {16, "REar"}, {17, "LEar"}
            },
            Bones = new[]
            {
                // Head
                (0, 14), (0, 15), (14, 16), (15, 17),
                // Torso
                (0, 1), (1, 2), (1, 5), (1, 8), (1, 11),
                // Right arm
                (2, 3), (3, 4),
                // Left arm
                (5, 6), (6, 7),
                // Right leg
                (8, 9), (9, 10),
                // Left leg
                (11, 12), (12, 13)
            }
        };

        // MPII
        public static readonly SkeletonDefinition Mpii = new SkeletonDefinition
        {
            Name = "MPII",
            JointMap = new Dictionary<int, string>
            {
                {0, "Nose"}, {1, "Neck"}, {2, "RShoulder"}, {3, "RElbow"}, {4, "RWrist"},
                {5, "LShoulder"}, {6, "LElbow"}, {7, "LWrist"}, {8, "RHip"},
                {9, "RKnee"}, {10, "RAnkle"}, {11, "LHip"}, {12, "LKnee"},
                {13, "LAnkle"}, {14, "CHip"}
            },
            Bones = new[]
            {
                // Head
                (0, 1),
                // Torso
                (1, 2), (1, 5), (1, 14), (14, 8), (14, 11),
                // Right arm
                (2, 3), (3, 4),
                // Left arm
                (5, 6), (6, 7),
                // Right leg
                (8, 9), (9, 10),
                // Left leg
                (11, 12), (12, 13)
            }
        };

        // COCO_133 (full-body with hands and face, from MMPose RTMPose)
        public static readonly SkeletonDefinition Coco133 = new SkeletonDefinition
        {
            Name = "COCO_133",
            JointMap = new Dictionary<int, string>
            {
                // Body (0-22)
                {0, "Nose"}, {1, "LEye"}, {2, "REye"}, {3, "LEar"}, {4, "REar"},
                {5, "LShoulder"}, {6, "RShoulder"}, {7, "LElbow"}, {8, "RElbow"},
                {9, "LWrist"}, {10, "RWrist"}, {11, "LHip"}, {12, "RHip"},
                {13, "LKnee"}, {14, "RKnee"}, {15, "LAnkle"}, {16, "RAnkle"},
                {17, "LBigToe"}, {18, "LSmallToe"}, {19, "LHeel"},
                {20, "RBigToe"}, {21, "RSmallToe"}, {22, "RHeel"},
                // Left hand (92-111)
                {92, "LThumb1"}, {93, "LThumb"}, {94, "LThumb3"}, {95, "LThumb4"},
                {96, "LIndex"}, {97, "LIndex2"}, {98, "LIndex3"}, {99, "LIndex4"},
                {100, "LMiddle1"}, {101, "LMiddle2"}, {102, "LMiddle3"}, {103, "LMiddle4"},
                {104, "LRing1"}, {105, "LRing2"}, {106, "LRing3"}, {107, "LRing4"},
                {108, "LPinky"}, {109, "LPinky2"}, {110, "LPinky3"}, {111, "LPinky4"},
                // Right hand (113-132)
                {113, "RThumb1"}, {114, "RThumb"}, {115, "RThumb3"}, {116, "RThumb4"},
                {117, "RIndex"}, {118, "RIndex2"}, {119, "RIndex3"}, {120, "RIndex4"},
                {121, "RMiddle1"}, {122, "RMiddle2"}, {123, "RMiddle3"}, {124, "RMiddle4"},
                {125, "RRing1"}, {126, "RRing2"}, {127, "RRing3"}, {128, "RRing4"},
                {129, "RPinky"}, {130, "RPinky2"}, {131, "RPinky3"}, {132, "RPinky4"}
            },
            Bones = new[]
            {
                // Torso
                (5, 6), (5, 11), (6, 12), (11, 12),
                // Head
                (0, 1), (0, 2), (1, 3), (2, 4),
                // Right leg
                (12, 14), (14, 16), (16, 20), (20, 21), (16, 22),
                // Left leg
                (11, 13), (13, 15), (15, 17), (17, 18), (15, 19),
                // Right arm
                (6, 8), (8, 10),
                // Left arm
                (5, 7), (7, 9),
                // Right hand
                (10, 113), (113, 114), (114, 115), (115, 116),
                (10, 117), (117, 118), (118, 119), (119, 120),
                (10, 121), (121, 122), (122, 123), (123, 124),
                (10, 125), (125, 126), (126, 127), (127, 128),
                (10, 129), (129, 130), (130, 131), (131, 132),
                // Left hand
                (9, 92), (92, 93), (93, 94), (94, 95),
                (9, 96), (96, 97), (97, 98), (98, 99),
                (9, 100), (100, 101), (101, 102), (102, 103),
                (9, 104), (104, 105), (105, 106), (106, 107),
                (9, 108), (108, 109), (109, 110), (110, 111)
            }
        };

        // All available skeletons
        public static readonly IReadOnlyList<SkeletonDefinition> All = new[]
        {
            Halpe26,
            Coco17,
            Coco133,
            Body25,
            Body25B,
            Coco,
            Mpii
        };

        // Get skeleton by name
        public static SkeletonDefinition? GetByName(string name)
        {
            foreach (var skeleton in All)
            {
                if (string.Equals(skeleton.Name, name, System.StringComparison.OrdinalIgnoreCase))
                    return skeleton;
            }
            return null;
        }
    }
}
