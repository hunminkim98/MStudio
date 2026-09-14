//! Automatic segment / joint detection from marker names, a port of
//! `utils/skeleton_config.py`. The first pattern whose markers all exist wins.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentDef {
    pub name: String,
    pub a: usize,
    pub b: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointDef {
    pub name: String,
    pub a: usize,
    pub vertex: usize,
    pub c: usize,
}

const SEGMENT_PATTERNS: &[(&str, &[[&str; 2]])] = &[
    (
        "Trunk",
        &[
            ["Neck", "Hip"],
            ["Neck", "CHip"],
            ["Neck", "RHip"],
            ["Neck", "LHip"],
            ["C7", "Hip"],
            ["C7", "CHip"],
            ["C7", "RHip"],
            ["C7", "LHip"],
            ["nose", "right_hip"],
            ["nose", "left_hip"],
        ],
    ),
    (
        "Head",
        &[
            ["Neck", "Head"],
            ["Neck", "Nose"],
            ["C7", "Head"],
            ["C7", "Nose"],
            ["nose", "right_eye"],
            ["nose", "left_eye"],
        ],
    ),
    ("Upper_Arm_R", &[["RShoulder", "RElbow"], ["right_shoulder", "right_elbow"]]),
    ("Upper_Arm_L", &[["LShoulder", "LElbow"], ["left_shoulder", "left_elbow"]]),
    ("Forearm_R", &[["RElbow", "RWrist"], ["right_elbow", "right_wrist"]]),
    ("Forearm_L", &[["LElbow", "LWrist"], ["left_elbow", "left_wrist"]]),
    ("Thigh_R", &[["RHip", "RKnee"], ["right_hip", "right_knee"]]),
    ("Thigh_L", &[["LHip", "LKnee"], ["left_hip", "left_knee"]]),
    ("Shank_R", &[["RKnee", "RAnkle"], ["right_knee", "right_ankle"]]),
    ("Shank_L", &[["LKnee", "LAnkle"], ["left_knee", "left_ankle"]]),
    (
        "Foot_R",
        &[
            ["RAnkle", "RBigToe"],
            ["RAnkle", "RSmallToe"],
            ["RAnkle", "RHeel"],
            ["right_ankle", "right_foot_index"],
            ["right_ankle", "right_heel"],
        ],
    ),
    (
        "Foot_L",
        &[
            ["LAnkle", "LBigToe"],
            ["LAnkle", "LSmallToe"],
            ["LAnkle", "LHeel"],
            ["left_ankle", "left_foot_index"],
            ["left_ankle", "left_heel"],
        ],
    ),
];

const JOINT_PATTERNS: &[(&str, &[[&str; 3]])] = &[
    (
        "Hip_R",
        &[
            ["Neck", "RHip", "RKnee"],
            ["C7", "RHip", "RKnee"],
            ["RShoulder", "RHip", "RKnee"],
            ["LShoulder", "RHip", "RKnee"],
            ["nose", "right_hip", "right_knee"],
            ["right_shoulder", "right_hip", "right_knee"],
        ],
    ),
    (
        "Hip_L",
        &[
            ["Neck", "LHip", "LKnee"],
            ["C7", "LHip", "LKnee"],
            ["RShoulder", "LHip", "LKnee"],
            ["LShoulder", "LHip", "LKnee"],
            ["nose", "left_hip", "left_knee"],
            ["left_shoulder", "left_hip", "left_knee"],
        ],
    ),
    ("Knee_R", &[["RHip", "RKnee", "RAnkle"], ["right_hip", "right_knee", "right_ankle"]]),
    ("Knee_L", &[["LHip", "LKnee", "LAnkle"], ["left_hip", "left_knee", "left_ankle"]]),
    (
        "Ankle_R",
        &[
            ["RKnee", "RAnkle", "RBigToe"],
            ["RKnee", "RAnkle", "RHeel"],
            ["right_knee", "right_ankle", "right_foot_index"],
            ["right_knee", "right_ankle", "right_heel"],
        ],
    ),
    (
        "Ankle_L",
        &[
            ["LKnee", "LAnkle", "LBigToe"],
            ["LKnee", "LAnkle", "LHeel"],
            ["left_knee", "left_ankle", "left_foot_index"],
            ["left_knee", "left_ankle", "left_heel"],
        ],
    ),
    (
        "Shoulder_R",
        &[["Neck", "RShoulder", "RElbow"], ["C7", "RShoulder", "RElbow"], ["nose", "right_shoulder", "right_elbow"]],
    ),
    (
        "Shoulder_L",
        &[["Neck", "LShoulder", "LElbow"], ["C7", "LShoulder", "LElbow"], ["nose", "left_shoulder", "left_elbow"]],
    ),
    ("Elbow_R", &[["RShoulder", "RElbow", "RWrist"], ["right_shoulder", "right_elbow", "right_wrist"]]),
    ("Elbow_L", &[["LShoulder", "LElbow", "LWrist"], ["left_shoulder", "left_elbow", "left_wrist"]]),
    (
        "Wrist_R",
        &[
            ["RElbow", "RWrist", "RThumb"],
            ["RElbow", "RWrist", "RIndex"],
            ["right_elbow", "right_wrist", "right_thumb"],
            ["right_elbow", "right_wrist", "right_index"],
        ],
    ),
    (
        "Wrist_L",
        &[
            ["LElbow", "LWrist", "LThumb"],
            ["LElbow", "LWrist", "LIndex"],
            ["left_elbow", "left_wrist", "left_thumb"],
            ["left_elbow", "left_wrist", "left_index"],
        ],
    ),
    (
        "Neck",
        &[
            ["RShoulder", "Neck", "Head"],
            ["LShoulder", "Neck", "Head"],
            ["RShoulder", "Neck", "Nose"],
            ["LShoulder", "Neck", "Nose"],
            ["right_shoulder", "nose", "right_eye"],
            ["left_shoulder", "nose", "left_eye"],
        ],
    ),
];

fn index(markers: &[String], name: &str) -> Option<usize> {
    markers.iter().position(|m| m == name)
}

/// Standard body segments present in `markers`, in pattern order.
pub fn auto_segments(markers: &[String]) -> Vec<SegmentDef> {
    SEGMENT_PATTERNS
        .iter()
        .filter_map(|(name, pats)| {
            pats.iter().find_map(|[a, b]| Some((index(markers, a)?, index(markers, b)?))).map(|(a, b)| SegmentDef {
                name: (*name).to_string(),
                a,
                b,
            })
        })
        .collect()
}

/// Standard joints (three markers, middle = vertex) present in `markers`.
pub fn auto_joints(markers: &[String]) -> Vec<JointDef> {
    JOINT_PATTERNS
        .iter()
        .filter_map(|(name, pats)| {
            pats.iter()
                .find_map(|[a, v, c]| Some((index(markers, a)?, index(markers, v)?, index(markers, c)?)))
                .map(|(a, vertex, c)| JointDef { name: (*name).to_string(), a, vertex, c })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn first_matching_pattern_wins() {
        let m = names(&["Hip", "Neck", "RHip", "RKnee", "RAnkle", "RHeel", "LShoulder", "LElbow", "LWrist"]);
        let segs = auto_segments(&m);
        let trunk = segs.iter().find(|s| s.name == "Trunk").unwrap();
        assert_eq!((trunk.a, trunk.b), (1, 0)); // Neck-Hip, not Neck-RHip
        assert!(segs.iter().any(|s| s.name == "Foot_R" && s.b == 5)); // RAnkle-RHeel (BigToe/SmallToe absent)
        let joints = auto_joints(&m);
        assert!(joints.iter().any(|j| j.name == "Knee_R" && j.vertex == 3));
        assert!(joints.iter().any(|j| j.name == "Elbow_L"));
        assert!(!joints.iter().any(|j| j.name == "Knee_L"));
    }

    #[test]
    fn snake_case_names_are_supported() {
        let m = names(&["right_hip", "right_knee", "right_ankle"]);
        assert_eq!(auto_segments(&m).iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["Thigh_R", "Shank_R"]);
        assert_eq!(auto_joints(&m)[0].name, "Knee_R");
    }
}
