"""
Skeleton Configuration for MStudio Report Generator

This module contains the standard biomechanical segment and joint patterns
used for analysis in the report generator.
"""

## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = ""
__credits__ = [""]
__license__ = ""
__maintainer__ = "HunMin Kim"
__email__ = "hunminkim98@gmail.com"
__status__ = "Development"


# Define standard segment patterns for biomechanical analysis
SEGMENT_PATTERNS = {
    'Trunk': [
        ['Neck', 'Hip'], ['Neck', 'CHip'], ['Neck', 'RHip'], ['Neck', 'LHip'],
        ['C7', 'Hip'], ['C7', 'CHip'], ['C7', 'RHip'], ['C7', 'LHip'],
        # For BLAZEPOSE (no central hip, use nose as reference)
        ['nose', 'right_hip'], ['nose', 'left_hip']
    ],
    'Head': [
        ['Neck', 'Head'], ['Neck', 'Nose'], ['C7', 'Head'], ['C7', 'Nose'],
        # BLAZEPOSE patterns (no explicit head, use nose-eye)
        ['nose', 'right_eye'], ['nose', 'left_eye']
    ],
    'Upper_Arm_R': [
        ['RShoulder', 'RElbow'], ['right_shoulder', 'right_elbow']
    ],
    'Upper_Arm_L': [
        ['LShoulder', 'LElbow'], ['left_shoulder', 'left_elbow']
    ],
    'Forearm_R': [
        ['RElbow', 'RWrist'], ['right_elbow', 'right_wrist']
    ],
    'Forearm_L': [
        ['LElbow', 'LWrist'], ['left_elbow', 'left_wrist']
    ],
    'Thigh_R': [
        ['RHip', 'RKnee'], ['right_hip', 'right_knee']
    ],
    'Thigh_L': [
        ['LHip', 'LKnee'], ['left_hip', 'left_knee']
    ],
    'Shank_R': [
        ['RKnee', 'RAnkle'], ['right_knee', 'right_ankle']
    ],
    'Shank_L': [
        ['LKnee', 'LAnkle'], ['left_knee', 'left_ankle']
    ],
    'Foot_R': [
        ['RAnkle', 'RBigToe'], ['RAnkle', 'RSmallToe'], ['RAnkle', 'RHeel'],
        ['right_ankle', 'right_foot_index'], ['right_ankle', 'right_heel']
    ],
    'Foot_L': [
        ['LAnkle', 'LBigToe'], ['LAnkle', 'LSmallToe'], ['LAnkle', 'LHeel'],
        ['left_ankle', 'left_foot_index'], ['left_ankle', 'left_heel']
    ]
}


# Define standard joint patterns for biomechanical analysis (3-point angles)
JOINT_PATTERNS = {
    'Hip_R': [
        ['Neck', 'RHip', 'RKnee'], ['C7', 'RHip', 'RKnee'],
        ['RShoulder', 'RHip', 'RKnee'], ['LShoulder', 'RHip', 'RKnee'],
        # BLAZEPOSE patterns
        ['nose', 'right_hip', 'right_knee'], ['right_shoulder', 'right_hip', 'right_knee']
    ],
    'Hip_L': [
        ['Neck', 'LHip', 'LKnee'], ['C7', 'LHip', 'LKnee'],
        ['RShoulder', 'LHip', 'LKnee'], ['LShoulder', 'LHip', 'LKnee'],
        # BLAZEPOSE patterns
        ['nose', 'left_hip', 'left_knee'], ['left_shoulder', 'left_hip', 'left_knee']
    ],
    'Knee_R': [
        ['RHip', 'RKnee', 'RAnkle'], ['right_hip', 'right_knee', 'right_ankle']
    ],
    'Knee_L': [
        ['LHip', 'LKnee', 'LAnkle'], ['left_hip', 'left_knee', 'left_ankle']
    ],
    'Ankle_R': [
        ['RKnee', 'RAnkle', 'RBigToe'], ['RKnee', 'RAnkle', 'RHeel'],
        ['right_knee', 'right_ankle', 'right_foot_index'], ['right_knee', 'right_ankle', 'right_heel']
    ],
    'Ankle_L': [
        ['LKnee', 'LAnkle', 'LBigToe'], ['LKnee', 'LAnkle', 'LHeel'],
        ['left_knee', 'left_ankle', 'left_foot_index'], ['left_knee', 'left_ankle', 'left_heel']
    ],
    'Shoulder_R': [
        ['Neck', 'RShoulder', 'RElbow'], ['C7', 'RShoulder', 'RElbow'],
        # BLAZEPOSE patterns
        ['nose', 'right_shoulder', 'right_elbow']
    ],
    'Shoulder_L': [
        ['Neck', 'LShoulder', 'LElbow'], ['C7', 'LShoulder', 'LElbow'],
        # BLAZEPOSE patterns
        ['nose', 'left_shoulder', 'left_elbow']
    ],
    'Elbow_R': [
        ['RShoulder', 'RElbow', 'RWrist'], ['right_shoulder', 'right_elbow', 'right_wrist']
    ],
    'Elbow_L': [
        ['LShoulder', 'LElbow', 'LWrist'], ['left_shoulder', 'left_elbow', 'left_wrist']
    ],
    'Wrist_R': [
        ['RElbow', 'RWrist', 'RThumb'], ['RElbow', 'RWrist', 'RIndex'],
        ['right_elbow', 'right_wrist', 'right_thumb'], ['right_elbow', 'right_wrist', 'right_index']
    ],
    'Wrist_L': [
        ['LElbow', 'LWrist', 'LThumb'], ['LElbow', 'LWrist', 'LIndex'],
        ['left_elbow', 'left_wrist', 'left_thumb'], ['left_elbow', 'left_wrist', 'left_index']
    ],
    'Neck': [
        ['RShoulder', 'Neck', 'Head'], ['LShoulder', 'Neck', 'Head'],
        ['RShoulder', 'Neck', 'Nose'], ['LShoulder', 'Neck', 'Nose'],
        # BLAZEPOSE patterns (no explicit neck, use shoulder-nose-eye)
        ['right_shoulder', 'nose', 'right_eye'], ['left_shoulder', 'nose', 'left_eye']
    ]
}


def get_segment_patterns():
    """
    Get the standard segment patterns for biomechanical analysis.
    
    Returns:
        dict: Dictionary of segment patterns where keys are segment names
              and values are lists of possible marker combinations.
    """
    return SEGMENT_PATTERNS.copy()


def get_joint_patterns():
    """
    Get the standard joint patterns for biomechanical analysis.
    
    Returns:
        dict: Dictionary of joint patterns where keys are joint names
              and values are lists of possible 3-point marker combinations.
    """
    return JOINT_PATTERNS.copy()


def add_custom_segment_pattern(segment_name: str, marker_combinations: list):
    """
    Add a custom segment pattern to the existing patterns.
    
    Args:
        segment_name (str): Name of the segment
        marker_combinations (list): List of marker combinations for this segment
    """
    SEGMENT_PATTERNS[segment_name] = marker_combinations


def add_custom_joint_pattern(joint_name: str, marker_combinations: list):
    """
    Add a custom joint pattern to the existing patterns.
    
    Args:
        joint_name (str): Name of the joint
        marker_combinations (list): List of 3-point marker combinations for this joint
    """
    JOINT_PATTERNS[joint_name] = marker_combinations
