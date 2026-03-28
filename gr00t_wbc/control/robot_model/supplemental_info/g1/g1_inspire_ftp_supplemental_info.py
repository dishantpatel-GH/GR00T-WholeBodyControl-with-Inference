from dataclasses import dataclass

import numpy as np

from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    WaistLocation,
)
from gr00t_wbc.control.robot_model.supplemental_info.robot_supplemental_info import (
    RobotSupplementalInfo,
)


@dataclass
class G1InspireFTPSupplementalInfo(RobotSupplementalInfo):
    """
    Supplemental information for the G1 robot with Inspire FTP hands (6 DOF per hand).

    Args:
        waist_location: Where to place waist joints in the joint groups
        elbow_pose: Which elbow pose configuration to use for default joint positions
    """

    def __init__(
        self,
        waist_location: WaistLocation = WaistLocation.LOWER_BODY,
        elbow_pose: ElbowPose = ElbowPose.LOW,
    ):
        name = "G1_InspireFTP"

        # Define all actuated joints
        body_actuated_joints = [
            # Left leg
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            # Right leg
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            # Waist
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            # Left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        left_hand_actuated_joints = [
            "left_hand_pinky_joint",
            "left_hand_ring_joint",
            "left_hand_middle_joint",
            "left_hand_index_joint",
            "left_hand_thumb_bend_joint",
            "left_hand_thumb_rotation_joint",
        ]

        right_hand_actuated_joints = [
            "right_hand_pinky_joint",
            "right_hand_ring_joint",
            "right_hand_middle_joint",
            "right_hand_index_joint",
            "right_hand_thumb_bend_joint",
            "right_hand_thumb_rotation_joint",
        ]

        # Define joint limits from URDF
        joint_limits = {
            # Left leg
            "left_hip_pitch_joint": [-2.5307, 2.8798],
            "left_hip_roll_joint": [-0.5236, 2.9671],
            "left_hip_yaw_joint": [-2.7576, 2.7576],
            "left_knee_joint": [-0.087267, 2.8798],
            "left_ankle_pitch_joint": [-0.87267, 0.5236],
            "left_ankle_roll_joint": [-0.2618, 0.2618],
            # Right leg
            "right_hip_pitch_joint": [-2.5307, 2.8798],
            "right_hip_roll_joint": [-2.9671, 0.5236],
            "right_hip_yaw_joint": [-2.7576, 2.7576],
            "right_knee_joint": [-0.087267, 2.8798],
            "right_ankle_pitch_joint": [-0.87267, 0.5236],
            "right_ankle_roll_joint": [-0.2618, 0.2618],
            # Waist
            "waist_yaw_joint": [-2.618, 2.618],
            "waist_roll_joint": [-0.52, 0.52],
            "waist_pitch_joint": [-0.52, 0.52],
            # Left arm
            "left_shoulder_pitch_joint": [-3.0892, 2.6704],
            "left_shoulder_roll_joint": [0.19, 2.2515],
            "left_shoulder_yaw_joint": [-2.618, 2.618],
            "left_elbow_joint": [-1.0472, 2.0944],
            "left_wrist_roll_joint": [-1.972222054, 1.972222054],
            "left_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "left_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Right arm
            "right_shoulder_pitch_joint": [-3.0892, 2.6704],
            "right_shoulder_roll_joint": [-2.2515, -0.19],
            "right_shoulder_yaw_joint": [-2.618, 2.618],
            "right_elbow_joint": [-1.0472, 2.0944],
            "right_wrist_roll_joint": [-1.972222054, 1.972222054],
            "right_wrist_pitch_joint": [-1.614429558, 1.614429558],
            "right_wrist_yaw_joint": [-1.614429558, 1.614429558],
            # Left hand (normalized range matching dataset values)
            "left_hand_pinky_joint": [0.0, 1.0],
            "left_hand_ring_joint": [0.0, 1.0],
            "left_hand_middle_joint": [0.0, 1.0],
            "left_hand_index_joint": [0.0, 1.0],
            "left_hand_thumb_bend_joint": [0.0, 1.0],
            "left_hand_thumb_rotation_joint": [0.0, 1.0],
            # Right hand (normalized range matching dataset values)
            "right_hand_pinky_joint": [0.0, 1.0],
            "right_hand_ring_joint": [0.0, 1.0],
            "right_hand_middle_joint": [0.0, 1.0],
            "right_hand_index_joint": [0.0, 1.0],
            "right_hand_thumb_bend_joint": [0.0, 1.0],
            "right_hand_thumb_rotation_joint": [0.0, 1.0],
        }

        # Define joint groups
        joint_groups = {
            # Body groups
            "waist": {
                "joints": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
                "groups": [],
            },
            # Leg groups
            "left_leg": {
                "joints": [
                    "left_hip_pitch_joint",
                    "left_hip_roll_joint",
                    "left_hip_yaw_joint",
                    "left_knee_joint",
                    "left_ankle_pitch_joint",
                    "left_ankle_roll_joint",
                ],
                "groups": [],
            },
            "right_leg": {
                "joints": [
                    "right_hip_pitch_joint",
                    "right_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_knee_joint",
                    "right_ankle_pitch_joint",
                    "right_ankle_roll_joint",
                ],
                "groups": [],
            },
            "legs": {"joints": [], "groups": ["left_leg", "right_leg"]},
            # Arm groups
            "left_arm": {
                "joints": [
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "right_arm": {
                "joints": [
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "groups": [],
            },
            "arms": {"joints": [], "groups": ["left_arm", "right_arm"]},
            # Hand groups (6 DOF Inspire FTP hands)
            "left_hand": {
                "joints": [
                    "left_hand_pinky_joint",
                    "left_hand_ring_joint",
                    "left_hand_middle_joint",
                    "left_hand_index_joint",
                    "left_hand_thumb_bend_joint",
                    "left_hand_thumb_rotation_joint",
                ],
                "groups": [],
            },
            "right_hand": {
                "joints": [
                    "right_hand_pinky_joint",
                    "right_hand_ring_joint",
                    "right_hand_middle_joint",
                    "right_hand_index_joint",
                    "right_hand_thumb_bend_joint",
                    "right_hand_thumb_rotation_joint",
                ],
                "groups": [],
            },
            "hands": {"joints": [], "groups": ["left_hand", "right_hand"]},
            # Full body groups
            "lower_body": {"joints": [], "groups": ["waist", "legs"]},
            "upper_body_no_hands": {"joints": [], "groups": ["arms"]},
            "body": {"joints": [], "groups": ["lower_body", "upper_body_no_hands"]},
            "upper_body": {"joints": [], "groups": ["upper_body_no_hands", "hands"]},
        }

        # Define joint name mapping from generic types to robot-specific names
        joint_name_mapping = {
            # Waist joints
            "waist_pitch": "waist_pitch_joint",
            "waist_roll": "waist_roll_joint",
            "waist_yaw": "waist_yaw_joint",
            # Shoulder joints
            "shoulder_pitch": {
                "left": "left_shoulder_pitch_joint",
                "right": "right_shoulder_pitch_joint",
            },
            "shoulder_roll": {
                "left": "left_shoulder_roll_joint",
                "right": "right_shoulder_roll_joint",
            },
            "shoulder_yaw": {
                "left": "left_shoulder_yaw_joint",
                "right": "right_shoulder_yaw_joint",
            },
            # Elbow joints
            "elbow_pitch": {"left": "left_elbow_joint", "right": "right_elbow_joint"},
            # Wrist joints
            "wrist_pitch": {"left": "left_wrist_pitch_joint", "right": "right_wrist_pitch_joint"},
            "wrist_roll": {"left": "left_wrist_roll_joint", "right": "right_wrist_roll_joint"},
            "wrist_yaw": {"left": "left_wrist_yaw_joint", "right": "right_wrist_yaw_joint"},
        }

        root_frame_name = "pelvis"

        hand_frame_names = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

        elbow_calibration_joint_angles = {"left": 0.0, "right": 0.0}

        hand_rotation_correction = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # Configure default joint positions based on elbow pose
        if elbow_pose == ElbowPose.HIGH:
            default_joint_q = {
                "shoulder_roll": {"left": 0.5, "right": -0.5},
                "shoulder_pitch": {"left": -0.2, "right": -0.2},
                "shoulder_yaw": {"left": -0.5, "right": 0.5},
                "wrist_roll": {"left": -0.5, "right": 0.5},
                "wrist_yaw": {"left": 0.5, "right": -0.5},
                "wrist_pitch": {"left": -0.2, "right": -0.2},
            }
        else:  # ElbowPose.LOW
            default_joint_q = {
                "shoulder_roll": {"left": 0.2, "right": -0.2},
            }

        teleop_upper_body_motion_scale = 1.0

        # Configure joint groups based on waist location
        modified_joint_groups = joint_groups.copy()
        if waist_location == WaistLocation.UPPER_BODY:
            # Move waist from lower_body to upper_body_no_hands
            modified_joint_groups["lower_body"] = {"joints": [], "groups": ["legs"]}
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }
        elif waist_location == WaistLocation.LOWER_AND_UPPER_BODY:
            # Add waist to upper_body_no_hands while keeping it in lower_body
            modified_joint_groups["upper_body_no_hands"] = {
                "joints": [],
                "groups": ["arms", "waist"],
            }
        # For LOWER_BODY, keep default joint_groups as is

        super().__init__(
            name=name,
            body_actuated_joints=body_actuated_joints,
            left_hand_actuated_joints=left_hand_actuated_joints,
            right_hand_actuated_joints=right_hand_actuated_joints,
            joint_limits=joint_limits,
            joint_groups=modified_joint_groups,
            root_frame_name=root_frame_name,
            hand_frame_names=hand_frame_names,
            elbow_calibration_joint_angles=elbow_calibration_joint_angles,
            joint_name_mapping=joint_name_mapping,
            hand_rotation_correction=hand_rotation_correction,
            default_joint_q=default_joint_q,
            teleop_upper_body_motion_scale=teleop_upper_body_motion_scale,
        )
