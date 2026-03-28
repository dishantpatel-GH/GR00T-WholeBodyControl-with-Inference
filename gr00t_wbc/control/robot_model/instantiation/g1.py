import os
from pathlib import Path
from typing import Literal, Optional

from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_supplemental_info import (
    ElbowPose,
    G1SupplementalInfo,
    WaistLocation,
)
from gr00t_wbc.control.robot_model.supplemental_info.g1.g1_inspire_ftp_supplemental_info import (
    G1InspireFTPSupplementalInfo,
)


def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body",
    high_elbow_pose: bool = False,
    hand_type: Optional[str] = None,
):
    """
    Instantiate a G1 robot model with configurable waist location and pose.

    Args:
        waist_location: Whether to put waist in "lower_body" (default G1 behavior),
                        "upper_body" (waist controlled with arms/manipulation via IK),
                        or "lower_and_upper_body" (waist reference from arms/manipulation
                        via IK then passed to lower body policy)
        high_elbow_pose: Whether to use high elbow pose configuration for default joint positions
        hand_type: Type of hand to use. None or "dex3" for default 7-DOF Dex3 hands,
                   "inspire_ftp" for 6-DOF Inspire FTP hands.

    Returns:
        RobotModel: Configured G1 robot model
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    model_data_dir = os.path.join(project_root, "gr00t_wbc/control/robot_model/model_data/g1")

    # Select URDF based on hand type
    if hand_type == "inspire_ftp":
        urdf_filename = "g1_29dof_with_inspire_ftp_hand.urdf"
    else:
        urdf_filename = "g1_29dof_with_hand.urdf"

    robot_model_config = {
        "asset_path": model_data_dir,
        "urdf_path": os.path.join(model_data_dir, urdf_filename),
    }

    assert waist_location in [
        "lower_body",
        "upper_body",
        "lower_and_upper_body",
    ], f"Invalid waist_location: {waist_location}. Must be 'lower_body' or 'upper_body' or 'lower_and_upper_body'"

    # Map string values to enums
    waist_location_enum = {
        "lower_body": WaistLocation.LOWER_BODY,
        "upper_body": WaistLocation.UPPER_BODY,
        "lower_and_upper_body": WaistLocation.LOWER_AND_UPPER_BODY,
    }[waist_location]

    elbow_pose_enum = ElbowPose.HIGH if high_elbow_pose else ElbowPose.LOW

    # Create supplemental info based on hand type
    if hand_type == "inspire_ftp":
        robot_model_supplemental_info = G1InspireFTPSupplementalInfo(
            waist_location=waist_location_enum, elbow_pose=elbow_pose_enum
        )
    else:
        robot_model_supplemental_info = G1SupplementalInfo(
            waist_location=waist_location_enum, elbow_pose=elbow_pose_enum
        )

    robot_model = RobotModel(
        robot_model_config["urdf_path"],
        robot_model_config["asset_path"],
        supplemental_info=robot_model_supplemental_info,
    )
    return robot_model
