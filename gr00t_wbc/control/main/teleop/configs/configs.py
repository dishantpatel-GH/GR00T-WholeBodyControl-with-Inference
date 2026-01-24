"""
Configuration classes for G1 robot control loop.

This module provides configuration dataclasses for running the G1 control loop
on real robot hardware.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Optional

import yaml

import gr00t_wbc
from gr00t_wbc.control.main.config_template import ArgsConfig as ArgsConfigTemplate
from gr00t_wbc.control.policy.wbc_policy_factory import WBC_VERSIONS
from gr00t_wbc.control.utils.network_utils import resolve_interface


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values.

    Args:
        wbc_config: The loaded WBC YAML configuration dictionary
        config: The BaseConfig dataclass instance with override values
        missed_keys_only: If True, only add keys that don't exist in wbc_config.
                          If False, validate all keys exist and override all.

    Returns:
        Updated wbc_config dictionary with overridden values
    """
    # Override yaml values with dataclass values
    key_to_value = {
        "INTERFACE": config.interface,
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "model_path": config.wbc_model_path,
        "enable_waist": config.enable_waist,
        "with_hands": config.with_hands,
        "HAND_TYPE": config.hand_type,
        "verbose": config.verbose,
        "verbose_timing": config.verbose_timing,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
        "keyboard_dispatcher_type": config.keyboard_dispatcher_type,
        "enable_gravity_compensation": config.enable_gravity_compensation,
        "gravity_compensation_joints": config.gravity_compensation_joints,
        "high_elbow_pose": config.high_elbow_pose,
    }

    if missed_keys_only:
        # Only add keys that don't exist in wbc_config
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        # Set all keys (overwrite existing)
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    # g1 kp, kd, sim2real gap - apply real robot tuning
    if config.env_type == "real":
        # update waist pitch damping, index 14
        wbc_config["MOTOR_KD"][14] = wbc_config["MOTOR_KD"][14] - 10

    return wbc_config


@dataclass
class BaseConfig(ArgsConfigTemplate):
    """Base config inherited by all G1 control loops."""

    # WBC Configuration
    wbc_version: Literal[tuple(WBC_VERSIONS)] = "gear_wbc"
    """Version of the whole body controller."""

    wbc_model_path: str = (
        "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"
    )
    """Path to WBC model file."""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration
    interface: str = "real"
    """Interface to use for the control loop (network interface name)."""

    control_frequency: int = 50
    """Frequency of the control loop."""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    with_hands: bool = True
    """Enable hand functionality. When False, robot operates without hands."""

    hand_type: Literal["dex3", "inspire"] = "dex3"
    """Type of hand to use. Options: 'dex3' (default), 'inspire'."""

    high_elbow_pose: bool = False
    """Enable high elbow pose configuration for default joint positions."""

    verbose: bool = True
    """Whether to print verbose output."""

    upper_body_joint_speed: float = 1000
    """Upper body joint speed."""

    env_name: str = "default"
    """Environment name."""

    verbose_timing: bool = False
    """Enable verbose timing output every iteration."""

    keyboard_dispatcher_type: str = "raw"
    """Keyboard dispatcher to use. [raw, ros]"""

    # Gravity Compensation Configuration
    enable_gravity_compensation: bool = False
    """Enable gravity compensation using pinocchio dynamics."""

    gravity_compensation_joints: Optional[list[str]] = None
    """Joint groups to apply gravity compensation to (e.g., ['arms', 'left_arm', 'right_arm'])."""

    # Deployment/Camera Configuration
    robot_ip: str = "192.168.123.164"
    """Robot IP address"""

    def __post_init__(self):
        # Resolve interface (handles sim/real shortcuts, platform differences, and error handling)
        self.interface, self.env_type = resolve_interface(self.interface)

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides."""
        # Get the base path to gr00t_wbc and convert to Path object
        package_path = Path(os.path.dirname(gr00t_wbc.__file__))

        if self.wbc_version == "gear_wbc":
            config_path = str(package_path / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: gear_wbc"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        # Override yaml values with dataclass values
        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config


@dataclass
class ControlLoopConfig(BaseConfig):
    """Config for running the G1 control loop on real robot."""
    pass


@dataclass
class ComposedCameraClientConfig:
    """Config for running the composed camera client."""
    camera_port: int = 5555
    camera_host: str = "localhost"
    fps: float = 20.0


@dataclass
class DeploymentConfig(BaseConfig, ComposedCameraClientConfig):
    """G1 Robot Deployment Configuration."""
    camera_publish_rate: float = 30.0
    view_camera: bool = True
    enable_webcam_recording: bool = True
    webcam_output_dir: str = "logs_experiment"
    skip_img_transform: bool = False
    image_publish: bool = False
    add_stereo_camera: bool = False
    """Enable stereo camera features (ego_view_left_mono, ego_view_right_mono)"""
