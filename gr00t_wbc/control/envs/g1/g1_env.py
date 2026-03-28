"""
G1 Robot Environment for Real Robot Control.

This module provides the G1Env class for interfacing with the Unitree G1 robot.
"""

from copy import deepcopy
from typing import Dict

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from gr00t_wbc.control.base.humanoid_env import Hands, HumanoidEnv
from gr00t_wbc.control.envs.g1.g1_body import G1Body
from gr00t_wbc.control.envs.g1.g1_hand import G1InspireFTPHand, G1InspireHand, G1ThreeFingerHand
from gr00t_wbc.control.envs.g1.sim.simulator_factory import init_channel
from gr00t_wbc.control.envs.g1.utils.joint_safety import JointSafetyMonitor
from gr00t_wbc.control.robot_model.robot_model import RobotModel
from gr00t_wbc.control.utils.ros_utils import ROSManager


class G1Env(HumanoidEnv):
    """Environment for controlling the Unitree G1 robot."""

    def __init__(
        self,
        env_name: str = "default",
        robot_model: RobotModel = None,
        wbc_version: str = "v2",
        config: Dict[str, any] = None,
        **kwargs,
    ):
        super().__init__()
        self.robot_model = deepcopy(robot_model)  # need to cache FK results
        self.config = config

        # Initialize safety monitor (visualization disabled)
        self.safety_monitor = JointSafetyMonitor(
            robot_model, enable_viz=False, env_type="real"
        )
        self.last_obs = None
        self.last_safety_ok = True  # Track last safety status from queue_action

        # Initialize Unitree SDK communication channel
        init_channel(config=self.config)

        # Initialize body and hands
        self._body = G1Body(config=self.config)

        self.with_hands = config.get("with_hands", False)
        # Check config for hand type (Default to 'dex3' if not specified)
        self.hand_type = config.get("HAND_TYPE", "dex3")

        # Gravity compensation settings
        self.enable_gravity_compensation = config.get("enable_gravity_compensation", False)
        self.gravity_compensation_joints = config.get("gravity_compensation_joints", ["arms"])

        if self.enable_gravity_compensation:
            print(
                f"Gravity compensation enabled for joint groups: {self.gravity_compensation_joints}"
            )

        if self.with_hands:
            self._hands = Hands()
            if self.hand_type == "inspire_ftp":
                print("[G1Env] Initializing Inspire FTP Hands...")
                self._hands.left = G1InspireFTPHand(is_left=True)
                self._hands.right = G1InspireFTPHand(is_left=False)
            elif self.hand_type == "inspire":
                print("[G1Env] Initializing Inspire Hands...")
                self._hands.left = G1InspireHand(is_left=True)
                self._hands.right = G1InspireHand(is_left=False)
            else:
                print("[G1Env] Initializing Standard Dex3 Hands...")
                self._hands.left = G1ThreeFingerHand(is_left=True)
                self._hands.right = G1ThreeFingerHand(is_left=False)

        # Calibrate hands for real robot
        self.calibrate_hands()

        # Initialize ROS 2 node
        self.ros_manager = ROSManager(node_name="g1_env")
        self.ros_node = self.ros_manager.node

        self.delay_list = []
        self.visualize_delay = False
        self.print_delay_interval = 100
        self.cnt = 0

    def body(self) -> G1Body:
        return self._body

    def hands(self) -> Hands:
        if not self.with_hands:
            raise RuntimeError(
                "Hands not initialized. Use --with_hands True to enable hand functionality."
            )
        return self._hands

    def observe(self) -> Dict[str, any]:
        """Get observations from body and hands."""
        body_obs = self.body().observe()

        body_q = body_obs["body_q"]
        body_dq = body_obs["body_dq"]
        body_ddq = body_obs["body_ddq"]
        body_tau_est = body_obs["body_tau_est"]

        if self.with_hands:
            left_hand_obs = self.hands().left.observe()
            right_hand_obs = self.hands().right.observe()

            # These will now come from the appropriate hand class (Dex3 or Inspire)
            left_hand_q = left_hand_obs["hand_q"]
            right_hand_q = right_hand_obs["hand_q"]
            left_hand_dq = left_hand_obs["hand_dq"]
            right_hand_dq = right_hand_obs["hand_dq"]
            left_hand_ddq = left_hand_obs["hand_ddq"]
            right_hand_ddq = right_hand_obs["hand_ddq"]
            left_hand_tau_est = left_hand_obs["hand_tau_est"]
            right_hand_tau_est = right_hand_obs["hand_tau_est"]

            # Body and hand joint measurements come in actuator order, convert to joint order
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
                left_hand_actuated_joint_values=left_hand_q,
                right_hand_actuated_joint_values=right_hand_q,
            )
            whole_dq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_dq,
                left_hand_actuated_joint_values=left_hand_dq,
                right_hand_actuated_joint_values=right_hand_dq,
            )
            whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_ddq,
                left_hand_actuated_joint_values=left_hand_ddq,
                right_hand_actuated_joint_values=right_hand_ddq,
            )
            whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_tau_est,
                left_hand_actuated_joint_values=left_hand_tau_est,
                right_hand_actuated_joint_values=right_hand_tau_est,
            )
        else:
            # Body only - convert from actuator order to joint order
            whole_q = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_q,
            )
            whole_dq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_dq,
            )
            whole_ddq = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_ddq,
            )
            whole_tau_est = self.robot_model.get_configuration_from_actuated_joints(
                body_actuated_joint_values=body_tau_est,
            )

        eef_obs = self.get_eef_obs(whole_q)

        obs = {
            "q": whole_q,
            "dq": whole_dq,
            "ddq": whole_ddq,
            "tau_est": whole_tau_est,
            "floating_base_pose": body_obs["floating_base_pose"],
            "floating_base_vel": body_obs["floating_base_vel"],
            "floating_base_acc": body_obs["floating_base_acc"],
            "wrist_pose": np.concatenate([eef_obs["left_wrist_pose"], eef_obs["right_wrist_pose"]]),
            "torso_quat": body_obs["torso_quat"],
            "torso_ang_vel": body_obs["torso_ang_vel"],
        }

        # Store last observation for safety checking
        self.last_obs = obs

        return obs

    @property
    def observation_space(self) -> gym.Space:
        q_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        dq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        ddq_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        tau_est_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))
        floating_base_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        floating_base_vel_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        floating_base_acc_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        wrist_pose_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7 + 7,))
        return gym.spaces.Dict(
            {
                "floating_base_pose": floating_base_pose_space,
                "floating_base_vel": floating_base_vel_space,
                "floating_base_acc": floating_base_acc_space,
                "q": q_space,
                "dq": dq_space,
                "ddq": ddq_space,
                "tau_est": tau_est_space,
                "wrist_pose": wrist_pose_space,
            }
        )

    def queue_action(self, action: Dict[str, any]):
        """Queue action to be sent to the robot."""
        # Safety check
        if self.last_obs is not None:
            safety_result = self.safety_monitor.handle_violations(self.last_obs, action)
            action = safety_result["action"]

        # Map action from joint order to actuator order
        body_actuator_q = self.robot_model.get_body_actuated_joints(action["q"])

        self.body().queue_action(
            {
                "body_q": body_actuator_q,
                "body_dq": np.zeros_like(body_actuator_q),
                "body_tau": np.zeros_like(body_actuator_q),
            }
        )

        if self.with_hands:
            left_hand_actuator_q = self.robot_model.get_hand_actuated_joints(
                action["q"], side="left"
            )
            right_hand_actuator_q = self.robot_model.get_hand_actuated_joints(
                action["q"], side="right"
            )
            self.hands().left.queue_action({"hand_q": left_hand_actuator_q})
            self.hands().right.queue_action({"hand_q": right_hand_actuator_q})

    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.robot_model.num_dofs,))

    def calibrate_hands(self):
        """Calibrate the hand joint qpos for real robot."""
        if self.with_hands:
            print("calibrating left hand")
            self.hands().left.calibrate_hand()
            print("calibrating right hand")
            self.hands().right.calibrate_hand()
        else:
            print("Skipping hand calibration - hands disabled")

    def reset(self):
        """Reset the environment (no-op for real robot)."""
        pass

    def close(self):
        """Close the environment (no-op for real robot)."""
        pass

    def get_eef_obs(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Get end-effector observations using forward kinematics."""
        self.robot_model.cache_forward_kinematics(q)
        eef_obs = {}
        for side in ["left", "right"]:
            wrist_placement = self.robot_model.frame_placement(
                self.robot_model.supplemental_info.hand_frame_names[side]
            )
            wrist_pos, wrist_quat = wrist_placement.translation[:3], R.from_matrix(
                wrist_placement.rotation
            ).as_quat(scalar_first=True)
            eef_obs[f"{side}_wrist_pose"] = np.concatenate([wrist_pos, wrist_quat])

        return eef_obs

    def get_joint_safety_status(self) -> bool:
        """Get current joint safety status from the last queue_action safety check.

        Returns:
            bool: True if joints are safe (no shutdown required), False if unsafe
        """
        return self.last_safety_ok

    def handle_keyboard_button(self, key):
        """Handle keyboard button press (no-op for real robot)."""
        pass
