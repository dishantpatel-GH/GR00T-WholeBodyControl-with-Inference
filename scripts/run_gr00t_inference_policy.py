"""
GR00T Inference Policy Loop for Real Robot Evaluation

This script replaces the teleop policy loop with a GR00T model inference policy.
It connects to the GR00T server to get action predictions and sends them to the
control loop via ROS.

Usage:
    python scripts/run_gr00t_inference_policy.py \
        --model_host <server_ip> \
        --model_port 5555 \
        --camera_host <camera_ip> \
        --camera_port 5555 \
        --task_description "pick and place the box on the platform"
"""

from collections import deque
from dataclasses import dataclass
import time
from typing import Literal, Optional

import numpy as np
import tyro

# from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import PolicyClient
from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    ROBOT_CONFIG_TOPIC,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
from gr00t_wbc.control.utils.telemetry import Telemetry


import cv2
import pickle
import zmq


class ZMQCameraSensor:
    """ZMQ camera client compatible with the xr_teleoperate ImageServer.

    The server publishes pickle-serialized dicts containing:
      - 'image': JPEG-encoded bytes (concatenated head+wrist frames)
      - 'depth_raw': optional uint16 numpy array
    """

    def __init__(self, server_ip: str = "localhost", port: int = 5556):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, True)  # keep only latest
        self._socket.setsockopt(zmq.RCVHWM, 3)
        self._socket.connect(f"tcp://{server_ip}:{port}")
        print(f"[ZMQCameraSensor] Connected to tcp://{server_ip}:{port}")

    def read(self):
        """Read latest frame. Returns dict with 'images' key or None."""
        if self._socket.poll(timeout=100) == 0:
            return None
        message = self._socket.recv()

        # Decode: raw JPEG or pickle dict
        is_jpg = len(message) > 2 and message[:2] == b"\xff\xd8"
        if is_jpg:
            jpg_bytes = message
        else:
            try:
                data = pickle.loads(message)
                if isinstance(data, dict):
                    jpg_bytes = data.get("image") or data.get("rgb")
                    if jpg_bytes is None:
                        return None
                else:
                    jpg_bytes = message
            except Exception:
                jpg_bytes = message

        np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if image is None:
            return None

        return {"images": {"ego_view": image}, "timestamps": {"ego_view": time.time()}}

    def close(self):
        self._socket.close()
        self._context.term()


INFERENCE_NODE_NAME = "Gr00tInferencePolicy"


@dataclass
class Gr00tInferenceConfig:
    """Configuration for GR00T inference policy loop."""
    
    # Model server configuration
    model_host: str = "localhost"
    """Host address for the GR00T model server"""
    
    model_port: int = 5555
    """Port number for the GR00T model server"""
    
    # Camera configuration
    camera_host: str = "localhost"
    """Host address for the camera server"""
    
    camera_port: int = 5556
    """Port number for the camera server"""
    
    # Task configuration
    task_description: str = "pick and place the box on the platform"
    """Task description/instruction for the model"""
    
    # Robot configuration
    enable_waist: bool = False
    """Whether waist is controlled by upper body IK"""
    
    high_elbow_pose: bool = False
    """Whether to use high elbow pose configuration"""
    
    with_hands: bool = False
    """Whether robot has hands"""

    hand_type: Literal["dex3", "inspire", "inspire_ftp"] = "inspire_ftp"
    """Type of hand to use. Options: 'dex3', 'inspire', 'inspire_ftp'."""

    # Inference configuration
    inference_frequency: float = 50.0
    """Frequency of inference loop (Hz). Higher = smoother motion."""
    
    n_action_steps: int = 32
    """Number of action steps to execute before re-querying the model.
    Higher values reduce stuttering but may reduce responsiveness."""
    
    action_horizon: int = 64
    """Total action horizon from model. Should match server's --execution-horizon."""
    
    # Connection configuration  
    connection_timeout_ms: int = 15000
    """Timeout for model server connection (ms)"""
    
    strict_validation: bool = False
    """Whether to enforce strict input/output validation on the policy client"""
    
    # Debug configuration
    verbose: bool = True
    """Whether to print verbose output"""


class Gr00tInferencePolicy:
    """
    Inference policy that queries a GR00T model server for actions.
    
    This replaces the teleop policy by:
    1. Receiving robot state observations via ROS
    2. Receiving camera images via the camera sensor client
    3. Formatting observations for the GR00T model
    4. Querying the model server for action predictions
    5. Publishing actions to the control loop
    """
    
    def __init__(self, config: Gr00tInferenceConfig):
        self.config = config
        
        # Initialize robot model for joint indexing
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        hand_type = config.hand_type if config.with_hands else None
        self.robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=config.high_elbow_pose,
            hand_type=hand_type,
        )
        
        # Initialize policy client
        print(f"Connecting to GR00T model server at {config.model_host}:{config.model_port}...")
        self.policy_client = PolicyClient(
            host=config.model_host,
            port=config.model_port,
            timeout_ms=config.connection_timeout_ms,
            strict=config.strict_validation,
        )
        
        # Test connection
        if not self.policy_client.ping():
            raise ConnectionError(
                f"Failed to connect to GR00T model server at {config.model_host}:{config.model_port}"
            )
        print("Successfully connected to GR00T model server!")
        
        # Get modality configuration from server
        try:
            self.modality_config = self.policy_client.get_modality_config()
            print(f"Received modality config from server")
            if self.modality_config:
                print(f"  Action keys: {self.modality_config.get('action', {}).modality_keys if hasattr(self.modality_config.get('action', {}), 'modality_keys') else 'N/A'}")
        except Exception as e:
            print(f"Warning: Could not get modality config from server: {e}")
            self.modality_config = None
        
        # Initialize ZMQ camera sensor client (compatible with xr_teleoperate ImageServer)
        print(f"Connecting to camera server at {config.camera_host}:{config.camera_port}...")
        self.camera_sensor = ZMQCameraSensor(
            server_ip=config.camera_host,
            port=config.camera_port,
        )
        print("Successfully connected to camera server!")
        
        # Action buffer for temporal action chunking
        self.action_buffer = deque(maxlen=config.action_horizon)
        self.action_step_counter = 0
        
        # State tracking
        self.latest_state = None
        self.latest_image = None
        self.last_action_query_time = 0
        
        # Joint indices for state formatting
        self._setup_joint_indices()
        
    def _setup_joint_indices(self):
        """Setup joint indices for state formatting based on robot model."""
        # Get indices from robot model for proper mapping
        # These should match the training data format (modality.json)
        
        # Use robot model to get actual indices
        left_leg_idx = sorted(self.robot_model.get_joint_group_indices("left_leg"))
        right_leg_idx = sorted(self.robot_model.get_joint_group_indices("right_leg"))
        waist_idx = sorted(self.robot_model.get_joint_group_indices("waist"))
        left_arm_idx = sorted(self.robot_model.get_joint_group_indices("left_arm"))
        right_arm_idx = sorted(self.robot_model.get_joint_group_indices("right_arm"))
        left_hand_idx = sorted(self.robot_model.get_joint_group_indices("left_hand"))
        right_hand_idx = sorted(self.robot_model.get_joint_group_indices("right_hand"))
        
        # State indices for extracting from robot state
        self.state_indices = {
            "left_leg": (left_leg_idx[0], left_leg_idx[-1] + 1),
            "right_leg": (right_leg_idx[0], right_leg_idx[-1] + 1),
            "waist": (waist_idx[0], waist_idx[-1] + 1),
            "left_arm": (left_arm_idx[0], left_arm_idx[-1] + 1),
            "left_hand": (left_hand_idx[0], left_hand_idx[-1] + 1),
            "right_arm": (right_arm_idx[0], right_arm_idx[-1] + 1),
            "right_hand": (right_hand_idx[0], right_hand_idx[-1] + 1),
        }
        
        # Store raw indices for action mapping
        self.action_joint_indices = {
            "left_arm": left_arm_idx,
            "right_arm": right_arm_idx,
            "left_hand": left_hand_idx,
            "right_hand": right_hand_idx,
            "waist": waist_idx,
        }
        
        # Upper body indices for the control loop
        self.upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
        
        if self.config.verbose:
            # Dataset joint order (ground truth from training data)
            dataset_joint_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "left_hand_pinky_joint", "left_hand_ring_joint", "left_hand_middle_joint",
                "left_hand_index_joint", "left_hand_thumb_bend_joint", "left_hand_thumb_rotation_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
                "right_hand_pinky_joint", "right_hand_ring_joint", "right_hand_middle_joint",
                "right_hand_index_joint", "right_hand_thumb_bend_joint", "right_hand_thumb_rotation_joint",
            ]

            print(f"\n{'='*80}")
            print(f"JOINT ORDER VERIFICATION (Robot Model vs Dataset)")
            print(f"{'='*80}")
            robot_joint_names = self.robot_model.joint_names
            all_ok = True
            for i, (robot_name, dataset_name) in enumerate(zip(robot_joint_names, dataset_joint_names)):
                match = "OK" if robot_name == dataset_name else "MISMATCH!"
                if robot_name != dataset_name:
                    all_ok = False
                print(f"  idx {i:2d}: robot={robot_name:40s} dataset={dataset_name:40s} {match}")
            if len(robot_joint_names) != len(dataset_joint_names):
                all_ok = False
                print(f"  LENGTH MISMATCH: robot={len(robot_joint_names)} dataset={len(dataset_joint_names)}")
            print(f"{'='*80}")
            print(f"Joint order: {'ALL MATCH' if all_ok else 'MISMATCHES FOUND!'}")
            print(f"{'='*80}\n")

            print(f"Joint group index mapping:")
            for name, (start, end) in self.state_indices.items():
                joint_names_in_group = robot_joint_names[start:end]
                print(f"  {name}[{start}:{end}]: {joint_names_in_group}")

            print(f"\nUpper body indices ({len(self.upper_body_indices)} joints):")
            for idx in self.upper_body_indices:
                print(f"  idx {idx}: {robot_joint_names[idx]}")
            print()
    
    def format_observation(self, state: dict, image: dict) -> dict:
        """
        Format robot state and image into model-expected observation format.
        
        Expected format for GR00T model (unitree_g1):
        - video: {"ego_view": np.ndarray[uint8, (1, 1, H, W, 3)]}
        - state: {"left_leg": np.ndarray[float32, (1, 1, 6)], ...}
        - language: {"annotation.human.task_description": [["task text"]]}
        """
        # Get joint positions from state
        q = state["q"]  # joint positions (41 for inspire_ftp, 43 for dex3)

        # Debug: print state info periodically
        if not hasattr(self, '_obs_count'):
            self._obs_count = 0
        self._obs_count += 1
        if self._obs_count <= 3 or self._obs_count % 100 == 0:
            print(f"\n[DEBUG] Observation #{self._obs_count}:")
            print(f"  State q shape: {q.shape}, dtype: {q.dtype}")
            for part_name, (start, end) in self.state_indices.items():
                vals = q[start:end]
                print(f"  {part_name}[{start}:{end}]: {np.array2string(vals, precision=3, suppress_small=True)}")

        # Format state observations by body part
        state_obs = {}
        for part_name, (start, end) in self.state_indices.items():
            # Shape: (batch=1, time=1, dim)
            part_state = q[start:end].astype(np.float32)
            state_obs[part_name] = np.expand_dims(np.expand_dims(part_state, 0), 0)

        # Format video observation
        ego_view = None
        if "images" in image:
            images = image["images"]
            possible_keys = [
                "ego_view",
                "ego_view/image",
                "ego_view_rgb",
                "ego_view_mono",
            ]
            for key in possible_keys:
                if key in images:
                    ego_view = images[key]
                    break

        if ego_view is None:
            raise ValueError(
                f"No ego_view image found in camera data. Available images: {list(image.get('images', {}).keys())}"
            )

        # Convert BGR to RGB (camera captures in BGR, model expects RGB)
        if len(ego_view.shape) == 3 and ego_view.shape[-1] == 3:
            ego_view = ego_view[..., ::-1].copy()  # BGR -> RGB

        # Ensure image is uint8 and correct shape
        if ego_view.dtype != np.uint8:
            ego_view = ego_view.astype(np.uint8)

        if self._obs_count <= 3:
            print(f"  Image shape: {ego_view.shape}, dtype: {ego_view.dtype}")

        # Shape: (batch=1, time=1, H, W, C)
        video_obs = {
            "ego_view": np.expand_dims(np.expand_dims(ego_view, 0), 0)
        }

        # Format language observation
        language_obs = {
            "annotation.human.task_description": [[self.config.task_description]]
        }

        observation = {
            "video": video_obs,
            "state": state_obs,
            "language": language_obs,
        }

        return observation
    
    def parse_action(self, action: dict) -> list:
        """
        Parse model action output into control commands.
        
        Model output format (unitree_g1):
        - left_arm: (batch, horizon, 7)
        - right_arm: (batch, horizon, 7)
        - left_hand: (batch, horizon, 7)
        - right_hand: (batch, horizon, 7)
        - waist: (batch, horizon, 3)
        - base_height_command: (batch, horizon, 1)
        - navigate_command: (batch, horizon, 3)
        
        Note: Model outputs are relative actions that need to be added to current state
        for some action keys (based on processor_config.json action_configs)
        """
        # Extract actions (remove batch dimension)
        left_arm = action["left_arm"][0]    # (horizon, 7)
        right_arm = action["right_arm"][0]  # (horizon, 7)

        # Handle optional hand actions
        if "left_hand" in action:
            left_hand = action["left_hand"][0]
        else:
            left_hand = np.zeros((left_arm.shape[0], len(self.action_joint_indices["left_hand"])))

        if "right_hand" in action:
            right_hand = action["right_hand"][0]
        else:
            right_hand = np.zeros((right_arm.shape[0], len(self.action_joint_indices["right_hand"])))

        waist = action["waist"][0]          # (horizon, 3)
        base_height = action["base_height_command"][0]  # (horizon, 1)
        nav_cmd = action["navigate_command"][0]  # (horizon, 3)

        # Debug: print action shapes and first-step values
        if not hasattr(self, '_action_count'):
            self._action_count = 0
        self._action_count += 1
        if self._action_count <= 3 or self._action_count % 100 == 0:
            robot_joint_names = self.robot_model.joint_names
            print(f"\n[DEBUG] Action #{self._action_count} from model (first step):")
            print(f"  left_arm:  shape={left_arm.shape}")
            for i, idx in enumerate(self.action_joint_indices["left_arm"]):
                if i < left_arm.shape[1]:
                    print(f"    [{i}] -> q[{idx}] {robot_joint_names[idx]:40s} = {left_arm[0, i]:.4f}")
            print(f"  right_arm: shape={right_arm.shape}")
            for i, idx in enumerate(self.action_joint_indices["right_arm"]):
                if i < right_arm.shape[1]:
                    print(f"    [{i}] -> q[{idx}] {robot_joint_names[idx]:40s} = {right_arm[0, i]:.4f}")
            print(f"  left_hand: shape={left_hand.shape}")
            for i, idx in enumerate(self.action_joint_indices["left_hand"]):
                if i < left_hand.shape[1]:
                    print(f"    [{i}] -> q[{idx}] {robot_joint_names[idx]:40s} = {left_hand[0, i]:.4f}")
            print(f"  right_hand: shape={right_hand.shape}")
            for i, idx in enumerate(self.action_joint_indices["right_hand"]):
                if i < right_hand.shape[1]:
                    print(f"    [{i}] -> q[{idx}] {robot_joint_names[idx]:40s} = {right_hand[0, i]:.4f}")
            print(f"  waist:     shape={waist.shape}")
            for i, idx in enumerate(self.action_joint_indices["waist"]):
                if i < waist.shape[1]:
                    print(f"    [{i}] -> q[{idx}] {robot_joint_names[idx]:40s} = {waist[0, i]:.4f}")
            print(f"  nav_cmd:   {np.array2string(nav_cmd[0], precision=3)}")
            print(f"  base_height: {base_height[0]}")

            # Show the final upper body pose that goes to control loop
            full_q = np.zeros(self.robot_model.num_dofs)
            for i, idx in enumerate(self.action_joint_indices["left_arm"]):
                if i < left_arm.shape[1]:
                    full_q[idx] = left_arm[0, i]
            for i, idx in enumerate(self.action_joint_indices["right_arm"]):
                if i < right_arm.shape[1]:
                    full_q[idx] = right_arm[0, i]
            if self.config.with_hands:
                for i, idx in enumerate(self.action_joint_indices["left_hand"]):
                    if i < left_hand.shape[1]:
                        full_q[idx] = left_hand[0, i]
                for i, idx in enumerate(self.action_joint_indices["right_hand"]):
                    if i < right_hand.shape[1]:
                        full_q[idx] = right_hand[0, i]
            for i, idx in enumerate(self.action_joint_indices["waist"]):
                if i < waist.shape[1]:
                    full_q[idx] = waist[0, i]
            upper_body_pose = full_q[self.upper_body_indices]
            print(f"\n  Upper body pose sent to control loop ({len(upper_body_pose)} joints):")
            for i, idx in enumerate(self.upper_body_indices):
                print(f"    upper[{i}] = q[{idx}] {robot_joint_names[idx]:40s} = {upper_body_pose[i]:.4f}")
        
        # Store all action steps
        action_steps = []
        horizon = left_arm.shape[0]
        
        for t in range(horizon):
            # Construct full joint position array
            full_q = np.zeros(self.robot_model.num_dofs)
            
            # Fill in arm joints using robot model indices
            for idx, joint_idx in enumerate(self.action_joint_indices["left_arm"]):
                if idx < left_arm.shape[1]:
                    full_q[joint_idx] = left_arm[t, idx]
                    
            for idx, joint_idx in enumerate(self.action_joint_indices["right_arm"]):
                if idx < right_arm.shape[1]:
                    full_q[joint_idx] = right_arm[t, idx]
            
            # Fill in hand joints if robot has hands
            if self.config.with_hands:
                for idx, joint_idx in enumerate(self.action_joint_indices["left_hand"]):
                    if idx < left_hand.shape[1]:
                        full_q[joint_idx] = left_hand[t, idx]
                        
                for idx, joint_idx in enumerate(self.action_joint_indices["right_hand"]):
                    if idx < right_hand.shape[1]:
                        full_q[joint_idx] = right_hand[t, idx]
            
            # Fill in waist joints
            for idx, joint_idx in enumerate(self.action_joint_indices["waist"]):
                if idx < waist.shape[1]:
                    full_q[joint_idx] = waist[t, idx]
            
            # Extract upper body pose for the control loop
            upper_body_pose = full_q[self.upper_body_indices]
            
            step_action = {
                "target_upper_body_pose": upper_body_pose.astype(np.float64),
                "wrist_pose": DEFAULT_WRIST_POSE,  # Model outputs joint space, not task space
                "base_height_command": float(base_height[t, 0]) if base_height.shape[1] > 0 else DEFAULT_BASE_HEIGHT,
                "navigate_cmd": nav_cmd[t].tolist() if nav_cmd.shape[1] >= 3 else DEFAULT_NAV_CMD,
            }
            action_steps.append(step_action)
        
        return action_steps
    
    def get_action(self, state: dict, image: dict) -> dict:
        """
        Get action from model or action buffer.
        
        Uses temporal action chunking: query model every n_action_steps,
        execute buffered actions in between.
        """
        # Store current state for fallback
        current_q = state.get("q", np.zeros(self.robot_model.num_dofs))
        
        # Check if we need to query the model
        need_query = (
            len(self.action_buffer) == 0 or
            self.action_step_counter >= self.config.n_action_steps
        )
        
        if need_query:
            try:
                # Format observation
                observation = self.format_observation(state, image)
                
                # Query model
                t_start = time.monotonic()
                action, info = self.policy_client.get_action(observation)
                query_time = (time.monotonic() - t_start) * 1000
                
                if self.config.verbose:
                    print(f"Model query took {query_time:.1f}ms, action keys: {list(action.keys())}")
                
                # Parse and buffer actions
                action_steps = self.parse_action(action)
                self.action_buffer.clear()
                self.action_buffer.extend(action_steps)
                self.action_step_counter = 0
                
            except Exception as e:
                print(f"Error querying model: {e}")
                import traceback
                traceback.print_exc()
                # On error, return current state to maintain position
                return self._get_hold_position_action(current_q)
        
        # Get next action from buffer
        if len(self.action_buffer) > 0:
            action = self.action_buffer.popleft()
            self.action_step_counter += 1
            return action
        else:
            # Fallback to holding current position
            return self._get_hold_position_action(current_q)
    
    def _get_hold_position_action(self, current_q: np.ndarray) -> dict:
        """Get action that holds the current position."""
        upper_body_pose = current_q[self.upper_body_indices]
        return {
            "target_upper_body_pose": upper_body_pose.astype(np.float64),
            "wrist_pose": DEFAULT_WRIST_POSE,
            "base_height_command": DEFAULT_BASE_HEIGHT,
            "navigate_cmd": DEFAULT_NAV_CMD,
        }
    
    def reset(self):
        """Reset the policy state."""
        self.action_buffer.clear()
        self.action_step_counter = 0
        self.policy_client.reset()
    
    def close(self):
        """Close connections."""
        self.camera_sensor.close()


def main(config: Gr00tInferenceConfig):
    """Main inference loop."""
    
    ros_manager = ROSManager(node_name=INFERENCE_NODE_NAME)
    node = ros_manager.node
    
    # Initialize inference policy
    try:
        policy = Gr00tInferencePolicy(config)
    except Exception as e:
        print(f"Failed to initialize inference policy: {e}")
        import traceback
        traceback.print_exc()
        ros_manager.shutdown()
        return
    
    # Create ROS publisher for control commands
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    
    # Create ROS subscriber for state (using ROSMsgSubscriber like teleop does)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    
    # Create rate controller
    rate = node.create_rate(config.inference_frequency)
    
    telemetry = Telemetry(window_size=100)
    iteration = 0
    time_to_get_to_initial_pose = 2  # seconds
    first_action_sent = False
    
    latest_state = None
    
    print("\n" + "="*60)
    print("GR00T Inference Policy Started")
    print(f"Task: {config.task_description}")
    print("="*60 + "\n")
    
    try:
        while ros_manager.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()
                
                # Get latest state from subscriber
                with telemetry.timer("get_state"):
                    msg = state_subscriber.get_msg()
                    if msg is not None:
                        latest_state = msg
                
                # Get latest image from camera
                with telemetry.timer("get_image"):
                    latest_image = policy.camera_sensor.read()
                
                # Skip if we don't have both state and image
                if latest_state is None or latest_image is None:
                    if config.verbose and iteration % 20 == 0:
                        print(
                            f"Waiting for data... "
                            f"state: {'OK' if latest_state else 'WAITING'}, "
                            f"image: {'OK' if latest_image else 'WAITING'}"
                        )
                    rate.sleep()
                    iteration += 1
                    continue
                
                # Get action from policy
                with telemetry.timer("get_action"):
                    action = policy.get_action(latest_state, latest_image)
                
                # Add timing information
                t_now = time.monotonic()
                action["timestamp"] = t_now
                
                # Set target completion time
                if not first_action_sent:
                    action["target_time"] = t_now + time_to_get_to_initial_pose
                else:
                    action["target_time"] = t_now + (1 / config.inference_frequency)
                
                # Publish action
                with telemetry.timer("publish_action"):
                    control_publisher.publish(action)
                
                # For initial pose, wait the full duration
                if not first_action_sent:
                    print(f"Moving to initial pose for {time_to_get_to_initial_pose} seconds")
                    time.sleep(time_to_get_to_initial_pose)
                    first_action_sent = True
                
                iteration += 1
                end_time = time.monotonic()
            
            # Log timing if we missed our target frequency
            if (end_time - t_start) > (1 / config.inference_frequency):
                telemetry.log_timing_info(
                    context="Inference Policy Loop Missed",
                    threshold=0.001
                )
            
            rate.sleep()
    
    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted: {e}")
    
    except KeyboardInterrupt:
        print("\nInference policy terminated by user")
    
    finally:
        print("Cleaning up...")
        policy.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(Gr00tInferenceConfig)
    main(config)
