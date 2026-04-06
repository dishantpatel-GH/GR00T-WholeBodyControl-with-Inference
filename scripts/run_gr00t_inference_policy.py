"""
GR00T Inference Policy Loop for Real Robot (Unitree G1 + Inspire FTP Hands)

Queries a GR00T model server for action predictions and sends them to the
control loop via ROS. Uses ZMQ camera feed from final_image_server.py for
ego_view (head RealSense) and ego_view_right_mono (right wrist camera).

Trained modality (unitree_g1):
    Video:  ego_view (848x480) + ego_view_right_mono (640x480)
    State:  41D = legs(12) + waist(3) + arms(14) + hands(12)
    Action: left_arm(7) RELATIVE, right_arm(7) RELATIVE,
            left_hand(6) RELATIVE, right_hand(6) RELATIVE,
            waist(3) ABSOLUTE, base_height_command(1), navigate_command(3)

Prerequisites:
    Terminal 1 - GR00T Model Server:
        python gr00t/eval/run_gr00t_server.py \
            --model-path <checkpoint> --embodiment-tag UNITREE_G1 \
            --device cuda --host 0.0.0.0 --port 5555

    Terminal 2 - Control Loop:
        python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
            --interface eno1 --control_frequency 50 \
            --with_hands --hand_type inspire

    Terminal 3 - (On robot) Inspire FTP hand driver:
        python Headless_driver_double.py

    Terminal 4 - (On robot) Camera server:
        python final_image_server.py --wrist <wrist_port>

    Terminal 5 - This inference script:
        python scripts/run_gr00t_inference_policy.py \
            --model_host <server_ip> --model_port 5555 \
            --camera_host <robot_ip> --camera_port 5555 \
            --head_cam_width 848 --wrist_cam_width 640 \
            --task_description "Pick up biscuit and place it in the plate"
"""

from collections import deque
from dataclasses import dataclass
import pickle
import time

import cv2
import numpy as np
import tyro
import zmq

from gr00t.policy.server_client import PolicyClient
from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber
from gr00t_wbc.control.utils.telemetry import Telemetry


INFERENCE_NODE_NAME = "Gr00tInferencePolicy"

# Dataset joint index slicing (from pick_place_biscuit/meta/modality.json)
# 41D total (no padding on right hand)
DATASET_SLICES = {
    "left_leg":   (0, 6),
    "right_leg":  (6, 12),
    "waist":      (12, 15),
    "left_arm":   (15, 22),
    "right_arm":  (22, 29),
    "left_hand":  (29, 35),   # 6D Inspire FTP (pinky→thumb)
    "right_hand": (35, 41),   # 6D Inspire FTP (pinky→thumb)
}

DATASET_DIM = 41

# Inspire FTP hands: model uses normalized 0-1, hardware expects 0-1000
INSPIRE_HAND_SCALE = 1000.0
HAND_PARTS = {"left_hand", "right_hand"}


# --------------------------------------------------------------------------- #
#  ZMQ Camera Client (receives from final_image_server.py)
# --------------------------------------------------------------------------- #

class ZMQCameraClient:
    """
    Receives camera frames from final_image_server.py via ZMQ.

    The server concatenates head + wrist images horizontally and sends as JPEG
    (or pickle with depth). This client splits them into ego_view and
    ego_view_right_mono based on known widths.

    The wrist camera may stream at a different resolution (e.g. 640x480) than
    the model expects (e.g. 848x480). If model_wrist_size is set, the wrist
    image is resized to match the training resolution.
    """

    def __init__(self, server_ip: str, port: int,
                 head_cam_width: int = 848, wrist_cam_width: int = 640,
                 model_head_size: tuple = (848, 480),
                 model_wrist_size: tuple = (848, 480)):
        self.head_cam_width = head_cam_width
        self.wrist_cam_width = wrist_cam_width
        self.model_head_size = model_head_size    # (w, h) model expects
        self.model_wrist_size = model_wrist_size  # (w, h) model expects

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{server_ip}:{port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Only keep latest frame
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

        print(f"[ZMQCamera] Connected to tcp://{server_ip}:{port}")
        print(f"[ZMQCamera] Stream widths: head={head_cam_width}, wrist={wrist_cam_width}")
        print(f"[ZMQCamera] Model sizes:   head={model_head_size}, wrist={model_wrist_size}")

    def read(self) -> dict:
        """
        Read latest frame. Returns dict with 'images' key containing
        'ego_view' and optionally 'ego_view_right_mono' as BGR numpy arrays.
        Returns None if no frame available.
        """
        try:
            message = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return None

        # Decode message — could be raw JPEG or pickle with depth
        jpg_bytes = None
        is_jpg = len(message) > 2 and message[0:2] == b'\xff\xd8'

        if is_jpg:
            jpg_bytes = message
        else:
            try:
                data = pickle.loads(message)
                if isinstance(data, dict):
                    jpg_bytes = data.get('image') or data.get('rgb')
                else:
                    jpg_bytes = message
            except Exception:
                jpg_bytes = message

        if jpg_bytes is None:
            return None

        np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        h, w = frame.shape[:2]
        images = {}

        # Split concatenated image: [head | depth(optional) | wrist]
        # Head camera is always the leftmost portion
        if w >= self.head_cam_width:
            ego = frame[:, :self.head_cam_width]
            # Resize to model-expected resolution if different
            eh, ew = ego.shape[:2]
            mw, mh = self.model_head_size
            if ew != mw or eh != mh:
                ego = cv2.resize(ego, (mw, mh))
            images["ego_view"] = ego

        # Wrist camera is the rightmost portion
        if w > self.head_cam_width:
            wrist = frame[:, -self.wrist_cam_width:]
            # Resize to model-expected resolution (e.g. 640x480 → 848x480)
            wh, ww = wrist.shape[:2]
            mw, mh = self.model_wrist_size
            if ww != mw or wh != mh:
                wrist = cv2.resize(wrist, (mw, mh))
            images["ego_view_right_mono"] = wrist
        elif "ego_view" not in images:
            # Fallback: use entire frame as ego_view
            images["ego_view"] = frame

        return {"images": images}

    def close(self):
        self._socket.close()
        self._context.term()


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #

@dataclass
class Gr00tInferenceConfig:
    """Configuration for GR00T inference policy loop."""

    # Model server
    model_host: str = "localhost"
    """Host address for the GR00T model server"""
    model_port: int = 5555
    """Port number for the GR00T model server"""

    # Camera (ZMQ from final_image_server.py)
    camera_host: str = "192.168.123.164"
    """Host address for the ZMQ camera server (robot IP)"""
    camera_port: int = 5555
    """Port number for the ZMQ camera server"""
    head_cam_width: int = 848
    """Width of head camera in the stream (for splitting concatenated image)"""
    wrist_cam_width: int = 640
    """Width of wrist camera in the stream (for splitting). Will be resized to model size."""

    # Task
    task_description: str = "Pick up biscuit and place it in the plate"
    """Task description for the model"""

    # Robot
    enable_waist: bool = False
    """Whether waist is in upper body IK"""
    high_elbow_pose: bool = False
    """High elbow pose configuration"""
    with_hands: bool = True
    """Inspire FTP hands enabled"""

    # Inference
    inference_frequency: float = 30.0
    """Inference loop frequency (Hz)"""
    n_action_steps: int = 16
    """Action steps to execute before re-querying model"""
    action_horizon: int = 30
    """Total action horizon from model"""

    # Connection
    connection_timeout_ms: int = 15000
    """Model server connection timeout (ms)"""
    strict_validation: bool = False
    """Strict input/output validation"""

    verbose: bool = True

    # Debug logging
    debug_log: bool = False
    """Save observations & actions to logs_debug/ for offline comparison with dataset"""
    debug_log_dir: str = "logs_debug"
    """Directory for debug logs"""


# --------------------------------------------------------------------------- #
#  Debug Logger
# --------------------------------------------------------------------------- #

class InferenceDebugLogger:
    """
    Logs observations sent to the model and raw actions received, for offline
    comparison against dataset distribution.

    Saves to logs_debug/:
        obs_states.npy    — (N, 41) state vectors sent to model (normalized, 0-1 hands)
        raw_actions.npz   — raw action arrays as returned by model (before scaling)
        actions_hw.npy    — (N, 41) actions mapped to dataset format (0-1 hands, pre-hw-scale)
        images/           — sampled ego_view frames (every 30th)
    """

    def __init__(self, log_dir: str, dataset_slices: dict):
        import os
        self.log_dir = log_dir
        self.ds_slices = dataset_slices
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "images"), exist_ok=True)

        self.obs_states = []       # 41D states sent to model
        self.raw_actions = []      # raw model output dicts (first timestep only)
        self.actions_41d = []      # 41D action in dataset format (normalized hands)
        self.query_count = 0
        print(f"[DebugLog] Logging to {log_dir}/")

    def log_observation(self, state_41d: np.ndarray, video_obs: dict):
        """Log the state observation sent to model."""
        self.obs_states.append(state_41d.copy())

        # Save a sample image every 30 queries
        if self.query_count % 30 == 0 and "ego_view" in video_obs:
            img = video_obs["ego_view"][0, 0]  # (H, W, 3) RGB
            img_bgr = img[..., ::-1]  # RGB→BGR for cv2
            path = os.path.join(self.log_dir, "images", f"ego_{self.query_count:06d}.jpg")
            cv2.imwrite(path, img_bgr)

    def log_action(self, raw_action: dict):
        """Log the raw action dict from model (first timestep)."""
        entry = {}
        for k, v in raw_action.items():
            if hasattr(v, 'shape'):
                # Store first timestep: (batch, horizon, dim) → (dim,)
                if v.ndim == 3:
                    entry[k] = v[0, 0].copy()
                elif v.ndim == 2:
                    entry[k] = v[0].copy()
                else:
                    entry[k] = v.copy()
        self.raw_actions.append(entry)

        # Also reconstruct a 41D action in dataset format (normalized, no hw scaling)
        action_41d = np.zeros(41, dtype=np.float64)
        for part_name in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]:
            if part_name in entry:
                ds_s, ds_e = self.ds_slices[part_name]
                vals = entry[part_name]
                n = min(len(vals), ds_e - ds_s)
                action_41d[ds_s:ds_s + n] = vals[:n]
        self.actions_41d.append(action_41d)
        self.query_count += 1

    def save(self):
        """Save all logged data to disk."""
        if not self.obs_states:
            print("[DebugLog] No data to save.")
            return

        obs_path = os.path.join(self.log_dir, "obs_states.npy")
        np.save(obs_path, np.array(self.obs_states))
        print(f"[DebugLog] Saved {len(self.obs_states)} observations → {obs_path}")

        if self.actions_41d:
            act_path = os.path.join(self.log_dir, "actions_41d.npy")
            np.save(act_path, np.array(self.actions_41d))
            print(f"[DebugLog] Saved {len(self.actions_41d)} actions → {act_path}")

        if self.raw_actions:
            raw_path = os.path.join(self.log_dir, "raw_actions.npz")
            # Collect per-key arrays
            keys = self.raw_actions[0].keys()
            arrays = {k: np.array([a[k] for a in self.raw_actions if k in a]) for k in keys}
            np.savez(raw_path, **arrays)
            print(f"[DebugLog] Saved raw actions → {raw_path}")

        print(f"[DebugLog] Done. Run: python scripts/compare_debug_to_dataset.py "
              f"--debug_dir {self.log_dir} --dataset_path <dataset>")


import os  # needed by debug logger


# --------------------------------------------------------------------------- #
#  Inference Policy
# --------------------------------------------------------------------------- #

class Gr00tInferencePolicy:
    """
    Queries GR00T model server for actions on Unitree G1 + Inspire FTP hands.
    Uses ZMQ camera for ego_view + ego_view_right_mono.
    """

    def __init__(self, config: Gr00tInferenceConfig):
        self.config = config

        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(
            waist_location=waist_location,
            high_elbow_pose=config.high_elbow_pose,
        )

        # Policy client
        print(f"Connecting to model server at {config.model_host}:{config.model_port}...")
        self.policy_client = PolicyClient(
            host=config.model_host,
            port=config.model_port,
            timeout_ms=config.connection_timeout_ms,
            strict=config.strict_validation,
        )
        if not self.policy_client.ping():
            raise ConnectionError(
                f"Cannot reach model server at {config.model_host}:{config.model_port}"
            )
        print("Connected to model server!")

        try:
            self.modality_config = self.policy_client.get_modality_config()
            if self.modality_config:
                action_cfg = self.modality_config.get("action", {})
                if hasattr(action_cfg, "modality_keys"):
                    print(f"  Action keys: {action_cfg.modality_keys}")
        except Exception as e:
            print(f"Warning: Could not get modality config: {e}")
            self.modality_config = None

        # ZMQ Camera
        print(f"Connecting to camera at {config.camera_host}:{config.camera_port}...")
        self.camera = ZMQCameraClient(
            server_ip=config.camera_host,
            port=config.camera_port,
            head_cam_width=config.head_cam_width,
            wrist_cam_width=config.wrist_cam_width,
        )

        # Action buffer
        self.action_buffer = deque(maxlen=config.action_horizon)
        self.action_step_counter = 0

        # Debug logger
        self.debug_logger = None
        if config.debug_log:
            self.debug_logger = InferenceDebugLogger(config.debug_log_dir, DATASET_SLICES)

        self._cache_joint_indices()

    def _cache_joint_indices(self):
        """Cache robot model joint indices. Hands use actuated joint order."""
        self.model_indices = {}
        for part_name in ["left_leg", "right_leg", "waist", "left_arm", "right_arm"]:
            self.model_indices[part_name] = sorted(
                self.robot_model.get_joint_group_indices(part_name)
            )
        self.model_indices["left_hand"] = self.robot_model.get_hand_actuated_joint_indices("left")
        self.model_indices["right_hand"] = self.robot_model.get_hand_actuated_joint_indices("right")
        self.upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")

        if self.config.verbose:
            print(f"\nRobot model: {self.robot_model.num_dofs} DOF, "
                  f"Upper body: {len(self.upper_body_indices)} joints")
            for name in ["waist", "left_arm", "right_arm", "left_hand", "right_hand"]:
                ds_s, ds_e = DATASET_SLICES[name]
                print(f"  {name}: dataset[{ds_s}:{ds_e}] ({ds_e-ds_s}D) "
                      f"-> model {len(self.model_indices[name])} indices")

    def _robot_q_to_state_41d(self, q: np.ndarray) -> np.ndarray:
        """Convert robot model q (43D) to dataset format (41D).

        Legs and waist are zeroed out because the training data had zeros
        for these joints (they weren't recorded during data collection).
        Only arms and hands carry actual robot state.
        """
        state = np.zeros(DATASET_DIM, dtype=np.float64)

        # Only populate arms and hands — legs/waist stay zero (matching training data)
        for part_name in ["left_arm", "right_arm", "left_hand", "right_hand"]:
            ds_start, ds_end = DATASET_SLICES[part_name]
            model_idx = self.model_indices[part_name]
            ds_dim = ds_end - ds_start
            n = min(ds_dim, len(model_idx))
            state[ds_start:ds_start + n] = q[model_idx[:n]]

        # Hands: hardware 0-1000 → model normalized 0-1
        for hand_name in HAND_PARTS:
            ds_start, ds_end = DATASET_SLICES[hand_name]
            state[ds_start:ds_end] /= INSPIRE_HAND_SCALE
        return state

    def _state_41d_to_model_state(self, state_41d: np.ndarray) -> dict:
        """Split 41D state into per-body-part arrays for model input."""
        state_obs = {}
        for part_name, (ds_start, ds_end) in DATASET_SLICES.items():
            part = state_41d[ds_start:ds_end].astype(np.float32)
            state_obs[part_name] = part[np.newaxis, np.newaxis, :]  # (1, 1, dim)
        return state_obs

    def _model_action_to_robot_q(self, action_parts: dict) -> np.ndarray:
        """Map single-timestep model action to robot model q."""
        q = self.robot_model.default_body_pose.copy()

        for part_name in ["left_arm", "right_arm", "waist"]:
            if part_name in action_parts:
                vals = action_parts[part_name]
                idx = self.model_indices[part_name]
                n = min(vals.shape[-1], len(idx))
                q[idx[:n]] = vals[:n]

        # Hands: model outputs normalized [0,1], scale to hardware [0,1000]
        for hand_name in HAND_PARTS:
            if hand_name in action_parts and self.config.with_hands:
                vals = action_parts[hand_name] * INSPIRE_HAND_SCALE
                idx = self.model_indices[hand_name]
                n = min(vals.shape[-1], len(idx))
                q[idx[:n]] = vals[:n]

        return q

    def format_observation(self, state: dict, camera_data: dict) -> dict:
        """
        Format robot state + camera images for the model.

        Video: ego_view (848x480) + ego_view_right_mono (640x480)
        State: 41D split into 7 body parts
        Language: task description
        """
        q = state["q"]

        # Diagnostic: check if hand state is being received (first 5 queries only)
        if self.config.verbose and (not hasattr(self, '_obs_count') or self._obs_count < 5):
            self._obs_count = getattr(self, '_obs_count', 0) + 1
            lh_idx = self.model_indices["left_hand"]
            rh_idx = self.model_indices["right_hand"]
            lh_raw = q[lh_idx]
            rh_raw = q[rh_idx]
            print(f"[DiagHand] Raw q hand values (hardware units):")
            print(f"  left_hand  idx={lh_idx}: {lh_raw}")
            print(f"  right_hand idx={rh_idx}: {rh_raw}")
            if np.all(lh_raw == 0) and np.all(rh_raw == 0):
                print("  WARNING: All hand values are ZERO! "
                      "Is Headless_driver_double.py running?")

        state_41d = self._robot_q_to_state_41d(q)
        state_obs = self._state_41d_to_model_state(state_41d)

        # Video: BGR→RGB, uint8, shape (1, 1, H, W, 3)
        video_obs = {}
        images = camera_data.get("images", {})

        # Debug: will log after video_obs is built (below)

        for view_key in ["ego_view", "ego_view_right_mono"]:
            if view_key in images:
                img = images[view_key]
                if len(img.shape) == 3 and img.shape[-1] == 3:
                    img = img[..., ::-1].copy()  # BGR→RGB
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                video_obs[view_key] = img[np.newaxis, np.newaxis, ...]

        if "ego_view" not in video_obs:
            raise ValueError(
                f"No ego_view in camera data. Available: {list(images.keys())}"
            )

        # Debug log: observation state + sample image
        if self.debug_logger:
            self.debug_logger.log_observation(state_41d, video_obs)

        language_obs = {
            "annotation.human.task_description": [[self.config.task_description]]
        }

        return {
            "video": video_obs,
            "state": state_obs,
            "language": language_obs,
        }

    def parse_action(self, action: dict) -> list:
        """
        Parse model output into list of control commands.

        Model outputs (unitree_g1):
            left_arm:  (1, horizon, 7)  RELATIVE
            right_arm: (1, horizon, 7)  RELATIVE
            left_hand: (1, horizon, 6)  RELATIVE
            right_hand:(1, horizon, 6)  RELATIVE
            waist:     (1, horizon, 3)  ABSOLUTE
            base_height_command: (1, horizon, 1)
            navigate_command:    (1, horizon, 3)
        """
        left_arm = action["left_arm"][0]
        right_arm = action["right_arm"][0]
        waist = action["waist"][0]
        nav_cmd = action["navigate_command"][0]

        left_hand = action.get("left_hand", [None])[0] if self.config.with_hands else None
        right_hand = action.get("right_hand", [None])[0] if self.config.with_hands else None

        horizon = left_arm.shape[0]
        steps = []

        for t in range(horizon):
            parts = {
                "left_arm": left_arm[t],
                "right_arm": right_arm[t],
                "waist": waist[t],
            }
            if left_hand is not None:
                parts["left_hand"] = left_hand[t]
            if right_hand is not None:
                parts["right_hand"] = right_hand[t]

            full_q = self._model_action_to_robot_q(parts)
            upper_body_pose = full_q[self.upper_body_indices]

            steps.append({
                "target_upper_body_pose": upper_body_pose.astype(np.float64),
                "wrist_pose": DEFAULT_WRIST_POSE,
                "base_height_command": DEFAULT_BASE_HEIGHT,
                "navigate_cmd": (
                    nav_cmd[t].tolist() if nav_cmd.shape[-1] >= 3 else DEFAULT_NAV_CMD
                ),
            })

        return steps

    def get_action(self, state: dict, camera_data: dict) -> dict:
        """Get action from model or buffer (temporal action chunking)."""
        current_q = state.get("q", np.zeros(self.robot_model.num_dofs))

        need_query = (
            len(self.action_buffer) == 0
            or self.action_step_counter >= self.config.n_action_steps
        )

        if need_query:
            try:
                obs = self.format_observation(state, camera_data)
                t0 = time.monotonic()
                action, info = self.policy_client.get_action(obs)
                dt_ms = (time.monotonic() - t0) * 1000

                if self.config.verbose:
                    shapes = {k: v.shape for k, v in action.items() if hasattr(v, 'shape')}
                    print(f"Model query: {dt_ms:.1f}ms, shapes: {shapes}")

                # Debug log: raw model output
                if self.debug_logger:
                    self.debug_logger.log_action(action)

                steps = self.parse_action(action)
                self.action_buffer.clear()
                self.action_buffer.extend(steps)
                self.action_step_counter = 0

            except Exception as e:
                print(f"Error querying model: {e}")
                import traceback
                traceback.print_exc()
                return self._hold_position(current_q)

        if self.action_buffer:
            act = self.action_buffer.popleft()
            self.action_step_counter += 1
            return act
        return self._hold_position(current_q)

    def _hold_position(self, current_q: np.ndarray) -> dict:
        return {
            "target_upper_body_pose": current_q[self.upper_body_indices].astype(np.float64),
            "wrist_pose": DEFAULT_WRIST_POSE,
            "base_height_command": DEFAULT_BASE_HEIGHT,
            "navigate_cmd": DEFAULT_NAV_CMD,
        }

    def reset(self):
        self.action_buffer.clear()
        self.action_step_counter = 0
        self.policy_client.reset()

    def close(self):
        if self.debug_logger:
            self.debug_logger.save()
        self.camera.close()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main(config: Gr00tInferenceConfig):
    ros_manager = ROSManager(node_name=INFERENCE_NODE_NAME)
    node = ros_manager.node

    try:
        policy = Gr00tInferencePolicy(config)
    except Exception as e:
        print(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        ros_manager.shutdown()
        return

    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
    rate = node.create_rate(config.inference_frequency)

    telemetry = Telemetry(window_size=100)
    first_action_sent = False
    time_to_initial_pose = 2.0
    iteration = 0
    latest_state = None

    print("\n" + "=" * 60)
    print("GR00T Inference Policy Ready")
    print(f"  Task: {config.task_description}")
    print(f"  Hands: {'Inspire FTP (6 DOF/hand)' if config.with_hands else 'DISABLED'}")
    print(f"  Video: ego_view ({config.head_cam_width}px) + ego_view_right_mono ({config.wrist_cam_width}px)")
    print(f"  Frequency: {config.inference_frequency} Hz")
    print(f"  Action chunking: {config.n_action_steps} steps / {config.action_horizon} horizon")
    print("=" * 60)

    print("\n  Press 's' + Enter to START inference")
    print("  Press 'q' + Enter to QUIT")
    while True:
        try:
            key = input(">>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            policy.close()
            ros_manager.shutdown()
            return
        if key == "s":
            break
        if key == "q":
            policy.close()
            ros_manager.shutdown()
            return
        print("  Press 's' to start or 'q' to quit.")

    print("\nStarting inference loop...\n")

    try:
        while ros_manager.ok():
            with telemetry.timer("total_loop"):
                t_start = time.monotonic()

                with telemetry.timer("get_state"):
                    msg = state_subscriber.get_msg()
                    if msg is not None:
                        latest_state = msg

                with telemetry.timer("get_image"):
                    latest_image = policy.camera.read()

                if latest_state is None or latest_image is None:
                    if config.verbose and iteration % 20 == 0:
                        has_views = list(latest_image["images"].keys()) if latest_image else []
                        print(
                            f"Waiting... state: {'OK' if latest_state else 'WAIT'}, "
                            f"image: {has_views if latest_image else 'WAIT'}"
                        )
                    rate.sleep()
                    iteration += 1
                    continue

                with telemetry.timer("get_action"):
                    action = policy.get_action(latest_state, latest_image)

                t_now = time.monotonic()
                action["timestamp"] = t_now
                action["target_time"] = (
                    t_now + time_to_initial_pose if not first_action_sent
                    else t_now + (1 / config.inference_frequency)
                )

                with telemetry.timer("publish"):
                    control_publisher.publish(action)

                if not first_action_sent:
                    print(f"Moving to initial pose for {time_to_initial_pose}s")
                    time.sleep(time_to_initial_pose)
                    first_action_sent = True

                iteration += 1
                end_time = time.monotonic()

            if (end_time - t_start) > (1 / config.inference_frequency):
                telemetry.log_timing_info(context="Inference Loop Missed", threshold=0.001)

            rate.sleep()

    except ros_manager.exceptions() as e:
        print(f"ROSManager interrupted: {e}")
    except KeyboardInterrupt:
        print("\nTerminated by user")
    finally:
        print("Cleaning up...")
        policy.close()
        ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(Gr00tInferenceConfig)
    main(config)
