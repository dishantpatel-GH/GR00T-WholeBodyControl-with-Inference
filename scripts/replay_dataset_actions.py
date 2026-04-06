"""
Replay Dataset Actions on Real Robot

Reads a recorded episode from the pick_place_biscuit dataset and replays the
UPPER BODY actions (arms + hands + waist) on the real robot through the
existing control loop. Legs are NOT replayed — the WBC lower body policy
handles locomotion via base_height_command and navigate_command.

Trained model action modality (unitree_g1):
    left_arm(7)  RELATIVE, right_arm(7)  RELATIVE,
    left_hand(6) RELATIVE, right_hand(6) RELATIVE,
    waist(3)     ABSOLUTE,
    base_height_command(1) ABSOLUTE, navigate_command(3) ABSOLUTE

Dataset: 41D state/action (no padding on right hand)

Prerequisites:
    Terminal 1 - Control loop with Inspire FTP hands:
        python gr00t_wbc/control/main/teleop/run_g1_control_loop.py \
            --interface eno1 --control_frequency 50 \
            --with_hands --hand_type inspire

    Terminal 2 - (On robot) Inspire FTP hand driver:
        python Headless_driver_double.py

    Terminal 3 - This replay script:
        python scripts/replay_dataset_actions.py \
            --dataset_path pick_place_biscuit --episode_index 0

Usage:
    python scripts/replay_dataset_actions.py --help
"""

from dataclasses import dataclass
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro

from gr00t_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    DEFAULT_BASE_HEIGHT,
    DEFAULT_NAV_CMD,
    DEFAULT_WRIST_POSE,
    STATE_TOPIC_NAME,
)
from gr00t_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gr00t_wbc.control.utils.ros_utils import ROSManager, ROSMsgPublisher, ROSMsgSubscriber


# Action modality keys — only these are output by the trained model.
# NO leg joints. Legs handled by WBC lower body policy.
DATASET_ACTION_PARTS = ["waist", "left_arm", "right_arm", "left_hand", "right_hand"]

# All body part keys we need from modality.json
ALL_BODY_PARTS = [
    "left_leg", "right_leg", "waist",
    "left_arm", "right_arm", "left_hand", "right_hand",
]


def load_dataset_slices(dataset_path: str) -> dict:
    """Load joint index slicing from the dataset's own modality.json.
    This avoids hardcoding indices which differ between datasets."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_absolute():
        if not dataset_dir.exists():
            project_root = Path(__file__).resolve().parent.parent
            dataset_dir = project_root / dataset_path

    modality_path = dataset_dir / "meta" / "modality.json"
    with open(modality_path) as f:
        modality = json.load(f)

    slices = {}
    for part_name in ALL_BODY_PARTS:
        entry = modality["state"][part_name]
        slices[part_name] = (entry["start"], entry["end"])

    return slices

# Inspire FTP hands: dataset stores normalized 0-1, hardware expects 0-1000
INSPIRE_HAND_SCALE = 1000.0
HAND_PARTS = {"left_hand", "right_hand"}

REPLAY_NODE_NAME = "DatasetReplay"


@dataclass
class ReplayConfig:
    """Configuration for dataset action replay."""

    dataset_path: str = "pick_place_biscuit"
    """Path to the dataset directory."""

    episode_index: int = 0
    """Episode index to replay."""

    replay_speed: float = 1.0
    """Replay speed multiplier. 1.0 = original speed, 0.5 = half speed."""

    enable_waist: bool = False
    """Whether waist is controlled by upper body IK (must match control loop)."""

    start_frame: int = 0
    """Frame index to start replay from."""

    end_frame: int = -1
    """Frame index to end replay at (-1 = end of episode)."""

    move_to_start_duration: float = 3.0
    """Seconds to interpolate to the starting pose before replay begins."""

    dry_run: bool = False
    """If True, only print actions without publishing to robot."""

    verbose: bool = True
    """Print detailed action info during replay."""


def load_episode(dataset_path: str, episode_index: int) -> dict:
    """Load a single episode from the dataset."""
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_absolute():
        if not dataset_dir.exists():
            project_root = Path(__file__).resolve().parent.parent
            dataset_dir = project_root / dataset_path

    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    fps = info["fps"]

    if episode_index < 0 or episode_index >= total_episodes:
        raise ValueError(
            f"Episode index {episode_index} out of range [0, {total_episodes})"
        )

    chunk_index = episode_index // info.get("chunks_size", 1000)
    data_path_template = info["data_path"]
    parquet_path = dataset_dir / data_path_template.format(
        episode_chunk=chunk_index, episode_index=episode_index
    )

    print(f"Loading episode {episode_index} from {parquet_path}")
    table = pq.read_table(parquet_path)

    # Extract columns using pyarrow (no pandas dependency)
    episode_data = {
        "action": np.array([row.as_py() for row in table.column("action")], dtype=np.float64),
        "observation_state": np.array(
            [row.as_py() for row in table.column("observation.state")], dtype=np.float64
        ),
        "fps": fps,
        "num_frames": table.num_rows,
    }

    if "teleop.navigate_command" in table.column_names:
        episode_data["navigate_command"] = np.array(
            [row.as_py() for row in table.column("teleop.navigate_command")], dtype=np.float64
        )
    if "action.eef" in table.column_names:
        episode_data["action_eef"] = np.array(
            [row.as_py() for row in table.column("action.eef")], dtype=np.float64
        )

    # Load task description
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                ep = json.loads(line)
                if ep["episode_index"] == episode_index:
                    episode_data["task"] = ep.get("tasks", ["unknown"])
                    break

    print(f"  Frames: {episode_data['num_frames']}, FPS: {fps}, "
          f"Duration: {episode_data['num_frames'] / fps:.1f}s")
    print(f"  Action dim: {episode_data['action'].shape[1]}")
    if "task" in episode_data:
        print(f"  Task: {episode_data['task']}")

    return episode_data


def dataset_action_to_upper_body_pose(
    dataset_action: np.ndarray,
    robot_model,
    model_indices: dict,
    upper_body_indices: list,
    dataset_slices: dict,
) -> np.ndarray:
    """
    Map dataset action to the robot model's target_upper_body_pose.

    Only maps UPPER BODY joints: arms + hands + waist.
    Legs are NOT mapped. Hand values are scaled from [0,1] to [0,1000].
    """
    q = robot_model.default_body_pose.copy()

    for part_name in DATASET_ACTION_PARTS:
        ds_start, ds_end = dataset_slices[part_name]
        idx = model_indices[part_name]
        ds_values = dataset_action[ds_start:ds_end].copy()

        # Inspire FTP hands: dataset stores normalized [0, 1], hardware expects [0, 1000]
        if part_name in HAND_PARTS:
            ds_values = ds_values * INSPIRE_HAND_SCALE

        n = min(len(ds_values), len(idx))
        q[idx[:n]] = ds_values[:n]
        for i in range(n, len(idx)):
            q[idx[i]] = 0.0

    return q[upper_body_indices]


def wait_for_start():
    """Wait for user to press 's' to start."""
    print("\n" + "=" * 50)
    print("  Robot is ready.")
    print("  Press 's' + Enter to START replay")
    print("  Press 'q' + Enter to QUIT")
    print("=" * 50)
    while True:
        try:
            key = input(">>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False
        if key == "s":
            return True
        if key == "q":
            return False
        print("  Invalid input. Press 's' to start or 'q' to quit.")


def main(config: ReplayConfig):
    """Main replay loop."""

    # Load dataset slices from modality.json (avoids hardcoded index mismatch)
    ds_slices = load_dataset_slices(config.dataset_path)
    print(f"Loaded slices from {config.dataset_path}/meta/modality.json:")
    for name in ALL_BODY_PARTS:
        print(f"  {name}: [{ds_slices[name][0]}:{ds_slices[name][1]}]")

    episode_data = load_episode(config.dataset_path, config.episode_index)
    actions = episode_data["action"]
    fps = episode_data["fps"]
    num_frames = episode_data["num_frames"]

    start_frame = config.start_frame
    end_frame = config.end_frame if config.end_frame > 0 else num_frames
    end_frame = min(end_frame, num_frames)

    if start_frame >= end_frame:
        print(f"Invalid frame range: [{start_frame}, {end_frame})")
        return

    print(f"\nWill replay frames [{start_frame}, {end_frame}) "
          f"({end_frame - start_frame} frames)")

    # Initialize robot model
    waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(waist_location=waist_location)

    # Cache indices — use get_hand_actuated_joint_indices for hands
    # to preserve finger order (pinky→thumb) matching queue_action extraction
    model_indices = {}
    for part_name in ALL_BODY_PARTS:
        if part_name == "left_hand":
            model_indices[part_name] = robot_model.get_hand_actuated_joint_indices("left")
        elif part_name == "right_hand":
            model_indices[part_name] = robot_model.get_hand_actuated_joint_indices("right")
        else:
            model_indices[part_name] = sorted(
                robot_model.get_joint_group_indices(part_name)
            )
    upper_body_indices = robot_model.get_joint_group_indices("upper_body")

    print(f"\nRobot model: {robot_model.num_dofs} DOF, Upper body: {len(upper_body_indices)} joints")
    for name in DATASET_ACTION_PARTS:
        ds_s, ds_e = ds_slices[name]
        print(f"  {name}: dataset[{ds_s}:{ds_e}] ({ds_e - ds_s}D) "
              f"-> model {len(model_indices[name])} indices")

    # --- Dry run ---
    if config.dry_run:
        print("\n=== DRY RUN ===")
        for i in range(start_frame, min(start_frame + 5, end_frame)):
            ub_pose = dataset_action_to_upper_body_pose(
                actions[i], robot_model, model_indices, upper_body_indices, ds_slices
            )
            lh_s, lh_e = ds_slices["left_hand"]
            rh_s, rh_e = ds_slices["right_hand"]
            print(f"\nFrame {i}: ub_pose shape={ub_pose.shape}")
            print(f"  L-hand (raw): {actions[i][lh_s:lh_e]}")
            print(f"  R-hand (raw): {actions[i][rh_s:rh_e]}")
            print(f"  L-hand (hw):  {actions[i][lh_s:lh_e] * INSPIRE_HAND_SCALE}")
            print(f"  R-hand (hw):  {actions[i][rh_s:rh_e] * INSPIRE_HAND_SCALE}")
        print("\nDry run complete.")
        return

    # --- Real robot ---
    ros_manager = ROSManager(node_name=REPLAY_NODE_NAME)
    control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
    state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)

    print("\nWaiting for control loop...")
    timeout_start = time.monotonic()
    while ros_manager.ok():
        msg = state_subscriber.get_msg()
        if msg is not None:
            print("  Control loop active!")
            break
        if time.monotonic() - timeout_start > 10.0:
            print("  ERROR: No state from control loop after 10s.")
            ros_manager.shutdown()
            return
        time.sleep(0.1)

    if not wait_for_start():
        print("Aborted.")
        ros_manager.shutdown()
        return

    # Phase 1: Move to starting pose
    print(f"\nMoving to starting pose over {config.move_to_start_duration}s...")
    start_ub_pose = dataset_action_to_upper_body_pose(
        actions[start_frame], robot_model, model_indices, upper_body_indices, ds_slices
    )

    start_nav_cmd = DEFAULT_NAV_CMD
    if "navigate_command" in episode_data:
        start_nav_cmd = episode_data["navigate_command"][start_frame].tolist()

    wrist_pose = DEFAULT_WRIST_POSE
    if "action_eef" in episode_data:
        wrist_pose = episode_data["action_eef"][start_frame].tolist()

    t_now = time.monotonic()
    control_publisher.publish({
        "target_upper_body_pose": start_ub_pose.astype(np.float64),
        "wrist_pose": wrist_pose,
        "base_height_command": DEFAULT_BASE_HEIGHT,
        "navigate_cmd": start_nav_cmd,
        "timestamp": t_now,
        "target_time": t_now + config.move_to_start_duration,
    })
    time.sleep(config.move_to_start_duration + 0.5)
    print("  Reached starting pose.")

    # Phase 2: Replay actions
    dt = (1.0 / fps) / config.replay_speed
    print(f"\nReplaying at {fps * config.replay_speed:.1f} Hz "
          f"(original {fps} Hz, speed {config.replay_speed}x)")
    print("Press Ctrl+C to stop.\n")

    lh_s, lh_e = ds_slices["left_hand"]
    rh_s, rh_e = ds_slices["right_hand"]

    frame_idx = start_frame
    try:
        for frame_idx in range(start_frame, end_frame):
            if not ros_manager.ok():
                break

            t_loop_start = time.monotonic()

            ub_pose = dataset_action_to_upper_body_pose(
                actions[frame_idx], robot_model, model_indices, upper_body_indices, ds_slices
            )

            nav_cmd = DEFAULT_NAV_CMD
            wrist_pose = DEFAULT_WRIST_POSE
            if "navigate_command" in episode_data:
                nav_cmd = episode_data["navigate_command"][frame_idx].tolist()
            if "action_eef" in episode_data:
                wrist_pose = episode_data["action_eef"][frame_idx].tolist()

            t_now = time.monotonic()
            control_publisher.publish({
                "target_upper_body_pose": ub_pose.astype(np.float64),
                "wrist_pose": wrist_pose,
                "base_height_command": DEFAULT_BASE_HEIGHT,
                "navigate_cmd": nav_cmd,
                "timestamp": t_now,
                "target_time": t_now + dt,
            })

            if config.verbose and frame_idx % 30 == 0:
                elapsed_frames = frame_idx - start_frame
                total_frames = end_frame - start_frame
                progress = elapsed_frames / total_frames * 100
                print(
                    f"  [{progress:5.1f}%] Frame {frame_idx}/{end_frame} | "
                    f"L-hand: [{actions[frame_idx][lh_s]*1000:.0f}..{actions[frame_idx][lh_e-1]*1000:.0f}] | "
                    f"R-hand: [{actions[frame_idx][rh_s]*1000:.0f}..{actions[frame_idx][rh_e-1]*1000:.0f}]"
                )

            elapsed = time.monotonic() - t_loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")

    replayed = min(frame_idx + 1, end_frame) - start_frame
    print(f"\nReplay finished. Replayed {replayed} frames.")
    print("Holding final pose for 2s...")
    time.sleep(2.0)
    ros_manager.shutdown()


if __name__ == "__main__":
    config = tyro.cli(ReplayConfig)
    main(config)
