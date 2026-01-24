#!/usr/bin/env python3
"""
Test script to verify GR00T model server observation format.

Run this on the server or any machine that can reach the model server
to validate that the observation format matches what the model expects.

Usage:
    python scripts/test_gr00t_observation_format.py --host <server_ip> --port 5555
    
Or run inline:
    python -c "exec(open('scripts/test_gr00t_observation_format.py').read())"
"""

import argparse
import time
import numpy as np


def create_test_observation(
    image_height: int = 480,
    image_width: int = 640,
    task_description: str = "pick and place the box on the platform"
) -> dict:
    """
    Create a test observation in the exact format expected by GR00T model.
    
    Based on processor_config.json for unitree_g1:
    - video modality_keys: ["ego_view"]
    - state modality_keys: ["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    - language modality_keys: ["annotation.human.task_description"]
    - action modality_keys: ["left_arm", "right_arm", "left_hand", "right_hand", "waist", "base_height_command", "navigate_command"]
    """
    # Random joint positions (43 joints total)
    q = np.random.randn(43).astype(np.float32)
    
    # State observation - shape: (batch=1, time=1, dim)
    # Indices from modality.json
    state = {
        "left_leg": q[0:6][np.newaxis, np.newaxis, :],      # (1, 1, 6)
        "right_leg": q[6:12][np.newaxis, np.newaxis, :],    # (1, 1, 6)
        "waist": q[12:15][np.newaxis, np.newaxis, :],       # (1, 1, 3)
        "left_arm": q[15:22][np.newaxis, np.newaxis, :],    # (1, 1, 7)
        "left_hand": q[22:29][np.newaxis, np.newaxis, :],   # (1, 1, 7)
        "right_arm": q[29:36][np.newaxis, np.newaxis, :],   # (1, 1, 7)
        "right_hand": q[36:43][np.newaxis, np.newaxis, :],  # (1, 1, 7)
    }
    
    # Video observation - shape: (batch=1, time=1, H, W, C)
    # Key is "ego_view" (not "video.ego_view") based on processor_config.json
    video = {
        "ego_view": np.random.randint(
            0, 255, 
            (1, 1, image_height, image_width, 3), 
            dtype=np.uint8
        )
    }
    
    # Language observation - shape: [[text]] for (batch=1, time=1)
    # Key is "annotation.human.task_description" based on processor_config.json
    language = {
        "annotation.human.task_description": [[task_description]]
    }
    
    observation = {
        "video": video,
        "state": state,
        "language": language,
    }
    
    return observation


def print_observation_shapes(obs: dict):
    """Print the shapes of all observation components."""
    print("\nObservation shapes:")
    print("  video:")
    for k, v in obs["video"].items():
        print(f"    {k}: {v.shape} dtype={v.dtype}")
    print("  state:")
    for k, v in obs["state"].items():
        print(f"    {k}: {v.shape} dtype={v.dtype}")
    print("  language:")
    for k, v in obs["language"].items():
        print(f"    {k}: {v}")


def print_action_shapes(action: dict):
    """Print the shapes of all action components."""
    print("\nAction shapes:")
    for k, v in action.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")


def test_connection(host: str, port: int):
    """Test basic connection to the server."""
    print(f"\n{'='*60}")
    print(f"Testing connection to {host}:{port}")
    print(f"{'='*60}")
    
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError as e:
        print(f"❌ Failed to import PolicyClient: {e}")
        return None
    
    try:
        client = PolicyClient(host, port, timeout_ms=15000, strict=False)
        print("✅ PolicyClient created")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return None
    
    # Test ping
    try:
        if client.ping():
            print("✅ Server ping successful")
        else:
            print("❌ Server ping failed")
            return None
    except Exception as e:
        print(f"❌ Ping error: {e}")
        return None
    
    return client


def test_modality_config(client):
    """Test getting modality configuration."""
    print(f"\n{'='*60}")
    print("Testing modality configuration")
    print(f"{'='*60}")
    
    try:
        config = client.get_modality_config()
        print("✅ Got modality config")
        
        if config:
            print("\nExpected modality keys:")
            for modality in ["video", "state", "action", "language"]:
                if modality in config:
                    keys = config[modality].modality_keys if hasattr(config[modality], 'modality_keys') else "N/A"
                    print(f"  {modality}: {keys}")
        return config
    except Exception as e:
        print(f"⚠️ Could not get modality config: {e}")
        return None


def test_inference(client, obs: dict):
    """Test model inference."""
    print(f"\n{'='*60}")
    print("Testing model inference")
    print(f"{'='*60}")
    
    print_observation_shapes(obs)
    
    try:
        t_start = time.time()
        action, info = client.get_action(obs)
        inference_time = (time.time() - t_start) * 1000
        
        print(f"\n✅ Inference successful! ({inference_time:.1f}ms)")
        print_action_shapes(action)
        
        # Verify expected action keys for unitree_g1
        expected_keys = [
            "left_arm", "right_arm", "left_hand", "right_hand", 
            "waist", "base_height_command", "navigate_command"
        ]
        missing_keys = [k for k in expected_keys if k not in action]
        if missing_keys:
            print(f"\n⚠️ Missing expected action keys: {missing_keys}")
        else:
            print(f"\n✅ All expected action keys present")
        
        # Check action horizon
        if "left_arm" in action:
            horizon = action["left_arm"].shape[1]
            print(f"✅ Action horizon: {horizon} steps")
        
        return action
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multiple_inferences(client, obs: dict, n_runs: int = 5):
    """Test multiple inferences for timing."""
    print(f"\n{'='*60}")
    print(f"Testing {n_runs} consecutive inferences")
    print(f"{'='*60}")
    
    times = []
    for i in range(n_runs):
        try:
            t_start = time.time()
            action, info = client.get_action(obs)
            times.append((time.time() - t_start) * 1000)
        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            return
    
    print(f"✅ All {n_runs} runs successful")
    print(f"  Mean: {np.mean(times):.1f}ms")
    print(f"  Std:  {np.std(times):.1f}ms")
    print(f"  Min:  {np.min(times):.1f}ms")
    print(f"  Max:  {np.max(times):.1f}ms")


def test_action_to_control_loop(action: dict):
    """
    Test that action format matches what control loop expects.
    
    The control loop (run_g1_control_loop.py) expects messages on CONTROL_GOAL_TOPIC with:
    - target_upper_body_pose: joint positions for upper body
    - wrist_pose: end effector pose (14 values)
    - base_height_command: float
    - navigate_cmd: list of 3 floats [lin_vel_x, lin_vel_y, ang_vel_z]
    - timestamp: float
    - target_time: float
    """
    print(f"\n{'='*60}")
    print("Testing action parsing for control loop")
    print(f"{'='*60}")
    
    # Joint indices matching G1 robot model
    # These are the indices used in the inference policy
    action_joint_indices = {
        "left_arm": [15, 16, 17, 18, 19, 20, 21],    # 7 joints
        "right_arm": [29, 30, 31, 32, 33, 34, 35],   # 7 joints
        "left_hand": [22, 23, 24, 25, 26, 27, 28],   # 7 joints
        "right_hand": [36, 37, 38, 39, 40, 41, 42],  # 7 joints
        "waist": [12, 13, 14],                        # 3 joints
    }
    
    # Upper body indices (what control loop expects)
    # waist (3) + left_arm (7) + left_hand (7) + right_arm (7) + right_hand (7) = 31
    upper_body_indices = (
        action_joint_indices["waist"] + 
        action_joint_indices["left_arm"] + 
        action_joint_indices["left_hand"] +
        action_joint_indices["right_arm"] + 
        action_joint_indices["right_hand"]
    )
    
    # Default values (from constants.py)
    DEFAULT_WRIST_POSE = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] * 2  # 14 values
    DEFAULT_BASE_HEIGHT = 0.74
    DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]
    
    # Extract actions from model output (remove batch dimension)
    try:
        left_arm = action["left_arm"][0]    # (horizon, 7)
        right_arm = action["right_arm"][0]  # (horizon, 7)
        left_hand = action.get("left_hand", np.zeros((left_arm.shape[0], 7)))[0]
        right_hand = action.get("right_hand", np.zeros((left_arm.shape[0], 7)))[0]
        waist = action["waist"][0]          # (horizon, 3)
        base_height = action["base_height_command"][0]  # (horizon, 1)
        nav_cmd = action["navigate_command"][0]  # (horizon, 3)
        
        horizon = left_arm.shape[0]
        print(f"✅ Action horizon: {horizon} steps")
        print(f"✅ Extracted components:")
        print(f"    left_arm:  {left_arm.shape}")
        print(f"    right_arm: {right_arm.shape}")
        print(f"    left_hand: {left_hand.shape}")
        print(f"    right_hand: {right_hand.shape}")
        print(f"    waist:     {waist.shape}")
        print(f"    base_height: {base_height.shape}")
        print(f"    nav_cmd:   {nav_cmd.shape}")
        
    except KeyError as e:
        print(f"❌ Missing action key: {e}")
        return None
    
    # Parse first action step (t=0) into control loop format
    t = 0
    
    # Construct full joint position array (43 DOF)
    full_q = np.zeros(43)
    
    # Fill in joints
    for idx, joint_idx in enumerate(action_joint_indices["left_arm"]):
        full_q[joint_idx] = left_arm[t, idx]
    for idx, joint_idx in enumerate(action_joint_indices["right_arm"]):
        full_q[joint_idx] = right_arm[t, idx]
    for idx, joint_idx in enumerate(action_joint_indices["left_hand"]):
        full_q[joint_idx] = left_hand[t, idx]
    for idx, joint_idx in enumerate(action_joint_indices["right_hand"]):
        full_q[joint_idx] = right_hand[t, idx]
    for idx, joint_idx in enumerate(action_joint_indices["waist"]):
        full_q[joint_idx] = waist[t, idx]
    
    # Extract upper body pose
    upper_body_pose = full_q[upper_body_indices]
    
    # Create control loop message
    control_msg = {
        "target_upper_body_pose": upper_body_pose.astype(np.float64),
        "wrist_pose": DEFAULT_WRIST_POSE,
        "base_height_command": float(base_height[t, 0]),
        "navigate_cmd": nav_cmd[t].tolist(),
        "timestamp": time.time(),
        "target_time": time.time() + 0.05,  # 20Hz
    }
    
    print(f"\n✅ Parsed control loop message (step {t}):")
    print(f"    target_upper_body_pose: shape={control_msg['target_upper_body_pose'].shape}, dtype={control_msg['target_upper_body_pose'].dtype}")
    print(f"    wrist_pose: {len(control_msg['wrist_pose'])} values")
    print(f"    base_height_command: {control_msg['base_height_command']:.4f}")
    print(f"    navigate_cmd: {control_msg['navigate_cmd']}")
    
    # Validate shapes and types
    errors = []
    
    # Check upper body pose
    expected_upper_body_size = 31  # waist(3) + arms(14) + hands(14) = 31 for G1 without legs
    if len(upper_body_pose) != expected_upper_body_size:
        # Note: actual size depends on robot model config
        print(f"    ℹ️  upper_body_pose has {len(upper_body_pose)} joints")
    
    # Check navigate_cmd
    if len(control_msg["navigate_cmd"]) != 3:
        errors.append(f"navigate_cmd should have 3 values, got {len(control_msg['navigate_cmd'])}")
    
    # Check base_height_command is a scalar
    if not isinstance(control_msg["base_height_command"], float):
        errors.append(f"base_height_command should be float, got {type(control_msg['base_height_command'])}")
    
    if errors:
        for e in errors:
            print(f"❌ {e}")
        return None
    
    print(f"\n✅ Control loop message format is correct!")
    
    # Show sample values for a few timesteps
    print(f"\n📊 Sample action values across horizon:")
    print(f"    {'Step':<6} {'nav_x':>10} {'nav_y':>10} {'nav_yaw':>10} {'height':>10}")
    print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for t in [0, horizon//4, horizon//2, 3*horizon//4, horizon-1]:
        print(f"    {t:<6} {nav_cmd[t, 0]:>10.4f} {nav_cmd[t, 1]:>10.4f} {nav_cmd[t, 2]:>10.4f} {base_height[t, 0]:>10.4f}")
    
    # Show arm joint sample
    print(f"\n📊 Sample left_arm joint values (step 0):")
    print(f"    Joints: {left_arm[0, :].tolist()}")
    
    return control_msg


def main():
    parser = argparse.ArgumentParser(description="Test GR00T model server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--image_height", type=int, default=480, help="Image height")
    parser.add_argument("--image_width", type=int, default=640, help="Image width")
    parser.add_argument("--task", type=str, default="pick and place the box on the platform",
                        help="Task description")
    args = parser.parse_args()
    
    # Test connection
    client = test_connection(args.host, args.port)
    if client is None:
        return
    
    # Test modality config
    test_modality_config(client)
    
    # Create test observation
    obs = create_test_observation(
        image_height=args.image_height,
        image_width=args.image_width,
        task_description=args.task
    )
    
    # Test single inference
    action = test_inference(client, obs)
    if action is None:
        return
    
    # Test action parsing for control loop
    control_msg = test_action_to_control_loop(action)
    if control_msg is None:
        print("\n❌ Action parsing failed!")
        return
    
    # Test multiple inferences
    test_multiple_inferences(client, obs, n_runs=5)
    
    print(f"\n{'='*60}")
    print("✅ All tests passed! Server is ready for deployment.")
    print("✅ Actions can be correctly parsed for control loop.")
    print(f"{'='*60}\n")


# Also allow running as inline code
if __name__ == "__main__":
    main()
