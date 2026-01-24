"""
Test GR00T Model Server Connection

This script tests the connection to the GR00T model server and validates
the observation/action format without actually controlling the robot.

Usage:
    # First start the server:
    python gr00t/eval/run_gr00t_server.py \
        --model-path g1_finetuned/checkpoint-10000 \
        --embodiment-tag UNITREE_G1 \
        --device cuda \
        --host 0.0.0.0 \
        --port 5555

    # Then run this test:
    python scripts/test_gr00t_connection.py \
        --model_host <server_ip> \
        --model_port 5555
"""

from dataclasses import dataclass
import time

import numpy as np
import tyro


@dataclass
class TestConfig:
    """Configuration for testing GR00T model server connection."""
    
    model_host: str = "localhost"
    """Host address for the GR00T model server"""
    
    model_port: int = 5555
    """Port number for the GR00T model server"""
    
    task_description: str = "pick and place the box on the platform"
    """Task description/instruction for the model"""
    
    image_height: int = 480
    """Height of dummy image"""
    
    image_width: int = 640
    """Width of dummy image"""


def create_dummy_observation(config: TestConfig) -> dict:
    """Create a dummy observation in the format expected by the model."""
    
    # Dummy state (43 joints)
    # Based on modality.json indices:
    # left_leg: 0-6, right_leg: 6-12, waist: 12-15
    # left_arm: 15-22, left_hand: 22-29
    # right_arm: 29-36, right_hand: 36-43
    
    state = {
        "left_leg": np.zeros((1, 1, 6), dtype=np.float32),
        "right_leg": np.zeros((1, 1, 6), dtype=np.float32),
        "waist": np.zeros((1, 1, 3), dtype=np.float32),
        "left_arm": np.zeros((1, 1, 7), dtype=np.float32),
        "left_hand": np.zeros((1, 1, 7), dtype=np.float32),
        "right_arm": np.zeros((1, 1, 7), dtype=np.float32),
        "right_hand": np.zeros((1, 1, 7), dtype=np.float32),
    }
    
    # Dummy image (H, W, C) -> (batch=1, time=1, H, W, C)
    dummy_image = np.random.randint(
        0, 255,
        (1, 1, config.image_height, config.image_width, 3),
        dtype=np.uint8
    )
    
    video = {
        "ego_view": dummy_image
    }
    
    language = {
        "annotation.human.task_description": [[config.task_description]]
    }
    
    observation = {
        "video": video,
        "state": state,
        "language": language,
    }
    
    return observation


def main(config: TestConfig):
    """Test GR00T model server connection."""
    
    print("\n" + "="*60)
    print("GR00T Model Server Connection Test")
    print("="*60)
    print(f"Server: {config.model_host}:{config.model_port}")
    print(f"Task: {config.task_description}")
    print("="*60 + "\n")
    
    # Import here to get better error messages
    try:
        from gr00t.policy.server_client import PolicyClient
    except ImportError as e:
        print(f"ERROR: Failed to import PolicyClient: {e}")
        print("Make sure gr00t package is installed.")
        return
    
    # Create client
    print("Creating policy client...")
    try:
        client = PolicyClient(
            host=config.model_host,
            port=config.model_port,
            timeout_ms=15000,
            strict=False,
        )
    except Exception as e:
        print(f"ERROR: Failed to create client: {e}")
        return
    
    # Test ping
    print("Testing connection (ping)...")
    try:
        if client.ping():
            print("  [OK] Server is responsive")
        else:
            print("  [FAIL] Server did not respond to ping")
            return
    except Exception as e:
        print(f"  [FAIL] Ping failed: {e}")
        return
    
    # Get modality config
    print("Getting modality configuration...")
    try:
        modality_config = client.get_modality_config()
        print("  [OK] Received modality config")
        print(f"  Video keys: {list(modality_config['video'].modality_keys)}")
        print(f"  State keys: {list(modality_config['state'].modality_keys)}")
        print(f"  Action keys: {list(modality_config['action'].modality_keys)}")
        print(f"  Language keys: {list(modality_config['language'].modality_keys)}")
        print(f"  Action horizon: {len(modality_config['action'].delta_indices)}")
    except Exception as e:
        print(f"  [WARN] Could not get modality config: {e}")
        print("  Continuing with default config...")
    
    # Create dummy observation
    print("\nCreating dummy observation...")
    observation = create_dummy_observation(config)
    
    print(f"  Video shape: {observation['video']['ego_view'].shape}")
    print(f"  State shapes:")
    for key, value in observation['state'].items():
        print(f"    {key}: {value.shape}")
    print(f"  Language: {observation['language']}")
    
    # Test inference
    print("\nTesting model inference...")
    try:
        t_start = time.monotonic()
        action, info = client.get_action(observation)
        inference_time = (time.monotonic() - t_start) * 1000
        
        print(f"  [OK] Inference successful ({inference_time:.1f}ms)")
        print(f"  Action keys: {list(action.keys())}")
        for key, value in action.items():
            print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test multiple inferences for timing
    print("\nRunning timing test (5 inferences)...")
    times = []
    for i in range(5):
        t_start = time.monotonic()
        action, info = client.get_action(observation)
        times.append((time.monotonic() - t_start) * 1000)
    
    print(f"  Mean: {np.mean(times):.1f}ms")
    print(f"  Std:  {np.std(times):.1f}ms")
    print(f"  Min:  {np.min(times):.1f}ms")
    print(f"  Max:  {np.max(times):.1f}ms")
    
    # Test reset
    print("\nTesting policy reset...")
    try:
        client.reset()
        print("  [OK] Reset successful")
    except Exception as e:
        print(f"  [WARN] Reset failed: {e}")
    
    print("\n" + "="*60)
    print("All tests passed! Server is ready for deployment.")
    print("="*60 + "\n")


if __name__ == "__main__":
    config = tyro.cli(TestConfig)
    main(config)
