"""
G1 Real Robot Evaluation Deployment

This script deploys the finetuned GR00T model for evaluation on the real G1 robot.
It replaces the teleop policy with the GR00T inference policy.

Server-side (on machine with GPU):
    python gr00t/eval/run_gr00t_server.py \
        --model-path g1_finetuned/checkpoint-10000 \
        --embodiment-tag UNITREE_G1 \
        --device cuda \
        --host 0.0.0.0 \
        --port 5555

Client-side (on robot or connected machine):
    python scripts/deploy_g1_evaluation.py \
        --interface eno1 \
        --camera_host 192.168.123.164 \
        --model_host <server_ip> \
        --model_port 5555 \
        --task_description "pick and place the box on the platform" \
        --no-with_hands

Usage Notes:
- The model server should be started first on a machine with GPU
- The camera server should be running on the robot's companion computer
- This script starts the control loop and inference policy
"""

from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal, Optional

import tyro

from gr00t_wbc.control.main.teleop.configs.configs import DeploymentConfig
from gr00t_wbc.control.utils.run_real_checklist import show_deployment_checklist


@dataclass
class EvaluationConfig(DeploymentConfig):
    """Configuration for G1 evaluation deployment with GR00T model."""
    
    # Model server configuration
    model_host: str = "localhost"
    """Host address for the GR00T model server"""
    
    model_port: int = 5555
    """Port number for the GR00T model server"""
    
    # Task configuration
    task_description: str = "pick and place the box on the platform"
    """Task description/instruction for the model"""
    
    # Inference configuration
    inference_frequency: float = 20.0
    """Frequency of inference loop (Hz)"""
    
    n_action_steps: int = 10
    """Number of action steps to execute before re-querying the model"""
    
    # Override defaults for evaluation
    data_collection: bool = True
    """Enable data collection during evaluation"""
    
    enable_upper_body_operation: bool = True
    """Enable upper body operation (required for inference)"""
    
    upper_body_operation_mode: Literal["teleop", "inference"] = "inference"
    """Upper body operation mode - set to inference for model evaluation"""


class G1EvaluationDeployment:
    """
    Deployment manager for G1 robot evaluation with GR00T model.
    
    This is similar to G1Deployment but uses the inference policy instead of teleop.
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.project_root = Path(__file__).resolve().parent.parent
        self.session_name = "g1_evaluation"
        
        # Create tmux session
        self._create_tmux_session()
    
    def _create_tmux_session(self):
        """Create a new tmux session if it doesn't exist"""
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", self.session_name])
            print(f"Created new tmux session: {self.session_name}")
            
            # Set up the default window layout
            subprocess.run(
                ["tmux", "rename-window", "-t", f"{self.session_name}:0", "control_inference_data"]
            )
            # Split horizontally (left and right)
            subprocess.run(["tmux", "split-window", "-t", f"{self.session_name}:0", "-h"])
            # Split right pane vertically
            subprocess.run(["tmux", "split-window", "-t", f"{self.session_name}:0.1", "-v"])
            # Select left pane
            subprocess.run(["tmux", "select-pane", "-t", f"{self.session_name}:0.0"])
    
    def _run_in_tmux(self, name, cmd, wait_time=2, pane_index=None):
        """Run a command in a tmux window or pane"""
        if pane_index is not None:
            target = f"{self.session_name}:0.{pane_index}"
        else:
            subprocess.run(["tmux", "new-window", "-t", self.session_name, "-n", name])
            target = f"{self.session_name}:{name}"
        
        trap_cmd = f"trap 'tmux kill-session -t {self.session_name}' QUIT"
        env_cmd = f"export GR00T_WBC_TMUX_SESSION={self.session_name}"
        cmd_str = " ".join(str(x) for x in cmd)
        full_cmd = f"{trap_cmd}; {env_cmd}; {cmd_str}"
        
        subprocess.run(["tmux", "send-keys", "-t", target, full_cmd, "C-m"])
        time.sleep(wait_time)
        
        result = subprocess.run(
            ["tmux", "list-panes", "-t", target, "-F", "#{pane_dead}"],
            capture_output=True,
            text=True,
        )
        
        if result.stdout.strip() == "1":
            print(f"ERROR: {name} failed to start!")
            return False
        
        return True
    
    def start_camera_viewer(self):
        """Start the camera viewer"""
        if not self.config.view_camera:
            return
        
        print("Starting camera viewer...")
        cmd = [
            sys.executable,
            str(self.project_root / "gr00t_wbc/control/main/teleop/run_camera_viewer.py"),
            "--camera_host", self.config.camera_host,
            "--camera_port", str(self.config.camera_port),
            "--fps", str(self.config.fps),
        ]
        
        if not self._run_in_tmux("camera_viewer", cmd):
            print("ERROR: Camera viewer failed to start!")
            print("Continuing without camera viewer...")
        else:
            print("Camera viewer started successfully.")
    
    def start_control_loop(self):
        """Start the G1 control loop"""
        print("Starting G1 control loop...")
        cmd = [
            sys.executable,
            str(self.project_root / "gr00t_wbc/control/main/teleop/run_g1_control_loop.py"),
            "--wbc_version", self.config.wbc_version,
            "--wbc_model_path", self.config.wbc_model_path,
            "--wbc_policy_class", self.config.wbc_policy_class,
            "--interface", self.config.interface,
            "--simulator", "None",
            "--control_frequency", str(self.config.control_frequency),
        ]
        
        # Handle boolean flags
        if self.config.enable_waist:
            cmd.append("--enable_waist")
        else:
            cmd.append("--no-enable_waist")
        
        if self.config.with_hands:
            cmd.append("--with_hands")
        else:
            cmd.append("--no-with_hands")
        
        if self.config.high_elbow_pose:
            cmd.append("--high_elbow_pose")
        else:
            cmd.append("--no-high_elbow_pose")
        
        if self.config.enable_gravity_compensation:
            cmd.append("--enable_gravity_compensation")
            if self.config.gravity_compensation_joints:
                cmd.extend(["--gravity_compensation_joints"] + self.config.gravity_compensation_joints)
        else:
            cmd.append("--no-enable_gravity_compensation")
        
        if not self._run_in_tmux("control", cmd, wait_time=3, pane_index=0):
            print("ERROR: Control loop failed to start!")
            self.cleanup()
            sys.exit(1)
        
        print("Control loop started successfully.")
        print("Controls: 'i' for initial pose, ']' to activate locomotion")
    
    def start_inference_policy(self):
        """Start the GR00T inference policy"""
        print("Starting GR00T inference policy...")
        cmd = [
            sys.executable,
            str(self.project_root / "scripts/run_gr00t_inference_policy.py"),
            "--model_host", self.config.model_host,
            "--model_port", str(self.config.model_port),
            "--camera_host", self.config.camera_host,
            "--camera_port", str(self.config.camera_port),
            "--task_description", f'"{self.config.task_description}"',
            "--inference_frequency", str(self.config.inference_frequency),
            "--n_action_steps", str(self.config.n_action_steps),
        ]
        
        if self.config.enable_waist:
            cmd.append("--enable_waist")
        else:
            cmd.append("--no-enable_waist")
        
        if self.config.with_hands:
            cmd.append("--with_hands")
        else:
            cmd.append("--no-with_hands")
        
        if self.config.high_elbow_pose:
            cmd.append("--high_elbow_pose")
        else:
            cmd.append("--no-high_elbow_pose")
        
        if not self._run_in_tmux("inference", cmd, pane_index=2):
            print("ERROR: Inference policy failed to start!")
            print("Continuing without inference policy...")
        else:
            print("Inference policy started successfully.")
            print("Press 'l' in the control loop terminal to start policy execution.")
    
    def start_data_collection(self):
        """Start the data collection process"""
        if not self.config.data_collection:
            print("Data collection disabled in config.")
            return
        
        print("Starting data collection...")
        cmd = [
            sys.executable,
            str(self.project_root / "gr00t_wbc/control/main/teleop/run_g1_data_exporter.py"),
            "--data_collection_frequency", str(self.config.data_collection_frequency),
            "--root_output_dir", self.config.root_output_dir,
            "--lower_body_policy", self.config.wbc_version,
            "--wbc_model_path", self.config.wbc_model_path,
            "--camera_host", self.config.camera_host,
            "--camera_port", str(self.config.camera_port),
        ]
        
        if self.config.add_stereo_camera:
            cmd.append("--add_stereo_camera")
        else:
            cmd.append("--no-add_stereo_camera")
        
        if not self._run_in_tmux("data", cmd, pane_index=1):
            print("ERROR: Data collection failed to start!")
            print("Continuing without data collection...")
        else:
            print("Data collection started successfully.")
            print("Press 'c' in the control loop terminal to start/stop recording data.")
    
    def deploy(self):
        """Run the complete deployment process"""
        print("\n" + "="*60)
        print("G1 EVALUATION DEPLOYMENT")
        print("="*60)
        print(f"Model Server: {self.config.model_host}:{self.config.model_port}")
        print(f"Camera Server: {self.config.camera_host}:{self.config.camera_port}")
        print(f"Task: {self.config.task_description}")
        print(f"Robot IP: {self.config.robot_ip}")
        print(f"WBC Version: {self.config.wbc_version}")
        print(f"Interface: {self.config.interface}")
        print(f"With Hands: {self.config.with_hands}")
        print(f"Enable Waist: {self.config.enable_waist}")
        print(f"Inference Frequency: {self.config.inference_frequency} Hz")
        print(f"Action Steps: {self.config.n_action_steps}")
        print("="*60 + "\n")
        
        # Safety checklist for real robot
        if self.config.env_type == "real":
            if not show_deployment_checklist():
                sys.exit(1)
        
        # Register signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Start components in sequence
        self.start_control_loop()
        self.start_camera_viewer()
        self.start_inference_policy()
        self.start_data_collection()
        
        print("\n--- G1 EVALUATION DEPLOYMENT COMPLETE ---")
        print(f"All systems running in tmux session: {self.session_name}")
        print("Press Ctrl+b then d to detach from the session")
        print("Press Ctrl+\\ in any window to shutdown all components.")
        print("\n")
        print("IMPORTANT: Before starting inference:")
        print("  1. Press 'i' in control window to move to initial pose")
        print("  2. Press ']' to activate locomotion")
        print("  3. Press 'l' to start receiving inference commands")
        print("  4. Press 'c' to start/stop recording (optional)")
        
        try:
            subprocess.run([
                "tmux", "attach", "-t", self.session_name,
                ";", "select-window", "-t", "control_inference_data"
            ])
        except KeyboardInterrupt:
            print("\nShutdown requested...")
            self.cleanup()
            sys.exit(0)
        
        try:
            while True:
                result = subprocess.run(
                    ["tmux", "has-session", "-t", self.session_name],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print("Tmux session terminated. Exiting.")
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up tmux session"""
        print("Cleaning up tmux session...")
        try:
            subprocess.run(["tmux", "kill-session", "-t", self.session_name], timeout=5)
            print("Tmux session terminated successfully.")
        except subprocess.TimeoutExpired:
            print("Warning: Tmux session termination timed out, forcing kill...")
            subprocess.run(["tmux", "kill-session", "-t", self.session_name, "-9"])
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        print("Cleanup complete.")
    
    def signal_handler(self, sig, frame):
        """Handle SIGINT gracefully"""
        print("\nShutdown signal received...")
        self.cleanup()
        sys.exit(0)


def main():
    """Main entry point"""
    config = tyro.cli(EvaluationConfig)
    deployment = G1EvaluationDeployment(config)
    deployment.deploy()


if __name__ == "__main__":
    main()
