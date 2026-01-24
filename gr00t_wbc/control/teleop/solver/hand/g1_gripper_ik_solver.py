import numpy as np
from gr00t_wbc.control.teleop.solver.solver import Solver

class G1GripperInverseKinematicsSolver(Solver):
    def __init__(self, side) -> None:
        self.side = "L" if side.lower() == "left" else "R"
        self.tip_indices = [4, 9, 14, 19, 24]  # Thumb, Index, Middle, Ring, Pinky
        
        # --- FIX: Tighter Calibration for Easier Opening ---
        # Lowering 'open_thresholds' means you don't have to stretch as far to get 1.0
        self.open_thresholds =  {"thumb": 0.09, "fingers": 0.10} 
        
        # Raising 'close_thresholds' slightly ensures it snaps to 0.0 (Closed) tightly
        self.close_thresholds = {"thumb": 0.05, "fingers": 0.06}

    def register_robot(self, robot):
        pass

    def __call__(self, finger_data):
        q_desired = np.zeros(7) 

        fingertips = finger_data["position"]
        
        # Check if valid data
        if np.allclose(fingertips, 0):
            q_desired[:6] = 1.0 # Default to open if lost tracking
            return q_desired

        # Extract positions
        positions = np.array([finger[:3, 3] for finger in fingertips]).reshape(-1, 3)
        wrist_pos = positions[0]

        def get_curl(tip_idx, is_thumb=False):
            tip_pos = positions[tip_idx]
            dist = np.linalg.norm(tip_pos - wrist_pos)
            
            t_open = self.open_thresholds["thumb" if is_thumb else "fingers"]
            t_close = self.close_thresholds["thumb" if is_thumb else "fingers"]
            
            # Calculate ratio
            val = (dist - t_close) / (t_open - t_close)
            
            # --- Optional: Add a small multiplier gain (1.2x) to make it snappier ---
            val = val * 1.1 
            
            return np.clip(val, 0.0, 1.0)

        # Calculate Curls
        thumb_curl = get_curl(self.tip_indices[0], is_thumb=True)
        index_curl = get_curl(self.tip_indices[1])
        middle_curl = get_curl(self.tip_indices[2])
        ring_curl = get_curl(self.tip_indices[3])
        pinky_curl = get_curl(self.tip_indices[4])

        # Map to Inspire Hand Motors (0-5)
        q_desired[0] = pinky_curl
        q_desired[1] = ring_curl
        q_desired[2] = middle_curl
        q_desired[3] = index_curl
        q_desired[4] = thumb_curl
        q_desired[5] = 1.0  # Keep thumb rotation open
        
        # Debug Print (Uncomment to tune if still stuck)
        # if not np.allclose(q_desired[:5], 1.0):
        #     print(f"[{self.side}] Dist: {np.linalg.norm(positions[9]-wrist_pos):.3f} -> CMD: {index_curl:.2f}")

        return q_desired