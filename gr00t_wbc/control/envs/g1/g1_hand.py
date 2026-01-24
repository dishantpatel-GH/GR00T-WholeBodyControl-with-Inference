import time
import gymnasium as gym
import numpy as np
from gr00t_wbc.control.base.env import Env

# Import the Standard Dex3 classes
from gr00t_wbc.control.envs.g1.utils.command_sender import HandCommandSender
from gr00t_wbc.control.envs.g1.utils.state_processor import HandStateProcessor

# Import the New Inspire classes
from gr00t_wbc.control.envs.g1.utils.command_sender import InspireHandCommandSender
from gr00t_wbc.control.envs.g1.utils.state_processor import InspireHandStateProcessor


class G1ThreeFingerHand(Env):
    # ... (Keep existing implementation unchanged) ...
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        self.hand_state_processor = HandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = HandCommandSender(is_left=self.is_left)
        self.hand_q_offset = np.zeros(7)

    def observe(self) -> dict[str, any]:
        hand_state = self.hand_state_processor._prepare_low_state()  # (1, 28)
        assert hand_state.shape == (1, 28)

        # Apply offset to the hand state
        hand_state[0, :7] = hand_state[0, :7] + self.hand_q_offset

        hand_q = hand_state[0, :7]
        hand_dq = hand_state[0, 7:14]
        hand_ddq = hand_state[0, 21:28]
        hand_tau_est = hand_state[0, 14:21]

        # Return the state for this specific hand (left or right)
        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # Apply offset to the hand target
        action["hand_q"] = action["hand_q"] - self.hand_q_offset

        # action should contain hand_q
        self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict({"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))})

    def calibrate_hand(self):
        # ... (Keep existing implementation) ...
        pass


class G1InspireHand(Env):
    def __init__(self, is_left: bool = True):
        super().__init__()
        self.is_left = is_left
        # Use the Inspire specific processors
        self.hand_state_processor = InspireHandStateProcessor(is_left=self.is_left)
        self.hand_command_sender = InspireHandCommandSender(is_left=self.is_left)
        # Inspire hands are usually pre-calibrated, but we keep structure
        self.hand_q_offset = np.zeros(7) 

    def observe(self) -> dict[str, any]:
        # Processor returns padded array of shape (1, 28)
        hand_state = self.hand_state_processor._prepare_low_state() 

        # We can add offset logic here if needed, usually not for Inspire
        hand_q = hand_state[0, :7]
        hand_dq = hand_state[0, 7:14]
        hand_ddq = hand_state[0, 21:28]
        hand_tau_est = hand_state[0, 14:21]

        return {
            "hand_q": hand_q,
            "hand_dq": hand_dq,
            "hand_ddq": hand_ddq,
            "hand_tau_est": hand_tau_est,
        }

    def queue_action(self, action: dict[str, any]):
        # Send to Inspire Sender (which talks to Driver)
        if "hand_q" in action:
            self.hand_command_sender.send_command(action["hand_q"])

    def observation_space(self) -> gym.Space:
        # Maintaining 7-dim shape for compatibility
        return gym.spaces.Dict(
            {
                "hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_dq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_ddq": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
                "hand_tau_est": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            }
        )

    def action_space(self) -> gym.Space:
        return gym.spaces.Dict({"hand_q": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))})

    def calibrate_hand(self):
        print(f"Inspire Hand ({'Left' if self.is_left else 'Right'}) calibration not required.")