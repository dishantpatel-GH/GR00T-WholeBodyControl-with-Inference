import threading
import time
import numpy as np

# Use the message types from your original source
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

# Constants for Inspire Hands
TOPIC_CMD = "rt/inspire/cmd"
TOPIC_STATE = "rt/inspire/state"
NUM_MOTORS_PER_HAND = 6
TOTAL_MOTORS = 12  # 0-5 Right, 6-11 Left

class InspireHandDriver:
    """
    Singleton driver that handles the single DDS topic for BOTH hands.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, interface="eno1"): # Added interface arg
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InspireHandDriver, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, interface="eno1"):
        if self._initialized:
            return

        print("[InspireHandDriver] Initializing Shared DDS Driver...")
        
        # --- ROBUST INIT START ---
        # Try to initialize ChannelFactory only if it hasn't been done globally
        # We pass a distinct domain ID (e.g., 0) and the interface.
        try:
            # Domain 0 is standard for Unitree. 
            # We wrap this in try/except because if G1Env already did it, this might throw.
            ChannelFactoryInitialize(0, interface)
            print(f"[InspireHandDriver] Channel Factory Initialized on {interface}")
        except Exception as e:
            # If it fails, it usually means it was already initialized by G1Env. 
            # We verify connectivity by proceeding.
            print(f"[InspireHandDriver] Channel Factory already active or failed (Warning: {e})")
        # --- ROBUST INIT END ---

        # Internal State Storage (12 motors)
        self.joint_q = np.zeros(TOTAL_MOTORS)
        self.joint_dq = np.zeros(TOTAL_MOTORS)
        self.joint_tau = np.zeros(TOTAL_MOTORS)
        
        # Internal Command Storage
        self.cmd_q = np.ones(TOTAL_MOTORS) # Default to open/safe position
        self.cmd_lock = threading.Lock()

        # DDS Initialization
        self.pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
        self.pub.Init()
        self.sub = ChannelSubscriber(TOPIC_STATE, MotorStates_)
        self.sub.Init()

        # Start background threads
        self.running = True
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()
        
        self._initialized = True
        print("[InspireHandDriver] Driver Fully Initialized.")

    def _recv_loop(self):
        while self.running:
            msg = self.sub.Read()
            if msg is not None:
                # The msg.states is a list. Ensure we don't go out of bounds.
                count = min(len(msg.states), TOTAL_MOTORS)
                for i in range(count):
                    self.joint_q[i] = msg.states[i].q
                    self.joint_dq[i] = msg.states[i].dq
                    self.joint_tau[i] = msg.states[i].tau_est
            time.sleep(0.002) # ~500Hz receive

    def _send_loop(self):
        msg = MotorCmds_()
        msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(TOTAL_MOTORS)]
        
        while self.running:
            with self.cmd_lock:
                for i in range(TOTAL_MOTORS):
                    msg.cmds[i].mode = 1  # Servo Mode
                    msg.cmds[i].q = self.cmd_q[i]
                    msg.cmds[i].kp = 0.0  # Usually internal to hand
                    msg.cmds[i].kd = 0.0
            
            self.pub.Write(msg)
            time.sleep(0.01) # 100Hz publish

    def get_hand_state(self, is_left: bool):
        """Returns (q, dq, tau) for the specific hand (6 DOFs)"""
        start_idx = 6 if is_left else 0
        end_idx = start_idx + 6
        return (
            self.joint_q[start_idx:end_idx],
            self.joint_dq[start_idx:end_idx],
            self.joint_tau[start_idx:end_idx]
        )

    def set_hand_cmd(self, is_left: bool, q_cmd: np.ndarray):
        """Updates command for specific hand"""
        start_idx = 6 if is_left else 0
        # Valid range check
        limit = min(len(q_cmd), 6)
        
        with self.cmd_lock:
            for i in range(limit):
                self.cmd_q[start_idx + i] = q_cmd[i]