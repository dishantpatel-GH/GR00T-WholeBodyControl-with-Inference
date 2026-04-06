import threading
import time
from dataclasses import dataclass

import numpy as np

# CycloneDDS for native Inspire hand DDS message types
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

# Unitree SDK for ChannelPublisher/Subscriber
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize


# --- Inspire FTP DDS message types (matching inspire_sdkpy/inspire_dds) ---
# These MUST match the IDL definitions used by Headless_driver_double.py
# (inspire_sdkpy), otherwise DDS discovery won't work.

@dataclass
@annotate.final
@annotate.autoid("sequential")
class InspireHandCtrlMsg(idl.IdlStruct, typename="inspire.inspire_hand_ctrl"):
    pos_set: types.sequence[types.int16, 6]
    angle_set: types.sequence[types.int16, 6]
    force_set: types.sequence[types.int16, 6]
    speed_set: types.sequence[types.int16, 6]
    mode: types.int8


@dataclass
@annotate.final
@annotate.autoid("sequential")
class InspireHandStateMsg(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act: types.sequence[types.int16, 6]
    angle_act: types.sequence[types.int16, 6]
    force_act: types.sequence[types.int16, 6]
    current: types.sequence[types.int16, 6]
    err: types.sequence[types.uint8, 6]
    status: types.sequence[types.uint8, 6]
    temperature: types.sequence[types.uint8, 6]


# DDS topics matching Headless_driver_double.py / inspire_sdkpy
TOPIC_CTRL_LEFT = "rt/inspire_hand/ctrl/l"
TOPIC_CTRL_RIGHT = "rt/inspire_hand/ctrl/r"
TOPIC_STATE_LEFT = "rt/inspire_hand/state/l"
TOPIC_STATE_RIGHT = "rt/inspire_hand/state/r"

NUM_MOTORS_PER_HAND = 6


class InspireHandDriver:
    """
    Singleton driver for Inspire FTP hands via DDS.

    Communicates with the Headless_driver_double.py (inspire_sdkpy) using
    the native Inspire DDS message types on the correct topics:
        - Command: rt/inspire_hand/ctrl/{l,r}  (inspire_hand_ctrl)
        - State:   rt/inspire_hand/state/{l,r}  (inspire_hand_state)
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, interface="eno1"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InspireHandDriver, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, interface="eno1"):
        if self._initialized:
            return

        print("[InspireHandDriver] Initializing DDS Driver (inspire_hand topics)...")

        # Try to initialize ChannelFactory (may already be done by G1Env)
        try:
            ChannelFactoryInitialize(0, interface)
            print(f"[InspireHandDriver] Channel Factory Initialized on {interface}")
        except Exception as e:
            print(f"[InspireHandDriver] Channel Factory already active (OK): {e}")

        # Internal state storage (6 DOF per hand)
        self.left_angle = np.zeros(NUM_MOTORS_PER_HAND)
        self.right_angle = np.zeros(NUM_MOTORS_PER_HAND)
        self.left_force = np.zeros(NUM_MOTORS_PER_HAND)
        self.right_force = np.zeros(NUM_MOTORS_PER_HAND)

        # Internal command storage (default: open position)
        self.left_cmd_angle = [1000] * NUM_MOTORS_PER_HAND
        self.right_cmd_angle = [1000] * NUM_MOTORS_PER_HAND
        self.cmd_lock = threading.Lock()

        # DDS publishers (one per hand, matching inspire_sdkpy topics)
        self.pub_l = ChannelPublisher(TOPIC_CTRL_LEFT, InspireHandCtrlMsg)
        self.pub_l.Init()
        self.pub_r = ChannelPublisher(TOPIC_CTRL_RIGHT, InspireHandCtrlMsg)
        self.pub_r.Init()

        # DDS subscribers (one per hand)
        self.sub_l = ChannelSubscriber(TOPIC_STATE_LEFT, InspireHandStateMsg)
        self.sub_l.Init()
        self.sub_r = ChannelSubscriber(TOPIC_STATE_RIGHT, InspireHandStateMsg)
        self.sub_r.Init()

        # Start background threads
        self.running = True
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

        self._initialized = True
        print("[InspireHandDriver] Driver Fully Initialized.")

    def _recv_loop(self):
        """Read hand state from DDS at ~500 Hz."""
        while self.running:
            msg_l = self.sub_l.Read()
            if msg_l is not None:
                self.left_angle[:] = msg_l.angle_act
                self.left_force[:] = msg_l.force_act

            msg_r = self.sub_r.Read()
            if msg_r is not None:
                self.right_angle[:] = msg_r.angle_act
                self.right_force[:] = msg_r.force_act

            time.sleep(0.002)  # ~500 Hz

    def _send_loop(self):
        """Send hand commands via DDS at ~100 Hz."""
        while self.running:
            with self.cmd_lock:
                cmd_l = InspireHandCtrlMsg(
                    pos_set=[0] * 6,
                    angle_set=list(self.left_cmd_angle),
                    force_set=[0] * 6,
                    speed_set=[0] * 6,
                    mode=0b0001,
                )
                cmd_r = InspireHandCtrlMsg(
                    pos_set=[0] * 6,
                    angle_set=list(self.right_cmd_angle),
                    force_set=[0] * 6,
                    speed_set=[0] * 6,
                    mode=0b0001,
                )

            self.pub_l.Write(cmd_l)
            self.pub_r.Write(cmd_r)
            time.sleep(0.01)  # 100 Hz

    def get_hand_state(self, is_left: bool):
        """
        Returns (q, dq, tau) for the specified hand (6 DOFs each).

        q = angle_act (Inspire FTP position, 0-1000 range)
        dq = zeros (Inspire FTP doesn't report velocity)
        tau = force_act
        """
        if is_left:
            q = self.left_angle.copy()
            tau = self.left_force.copy()
        else:
            q = self.right_angle.copy()
            tau = self.right_force.copy()

        dq = np.zeros(NUM_MOTORS_PER_HAND)
        return (q, dq, tau)

    def set_hand_cmd(self, is_left: bool, q_cmd: np.ndarray):
        """
        Update angle command for the specified hand.

        Args:
            is_left: True for left hand, False for right hand
            q_cmd: Array of target angles (first 6 values used, 0-1000 range)
        """
        limit = min(len(q_cmd), NUM_MOTORS_PER_HAND)

        with self.cmd_lock:
            if is_left:
                for i in range(limit):
                    self.left_cmd_angle[i] = int(q_cmd[i])
            else:
                for i in range(limit):
                    self.right_cmd_angle[i] = int(q_cmd[i])
