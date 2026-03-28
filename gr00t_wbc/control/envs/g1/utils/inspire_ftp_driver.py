import threading
import time
from dataclasses import dataclass

import numpy as np

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)

# ---------------------------------------------------------------------------
# Cyclone DDS IDL type definitions for Inspire FTP hands
# (Defined inline so there is no external dependency on inspire_sdkpy)
# ---------------------------------------------------------------------------


@dataclass
@annotate.final
@annotate.autoid("sequential")
class InspireFTPHandCtrl(idl.IdlStruct, typename="inspire.inspire_hand_ctrl"):
    pos_set: types.sequence[types.int16, 6]
    angle_set: types.sequence[types.int16, 6]
    force_set: types.sequence[types.int16, 6]
    speed_set: types.sequence[types.int16, 6]
    mode: types.int8


@dataclass
@annotate.final
@annotate.autoid("sequential")
class InspireFTPHandState(idl.IdlStruct, typename="inspire.inspire_hand_state"):
    pos_act: types.sequence[types.int16, 6]
    angle_act: types.sequence[types.int16, 6]
    force_act: types.sequence[types.int16, 6]
    current: types.sequence[types.int16, 6]
    err: types.sequence[types.uint8, 6]
    status: types.sequence[types.uint8, 6]
    temperature: types.sequence[types.uint8, 6]


# ---------------------------------------------------------------------------
# DDS topics used by the Headless_driver_double.py service
# ---------------------------------------------------------------------------

TOPIC_CMD_LEFT = "rt/inspire_hand/ctrl/l"
TOPIC_CMD_RIGHT = "rt/inspire_hand/ctrl/r"
TOPIC_STATE_LEFT = "rt/inspire_hand/state/l"
TOPIC_STATE_RIGHT = "rt/inspire_hand/state/r"

NUM_MOTORS_PER_HAND = 6


class InspireFTPHandDriver:
    """
    Singleton driver that manages BOTH Inspire FTP hands via separate DDS
    publishers/subscribers for the left and right hand.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, interface="eno1"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InspireFTPHandDriver, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, interface="eno1"):
        if self._initialized:
            return

        print("[InspireFTPHandDriver] Initializing Shared DDS Driver...")

        # Try to initialize ChannelFactory -- it may already be initialized by G1Env
        try:
            ChannelFactoryInitialize(0, interface)
            print(f"[InspireFTPHandDriver] Channel Factory Initialized on {interface}")
        except Exception as e:
            print(
                f"[InspireFTPHandDriver] Channel Factory already active or failed (Warning: {e})"
            )

        # ---- Internal state storage (per hand) ----
        self._state_lock = threading.Lock()
        self._left_angle_act = np.zeros(NUM_MOTORS_PER_HAND)
        self._right_angle_act = np.zeros(NUM_MOTORS_PER_HAND)

        # ---- Internal command storage (per hand) ----
        self._cmd_lock = threading.Lock()
        self._left_cmd_angle = np.zeros(NUM_MOTORS_PER_HAND, dtype=np.int16)
        self._right_cmd_angle = np.zeros(NUM_MOTORS_PER_HAND, dtype=np.int16)

        # ---- Publishers ----
        self._pub_left = ChannelPublisher(TOPIC_CMD_LEFT, InspireFTPHandCtrl)
        self._pub_left.Init()
        self._pub_right = ChannelPublisher(TOPIC_CMD_RIGHT, InspireFTPHandCtrl)
        self._pub_right.Init()

        # ---- Subscribers ----
        self._sub_left = ChannelSubscriber(TOPIC_STATE_LEFT, InspireFTPHandState)
        self._sub_left.Init()
        self._sub_right = ChannelSubscriber(TOPIC_STATE_RIGHT, InspireFTPHandState)
        self._sub_right.Init()

        # ---- Start background threads ----
        self.running = True
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._send_loop, daemon=True).start()

        self._initialized = True
        print("[InspireFTPHandDriver] Driver Fully Initialized.")

    # ------------------------------------------------------------------
    # Background receive loop (~500 Hz)
    # ------------------------------------------------------------------
    def _recv_loop(self):
        recv_count = 0
        left_recv = 0
        right_recv = 0
        while self.running:
            left_msg = self._sub_left.Read()
            right_msg = self._sub_right.Read()

            with self._state_lock:
                if left_msg is not None:
                    left_recv += 1
                    for i in range(NUM_MOTORS_PER_HAND):
                        self._left_angle_act[i] = left_msg.angle_act[i] / 1000.0
                if right_msg is not None:
                    right_recv += 1
                    for i in range(NUM_MOTORS_PER_HAND):
                        self._right_angle_act[i] = right_msg.angle_act[i] / 1000.0

            recv_count += 1
            if recv_count % 2500 == 0:  # ~every 5 seconds
                print(
                    f"[InspireFTPHandDriver] recv stats: left={left_recv} right={right_recv} "
                    f"L_state={self._left_angle_act} R_state={self._right_angle_act}"
                )

            time.sleep(0.002)  # ~500 Hz

    # ------------------------------------------------------------------
    # Background send loop (~100 Hz)
    # ------------------------------------------------------------------
    def _send_loop(self):
        send_count = 0
        while self.running:
            with self._cmd_lock:
                left_angles = self._left_cmd_angle.copy()
                right_angles = self._right_cmd_angle.copy()

            send_count += 1
            if send_count % 500 == 0:  # ~every 5 seconds
                print(
                    f"[InspireFTPHandDriver] send stats: count={send_count} "
                    f"L_cmd={left_angles} R_cmd={right_angles}"
                )

            # Build left command
            left_ctrl = InspireFTPHandCtrl(
                pos_set=[0] * NUM_MOTORS_PER_HAND,
                angle_set=left_angles.tolist(),
                force_set=[0] * NUM_MOTORS_PER_HAND,
                speed_set=[0] * NUM_MOTORS_PER_HAND,
                mode=0b0001,  # angle control mode
            )
            self._pub_left.Write(left_ctrl)

            # Build right command
            right_ctrl = InspireFTPHandCtrl(
                pos_set=[0] * NUM_MOTORS_PER_HAND,
                angle_set=right_angles.tolist(),
                force_set=[0] * NUM_MOTORS_PER_HAND,
                speed_set=[0] * NUM_MOTORS_PER_HAND,
                mode=0b0001,  # angle control mode
            )
            self._pub_right.Write(right_ctrl)

            time.sleep(0.01)  # ~100 Hz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_hand_state(self, is_left: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (q, dq, tau) for the specified hand (6 DOFs each).

        q   -- angle_act / 1000.0 (normalized 0-1 range)
        dq  -- zeros (not reported by Inspire FTP)
        tau -- zeros (not reported by Inspire FTP)
        """
        with self._state_lock:
            if is_left:
                q = self._left_angle_act.copy()
            else:
                q = self._right_angle_act.copy()

        dq = np.zeros(NUM_MOTORS_PER_HAND)
        tau = np.zeros(NUM_MOTORS_PER_HAND)
        return q, dq, tau

    def set_hand_cmd(self, is_left: bool, q_cmd: np.ndarray) -> None:
        """
        Send an angle command for the specified hand.

        q_cmd is in normalized 0-1 range; it is scaled by 1000 and cast to
        int16 for the DDS message.
        """
        limit = min(len(q_cmd), NUM_MOTORS_PER_HAND)
        angles = np.zeros(NUM_MOTORS_PER_HAND, dtype=np.int16)
        for i in range(limit):
            angles[i] = int(np.clip(q_cmd[i] * 1000, 0, 1000))

        with self._cmd_lock:
            if is_left:
                self._left_cmd_angle[:] = angles
            else:
                self._right_cmd_angle[:] = angles
