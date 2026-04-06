"""
Quick diagnostic: can we receive Inspire FTP hand state via DDS?
Run this WHILE Headless_driver_double.py is running on the robot.

Usage:
    python scripts/test_inspire_dds_state.py
"""
import time
from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types
from unitree_sdk2py.core.channel import (
    ChannelSubscriber,
    ChannelFactoryInitialize,
)


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


def main():
    print("Initializing DDS (domain 0)...")
    try:
        ChannelFactoryInitialize(0)
    except Exception as e:
        print(f"ChannelFactory warning: {e}")

    print("Subscribing to rt/inspire_hand/state/l and /r ...")
    sub_l = ChannelSubscriber("rt/inspire_hand/state/l", InspireHandStateMsg)
    sub_l.Init()
    sub_r = ChannelSubscriber("rt/inspire_hand/state/r", InspireHandStateMsg)
    sub_r.Init()

    print("Waiting for messages (10s timeout)...\n")
    got_l = False
    got_r = False
    start = time.time()

    while time.time() - start < 10:
        msg_l = sub_l.Read()
        msg_r = sub_r.Read()

        if msg_l is not None and not got_l:
            print(f"  LEFT  state received!  angle_act={list(msg_l.angle_act)}")
            got_l = True
        if msg_r is not None and not got_r:
            print(f"  RIGHT state received!  angle_act={list(msg_r.angle_act)}")
            got_r = True

        if got_l and got_r:
            # Keep printing for a few more reads
            for _ in range(5):
                time.sleep(0.1)
                msg_l = sub_l.Read()
                msg_r = sub_r.Read()
                if msg_l:
                    print(f"  L: angle_act={list(msg_l.angle_act)}")
                if msg_r:
                    print(f"  R: angle_act={list(msg_r.angle_act)}")
            break

        time.sleep(0.05)

    print()
    if not got_l:
        print("FAIL: No LEFT hand state received!")
    if not got_r:
        print("FAIL: No RIGHT hand state received!")

    if not got_l or not got_r:
        print("\nPossible causes:")
        print("  1. Headless_driver_double.py not running on robot")
        print("  2. DDS domain mismatch (both must use domain 0)")
        print("  3. Network interface mismatch (try without 'eno1')")
        print("  4. Multicast not working across network")
        print("\nTry running this ON the robot instead of the workstation.")
    else:
        print("SUCCESS: Both hands reporting state via DDS.")


if __name__ == "__main__":
    main()
