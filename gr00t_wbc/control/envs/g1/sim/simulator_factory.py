"""
Unitree SDK communication channel initialization.

This module provides the init_channel function for initializing
the Unitree SDK2 DDS communication layer.
"""

from typing import Any, Dict

from unitree_sdk2py.core.channel import ChannelFactoryInitialize


def init_channel(config: Dict[str, Any]) -> None:
    """
    Initialize the communication channel for robot communication.

    Args:
        config: Configuration dictionary containing DOMAIN_ID and optionally INTERFACE
    """
    if config.get("INTERFACE", None):
        ChannelFactoryInitialize(config["DOMAIN_ID"], config["INTERFACE"])
    else:
        ChannelFactoryInitialize(config["DOMAIN_ID"])
