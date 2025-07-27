#!/usr/bin/env python3

"""Loads a world to act as a server for the RL problem."""

import argparse
import os
import rclpy
import threading

from pyrobosim.core import WorldYamlLoader
from pyrobosim.gui import start_gui
from pyrobosim_ros.ros_interface import WorldROSWrapper


def create_ros_node() -> WorldROSWrapper:
    """Initializes ROS node"""
    rclpy.init()
    world = WorldYamlLoader().from_file(os.path.join("pyrobosim_ros_gym", "test_world.yaml"))
    return WorldROSWrapper(world=world, state_pub_rate=0.1, dynamics_rate=0.01)


if __name__ == "__main__":
    node = create_ros_node()

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Enables headless world loading.")
    args = parser.parse_args()

    if args.headless:
        # Start ROS node in main thread if there is no GUI.
        node.start(wait_for_gui=False)
    else:
        # Start ROS node in separate thread.
        ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
        ros_thread.start()

        # Start GUI in main thread.
        start_gui(node.world)
