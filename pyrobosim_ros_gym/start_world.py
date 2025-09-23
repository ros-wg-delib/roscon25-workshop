#!/usr/bin/env python3

"""Loads a world to act as a server for the RL problem."""

import argparse
import os
import rclpy
import threading

from pyrobosim.core import WorldYamlLoader
from pyrobosim.gui import start_gui
from pyrobosim_ros.ros_interface import WorldROSWrapper

from pyrobosim_ros_gym.envs import get_env_ENV_CLASS_FROM_NAME, available_env_classes


def create_ros_node(world_file_path) -> WorldROSWrapper:
    """Initializes ROS node"""
    rclpy.init()
    world = WorldYamlLoader().from_file(world_file_path)
    return WorldROSWrapper(world=world, state_pub_rate=0.1, dynamics_rate=0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless", action="store_true", help="Enables headless world loading."
    )
    parser.add_argument(
        "--env",
        choices=available_env_classes(),
        help="The environment to use.",
        required=True,
    )
    args = parser.parse_args()

    env_class = get_env_ENV_CLASS_FROM_NAME(args.env)
    node = create_ros_node(env_class.world_file_path)

    if args.headless:
        # Start ROS node in main thread if there is no GUI.
        node.start(wait_for_gui=False)
    else:
        # Start ROS node in separate thread.
        ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
        ros_thread.start()

        # Start GUI in main thread.
        start_gui(node.world)
