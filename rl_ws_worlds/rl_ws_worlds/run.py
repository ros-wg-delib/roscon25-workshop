#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""
Runner for ROS 2 Deliberation workshop worlds.
"""

import os
import rclpy
import threading

from pyrobosim.core import WorldYamlLoader
from pyrobosim.gui import start_gui
from pyrobosim_ros.ros_interface import WorldROSWrapper
from ament_index_python.packages import get_package_share_directory


def create_ros_node() -> WorldROSWrapper:
    """Initializes ROS node"""
    rclpy.init()
    node = WorldROSWrapper(state_pub_rate=0.1, dynamics_rate=0.01)
    node.declare_parameter("world_name", "greenhouse")
    node.declare_parameter("headless", False)

    # Set the world file.
    world_name = node.get_parameter("world_name").value
    node.get_logger().info(f"Starting world '{world_name}'")
    world_file = os.path.join(
        get_package_share_directory("rl_ws_worlds"),
        "worlds",
        f"{world_name}.yaml",
    )
    world = WorldYamlLoader().from_file(world_file)
    node.set_world(world)

    return node


def start_node(node: WorldROSWrapper):
    headless = node.get_parameter("headless").value
    if headless:
        # Start ROS node in main thread if there is no GUI.
        node.start(wait_for_gui=False)
    else:
        # Start ROS node in separate thread
        ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
        ros_thread.start()

        # Start GUI in main thread
        start_gui(node.world)


def main():
    node = create_ros_node()
    start_node(node)


if __name__ == "__main__":
    main()
