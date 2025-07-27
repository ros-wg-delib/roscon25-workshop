#!/usr/bin/env python3

"""Trains an RL policy."""

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    rclpy.init()
    node = Node("pyrobosim_ros_env")

    env = PyRoboSimRosEnv(node)

    out = env.reset()
    print(f"Reset output:\n{out}")

    out = env.step(1.0)
    print(f"Step output:\n{out}")

    check_env(env)

    print("Training...")
    model = DQN("MlpPolicy", env=env, verbose=1)
    model.learn(total_timesteps=100, progress_bar=True)

    rclpy.shutdown()
