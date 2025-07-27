#!/usr/bin/env python3

"""Evaluates an RL policy."""

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    rclpy.init()
    node = Node("pyrobosim_ros_env")

    env = PyRoboSimRosEnv(node)

    # Load a model
    model_name = "PPO_model_2025_07_27_16_27_36.pt"  # Change me
    model_type = "PPO"
    if model_type == "DQN":
        model = DQN.load(model_name, env=env)
    elif model_type == "PPO":
        model = PPO.load(model_name, env=env)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    # Evaluate it for some steps
    vec_env = model.get_env()
    obs = vec_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)

    rclpy.shutdown()
