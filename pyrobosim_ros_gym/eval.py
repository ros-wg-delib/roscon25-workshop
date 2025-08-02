#!/usr/bin/env python3

"""Evaluates a trained RL policy."""

import argparse

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="The name of the model to evaluate.")
    parser.add_argument(
        "--num-episodes",
        default=10,
        type=int,
        help="The number of episodes to evaluate.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = PyRoboSimRosEnv(node, max_steps_per_episode=10)

    # Load a model
    model_type = args.model.split("_")[0]
    if model_type == "DQN":
        model = DQN.load(args.model, env=env)
    elif model_type == "PPO":
        model = PPO.load(args.model, env=env)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    # Evaluate it for some steps
    vec_env = model.get_env()
    obs = vec_env.reset()
    num_episodes = 0
    while num_episodes < args.num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            num_episodes += 1

    rclpy.shutdown()
