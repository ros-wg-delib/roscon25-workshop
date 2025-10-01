#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Evaluates a trained RL policy."""

import argparse

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from pyrobosim_ros_gym.envs import get_env_by_name, BananaEnv, GreenhouseEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="The name of the model to evaluate.")
    parser.add_argument(
        "--discrete-actions",
        action="store_true",
        help="If true, uses discrete action space. Otherwise, uses continuous action space.",
    )
    parser.add_argument(
        "--num-episodes",
        default=3,
        type=int,
        help="The number of episodes to evaluate.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    args = parser.parse_args()

    model_name_parts = args.model.split("_")
    env_type = model_name_parts[0]
    model_type = model_name_parts[1]

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")

    env = get_env_by_name(
        env_type,
        node,
        max_steps_per_episode=10,
        realtime=True,
        discrete_actions=args.discrete_actions,
    )
    env.reset()

    # Load a model
    if model_type == "DQN":
        model: BaseAlgorithm = DQN.load(args.model, env=None)
    elif model_type == "PPO":
        model = PPO.load(args.model, env=None)
    elif model_type == "SAC":
        model = SAC.load(args.model, env=None)
    elif model_type == "A2C":
        model = A2C.load(args.model, env=None)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    # Evaluate it for some steps
    num_episodes = 0
    obs, _ = env.reset(seed=num_episodes)
    survived_episodes = 0
    watered_perc_s = []
    while num_episodes < args.num_episodes:
        print("." * 10)
        print(f"{obs=}")
        action, _ = model.predict(obs, deterministic=True)
        print(f"{action=}")
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"{reward=}")
        print(f"{terminated=}")
        print(f"{truncated=}")
        if terminated:
            num_episodes += 1
            if isinstance(env, GreenhouseEnv):
                survived_episodes += not env.dead()
                watered_plant_percent = env.watered_plant_percent()
                print(f".. {watered_plant_percent*100}% Plants watered.")
                watered_perc_s.append(watered_plant_percent)
            elif isinstance(env, BananaEnv):
                survived_episodes += info["success"]

            env.reset(seed=num_episodes)

    print(
        f"{survived_episodes} of {num_episodes} ({100.0*survived_episodes/num_episodes}%) episodes survived."
    )
    if isinstance(env, GreenhouseEnv):
        print(f"{watered_perc_s=}")

    rclpy.shutdown()
