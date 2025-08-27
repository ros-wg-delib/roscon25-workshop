#!/usr/bin/env python3

"""Evaluates a trained RL policy."""

import argparse

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="The name of the model to evaluate.")
    parser.add_argument(
        "--num-episodes",
        default=3,
        type=int,
        help="The number of episodes to evaluate.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = PyRoboSimRosEnv(node, max_steps_per_episode=100, realtime=True)

    # Load a model
    model_type = args.model.split("_")[0]
    if model_type == "DQN":
        model = DQN.load(args.model, env=None)
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
        obs, reward, terminated, truncated, _ = env.step(action)

        print(f"{reward=}")
        print(f"{terminated=}")
        print(f"{truncated=}")
        if terminated:
            num_episodes += 1
            survived_episodes += not env.dead()
            watered_plant_percent = env.watered_plant_percent()
            print(f".. {watered_plant_percent*100}% Plants watered.")
            watered_perc_s.append(watered_plant_percent)
            env.reset(seed=num_episodes)
    print(
        f"{survived_episodes} of {num_episodes} ({100.*survived_episodes/num_episodes}%) episodes survived."
    )
    print(f"{watered_perc_s=}")

    rclpy.shutdown()
