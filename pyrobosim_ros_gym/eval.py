#!/usr/bin/env python3

"""Evaluates a trained RL policy."""

import argparse

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C

from pyrobosim_ros_env import PyRoboSimRosEnv
from envs.banana import banana_picked_reward, banana_on_table_reward


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

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env_type = args.model.split("_")[0]
    if env_type == "PickBanana":
        reward_fn = banana_picked_reward
    elif env_type == "PlaceBanana":
        reward_fn = banana_on_table_reward
    else:  # TODO: Add another "fire avoidance" type env
        raise ValueError(f"Invalid environment name: {args.env}")

    env = PyRoboSimRosEnv(
        node,
        reward_fn=reward_fn,
        max_steps_per_episode=10,
    )

    # Load a model
    model_type = args.model.split("_")[1]
    if model_type == "DQN":
        model = DQN.load(args.model, env=env)
    elif model_type == "PPO":
        model = PPO.load(args.model, env=env)
    elif model_type == "SAC":
        model = SAC.load(args.model, env=env)
    elif model_type == "A2C":
        model = A2C.load(args.model, env=env)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    # Evaluate it for some steps
    vec_env = model.get_env()
    assert vec_env is not None, "Environment must be defined."
    obs = vec_env.reset()
    num_episodes = 0
    successful_episodes = 0
    while num_episodes < args.num_episodes:
        # print("." * 10)
        # print(f"{obs=}")
        action, _ = model.predict(obs, deterministic=True)
        # print(f"{action=}")
        obs, rewards, dones, infos = vec_env.step(action)
        # print(f"{rewards=}")
        if dones[0]:
            num_episodes += 1
            if infos[0]["success"]:
                successful_episodes += 1
    print(
        f"{successful_episodes} of {num_episodes} ({100.*successful_episodes/num_episodes}%) episodes successful."
    )

    rclpy.shutdown()
