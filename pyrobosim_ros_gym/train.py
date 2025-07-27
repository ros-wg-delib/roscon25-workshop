#!/usr/bin/env python3

"""Trains an RL policy."""

import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="DQN", choices=["DQN", "PPO"], help="The model type to train.")
    parser.add_argument("--total-timesteps", default=1000, help="The number of total timesteps to train for.")
    parser.add_argument("--gamma", default=0.95, help="The discount factor for RL.")
    args = parser.parse_args()

    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = PyRoboSimRosEnv(node)

    # Train a model
    if args.model_type == "DQN":
        model = DQN("MlpPolicy", env=env, verbose=1, gamma=args.gamma, exploration_initial_eps=0.2, learning_starts=50, learning_rate=0.0001, batch_size=32, train_freq=(16, "step"))
        print(f"\nTraining with DQN...\n")
    elif args.model_type == "PPO":
        model = PPO("MlpPolicy", env=env, verbose=1, gamma=args.gamma, learning_rate=0.0005, batch_size=32, n_steps=32)
        print(f"\nTraining with PPO...\n")
    else:
        raise RuntimeError(f"Invalid model type: {args.model_type}")

    model.learn(total_timesteps=args.total_timesteps, progress_bar=True)

    # Evaluate the trained model
    print("\nEvaluating...\n")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

    # Save the trained model
    model_name = f"{args.model_type}_" + datetime.now().strftime("model_%Y_%m_%d_%H_%M_%S" + ".pt")
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
