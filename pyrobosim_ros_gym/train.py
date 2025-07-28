#!/usr/bin/env python3

"""Trains an RL policy."""

import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="DQN", choices=["DQN", "PPO"], help="The model type to train.")
    parser.add_argument("--total-timesteps", default=10000, type=int, help="The number of total timesteps to train for.")
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    parser.add_argument("--realtime", action="store_true", help="If true, slows down to real time.")
    args = parser.parse_args()

    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = PyRoboSimRosEnv(node, realtime=args.realtime, max_steps_per_episode=50)

    # Train a model
    if args.model_type == "DQN":
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": [64, 64, 32],
        }
        model = DQN("MlpPolicy", env=env, seed=args.seed, policy_kwargs=policy_kwargs, gamma=0.99, exploration_initial_eps=0.2, exploration_fraction=0.2, learning_starts=100, learning_rate=0.0005, batch_size=32, train_freq=(4, "step"), target_update_interval=100, tau=1.0)
        print(f"\nTraining with DQN...\n")
    elif args.model_type == "PPO":
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": {
                "pi": [64, 64, 32],
                "vf": [64, 64],
            }
        }
        model = PPO("MlpPolicy", env=env, seed=args.seed, policy_kwargs=policy_kwargs, gamma=0.99, learning_rate=0.0005, batch_size=32, n_steps=32)
        print(f"\nTraining with PPO...\n")
    else:
        raise RuntimeError(f"Invalid model type: {args.model_type}")

    # Train the model until it exceeds a specified reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=5.0, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, n_eval_episodes = 5, eval_freq=500, verbose=1)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=eval_callback)

    # Save the trained model
    model_name = f"{args.model_type}_" + datetime.now().strftime("model_%Y_%m_%d_%H_%M_%S" + ".pt")
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
