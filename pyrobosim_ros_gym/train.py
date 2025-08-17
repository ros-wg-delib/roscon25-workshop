#!/usr/bin/env python3

"""Trains an RL policy."""

import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from torch import nn

from pyrobosim_ros_env import PyRoboSimRosEnv
from envs.banana import banana_picked_reward, banana_on_table_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        default="PickBanana",
        choices=["PickBanana", "PlaceBanana"],
        help="The environment to use.",
    )
    parser.add_argument(
        "--model-type",
        default="DQN",
        choices=["DQN", "PPO", "SAC", "A2C"],
        help="The model type to train.",
    )
    parser.add_argument(
        "--total-timesteps",
        default=25000,
        type=int,
        help="The number of total timesteps to train for.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    parser.add_argument(
        "--realtime", action="store_true", help="If true, slows down to real time."
    )
    parser.add_argument(
        "--log", action="store_true", help="If true, logs data to Tensorboard."
    )
    args = parser.parse_args()

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")
    if args.env == "PickBanana":
        env = PyRoboSimRosEnv(
            node,
            reward_fn=banana_picked_reward,
            realtime=args.realtime,
            max_steps_per_episode=25,
        )
        eval_freq = 1000
    elif args.env == "PlaceBanana":
        env = PyRoboSimRosEnv(
            node,
            reward_fn=banana_on_table_reward,
            realtime=args.realtime,
            max_steps_per_episode=50,
        )
        eval_freq = 2000
    else:  # TODO: Add another "fire avoidance" type env.
        raise ValueError(f"Invalid environment name: {args.env}")

    # Train a model
    log_path = "train_logs" if args.log else None
    if args.model_type == "DQN":
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": [64, 64],
        }
        model = DQN(  # type: BaseAlgorithm
            "MlpPolicy",
            env=env,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            exploration_initial_eps=0.75,
            exploration_final_eps=0.1,
            exploration_fraction=0.25,
            learning_starts=100,
            learning_rate=0.0001,
            batch_size=32,
            gradient_steps=10,
            train_freq=(4, "step"),
            target_update_interval=500,
            tensorboard_log=log_path,
        )
    elif args.model_type == "PPO":
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": {
                "pi": [64, 64],
                "vf": [64, 32],
            },
        }
        model = PPO(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            learning_rate=0.0003,
            batch_size=32,
            n_steps=64,
            tensorboard_log=log_path,
        )
    elif args.model_type == "SAC":
        model = SAC(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            # policy_kwargs=policy_kwargs,    .. Let's try default values
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            tensorboard_log=log_path,
        )
    elif args.model_type == "A2C":
        model = A2C(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            # policy_kwargs=policy_kwargs,    .. Let's try default values
            learning_rate=0.0003,
            gamma=0.99,
            tensorboard_log=log_path,
        )
    else:
        raise RuntimeError(f"Invalid model type: {args.model_type}")
    print(f"\nTraining with {args.model_type}...\n")

    # Train the model until it exceeds a specified reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=9.5, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        n_eval_episodes=10,
        eval_freq=eval_freq,
        verbose=1,
    )
    model.learn(
        total_timesteps=args.total_timesteps, progress_bar=True, callback=eval_callback
    )

    # Save the trained model
    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = f"{args.env}_{args.model_type}_{date_str}.pt"
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
