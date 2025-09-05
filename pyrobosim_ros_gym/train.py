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
from typing import Dict
from pyrobosim_ros_env import PyRoboSimRosEnv

def train_w_args(
    kwargs: Dict
):
    # params
    learning_rate = dict(kwargs).pop('learning_rate', .001)

    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = PyRoboSimRosEnv(node, realtime=kwargs['realtime'], max_steps_per_episode=25)

    # Train a model
    log_path = "train_logs" if kwargs['log'] else None
    if kwargs['model_type'] == "DQN":
        policy_kwargs = {
            "activation_fn": nn.ReLU,
            "net_arch": [8, 4],
        }
        model = DQN(  # type: BaseAlgorithm
            "MlpPolicy",
            env=env,
            seed=kwargs['seed'],
            # policy_kwargs=policy_kwargs,
            gamma=0.99,
            exploration_initial_eps=0.75,
            exploration_final_eps=0.2,
            exploration_fraction=0.25,
            learning_starts=kwargs['total_timesteps'] // 4,
            learning_rate=learning_rate,
            batch_size=2,
            train_freq=(1, "step"),
            target_update_interval=1,
            tensorboard_log=log_path,
        )
    elif kwargs['model_type'] == "PPO":
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
            seed=kwargs['seed'],
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            learning_rate=0.0003,
            batch_size=2,
            n_steps=2,
            tensorboard_log=log_path,
        )
    elif kwargs['model_type'] == "SAC":
        model = SAC(
            "MlpPolicy",
            env=env,
            seed=kwargs['seed'],
            # policy_kwargs=policy_kwargs,    .. Let's try default values
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=32,
            tensorboard_log=log_path,
        )
    elif kwargs['model_type'] == "A2C":
        # policy_kwargs = {
        #     "activation_fn": nn.ReLU,
        #     "net_arch": [8, 4],
        # }
        model = A2C(
            "MlpPolicy",
            env=env,
            seed=kwargs['seed'],
            # policy_kwargs=policy_kwargs,
            learning_rate=0.0003,
            gamma=0.99,
            # n_steps=1,
            stats_window_size=2,
            tensorboard_log=log_path,
        )
    else:
        raise RuntimeError(f"Invalid model type: {kwargs['model_type']}")
    print(f"\nTraining with {kwargs['model_type']}...\n")

    # Train the model until it exceeds a specified reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=9.5, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        n_eval_episodes=10,
        eval_freq=500,
        verbose=1,
    )
    model.learn(
        total_timesteps=kwargs['total_timesteps'], progress_bar=True, callback=eval_callback
    )

    # Save the trained model
    model_name = f"{kwargs['model_type']}_" + datetime.now().strftime(
        "model_%Y_%m_%d_%H_%M_%S" + ".pt"
    )
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        default="DQN",
        choices=["DQN", "PPO", "SAC", "A2C"],
        help="The model type to train.",
    )
    parser.add_argument(
        "--total-timesteps",
        default=100,
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

    train_w_args(vars(args))
