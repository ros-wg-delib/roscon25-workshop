#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Trains an RL policy."""

import argparse
from datetime import datetime
import importlib
import os
from typing import Any

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.base_class import BaseAlgorithm
import yaml

from pyrobosim_ros_gym.envs import get_env_by_name, available_envs_w_subtype


def get_args() -> argparse.Namespace:
    """Helper function to parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=available_envs_w_subtype(),
        help="The environment to use.",
        required=True,
    )
    parser.add_argument(
        "--config",
        help="Path to the training configuration YAML file.",
        required=True,
    )
    parser.add_argument(
        "--model-type",
        default="DQN",
        choices=["DQN", "PPO", "SAC", "A2C"],
        help="The model type to train.",
    )
    parser.add_argument(
        "--discrete-actions",
        action="store_true",
        help="If true, uses discrete action space. Otherwise, uses continuous action space.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    parser.add_argument(
        "--realtime", action="store_true", help="If true, slows down to real time."
    )
    parser.add_argument(
        "--log", action="store_true", help="If true, logs data to Tensorboard."
    )
    args = parser.parse_args()
    return args


def get_config(config_path: str) -> dict[str, Any]:
    """Helper function to parse the configuration YAML file."""
    config_path = args.config
    if not os.path.isabs(config_path):
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config"
        )
        config_path = os.path.join(default_path, config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Handle special case of policy_kwargs activation function needing to be a class instance.
    for subtype in config.get("training", {}):
        subtype_config = config["training"][subtype]
        if not isinstance(subtype_config, dict):
            continue
        policy_kwargs = subtype_config.get("policy_kwargs", {})
        if "activation_fn" in policy_kwargs:
            module_name, class_name = policy_kwargs["activation_fn"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            policy_kwargs["activation_fn"] = getattr(module, class_name)

    return config


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config)

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = get_env_by_name(
        args.env,
        node,
        max_steps_per_episode=25,
        realtime=False,
        discrete_actions=args.discrete_actions,
    )

    # Train a model
    log_path = "train_logs" if args.log else None
    if args.model_type == "DQN":
        dqn_config = config.get("training", {}).get("DQN", {})
        if "policy_kwargs" in dqn_config:
            policy_kwargs = dqn_config["policy_kwargs"]
            del dqn_config["policy_kwargs"]
        model: BaseAlgorithm = DQN(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **dqn_config,
        )
    elif args.model_type == "PPO":
        ppo_config = config.get("training", {}).get("PPO", {})
        if "policy_kwargs" in ppo_config:
            policy_kwargs = ppo_config["policy_kwargs"]
            del ppo_config["policy_kwargs"]
        model = PPO(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **ppo_config,
        )
    elif args.model_type == "SAC":
        sac_config = config.get("training", {}).get("SAC", {})
        if "policy_kwargs" in sac_config:
            policy_kwargs = sac_config["policy_kwargs"]
            del sac_config["policy_kwargs"]
        model = SAC(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **sac_config,
        )
    elif args.model_type == "A2C":
        a2c_config = config.get("training", {}).get("A2C", {})
        if "policy_kwargs" in a2c_config:
            policy_kwargs = a2c_config["policy_kwargs"]
            del a2c_config["policy_kwargs"]
        model = A2C(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **a2c_config,
        )
    else:
        raise RuntimeError(f"Invalid model type: {args.model_type}")
    print(f"\nTraining with {args.model_type}...\n")

    # Train the model until it exceeds a specified reward threshold
    training_config = config["training"]
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=training_config["reward_threshold"],
        verbose=1,
    )
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        **training_config.get("eval", {}),
    )

    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_name = f"{args.env}_{args.model_type}_{date_str}"
    model.learn(
        total_timesteps=training_config["max_training_steps"],
        progress_bar=True,
        tb_log_name=log_name,
        callback=eval_callback,
    )

    # Save the trained model
    model_name = f"{log_name}.pt"
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
