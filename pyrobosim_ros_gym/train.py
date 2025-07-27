#!/usr/bin/env python3

"""Trains an RL policy."""

from datetime import datetime

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from pyrobosim_ros_env import PyRoboSimRosEnv


if __name__ == "__main__":
    rclpy.init()
    node = Node("pyrobosim_ros_env")

    env = PyRoboSimRosEnv(node)

    # Train a model
    model_type = "PPO"
    if model_type == "DQN":
        model = DQN("MlpPolicy", env=env, verbose=1, learning_starts=10, learning_rate=0.001, batch_size=32, train_freq=(5, "step"))
    elif model_type == "PPO":
        model = PPO("MlpPolicy", env=env, verbose=1, learning_rate=0.001, batch_size=32, n_steps=20)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    print(f"\nTraining {model_type}...\n")
    model.learn(total_timesteps=1000, progress_bar=True)

    # Evaluate the trained model
    print("\nEvaluating...\n")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

    # Save the trained model
    model_name = f"{model_type}_" + datetime.now().strftime("model_%Y_%m_%d_%H_%M_%S" + ".pt")
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
