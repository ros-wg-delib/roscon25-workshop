import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import ExecutionResult
from pyrobosim_msgs.srv import RequestWorldState


class PyRoboSimRosEnv(gym.Env):
    """Gym environment wrapping around the PyRoboSim ROS Interface."""

    def __init__(self, node: Node):
        super().__init__()
        self.node = node

        self.request_state_client = node.create_client(RequestWorldState, "/request_world_state")

        self.execute_action_client = ActionClient(node, ExecuteTaskAction, "/execute_action")

        # Action space is defined by:
        #   Move: To all possible object spawns
        #   Pick: To all specific object types
        #   Place the current object
        #   Detect at current location (optional)
        self.action_space = spaces.Discrete(5)

        # Observation space is defined by:
        #  Location of robot
        #  Type of object robot is holding (including None)
        #  How many of each object type at each location
        self.observation_space = spaces.Box(
            low=np.zeros(30, dtype=np.float32),
            high=5.0 * np.ones(30, dtype=np.float32),
        )

        self.step_number = 0
        self.max_steps_per_episode = 20


    def step(self, action: int):
        observation = self._get_obs()
        t_start = time.time()
        info = {}

        goal = ExecuteTaskAction.Goal()
        goal.action.robot = "robot"
        goal.action.type = "navigate"
        goal.action.target_location = np.random.choice(["kitchen", "bedroom", "bathroom"])

        goal_future = self.execute_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        t_elapsed = time.time() - t_start
        action_result = result_future.result().result
        terminated = (action_result.execution_result.status != ExecutionResult.SUCCESS)
        self.step_number += 1
        truncated = (self.step_number >= self.max_steps_per_episode)

        reward = -1.0 * t_elapsed
        print(f"Completed step {self.step_number} in {t_elapsed} seconds")

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        self.step_number = 0

        observation = self._get_obs()
        info = {}

        return observation, info

    # def render(self):
    #     pass

    def close(self):
        rclpy.shutdown()

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        world_state = future.result().state

        num_locations = len(world_state.locations)  # TODO: Should be object spawns
        num_object_types = 5  # TODO: Should come from world info

        obs_size = (
            num_locations   # number of locations robot can be in
            + (num_object_types + 1)   # Object type robot is holding (including nothing)
            + (num_locations * num_object_types)  # Number of object categories per location
        )

        return np.zeros(obs_size, dtype=np.float32)
