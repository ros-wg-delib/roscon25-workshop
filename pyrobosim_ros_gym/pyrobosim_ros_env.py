import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.action import ActionClient

from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import ExecutionResult, TaskAction
from pyrobosim_msgs.srv import RequestWorldInfo, RequestWorldState, ResetWorld


class PyRoboSimRosEnv(gym.Env):
    """Gym environment wrapping around the PyRoboSim ROS Interface."""

    def __init__(self, node, max_steps_per_episode=20):
        super().__init__()
        self.node = node

        self.request_info_client = node.create_client(RequestWorldInfo, "/request_world_info")
        self.request_state_client = node.create_client(RequestWorldState, "/request_world_state")
        self.execute_action_client = ActionClient(node, ExecuteTaskAction, "/execute_action")
        self.reset_world_client = node.create_client(ResetWorld, "reset_world")

        future = self.request_info_client.call_async(RequestWorldInfo.Request())
        rclpy.spin_until_future_complete(self.node, future)
        self.world_info = future.result().info

        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        world_state = future.result().state

        self.all_locations = []
        for loc in world_state.locations:
            self.all_locations.extend(loc.spawns)
        num_locations = sum(
            len(loc.spawns) for loc in world_state.locations
        )
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}

        num_object_types = len(self.world_info.object_categories)
        self.obj_to_idx = {obj: idx for idx, obj in enumerate(self.world_info.object_categories)}

        obs_size = (
            num_locations   # Number of locations robot can be in
            + num_object_types   # Object types robot is holding
            + (num_locations * num_object_types)  # Number of object categories per location
        )

        # Action space is defined by:
        #   Move: To all possible object spawns
        #   Pick: To all specific object types
        #   Place: the current object
        idx = 0
        self.integer_to_action = {}
        for loc in self.all_locations:
            self.integer_to_action[idx] = TaskAction(type="navigate", target_location=loc)
            idx += 1
        for obj_category in self.world_info.object_categories:
            self.integer_to_action[idx] = TaskAction(type="pick", object=obj_category)
            idx += 1
        self.integer_to_action[idx] = TaskAction(type="place")

        self.action_space = spaces.Discrete(len(self.integer_to_action))

        # Observation space is defined by:
        #  Location of robot
        #  Type of object robot is holding (including None)
        #  Whether there is at least one of a specific object type at each location
        self.observation_space = spaces.Box(
            low=-np.ones(obs_size, dtype=np.float32),
            high=np.ones(obs_size, dtype=np.float32),
        )

        self.step_number = 0
        self.max_steps_per_episode = max_steps_per_episode


    def step(self, action):
        t_start = time.time()
        info = {}

        goal = ExecuteTaskAction.Goal()
        goal.action = self.integer_to_action[action]
        goal.action.robot = "robot"

        goal_future = self.execute_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        t_elapsed = time.time() - t_start
        action_result = result_future.result().result
        self.step_number += 1
        truncated = (self.step_number >= self.max_steps_per_episode)
        if truncated:
            print("Maximum steps per episode exceeded. Truncated episode.")

        observation = self._get_obs()
        robot_state = self.world_state.robots[0]

        terminated = False

        # Compute reward (TODO: Factor out)
        # The farther away you travel, the less reward
        # TODO: This should hopefully get a cost/distance from PyRoboSim itself instead of using time
        reward = -1.0 * t_elapsed
        # Discourage repeating and failing actions
        if (goal.action.type == "navigate") and (goal.action.target_location == robot_state.last_visited_location):
            reward -= 1.0
        if (action_result.execution_result.status != ExecutionResult.SUCCESS):
            reward -= 0.5
        # Robot gets positive reward based on holding a banana.
        for obj in self.world_state.objects:
            if obj.category == "banana":
                if obj.name == robot_state.manipulated_object:
                    print(f"Robot is at {robot_state.last_visited_location} and holding {obj.name}. Episode succeeded!")
                    reward += 5.0
                    terminated = True
                    break

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        print(f"Resetting environment")
        self.step_number = 0

        future = self.reset_world_client.call_async(ResetWorld.Request())
        rclpy.spin_until_future_complete(self.node, future)

        observation = self._get_obs()
        info = {}

        return observation, info

    def _get_obs(self):
        """Calculate the observation. All elements are either -1.0 or +1.0."""
        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        world_state = future.result().state
        robot_state = world_state.robots[0]

        num_locations = len(self.all_locations)
        num_object_types = len(self.world_info.object_categories)

        obs_size = (
            num_locations   # Number of locations robot can be in
            + num_object_types   # Object type robot is holding
            + (num_locations * num_object_types)  # Number of object categories per location
        )

        obs = -np.ones(obs_size, dtype=np.float32)

        # Robot's current location
        if robot_state.last_visited_location in self.loc_to_idx:
            loc_idx = self.loc_to_idx[robot_state.last_visited_location]
            obs[loc_idx] = 1.0

        # Robot's currently held object
        if robot_state.manipulated_object:
            for obj in world_state.objects:
                if obj.name == robot_state.manipulated_object:
                     obs[num_locations + self.obj_to_idx[obj.category]] = 1.0
                     break

        # Object categories per location
        for obj in world_state.objects:
            obj_idx = self.obj_to_idx[obj.category]
            loc_idx = self.loc_to_idx.get(obj.parent)
            if loc_idx is None:
                continue
            obs[num_locations + num_object_types + (loc_idx * num_object_types) + obj_idx] = 1.0

        self.world_state = world_state
        return obs
