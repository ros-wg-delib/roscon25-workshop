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

    def __init__(self, node, max_steps_per_episode=50, realtime=True):
        super().__init__()
        self.node = node
        self.realtime = realtime
        self.max_steps_per_episode = max_steps_per_episode
        self.step_number = 0

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
        self.num_locations = sum(
            len(loc.spawns) for loc in world_state.locations
        )
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}

        self.num_object_types = len(self.world_info.object_categories)
        self.obj_to_idx = {obj: idx for idx, obj in enumerate(self.world_info.object_categories)}

        # Action space is defined by:
        #   Move: To all possible object spawns
        #   Pick: All possible object categories
        #   Place: The current manipulated object
        idx = 0
        self.integer_to_action = {}
        for loc in self.all_locations:
            self.integer_to_action[idx] = TaskAction(type="navigate", target_location=loc)
            idx += 1
        for obj_category in self.world_info.object_categories:
            self.integer_to_action[idx] = TaskAction(type="pick", object=obj_category)
            idx += 1
        self.integer_to_action[idx] = TaskAction(type="place")
        self.num_actions = len(self.integer_to_action)

        self.action_space = spaces.Discrete(self.num_actions)

        # Observation space is defined by:
        #  Previous action
        #  Location of robot
        #  Type of object robot is holding (if any)
        #  Whether there is at least one of a specific object type at each location
        self.obs_size = (
            + self.num_locations   # Number of locations robot can be in
            + self.num_object_types   # Object types robot is holding
            + (self.num_locations * self.num_object_types)  # Number of object categories per location
        )

        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_size, dtype=np.float32),
            high=np.ones(self.obs_size, dtype=np.float32),
        )


    def step(self, action):
        info = {}

        previous_location = self.world_state.robots[0].last_visited_location

        goal = ExecuteTaskAction.Goal()
        goal.action = self.integer_to_action[action]
        goal.action.robot = "robot"
        goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        action_result = result_future.result().result
        self.step_number += 1
        truncated = (self.step_number >= self.max_steps_per_episode)
        if truncated:
            print(f"Maximum steps ({self.max_steps_per_episode}) exceeded. Truncated episode.")

        observation = self._get_obs()
        robot_state = self.world_state.robots[0]

        # Calculate reward
        reward = 0.0
        terminated = False
        # Discourage repeating the same navigation action or failing to pick/place.
        if (goal.action.type == "navigate") and (goal.action.target_location == previous_location):
            reward -= 1.0
        if (action_result.execution_result.status != ExecutionResult.SUCCESS):
            reward -= 0.5
        # Robot gets positive reward based on holding a banana,
        # and negative reward for being in locations without bananas.
        at_banana_location = False
        for obj in self.world_state.objects:
            if obj.category == "banana":
                if obj.parent == robot_state.last_visited_location:
                    at_banana_location = True
                elif obj.name == robot_state.manipulated_object:
                    print(
                        f"Robot is at {robot_state.last_visited_location} and holding {obj.name}. "
                        f"Episode succeeded in {self.step_number} steps!"
                    )
                    reward += 10.0
                    terminated = True
                    break
        if not at_banana_location:
            reward -= 0.5

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        print(f"Resetting environment")
        self.step_number = 0

        future = self.reset_world_client.call_async(ResetWorld.Request(seed=(seed or -1)))
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

        obs = -np.ones(self.obs_size, dtype=np.float32)

        # Robot's current location
        if robot_state.last_visited_location in self.loc_to_idx:
            loc_idx = self.loc_to_idx[robot_state.last_visited_location]
            obs[loc_idx] = 1.0

        # Object categories per location (including currently held object, if any)
        for obj in world_state.objects:
            obj_idx = self.obj_to_idx[obj.category]
            if obj.name == robot_state.manipulated_object:
                obs[self.num_locations + obj_idx] = 1.0
            else:
                loc_idx = self.loc_to_idx[obj.parent]
                obs[self.num_locations + self.num_object_types + (loc_idx * self.num_object_types) + obj_idx] = 1.0

        self.world_state = world_state
        return obs
