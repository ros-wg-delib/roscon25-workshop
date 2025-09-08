"""Utilities for the banana test environment."""

import os
from enum import Enum
import rclpy
import numpy as np
from gymnasium import spaces
from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import ExecutionResult, ObjectState, TaskAction, WorldState
from pyrobosim_msgs.srv import (
    RequestWorldInfo,
    RequestWorldState,
    ResetWorld,
    SetLocationState,
)

from pyrobosim_ros_gym.envs.pyrobosim_ros_env import PyRoboSimRosEnv


class BananaEnv(PyRoboSimRosEnv):
    sub_type = Enum("sub_type", "Pick Place PlaceNoSoda")
    world_file_path = os.path.join("rl_ws_worlds", "worlds", "banana.yaml")

    def __init__(
        self,
        sub_type: sub_type,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
    ):
        if sub_type == BananaEnv.sub_type.Pick:
            reward_fn = banana_picked_reward
            reset_validation_fn = None
            # eval_freq = 1000
        elif sub_type == BananaEnv.sub_type.Place:
            reward_fn = banana_on_table_reward
            reset_validation_fn = None
            # eval_freq = 2000
        elif sub_type == BananaEnv.sub_type.PlaceNoSoda:
            reward_fn = banana_on_table_avoid_soda_reward
            reset_validation_fn = avoid_soda_reset_validation
            # eval_freq = 2000
        else:
            raise ValueError(f"Invalid environment: {sub_type}")

        super().__init__(
            node,
            reward_fn,
            reset_validation_fn,
            max_steps_per_episode,
            realtime,
            discrete_actions,
        )

        self.num_locations = sum(len(loc.spawns) for loc in self.world_state.locations)
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}

        self.num_object_types = len(self.world_info.object_categories)
        self.obj_to_idx = {
            obj: idx for idx, obj in enumerate(self.world_info.object_categories)
        }

        # Observation space is defined by:
        #  Previous action
        #  Location of robot
        #  Type of object robot is holding (if any)
        #  Whether there is at least one of a specific object type at each location
        self.obs_size = (
            +self.num_locations  # Number of locations robot can be in
            + self.num_object_types  # Object types robot is holding
            + (
                self.num_locations * self.num_object_types
            )  # Number of object categories per location
        )

        self.observation_space = spaces.Box(
            low=-np.ones(self.obs_size, dtype=np.float32),
            high=np.ones(self.obs_size, dtype=np.float32),
        )
        print(f"{self.observation_space=}")

    def _action_space(self):
        # Action space is defined by:
        #   Move: To all possible object spawns
        #   Pick: All possible object categories
        #   Place: The current manipulated object
        idx = 0
        self.integer_to_action = {}
        for loc in self.all_locations:
            self.integer_to_action[idx] = TaskAction(
                type="navigate", target_location=loc
            )
            idx += 1
        for obj_category in self.world_info.object_categories:
            self.integer_to_action[idx] = TaskAction(type="pick", object=obj_category)
            idx += 1
        self.integer_to_action[idx] = TaskAction(type="place")
        self.num_actions = len(self.integer_to_action)

        if self.discrete_actions:
            return spaces.Discrete(self.num_actions)
        else:
            return spaces.Box(
                low=np.zeros(self.num_actions, dtype=np.float32),
                high=np.ones(self.num_actions, dtype=np.float32),
            )

    def step(self, action):
        """Steps the environment with a specific action."""
        self.previous_location = self.world_state.robots[0].last_visited_location

        goal = ExecuteTaskAction.Goal()
        if self.discrete_actions:
            goal.action = self.integer_to_action[action]
        else:
            goal.action = self.integer_to_action[np.argmax(action)]
        goal.action.robot = "robot"
        goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

        action_result = result_future.result().result
        self.step_number += 1
        truncated = self.step_number >= self.max_steps_per_episode
        if truncated:
            print(
                f"Maximum steps ({self.max_steps_per_episode}) exceeded. Truncated episode."
            )

        observation = self._get_obs()
        reward, terminated, info = self.reward_fn(goal, action_result)
        self.previous_action_type = goal.action.type

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_number = 0
        info = {}

        valid_reset = False
        num_reset_attempts = 0
        while not valid_reset:
            future = self.reset_world_client.call_async(
                ResetWorld.Request(seed=(seed or -1))
            )
            rclpy.spin_until_future_complete(self.node, future)

            observation = self._get_obs()

            valid_reset = self.reset_validation_fn()
            num_reset_attempts += 1

        print(f"Reset environment in {num_reset_attempts} attempt(s).")
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
                obs[
                    self.num_locations
                    + self.num_object_types
                    + (loc_idx * self.num_object_types)
                    + obj_idx
                ] = 1.0

        self.world_state = world_state
        return obs


def banana_picked_reward(env, goal, action_result):
    """
    Checks whether the robot has picked a banana.

    :param: The environment.
    :goal: The ROS action goal sent to the robot.
    :action_result: The result of the above goal.
    :
    :return: A tuple of (reward, terminated, info)
    """
    # Calculate reward
    reward = 0.0
    terminated = False
    info = {"success": False}

    robot_state = env.world_state.robots[0]
    # Discourage repeating the same navigation action or failing to pick/place.
    if (goal.action.type == "navigate") and (
        goal.action.target_location == env.previous_location
    ):
        reward -= 1.0
    if action_result.execution_result.status != ExecutionResult.SUCCESS:
        reward -= 1.0
    # Discourage repeat action types.
    if goal.action.type == env.previous_action_type:
        reward -= 0.5
    # Robot gets positive reward based on holding a banana,
    # and negative reward for being in locations without bananas.
    at_banana_location = False
    for obj in env.world_state.objects:
        if obj.category == "banana":
            if obj.parent == robot_state.last_visited_location:
                at_banana_location = True
            elif obj.name == robot_state.manipulated_object:
                print(
                    f"🍌 At {robot_state.last_visited_location} and holding {obj.name}. "
                    f"Episode succeeded in {env.step_number} steps!"
                )
                reward += 10.0
                terminated = True
                at_banana_location = True
                info["success"] = True
                break

    # Reward shaping: Penalty if the robot is not at a location containing a banana.
    if not terminated and not at_banana_location:
        reward -= 0.5
    return reward, terminated, info


def banana_on_table_reward(env, goal, action_result):
    """
    Checks whether the robot has placed a banana on the table.

    :param: The environment.
    :goal: The ROS action goal sent to the robot.
    :action_result: The result of the above goal.
    :
    :return: A tuple of (reward, terminated, info)
    """
    # Calculate reward
    reward = 0.0
    terminated = False
    info = {"success": False}

    robot_state = env.world_state.robots[0]
    # Discourage repeating the same navigation action or failing to pick/place.
    if (goal.action.type == "navigate") and (
        goal.action.target_location == env.previous_location
    ):
        reward -= 1.0
    if action_result.execution_result.status != ExecutionResult.SUCCESS:
        reward -= 1.0
    # Discourage repeat action types.
    if goal.action.type == env.previous_action_type:
        reward -= 0.5
    # Robot gets positive reward based on a banana being at the table.
    at_banana_location = False
    holding_banana = False
    for obj in env.world_state.objects:
        if obj.category == "banana":
            if obj.parent == robot_state.last_visited_location:
                at_banana_location = True
            if obj.parent == "table0_tabletop":
                print(
                    f"🍌 Banana is on the table. "
                    f"Episode succeeded in {env.step_number} steps!"
                )
                reward += 10.0
                terminated = True
                info["success"] = True
                break

        if obj.category == "banana" and obj.name == robot_state.manipulated_object:
            holding_banana = True

    # Reward shaping: Adjust the reward related to how close the robot is to completing the task.
    if not terminated:
        if holding_banana and env.previous_location != "table0_tabletop":
            reward += 0.25
        elif holding_banana:
            reward += 0.1
        elif at_banana_location:
            reward -= 0.1
        else:
            reward -= 0.25

    return reward, terminated, info


def banana_on_table_avoid_soda_reward(env, goal, action_result):
    """
    Checks whether the robot has placed a banana on the table without touching the soda.

    :param: The environment.
    :goal: The ROS action goal sent to the robot.
    :action_result: The result of the above goal.
    :
    :return: A tuple of (reward, terminated, info)
    """
    # Start with the same reward as the no-soda case.
    reward, terminated, info = banana_on_table_reward(env, goal, action_result)

    # Robot gets additional negative reward being near a soda,
    # and fails if it tries to pick or place at a location with a soda.
    robot_state = env.world_state.robots[0]
    for obj in env.world_state.objects:
        if obj.category == "coke" and obj.parent == robot_state.last_visited_location:
            if goal.action.type == "navigate":
                reward -= 2.5
            else:
                print(
                    "🔥 Tried to pick and place near a soda. "
                    f"Episode failed in {env.step_number} steps!"
                )
                reward -= 25.0
                terminated = True
                info["success"] = False

    return reward, terminated, info


def avoid_soda_reset_validation(env):
    """
    Checks whether an environment has been properly reset to avoid soda.

    Specifically, we are checking that:
      - There is at least one banana not next to a soda
      - The robot is not at a starting location where there is a soda

    :param: The environment.
    :return: True if valid, else False.
    """
    soda_location = None
    for obj in env.world_state.objects:
        if obj.category == "coke":
            soda_location = obj.parent

    valid_banana_locations = False
    for obj in env.world_state.objects:
        if obj.category == "banana" and obj.parent != soda_location:
            valid_banana_locations = True

    robot_location = env.world_state.robots[0].last_visited_location
    valid_robot_location = robot_location != soda_location

    return valid_banana_locations and valid_robot_location
