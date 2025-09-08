import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.action import ActionClient

from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import ExecutionResult, TaskAction, WorldState, ObjectState
from pyrobosim_msgs.srv import (
    RequestWorldInfo,
    RequestWorldState,
    ResetWorld,
    SetLocationState,
)



class PyRoboSimRosEnv(gym.Env):
    """Gym environment wrapping around the PyRoboSim ROS Interface."""

    world_file_path = "DEFINE_IN_SUBCLASS"

    def __init__(
        self,
        node,
        reward_fn,
        reset_validation_fn=None,
        max_steps_per_episode=50,
        realtime=True,
        discrete_actions=True,
    ):
        """
        Instantiates a PyRoboSim ROS environment.

        :param node: The ROS node to use for creating clients.
        :param reward_fn: Function that calculates the reward and termination criteria.
        :param reset_validation_fn: Function that calculates whether a reset is valid.
            If None (default), all resets are valid.
        :param max_steps_per_episode: Maximum number of steps before truncating an episode.
        :param realtime: If True, commands PyRoboSim to run actions in real time.
            If False, actions run as quickly as possible for faster training.
        :param discrete_actions: If True, uses discrete actions, else uses continuous.
        """
        super().__init__()
        self.node = node
        self.realtime = realtime
        self.max_steps_per_episode = max_steps_per_episode
        self.discrete_actions = discrete_actions
        self.reward_fn = lambda goal, result: reward_fn(self, goal, result)
        if reset_validation_fn is None:
            self.reset_validation_fn = lambda: True
        else:
            self.reset_validation_fn = lambda: reset_validation_fn(self)

        self.step_number = 0
        self.previous_location = None
        self.previous_action_type = None

        self.request_info_client = node.create_client(
            RequestWorldInfo, "/request_world_info"
        )
        self.request_state_client = node.create_client(
            RequestWorldState, "/request_world_state"
        )
        self.execute_action_client = ActionClient(
            node, ExecuteTaskAction, "/execute_action"
        )
        self.reset_world_client = node.create_client(ResetWorld, "reset_world")
        self.set_location_state_client = node.create_client(
            SetLocationState, "set_location_state"
        )

        self.request_info_client.wait_for_service()
        self.request_state_client.wait_for_service()
        self.execute_action_client.wait_for_server()
        self.reset_world_client.wait_for_service()
        self.set_location_state_client.wait_for_service()

        future = self.request_info_client.call_async(RequestWorldInfo.Request())
        rclpy.spin_until_future_complete(self.node, future)
        self.world_info = future.result().info

        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        self.world_state = future.result().state

        self.all_locations = []
        for loc in self.world_state.locations:
            self.all_locations.extend(loc.spawns)
        self.num_locations = sum(len(loc.spawns) for loc in self.world_state.locations)
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}
        print(f"{self.all_locations=}")

        self.action_space = self._action_space()
        print(f"{self.action_space=}")

    def _action_space(self):
        raise NotImplementedError("implement in sub-class")

    def get_next_navigation_action(self):
        self.waypoint_i = (self.waypoint_i + 1) % len(self.waypoints)
        return self.get_current_navigation_action()
    
    def get_current_navigation_action(self):
        return TaskAction(
            type="navigate", target_location=self.get_current_location()
        )
    
    def get_current_location(self):
        return self.waypoints[self.waypoint_i]

    def step(self, action):
        raise NotImplementedError("implement in sub-class")
    
    def go_to_next_wp(self):
        nav_goal = ExecuteTaskAction.Goal()
        nav_goal.action = self.get_next_navigation_action()
        nav_goal.action.robot = "robot"
        nav_goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

    def reset(self, seed=None, options=None):
        """Resets the environment with a specified seed and options."""
        print(f"Resetting environment {seed=}")
        super().reset(seed=seed)
