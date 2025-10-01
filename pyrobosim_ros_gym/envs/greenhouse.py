from enum import Enum
import os
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from gymnasium import spaces
from pyrobosim_msgs.msg import TaskAction, WorldState

from .pyrobosim_ros_env import PyRoboSimRosEnv

from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.msg import TaskAction, WorldState
from pyrobosim_msgs.srv import RequestWorldState, ResetWorld


def _dist(a: Point, b: Point) -> float:
    """Calculate distance between two (geometry_msgs.msg) Points."""
    return float(np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z]))


class GreenhouseEnv(PyRoboSimRosEnv):
    sub_types = Enum("sub_types", "Deterministic Random")
    world_file_path = os.path.join("rl_ws_worlds", "worlds", "greenhouse.yaml")

    def __init__(
        self,
        sub_type: sub_types,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
    ):
        """
        Instantiate Greenhouse environment.

        :param sub_type: Subtype of this environment, e.g. `GreenhouseEnv.sub_types.Deterministic`.
        :param node: Node instance needed for ROS communication.
        :param max_steps_per_episode: Limit the steps (when to end the episode).
        :param realtime: Whether actions take time.
        :param discrete_actions: Choose discrete actions (needed for DQN).
        """
        if sub_type == GreenhouseEnv.sub_types.Deterministic:
            # TODO
            pass
        elif sub_type == GreenhouseEnv.sub_types.Random:
            # TODO
            pass
        else:
            raise ValueError(f"Invalid environment: {sub_type}")

        super().__init__(
            node,
            None,
            None,
            max_steps_per_episode,
            realtime,
            discrete_actions,
        )

        # Observation space is defined by:
        self.max_n_objects = 3
        self.max_dist = 10
        # array of n objects with a class and distance each
        self.obs_size = (self.max_n_objects, 2)
        self.observation_space = spaces.Box(
            low=-1, high=self.max_dist, shape=self.obs_size
        )
        print(f"{self.observation_space=}")

        self.plants = [obj.name for obj in self.world_state.objects]
        # print(f"{self.plants=}")
        self.good_plants = [
            obj.name for obj in self.world_state.objects if obj.category == "plant_good"
        ]
        # print(f"{self.good_plants=}")

        self.waypoints = [
            "table_c",
            "table_ne",
            "table_e",
            "table_se",
            "table_s",
            "table_sw",
            "table_w",
            "table_nw",
            "table_n",
        ]

    def _action_space(self):
        self.num_actions = 2  # stay ducked or water plant
        return spaces.Discrete(self.num_actions)

    def step(self, action):
        info = {}
        truncated = self.step_number >= self.max_steps_per_episode
        if truncated:
            print(
                f"Maximum steps ({self.max_steps_per_episode}) exceeded. Truncated episode."
            )

        # print(f"{'*'*10}")
        # print(f"{action=}")

        if action:
            self.mark_table(self.get_current_location())

        reward, terminated = self._calculate_reward(action)
        # print(f"{reward=}")

        if not terminated:
            self.go_to_next_wp()
            # action_result = result_future.result().result
            self.step_number += 1

        observation = self._get_obs()  # update self.world_state
        # print(f"{observation=}")

        return observation, reward, terminated, truncated, info

    def mark_table(self, loc):
        close_goal = ExecuteTaskAction.Goal()
        close_goal.action = TaskAction()
        close_goal.action.robot = "robot"
        close_goal.action.type = "close"
        close_goal.action.target_location = loc

        goal_future = self.execute_action_client.send_goal_async(close_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)

    def _calculate_reward(self, action):
        # Calculate reward
        reward = 0.0
        terminated = False
        plants_by_distance = self._get_plants_by_distance(self.world_state)
        # print(f"{action=}")

        close_radius = 1.0
        self.is_dead = False

        if action == 0:  # stay ducked
            return 0.0, False

        # move up to water
        for dist, plant in plants_by_distance.items():
            if dist > close_radius:
                continue
            if plant.category == "plant_good":
                if not self.watered[plant.name]:
                    self.watered[plant.name] = True
                    reward += 2
            elif plant.category == "plant_evil":
                reward += -10
                terminated = True
                self.is_dead = True
            else:
                raise RuntimeError(f"Unknown category {plant.category}")
        if reward == 0.0:  # nothing watered, wasted water
            reward = -0.1

        # print(f"{self.watered=}")
        if all(self.watered.values()):
            terminated = True

        return reward, terminated

    def dead(self):
        return self.is_dead

    def watered_plant_percent(self):
        n_watered = 0
        for w in self.watered.values():
            if w:
                n_watered += 1
        return n_watered / len(self.watered)

    def _get_plants_by_distance(self, world_state: WorldState):
        robot_state = world_state.robots[0]
        robot_pos = robot_state.pose.position
        # print(robot_pos)

        plants_by_distance = {}
        for obj in world_state.objects:
            pos = obj.pose.position
            dist = _dist(robot_pos, pos)
            dist = min(dist, self.max_dist)
            plants_by_distance[dist] = obj

        return plants_by_distance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        future = self.reset_world_client.call_async(
            ResetWorld.Request(seed=(seed or -1))
        )
        rclpy.spin_until_future_complete(self.node, future)

        # Reset helper vars
        self.step_number = 0
        self.waypoint_i = -1
        self.watered = {plant: False for plant in self.good_plants}
        self.go_to_next_wp()

        # Very first observation
        observation = self._get_obs()

        return observation, {}

    def _get_obs(self):
        """Calculate the observations"""
        future = self.request_state_client.call_async(RequestWorldState.Request())
        rclpy.spin_until_future_complete(self.node, future)
        world_state = future.result().state
        plants_by_distance = self._get_plants_by_distance(world_state)

        obs = np.zeros(self.obs_size, dtype=np.float32)
        obs[:, 0] = -1  # unknown class
        n_observations = 0
        while n_observations < self.max_n_objects:
            closest_d = min(plants_by_distance.keys())
            plant = plants_by_distance.pop(closest_d)
            plant_class = 0 if plant.category == "plant_good" else 1
            obs[n_observations] = [plant_class, closest_d]
            n_observations += 1

        self.world_state = world_state
        return obs

    def get_next_navigation_action(self):
        self.waypoint_i = (self.waypoint_i + 1) % len(self.waypoints)
        return self.get_current_navigation_action()

    def get_current_navigation_action(self):
        return TaskAction(type="navigate", target_location=self.get_current_location())

    def get_current_location(self):
        return self.waypoints[self.waypoint_i]

    def go_to_next_wp(self):
        nav_goal = ExecuteTaskAction.Goal()
        nav_goal.action = self.get_next_navigation_action()
        nav_goal.action.robot = "robot"
        nav_goal.realtime_factor = 1.0 if self.realtime else -1.0

        goal_future = self.execute_action_client.send_goal_async(nav_goal)
        rclpy.spin_until_future_complete(self.node, goal_future)

        result_future = goal_future.result().get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
