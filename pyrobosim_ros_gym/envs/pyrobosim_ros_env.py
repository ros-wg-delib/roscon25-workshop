import time

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
from geometry_msgs.msg import Point


def _dist(a: Point, b: Point) -> float:
    return np.linalg.norm([a.x - b.x, a.y - b.y, a.z - b.z])


class PyRoboSimRosEnv(gym.Env):
    """Gym environment wrapping around the PyRoboSim ROS Interface."""

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
        world_state = future.result().state

        self.all_locations = []
        for loc in world_state.locations:
            self.all_locations.extend(loc.spawns)
        self.num_locations = sum(len(loc.spawns) for loc in world_state.locations)
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}

        self.plants = [obj.name for obj in world_state.objects]
        # print(f"{self.plants=}")
        self.good_plants = [
            obj.name for obj in world_state.objects if obj.category == "plant_good"
        ]
        # print(f"{self.good_plants=}")

<<<<<<< HEAD
        self.num_actions = 2  # stay ducked or water plant
        self.action_space = spaces.Discrete(self.num_actions)
        # print(f"{self.action_space=}")
=======
        if self.discrete_actions:
            self.action_space = spaces.Discrete(self.num_actions)
        else:
            self.action_space = spaces.Box(
                low=np.zeros(self.num_actions, dtype=np.float32),
                high=np.ones(self.num_actions, dtype=np.float32),
            )
        print(f"{self.action_space=}")
>>>>>>> origin/continuous-actions

        # Observation space is defined by:
        self.max_n_objects = 3
        self.max_dist = 10
        # array of n objects with a class and distance each
        self.obs_size = (self.max_n_objects, 2)
        self.observation_space = spaces.Box(
            low=-1, high=self.max_dist, shape=self.obs_size
        )
        # print(f"{self.observation_space=}")

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
<<<<<<< HEAD
        info = {}
=======
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
>>>>>>> origin/continuous-actions
        truncated = self.step_number >= self.max_steps_per_episode
        if truncated:
            print(
                f"Maximum steps ({self.max_steps_per_episode}) exceeded. Truncated episode."
            )

<<<<<<< HEAD
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
=======
        observation = self._get_obs()
        reward, terminated, info = self.reward_fn(goal, action_result)
        self.previous_action_type = goal.action.type
>>>>>>> origin/continuous-actions

        return observation, reward, terminated, truncated, info
    
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
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
<<<<<<< HEAD
        print(f"Resetting environment {seed=}")
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
=======
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
>>>>>>> origin/continuous-actions

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
<<<<<<< HEAD
        
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
            reward = -.1

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
=======
>>>>>>> origin/continuous-actions
