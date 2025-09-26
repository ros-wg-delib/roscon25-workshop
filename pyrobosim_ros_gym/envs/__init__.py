from typing import List, Dict

from pyrobosim_ros_gym.envs.banana import BananaEnv
from pyrobosim_ros_gym.envs.greenhouse import GreenhouseEnv
from pyrobosim_ros_gym.envs.pyrobosim_ros_env import PyRoboSimRosEnv
import rclpy

ENV_CLASS_FROM_NAME: Dict[str, type[PyRoboSimRosEnv]] = {
    "Banana": BananaEnv,
    "Greenhouse": GreenhouseEnv,
}


def available_envs_w_subtype() -> List[str]:
    """Return a list of environment types including subtypes."""
    envs: List[str] = []
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        for sub_type in env_class.sub_type:
            envs.append("".join((name, sub_type.name)))
    return envs


def available_env_classes() -> List[str]:
    """Return names of environment classes"""
    return list(ENV_CLASS_FROM_NAME.keys())


def get_env_env_class_from_name(req_name: str):
    """Return the class of a chosen environment name (ignoring `sub_type`s)."""
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        if req_name.startswith(name):
            return env_class
    raise RuntimeError(f"No environment found for {req_name}.")


def get_env_by_name(
    env_name: str,
    node: rclpy.Node,
    max_steps_per_episode: int,
    realtime: bool,
    discrete_actions: bool,
) -> PyRoboSimRosEnv:
    """
    Instantiate an environment class for a given type and `sub_type`.

    :param env_name: Name of environment, with subtype, e.g. BananaPick.
    :param node: Node instance needed for ROS communication.
    :param max_steps_per_episode: Limit the steps (when to end the episode).
    :param realtime: Whether actions take time.
    :param discrete_actions: Choose discrete actions (needed for DQN).
    """
    base_class = None
    sub_type_str = None
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        if env_name.startswith(name):
            base_class = env_class
            sub_type_str = env_name.replace(name, "")
            break
    assert base_class == get_env_env_class_from_name(env_name)
    if base_class is None:
        raise RuntimeError(f"No environment found for {env_name}.")
    sub_type = None
    for st in base_class.sub_types:
        if st.name == sub_type_str:
            sub_type = st
            break
    if sub_type is None:
        raise RuntimeError(f"No sub_type found for {sub_type_str} in {base_class}.")
    return base_class(
        sub_type,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
    )
