from typing import List, Dict

from pyrobosim_ros_gym.envs.banana import BananaEnv
from pyrobosim_ros_gym.envs.greenhouse import GreenhouseEnv
from pyrobosim_ros_gym.envs.pyrobosim_ros_env import PyRoboSimRosEnv

ENV_CLASS_FROM_NAME: Dict[str, type[PyRoboSimRosEnv]] = {
    "Banana": BananaEnv,
    "Greenhouse": GreenhouseEnv,
}


def available_envs_w_subtype() -> List[str]:
    envs: List[str] = []
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        for sub_type in env_class.sub_type:
            envs.append("".join((name, sub_type.name)))
    return envs


def available_env_classes() -> List[str]:
    return list(ENV_CLASS_FROM_NAME.keys())


def get_env_ENV_CLASS_FROM_NAME(req_name: str):
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        if req_name.startswith(name):
            return env_class
    raise RuntimeError(f"No environment found for {req_name}.")


def get_env_by_name(
    req_name: str, node, max_steps_per_episode, realtime, discrete_actions
) -> PyRoboSimRosEnv:
    base_class = None
    sub_type_str = None
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        if req_name.startswith(name):
            base_class = env_class
            sub_type_str = req_name.replace(name, "")
            break
    assert base_class == get_env_ENV_CLASS_FROM_NAME(req_name)
    if base_class is None:
        raise RuntimeError(f"No environment found for {req_name}.")
    sub_type = None
    for st in base_class.sub_type:
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
