from pyrobosim_ros_gym.envs.pyrobosim_ros_env import PyRoboSimRosEnv
from pyrobosim_ros_gym.envs.banana import BananaEnv

CLASS_BY_NAME = {"Banana": BananaEnv}


def get_env_by_name(
    req_name: str, node, max_steps_per_episode, realtime, discrete_actions
) -> PyRoboSimRosEnv:
    base_class = None
    sub_type = None
    for name, env_class in CLASS_BY_NAME.items():
        if req_name.startswith(name):
            base_class = env_class
            sub_type = req_name.replace(name, "")
            break
    if base_class is None:
        raise RuntimeError(f"No environment found for {req_name}")
    print(base_class.SUB_TYPE)
    print(sub_type)
    print("TODO")
    return base_class(
        base_class.SUB_TYPE.Place,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
    )
