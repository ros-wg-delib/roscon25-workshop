"""Utilities for the banana test environment."""

from pyrobosim_msgs.msg import ExecutionResult


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
    if not at_banana_location:
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
    holding_banana = False
    for obj in env.world_state.objects:
        if obj.category == "banana":
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

    # Slight reward penalty if the robot is not holding a banana.
    # This encourages the robot to try pick up bananas.
    if not terminated and not holding_banana:
        reward -= 0.1

    return reward, terminated, info
