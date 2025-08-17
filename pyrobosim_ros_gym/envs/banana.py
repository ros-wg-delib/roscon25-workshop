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
