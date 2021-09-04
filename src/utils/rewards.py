import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reward_basic(depth_controlled_nodes, **kwargs):
    target_depths = kwargs.get(
        "target_depths", [5.0 for i in range(len(depth_controlled_nodes))]
    )
    depth_penalty = kwargs.get(
        "depth_penalty", [-1.0 for i in range(len(depth_controlled_nodes))]
    )
    depth_advantage = kwargs.get(
        "depth_advantage", [0.2 for i in range(len(depth_controlled_nodes))]
    )
    reward_depth = 0
    for i in range(len(depth_controlled_nodes)):
        temp = depth_controlled_nodes[i] - target_depths[i]
        if temp > 0:
            temp *= depth_penalty[i]
        else:
            # depth <= target depth
            temp *= -depth_advantage[i]
        reward_depth += temp
    """
    reward = (
        -10 * (flooding_after_step - flooding_before_step)
        - 30 * (abs(depth1 - 3.4) + depth1 - 3.4)
        - 100 * (abs(depth2 - 4.7) + depth2 - 4.7)
    )
    """
    return reward_depth


def reward_3obj_lowdepthpenalty(
    depth_controlled_nodes, action_list, action_step, old_action_step, **kwargs
):
    target_depths = kwargs.get(
        "target_depths", [5.0 for i in range(len(depth_controlled_nodes))]
    )
    low_target_depths = kwargs.get(
        "low_target_depths", [1.0 for i in range(len(depth_controlled_nodes))]
    )
    depth_penalty = kwargs.get(
        "depth_penalty", [-1.0 for i in range(len(depth_controlled_nodes))]
    )
    low_depth_penalty = kwargs.get(
        "low_depth_penalty", [-1.0 for i in range(len(depth_controlled_nodes))]
    )
    depth_advantage = kwargs.get(
        "depth_advantage", [0.2 for i in range(len(depth_controlled_nodes))]
    )
    energy_coeff = kwargs.get("energy_coeff", [-100])
    safety_coeff = kwargs.get("safety_coeff", [-10])
    k_coeff = kwargs.get("k_coeff", [-1 / 3, -1 / 3, -1 / 3])
    reward_depth = 0

    for i in range(len(depth_controlled_nodes)):
        temp = depth_controlled_nodes[i] - target_depths[i]
        temp2 = depth_controlled_nodes[i] - low_target_depths[i]
        if temp > 0:
            temp *= depth_penalty[i]
            reward_depth += temp - 20
        elif temp2 < 0:
            temp2 *= low_depth_penalty[i]
            reward_depth += temp2 - 50
        else:
            temp *= -depth_advantage[i]
            reward_depth += temp
    reward_energy = 0
    for j in range(3):
        reward_energy += action_list[action_step][j] * 0.0278
    for j in range(3, 7):
        reward_energy += action_list[action_step][j] * 0.167
    reward_energy *= energy_coeff[0]

    reward_safety = 0
    for j in range(7):
        reward_safety += abs(
            action_list[action_step][j] - action_list[old_action_step][j]
        )
    reward_safety *= safety_coeff[0]
    reward = (
        k_coeff[0] * reward_depth
        + k_coeff[1] * reward_energy
        + k_coeff[2] * reward_safety
    )

    return reward, reward_depth, reward_energy, reward_safety
