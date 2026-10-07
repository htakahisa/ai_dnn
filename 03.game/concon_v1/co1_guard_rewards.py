"""Postplant shaping; victory/defeat and death are applied by the rollout."""

from concon_v1.co1_guard_common import aim_alignment

REWARD_VERSION = 3
GAMMA = 0.99
DEATH_PENALTY = 0.5
ROUND_REWARD = 10.0
QUIET_UTILITY_COST = 0.08


def decision_reward(action, context, position, facing, new_distance):
    stationary = position == context["position"]
    operation = action // 8
    reward = -0.005
    if context["fireable"]:
        # Signed, continuous alignment distinguishes head-on aim from a
        # sideways shot and penalizes facing away. Moving still costs accuracy.
        alignment = aim_alignment(position, context["target"], facing)
        reward += 0.04 * alignment
        if stationary:
            if context["stopped"] >= 1:
                reward += 0.03 * max(0.0, alignment)
        else:
            reward -= 0.04
        return reward

    tapping = context["tap"]
    previous_distance = context["distance_spike"] if tapping else context["distance_goal"]
    if previous_distance >= 0 and new_distance >= 0:
        reward += 0.025 * (previous_distance - GAMMA * new_distance)
    if tapping:
        # Keep approach and utility available when smoke blocks the defuser.
        # A blocked actor is not penalized for an impossible movement.
        if stationary and operation < 5 and context["can_move"]:
            reward -= 0.03
        return reward

    # Spending a resource used to earn the same holding bonus as waiting,
    # and even avoided the off-post waiting penalty. Charge quiet casts while
    # leaving contact/defuse responses above unchanged. Outcomes can still
    # justify proactive utility; every executable cast remains a candidate.
    if operation >= 5:
        reward -= QUIET_UTILITY_COST
    reward += 0.02 * aim_alignment(position, context["aim"], facing)
    if not stationary:
        reward -= 0.02
        if context["position"] == context["goal"]:
            reward -= 0.06
    elif position == context["goal"]:
        reward += 0.04
    elif context["can_move"]:
        reward -= 0.03
    return reward
