"""Training penalties for idle route decisions, using the actor's perceived input."""

from concon_v1.co1_attacker_common import ACTION_WAIT, CARDINAL_MOVES

AVOIDABLE_WAIT_PENALTY = 0.1


def avoidable_wait_penalty(action, observation, mask, distance):
    # Contact stops/abilities bypass the route policy. Keep voluntary waiting
    # available in contact, at a goal, and when teammates block every move.
    if (action == ACTION_WAIT and distance > 0
            and any(mask[:len(CARDINAL_MOVES)])
            and not any(observation[-2:])):
        return AVOIDABLE_WAIT_PENALTY
    return 0.0
