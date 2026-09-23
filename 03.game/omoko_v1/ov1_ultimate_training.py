"""Shared ultimate primitives for the six omoko_v1 simulation trainers.

The live game keeps ultimate points between rounds.  The small phase trainers
simulate one round at a time, so reset-time point sampling is used to expose
both ready and non-ready states to the policy.  Whether to cast remains a DQN
action; this module only provides observable context and common accounting.
"""

import random

import numpy as np

from ov1_map_data_base import NEW_MAZE_STR as BASE_MAZE_STR


ULTIMATE_BY_ABILITY = {
    "HUNT": "RAID",
    "SMOKE": "ESCAPE",
    "RECON": "MONITOR",
    "FLASH": "TUNNEL",
}
ULTIMATE_COST_BY_ABILITY = {
    "HUNT": 3,
    "SMOKE": 6,
    "RECON": 8,
    "FLASH": 5,
}

ULTIMATE_CONTEXT_DIM = 4
ULTIMATE_GOOD_USE_REWARD = 0.75
ULTIMATE_BAD_USE_PENALTY = -0.45
ORB_COLLECT_REQUIRED_TICKS = 5
ORB_ULTIMATE_POINTS = 2
ORB_COMPLETION_REWARD = 1.0


def _base_orb_cells():
    lines = [line.strip() for line in BASE_MAZE_STR.splitlines() if line.strip()]
    return tuple(
        (row, col)
        for row, line in enumerate(lines)
        for col, cell in enumerate(line)
        if cell == "5"
    )


BASE_ORB_CELLS = _base_orb_cells()


def valid_orb_cells(grid):
    """Base-map orb cells which are walkable in this phase's overlay map."""
    height, width = grid.shape
    return {
        cell for cell in BASE_ORB_CELLS
        if 0 <= cell[0] < height and 0 <= cell[1] < width and grid[cell] != 1
    }


def initialize_ultimate(unit, ability_name, rng=None, ready_probability=0.5):
    """Attach live-game-compatible ultimate fields to a simulation unit."""
    rng = rng or random
    ability_name = str(ability_name).upper()
    if ability_name not in ULTIMATE_BY_ABILITY:
        unit.ultimate_name = ""
        unit.ultimate_cost = 1
        unit.ultimate_points = 0
        unit.orb_collect_timer = 0
        return
    unit.ultimate_name = ULTIMATE_BY_ABILITY[ability_name]
    unit.ultimate_cost = ULTIMATE_COST_BY_ABILITY[ability_name]
    if rng.random() < ready_probability:
        unit.ultimate_points = unit.ultimate_cost
    else:
        unit.ultimate_points = rng.randrange(unit.ultimate_cost)
    unit.orb_collect_timer = 0


def ultimate_ready(unit):
    return (
        getattr(unit, "ultimate_cost", 0) > 0
        and getattr(unit, "ultimate_points", 0) >= unit.ultimate_cost
    )


def ultimate_context(unit, self_sighting=False, team_sighting=False, tactical=False):
    """Observable features: ready, own sighting, shared sighting, cast window."""
    return np.asarray(
        (
            float(ultimate_ready(unit)),
            float(bool(self_sighting)),
            float(bool(team_sighting)),
            float(bool(tactical)),
        ),
        dtype=np.float32,
    )


def spend_ultimate(unit):
    if not ultimate_ready(unit):
        return False
    unit.ultimate_points = 0
    return True


def ultimate_use_reward(tactical):
    return ULTIMATE_GOOD_USE_REWARD if tactical else ULTIMATE_BAD_USE_PENALTY


def ultimate_points_needed(unit):
    return max(0, int(getattr(unit, "ultimate_cost", 0)) - int(getattr(unit, "ultimate_points", 0)))


def orb_priority(unit, allies):
    """Whether this unit should get a nearby orb before its teammates.

    A one-orb pickup gives two points.  A unit that becomes ready from that
    pickup is always preferred; otherwise the smallest remaining deficit wins.
    """
    own_needed = ultimate_points_needed(unit)
    if own_needed <= 0:
        return False
    needed = [
        ultimate_points_needed(ally)
        for ally in allies
        if getattr(ally, "is_alive", getattr(ally, "alive", True))
        and ultimate_points_needed(ally) > 0
    ]
    if not needed:
        return False
    return own_needed == min(needed)


def nearest_orb(pos, available_orbs):
    if not available_orbs:
        return None
    return min(
        available_orbs,
        key=lambda cell: (max(abs(cell[0] - pos[0]), abs(cell[1] - pos[1])), cell),
    )


def orb_context(unit, available_orbs, grid, allies):
    """Observable orb state: need, availability, relative target, progress, priority."""
    pos = tuple(map(int, unit.pos))
    target = nearest_orb(pos, available_orbs)
    height, width = grid.shape
    needed = ultimate_points_needed(unit)
    if target is None:
        target_dr = target_dc = 0.0
        target_dist = 1.0
    else:
        target_dr = float(np.clip((target[0] - pos[0]) / max(1, height), -1, 1))
        target_dc = float(np.clip((target[1] - pos[1]) / max(1, width), -1, 1))
        target_dist = min(1.0, max(abs(target[0] - pos[0]), abs(target[1] - pos[1])) / max(height, width))
    return np.asarray(
        (
            min(1.0, needed / max(1, getattr(unit, "ultimate_cost", 1))),
            float(target is not None),
            target_dr,
            target_dc,
            target_dist,
            min(1.0, getattr(unit, "orb_collect_timer", 0) / ORB_COLLECT_REQUIRED_TICKS),
            float(orb_priority(unit, allies)),
        ),
        dtype=np.float32,
    )


def collect_orb_tick(unit, available_orbs):
    """Apply one stationary collection tick and return (completed, reward)."""
    pos = tuple(map(int, unit.pos))
    if pos not in available_orbs or ultimate_points_needed(unit) <= 0:
        unit.orb_collect_timer = 0
        return False, 0.0
    unit.orb_collect_timer = int(getattr(unit, "orb_collect_timer", 0)) + 1
    if unit.orb_collect_timer < ORB_COLLECT_REQUIRED_TICKS:
        return False, 0.0
    needed_before = ultimate_points_needed(unit)
    unit.ultimate_points = min(unit.ultimate_cost, unit.ultimate_points + ORB_ULTIMATE_POINTS)
    unit.orb_collect_timer = 0
    available_orbs.discard(pos)
    # Prefer a pickup which completes the ultimate, without rewarding a full unit.
    completed_ultimate = needed_before <= ORB_ULTIMATE_POINTS
    return True, ORB_COMPLETION_REWARD * (2.0 if completed_ultimate else 1.0)
