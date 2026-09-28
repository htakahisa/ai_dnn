"""Training-only shortest-path demonstrations for the fixed-map attacker.

This module is never loaded by a coach actor. It supplies behavior-cloning
labels from safe actor observations, with no enemy truth or game reference.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from map_data import NEW_MAZE_STR

from coach_v1.common.types import MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.training.scenario_generator import _distances


_ROWS = tuple(NEW_MAZE_STR.strip().splitlines())
_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


@lru_cache(maxsize=64)
def _distance_map(target: tuple[int, int]) -> dict[tuple[int, int], int]:
    return _distances(_ROWS, (target,))


def teacher_actions(observation, mask) -> tuple[CoachInstruction, ...]:
    """Return legal demonstration labels using only the actor's public view."""
    grid, slots = observation.grid, observation.vector[-70:].reshape(5, 14)
    dropped = np.argwhere(grid[20] > 0)
    planted = bool(np.any(grid[21] > 0))
    sites = np.argwhere(grid[2] > 0)
    carriers = [i for i in range(5) if slots[i, 0] and slots[i, 3]]
    if len(dropped):
        target = tuple(map(int, dropped[0]))
    elif carriers:
        carrier = slots[carriers[0]]
        position = (round(float(carrier[4]) * 25), round(float(carrier[5]) * 43))
        target = min((tuple(map(int, site)) for site in sites),
                     key=lambda site: (_distance_map(site).get(position, 10_000), site))
    else:
        target = tuple(map(int, sites[0]))
    distances = _distance_map(target)
    instructions = []
    for slot in range(5):
        objective = (ObjectiveAction.PLANT
                     if mask.objective[slot, tuple(ObjectiveAction).index(ObjectiveAction.PLANT)]
                     else ObjectiveAction.NONE)
        move_index = 0
        if slots[slot, 0] and not planted and objective is ObjectiveAction.NONE:
            row = round(float(slots[slot, 4]) * 25)
            col = round(float(slots[slot, 5]) * 43)
            current_distance = distances.get((row, col), 10_000)
            choices = [(distances.get((row + dr, col + dc), 10_000), index)
                       for index, (dr, dc) in enumerate(_DELTAS)
                       if mask.movement[slot, index]]
            best = min(choices)
            if best[0] < current_distance:
                move_index = best[1]
        instructions.append(CoachInstruction(
            tuple(MovementAction)[move_index], objective,
            TacticalIntent.HOLD if planted else TacticalIntent.ADVANCE,
        ))
    return tuple(instructions)
