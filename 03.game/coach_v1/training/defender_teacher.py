"""Training-only defender demonstrations derived from actor-safe observations."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from map_data import NEW_MAZE_STR
from map_data_defender_setup import DEFENDER_SETUP_TICKS

from coach_v1.common.types import MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import COACH_GRID_CHANNELS, COACH_VECTOR_FIELDS
from coach_v1.training.scenario_generator import _distances


_ROWS = tuple(NEW_MAZE_STR.strip().splitlines())
_DELTAS = ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1))


@lru_cache(maxsize=128)
def _distance_map(target: tuple[int, int]) -> dict[tuple[int, int], int]:
    return _distances(_ROWS, (target,))


def teacher_actions(observation, mask) -> tuple[CoachInstruction, ...]:
    """Return legal labels using no live game or hidden enemy coordinates."""
    grid = observation.grid
    fields = {name: index for index, name in enumerate(COACH_VECTOR_FIELDS)}
    slots = observation.vector[-70:].reshape(5, 14)
    positions = [_slot_position(slot) for slot in slots]
    alive = [index for index, slot in enumerate(slots) if slot[0]]
    setup = bool(observation.vector[fields["defender_setup"]])
    planted_cells = np.argwhere(grid[COACH_GRID_CHANNELS.index("spike_planted")] > 0)
    sightings = np.argwhere(grid[COACH_GRID_CHANNELS.index("current_enemy_sighting")] > 0)

    objectives = [ObjectiveAction.NONE] * 5
    intents = [TacticalIntent.HOLD] * 5
    if setup:
        targets = _assign_distinct(positions, alive, _setup_targets(grid))
    elif len(planted_cells):
        planted = tuple(map(int, planted_cells[0]))
        eligible = [slot for slot in alive
                    if mask.objective[slot, tuple(ObjectiveAction).index(ObjectiveAction.DEFUSE)]]
        if eligible:
            defuser = min(eligible, key=lambda slot: (_chebyshev(positions[slot], planted), slot))
            objectives[defuser] = ObjectiveAction.DEFUSE
            targets = _retake_targets(planted, positions, alive, defuser=defuser)
        else:
            # Before anyone can defuse, converge visibly as a team. Assigning
            # a latent "future defuser" here made the actor approximation lose
            # the only site entrant while four allies waited on a guard ring.
            targets = {slot: planted for slot in alive}
        detonation = float(observation.vector[fields["detonation_time_remaining"]])
        nearby = sum(_chebyshev(positions[slot], planted) <= 5 for slot in alive)
        for slot in alive:
            if objectives[slot] is ObjectiveAction.DEFUSE:
                intents[slot] = TacticalIntent.HOLD
            elif detonation >= 0.9 and nearby < 2:
                intents[slot] = TacticalIntent.WAIT_TEAM
            elif 0.55 <= detonation < 0.9 and slots[slot, 2]:
                intents[slot] = TacticalIntent.UTILITY_REQUEST
            else:
                intents[slot] = TacticalIntent.SUPPORT_ENTRY
    elif len(sightings):
        target = tuple(map(int, sightings[0]))
        targets = {slot: target for slot in alive}
        for slot in alive:
            intents[slot] = TacticalIntent.MULTI_PEEK
    else:
        last_seen = np.argwhere(grid[COACH_GRID_CHANNELS.index("last_seen_known")] > 0)
        if len(last_seen):
            age = grid[COACH_GRID_CHANNELS.index("last_seen_age")]
            target = min((tuple(map(int, cell)) for cell in last_seen),
                         key=lambda cell: (float(age[cell]), cell))
            targets = {slot: target for slot in alive}
        else:
            targets = _assign_distinct(positions, alive, _search_targets(grid))
        for slot in alive:
            intents[slot] = TacticalIntent.CLEAR_AREA

    instructions = []
    reserved = set()
    for slot in range(5):
        move_index = 0
        target = targets.get(slot) if slot in alive else None
        if target is not None and objectives[slot] is ObjectiveAction.NONE:
            stop_adjacent = len(sightings) and target in {
                tuple(map(int, cell)) for cell in sightings
            }
            move_index = _next_move(positions[slot], target, mask.movement[slot],
                                    reserved, stop_adjacent=bool(stop_adjacent))
            dr, dc = _DELTAS[move_index]
            reserved.add((positions[slot][0] + dr, positions[slot][1] + dc))
        instructions.append(CoachInstruction(
            tuple(MovementAction)[move_index], objectives[slot], intents[slot],
        ))
    return tuple(instructions)


def _slot_position(slot: np.ndarray) -> tuple[int, int]:
    return round(float(slot[4]) * 25), round(float(slot[5]) * 43)


def _setup_targets(grid: np.ndarray) -> list[tuple[int, int]]:
    importance = grid[COACH_GRID_CHANNELS.index("watch_importance")]
    defender_spawns = [(row, column) for row, cells in enumerate(_ROWS)
                       for column, value in enumerate(cells) if value == "4"]
    reachable = []
    for cell in map(tuple, np.argwhere(importance > 0)):
        distance = min(_distance_map(cell).get(spawn, 10_000)
                       for spawn in defender_spawns)
        if distance <= DEFENDER_SETUP_TICKS:
            reachable.append(cell)
    return _spread_targets(reachable, importance)


def _search_targets(grid: np.ndarray) -> list[tuple[int, int]]:
    importance = grid[COACH_GRID_CHANNELS.index("watch_importance")]
    confirmed = grid[COACH_GRID_CHANNELS.index("watch_confirmed")]
    age = grid[COACH_GRID_CHANNELS.index("watch_confirmation_age")]
    candidates = [tuple(map(int, cell)) for cell in np.argwhere(importance > 0)]
    candidates.sort(key=lambda cell: (float(confirmed[cell]), -float(age[cell]),
                                      -float(importance[cell]), cell))
    return candidates


def _spread_targets(candidates: list[tuple[int, int]],
                    importance: np.ndarray) -> list[tuple[int, int]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda cell: (cell[1], cell[0]))
    buckets = np.array_split(np.asarray(ordered), min(5, len(ordered)))
    return [max((tuple(map(int, cell)) for cell in bucket),
                key=lambda cell: (float(importance[cell]), -cell[0], -cell[1]))
            for bucket in buckets if len(bucket)]


def _retake_targets(planted: tuple[int, int], positions: list[tuple[int, int]],
                    alive: list[int], *, defuser: int) -> dict[int, tuple[int, int]]:
    distances = _distance_map(planted)
    ring = sorted((cell for cell, distance in distances.items() if 2 <= distance <= 4),
                  key=lambda cell: (distances[cell], cell))
    targets = {defuser: planted}
    guards = [slot for slot in alive if slot != defuser]
    targets.update(_assign_distinct(positions, guards, ring))
    return targets


def _assign_distinct(positions: list[tuple[int, int]], alive: list[int],
                     candidates: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    remaining = set(candidates)
    result = {}
    for slot in alive:
        if not remaining:
            break
        target = min(remaining, key=lambda cell: (
            _distance_map(cell).get(positions[slot], 10_000), cell,
        ))
        result[slot] = target
        remaining.remove(target)
    return result


def _next_move(position: tuple[int, int], target: tuple[int, int], legal: np.ndarray,
               reserved: set[tuple[int, int]], *, stop_adjacent: bool) -> int:
    distances = _distance_map(target)
    current = distances.get(position, 10_000)
    if current == 0 or stop_adjacent and current <= 1:
        return 0
    choices = []
    for index, (dr, dc) in enumerate(_DELTAS):
        destination = position[0] + dr, position[1] + dc
        if legal[index] and destination not in reserved:
            choices.append((distances.get(destination, 10_000), index))
    if not choices:
        return 0
    best = min(choices)
    return best[1] if best[0] < current else 0


def _chebyshev(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))
