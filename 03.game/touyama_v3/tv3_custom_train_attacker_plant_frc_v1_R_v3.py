"""Deadline completion examples for FRC/right; never force policy actions."""
from functools import lru_cache
import numpy as np
from dataclasses import replace
from frc_v1.actions import MOVE_STEPS
from game_core import PLANT_REQUIRED_TICKS
from grid_paths import distance_map
from touyama_v3.tv3_custom_train_attacker_plant_frc_v1_R import (
    adjust_inputs as utility_inputs, _PublicGeometry)
from touyama_v3.tv3_learn_attacker_recon_cover import incoming_clear

EXTENSION_SPEC = dict(id='attacker_plant_frc_v1_R', revision=3,
                      role='attacker_plant', opponent='frc_v1', site='R')

# 通常設定：移動＋残り設置時間に対する余裕がこの値以下なら完遂を教える。
DEADLINE_SLACK_TICKS = 12
DEADLINE_IDLE_PENALTY = .25
DEADLINE_REVERSE_PENALTY = .15


@lru_cache(maxsize=4)
def _plant_distance(grid):
    board = np.asarray(grid)
    cells = np.argwhere((board == 2) & (np.indices(board.shape)[1] >= board.shape[1] // 2))
    if not len(cells):
        raise ValueError('Right plant extension requires right site floors')
    fields = [distance_map(board, tuple(p)) for p in cells]
    field = np.minimum.reduce([np.where(f >= 0, f, np.inf) for f in fields])
    return field


def _deadline(context):
    ally = context.ally
    if (context.role != 'attacker_plant' or context.opponent != 'frc_v1'
            or context.site != 'R' or ally is None or not ally.has_spike):
        return None
    field = _plant_distance(tuple(map(tuple, context.snapshot.grid)))
    distance = field[ally.position]
    work = max(1, PLANT_REQUIRED_TICKS - ally.plant_progress)
    if not np.isfinite(distance) or context.snapshot.round_timer > distance + work + DEADLINE_SLACK_TICKS:
        return None
    return field, work


@lru_cache(maxsize=64)
def _goal_distance(grid, goal):
    return distance_map(np.asarray(grid), goal)


def adjust_goal(context, goal):
    ally = context.ally
    if (context.role == 'attacker_plant' and context.opponent == 'frc_v1' and context.site == 'R'
            and ally is not None and not ally.has_spike and context.carrier_goal is not None
            and (ally.position == context.carrier_goal or goal == context.carrier_goal)):
        holder = next((a for a in context.snapshot.allies if a.alive and a.has_spike), None)
        if holder is not None and _deadline(replace(context, ally=holder)) is not None:
            geometry = _PublicGeometry(context.snapshot)
            board = np.asarray(context.snapshot.grid)
            grid = tuple(map(tuple, board))
            occupied = {a.position for a in context.snapshot.allies if a.alive and a.slot != ally.slot}
            choices = []
            for cell in np.argwhere((board == 2) & (np.indices(board.shape)[1] >= board.shape[1] // 2)):
                cell = tuple(map(int, cell))
                travel = _goal_distance(grid, cell)[ally.position]
                if cell == context.carrier_goal or cell in occupied or not 0 <= travel <= 4:
                    continue
                risk = sum(incoming_clear(geometry, context.snapshot, ally, cell, threat)
                           for threat in context.threats or context.snapshot.sightings)
                choices.append((risk, travel, cell))
            if choices:
                return min(choices)[-1]
    deadline = _deadline(context)
    if deadline is None:
        return goal
    geometry = _PublicGeometry(context.snapshot)
    field, work = deadline
    ally = context.ally
    threats = context.threats or context.snapshot.sightings
    choices = []
    board = np.asarray(context.snapshot.grid)
    grid = tuple(map(tuple, board))
    for cell in np.argwhere((board == 2) & (np.indices(board.shape)[1] >= board.shape[1] // 2)):
        cell = tuple(map(int, cell))
        travel = _goal_distance(grid, cell)[ally.position]
        if travel < 0 or travel + work > context.snapshot.round_timer:
            continue
        risk = sum(incoming_clear(geometry, context.snapshot, ally, cell, threat) for threat in threats)
        choices.append((risk, travel, cell != goal, cell))
    return min(choices)[-1] if choices else goal


def adjust_inputs(context, inputs):
    inputs = utility_inputs(context, inputs)
    deadline = _deadline(context)
    if deadline is None:
        return inputs
    field, work = deadline
    ally = context.ally
    if context.goal is not None and context.snapshot.grid[context.goal[0]][context.goal[1]] == 2:
        field = _goal_distance(tuple(map(tuple, context.snapshot.grid)), tuple(context.goal))
    # Already too late: do not label an impossible completion as a valid example.
    if field[ally.position] + work > context.snapshot.round_timer:
        return inputs
    if inputs.mask[56] and field[ally.position] == 0:
        inputs.teacher = 56
        return inputs
    candidates = []
    for i in range(40):
        action = inputs.actions[i]
        if not inputs.mask[i] or action.kind not in MOVE_STEPS:
            continue
        dr, dc = MOVE_STEPS[action.kind]
        p = ally.position[0] + dr, ally.position[1] + dc
        if field[p] < field[ally.position]:
            candidates.append((field[p], action.facing != inputs.combat.facing, i))
    if candidates:
        inputs.teacher = min(candidates)[-1]
    return inputs


def adjust_reward(context, reward):
    deadline = _deadline(context)
    action = context.action
    if deadline is None or action is None or action.kind == 'PLANT':
        return reward
    field, _ = deadline
    if context.goal is not None and context.snapshot.grid[context.goal[0]][context.goal[1]] == 2:
        field = _goal_distance(tuple(map(tuple, context.snapshot.grid)), tuple(context.goal))
    if action.kind in ('STAY', 'ABILITY', 'ULTIMATE', 'COLLECT_ORB'):
        return reward - DEADLINE_IDLE_PENALTY
    if action.kind in MOVE_STEPS:
        dr, dc = MOVE_STEPS[action.kind]
        p = context.ally.position[0] + dr, context.ally.position[1] + dc
        if field[p] >= field[context.ally.position]:
            return reward - DEADLINE_REVERSE_PENALTY
    return reward
