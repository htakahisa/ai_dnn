"""Guard observations and learned movement, facing and utility actions."""

import math

import numpy as np
import torch
from torch import nn

from game_core import FACING_DIRECTIONS, FACING_VECTORS
from grid_lines import line_cells
from concon_v1.co1_attacker_common import bfs_distance_map, GORIGONS

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
ABILITIES = ("SMOKE", "FLASH", "RECON")
TARGET_COUNT = 3  # planted spike, assigned facing point, known enemy
ACTION_DIM = (len(MOVES) + len(ABILITIES) * TARGET_COUNT) * 8
WAIT_ACTION = 4 * 8
MAP_CHANNELS = 6
FEATURE_DIM = 34 + 5 * 15 + 5 * 16
MEMORY_TICKS = 12


def observation_dim(scenario):
    return MAP_CHANNELS * scenario.grid.size + FEATURE_DIM


def facing_onehot(direction):
    return [float(direction == item) for item in FACING_DIRECTIONS]


def aim_alignment(source, target, direction):
    dr, dc = target[0] - source[0], target[1] - source[1]
    length = math.hypot(dr, dc)
    if not length:
        return 1.0
    vx, vy = FACING_VECTORS[direction]
    return (dc * vx + dr * vy) / length


def clear_shot(char, enemy, chars, grid, smoke, game=None):
    """Geometry from perceived positions only, with the engine's smoke rules."""
    start, end = tuple(char.pos), tuple(enemy.pos)
    height, width = grid.shape
    if not all(0 <= r < height and 0 <= c < width for r, c in (start, end)):
        return False
    cells = line_cells(start, end)
    if any(grid[cell] == 1 for cell in cells):
        return False
    ignore_smoke = (getattr(char, "sees_through_smoke", False)
                    or getattr(enemy, "reveal_remaining", 0) > 0)
    if len(cells) > 2 and not ignore_smoke and any(cell in smoke for cell in cells):
        return False
    occupied = {tuple(other.pos) for other in chars
                if other.name not in (char.name, enemy.name)
                and getattr(other, "is_alive", True)
                and getattr(other, "position_known", True)}
    return not occupied.intersection(cells[1:-1])


class GuardDQN(nn.Module):
    def __init__(self, scenario):
        super().__init__()
        self.height, self.width = scenario.grid.shape
        self.map_size = MAP_CHANNELS * scenario.grid.size
        self.encoder = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * 4 * 6 + FEATURE_DIM, 256),
                                  nn.ReLU(), nn.Linear(256, ACTION_DIM))

    def forward(self, observations):
        maps = observations[:, :self.map_size].reshape(-1, MAP_CHANNELS, self.height, self.width)
        return self.head(torch.cat((self.encoder(maps), observations[:, self.map_size:]), dim=1))


def build_inputs(controller, char, state):
    scenario = controller.scenario
    grid = np.asarray(state["grid"])
    height, width = grid.shape
    tick = int(state.get("battle_tick", 0))
    chars = state.get("chars", [])
    slot = controller.assignments[char.name]
    goal, aim = scenario.positions[slot], scenario.facing_points[slot]
    spike = state.get("planted_pos")
    spike = tuple(map(int, spike)) if spike is not None else goal
    position = tuple(map(int, char.pos))
    smoke = set(map(tuple, state.get("smoke_cells", ())))
    tap_info = state.get("defender_defuse_info") or {}
    tap_progress = max((float(value[0]) / max(1, float(value[1]))
                        for value in tap_info.values()), default=0.0)
    known = [other for other in chars if other.team != char.team
             and getattr(other, "is_alive", True) and getattr(other, "position_known", True)
             and 0 <= other.pos[0] < height and 0 <= other.pos[1] < width]
    # Store only IQ-filtered disclosures; never read hidden live enemy positions.
    for enemy in known:
        controller.sightings[enemy.name] = (tuple(enemy.pos), tick)
    for name, (_, seen_tick) in list(controller.sightings.items()):
        if tick < seen_tick or tick - seen_tick >= MEMORY_TICKS:
            del controller.sightings[name]
    fireable = [enemy for enemy in known if clear_shot(char, enemy, chars, grid, smoke)]
    fireable.sort(key=lambda enemy: (
        -float(tap_info.get(enemy.name, (0, 1))[0]),
        max(abs(enemy.pos[0] - position[0]), abs(enemy.pos[1] - position[1])), enemy.name))
    target = fireable[0] if fireable else None
    known.sort(key=lambda enemy: (-float(tap_info.get(enemy.name, (0, 1))[0]),
                                 math.dist(position, enemy.pos), enemy.name))
    targets = (spike, aim, tuple(known[0].pos) if known else None)
    stopped = controller.stationary_ticks(char, tick)
    distance_goal = int(bfs_distance_map(grid, goal)[position])
    spike_in_bounds = 0 <= spike[0] < height and 0 <= spike[1] < width
    distance_spike = int(bfs_distance_map(grid, spike)[position]) if spike_in_bounds else -1
    spike_line = line_cells(position, spike) if spike_in_bounds else []
    spike_clear = bool(spike_line and all(grid[cell] != 1 for cell in spike_line)
                       and (len(spike_line) <= 2 or not any(cell in smoke for cell in spike_line)))

    maps = np.zeros((MAP_CHANNELS, height, width), dtype=np.float32)
    maps[0] = np.where(grid == 1, 1.0, np.where(grid == 2, 0.5, 0.0))
    for r, c in smoke:
        if 0 <= r < height and 0 <= c < width:
            maps[1, r, c] = 1
    maps[2, position[0], position[1]] = 1
    maps[5, goal[0], goal[1]] = 1
    for other in chars:
        if other.team == char.team and getattr(other, "is_alive", True):
            r, c = map(int, other.pos)
            if 0 <= r < height and 0 <= c < width:
                maps[3, r, c] = 1
    for enemy_pos, seen_tick in controller.sightings.values():
        r, c = map(int, enemy_pos)
        if 0 <= r < height and 0 <= c < width:
            maps[4, r, c] = 1 - (tick - seen_tick) / MEMORY_TICKS

    def normalized(pos):
        return [pos[0] / height, pos[1] / width]

    features = (normalized(position) + normalized(goal) + normalized(aim) + normalized(spike)
                + [distance_goal / 100, distance_spike / 100,
                   char.hp / max(1, char.max_hp), float(state.get("detonate_timer", 0)) / 50,
                   float(tap_progress > 0), tap_progress, min(stopped, 4) / 4,
                   float(getattr(char, "moved_this_tick", False)),
                   float(target is not None), float(spike_clear)]
                + facing_onehot(char.facing)
                + [min(2, getattr(char, ability.lower() + "_charges", 0)) / 2
                   for ability in ABILITIES]
                + [float(slot == letter) for letter in "abcde"])
    allies = {other.name: other for other in chars if other.team == char.team}
    for name in GORIGONS.players:
        ally = allies.get(name)
        if ally is None or not ally.is_alive:
            features.extend([0.0] * 15)
        else:
            features.extend([1.0] + normalized(ally.pos) + [ally.hp / max(1, ally.max_hp)]
                            + facing_onehot(ally.facing)
                            + [min(2, getattr(ally, ability.lower() + "_charges", 0)) / 2
                               for ability in ABILITIES])
    enemies = sorted([other for other in chars if other.team != char.team], key=lambda e: e.name)[:5]
    for index in range(5):
        enemy = enemies[index] if index < len(enemies) else None
        memory = controller.sightings.get(enemy.name) if enemy is not None else None
        disclosed = enemy is not None and enemy in known
        if enemy is None:
            features.extend([0.0] * 16)
            continue
        pos = tuple(enemy.pos) if disclosed else memory[0] if memory else None
        features.extend([float(pos is not None), float(enemy.is_alive)]
                        + (normalized(pos) if pos is not None else [0.0, 0.0])
                        + [enemy.hp / max(1, enemy.max_hp) if disclosed else 0.0]
                        + (facing_onehot(enemy.facing) if disclosed else [0.0] * 8)
                        + [float(getattr(enemy, "reveal_remaining", 0) > 0) if disclosed else 0.0,
                           float(tap_info.get(enemy.name, (0, 1))[0]) > 0,
                           (tick - memory[1]) / MEMORY_TICKS if memory else 1.0])
    observation = np.concatenate((maps.ravel(), np.asarray(features, dtype=np.float32)))
    if len(observation) != observation_dim(scenario):
        raise RuntimeError("guard feature dimension changed")
    mask = np.zeros(ACTION_DIM, dtype=bool)
    occupied = {tuple(other.pos) for other in chars if other.name != char.name
                and getattr(other, "is_alive", True) and getattr(other, "position_known", True)}
    for move_index, (dr, dc) in enumerate(MOVES):
        destination = (position[0] + dr, position[1] + dc)
        r, c = destination
        if (0 <= r < height and 0 <= c < width and grid[r, c] != 1
                and (move_index == 4 or destination not in occupied)):
            mask[move_index * 8:(move_index + 1) * 8] = True
    for ability_index, ability in enumerate(ABILITIES):
        if getattr(char, ability.lower() + "_charges", 0) <= 0:
            continue
        for target_index, destination in enumerate(targets):
            if destination is None:
                continue
            r, c = destination
            if not (0 <= r < height and 0 <= c < width) or grid[r, c] == 1:
                continue
            game = getattr(controller, "game", None)
            if ability != "SMOKE" and (destination == position or (
                    game is not None and len(game._projectile_path(position, destination)) <= 1)):
                continue
            start = (len(MOVES) + ability_index * TARGET_COUNT + target_index) * 8
            # The engine applies explicit facing to movement/turn actions;
            # ability ticks retain the current facing. Do not offer fictitious turns.
            mask[start + FACING_DIRECTIONS.index(char.facing)] = True
    context = {
        "position": position, "goal": goal, "aim": aim, "spike": spike,
        "distance_goal": distance_goal, "distance_spike": distance_spike,
        "tap": tap_progress > 0, "fireable": target is not None,
        "target": tuple(target.pos) if target is not None else None,
        "targets": targets, "stopped": stopped, "tick": tick,
    }
    return observation, mask, context


def decode_action(action, position, targets):
    operation, facing_index = divmod(int(action), 8)
    payload = {"facing": FACING_DIRECTIONS[facing_index]}
    if operation < len(MOVES):
        dr, dc = MOVES[operation]
        return [position[0] + dr, position[1] + dc], payload
    ability_index, target_index = divmod(operation - len(MOVES), TARGET_COUNT)
    payload.update(ability=ABILITIES[ability_index], target=targets[target_index])
    return list(position), payload
