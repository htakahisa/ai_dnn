"""Learned defender positioning values; no route search during inference.

The foundation supports movement and eight facing directions. Observations
retain terrain, allies, disclosed enemies and facing for subsequent search
training. This first skill uses phase, position, post and ally occupancy.
"""

import numpy as np
import torch
from torch import nn

from game_core import FACING_DIRECTIONS
from concon_v1.co1_guard_common import MOVES, GORIGONS, facing_onehot

ACTION_DIM = 40
MAP_CHANNELS = 6
FEATURE_DIM = 3 + 5 + 5 * 11 + 5 * 11 + 8


def observation_dim(scenario):
    return MAP_CHANNELS * scenario.grid.size + FEATURE_DIM


class DefenderSearchDQN(nn.Module):
    def __init__(self, scenario):
        super().__init__()
        self.height, self.width = scenario.grid.shape
        self.map_size = MAP_CHANNELS * scenario.grid.size
        self.phase_size = 5 * scenario.grid.size * 16
        self.navigation = True
        self.navigation_values = nn.Embedding(2 * self.phase_size, ACTION_DIM)
        self.navigation_yield_values = nn.Embedding(5 * 16, ACTION_DIM)
        nn.init.zeros_(self.navigation_values.weight)
        nn.init.zeros_(self.navigation_yield_values.weight)
        self.register_buffer("other_posts", torch.tensor([
            [positions[other][0] * self.width + positions[other][1]
             for other in "abcde" if other != letter]
            for positions in (scenario.positions, scenario.setup_positions) for letter in "abcde"]))

    def forward(self, observation):
        maps = observation[:, :self.map_size].reshape(-1, MAP_CHANNELS, self.height, self.width)
        features = observation[:, self.map_size:]
        rows = (features[:, 0] * self.height).round().long().clamp(0, self.height - 1)
        cols = (features[:, 1] * self.width).round().long().clamp(0, self.width - 1)
        setup = features[:, 2].long()
        slots = features[:, 3:8].argmax(1)
        occupied = (maps[:, 3] - maps[:, 2]).flatten(1).gather(1, self.other_posts[setup * 5 + slots]) > 0
        occupancy = (occupied.long() * observation.new_tensor([1, 2, 4, 8]).long()).sum(1)
        indices = setup * self.phase_size + (slots * self.height * self.width + rows * self.width + cols) * 16 + occupancy
        values = self.navigation_values(indices)
        allies = features[:, 8:63].reshape(-1, 5, 11)
        self_indices = ((allies[:, :, 1:3] - features[:, None, :2]).square().sum(2)
                        + (1 - allies[:, :, 0]) * 100).argmin(1)
        ally_rows = (allies[:, :, 1] * self.height).round().long()
        ally_cols = (allies[:, :, 2] * self.width).round().long()
        higher = ((torch.arange(5, device=observation.device)[None, :] < self_indices[:, None])
                  & (allies[:, :, 0] > 0))
        neighbors = torch.zeros(len(values), dtype=torch.long, device=observation.device)
        for bit, (dr, dc) in enumerate(MOVES[:4]):
            neighbor = higher & (ally_rows == rows[:, None] + dr) & (ally_cols == cols[:, None] + dc)
            neighbors += neighbor.any(1).long() * (1 << bit)
        preferred = values.reshape(-1, 5, 8).amax(2).argmax(1)
        return values + self.navigation_yield_values(preferred * 16 + neighbors)


def build_inputs(controller, char, state):
    scenario = controller.scenario
    grid = np.asarray(state["grid"])
    height, width = grid.shape
    if grid.shape != scenario.grid.shape or not np.array_equal(grid, scenario.grid):
        raise ValueError("defender positioning requires the checkpoint terrain")
    setup = bool(state.get("defender_setup_active", False))
    position = tuple(map(int, char.pos))
    slot = controller.assignments[char.name]
    maps = np.zeros((MAP_CHANNELS, height, width), dtype=np.float32)
    maps[0] = np.where(grid == 1, 1.0, np.where(grid == 2, .5, 0.0))
    maps[1] = (scenario.setup_grid == 1) & (grid != 1) if setup else 0
    maps[2, position[0], position[1]] = 1
    goal = (scenario.setup_positions if setup else scenario.positions)[slot]
    maps[5, goal[0], goal[1]] = 1
    chars = state.get("chars", [])
    allies = {other.name: other for other in chars if other.team == char.team}
    known_enemies = [other for other in chars if other.team != char.team
                     and other.is_alive and getattr(other, "position_known", False)
                     and 0 <= other.pos[0] < height and 0 <= other.pos[1] < width]
    for other in allies.values():
        if other.is_alive:
            maps[3, int(other.pos[0]), int(other.pos[1])] = 1
    for other in known_enemies:
        maps[4, int(other.pos[0]), int(other.pos[1])] = 1
    features = [position[0] / height, position[1] / width, float(setup)]
    features += [float(slot == letter) for letter in "abcde"]
    for name in GORIGONS.players:
        other = allies.get(name)
        features += ([1., other.pos[0] / height, other.pos[1] / width] + facing_onehot(other.facing)
                     if other is not None and other.is_alive else [0.] * 11)
    enemies = sorted(known_enemies, key=lambda other: other.name)[:5]
    for index in range(5):
        other = enemies[index] if index < len(enemies) else None
        features += ([1., other.pos[0] / height, other.pos[1] / width] + facing_onehot(other.facing)
                     if other is not None else [0.] * 11)
    features += facing_onehot(char.facing)
    observation = np.concatenate((maps.ravel(), np.asarray(features, dtype=np.float32)))
    mask = np.zeros(ACTION_DIM, dtype=bool)
    occupied = {tuple(other.pos) for other in allies.values() if other.name != char.name and other.is_alive}
    occupied.update(tuple(other.pos) for other in known_enemies)
    for move, (dr, dc) in enumerate(MOVES):
        r, c = position[0] + dr, position[1] + dc
        legal = (0 <= r < height and 0 <= c < width and grid[r, c] != 1
                 and (move == 4 or (r, c) not in occupied)
                 and (move == 4 or not setup or scenario.setup_grid[r, c] != 1))
        if legal:
            if getattr(char, "facing_forced_this_tick", False):
                mask[move * 8 + FACING_DIRECTIONS.index(char.facing)] = True
            else:
                mask[move * 8:(move + 1) * 8] = True
    return observation, mask
