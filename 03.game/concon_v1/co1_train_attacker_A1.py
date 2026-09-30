"""Train a shared DQN policy to route attackers through concon_v1 and plant left."""

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from map_data import NEW_MAZE_STR as GAME_MAZE_STR

try:
    from .co1_map_attacker_A1 import MAZE_STR as STRATEGY_MAZE_STR
except ImportError:
    from co1_map_attacker_A1 import MAZE_STR as STRATEGY_MAZE_STR


WAYPOINT_ORDER = "abcd"
SPLIT_PATTERNS = ((2, 3), (3, 2), (0, 5), (5, 0))
CARDINAL_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_WAIT = 4
ACTION_PLANT = 5
ACTION_DIM = 6
OBS_DIM = 28
MAX_TICKS = 100
PLANT_REQUIRED_TICKS = 4
DEFAULT_EPISODES = 5000
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "data" / "attacker_A1_data"


def _rows(map_text):
    rows = [line.strip() for line in map_text.strip().splitlines() if line.strip()]
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("map rows must be non-empty and have equal width")
    return rows


def parse_game_grid(map_text):
    """Read terrain digits only; strategic markers never become terrain."""
    rows = _rows(map_text)
    if any(not row.isdigit() for row in rows):
        raise ValueError("game terrain map must contain digits only")
    return np.asarray([[int(value) for value in row] for row in rows], dtype=np.int32)


def parse_strategy_points(map_text):
    """Keep marker coordinates in the strategy map's original row/column frame."""
    rows = _rows(map_text)
    points = {marker: [] for marker in WAYPOINT_ORDER}
    for row_index, row in enumerate(rows):
        for col_index, value in enumerate(row):
            if value in points:
                points[value].append((row_index, col_index))
            elif not value.isdigit():
                raise ValueError(f"unsupported strategy-map character: {value!r}")
    if any(not points[marker] for marker in WAYPOINT_ORDER):
        raise ValueError("strategy map must contain at least one a, b, c, and d point")
    return points


GRID = parse_game_grid(GAME_MAZE_STR)
HEIGHT, WIDTH = GRID.shape
WAYPOINT_POINTS = parse_strategy_points(STRATEGY_MAZE_STR)
if len(_rows(GAME_MAZE_STR)) != len(_rows(STRATEGY_MAZE_STR)) or WIDTH != len(_rows(STRATEGY_MAZE_STR)[0]):
    raise ValueError("strategy map dimensions must match the game terrain map")
if any(GRID[row, col] == 1 for cells in WAYPOINT_POINTS.values() for row, col in cells):
    raise ValueError("a strategy waypoint overlays a wall in the game terrain map")

LEFT_PLANT_CELLS = [
    (row, col)
    for row in range(HEIGHT)
    for col in range(WIDTH)
    if GRID[row, col] == 2 and col < WIDTH // 2
]
ATTACKER_SPAWNS = [
    (row, col)
    for row in range(HEIGHT)
    for col in range(WIDTH)
    if GRID[row, col] == 3
]
if not LEFT_PLANT_CELLS or len(ATTACKER_SPAWNS) != 5:
    raise ValueError("concon_v1 requires left plant cells and exactly five attacker spawns")


def bfs_distance_map(grid, goal):
    """Return cardinal movement distances through non-wall terrain."""
    height, width = grid.shape
    goal = tuple(map(int, goal))
    distances = np.full((height, width), -1, dtype=np.int32)
    if not (0 <= goal[0] < height and 0 <= goal[1] < width) or grid[goal] == 1:
        return distances
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for row_delta, col_delta in CARDINAL_MOVES:
            next_pos = (row + row_delta, col + col_delta)
            if (
                0 <= next_pos[0] < height
                and 0 <= next_pos[1] < width
                and grid[next_pos] != 1
                and distances[next_pos] < 0
            ):
                distances[next_pos] = distances[row, col] + 1
                queue.append(next_pos)
    return distances


def select_nearest_candidate(grid, start, candidates):
    """Select the first reachable candidate with the shortest wall-aware BFS distance."""
    candidates = [tuple(map(int, point)) for point in candidates]
    if not candidates:
        raise ValueError("at least one candidate is required")
    distances = bfs_distance_map(grid, start)
    reachable = [(int(distances[point]), index, point) for index, point in enumerate(candidates)
                 if distances[point] >= 0]
    if not reachable:
        raise ValueError(f"no candidate is reachable from {tuple(start)}")
    distance, index, point = min(reachable)
    return point, index, distance


def choose_split_assignment(rng, player_count=5, pattern_index=None):
    """Choose a round split and assign exactly its requested number to each a-point."""
    if player_count != 5:
        raise ValueError("concon_v1 route scenarios require five attackers")
    if pattern_index is None:
        pattern_index = rng.randrange(len(SPLIT_PATTERNS))
    if not 0 <= pattern_index < len(SPLIT_PATTERNS):
        raise ValueError("invalid split pattern index")
    group_counts = SPLIT_PATTERNS[pattern_index]
    groups = [0] * group_counts[0] + [1] * group_counts[1]
    rng.shuffle(groups)
    return pattern_index, groups


class RouteProgress:
    """Per-character route state; BFS chooses goals but never issues movement."""

    def __init__(self, group, pattern_index, start_pos, grid=GRID):
        self.group = int(group)
        self.pattern_index = int(pattern_index)
        self.stage = 0
        self.goal_index = self.group
        self.goal = WAYPOINT_POINTS["a"][self.group]
        self._set_goal_for_stage(tuple(start_pos), grid)

    def _set_goal_for_stage(self, pos, grid):
        if self.stage < len(WAYPOINT_ORDER):
            marker = WAYPOINT_ORDER[self.stage]
            if marker == "a":
                candidates = WAYPOINT_POINTS[marker]
                self.goal = candidates[self.group]
                self.goal_index = self.group
            else:
                self.goal, self.goal_index, _ = select_nearest_candidate(
                    grid, pos, WAYPOINT_POINTS[marker]
                )
        else:
            self.goal, self.goal_index, _ = select_nearest_candidate(
                grid, pos, LEFT_PLANT_CELLS
            )

    def advance_if_reached(self, pos, grid=GRID):
        if tuple(map(int, pos)) != self.goal or self.stage >= len(WAYPOINT_ORDER) + 1:
            return False
        self.stage += 1
        self._set_goal_for_stage(tuple(map(int, pos)), grid)
        return True

    @property
    def at_plant_stage(self):
        return self.stage == len(WAYPOINT_ORDER)


def build_action_mask(grid, pos, occupied_allies, is_carrier, at_plant_stage, plant_goal):
    """The actor sees only terrain and friendly occupancy, never hidden enemy positions."""
    row, col = map(int, pos)
    occupied = {tuple(map(int, point)) for point in occupied_allies}
    mask = np.zeros(ACTION_DIM, dtype=bool)
    for action, (row_delta, col_delta) in enumerate(CARDINAL_MOVES):
        next_pos = (row + row_delta, col + col_delta)
        mask[action] = (
            0 <= next_pos[0] < grid.shape[0]
            and 0 <= next_pos[1] < grid.shape[1]
            and grid[next_pos] != 1
            and next_pos not in occupied
        )
    mask[ACTION_WAIT] = True
    mask[ACTION_PLANT] = bool(
        is_carrier and at_plant_stage and (row, col) == tuple(plant_goal)
        and 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        and grid[row, col] == 2
    )
    return mask


def build_observation(route, pos, is_carrier, occupied_allies, plant_progress, elapsed_ticks, grid=GRID):
    """Shared actor observation; enemies and their unobserved coordinates are excluded."""
    row, col = map(int, pos)
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:2] = (row / max(1, HEIGHT - 1), col / max(1, WIDTH - 1))
    observation[2 + route.group] = 1.0
    observation[4 + route.pattern_index] = 1.0
    observation[8 + route.stage] = 1.0
    observation[13 + route.goal_index] = 1.0
    observation[18:20] = (route.goal[0] / max(1, HEIGHT - 1), route.goal[1] / max(1, WIDTH - 1))
    distance = bfs_distance_map(grid, route.goal)[row, col]
    observation[20] = max(0, int(distance)) / (HEIGHT + WIDTH)
    observation[21] = float(is_carrier)
    occupied = {tuple(map(int, point)) for point in occupied_allies}
    for action, (row_delta, col_delta) in enumerate(CARDINAL_MOVES):
        observation[22 + action] = float((row + row_delta, col + col_delta) in occupied)
    observation[26] = min(int(plant_progress), PLANT_REQUIRED_TICKS) / PLANT_REQUIRED_TICKS
    observation[27] = min(int(elapsed_ticks), MAX_TICKS) / MAX_TICKS
    return observation


class SharedRouteDQN(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, action_dim)
        )

    def forward(self, observations):
        features = self.features(observations)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class RouteEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.pattern_index, groups = choose_split_assignment(self.rng)
        self.positions = list(ATTACKER_SPAWNS)
        self.routes = [RouteProgress(group, self.pattern_index, pos) for group, pos in zip(groups, self.positions)]
        self.plant_progress = 0
        self.elapsed_ticks = 0
        self.done = False
        self.success = False
        return self._collect()

    def _collect(self):
        observations = []
        masks = []
        for index, (position, route) in enumerate(zip(self.positions, self.routes)):
            allies = [other for other_index, other in enumerate(self.positions) if other_index != index]
            observations.append(build_observation(
                route, position, index == 0, allies,
                self.plant_progress if index == 0 else 0, self.elapsed_ticks,
            ))
            masks.append(build_action_mask(
                GRID, position, allies, index == 0, route.at_plant_stage, route.goal,
            ))
        return observations, masks

    def step(self, actions):
        observations, masks = self._collect()
        previous_stages = [route.stage for route in self.routes]
        previous_distances = [
            int(bfs_distance_map(GRID, route.goal)[position])
            for route, position in zip(self.routes, self.positions)
        ]
        rewards = [-0.005] * len(self.positions)

        for index, action in enumerate(actions):
            if not masks[index][int(action)]:
                action = ACTION_WAIT
            row, col = self.positions[index]
            if action < len(CARDINAL_MOVES):
                row_delta, col_delta = CARDINAL_MOVES[action]
                destination = (row + row_delta, col + col_delta)
                if destination not in self.positions:
                    self.positions[index] = destination
                if index == 0 and self.positions[index] != (row, col):
                    self.plant_progress = 0
            elif action == ACTION_PLANT and index == 0:
                self.plant_progress += 1
                rewards[index] += 0.05
                if self.plant_progress >= PLANT_REQUIRED_TICKS:
                    self.done = True
                    self.success = True
            elif index == 0:
                self.plant_progress = 0

        for route, position in zip(self.routes, self.positions):
            route.advance_if_reached(position)

        self.elapsed_ticks += 1
        for index, (route, position) in enumerate(zip(self.routes, self.positions)):
            if route.stage != previous_stages[index]:
                rewards[index] += 0.25
            elif previous_distances[index] >= 0:
                distance = int(bfs_distance_map(GRID, route.goal)[position])
                if distance >= 0:
                    rewards[index] += 0.04 * (previous_distances[index] - distance)

        if self.success:
            rewards = [reward + 10.0 for reward in rewards]
        elif self.elapsed_ticks >= MAX_TICKS:
            self.done = True
            rewards = [reward - 3.0 for reward in rewards]

        next_observations, next_masks = self._collect()
        return observations, masks, rewards, next_observations, next_masks, self.done


def _choose_action(model, observation, mask, epsilon, rng):
    valid_actions = np.flatnonzero(mask)
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions.tolist()))
    with torch.no_grad():
        values = model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0]
        values[~torch.as_tensor(mask, dtype=torch.bool)] = -torch.inf
        return int(values.argmax().item())


def _optimize(model, target, optimizer, replay, batch_size, gamma):
    if len(replay) < batch_size:
        return
    batch = random.sample(replay, batch_size)
    observations, actions, rewards, next_observations, next_masks, dones = zip(*batch)
    observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.int64)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_observations = torch.as_tensor(np.asarray(next_observations), dtype=torch.float32)
    next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool)
    dones = torch.as_tensor(dones, dtype=torch.float32)

    selected = model(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_policy = model(next_observations).masked_fill(~next_masks, -torch.inf)
        next_actions = next_policy.argmax(dim=1)
        next_values = target(next_observations).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        expected = rewards + gamma * next_values * (1.0 - dones)
    loss = nn.functional.smooth_l1_loss(selected, expected)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()


def train(episodes=DEFAULT_EPISODES, save_dir=DEFAULT_SAVE_DIR, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SharedRouteDQN()
    target = SharedRouteDQN()
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    replay = deque(maxlen=100_000)
    env = RouteEnv(seed)
    recent_success = deque(maxlen=100)
    best_rate = -1.0
    started = time.perf_counter()

    for episode in range(1, episodes + 1):
        observations, masks = env.reset()
        epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, episodes))
        total_reward = 0.0
        while not env.done:
            actions = [
                _choose_action(model, observations[index], masks[index], epsilon, rng)
                for index in range(len(observations))
            ]
            old_obs, old_masks, rewards, next_obs, next_masks, done = env.step(actions)
            for index, action in enumerate(actions):
                replay.append((old_obs[index], action, rewards[index], next_obs[index], next_masks[index], float(done)))
                total_reward += rewards[index]
            observations, masks = next_obs, next_masks
            _optimize(model, target, optimizer, replay, 128, 0.99)
            if len(replay) and len(replay) % 1000 == 0:
                target.load_state_dict(model.state_dict())

        recent_success.append(float(env.success))
        success_rate = sum(recent_success) / len(recent_success)
        if episode % 20 == 0:
            print(
                f"episode={episode}/{episodes} success100={success_rate:.3f} "
                f"reward={total_reward:.2f} ticks={env.elapsed_ticks} epsilon={epsilon:.3f} "
                f"elapsed={time.perf_counter() - started:.1f}s"
            )
        if episode % 100 == 0 or episode == episodes:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": OBS_DIM,
                "n_actions": ACTION_DIM,
                "episode": episode,
                "success_rate": success_rate,
                "split_patterns": SPLIT_PATTERNS,
                "waypoint_points": WAYPOINT_POINTS,
                "left_plant_cells": LEFT_PLANT_CELLS,
            }
            torch.save(checkpoint, save_dir / "co1_attacker_A1_latest.pt")
            if success_rate >= best_rate:
                best_rate = success_rate
                torch.save(checkpoint, save_dir / "co1_attacker_A1_best.pt")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    args = parser.parse_args()
    train(args.episodes, args.save_dir, args.seed)


if __name__ == "__main__":
    main()