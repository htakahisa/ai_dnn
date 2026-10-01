"""Train ConCon A1 routes in real 5v5 rounds (or the legacy route-only simulator)."""

import argparse
import math
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
from party_presets import get_preset
from controllers import BaseController
from game_core import FACING_VECTORS, SHOOTING_SITE_DIGREE

try:
    from .co1_map_attacker_A1 import MAZE_STR as STRATEGY_MAZE_STR
except ImportError:
    from co1_map_attacker_A1 import MAZE_STR as STRATEGY_MAZE_STR


WAYPOINT_ORDER = "abcd"
GORIGONS = get_preset("Gorigons")
SPIKE_CARRIER_INDEX = GORIGONS.players.index(GORIGONS.spike_holder)
SPLIT_PATTERNS = ((2, 3), (3, 2), (0, 5), (5, 0))
MAX_CANDIDATE_BFS_DISTANCE = 12
CARDINAL_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_WAIT = 4
ACTION_PLANT = 5
ACTION_DIM = 6
OBS_DIM = 28
MAX_TICKS = 100
PLANT_REQUIRED_TICKS = 4
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "data" / "attacker_A1_data"
TARGET_UPDATE_INTERVAL = 1000

DEFAULT_EPISODES = 1000
CHECKPOINT_INTERVAL = 50  # bestモデル算出episode間隔
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_RATIO = 0.7


def epsilon_by_episode(episode, total_episodes=DEFAULT_EPISODES):
    decay_episodes = max(1, int(total_episodes * EPSILON_DECAY_RATIO))
    fraction = min(max(float(episode) / decay_episodes, 0.0), 1.0)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * fraction


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


def select_nearest_candidate(
    grid, start, candidates, max_distance=MAX_CANDIDATE_BFS_DISTANCE,
):
    """Select the nearest reachable candidate, optionally within a BFS limit."""
    candidates = [tuple(map(int, point)) for point in candidates]
    if not candidates:
        raise ValueError("at least one candidate is required")
    distances = bfs_distance_map(grid, start)
    reachable = [(int(distances[point]), index, point) for index, point in enumerate(candidates)
                 if 0 <= distances[point]
                 and (max_distance is None or distances[point] <= max_distance)]
    if not reachable:
        if max_distance is None:
            raise ValueError(f"no candidate is reachable from {tuple(start)}")
        raise ValueError(f"no candidate within {max_distance} BFS steps from {tuple(start)}")
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
                grid, pos, LEFT_PLANT_CELLS, max_distance=None
            )
        self.distance_map = bfs_distance_map(grid, self.goal)

    def set_stage(self, stage, pos, grid=GRID, goal=None, goal_index=None):
        self.stage = int(stage)
        if goal is None:
            self._set_goal_for_stage(tuple(map(int, pos)), grid)
            return
        self.goal = tuple(map(int, goal))
        self.goal_index = int(goal_index)
        self.distance_map = bfs_distance_map(grid, self.goal)

    def advance_if_reached(self, pos, grid=GRID):
        if tuple(map(int, pos)) != self.goal or self.stage >= len(WAYPOINT_ORDER) + 1:
            return False
        self.set_stage(self.stage + 1, pos, grid)
        return True

    @property
    def at_plant_stage(self):
        return self.stage >= len(WAYPOINT_ORDER)


def build_action_mask(
    grid, pos, occupied_allies, is_carrier, at_plant_stage, plant_goal,
    route_distance_map=None, wait_for_other_group=False,
):
    """Allow movement toward the current goal, or wait if progress is blocked."""
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
    if route_distance_map is not None:
        current_distance = int(route_distance_map[row, col])
        if current_distance >= 0:
            for action, (row_delta, col_delta) in enumerate(CARDINAL_MOVES):
                next_pos = (row + row_delta, col + col_delta)
                if mask[action] and int(route_distance_map[next_pos]) != current_distance - 1:
                    mask[action] = False
    mask[ACTION_WAIT] = True
    if wait_for_other_group:
        mask[:len(CARDINAL_MOVES)] = False
    mask[ACTION_PLANT] = bool(
        is_carrier and at_plant_stage and (row, col) == tuple(plant_goal)
        and 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
        and grid[row, col] == 2
    )
    return mask


def plant_stage_action_mask(grid, pos, occupied_allies, carrier_pos, carrier_goal):
    """Park an escort, moving it aside only if it blocks the carrier's route."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[ACTION_WAIT] = True
    if carrier_pos is None or carrier_goal is None:
        return mask
    from_carrier = bfs_distance_map(grid, carrier_pos)
    to_goal = bfs_distance_map(grid, carrier_goal)
    total = int(from_carrier[carrier_goal])
    pos = tuple(map(int, pos))
    if total < 0 or int(from_carrier[pos]) + int(to_goal[pos]) != total:
        return mask
    occupied = {tuple(map(int, ally)) for ally in occupied_allies}
    for action, (dr, dc) in enumerate(CARDINAL_MOVES):
        next_pos = (pos[0] + dr, pos[1] + dc)
        if (not (0 <= next_pos[0] < grid.shape[0]
                 and 0 <= next_pos[1] < grid.shape[1])
                or grid[next_pos] in (1, 2) or next_pos in occupied):
            continue
        if int(from_carrier[next_pos]) + int(to_goal[next_pos]) != total:
            mask[ACTION_WAIT] = False
            mask[action] = True
            break
    return mask


def advance_team_routes(routes, positions, alive, completed_a_groups,
                        grid=GRID, carrier_index=SPIKE_CARRIER_INDEX):
    """Release a when each surviving split group has arrived, then share later goals."""
    active = [i for i, is_alive in enumerate(alive) if is_alive]
    if not active:
        return
    required_groups = {routes[i].group for i in active}
    for i in active:
        if routes[i].stage == 0 and tuple(positions[i]) == routes[i].goal:
            completed_a_groups.add(routes[i].group)
    if not required_groups.issubset(completed_a_groups):
        return

    if any(routes[i].stage == 0 for i in active):
        arrived = next(
            (i for i in active if routes[i].stage == 0
             and tuple(positions[i]) == routes[i].goal), None
        )
        a_goal = (tuple(positions[arrived]) if arrived is not None else
                  WAYPOINT_POINTS["a"][min(required_groups)])
        goal, goal_index, _ = select_nearest_candidate(
            grid, a_goal, WAYPOINT_POINTS["b"]
        )
        for i in active:
            routes[i].set_stage(1, positions[i], grid, goal, goal_index)

    for i in active:
        route = routes[i]
        if route.stage not in (1, 2, 3) or tuple(positions[i]) != route.goal:
            continue
        next_stage = route.stage + 1
        if next_stage < len(WAYPOINT_ORDER):
            goal, goal_index, _ = select_nearest_candidate(
                grid, positions[i], WAYPOINT_POINTS[WAYPOINT_ORDER[next_stage]]
            )
            for j in active:
                routes[j].set_stage(next_stage, positions[j], grid, goal, goal_index)
        else:
            for j in active:
                if j == carrier_index:
                    routes[j].set_stage(next_stage, positions[j], grid)
                else:
                    routes[j].set_stage(next_stage, positions[j], grid,
                                        goal=positions[j], goal_index=0)
        break

    for j in active:
        if routes[j].stage != len(WAYPOINT_ORDER):
            continue
        if j == carrier_index:
            if routes[j].goal not in LEFT_PLANT_CELLS:
                routes[j].set_stage(len(WAYPOINT_ORDER), positions[j], grid)
        else:
            routes[j].set_stage(len(WAYPOINT_ORDER), positions[j], grid,
                                goal=positions[j], goal_index=0)


def build_observation(route, pos, is_carrier, occupied_allies, plant_progress, elapsed_ticks, grid=GRID):
    """Shared actor observation; enemies and their unobserved coordinates are excluded."""
    row, col = map(int, pos)
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[0:2] = (row / max(1, HEIGHT - 1), col / max(1, WIDTH - 1))
    observation[2 + route.group] = 1.0
    observation[4 + route.pattern_index] = 1.0
    observation[8 + min(route.stage, len(WAYPOINT_ORDER))] = 1.0
    observation[13 + route.goal_index] = 1.0
    observation[18:20] = (route.goal[0] / max(1, HEIGHT - 1), route.goal[1] / max(1, WIDTH - 1))
    distance = route.distance_map[row, col]
    observation[20] = max(0, int(distance)) / (HEIGHT + WIDTH)
    observation[21] = float(is_carrier)
    occupied = {tuple(map(int, point)) for point in occupied_allies}
    for action, (row_delta, col_delta) in enumerate(CARDINAL_MOVES):
        observation[22 + action] = float((row + row_delta, col + col_delta) in occupied)
    observation[26] = min(int(plant_progress), PLANT_REQUIRED_TICKS) / PLANT_REQUIRED_TICKS
    observation[27] = min(int(elapsed_ticks), MAX_TICKS) / MAX_TICKS
    return observation


def _shot_visible(shooter, target, chars, grid, smoke_cells=(), game=None):
    """Use the game's firing line when available, including smoke and body blocking."""
    if game is not None and hasattr(game, "check_shot_line_of_sight"):
        if not game.check_shot_line_of_sight(shooter, target):
            return False
        if hasattr(game, "check_line_of_sight"):
            return (game.check_line_of_sight(shooter, target)
                    or bool(getattr(shooter, "sees_through_smoke", False))
                    or getattr(target, "reveal_remaining", 0) > 0)
        return True
    if not BaseController.has_line_of_sight(shooter.pos, target.pos, grid):
        return False
    # The fallback is used by the route tests, where no Game instance exists.
    row, col = map(int, shooter.pos)
    end_row, end_col = map(int, target.pos)
    dr, dc = abs(end_row - row), abs(end_col - col)
    step_r = 1 if row < end_row else -1
    step_c = 1 if col < end_col else -1
    error = dc - dr
    cells = []
    while True:
        cells.append((row, col))
        if (row, col) == (end_row, end_col):
            break
        doubled = 2 * error
        if doubled > -dr:
            error -= dr
            col += step_c
        if doubled < dc:
            error += dc
            row += step_r
    if len(cells) > 2 and any(cell in smoke_cells for cell in cells):
        return False
    occupied = {tuple(map(int, other.pos)) for other in chars
                if other is not shooter and other is not target
                and getattr(other, "is_alive", True)}
    return not occupied.intersection(cells[1:-1])


def choose_team_fire_target(shooter, chars, grid, smoke_cells=(), game=None):
    """Prefer a visible enemy that the most living allies can also shoot."""
    allies = [char for char in chars if getattr(char, "is_alive", True)
              and getattr(char, "team", None) == shooter.team]
    enemies = [char for char in chars if getattr(char, "is_alive", True)
               and getattr(char, "team", None) != shooter.team]
    visible = [enemy for enemy in enemies
               if _shot_visible(shooter, enemy, chars, grid, smoke_cells, game)]
    if not visible:
        return None
    return min(visible, key=lambda enemy: (
        -sum(_shot_visible(ally, enemy, chars, grid, smoke_cells, game)
             for ally in allies),
        max(abs(int(enemy.pos[0]) - int(shooter.pos[0])),
            abs(int(enemy.pos[1]) - int(shooter.pos[1]))),
        getattr(enemy, "hp", 100), str(enemy.name),
    ))


def facing_for_fire_target(shooter, target, chars, grid, smoke_cells=(), game=None):
    """Pick an aim direction that makes the game's automatic shot select target."""
    visible = [enemy for enemy in chars if getattr(enemy, "is_alive", True)
               and getattr(enemy, "team", None) != shooter.team
               and _shot_visible(shooter, enemy, chars, grid, smoke_cells, game)]
    if not visible:
        return getattr(shooter, "facing", "N")
    if target not in visible:
        # The selected enemy may move, die, or become blocked during this tick.
        target = min(visible, key=lambda enemy: (
            max(abs(enemy.pos[0] - shooter.pos[0]),
                abs(enemy.pos[1] - shooter.pos[1])),
            getattr(enemy, "hp", 100), enemy.name,
        ))

    def angle(direction, enemy):
        dc = float(enemy.pos[1] - shooter.pos[1])
        dr = float(enemy.pos[0] - shooter.pos[0])
        distance = math.hypot(dc, dr)
        if not distance:
            return 0.0
        fx, fy = FACING_VECTORS[direction]
        return math.degrees(math.acos(max(-1.0, min(1.0,
            (fx * dc + fy * dr) / distance))))

    candidates = []
    for direction in FACING_VECTORS:
        target_angle = angle(direction, target)
        if target_angle > SHOOTING_SITE_DIGREE:
            continue
        in_cone = [enemy for enemy in visible
                   if angle(direction, enemy) <= SHOOTING_SITE_DIGREE]
        if not in_cone:
            continue
        auto_target = min(in_cone, key=lambda enemy: (
            max(abs(enemy.pos[0] - shooter.pos[0]),
                abs(enemy.pos[1] - shooter.pos[1])),
            getattr(enemy, "hp", 100), enemy.name,
        ))
        candidates.append((auto_target is not target, target_angle, direction))
    return min(candidates)[2] if candidates else getattr(shooter, "facing", "N")


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
        self.alive = [True] * len(self.positions)
        self._a_completed_groups = set()
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
                route, position, index == SPIKE_CARRIER_INDEX, allies,
                self.plant_progress if index == SPIKE_CARRIER_INDEX else 0, self.elapsed_ticks,
            ))
            masks.append(build_action_mask(
                GRID, position, allies, index == SPIKE_CARRIER_INDEX,
                route.at_plant_stage, route.goal,
                route.distance_map,
                route.stage == 0 and position == route.goal
                and bool({self.routes[i].group for i, alive in enumerate(self.alive) if alive}
                         - self._a_completed_groups),
            ))
            if route.at_plant_stage and index != SPIKE_CARRIER_INDEX:
                masks[-1] = plant_stage_action_mask(
                    GRID, position, allies, self.positions[SPIKE_CARRIER_INDEX],
                    self.routes[SPIKE_CARRIER_INDEX].goal,
                )
        return observations, masks

    def _advance_routes_if_reached(self):
        advance_team_routes(self.routes, self.positions, self.alive, self._a_completed_groups)

    def step(self, actions):
        observations, masks = self._collect()
        previous_stages = [route.stage for route in self.routes]
        previous_distances = [
            int(route.distance_map[position])
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
                if index == SPIKE_CARRIER_INDEX and self.positions[index] != (row, col):
                    self.plant_progress = 0
            elif action == ACTION_PLANT and index == SPIKE_CARRIER_INDEX:
                self.plant_progress += 1
                rewards[index] += 0.05
                if self.plant_progress >= PLANT_REQUIRED_TICKS:
                    self.done = True
                    self.success = True
            elif index == SPIKE_CARRIER_INDEX:
                self.plant_progress = 0

        self._advance_routes_if_reached()

        self.elapsed_ticks += 1
        for index, (route, position) in enumerate(zip(self.routes, self.positions)):
            if route.stage != previous_stages[index]:
                rewards[index] += 0.25
            elif previous_distances[index] >= 0:
                distance = int(route.distance_map[position])
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


def summarize_team_plants(opponents, results):
    """Count plants and played episodes for each sampled opponent."""
    summary = {name: {"plants": 0, "episodes": 0} for name in dict.fromkeys(opponents)}
    for name, planted in results:
        summary[name]["episodes"] += 1
        summary[name]["plants"] += int(planted)
    for counts in summary.values():
        games = counts["episodes"]
        counts["plant_rate"] = counts["plants"] / games if games else None
    return summary


def format_team_plants(summary):
    parts = []
    for name, counts in summary.items():
        rate = counts["plant_rate"]
        rate_text = f"{rate:.3f}" if rate is not None else "-"
        parts.append(f"{name}={counts['plants']}/{counts['episodes']}({rate_text})")
    return " ".join(parts)


def train(episodes=DEFAULT_EPISODES, save_dir=DEFAULT_SAVE_DIR, seed=0,
          mode="battle", opponents=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = SharedRouteDQN()
    target = SharedRouteDQN()
    target.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    replay = deque(maxlen=100_000)
    if mode == "battle":
        from concon_v1.co1_battle_training import BattleRouteEnv
        env = BattleRouteEnv(seed, opponents)
    elif mode == "route":
        env = RouteEnv(seed)
    else:
        raise ValueError("mode must be 'battle' or 'route'")
    global_step = 0
    recent_success = deque(maxlen=100)
    recent_drops = deque(maxlen=100)
    recent_recoveries = deque(maxlen=100)
    total_plants = 0
    team_results = []
    recent_team_results = deque(maxlen=100)
    best_rate = -1.0
    started = time.perf_counter()

    for episode in range(1, episodes + 1):
        observations, masks = env.reset()
        epsilon = epsilon_by_episode(episode, episodes)
        total_reward = 0.0
        while not env.done:
            active_before = list(env.alive)
            actions = [
                _choose_action(model, observations[index], masks[index], epsilon, rng)
                for index in range(len(observations))
            ]
            if mode == "battle":
                transition = env.step(actions, current=(observations, masks))
            else:
                transition = env.step(actions)
            old_obs, old_masks, rewards, next_obs, next_masks, done = transition
            route_active = mode != "battle" or env.route_active_before_step
            route_interrupted = mode == "battle" and env.retrieve_active
            for index, action in enumerate(actions):
                if (not active_before[index] or not route_active
                        or (mode == "battle" and not env.policy_action_applied[index])):
                    continue
                # Retrieval is controlled by its own phase. A dropped spike
                # ends this route transition, but the real round continues.
                applied_action = env.actions[index] if mode == "battle" else action
                replay.append((old_obs[index], applied_action, rewards[index], next_obs[index],
                               next_masks[index], float(done or route_interrupted)))
                total_reward += rewards[index]
            observations, masks = next_obs, next_masks
            if route_active:
                global_step += 1
                _optimize(model, target, optimizer, replay, 128, 0.99)
                if global_step % TARGET_UPDATE_INTERVAL == 0:
                    target.load_state_dict(model.state_dict())

        planted = bool(env.success)
        total_plants += int(planted)
        recent_success.append(float(planted))
        if mode == "battle":
            result = (env.opponent, planted)
            team_results.append(result)
            recent_team_results.append(result)
            recent_drops.append(int(env.had_spike_drop))
            recent_recoveries.append(int(env.spike_recovered))
        success_rate = sum(recent_success) / len(recent_success)
        plant_rate_total = total_plants / episode
        if episode % 20 == 0:
            recovery = (f" recovered100={sum(recent_recoveries)}/{sum(recent_drops)}"
                        if mode == "battle" else "")
            print(
                f"episode={episode}/{episodes} success100={success_rate:.3f}"
                f" plant_total={total_plants}/{episode}"
                f" plant_rate_total={plant_rate_total:.3f}"
                f"{recovery} reward={total_reward:.2f} ticks={env.elapsed_ticks} "
                f"epsilon={epsilon:.3f} elapsed={time.perf_counter() - started:.1f}s"
            )
            if mode == "battle":
                print("  team_total " + format_team_plants(
                    summarize_team_plants(env.opponents, team_results)))
                print("  team100 " + format_team_plants(
                    summarize_team_plants(env.opponents, recent_team_results)))
        if episode % CHECKPOINT_INTERVAL == 0 or episode == episodes:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "obs_dim": OBS_DIM,
                "n_actions": ACTION_DIM,
                "episode": episode,
                "success_rate": success_rate,
                "plant_count_total": total_plants,
                "plant_rate_total": plant_rate_total,
                "split_patterns": SPLIT_PATTERNS,
                "waypoint_points": WAYPOINT_POINTS,
                "left_plant_cells": LEFT_PLANT_CELLS,
                "training_roster": GORIGONS.players,
                "spike_carrier": GORIGONS.spike_holder,
                "training_mode": mode,
                "attacker_perception": "iq" if mode == "battle" else "route_simulator",
                "opponents": env.opponents if mode == "battle" else (),
                "team_plant_total": (
                    summarize_team_plants(env.opponents, team_results)
                    if mode == "battle" else {}
                ),
                "team_plant100": (
                    summarize_team_plants(env.opponents, recent_team_results)
                    if mode == "battle" else {}
                ),
            }
            torch.save(checkpoint, save_dir / "co1_attacker_A1_latest.pt")
            if success_rate >= best_rate:
                best_rate = success_rate
                torch.save(checkpoint, save_dir / "co1_attacker_A1_best.pt")
                print(
                    f"Saved best model with success100={best_rate:.3f} "
                    f"plant_total={total_plants}/{episode} "
                    f"plant_rate_total={plant_rate_total:.3f} at episode {episode}"
                )
                if mode == "battle" and episode % 20 != 0:
                    print("  team_total " + format_team_plants(checkpoint["team_plant_total"]))
                    print("  team100 " + format_team_plants(checkpoint["team_plant100"]))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--mode", choices=("battle", "route"), default="battle")
    parser.add_argument("--opponents", nargs="+", choices=(
        "omoko_v1", "touyama_v2", "fnatic_v3", "gc_v1", "toru_ai_v3.1",
    ))
    args = parser.parse_args()
    train(args.episodes, args.save_dir, args.seed, args.mode, args.opponents)


if __name__ == "__main__":
    main()
