"""Route observations, actions, progress, and network shared by all attacker maps."""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from controllers import BaseController
from game_core import FACING_VECTORS, SHOOTING_SITE_DIGREE
from party_presets import get_preset
from concon_v1.co1_attacker_scenarios import (
    GAME_MAZE_STR, WAYPOINT_ORDER, get_scenario,
    parse_game_grid, parse_strategy_points,
)

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
DEFAULT_SCENARIO = get_scenario("A1")
DEFAULT_SAVE_DIR = DEFAULT_SCENARIO.save_dir
STRATEGY_MAZE_STR = DEFAULT_SCENARIO.strategy_map
GRID = DEFAULT_SCENARIO.grid
HEIGHT, WIDTH = GRID.shape
WAYPOINT_POINTS = DEFAULT_SCENARIO.waypoint_points
LEFT_PLANT_CELLS = DEFAULT_SCENARIO.plant_cells
ATTACKER_SPAWNS = DEFAULT_SCENARIO.attacker_spawns


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


def _choose_action(model, observation, mask, epsilon, rng):
    valid_actions = np.flatnonzero(mask)
    if rng.random() < epsilon:
        return int(rng.choice(valid_actions.tolist()))
    with torch.no_grad():
        values = model(torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0))[0]
        values[~torch.as_tensor(mask, dtype=torch.bool)] = -torch.inf
        return int(values.argmax().item())


class RouteProgress:
    """Per-character route state; BFS chooses goals but never issues movement."""

    def __init__(self, group, pattern_index, start_pos, grid=None, scenario="A1"):
        self.scenario = get_scenario(scenario)
        grid = self.scenario.grid if grid is None else grid
        self.group = int(group)
        self.pattern_index = int(pattern_index)
        self.stage = 0
        self.goal_index = self.group
        self.goal = self.scenario.waypoint_points["a"][self.group]
        self._set_goal_for_stage(tuple(start_pos), grid)

    def _set_goal_for_stage(self, pos, grid):
        if self.stage < len(self.scenario.waypoint_order):
            marker = self.scenario.waypoint_order[self.stage]
            if marker == "a":
                candidates = self.scenario.waypoint_points[marker]
                self.goal = candidates[self.group]
                self.goal_index = self.group
            else:
                self.goal, self.goal_index, _ = select_nearest_candidate(
                    grid, pos, self.scenario.waypoint_points[marker],
                    max_distance=self.scenario.max_candidate_bfs_distance,
                )
        else:
            self.goal, self.goal_index, _ = select_nearest_candidate(
                grid, pos, self.scenario.plant_cells, max_distance=None
            )
        self.distance_map = bfs_distance_map(grid, self.goal)

    def set_stage(self, stage, pos, grid=None, goal=None, goal_index=None):
        grid = self.scenario.grid if grid is None else grid
        self.stage = int(stage)
        if goal is None:
            self._set_goal_for_stage(tuple(map(int, pos)), grid)
            return
        self.goal = tuple(map(int, goal))
        self.goal_index = int(goal_index)
        self.distance_map = bfs_distance_map(grid, self.goal)

    def advance_if_reached(self, pos, grid=None):
        if tuple(map(int, pos)) != self.goal or self.stage >= len(self.scenario.waypoint_order) + 1:
            return False
        self.set_stage(self.stage + 1, pos, grid)
        return True

    @property
    def at_plant_stage(self):
        return self.stage >= len(self.scenario.waypoint_order)


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
                        grid=None, carrier_index=SPIKE_CARRIER_INDEX):
    """Release a when each surviving split group has arrived, then share later goals."""
    active = [i for i, is_alive in enumerate(alive) if is_alive]
    if not active:
        return
    scenario = routes[active[0]].scenario
    waypoint_order = scenario.waypoint_order
    grid = scenario.grid if grid is None else grid
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
                  scenario.waypoint_points["a"][min(required_groups)])
        if len(waypoint_order) > 1:
            goal, goal_index, _ = select_nearest_candidate(
                grid, a_goal, scenario.waypoint_points[waypoint_order[1]],
                max_distance=scenario.max_candidate_bfs_distance,
            )
            for i in active:
                routes[i].set_stage(1, positions[i], grid, goal, goal_index)
        else:
            for i in active:
                if i == carrier_index:
                    routes[i].set_stage(1, positions[i], grid)
                else:
                    routes[i].set_stage(1, positions[i], grid,
                                        goal=positions[i], goal_index=0)

    for i in active:
        route = routes[i]
        if not 1 <= route.stage < len(waypoint_order) or tuple(positions[i]) != route.goal:
            continue
        next_stage = route.stage + 1
        if next_stage < len(waypoint_order):
            goal, goal_index, _ = select_nearest_candidate(
                grid, positions[i], scenario.waypoint_points[waypoint_order[next_stage]],
                max_distance=scenario.max_candidate_bfs_distance,
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
        if routes[j].stage != len(waypoint_order):
            continue
        if j == carrier_index:
            if routes[j].goal not in scenario.plant_cells:
                routes[j].set_stage(len(waypoint_order), positions[j], grid)
        else:
            routes[j].set_stage(len(waypoint_order), positions[j], grid,
                                goal=positions[j], goal_index=0)


def build_observation(route, pos, is_carrier, occupied_allies, plant_progress, elapsed_ticks, grid=None):
    """Shared actor observation; enemies and their unobserved coordinates are excluded."""
    row, col = map(int, pos)
    grid = route.scenario.grid if grid is None else grid
    height, width = grid.shape
    stage_count = len(route.scenario.waypoint_order) + 1
    goal_offset = 8 + stage_count
    status_offset = goal_offset + 5
    observation = np.zeros(route.scenario.obs_dim, dtype=np.float32)
    observation[0:2] = (row / max(1, height - 1), col / max(1, width - 1))
    observation[2 + route.group] = 1.0
    observation[4 + route.pattern_index] = 1.0
    observation[8 + min(route.stage, stage_count - 1)] = 1.0
    # Preserve A1's 28-feature model. Extra plant-cell indices share the last
    # index slot; the exact goal is still represented by its coordinates.
    observation[goal_offset + min(route.goal_index, 4)] = 1.0
    observation[status_offset:status_offset + 2] = (
        route.goal[0] / max(1, height - 1), route.goal[1] / max(1, width - 1),
    )
    distance = route.distance_map[row, col]
    observation[status_offset + 2] = max(0, int(distance)) / (height + width)
    observation[status_offset + 3] = float(is_carrier)
    occupied = {tuple(map(int, point)) for point in occupied_allies}
    for action, (row_delta, col_delta) in enumerate(CARDINAL_MOVES):
        observation[status_offset + 4 + action] = float((row + row_delta, col + col_delta) in occupied)
    observation[status_offset + 8] = min(int(plant_progress), PLANT_REQUIRED_TICKS) / PLANT_REQUIRED_TICKS
    observation[status_offset + 9] = min(int(elapsed_ticks), MAX_TICKS) / MAX_TICKS
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


