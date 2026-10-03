"""Route observations, actions, progress, and network shared by all attacker maps."""

import math
import random
from collections import deque
from functools import lru_cache

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
    goal = tuple(map(int, goal))
    # Distances depend only on walls and the goal, not players or site labels.
    # Content keys also handle in-place terrain changes without stale results.
    walls = (grid == 1).tobytes()
    return _cached_bfs_distance_map(grid.shape, walls, goal).copy()


@lru_cache(maxsize=256)
def _cached_bfs_distance_map(shape, wall_bytes, goal):
    """Keep bounded, immutable distance maps shared across actors and rounds."""
    height, width = shape
    walls = np.frombuffer(wall_bytes, dtype=np.bool_).reshape(shape)
    distances = np.full((height, width), -1, dtype=np.int32)
    if not (0 <= goal[0] < height and 0 <= goal[1] < width) or walls[goal]:
        distances.setflags(write=False)
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
                and not walls[next_pos]
                and distances[next_pos] < 0
            ):
                distances[next_pos] = distances[row, col] + 1
                queue.append(next_pos)
    distances.setflags(write=False)
    return distances


def _reachable_candidates(
    grid, start, candidates, max_distance=MAX_CANDIDATE_BFS_DISTANCE,
):
    """Return reachable candidates with their BFS distances and original indices."""
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
    return sorted(reachable)


def select_nearest_candidate(
    grid, start, candidates, max_distance=MAX_CANDIDATE_BFS_DISTANCE,
):
    """Select the nearest reachable candidate, optionally within a BFS limit."""
    distance, index, point = _reachable_candidates(grid, start, candidates, max_distance)[0]
    return point, index, distance


def choose_split_assignment(rng, player_count=5, pattern_index=None, *, a_point_count=2,
                            balanced_only=False):
    """Send everyone to a single a-point, or sample the existing two-point split."""
    if player_count != 5:
        raise ValueError("concon_v1 route scenarios require five attackers")
    if a_point_count == 1:
        return SPLIT_PATTERNS.index((5, 0)), [0] * player_count
    if a_point_count != 2:
        raise ValueError("the first waypoint a must have one or two points")
    if pattern_index is None:
        pattern_index = rng.randrange(2 if balanced_only else len(SPLIT_PATTERNS))
    if not 0 <= pattern_index < len(SPLIT_PATTERNS):
        raise ValueError("invalid split pattern index")
    if balanced_only and pattern_index >= 2:
        raise ValueError("uppercase waypoints require a 2:3 or 3:2 split")
    group_counts = SPLIT_PATTERNS[pattern_index]
    groups = [0] * group_counts[0] + [1] * group_counts[1]
    rng.shuffle(groups)
    return pattern_index, groups


def _choose_action(model, observation, mask, epsilon, rng, *, route=None, position=None):
    valid_actions = np.flatnonzero(mask)
    if rng.random() < epsilon:
        # Unrestricted random walks rarely finish the longer A2/A3 routes
        # within a round. Explore among the best available route moves while
        # leaving the policy's full mask (including detours) unchanged.
        if route is not None and position is not None and not mask[ACTION_PLANT]:
            row, col = map(int, position)
            if int(route.distance_map[row, col]) == 0 and mask[ACTION_WAIT]:
                return ACTION_WAIT
            moves = [int(action) for action in valid_actions if action < len(CARDINAL_MOVES)]
            if moves:
                distances = {
                    action: int(route.distance_map[row + CARDINAL_MOVES[action][0],
                                                  col + CARDINAL_MOVES[action][1]])
                    for action in moves
                }
                reachable = [action for action in moves if distances[action] >= 0]
                if reachable:
                    nearest = min(distances[action] for action in reachable)
                    valid_actions = np.asarray([action for action in reachable
                                                if distances[action] == nearest])
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
        self.group = 0 if len(self.scenario.waypoint_points["a"]) == 1 else int(group)
        self.pattern_index = int(pattern_index)
        self.stage = 0
        self.completed_goals = frozenset()
        self.required_goals = frozenset()
        self.yield_for = None
        self.yield_origin = None
        self.yield_priority_goal = None
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
        if self.stage != int(stage):
            self.completed_goals = frozenset()
            self.required_goals = frozenset()
            self.yield_for = None
            self.yield_origin = None
            self.yield_priority_goal = None
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
    """Allow free moves that reduce the distance to the assigned goal."""
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
            for action, (dr, dc) in enumerate(CARDINAL_MOVES):
                next_pos = (row + dr, col + dc)
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
    if mask[ACTION_PLANT]:
        mask[:len(CARDINAL_MOVES)] = False
        mask[ACTION_WAIT] = False
    return mask


def plant_stage_action_mask(grid, pos, occupied_allies, carrier_pos, carrier_goal):
    """Park an escort or clear the carrier's path, including multi-step retreats."""
    return _yield_path_mask(grid, pos, occupied_allies, carrier_pos, carrier_goal)


def _yield_path_mask(grid, pos, occupied_allies, priority_pos, priority_goal):
    """Find a free pullout off the priority actor's shortest-path corridor."""
    mask = np.zeros(ACTION_DIM, dtype=bool)
    mask[ACTION_WAIT] = True
    if priority_pos is None or priority_goal is None:
        return mask
    priority_pos, priority_goal = tuple(priority_pos), tuple(priority_goal)
    from_priority = bfs_distance_map(grid, priority_pos)
    to_goal = bfs_distance_map(grid, priority_goal)
    total = int(from_priority[priority_goal])
    pos = tuple(map(int, pos))
    if total < 0 or int(from_priority[pos]) + int(to_goal[pos]) != total:
        return mask
    occupied = {tuple(map(int, ally)) for ally in occupied_allies}
    queue = deque([(pos, None)])
    visited = {pos}
    while queue:
        current, first_action = queue.popleft()
        for action, (dr, dc) in enumerate(CARDINAL_MOVES):
            next_pos = (current[0] + dr, current[1] + dc)
            if (not (0 <= next_pos[0] < grid.shape[0] and 0 <= next_pos[1] < grid.shape[1])
                    or grid[next_pos] == 1 or next_pos in occupied or next_pos in visited):
                continue
            first = action if first_action is None else first_action
            if int(from_priority[next_pos]) + int(to_goal[next_pos]) != total:
                mask[ACTION_WAIT] = False
                mask[first] = True
                return mask
            visited.add(next_pos)
            queue.append((next_pos, first))
    return mask


def build_team_route_action_mask(routes, positions, alive, index, grid,
                                 carrier_index, completed_a_groups):
    """Use identical movement and yielding rules in training and production."""
    route, pos = routes[index], tuple(positions[index])
    active = [i for i, living in enumerate(alive) if living]
    allies = [positions[i] for i in active if i != index]
    # Planting has priority over stale retreats and waypoint yielding.
    if index == carrier_index and route.at_plant_stage and pos == route.goal:
        planting = build_action_mask(grid, pos, allies, True, True, route.goal)
        if planting[ACTION_PLANT]:
            route.yield_for = route.yield_origin = route.yield_priority_goal = None
            return planting
    if route.at_plant_stage and index != carrier_index:
        return plant_stage_action_mask(
            grid, pos, allies, positions[carrier_index] if carrier_index is not None else None,
            routes[carrier_index].goal if carrier_index is not None else None,
        )
    if route.yield_for is not None:
        priority = route.yield_for
        if (priority in active and routes[priority].goal == route.yield_priority_goal
                and int(routes[priority].distance_map[tuple(positions[priority])])
                > int(routes[priority].distance_map[route.yield_origin])):
            # Continue the retreat even if the priority actor has not moved yet.
            # Once outside the corridor, hold there until it passes the vacated cell.
            return _yield_path_mask(grid, pos, allies, positions[priority], routes[priority].goal)
        route.yield_for = route.yield_origin = route.yield_priority_goal = None
    waiting_at_waypoint = (
        route.stage == 0 and pos == route.goal
        and not {routes[i].group for i in active}.issubset(completed_a_groups)
    )
    mask = build_action_mask(
        grid, pos, allies, index == carrier_index, route.at_plant_stage, route.goal,
        route.distance_map,
        waiting_at_waypoint,
    )
    occupied = {tuple(positions[i]) for i in active}
    for other in sorted((i for i in active if i != index),
                        key=lambda i: (i != carrier_index, i)):
        other_pos, other_route = tuple(positions[other]), routes[other]
        distance = int(other_route.distance_map[other_pos])
        if (distance <= 0 or abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1]) != 1
                or int(other_route.distance_map[pos]) != distance - 1):
            continue
        free_progress = any(
            0 <= other_pos[0] + dr < grid.shape[0] and 0 <= other_pos[1] + dc < grid.shape[1]
            and grid[other_pos[0] + dr, other_pos[1] + dc] != 1
            and (other_pos[0] + dr, other_pos[1] + dc) not in occupied
            and int(other_route.distance_map[other_pos[0] + dr, other_pos[1] + dc]) == distance - 1
            for dr, dc in CARDINAL_MOVES
        )
        if free_progress:
            continue
        mutual = int(route.distance_map[pos]) > 0 and (
            int(route.distance_map[other_pos]) == int(route.distance_map[pos]) - 1
        )
        lower_priority = (index != carrier_index, index) > (other != carrier_index, other)
        if pos == route.goal or (mutual and lower_priority):
            yielding = _yield_path_mask(grid, pos, allies, other_pos, other_route.goal)
            if yielding[:4].any():
                route.yield_for = other
                route.yield_origin = pos
                route.yield_priority_goal = other_route.goal
                return yielding
    return mask


def _random_candidate_assignments(candidates, count, rng, *, balanced_only=False):
    """Uppercase pairs use balanced splits; lowercase pairs also allow all-in."""
    assignments = [(point, index) for _, index, point in candidates]
    if len(assignments) == 2:
        left, _ = SPLIT_PATTERNS[rng.randrange(2 if balanced_only else len(SPLIT_PATTERNS))]
        left_count = (left * count + 2) // 5
        goals = [assignments[0]] * left_count + [assignments[1]] * (count - left_count)
    else:
        rng.shuffle(assignments)
        goals = [assignments[offset % len(assignments)] for offset in range(count)]
    rng.shuffle(goals)
    return goals


def _assign_next_waypoint(routes, positions, active, grid, stage, source, rng):
    """Split from a single map point; otherwise choose the nearest goal per origin."""
    scenario = routes[active[0]].scenario
    next_marker = scenario.waypoint_order[stage + 1]
    balanced_only = next_marker in scenario.uppercase_markers
    single_source = len(scenario.waypoint_points[scenario.waypoint_order[stage]]) == 1
    if single_source:
        candidates = _reachable_candidates(
            grid, source, scenario.waypoint_points[next_marker],
            max_distance=scenario.max_candidate_bfs_distance,
        )
        goals = _random_candidate_assignments(candidates, len(active), rng,
                                              balanced_only=balanced_only)
    else:
        origins = {}
        for i in active:
            origins.setdefault(routes[i].goal, []).append(i)
        choices = {}
        for origin, members in origins.items():
            goal, goal_index, _ = select_nearest_candidate(
                grid, origin, scenario.waypoint_points[next_marker],
                max_distance=scenario.max_candidate_bfs_distance,
            )
            choices.update((i, (goal, goal_index)) for i in members)
        goals = [choices[i] for i in active]
    for i, (goal, index) in zip(active, goals):
        routes[i].set_stage(stage + 1, positions[i], grid, goal, index)
    required_goals = frozenset(goal for goal, _ in goals)
    for i in active:
        routes[i].required_goals = required_goals


def advance_team_routes(routes, positions, alive, completed_a_groups,
                        grid=None, carrier_index=SPIKE_CARRIER_INDEX, rng=None):
    """Assign next goals using team arrival and the initial a group wait."""
    active = [i for i, is_alive in enumerate(alive) if is_alive]
    if not active:
        return
    rng = random if rng is None else rng
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
            _assign_next_waypoint(routes, positions, active, grid, 0, a_goal, rng)
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
            _assign_next_waypoint(routes, positions, active, grid, route.stage, positions[i], rng)
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


