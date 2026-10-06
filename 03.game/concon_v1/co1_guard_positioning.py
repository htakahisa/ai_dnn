"""Learn and independently roll out quiet navigation on each fixed guard map.

BFS supplies training targets only. Inference uses learned action values and
the production legality mask; it never searches for or forces a route.
"""

import numpy as np
import torch

from game_core import FACING_DIRECTIONS
from concon_v1.co1_guard_common import (
    MOVES, aim_alignment, bfs_distance_map, GORIGONS,
)
from concon_v1.co1_learn_guard import ConconGuardController

POSITIONING_VERSION = 1
DEFAULT_POSITIONING_STEPS = 250


def training_targets(scenario, device="cpu"):
    height, width = scenario.grid.shape
    indices, targets = [], []
    for slot, letter in enumerate("abcde"):
        goal, aim = scenario.positions[letter], scenario.facing_points[letter]
        other_posts = [scenario.positions[other] for other in "abcde" if other != letter]
        for occupancy in range(16):
            route_grid = scenario.grid.copy()
            reserved = {point for bit, point in enumerate(other_posts) if occupancy & (1 << bit)}
            for point in reserved:
                route_grid[point] = 1
            distances = bfs_distance_map(route_grid, goal)
            for r, c in zip(*np.where(distances >= 0)):
                position = (int(r), int(c))
                values = np.full(40, -6.0, dtype=np.float32)
                for operation, (dr, dc) in enumerate(MOVES):
                    rr, cc = int(r + dr), int(c + dc)
                    if not (0 <= rr < height and 0 <= cc < width) or distances[rr, cc] < 0:
                        continue
                    progress = int(distances[r, c]) - int(distances[rr, cc])
                    for facing_index, facing in enumerate(FACING_DIRECTIONS):
                        value = 2.0 * progress + 0.25 * aim_alignment((rr, cc), aim, facing)
                        if position == goal:
                            value += 4.0 if operation == 4 else -4.0
                        elif operation == 4:
                            value -= 0.5
                        elif progress == 0:
                            # Wait for a passing ally instead of repeatedly
                            # stepping sideways when every useful move is blocked.
                            value -= 1.0
                        values[operation * 8 + facing_index] = value
                indices.append((slot * height * width + int(r) * width + int(c)) * 16 + occupancy)
                targets.append(values)
    return (torch.tensor(indices, dtype=torch.long, device=device),
            torch.as_tensor(np.asarray(targets), device=device))


def learn_positioning(model, scenario, steps=DEFAULT_POSITIONING_STEPS):
    if not model.navigation:
        raise ValueError("positioning requires the learned navigation head")
    if steps < 1:
        raise ValueError("positioning updates must be positive")
    indices, targets = training_targets(scenario, next(model.parameters()).device)
    yield_targets = torch.zeros((80, 40), device=targets.device)
    for preferred in range(4):
        for neighbors in range(16):
            if neighbors & (1 << preferred):
                values = yield_targets[preferred * 16 + neighbors].reshape(5, 8)
                values[:4] = 2.0
                values[preferred ^ 1] = 5.0
                values[preferred] = -6.0
                values[4] = -2.0
    optimizer = torch.optim.Adam([model.navigation_values.weight, model.navigation_yield_values.weight], lr=0.1)
    for _ in range(steps):
        loss = (torch.nn.functional.mse_loss(model.navigation_values(indices), targets)
                + torch.nn.functional.mse_loss(model.navigation_yield_values.weight, yield_targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return indices, targets


def validate_positioning(model, scenario, seed=7919):
    result = positioning_evaluation(model, scenario, seed=seed)
    result["formation"] = formation_evaluation(model, scenario)
    result["passed"] &= result["formation"]["passed"]
    return result


def prepare_positioning(model, scenario, steps=DEFAULT_POSITIONING_STEPS,
                        reuse=False, force=False, seed=7919):
    if steps < 0:
        raise ValueError("positioning steps cannot be negative")
    if reuse and not force:
        result = validate_positioning(model, scenario, seed)
        if result["passed"]:
            return result, {"mode": "reused", "updates": 0, "requested_updates": steps}
    if steps == 0:
        raise RuntimeError("positioning validation did not pass; enable positioning updates")
    learn_positioning(model, scenario, steps=steps)
    result = validate_positioning(model, scenario, seed)
    if not result["passed"]:
        raise RuntimeError(f"guard failed positioning validation after {steps} updates: {result}")
    return result, {"mode": "trained", "updates": steps, "requested_updates": steps}


def positioning_evaluation(model, scenario, seed=7919, trials=100, hold_ticks=8):
    """Autonomous trajectories from held evaluation starts, not label accuracy.

    The production controller builds observations and masks and selects every
    move. Goal arrival, repeated holding, facing and reversals are measured.
    """
    from types import SimpleNamespace
    rng = np.random.default_rng(seed)
    arrived = held = aimed = reversals = moves = 0
    arrival_ticks = []
    was_training = model.training
    model.eval()
    try:
        for trial in range(trials):
            controller = ConconGuardController(map_name=scenario, model=model)
            letter = "abcde"[trial % 5]
            goal = scenario.positions[letter]
            distances = bfs_distance_map(scenario.grid, goal)
            cells = np.argwhere(distances >= 0)
            start = tuple(map(int, cells[int(rng.integers(len(cells)))]))
            char = SimpleNamespace(name=GORIGONS.players[trial % 5], team="A", pos=list(start),
                                   is_alive=True, hp=100, max_hp=100,
                                   facing=FACING_DIRECTIONS[int(rng.integers(8))],
                                   moved_this_tick=False, smoke_charges=0, flash_charges=0,
                                   recon_charges=0, facing_forced_this_tick=False)
            controller.assignments = {char.name: letter}
            controller._assigned = True
            state = dict(grid=scenario.grid, chars=[char], is_planted=True,
                         planted_pos=scenario.plant_cells[trial % len(scenario.plant_cells)],
                         battle_tick=0, detonate_timer=50)
            history = [start]
            first_arrival = 0 if start == goal else None
            stopped = 0
            max_ticks = int(distances[start]) + hold_ticks + 4
            for tick in range(max_ticks):
                state["battle_tick"] = tick
                destination, payload = controller.decide_move(char, state)
                previous = tuple(char.pos)
                # The training teacher is never consulted to execute a move.
                char.pos = list(destination)
                if "ability" not in payload:
                    char.facing = payload["facing"]
                char.moved_this_tick = tuple(char.pos) != previous
                position = tuple(char.pos)
                moves += int(char.moved_this_tick)
                history.append(position)
                if len(history) >= 3 and history[-1] == history[-3] != history[-2]:
                    reversals += 1
                if position == goal:
                    if first_arrival is None:
                        first_arrival = tick + 1
                    stopped = stopped + 1 if previous == goal else 0
                else:
                    stopped = 0
                if stopped >= hold_ticks:
                    break
            arrived += int(first_arrival is not None)
            held += int(stopped >= hold_ticks)
            aimed += int(stopped >= hold_ticks and char.facing == max(
                FACING_DIRECTIONS, key=lambda direction: aim_alignment(goal, scenario.facing_points[letter], direction)))
            if first_arrival is not None:
                arrival_ticks.append(first_arrival)
    finally:
        model.train(was_training)
    return dict(trials=trials, arrival_rate=arrived / trials, hold_rate=held / trials,
                facing_rate=aimed / trials, reversals=reversals, move_decisions=moves,
                mean_arrival_ticks=float(np.mean(arrival_ticks)) if arrival_ticks else None,
                hold_ticks=hold_ticks,
                passed=arrived == trials and held == trials and aimed == trials and reversals == 0)


def formation_evaluation(model, scenario, seed=500, trials=20, hold_ticks=8):
    """Five actors use the shared production controller and occupancy masks."""
    from types import SimpleNamespace
    arrived = held = aimed = departures = 0
    was_training = model.training
    model.eval()
    try:
        distances = bfs_distance_map(scenario.grid, scenario.plant_cells[0])
        cells = np.argwhere((distances >= 2) & (distances <= 8))
        for trial in range(trials):
            rng = np.random.default_rng(seed + trial)
            starts = cells[rng.choice(len(cells), 5, replace=False)]
            chars = [SimpleNamespace(name=name, team="A", pos=list(map(int, pos)), is_alive=True,
                                     hp=100, max_hp=100, facing="N", moved_this_tick=False,
                                     smoke_charges=0, flash_charges=0, recon_charges=0,
                                     facing_forced_this_tick=False)
                     for name, pos in zip(GORIGONS.players, starts)]
            controller = ConconGuardController(map_name=scenario, model=model, seed=trial)
            controller.prepare_assignments(chars)
            state = dict(grid=scenario.grid, chars=chars, is_planted=True,
                         planted_pos=scenario.plant_cells[0], detonate_timer=50)
            stopped = {char.name: 0 for char in chars}
            reached = set()
            for tick in range(50):
                state["battle_tick"] = tick
                for char in chars:
                    goal = scenario.positions[controller.assignments[char.name]]
                    previous = tuple(char.pos)
                    destination, payload = controller.decide_move(char, state)
                    char.pos = list(destination)
                    if "ability" not in payload:
                        char.facing = payload["facing"]
                    char.moved_this_tick = tuple(char.pos) != previous
                    departures += int(previous == goal and tuple(char.pos) != goal)
                    if tuple(char.pos) == goal:
                        reached.add(char.name)
                        stopped[char.name] = stopped[char.name] + 1 if previous == goal else 0
                    else:
                        stopped[char.name] = 0
            arrived += len(reached)
            for char in chars:
                letter = controller.assignments[char.name]
                held += int(stopped[char.name] >= hold_ticks)
                aimed += int(stopped[char.name] >= hold_ticks and char.facing == max(
                    FACING_DIRECTIONS, key=lambda direction: aim_alignment(
                        scenario.positions[letter], scenario.facing_points[letter], direction)))
    finally:
        model.train(was_training)
    total = trials * 5
    return dict(trials=trials, actors=total, arrival_rate=arrived / total, hold_rate=held / total,
                facing_rate=aimed / total, leave_goal_decisions=departures, hold_ticks=hold_ticks,
                passed=min(arrived, held, aimed) / total >= .95 and departures == 0)
