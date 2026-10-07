"""BFS-supervised basic movement/defusing and production-IQ engine checks."""

import contextlib
import io
import random

import numpy as np
import torch

from concon_v1.co1_attacker_common import bfs_distance_map, GORIGONS
from concon_v1.co1_retake_common import MOVES
from concon_v1.co1_retake_scenarios import plant_site
from concon_v1.co1_retake_start_positions import validate_starts
from concon_v1.co1_retake_navigation import assembly_navigation, assembly_step_allowed

FOUNDATION_VERSION = 1
PATH_JITTER_RADIUS = 2


def plant_cells(scenario):
    return [(int(r), int(c)) for r, c in zip(*np.where(scenario.grid == 2))
            if plant_site((r, c), scenario.grid) == scenario.map_name]


def perceived_plant_cells(scenario):
    """Include the engine's IQ-blurred plant coordinates, including floor cells."""
    from iq_perception import IQPerceptionEngine, PLANTED_SPIKE_MAX_ERROR
    engine = IQPerceptionEngine()
    cells = set()
    for spike in plant_cells(scenario):
        for dr in range(-PLANTED_SPIKE_MAX_ERROR, PLANTED_SPIKE_MAX_ERROR + 1):
            for dc in range(-PLANTED_SPIKE_MAX_ERROR, PLANTED_SPIKE_MAX_ERROR + 1):
                cells.add(engine._nearest_walkable(scenario.grid, (spike[0] + dr, spike[1] + dc), spike))
    return sorted(cells)


def navigation_cells(scenario):
    plants = perceived_plant_cells(scenario)
    # Entry goals have their own lookup even if IQ can report the same cell as
    # a planted spike: navigating to an entry must not teach defusing there.
    entries = [point for _, points in scenario.entry_points for point in points]
    return plants + sorted(set(scenario.rally_points) - set(plants)) + entries


def teacher_values(grid, spike, position, assembly=False, distances=None, front=None):
    distances = bfs_distance_map(grid, spike) if distances is None else distances
    before = distances[position]
    values = np.full(6, -2., dtype=np.float32)
    for operation, (dr, dc) in enumerate(MOVES):
        r, c = position[0] + dr, position[1] + dc
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and distances[r, c] >= 0:
            if front is not None and not assembly_step_allowed(front, position, (r, c)):
                continue
            values[operation] = float(before - distances[r, c]) if operation < 4 else -.2
    if assembly:
        if position == spike:
            values[4] = 2.
    elif max(abs(position[0] - spike[0]), abs(position[1] - spike[1])) <= 1:
        values[5] = 2.
    return values


def training_targets(scenario, seed=0):
    """Cover paths from every explicit start, nearby perturbations and defusing."""
    starts, rng = validate_starts(), random.Random(seed)
    indices, targets = [], []
    height, width = scenario.grid.shape
    plants = set(perceived_plant_cells(scenario))
    front, rally_routes = assembly_navigation(scenario)
    cells = navigation_cells(scenario)
    entry_count = sum(len(points) for _, points in scenario.entry_points)
    for plant_index, spike in enumerate(cells):
        entry = bool(entry_count and plant_index >= len(cells) - entry_count)
        assembly = not entry and spike not in plants
        distances = rally_routes[spike] if assembly else bfs_distance_map(scenario.grid, spike)
        covered = np.zeros(scenario.grid.shape, dtype=bool)
        if assembly or entry:
            # Any teammate may be assigned any A, including an optional advance
            # from another entrance. Cover every reachable approach cell.
            covered |= distances >= 0
        for candidates in starts.values():
            for start in candidates:
                if distances[start] < 0:
                    raise ValueError(f"foundation start {start} cannot reach {spike}")
                from_start = bfs_distance_map(scenario.grid, start)
                # Cover every shortest-path branch, plus detours corresponding
                # to two random steps away. A distance-decreasing move remains
                # inside this region even when it chooses a different branch.
                covered |= ((distances >= 0) & (from_start >= 0)
                            & (from_start + distances <= distances[start] + 2 * PATH_JITTER_RADIUS))
        positions = [(int(r), int(c)) for r, c in zip(*np.where(covered))]
        rng.shuffle(positions)
        for position in positions:
            indices.append(plant_index * height * width + position[0] * width + position[1])
            targets.append(teacher_values(scenario.grid, spike, position, assembly=assembly or entry,
                                          distances=distances, front=front if assembly else None))
    return np.asarray(indices, dtype=np.int64), np.asarray(targets, dtype=np.float32)


def learn_foundation(model, scenario, steps=250, learning_rate=.1, seed=0):
    if not model.foundation or steps < 1 or learning_rate <= 0:
        raise ValueError("foundation model, positive steps and learning rate are required")
    indices, targets = training_targets(scenario, seed)
    device = next(model.parameters()).device
    indices = torch.as_tensor(indices, device=device)
    targets = torch.as_tensor(targets, device=device)
    model.foundation_values.weight.requires_grad_(True)
    optimizer = torch.optim.Adam(model.foundation_values.parameters(), lr=learning_rate)
    loss = None
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model.foundation_values(indices), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.foundation_trained[indices] = True
    model.foundation_values.weight.requires_grad_(False)
    return dict(training_states=len(indices), steps=steps, loss=float(loss.detach()))


def evaluate_foundation(model, scenario, trials=20, seed=0, ability_distance=6):
    """Only model actions: actual IQ movement and six consecutive DEFUSE ticks."""
    from concon_v1.co1_battle_training import _run_from_project_root
    from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
    from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
    from controllers import DefaultAttackerController, DefaultDefenderController
    from team_ai import DualRoleTeamAI
    from game_core import SPIKE_DETONATION_TICKS, DEFUSE_REQUIRED_TICKS
    if trials < 5:
        raise ValueError("foundation evaluation needs at least five trials, one per search post")
    starts, rng = validate_starts(), random.Random(seed)

    @_run_from_project_root
    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle
        details = []
        was_training = model.training
        model.eval()
        try:
            for trial in range(trials):
                slot = "abcde"[trial % 5]
                start = rng.choice(starts[slot])
                spike = rng.choice(plant_cells(scenario))
                controller = ConconDefenderRetakeController(scenario.map_name, model=model, ability_distance=ability_distance)
                team = DualRoleTeamAI("ConCon retake foundation", DefaultAttackerController,
                                      lambda: controller, use_iq_perception=True)
                quiet = DualRoleTeamAI("quiet foundation", DefaultAttackerController, DefaultDefenderController)
                with contextlib.redirect_stdout(io.StringIO()):
                    game = VisualFPSBattle(GAME_MAZE_STR, quiet, team, headless=True,
                        defender_roster=list(GORIGONS.players), defender_spike_holder_name=GORIGONS.spike_holder,
                        defender_igl_name=GORIGONS.igl, disable_side_swap=True)
                game.analytics_tracker = None
                actor = next(char for char in game.chars if char.team == "D" and char.name == GORIGONS.players[trial % 5])
                for char in game.chars:
                    if char is not actor:
                        char.is_alive, char.hp = False, 0
                actor.pos = list(start)
                # Retain the normal masks/observations. The learned movement
                # should outrank unnecessary casts in this quiet task.
                game.defender_setup_phase.finish()
                game.is_planted, game.planted_pos, game.spike_pos = True, spike, None
                game.detonate_timer = SPIKE_DETONATION_TICKS
                reached = False
                longest_defuse = 0
                for tick in range(SPIKE_DETONATION_TICKS):
                    game.battle_tick = tick + 1
                    team.perception_engine.clear_cache()
                    with contextlib.redirect_stdout(io.StringIO()):
                        game.move_character(actor)
                        game._resolve_defuse_completion()
                    reached |= max(abs(actor.pos[0] - spike[0]), abs(actor.pos[1] - spike[1])) <= 1
                    longest_defuse = max(longest_defuse, actor.defuse_timer)
                    game.detonate_timer -= 1
                    if game.is_defused:
                        break
                details.append(dict(slot=slot, start=start, spike=spike, reached=reached,
                                    defused=bool(game.is_defused), ticks=tick + 1,
                                    max_defuse_ticks=longest_defuse))
        finally:
            model.train(was_training)
        return dict(trials=trials, arrival_rate=sum(row["reached"] for row in details) / trials,
                    defuse_rate=sum(row["defused"] for row in details) / trials,
                    passed=all(row["defused"] and row["max_defuse_ticks"] >= DEFUSE_REQUIRED_TICKS for row in details),
                    perception="production_iq", details=details)
    return run()
