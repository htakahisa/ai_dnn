"""Supervised setup/live movement targets and real-engine navigation checks."""

import contextlib
import io

import torch

from concon_v1.co1_guard_positioning import training_targets
from concon_v1.co1_defender_scenario import phase_scenario
from concon_v1.co1_defender_common import GORIGONS

POSITIONING_VERSION = 1


def learn_positioning(model, scenario, steps=None, learning_rate=None):
    from concon_v1.co1_train_defender_search import DEFAULT_POSITIONING_STEPS, POSITIONING_LEARNING_RATE
    if steps is None:
        steps = DEFAULT_POSITIONING_STEPS
    if learning_rate is None:
        learning_rate = POSITIONING_LEARNING_RATE
    if steps < 1:
        raise ValueError("positioning updates must be positive")
    device = next(model.parameters()).device
    indices, targets = [], []
    for setup in (False, True):
        phase_indices, phase_targets = training_targets(phase_scenario(scenario, setup), device)
        indices.append(phase_indices + int(setup) * model.phase_size)
        targets.append(phase_targets)
    indices, targets = torch.cat(indices), torch.cat(targets)
    yield_targets = torch.zeros((80, 40), device=device)
    for preferred in range(4):
        for neighbors in range(16):
            if neighbors & (1 << preferred):
                values = yield_targets[preferred * 16 + neighbors].reshape(5, 8)
                values[:4] = 2.
                values[preferred ^ 1] = 5.
                values[preferred] = -6.
                values[4] = -2.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        loss = (torch.nn.functional.mse_loss(model.navigation_values(indices), targets)
                + torch.nn.functional.mse_loss(model.navigation_yield_values.weight, yield_targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_positioning(model, scenario, seed=0, trials=20, live_ticks=100, hold_ticks=8):
    """No opponents: use actual setup movement and live IQ-wrapped movement.

    Shooting, abilities and victory are outside this basic-skill evaluation.
    No teacher or BFS is used to choose any evaluation move.
    """
    from concon_v1.co1_battle_training import _run_from_project_root
    from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
    from concon_v1.co1_guard_common import aim_alignment
    from concon_v1.co1_attacker_scenarios import GAME_MAZE_STR
    from controllers import DefaultAttackerController, DefaultDefenderController
    from game_core import FACING_DIRECTIONS
    from team_ai import DualRoleTeamAI

    @_run_from_project_root
    def run():
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle
        arrived = held = aimed = departures = setup_arrived = 0
        was_training = model.training
        model.eval()
        try:
            for trial in range(trials):
                controller = ConconDefenderSearchController(model=model, seed=seed + trial)
                team = DualRoleTeamAI("ConCon positioning", DefaultAttackerController,
                                      lambda: controller, use_iq_perception=True)
                opponent = DualRoleTeamAI("quiet navigation", DefaultAttackerController,
                                          DefaultDefenderController, use_iq_perception=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    game = VisualFPSBattle(GAME_MAZE_STR, opponent, team, headless=True,
                                           defender_roster=list(GORIGONS.players),
                                           defender_spike_holder_name=GORIGONS.spike_holder,
                                           defender_igl_name=GORIGONS.igl, disable_side_swap=True)
                game.analytics_tracker = None
                # No combat opponent in this basic-skill check.
                for char in game.chars:
                    if char.team == "A":
                        char.is_alive = False
                        char.hp = 0
                chars = [char for char in game.chars if char.team == "D"]
                controller.prepare_assignments(chars)
                while game.defender_setup_phase.active:
                    for char in chars:
                        game._move_character_during_defender_setup(char)
                    game.defender_setup_phase.advance_tick()
                setup_arrived += sum(tuple(char.pos) == scenario.setup_positions[controller.assignments[char.name]]
                                     for char in chars)
                stopped = {char.name: 0 for char in chars}
                reached = set()
                for tick in range(live_ticks):
                    game.battle_tick = tick + 1
                    team.perception_engine.clear_cache()
                    for char in chars:
                        goal = scenario.positions[controller.assignments[char.name]]
                        previous = tuple(char.pos)
                        game.move_character(char)
                        position = tuple(char.pos)
                        departures += int(previous == goal and position != goal)
                        if position == goal:
                            reached.add(char.name)
                            stopped[char.name] = stopped[char.name] + 1 if previous == goal else 0
                        else:
                            stopped[char.name] = 0
                arrived += len(reached)
                for char in chars:
                    letter = controller.assignments[char.name]
                    holding = stopped[char.name] >= hold_ticks
                    held += int(holding)
                    aimed += int(holding and char.facing == max(FACING_DIRECTIONS, key=lambda facing:
                        aim_alignment(scenario.positions[letter], scenario.facing_points[letter], facing)))
        finally:
            model.train(was_training)
        total = trials * 5
        return dict(trials=trials, actors=total, perception="production_iq_live",
                    setup_staging_rate=setup_arrived / total, arrival_rate=arrived / total,
                    hold_rate=held / total, facing_rate=aimed / total,
                    leave_goal_decisions=departures, live_ticks=live_ticks, hold_ticks=hold_ticks,
                    passed=arrived == total and held == total and aimed == total and departures == 0)

    if trials < 1 or live_ticks < hold_ticks + 1 or hold_ticks < 1:
        raise ValueError("invalid evaluation trials/ticks")
    return run()
