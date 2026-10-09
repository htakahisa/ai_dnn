"""Guard contract and real IQ/engine integration checks; no training run."""

import contextlib
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_guard_scenarios import get_scenario, build_scenario, validate_checkpoint
from concon_v1.co1_guard_common import (
    GuardDQN, observation_dim, ACTION_DIM, WAIT_ACTION, build_inputs, decode_action,
    clear_shot, FEATURE_DIM, MAP_CHANNELS,
)
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_guard_battle_training import GuardBattleEnv, OPPONENTS, START_MODES
from concon_v1.co1_train_guard import make_checkpoint, curriculum_modes, qualifies_as_best, optimize
from concon_v1.evaluate_co1_guard import evaluate
from concon_v1.co1_attacker_common import GORIGONS


def actor(name, team="A", pos=(7, 1), alive=True, known=True):
    return SimpleNamespace(name=name, team=team, pos=list(pos), is_alive=alive,
                           position_known=known, hp=100, max_hp=100, facing="E",
                           moved_this_tick=False, smoke_charges=0, flash_charges=0,
                           recon_charges=1, reveal_remaining=0)


class GuardRuntimeSelectionTests(unittest.TestCase):
    def test_configured_guard_uses_actual_site_and_dispatches_moves(self):
        from concon_v1.co1_attacker_controller import (
            ConconAttackerController, ConconRoundAttackerController,
        )
        from concon_v1.co1_attacker_scenarios import CONCON_ATTACKER_POSTPLANT_MODELS

        # Plant on the opposite site to the carry route to check actual-site selection.
        for carry_map, guard_map, position in (("A1", "R", (7, 40)),
                                               ("A2", "L", (8, 3))):
            with self.subTest(guard_map=guard_map):
                adapter = ConconAttackerController(route_controller=Mock(), map_name=carry_map)
                guard = Mock()
                guard.decide_move.return_value = ([8, 4], None)
                with patch("concon_v1.co1_attacker_controller.ConconAttackerController",
                           return_value=adapter), \
                        patch("concon_v1.co1_learn_guard.ConconGuardController",
                              return_value=guard) as load:
                    controller = ConconRoundAttackerController(
                        map_names=carry_map, postplant_factories=CONCON_ATTACKER_POSTPLANT_MODELS,
                    )
                    state = {"is_planted": True, "planted_pos": position,
                             "grid": get_scenario(guard_map).grid, "chars": []}
                    for _ in range(2):
                        controller.reset_round()
                        self.assertIsNone(adapter.postplant_controller)
                        self.assertEqual(controller.decide_move(actor(GORIGONS.players[0]), state),
                                         ([8, 4], None))
                        self.assertIs(adapter.postplant_controller, guard)
                    load.assert_called_once_with(map_name=guard_map)
                    self.assertEqual(guard.reset_round.call_count, 2)
                    self.assertEqual(guard.decide_move.call_count, 2)


class GuardContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self, site="L"):
        scenario = get_scenario(site)
        return ConconGuardController(map_name=scenario, model=GuardDQN(scenario))

    def test_markers_are_pairs_and_future_patterns_use_same_reader(self):
        from concon_v1.co1_map_guard_L import MAZE_STR
        future = build_scenario("L2", MAZE_STR, "left")
        self.assertEqual(future.positions, get_scenario("L").positions)
        self.assertEqual(future.checkpoint_filename(), "co1_guard_L2_best.pt")
        for site in ("L", "R"):
            scenario = get_scenario(site)
            self.assertEqual(set(scenario.positions), set("abcde"))
            self.assertEqual(set(scenario.facing_points), set("abcde"))
        with self.assertRaisesRegex(ValueError, "exactly one A"):
            build_scenario("bad", MAZE_STR.replace("A", "0"), "left")
        with self.assertRaisesRegex(ValueError, "terrain"):
            build_scenario("bad", MAZE_STR.replace("1111", "0111", 1), "left")

    def test_partial_team_has_unique_stable_assignments_after_death(self):
        controller = self.controller()
        chars = [actor(name, alive=index < 3) for index, name in enumerate(GORIGONS.players)]
        controller.prepare_assignments(chars)
        initial = controller.assignments.copy()
        self.assertEqual(len(initial), 3)
        self.assertEqual(len(set(initial.values())), 3)
        chars[0].is_alive = False
        controller.prepare_assignments(chars)
        self.assertEqual(controller.assignments, initial)
        controller.reset_round()
        controller.prepare_assignments(chars)
        self.assertEqual(len(controller.assignments), 2)

    def test_hidden_enemy_coordinates_do_not_enter_inputs_or_memory(self):
        controller = self.controller()
        char = actor(GORIGONS.players[0])
        enemy = actor("hidden", "D", (-1, -1), known=False)
        state = {"grid": controller.scenario.grid, "chars": [char, enemy],
                 "is_planted": True, "planted_pos": (8, 3), "battle_tick": 0,
                 "defender_defuse_info": {"hidden": (1, 6)}, "smoke_cells": {(8, 3)}}
        controller.prepare_assignments(state["chars"])
        obs1, mask1, context = build_inputs(controller, char, state)
        # Even a misbehaving source supplying hidden coordinates cannot disclose them.
        enemy.pos = [10, 2]
        obs2, mask2, _ = build_inputs(controller, char, state)
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(mask1, mask2)
        self.assertNotIn("hidden", controller.sightings)
        self.assertTrue(context["tap"])
        self.assertFalse(context["fireable"])
        self.assertEqual(len(obs1), MAP_CHANNELS * controller.scenario.grid.size + FEATURE_DIM)

    def test_smoke_adjacent_and_recon_shots_follow_engine_rules(self):
        grid = np.zeros((5, 5), dtype=np.int32)
        char = actor("attacker", pos=(2, 0))
        enemy = actor("defender", "D", (2, 3))
        smoke = {(2, 2), (2, 3)}
        self.assertFalse(clear_shot(char, enemy, [char, enemy], grid, smoke))
        enemy.reveal_remaining = 2
        self.assertTrue(clear_shot(char, enemy, [char, enemy], grid, smoke))
        enemy.reveal_remaining = 0
        char.pos = [2, 2]
        self.assertTrue(clear_shot(char, enemy, [char, enemy], grid, smoke))
        char.pos = [2, 0]
        enemy.reveal_remaining = 2
        ally = actor("ally", pos=(2, 1))
        self.assertFalse(clear_shot(char, enemy, [char, enemy, ally], grid, smoke))

    def test_model_can_choose_movement_facing_and_recon_without_override(self):
        controller = self.controller()
        char = actor(GORIGONS.players[0])
        state = {"grid": controller.scenario.grid, "chars": [char],
                 "is_planted": True, "planted_pos": (8, 3), "battle_tick": 0}
        obs, mask, context = controller.policy_inputs(char, state)
        self.assertTrue(mask[:40].any())
        recon_start = (5 + 2 * 3) * 8
        self.assertEqual(mask[recon_start:recon_start + 8].sum(), 1)
        recon = recon_start + 2  # current E facing on an ability tick
        destination, payload = decode_action(recon, char.pos, context["targets"])
        self.assertEqual(payload["ability"], "RECON")
        self.assertEqual(payload["target"], (8, 3))
        with torch.no_grad():
            for parameter in controller.model.parameters():
                parameter.zero_()
            controller.model.head[-1].bias[WAIT_ACTION + 6] = 100
        destination, payload = controller.decide_move(char, state)
        self.assertEqual(destination, char.pos)
        self.assertEqual(payload["facing"], "W")
        self.assertEqual(controller.model(torch.as_tensor(obs)[None]).shape, (1, ACTION_DIM))

    def test_stopping_counter_and_reset(self):
        controller = self.controller()
        char = actor(GORIGONS.players[0])
        self.assertEqual(controller.stationary_ticks(char, 0), 0)
        self.assertEqual(controller.stationary_ticks(char, 1), 1)
        self.assertEqual(controller.stationary_ticks(char, 1), 1)
        self.assertEqual(controller.stationary_ticks(char, 2), 2)
        char.pos = [8, 1]
        self.assertEqual(controller.stationary_ticks(char, 3), 0)
        controller.reset_round()
        self.assertEqual(controller.stationary_ticks(char, 4), 0)

    def test_incoming_fire_lock_masks_only_impossible_facing_not_movement(self):
        controller = self.controller()
        char = actor(GORIGONS.players[0])
        char.recon_charges = 0
        state = {"grid": controller.scenario.grid, "chars": [char],
                 "is_planted": True, "planted_pos": (8, 3), "battle_tick": 0}
        _, free, _ = controller.policy_inputs(char, state)
        char.facing_forced_this_tick = True
        _, locked, _ = controller.policy_inputs(char, state)
        self.assertEqual(locked[WAIT_ACTION:WAIT_ACTION + 8].sum(), 1)
        self.assertTrue(locked[WAIT_ACTION + 2])  # E
        for operation in range(5):
            start = operation * 8
            self.assertEqual(locked[start:start + 8].any(), free[start:start + 8].any())
            self.assertLessEqual(locked[start:start + 8].sum(), 1)
        char.facing_forced_this_tick = False
        _, restored, _ = controller.policy_inputs(char, state)
        np.testing.assert_array_equal(restored, free)

    def test_checkpoint_rejects_wrong_side_or_map_change(self):
        controller = self.controller()
        checkpoint = make_checkpoint(controller.model, controller.scenario, 1, OPPONENTS, START_MODES)
        validate_checkpoint(checkpoint, controller.scenario)
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, "R")
        checkpoint["scenario_signature"] = "changed"
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, "L")
        self.assertEqual(curriculum_modes(1, 100), ("hold",))
        self.assertIn("smoke", curriculum_modes(100, 100))
        self.assertFalse(qualifies_as_best({"mean_win_rate": .8, "min_team_win_rate": .2},
                                         {"mean_win_rate": .7, "min_team_win_rate": .3}))

    def test_terminal_replay_has_finite_loss_and_updates_selected_action(self):
        scenario = get_scenario("L")
        model, target = GuardDQN(scenario), GuardDQN(scenario)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        obs = np.zeros(observation_dim(scenario), dtype=np.float16)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[WAIT_ACTION] = True
        replay = [(obs, WAIT_ACTION, 10.0, obs.copy(), mask, 1.0, 1)]
        before = model.head[-1].bias[WAIT_ACTION].detach().clone()
        loss = optimize(model, target, optimizer, replay, batch_size=1)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(model.head[-1].bias[WAIT_ACTION]), float(before))


class GuardEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_five_opponents_both_sites_finish_and_flush_terminal_replay(self):
        for opponent in OPPONENTS:
            for site in ("L", "R"):
                with self.subTest(opponent=opponent, site=site):
                    env = GuardBattleEnv(seed=4, opponents=[opponent], map_name=site)
                    with contextlib.redirect_stdout(io.StringIO()):
                        env.reset("smoke", attacker_count=5, defender_count=5)
                    self.assertTrue(env.game.current_attacker_team_ai.use_iq_perception)
                    self.assertTrue(env.game.is_planted)
                    self.assertFalse(env.game.defender_setup_phase.active)
                    transitions = []
                    while not env.done:
                        current, _, _ = env.step(actions=[WAIT_ACTION] * 5)
                        transitions.extend(current)
                    self.assertFalse(any(env.pending))
                    self.assertTrue(transitions)
                    self.assertTrue(any(record[5] for record in transitions))
                    self.assertTrue(all(record[6] >= 1 for record in transitions))
                    self.assertTrue(env.result()["winner"] in ("A", "D"))
                    self.assertGreater(env.metrics["tap_ticks"], 0)

    def test_evaluation_uses_fixed_weights_all_modes_and_restores_rng(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            model = GuardDQN(scenario)
            checkpoint = make_checkpoint(model, scenario, 12, OPPONENTS, START_MODES)
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state().clone()
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate(site, rounds=len(START_MODES), opponents=["touyama_v2"],
                                  frozen_checkpoint=buffer.getvalue())
            self.assertEqual(result["epsilon"], 0)
            self.assertEqual(set(result["opponents"]["touyama_v2"]["by_start_mode"]), set(START_MODES))
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()
