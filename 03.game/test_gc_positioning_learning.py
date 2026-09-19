"""Regression coverage for GC positional incentives and model-owned actions."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
import train_attacker_carry_gc as carry
import train_attacker_guard_gc as guard
import learning_attacker_guard_gc as runtime
import learning_attacker_carry_gc as carry_runtime
from positioning_gc import (guard_candidates, PLANT_PATTERNS, REGISTERED_PLANT_CELLS,
                            team_plant_target, set_team_plant_target)
from positioning_evaluation_gc import evaluation_seed
from ghost_champions_v1_macro import GhostChampionsV1AttackerController


class PositioningLearningTests(unittest.TestCase):
    def setUp(self):
        random.seed(19)
        np.random.seed(19)

    def carry_reward(self, pos, completed=False, on_site=True):
        env = carry.CarryEnv()
        env.reset(pattern_marker=5)
        env.carrier.pos = list(pos)
        env.target_plant_pos = PLANT_PATTERNS[5][0]
        env.dist_map = carry.PLANT_DIST_MAPS[env.target_plant_pos]
        env._prev_dist = env.dist_map[pos]
        for a in env.attackers:
            a.pos = list(pos)
        env.defenders = []
        return env._compute_reward(False, False, completed, completed, completed, on_site)[0]

    def test_registered_plants_rewarded_above_ordinary_plants(self):
        ordinary = next(p for p in carry.PLANT_CELLS if p not in REGISTERED_PLANT_CELLS)
        preferred = PLANT_PATTERNS[5][0]
        self.assertGreater(self.carry_reward(preferred, True), self.carry_reward(ordinary, True))

    def test_crossing_ordinary_plant_cell_is_not_penalized_for_not_planting(self):
        ordinary = next(p for p in carry.PLANT_CELLS if p not in REGISTERED_PLANT_CELLS)
        self.assertAlmostEqual(self.carry_reward(ordinary, on_site=True),
                               self.carry_reward(ordinary, on_site=False))

    def test_carry_targets_registered_patterns_and_resets_distance(self):
        env = carry.CarryEnv()
        for marker in PLANT_PATTERNS:
            env._prev_dist = 9999
            env.reset(pattern_marker=marker)
            self.assertIn(env.target_plant_pos, PLANT_PATTERNS[marker])
            self.assertEqual(env._prev_dist, env.dist_map[tuple(env.carrier.pos)])

    def test_guard_learns_from_off_position_starts_with_unique_targets(self):
        env = guard.GuardEnv(start_mode="transition")
        for marker in guard.ACTIVE_GUARD_PATTERNS:
            env.reset(pattern_marker=marker)
            targets = [a.assigned_guard_pos for a in env.attackers]
            self.assertEqual(len(set(targets)), len(env.attackers))
            self.assertTrue(any(a.assigned_guard_dist_map[tuple(a.pos)] > 0
                                for a in env.attackers))
            for a in env.attackers:
                self.assertGreaterEqual(a.assigned_guard_dist_map[tuple(a.pos)], 0)

    def test_guard_candidate_shortage_fills_nearby_walkable_cells(self):
        positions = guard_candidates(guard.GRID, 6, 5)
        self.assertEqual(len(set(positions)), 5)
        self.assertTrue(set(guard.GUARD_PATTERN_CELLS[6]).issubset(positions))
        self.assertTrue(all(guard.GRID[p] != 1 for p in positions))

    def test_sighting_does_not_replace_guard_target(self):
        env = guard.GuardEnv(start_mode="transition")
        env.reset(pattern_marker=5)
        env.guard_memory.last_seen_enemy = {"pos": (22, 20), "name": "D", "tick_ago": 0}
        mode, distances, _ = env._priority_mode_and_distmap(env.attackers[0])
        self.assertEqual(mode, "position")
        self.assertIs(distances, env.attackers[0].assigned_guard_dist_map)

    def test_first_step_toward_guard_position_has_progress_reward(self):
        env = guard.GuardEnv(start_mode="transition")
        env.reset(pattern_marker=5)
        a = env.attackers[0]
        start = tuple(a.pos)
        distances = a.assigned_guard_dist_map
        dr, dc = guard.bfs_best_direction(distances, *start)
        self.assertNotEqual((dr, dc), (0, 0))
        env._pre_positions = {a.name: start}
        a.pos = [start[0] + dr, start[1] + dc]
        a.moved_this_tick = True
        env.defenders = []
        rewards = env._compute_rewards({}, {}, {}, {})
        self.assertGreater(rewards[a.name], guard.STEP_PENALTY)

    def test_guard_modern_training_and_runtime_observations_match(self):
        env = guard.GuardEnv(start_mode="transition")
        expected, _ = env.reset(pattern_marker=7)
        ctrl = runtime.LearningAttackerGuardGCController.__new__(runtime.LearningAttackerGuardGCController)
        ctrl.positioning_version = 1
        ctrl.spike_dist_map = env.spike_dist_map
        ctrl.sighting_dist_map = env.sighting_dist_map
        ctrl.team_memory = env.guard_memory
        ctrl._active_guard_pattern = env.pattern_marker
        ctrl._assigned_dist_maps = {a.name: a.assigned_guard_dist_map for a in env.attackers}
        for a in env.attackers:
            a.smoke_charges = a.charges if a.role == "SMOKE" else 0
            a.flash_charges = a.charges if a.role == "FLASH" else 0
            a.recon_charges = a.charges if a.role == "RECON" else 0
        state = {"grid": guard.GRID, "chars": env.attackers + env.defenders, "smoke_cells": set()}
        for a in env.attackers:
            los = any(guard.has_los(a.pos, cell) for cell in
                      guard.postplant_watch_cells(guard.GRID, env.planted_pos))
            obs, _ = ctrl._build_observation(a, state, los, None, env.detonate_timer)
            np.testing.assert_allclose(obs, expected[a.name])

    def test_carry_modern_observation_guides_selected_plant_even_before_waypoint(self):
        env = carry.CarryEnv()
        expected, _ = env.reset(pattern_marker=7)
        ctrl = carry_runtime.LearningAttackerCarryGCController.__new__(
            carry_runtime.LearningAttackerCarryGCController)
        ctrl.positioning_version = 2
        ctrl.game = NS(grid=carry.GRID, smokes=[])
        ctrl._priority_dist_map = carry.PRIORITY_DIST_MAP
        ctrl._priority_max_dist = carry.PRIORITY_MAX_DIST
        ctrl._cached_target_pos = None
        ctrl._target_dist_map = None
        ctrl._sighting = env.sighting.last_seen_enemy
        for a in env.attackers:
            a.smoke_charges = a.charges if a.role == "SMOKE" else 0
            a.flash_charges = a.charges if a.role == "FLASH" else 0
            a.recon_charges = a.charges if a.role == "RECON" else 0
        obs = ctrl._build_observation(env.carrier, env.attackers + env.defenders,
                                      set(), env.dist_map, 0, carry.MAX_TICKS,
                                      env.reached_waypoint, env.target_plant_pos)
        np.testing.assert_allclose(obs, expected)
        target_map = carry.PLANT_DIST_MAPS[env.target_plant_pos]
        self.assertEqual(tuple(obs[26:28]), carry.bfs_best_direction(target_map, *env.carrier.pos))
        env.carrier.moved_this_tick = False
        env.carrier.moved_last_tick = True
        ctrl.positioning_version = 3
        obs = ctrl._build_observation(env.carrier, env.attackers + env.defenders,
                                      set(), env.dist_map, 0, carry.MAX_TICKS,
                                      env.reached_waypoint, env.target_plant_pos)
        self.assertEqual(obs[3], 1.0)
        self.assertEqual(obs.shape, (31,))
        env.carrier.plant_timer = carry.PLANT_REQUIRED_TICKS - 1
        obs = ctrl._build_observation(env.carrier, env.attackers + env.defenders,
                                      set(), env.dist_map, 0, carry.MAX_TICKS,
                                      env.reached_waypoint, env.target_plant_pos)
        self.assertGreater(obs[30], 0)

    def test_modern_guard_action_is_not_overwritten_by_route_or_spike_watch(self):
        ctrl = runtime.LearningAttackerGuardGCController.__new__(runtime.LearningAttackerGuardGCController)
        ctrl.positioning_version = 1
        ctrl.verbose = False
        for name in ("_ensure_spike_dist_map", "_ensure_guard_assignment", "_maybe_advance_tick",
                     "_update_sighting_dist_map"):
            setattr(ctrl, name, Mock())
        ctrl._build_observation = Mock(return_value=(np.zeros(34, dtype=np.float32), []))
        ctrl._action_mask = Mock(return_value=np.ones(10, dtype=bool))
        ctrl._guard_position_step = Mock(side_effect=AssertionError("route override"))
        ctrl.model = lambda x: torch.tensor([[0., 0., 10., 0., 0., 0., 0., 0., 0., 0.]], device=x.device)
        char = NS(name="p", pos=[3, 3], is_alive=True)
        state = {"grid": np.zeros((7, 7), dtype=int), "chars": [char],
                 "is_planted": True, "planted_pos": [3, 4], "detonate_timer": 30}
        self.assertEqual(ctrl.decide_move(char, state), [2, 3])

    def test_modern_guard_wrapper_does_not_override_learned_action(self):
        from unittest.mock import patch
        import ghost_champions_v1_macro as wrapper
        ctrl = GhostChampionsV1AttackerController.__new__(GhostChampionsV1AttackerController)
        ctrl.macro_controller = object()
        ctrl.guard = NS(positioning_version=1)
        ctrl._micro_cover_result = Mock(side_effect=AssertionError("cover override"))
        result = [4, 3]
        with patch.object(wrapper._BaseGCAttacker, "decide_move", return_value=result):
            self.assertIs(ctrl.decide_move(NS(), {"is_planted": True}), result)

    def test_modern_carry_wrapper_keeps_model_movement(self):
        from unittest.mock import patch
        import ghost_champions_v1_macro as wrapper
        ctrl = GhostChampionsV1AttackerController.__new__(GhostChampionsV1AttackerController)
        ctrl.carry = NS(positioning_version=1)
        ctrl.macro_controller = Mock()
        ctrl.macro_controller.coordinate.side_effect = AssertionError("Macro movement override")
        char = NS(name="Absol", team="A", is_alive=True, has_spike=True)
        result = [4, 3]
        with patch.object(wrapper._BaseGCAttacker, "decide_move", return_value=result):
            self.assertIs(ctrl.decide_move(char, {"chars": [char]}), result)
        ctrl.macro_controller._sync_tick_once.assert_called_once()

    def test_partial_spike_los_does_not_reward_ignoring_hidden_defuser(self):
        from unittest.mock import patch
        grid = np.zeros((7, 7), dtype=int)
        a = guard.UnitStub("A", "A", (3, 1), "SMOKE")
        d = guard.UnitStub("D", "D", (3, 4), "NONE")
        d.defuse_timer = 1
        a.assigned_guard_dist_map = np.zeros_like(grid)
        a.guard_arrived = True
        a.smoke_charges = 1
        env = guard.GuardEnv()
        env.attackers, env.defenders = [a], [d]
        env.planted_pos = (3, 4)
        env.spike_dist_map = np.ones_like(grid)
        env.smokes = [{"cells": {(3, 3)}, "remaining_ticks": 2, "team": "A"}]
        smoke_cells = env._smoke_cells()
        with patch.object(guard, "GRID", grid):
            self.assertTrue(any(guard.has_los(a.pos, p, smoke_cells)
                                for p in guard.postplant_watch_cells(grid, env.planted_pos)))
            self.assertFalse(guard.has_los(a.pos, d.pos, smoke_cells))
            rewards = env._compute_rewards({"D": 1}, {}, {}, {})
            self.assertAlmostEqual(rewards[a.name], guard.STEP_PENALTY + guard.UNCONTESTED_DEFUSE_PENALTY)
            obs = guard.build_observation(a, [a], [d], env.guard_memory, smoke_cells,
                                          True, 30, env.spike_dist_map, None, True,
                                          env._active_defuse_info(), pattern_marker=5)
        self.assertEqual(obs[17], 0.0)
        ctrl = runtime.LearningAttackerGuardGCController.__new__(runtime.LearningAttackerGuardGCController)
        ctrl.positioning_version = 2
        ctrl.spike_dist_map = env.spike_dist_map
        ctrl.sighting_dist_map = None
        ctrl.team_memory = env.guard_memory
        ctrl._active_guard_pattern = 5
        ctrl._assigned_dist_maps = {a.name: a.assigned_guard_dist_map}
        state = {"grid": grid, "chars": [a, d], "smoke_cells": smoke_cells}
        runtime_obs, _ = ctrl._build_observation(a, state, True,
                                                {"name": "D", "progress_ratio": 0.1}, 30)
        self.assertEqual(runtime_obs[17], 0.0)

    def test_evaluation_preserves_training_random_streams(self):
        py_state, np_state = random.getstate(), np.random.get_state()
        with evaluation_seed(999):
            random.random()
            np.random.random()
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])

    def test_team_target_is_published_through_iq_view(self):
        from iq_perception import PerceivedGameView
        real = NS(target_plant_pos=(4, 41))
        first = PerceivedGameView(real, {"target_plant_pos": [5, 40]}, {})
        set_team_plant_target(first, (7, 40))
        self.assertEqual(real.target_plant_pos, (7, 40))
        self.assertEqual(first.target_plant_pos, (7, 40))
        second = PerceivedGameView(real, {"target_plant_pos": [8, 39]}, {})
        self.assertEqual(team_plant_target(second), (7, 40))

    def test_real_carry_warm_start_preserves_legacy_deployed_values(self):
        from gc_v1.train_attacker_carry_gc_real import expanded_state
        old = carry_runtime.AttackerCarryDuelingDQN()
        modern = carry_runtime.AttackerCarryDuelingDQN(obs_dim=31)
        modern.load_state_dict(expanded_state({"model_state_dict": old.state_dict()}))
        old_obs = torch.rand(8, 29)
        old_obs[:, 3] = 0  # Actual v2 decision timing.
        new_obs = torch.cat((old_obs, torch.rand(8, 2)), dim=1)
        new_obs[:, 3] = 1
        torch.testing.assert_close(old(old_obs), modern(new_obs))

    def test_series_evaluation_restores_model_paths_on_failure(self):
        from gc_v1 import evaluate_real_series_gc as evaluation
        import ghost_champions_v1 as gc
        from unittest.mock import patch
        original = gc.CARRY
        def fail(*a):
            gc.CARRY = (Path("candidate.pt"),)
            raise RuntimeError("evaluation failure")
        with patch.object(evaluation, "_evaluate_series", side_effect=fail):
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_series("not_read.json")
        self.assertEqual(gc.CARRY, original)

    def test_iq_rebinding_reuses_grid_maps_without_keeping_old_view(self):
        ctrl = carry_runtime.LearningAttackerCarryGCController.__new__(
            carry_runtime.LearningAttackerCarryGCController)
        ctrl.has_priority_cells = True
        ctrl.priority_cells = REGISTERED_PLANT_CELLS
        ctrl.waypoint_cells = carry.WAYPOINT_CELLS
        first, second = NS(grid=carry.GRID), NS(grid=carry.GRID)
        ctrl.set_game(first)
        distances = ctrl._priority_dist_map
        ctrl.set_game(second)
        self.assertIs(ctrl.game, second)
        self.assertIs(ctrl._priority_dist_map, distances)

    def test_carry_does_not_change_registered_target_with_iq_noise(self):
        from iq_perception import PerceivedGameView
        from unittest.mock import patch
        env = carry.CarryEnv()
        env.reset(pattern_marker=5)
        target = env.target_plant_pos
        real = NS(grid=carry.GRID, chars=env.attackers, smokes=[],
                  target_plant_pos=target, battle_tick=0, round_timer=100)
        ctrl = carry_runtime.LearningAttackerCarryGCController.__new__(
            carry_runtime.LearningAttackerCarryGCController)
        ctrl.positioning_version = 2
        ctrl.has_priority_cells = True
        ctrl.priority_cells = REGISTERED_PLANT_CELLS
        ctrl.waypoint_cells = carry.WAYPOINT_CELLS
        ctrl._build_observation = Mock(return_value=np.zeros(29))
        ctrl._build_mask = Mock(return_value=np.ones(11, dtype=bool))
        ctrl._select_action = Mock(return_value=0)
        ctrl.reset_round()
        with patch.object(carry_runtime, "choose_pre_entry_ability", return_value=None):
            for wrong in ((target[0] + 1, target[1]), (target[0] - 1, target[1])):
                view = PerceivedGameView(real, {"target_plant_pos": wrong}, {})
                ctrl.set_game(view)
                ctrl.decide_move(env.carrier, {"chars": env.attackers,
                                             "target_plant_pos": wrong})
                self.assertEqual(ctrl._build_observation.call_args.args[-1], target)
                self.assertEqual(real.target_plant_pos, target)


if __name__ == "__main__":
    unittest.main()
