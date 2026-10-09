"""Visibility and inference optimizations preserve masks, choices and training."""

import unittest
from collections import deque
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import torch

from abilities_los import AbilityLosMixin
from frc_v1 import FACING_STEPS
from frc_v1.baseline import _bfs_route_step, _cached_route_step, _grid_key, _route_distances, route_step
from frc_v1.navigation import _cached_walk_distances, _walk_distances
from frc_v1.model import FrcPolicy
from frc_v1.memory import FrcMemory
from frc_v1.observation import FrcObservationEncoder
from grid_paths import _distance, _distance_map, distance_map, walking_distance
from grid_lines import _line_cells as cached_line, line_cells, iter_line_cells
from grid_visibility import _clear_mask, _rays, visible_cells
from simulation_runtime import cpu_inference, match_inference_device
from test_frc_v1 import make_game, observe


class Geometry(AbilityLosMixin):
    pass


class VisibilityOptimizationTests(unittest.TestCase):
    def test_projectile_paths_preserve_all_directions_and_stop_at_walls(self):
        rng = np.random.default_rng(713)
        game = Geometry()
        game.height, game.width = 7, 9
        for _ in range(3):
            game.grid = (rng.random((game.height, game.width)) < .2).astype(np.int32)
            for start in ((0, 0), (3, 4), (6, 8)):
                for r in range(game.height):
                    for c in range(game.width):
                        scale = max(game.height, game.width) * 3
                        far = (start[0] + (r - start[0]) * scale,
                               start[1] + (c - start[1]) * scale)
                        expected = [start]
                        for rr, cc in line_cells(start, far)[1:]:
                            if not (0 <= rr < game.height and 0 <= cc < game.width):
                                break
                            if game.grid[rr, cc] == 1:
                                break
                            expected.append((rr, cc))
                        self.assertEqual(game._projectile_path(start, (r, c)), expected)

    def test_projectile_only_generates_cells_until_first_wall(self):
        game = Geometry()
        game.height, game.width = 26, 44
        game.grid = np.zeros((game.height, game.width), np.int32)
        game.grid[10, 12] = 1
        generated = []
        def counted(start, end):
            for cell in iter_line_cells(start, end):
                generated.append(cell)
                yield cell
        with patch("abilities_los.iter_line_cells", side_effect=counted):
            self.assertEqual(game._projectile_path((10, 10), (10, 40)),
                             [(10, 10), (10, 11)])
        self.assertEqual(generated, [(10, 10), (10, 11), (10, 12)])

    def test_geometry_cache_preserves_directed_ties_and_mutable_return_values(self):
        self.assertEqual(line_cells((0, 0), (1, 2)), [(0, 0), (1, 1), (1, 2)])
        self.assertEqual(line_cells((1, 2), (0, 0)), [(1, 2), (0, 1), (0, 0)])
        path = line_cells((0, 0), (1, 2))
        path.clear()
        self.assertEqual(len(line_cells((0, 0), (1, 2))), 3)
        self.assertEqual(cached_line.cache_info().maxsize, 4096)
        self.assertEqual(_rays.cache_info().maxsize, 128)

    def test_matches_engine_for_all_directions_walls_smoke_and_moving_viewers(self):
        rng = np.random.default_rng(713)
        for shape in ((5, 7), (11, 14)):
            game = Geometry()
            for iteration in range(12):
                game.grid = (rng.random(shape) < .23).astype(np.int32)
                smoke = {tuple(map(int, cell)) for cell in np.argwhere(rng.random(shape) < .15)}
                game.smokes = [{"cells": smoke}]
                for direction in FACING_STEPS:
                    viewers = [((int(rng.integers(shape[0])), int(rng.integers(shape[1]))), direction),
                               ((1, 1), (0, 1))]
                    expected = {(r, c) for r in range(shape[0]) for c in range(shape[1])
                        if game.grid[r, c] != 1 and (r, c) not in smoke and any(
                            dr * (r - origin[0]) + dc * (c - origin[1]) >= -1e-9
                            and game.check_cell_line_of_sight(origin, (r, c))
                            for origin, (dr, dc) in viewers)}
                    with self.subTest(shape=shape, iteration=iteration, direction=direction):
                        self.assertEqual(visible_cells(game.grid, viewers, smoke), expected)

    def test_cache_is_bounded_readonly_and_refreshes_after_wall_or_smoke_change(self):
        grid = np.zeros((5, 7), np.int32)
        viewers = [((2, 1), (0, 1))]
        before = visible_cells(grid, viewers, ())
        before.clear()
        self.assertIn((2, 6), visible_cells(grid, viewers, ()))
        self.assertNotIn((2, 6), visible_cells(grid, viewers, {(2, 3)}))
        grid[2, 3] = 1
        self.assertNotIn((2, 6), visible_cells(grid, viewers, ()))
        for i in range(300):
            _clear_mask(grid.shape, (grid == 1).tobytes(), (2, 1), frozenset({(i, i)}))
        self.assertLessEqual(_clear_mask.cache_info().currsize, 256)
        self.assertFalse(_clear_mask(grid.shape, (grid == 1).tobytes(), (2, 1), frozenset()).flags.writeable)

    def test_frc_sensor_preserves_custom_los(self):
        game = make_game()
        with patch.object(game, "check_cell_line_of_sight", return_value=False) as los:
            snapshot = observe(game)[0]
        self.assertEqual(snapshot.visible_cells, ())
        self.assertTrue(los.called)


class PathOptimizationTests(unittest.TestCase):
    def test_route_cannot_bypass_first_step_block_by_returning_through_origin(self):
        grid = np.ones((3, 5), np.int32)
        grid[1, 1:4] = 0
        grid[0, 1] = 0
        self.assertEqual(route_step(grid, (1, 1), {(1, 3)}), ("E", 2))
        self.assertEqual(route_step(grid, (1, 1), {(1, 3)}, first_step_blocked={(1, 2)}), ("STAY", 10000))
        self.assertEqual(_route_distances.cache_info().maxsize, 512)

    def test_cached_map_watch_points_refresh_after_map_changes(self):
        from frc_v1.navigation import (site_approaches, site_interdiction_cells, site_watch_points,
            _cached_site_approaches, _cached_site_interdiction, _cached_site_watch_points)
        from frc_v1.baseline import plant_sites
        grid = np.zeros((9, 12), np.int32)
        grid[3:6, 6:9] = 2
        grid[1, 1] = 3
        grid[7, 10] = 4
        before = site_watch_points(grid, plant_sites(grid)[0])
        for iteration in range(2):
            frozen = _grid_key(grid)
            site = frozenset(plant_sites(grid)[0])
            from frc_v1.actions import MOVE_STEPS
            moves = tuple(MOVE_STEPS.items())
            self.assertEqual(site_approaches(grid, site, 3),
                             _cached_site_approaches.__wrapped__(frozen, site, 3, moves))
            self.assertEqual(site_interdiction_cells(grid, site, 4),
                             _cached_site_interdiction.__wrapped__(frozen, site, 4, moves))
            self.assertEqual(site_watch_points(grid, site),
                             _cached_site_watch_points.__wrapped__(frozen, site, moves))
            grid[:, 4] = 1
        self.assertNotEqual(site_watch_points(grid, site), before)

    def test_toru_distance_fields_match_bfs_and_do_not_share_mutable_arrays(self):
        import run_game  # Installs the same controller import paths as a match.
        from attacker_v3.learning_attacker_retrieve import _bfs_distance_map as retrieve
        from defender_v3.learning_defender_search import _bfs_distance_map as search
        from defender_v3.learning_defender_retake import _bfs_distance_map as retake
        rng = np.random.default_rng(915)
        for _ in range(40):
            grid = (rng.random((8, 12)) < .25).astype(np.int32)
            goal = int(rng.integers(8)), int(rng.integers(12))
            expected = np.full(grid.shape, -1, np.int32)
            queue = deque()
            if grid[goal] != 1:
                expected[goal] = 0
                queue.append(goal)
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 12 and grid[nr, nc] != 1 and expected[nr, nc] == -1:
                        expected[nr, nc] = expected[r, c] + 1
                        queue.append((nr, nc))
            for calculate in (distance_map, retrieve, search, retake):
                actual = calculate(grid, goal)
                np.testing.assert_array_equal(actual, expected)
                actual[:] = 999
                np.testing.assert_array_equal(calculate(grid, goal), expected)
        self.assertEqual(_distance_map.cache_info().maxsize, 512)

    def test_general_distances_respect_occupants_walls_and_cached_grid_mutations(self):
        grid = np.zeros((4, 5), np.int32)
        self.assertEqual(walking_distance(grid, (1, 0), (1, 4)), 4)
        self.assertEqual(walking_distance(grid, (1, 0), (1, 4), {(1, 2)}), 6)
        self.assertEqual(walking_distance(grid, (1, 0), (1, 4), {(1, 4)}), float("inf"))
        grid[:, 2] = 1
        self.assertEqual(walking_distance(grid, (1, 0), (1, 4)), float("inf"))
        self.assertEqual(walking_distance(grid, (1, 0), (1, 0), {(1, 0)}), 0)
        self.assertEqual(_distance.cache_info().maxsize, 4096)

    def test_frc_routes_match_original_ties_goals_and_first_step_occupants(self):
        rng = np.random.default_rng(813)
        for i in range(900):
            grid = (rng.random((8, 12)) < .25).astype(np.int32)
            start = int(rng.integers(8)), int(rng.integers(12))
            goals = frozenset((int(rng.integers(8)), int(rng.integers(12))) for _ in range(3))
            blocked = frozenset((int(rng.integers(8)), int(rng.integers(12))) for _ in range(5))
            first = frozenset((int(rng.integers(8)), int(rng.integers(12))) for _ in range(4))
            from frc_v1.actions import MOVE_STEPS
            expected = _bfs_route_step(_grid_key(grid), start, goals, blocked - goals,
                                                       first, tuple(MOVE_STEPS.items()))
            self.assertEqual(route_step(grid, start, goals, blocked, first), expected)
        lengths = _walk_distances(grid, (start,))
        original = _cached_walk_distances.__wrapped__(_grid_key(grid), (start,), tuple(MOVE_STEPS.values()))
        self.assertEqual(lengths, original)
        lengths.clear()
        self.assertEqual(_walk_distances(grid, (start,)), original)


class RuntimeInferenceTests(unittest.TestCase):
    def test_empty_effect_encoding_matches_full_network_and_preserves_training(self):
        game = make_game()
        game.grid[3, 15] = 2
        snapshot, _, observation = observe(game)
        with cpu_inference(), torch.no_grad():
            model = FrcPolicy(snapshot.grid, "A").model
            expected = model.features([observation])
            with patch.object(model.token_encoder, "forward", side_effect=AssertionError("empty effects encoded")):
                self.assertTrue(torch.equal(model.features([observation], runtime=True), expected))
            mask = observation.token_mask.copy()
            mask[0] = True
            nonempty = replace(observation, token_mask=mask)
            with patch.object(model.token_encoder, "forward", wraps=model.token_encoder.forward) as encode:
                self.assertTrue(torch.equal(model.features([nonempty], runtime=True), model.features([nonempty])))
                self.assertEqual(encode.call_count, 2)
    def test_single_cpu_decoder_matches_training_decoder_for_targets_dead_and_forced_players(self):
        game = make_game()
        game.grid[3, 15] = 2
        snapshot, _, _ = observe(game)
        with cpu_inference(), torch.no_grad():
            policy = FrcPolicy(snapshot.grid, "A")
            for iteration in range(24):
                allies = tuple(replace(a, alive=iteration != 23 and (iteration % 4 != 0 or a.slot % 2 == 0),
                    charges=2, points=a.cost, hp=40, forced_facing=iteration % 3 == 0)
                    for a in snapshot.allies)
                current = replace(snapshot, allies=allies, phase="setup" if iteration == 1 else "live")
                observation = FrcObservationEncoder().encode(current, FrcMemory().update(current))
                for slot, head in enumerate(policy.model.role_heads):
                    head[2].bias[:10] = -80
                    head[2].bias[(iteration + slot) % 10] = 80
                packed, _, _, _ = policy.model.distribution([observation], deterministic=True)
                actual = policy.model.greedy_record(observation)
                for key, expected in packed.items():
                    with self.subTest(iteration=iteration, key=key):
                        np.testing.assert_array_equal(actual[key], expected[0].numpy())

    def test_torch_thread_count_restores_on_nested_context_and_error(self):
        previous = torch.get_num_threads()
        self.assertIsNone(match_inference_device())
        with self.assertRaisesRegex(RuntimeError, "test"):
            with cpu_inference():
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertEqual(match_inference_device(), "cpu")
                with cpu_inference():
                    self.assertEqual(torch.get_num_threads(), 1)
                raise RuntimeError("test")
        self.assertEqual(torch.get_num_threads(), previous)
        self.assertIsNone(match_inference_device())
        with cpu_inference(enabled=False):
            self.assertEqual(torch.get_num_threads(), previous)

    def test_headless_toru_uses_cpu_for_eager_and_lazy_models(self):
        from run_game import _build_team_ai
        from defender_v3.learning_defender_retake import DuelingQNet, N_ACTIONS
        with cpu_inference():
            team = _build_team_ai("toru_ai_v3.1")
            attacker = team.attacker_factory()
            defender = team.defender_factory()
            self.assertEqual(attacker.retrieve_controller.device.type, "cpu")
            self.assertEqual(defender.search_controller.device.type, "cpu")
            retake = defender.retake_controller
            self.assertEqual(retake.device.type, "cpu")
            model = DuelingQNet(20, N_ACTIONS)
            with patch("torch.load", return_value=model.state_dict()):
                retake._lazy_init_model(20)
            self.assertEqual(next(retake.model.parameters()).device.type, "cpu")

    def test_runtime_actions_match_full_actor_and_skip_statistics_without_affecting_training(self):
        game = make_game()
        game.grid[3, 15] = 2
        snapshot, _, observation = observe(game)
        policy = FrcPolicy(snapshot.grid, "A")
        expected = policy.sample(observation)
        self.assertIsNotNone(policy.last_sample)
        policy.collect_statistics = False
        with patch.object(policy.model.critic, "forward", side_effect=AssertionError("runtime critic executed")):
            actual = policy.sample(observation)
        self.assertEqual(actual, expected)
        self.assertIsNone(policy.last_sample)
        policy.collect_statistics = True
        self.assertEqual(policy.sample(observation), expected)
        self.assertIsNotNone(policy.last_sample)


if __name__ == "__main__":
    unittest.main()
