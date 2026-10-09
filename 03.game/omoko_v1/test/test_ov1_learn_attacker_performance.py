"""Geometry reuse and scoped CPU inference, without training or saved models."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

AI_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIRECTORY.parent))
sys.path.insert(0, str(AI_DIRECTORY))

import ov1_learning_attacker_carry as carry
import ov1_learning_attacker_escort as escort
import ov1_learning_attacker_guard as guard
import ov1_learning_attacker_retrieve as retrieve
from simulation_runtime import cpu_inference, match_inference_device


class AttackerPerformanceTests(unittest.TestCase):
    def make_carry(self):
        controller = carry.Ov1LearningAttackerCarryController.__new__(carry.Ov1LearningAttackerCarryController)
        controller._real_game = None
        controller.waypoint_cells_by_site = {"left": [(0, 0)]}
        controller.has_priority_cells = True
        controller.priority_cells = [(0, 2)]
        return controller

    def make_escort(self):
        controller = escort.Ov1LearningAttackerEscortController.__new__(escort.Ov1LearningAttackerEscortController)
        controller._goal_dist_cache = {}
        controller._carry_dist_cache = {}
        controller.team_sighting = SimpleNamespace(last_seen_enemy=None)
        controller.max_ticks = 100
        controller._char_state = {}
        return controller

    def test_carry_reuses_geometry_but_updates_perception_view(self):
        grid = np.array([[0, 0, 2], [0, 1, 0]], dtype=np.int32)
        controller = self.make_carry()
        original = SimpleNamespace(grid=grid)
        with patch.object(carry, "_bfs_distance_map", wraps=carry._bfs_distance_map) as bfs, \
                patch.object(carry, "_ray_openness", wraps=carry._ray_openness) as openness:
            controller.set_game(original)
            distances = controller._waypoint_dist_maps
            ray_count = openness.call_count
            # Different view/array instances must still reuse the same geometry.
            for _ in range(5):
                view = SimpleNamespace(grid=grid.copy())
                controller.set_game(view)
                self.assertIs(controller.game, view)
            self.assertIs(controller._real_game, original)
            self.assertIs(controller._waypoint_dist_maps, distances)
            self.assertEqual(bfs.call_count, 1)
            self.assertEqual(openness.call_count, ray_count)
        np.testing.assert_array_equal(distances["left"][0], carry._bfs_distance_map(grid, (0, 0)))

    def test_carry_rebuilds_when_tiles_change_in_place(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        controller = self.make_carry()
        game = SimpleNamespace(grid=grid)
        controller.set_game(game)
        grid[0, 1] = 1
        grid[0, 2] = 2
        controller.set_game(game)
        np.testing.assert_array_equal(controller._waypoint_dist_maps["left"][0],
                                      carry._bfs_distance_map(grid, (0, 0)))
        self.assertEqual(controller._plant_cells, [(0, 2)])
        self.assertNotIn((0, 1), controller._cell_openness)

    def test_escort_orb_selection_reuses_bfs_across_characters(self):
        controller = self.make_escort()
        grid = np.zeros((5, 5), dtype=np.int32)
        orbs = {(0, 1), (4, 4)}
        with patch.object(escort, "_build_distance_map_walls_only",
                          wraps=escort._build_distance_map_walls_only) as bfs:
            for i in range(4):
                char = SimpleNamespace(name=str(i), pos=(0, 0), team="A", ultimate_points=0, ultimate_cost=6)
                self.assertEqual(controller._opportunistic_orb_target(
                    char, grid, [char], orbs, {"round_timer": 100}), (0, 1))
            self.assertEqual(bfs.call_count, len(orbs))

    def test_escort_cached_next_step_preserves_blocking_and_tie_breaks(self):
        controller = self.make_escort()
        grid = np.zeros((6, 6), dtype=np.int32)
        grid[1:5, 2] = 1
        goal = (4, 4)
        distances = controller._get_goal_dist_map(grid, goal)
        for start in [(0, 0), (4, 0), goal, (2, 5)]:
            for blocked in [(), ((5, 0),), ((0, 1), (1, 0))]:
                expected = escort._bfs_next_step(grid, start, goal, blocked)
                with patch.object(escort, "_build_distance_map_walls_only",
                                  side_effect=AssertionError("Cached steps must not run BFS")):
                    self.assertEqual(escort._bfs_next_step(grid, start, goal, blocked, dist_map=distances), expected)

    def test_escort_cache_distinguishes_shape_and_in_place_changes(self):
        controller = self.make_escort()
        for method in [controller._get_goal_dist_map, controller._get_carry_dist_map]:
            grid = np.zeros((2, 6), dtype=np.int32)
            distances = method(grid, (0, 0))
            self.assertIs(method(grid.copy(), (0, 0)), distances)
            reshaped = grid.reshape(3, 4)
            self.assertEqual(method(reshaped, (0, 0)).shape, (3, 4))
            grid[0, 1] = 1
            np.testing.assert_array_equal(method(grid, (0, 0)),
                                          escort._build_distance_map_walls_only(grid, [(0, 0)]))

    def test_all_attacker_models_use_cpu_in_match_even_with_cuda_default(self):
        cases = [
            (carry, carry.Ov1LearningAttackerCarryController,
             {"model_state_dict": carry.AttackerCarryDuelingDQN().state_dict()}, "policy_net"),
            (escort, escort.Ov1LearningAttackerEscortController,
             {"model_state_dict": escort.DuelingQNetwork(escort.OBS_DIM, escort.N_ACTIONS).state_dict()}, "policy_net"),
            (retrieve, retrieve.Ov1LearningAttackerRetrieveController,
             {"model_state_dict": retrieve.DuelingQNet().state_dict()}, "model"),
            (guard, guard.Ov1LearningAttackerGuardController,
             guard.AttackerGuardDuelingDQN().state_dict(), "model"),
        ]
        previous = match_inference_device()
        with cpu_inference(), patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(retrieve, "DEVICE", torch.device("cuda")), \
                patch.object(guard, "DEVICE", torch.device("cuda")), \
                patch.object(carry.os.path, "isfile", return_value=True):
            for module, factory, checkpoint, model_attr in cases:
                with self.subTest(controller=factory.__name__), \
                        patch.object(torch, "load", return_value=checkpoint) as load:
                    controller = factory(model_path="unused.pt")
                    self.assertEqual(controller.device, torch.device("cpu"))
                    self.assertEqual(next(getattr(controller, model_attr).parameters()).device, torch.device("cpu"))
                    self.assertEqual(load.call_args.kwargs["map_location"], torch.device("cpu"))
        self.assertEqual(match_inference_device(), previous)

    def test_explicit_device_takes_priority_over_match_default(self):
        checkpoint = {"model_state_dict": carry.AttackerCarryDuelingDQN().state_dict()}
        with patch.object(carry, "match_inference_device", return_value="cuda"), \
                patch.object(carry.os.path, "isfile", return_value=True), \
                patch.object(torch, "load", return_value=checkpoint):
            controller = carry.Ov1LearningAttackerCarryController(model_path="unused.pt", device="cpu")
            self.assertEqual(controller.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
