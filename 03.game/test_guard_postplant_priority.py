import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
from gc_v1.learning_attacker_guard_gc import LearningAttackerGuardGCController
import gc_v1.learning_attacker_guard_gc as guard
from abilities_los import AbilityLosMixin


class GuardPostplantPriorityTests(unittest.TestCase):
    def test_postplant_los_matches_engine_smoke_rules(self):
        from gc_v1.learning_defender_retake_gc import _has_los as retake_los
        grid = np.zeros((7, 7), dtype=int)
        engine = AbilityLosMixin()
        engine.grid = grid
        engine.smokes = [{"cells": {(3, 3)}}]
        for start, end in [((3, 1), (3, 5)), ((3, 3), (3, 4)),
                           ((3, 3), (3, 5)), ((1, 1), (1, 5))]:
            expected = engine.check_cell_line_of_sight(start, end)
            for los in (guard._has_los, retake_los):
                self.assertEqual(los(grid, start, end, {(3, 3)}), expected)

    def test_active_defuse_releases_spike_watch_hold(self):
        import torch
        from unittest.mock import Mock
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController)
        controller.verbose = False
        controller._ensure_spike_dist_map = Mock()
        controller._ensure_guard_assignment = Mock()
        controller._maybe_advance_tick = Mock()
        controller._update_sighting_dist_map = Mock()
        controller._build_observation = Mock(return_value=(np.zeros(34, dtype=np.float32), []))
        controller._action_mask = Mock(return_value=np.ones(10, dtype=bool))
        controller.model = lambda x: torch.tensor([[0., 0., 10., 0., 0., 0., 0., 0., 0., 0.]], device=x.device)
        char = NS(name="p", pos=[3, 3], is_alive=True)
        state = {"grid": np.zeros((7, 7), dtype=int), "chars": [char],
                 "is_planted": True, "planted_pos": [3, 4],
                 "detonate_timer": 30, "defender_defuse_info": {"enemy": (2, 6)}}
        self.assertEqual(controller.decide_move(char, state), [2, 3])

    def test_guard_route_precedes_spike_watch_until_arrival(self):
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController
        )
        controller._assigned_guard_positions = {"p": (2, 4)}
        distances = np.array([
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [8, 7, 6, 5, 4],
            [9, 8, 7, 6, 5],
        ], dtype=np.int32)
        controller._assigned_dist_maps = {"p": distances}
        char = NS(name="p", pos=[2, 1], is_alive=True)
        step = controller._guard_position_step(char, np.zeros((5, 5), dtype=int), [char])
        self.assertEqual(step, [2, 2])

    def test_spike_watch_can_resume_after_guard_arrival(self):
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController
        )
        controller._assigned_guard_positions = {"p": (2, 2)}
        controller._assigned_dist_maps = {"p": np.zeros((5, 5), dtype=np.int32)}
        char = NS(name="p", pos=[2, 2], is_alive=True)
        self.assertIsNone(controller._guard_position_step(
            char, np.zeros((5, 5), dtype=int), [char]
        ))


if __name__ == "__main__":
    unittest.main()
