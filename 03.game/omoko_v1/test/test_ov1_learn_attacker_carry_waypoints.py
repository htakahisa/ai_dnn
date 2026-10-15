"""Current-map route discovery without training or writing checkpoints."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ov1_learning_attacker_carry as carry
import ov1_train_attacker_carry as training


class CarryWaypointTests(unittest.TestCase):
    def test_current_map_matches_training_waypoints_and_order(self):
        self.assertEqual(carry._parse_waypoint_cells(carry.CARRY_MAZE_STR),
                         training.WAYPOINT_CELLS_BY_SITE)

    def test_route_count_is_not_fixed_and_missing_markers_stay_missing(self):
        for count in (0, 1, 4, 8):
            with self.subTest(count=count):
                cells = carry._parse_waypoint_cells("0" + "6" * count + "7" * count)
                self.assertEqual(len(cells.get("right", [])), count)
                self.assertEqual(len(cells.get("left", [])), count)
                if count == 0:
                    self.assertEqual(cells, {})

    def test_model_load_uses_map_instead_of_stale_saved_waypoints(self):
        checkpoint = {
            "obs_dim": carry.OBS_DIM,
            "n_actions": carry.ACTION_DIM,
            "model_state_dict": carry.AttackerCarryDuelingDQN(
                obs_dim=carry.OBS_DIM, action_dim=carry.ACTION_DIM,
            ).state_dict(),
            "waypoint_cells": {"right": [(99, 99)]},
            "priority_cells": [(0, 0)],
            "has_priority_cells": True,
        }
        with patch.object(carry.os.path, "isfile", return_value=True), \
                patch.object(carry.torch, "load", return_value=checkpoint), \
                patch.object(carry, "CARRY_MAZE_STR", "06767\n06767"):
            controller = carry.Ov1LearningAttackerCarryController("unused.pt", device="cpu")
        self.assertEqual(controller.waypoint_cells_by_site,
                         {"right": [(0, 1), (0, 3), (1, 1), (1, 3)],
                          "left": [(0, 2), (0, 4), (1, 2), (1, 4)]})
        self.assertEqual(controller.priority_cells, [(0, 0)])
        controller._priority_cells_by_site = {"left": [(0, 0)], "right": [(0, 0)]}
        controller._plant_cells_by_site = {}
        # Existing selection enumerates every discovered site/waypoint pair.
        with patch.object(carry.random, "choice", side_effect=lambda routes: routes[-1]) as choose:
            self.assertEqual(controller._choose_route(1), ("right", 3))
        self.assertEqual(len(choose.call_args.args[0]), 8)


if __name__ == "__main__":
    unittest.main()
