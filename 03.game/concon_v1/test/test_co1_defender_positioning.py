"""Check foundation wiring and real setup/live motion without running training."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_defender_common import DefenderSearchDQN, build_inputs, GORIGONS
from concon_v1.co1_defender_positioning import evaluate_positioning
from concon_v1.co1_defender_scenario import get_scenario, phase_scenario, validate_checkpoint
from concon_v1.co1_guard_positioning import training_targets
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_train_defender_search import make_checkpoint


class DefenderPositioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scenario = get_scenario()
        cls.model = DefenderSearchDQN(cls.scenario)
        # Reference tables test runtime execution; no optimizer/training run.
        with torch.no_grad():
            for setup in (False, True):
                indices, values = training_targets(phase_scenario(cls.scenario, setup))
                cls.model.navigation_values.weight[indices + int(setup) * cls.model.phase_size] = values
            for preferred in range(4):
                for neighbors in range(16):
                    if neighbors & (1 << preferred):
                        values = cls.model.navigation_yield_values.weight[preferred * 16 + neighbors].reshape(5, 8)
                        values[:4] = 2.
                        values[preferred ^ 1] = 5.
                        values[preferred] = -6.
                        values[4] = -2.

    def test_engine_setup_to_live_uses_iq_and_learned_values_without_teacher(self):
        with patch("concon_v1.co1_guard_positioning.training_targets", side_effect=AssertionError("runtime teacher")), \
                patch("concon_v1.co1_defender_positioning.training_targets", side_effect=AssertionError("runtime teacher")):
            result = evaluate_positioning(self.model, self.scenario, trials=5)
        self.assertTrue(result["passed"], result)

    def test_setup_posts_are_distinct_reachable_staging_cells(self):
        from concon_v1.co1_attacker_common import bfs_distance_map
        self.assertEqual(len(set(self.scenario.setup_positions.values())), 5)
        starts = list(zip(*np.where(self.scenario.grid == 4)))
        for point in self.scenario.setup_positions.values():
            distances = bfs_distance_map(self.scenario.setup_grid, point)
            self.assertTrue(all(distances[start] >= 0 for start in starts))

    def test_hidden_enemies_do_not_enter_observations_or_masks(self):
        from types import SimpleNamespace
        char = SimpleNamespace(name=GORIGONS.players[0], team="D", is_alive=True, pos=[1, 18], facing="S")
        controller = ConconDefenderSearchController(model=self.model)
        controller.assignments = {char.name: "a"}
        enemy = SimpleNamespace(name="hidden", team="A", is_alive=True, pos=[1, 19], facing="N", position_known=False)
        state = dict(grid=self.scenario.grid, chars=[char], defender_setup_active=True)
        observation, mask = build_inputs(controller, char, state)
        state["chars"].append(enemy)
        hidden_observation, hidden_mask = build_inputs(controller, char, state)
        np.testing.assert_array_equal(observation, hidden_observation)
        np.testing.assert_array_equal(mask, hidden_mask)
        enemy.position_known = True
        disclosed, disclosed_mask = build_inputs(controller, char, state)
        self.assertFalse(np.array_equal(observation, disclosed))
        self.assertFalse(disclosed_mask[24:32].any())

    def test_setup_restriction_ends_without_reassigning_posts(self):
        from types import SimpleNamespace
        char = SimpleNamespace(name=GORIGONS.players[0], team="D", is_alive=True,
                               pos=[2, 35], facing="N", facing_forced_this_tick=True)
        controller = ConconDefenderSearchController(model=self.model)
        state = dict(grid=self.scenario.grid, chars=[char], defender_setup_active=True)
        _, setup_mask = controller.policy_inputs(char, state)
        assigned = dict(controller.assignments)
        self.assertFalse(setup_mask[24:32].any())  # (2, 36) is forbidden only in setup.
        state["defender_setup_active"] = False
        _, live_mask = controller.policy_inputs(char, state)
        self.assertTrue(live_mask[24])
        self.assertFalse(live_mask[25:32].any())  # Engine-locked facing is respected.
        self.assertEqual(controller.assignments, assigned)

    def test_checkpoint_roundtrip_and_adapter_retake_delegation(self):
        from concon_v1.co1_defender_controller import ConconDefenderController
        checkpoint = make_checkpoint(self.model, self.scenario, {"passed": True}, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "basic.pt"
            torch.save(checkpoint, path)
            adapter = ConconDefenderController(model_path=path)
            for name, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, adapter.search_controller.model.state_dict()[name]))
            with patch.object(adapter.default_controller, "decide_move", return_value=[1, 18]) as fallback:
                self.assertEqual(adapter.decide_move(None, {"is_planted": True}), [1, 18])
                fallback.assert_called_once()
        checkpoint["scenario_signature"] = "changed"
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, self.scenario)


if __name__ == "__main__":
    unittest.main()
