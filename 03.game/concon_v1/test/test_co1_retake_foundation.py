"""Foundation coverage, learned inference, real IQ defusing and legacy loading.

These tests construct known policy values in memory; they never run training.
"""

from pathlib import Path
import sys
import contextlib
import io
import copy
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_retake_start_positions import START_CELLS, validate_starts
from concon_v1.co1_retake_scenarios import get_scenario
from concon_v1.co1_retake_common import RetakeDQN, build_inputs, DEFUSE_ACTION, GORIGONS, FEATURE_DIM, MAP_CHANNELS, foundation_preservation_loss
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_retake_foundation import training_targets, teacher_values, perceived_plant_cells, evaluate_foundation
from concon_v1.co1_train_defender_retake import main, make_checkpoint, load_training_model, optimize
from concon_v1.co1_train_defender_retake_base import main as base_main
from concon_v1.test.test_co1_defender_retake import actor


def fixture_model(scenario):
    model = RetakeDQN(scenario, foundation=True)
    indices, values = training_targets(scenario)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.foundation_values.weight[indices] = torch.as_tensor(values)
        model.foundation_trained[indices] = True
    model.foundation_values.weight.requires_grad_(False)
    return model


class FoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_explicit_starts_cover_five_posts_and_each_site(self):
        self.assertEqual(validate_starts(), START_CELLS)
        self.assertTrue(all(len(cells) >= 5 for cells in START_CELLS.values()))
        for site in ("L", "R"):
            scenario = get_scenario(site)
            indices, targets = training_targets(scenario, seed=0)
            coverage = set(indices.tolist())
            height, width = scenario.grid.shape
            for plant, spike in enumerate(perceived_plant_cells(scenario)):
                for candidates in START_CELLS.values():
                    for r, c in candidates:
                        self.assertIn(plant * height * width + r * width + c, coverage)
            self.assertTrue((targets[:, 5] == 2).any())
            self.assertEqual(len(indices), len(set(indices.tolist())))
            # Sample ordering is shuffled reproducibly while preserving coverage.
            other, _ = training_targets(scenario, seed=123)
            self.assertFalse(np.array_equal(indices, other))

    def test_teacher_prefers_progress_and_defuse_over_leaving_spike(self):
        scenario = get_scenario("L")
        values = teacher_values(scenario.grid, (7, 3), (12, 3))
        self.assertEqual(values[0], 1.)
        self.assertGreater(values[0], values[4])
        values = teacher_values(scenario.grid, (7, 3), (8, 3))
        self.assertEqual(int(values.argmax()), 5)

    def test_production_iq_model_reaches_and_defuses_both_sites_from_search_posts(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            model = fixture_model(scenario)
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_foundation(model, scenario, trials=20, seed=7919)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["arrival_rate"], 1.)
            self.assertEqual(result["defuse_rate"], 1.)
            self.assertEqual({row["slot"] for row in result["details"]}, set("abcde"))
            self.assertTrue(all(row["max_defuse_ticks"] >= 6 for row in result["details"]))

    def test_foundation_remains_available_with_enemy_memory_but_not_firing_contact(self):
        scenario = get_scenario("L")
        model = fixture_model(scenario)
        controller = ConconDefenderRetakeController("L", model=model)
        char = actor(GORIGONS.players[0], (7, 3))
        state = dict(grid=scenario.grid, chars=[char], is_planted=True, planted_pos=(7, 3),
                     smoke_cells=[], battle_tick=1, detonate_timer=30, defender_defuse_info={})
        obs, _, _ = build_inputs(controller, char, state)
        self.assertGreater(float(model(torch.as_tensor(obs)[None])[0, DEFUSE_ACTION]), 1.)
        state["chars"].append(actor("known", (8, 3), team="A"))
        obs, _, _ = build_inputs(controller, char, state)
        self.assertEqual(float(model(torch.as_tensor(obs)[None])[0, DEFUSE_ACTION]), 0.)
        # An old sighting remains in map channel 4 after the live contact ends.
        state["chars"] = [char]
        state["battle_tick"] = 2
        obs, _, context = build_inputs(controller, char, state)
        self.assertFalse(context["fireable"])
        self.assertGreater(obs[:model.map_size].reshape(MAP_CHANNELS, model.height, model.width)[4].sum(), 0)
        self.assertGreater(float(model(torch.as_tensor(obs)[None])[0, DEFUSE_ACTION]), 1.)

    def test_auxiliary_loss_restores_defuse_priority_and_respects_combat_wait_and_legality(self):
        scenario = get_scenario("L")
        model = fixture_model(scenario)
        char = actor(GORIGONS.players[0], (7, 3))
        controller = ConconDefenderRetakeController("L", model=model)
        state = dict(grid=scenario.grid, chars=[char], is_planted=True, planted_pos=(7, 3),
                     smoke_cells=[], battle_tick=1, detonate_timer=30, defender_defuse_info={})
        observation, _, _ = build_inputs(controller, char, state)
        obs = torch.as_tensor(observation)[None]
        mask = torch.zeros((1, DEFUSE_ACTION + 1), dtype=torch.bool)
        mask[0, 32:40], mask[0, DEFUSE_ACTION] = True, True
        values = torch.zeros_like(mask, dtype=torch.float32)
        values[0, 32], values[0, 40] = 3., 10.
        values.requires_grad_(True)
        loss = foundation_preservation_loss(model, obs, values, mask)
        self.assertGreater(float(loss), 0.)
        loss.backward()
        self.assertGreater(float(values.grad[0, 32]), 0.)
        self.assertLess(float(values.grad[0, DEFUSE_ACTION]), 0.)
        self.assertEqual(float(values.grad[0, 40]), 0.)  # No utility/facing target.
        self.assertIsNone(model.foundation_values.weight.grad)
        for feature in (16,):
            blocked = obs.clone()
            blocked[0, model.map_size + feature] = 1.
            self.assertEqual(float(foundation_preservation_loss(model, blocked, values, mask)), 0.)
        assembly = obs.clone()
        assembly[0, model.map_size + FEATURE_DIM] = 1.
        self.assertGreater(float(foundation_preservation_loss(model, assembly, values, mask)), 0.)
        mask[0, DEFUSE_ACTION] = False
        self.assertEqual(float(foundation_preservation_loss(model, obs, values, mask)), 0.)

    def test_td_optimizer_preserves_legal_defuse_preference_without_updating_foundation(self):
        scenario = get_scenario("L")
        model = fixture_model(scenario)
        char = actor(GORIGONS.players[0], (7, 3))
        controller = ConconDefenderRetakeController("L", model=model)
        state = dict(grid=scenario.grid, chars=[char], is_planted=True, planted_pos=(7, 3),
                     smoke_cells=[], battle_tick=1, detonate_timer=30, defender_defuse_info={})
        obs, mask, _ = build_inputs(controller, char, state)
        with torch.no_grad():
            model.head[2].bias[32] = 4.
        target = copy.deepcopy(model)
        transition = (obs, 32, -10., obs, mask, mask, 1., 1)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        loss = optimize(model, target, optimizer, [transition], batch_size=1)
        self.assertTrue(np.isfinite(loss))
        self.assertLess(float(model.head[2].bias.grad[DEFUSE_ACTION]), 0.)
        self.assertIsNone(model.foundation_values.weight.grad)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_resume_can_add_foundation_without_replacing_old_battle_network(self):
        scenario = get_scenario("L")
        foundation = fixture_model(scenario)
        foundation.foundation_evaluation = dict(passed=True)
        base = make_checkpoint(foundation, "L", 0, 6, "unused", [], 0)
        legacy = RetakeDQN(scenario)
        with torch.no_grad():
            legacy.head[2].bias.fill_(.25)
        old = make_checkpoint(legacy, "L", 100, 6, "search", [], 10)
        with patch("pathlib.Path.is_file", return_value=True), patch(
                "concon_v1.co1_train_defender_retake.torch.load", side_effect=[old, base]), contextlib.redirect_stdout(io.StringIO()):
            model, checkpoint = load_training_model("L", Path("data"), dict(FLASH=6, RECON=6, SMOKE=6), Path("previous"))
        self.assertIs(checkpoint, old)
        torch.testing.assert_close(model.head[2].bias, legacy.head[2].bias)
        torch.testing.assert_close(model.foundation_values.weight, foundation.foundation_values.weight)
        self.assertFalse(model.foundation_values.weight.requires_grad)

    def test_battle_requires_a_validated_foundation_and_foundation_cli_is_separate(self):
        scenario = get_scenario("L")
        model = fixture_model(scenario)
        cp = make_checkpoint(model, "L", 0, 6, "unused", [], 0)
        with patch("pathlib.Path.is_file", return_value=True), patch("concon_v1.co1_train_defender_retake.torch.load", return_value=cp):
            with self.assertRaises(ValueError):
                load_training_model("L", Path("data"), dict(FLASH=6, RECON=6, SMOKE=6))
        with patch("concon_v1.co1_train_defender_retake_base.train_foundation") as foundation, patch("concon_v1.co1_train_defender_retake.train") as battle:
            base_main(["--steps", "400", "--eval-trials", "10"])
            self.assertEqual(foundation.call_args.args[:2], (400, 10))
            battle.assert_not_called()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--mode", "foundation"])
            battle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
