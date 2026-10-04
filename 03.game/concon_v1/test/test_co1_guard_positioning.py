"""Exercise learned routes, repeated holding and checkpoint runtime selection."""

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from concon_v1.co1_guard_common import GuardDQN
from concon_v1.co1_guard_positioning import learn_positioning, positioning_evaluation, formation_evaluation, prepare_positioning
from concon_v1.co1_guard_scenarios import get_scenario
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_train_guard import make_checkpoint, qualifies_as_best, optimize, train
from concon_v1.co1_guard_battle_training import OPPONENTS, START_MODES, GuardBattleEnv


class LearnedPositioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.models = {}
        for site in ("L", "R"):
            scenario = get_scenario(site)
            model = GuardDQN(scenario, navigation=True)
            learn_positioning(model, scenario)
            cls.models[site] = model

    def test_autonomous_arrival_hold_and_facing_for_both_maps(self):
        for site, model in self.models.items():
            with self.subTest(site=site), patch(
                    "concon_v1.co1_guard_positioning.training_targets", side_effect=AssertionError("teacher used at runtime")):
                result = positioning_evaluation(model, get_scenario(site), trials=100)
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["reversals"], 0)

    def test_five_actors_arrive_and_hold_without_leaving_posts(self):
        for site, model in self.models.items():
            with self.subTest(site=site):
                result = formation_evaluation(model, get_scenario(site))
                self.assertTrue(result["passed"], result)
                self.assertEqual(result["leave_goal_decisions"], 0)

    def test_checkpoint_uses_trained_head_and_legacy_weights_still_load(self):
        scenario = get_scenario("L")
        for model in (self.models["L"], GuardDQN(scenario)):
            checkpoint = make_checkpoint(model, scenario, 10, OPPONENTS, START_MODES)
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            controller = ConconGuardController(checkpoint_bytes=buffer.getvalue())
            self.assertEqual(controller.model.navigation, model.navigation)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, controller.model.state_dict()[key]), key)

    def test_failed_placement_cannot_be_best_even_with_higher_win_rate(self):
        bad = dict(mean_win_rate=.9, min_team_win_rate=.8, positioning={"passed": False})
        good = dict(mean_win_rate=.5, min_team_win_rate=.3, positioning={"passed": True})
        self.assertFalse(qualifies_as_best(bad, None))
        self.assertFalse(qualifies_as_best(bad, good))
        self.assertTrue(qualifies_as_best(good, bad))

    def test_positioning_only_cannot_publish_untrained_combat_weights(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            train(positioning_only=True)

    def test_combat_updates_do_not_erase_frozen_positioning(self):
        scenario = get_scenario("L")
        model = self.models["L"]
        model.navigation_values.weight.requires_grad_(False)
        model.navigation_yield_values.weight.requires_grad_(False)
        before = model.navigation_values.weight.detach().clone()
        target = GuardDQN(scenario, navigation=True)
        target.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        env = GuardBattleEnv(model=model, map_name=scenario, opponents=["touyama_v2"])
        env.reset("transition", attacker_count=5, defender_count=5)
        replay = []
        while not env.done:
            transitions, _, _ = env.step(epsilon=.5)
            replay.extend(transitions)
        loss = optimize(model, target, optimizer, replay, batch_size=min(64, len(replay)))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.equal(before, model.navigation_values.weight))

    def test_quiet_positioning_is_preserved_during_full_exploration(self):
        env = GuardBattleEnv(model=self.models["L"], map_name="L", opponents=["touyama_v2"])
        env.reset("hold", attacker_count=5, defender_count=5)
        for char in env.attackers:
            char.smoke_charges = char.flash_charges = char.recon_charges = 0
            char.ultimate_points = 0
        env.step(epsilon=1.0)
        quiet = [(action, ctx) for action, ctx in env.decisions.values()
                 if not ctx["tap"] and not ctx["fireable"] and ctx["position"] == ctx["goal"]]
        self.assertTrue(quiet)
        self.assertTrue(all(action // 8 == 4 for action, _ in quiet))


class PositioningConfigurationTests(unittest.TestCase):
    def test_checkpoint_distinguishes_requested_and_completed_additional_episodes(self):
        scenario = get_scenario("L")
        model = GuardDQN(scenario)
        model.battle_training_run = {"start_episode": 2500, "requested_additional_episodes": 3000}
        foundation = make_checkpoint(model, scenario, 2500, OPPONENTS, START_MODES)
        finished = make_checkpoint(model, scenario, 5500, OPPONENTS, START_MODES)
        self.assertEqual(foundation["battle_episodes_this_run"], 0)
        self.assertEqual(finished["battle_episodes_this_run"], 3000)
        self.assertEqual(finished["battle_training_run"]["requested_additional_episodes"], 3000)

    def test_validated_resume_skips_updates(self):
        with patch("concon_v1.co1_guard_positioning.validate_positioning", return_value={"passed": True}), \
                patch("concon_v1.co1_guard_positioning.learn_positioning") as learn:
            _, info = prepare_positioning(Mock(), Mock(), steps=250, reuse=True)
            learn.assert_not_called()
            self.assertEqual(info["updates"], 0)
            self.assertEqual(info["mode"], "reused")

    def test_requested_update_count_is_used_and_recorded(self):
        model, scenario = Mock(), Mock()
        with patch("concon_v1.co1_guard_positioning.validate_positioning", return_value={"passed": True}), \
                patch("concon_v1.co1_guard_positioning.learn_positioning") as learn:
            _, info = prepare_positioning(model, scenario, steps=37)
            learn.assert_called_once_with(model, scenario, steps=37)
            self.assertEqual(info["updates"], 37)

    def test_explicit_retraining_and_failed_resume_trigger_updates(self):
        for force, validations in ((True, [{"passed": True}]),
                                   (False, [{"passed": False}, {"passed": True}])):
            with self.subTest(force=force), \
                    patch("concon_v1.co1_guard_positioning.validate_positioning", side_effect=validations), \
                    patch("concon_v1.co1_guard_positioning.learn_positioning") as learn:
                _, info = prepare_positioning(Mock(), Mock(), steps=50, reuse=True, force=force)
                self.assertEqual(learn.call_count, 1)
                self.assertEqual(info["updates"], 50)

    def test_zero_updates_requires_successful_resume_validation(self):
        with patch("concon_v1.co1_guard_positioning.validate_positioning", return_value={"passed": True}):
            _, info = prepare_positioning(Mock(), Mock(), steps=0, reuse=True)
            self.assertEqual(info["updates"], 0)
        with patch("concon_v1.co1_guard_positioning.validate_positioning", return_value={"passed": False}):
            with self.assertRaises(RuntimeError):
                prepare_positioning(Mock(), Mock(), steps=0, reuse=True)
        with self.assertRaises(ValueError):
            prepare_positioning(Mock(), Mock(), steps=-1)

    def test_failed_training_validation_stops_before_battle(self):
        with patch("concon_v1.co1_guard_positioning.validate_positioning", return_value={"passed": False}), \
                patch("concon_v1.co1_guard_positioning.learn_positioning"):
            with self.assertRaisesRegex(RuntimeError, "after 10 updates"):
                prepare_positioning(Mock(), Mock(), steps=10)


if __name__ == "__main__":
    unittest.main()
