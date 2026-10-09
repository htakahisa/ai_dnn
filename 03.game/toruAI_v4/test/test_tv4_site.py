"""Check public-only inputs, chronology, decision timing and peek movement."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import replace
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch

from game_core import Character
from frc_v1.perception import FrcPerceptionBuilder, Sighting
from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.tv4_observer import FeatureHistory, ObserverController
from toruAI_v4.tv4_model import SiteModel, DecisionGate, optimize
from toruAI_v4.tv4_train_defender_analysis import summarize, preplant_counts, best_rank, evaluate_seeds, consider_best, parser, BEST_RULE, analysis_config
from toruAI_v4.tv4_model import VERSION


def world(scenario):
    class Game(Scenario):
        def _smoke_cells(self):
            return set()

        def _ramp_blocks_movement(self, char):
            return False
    game = Game()
    own = [Character(f"defender_{i}", "D", post.watch, "white", "green")
           for i, post in enumerate(scenario.posts)]
    enemies = [Character(f"enemy_{i}", "A", (22, 18 + i), "white", "red", has_spike=i == 0)
               for i in range(5)]
    game.chars = own + enemies
    game.defender_setup_phase = SimpleNamespace(active=False)
    game.current_round, game.battle_tick = 1, 0
    game.spike_pos = game.planted_pos = None
    game.is_planted = False
    game.round_timer, game.detonate_timer = 100, 55
    return game


class SitePredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario()
        torch.set_num_threads(1)

    def test_overlay_preserves_orb_and_posts_have_cover(self):
        self.assertEqual(self.scenario.grid[16, 23], 5)
        for post in self.scenario.posts[:3]:
            self.assertTrue(self.scenario.clear(post.watch, post.look))
            self.assertFalse(self.scenario.clear(post.retreat, post.look))

    def test_hidden_enemy_coordinates_and_carrier_do_not_change_inputs_or_actions(self):
        game = world(self.scenario)
        fields = FeatureHistory(self.scenario).fields
        model = SiteModel(len(fields))
        first = ObserverController(self.scenario, model)
        first.set_game(game)
        first.prepare_team_tick()
        features = first.frames[0]["features"].copy()
        actions = first.actions.copy()
        # Only hidden state changes; public observations remain identical.
        for i, enemy in enumerate(game.chars[5:]):
            enemy.pos = [4, 6 + i]
            enemy.has_spike = i == 3
        game.target_plant_pos = (7, 39)
        second = ObserverController(self.scenario, model)
        second.set_game(game)
        second.prepare_team_tick()
        np.testing.assert_array_equal(features, second.frames[0]["features"])
        self.assertEqual(actions, second.actions)

    def test_enemy_seen_by_team_is_counted_once_and_memory_ages(self):
        game = world(self.scenario)
        snapshot = FrcPerceptionBuilder("D").build(game)
        pos = self.scenario.branches["e"][0]
        snapshot = replace(snapshot, sightings=(Sighting(0, pos, "normal"),), tick=10)
        history = FeatureHistory(self.scenario)
        x = history.encode(snapshot, [])
        name = self.scenario.names[int(self.scenario.region[pos])]
        self.assertAlmostEqual(x[history.fields.index(f"branch_{name}_current_count")], .2)
        y = history.encode(replace(snapshot, sightings=(), tick=15), [])
        self.assertEqual(y[history.fields.index("enemy_0_current")], 0)
        self.assertEqual(y[history.fields.index("enemy_0_known")], 1)
        self.assertAlmostEqual(y[history.fields.index("enemy_0_age")], .05)

    def test_prior_rounds_and_round_number_are_input_and_do_not_reset_within_match(self):
        snapshot = FrcPerceptionBuilder("D").build(world(self.scenario))
        history = FeatureHistory(self.scenario)
        x = history.encode(replace(snapshot, round_number=2), [{"site": "L", "winner": "A"}])
        self.assertEqual(x[history.fields.index("round_2")], 1)
        self.assertEqual(x[history.fields.index("past_1_L")], 1)
        self.assertEqual(x[history.fields.index("past_2_known")], 0)
        y = history.encode(snapshot, [])
        self.assertEqual(y[history.fields.index("past_1_known")], 0)

    def test_public_dropped_spike_is_encoded_and_disappearance_retains_location(self):
        game = world(self.scenario)
        snapshot = FrcPerceptionBuilder("D").build(game)
        history = FeatureHistory(self.scenario)
        position = self.scenario.branches["e"][0]
        x = history.encode(replace(snapshot, tick=10, spike_dropped=position), [])
        self.assertEqual(x[history.fields.index("spike_on_ground")], 1)
        self.assertEqual(x[history.fields.index("spike_ever_dropped")], 1)
        self.assertGreater(x[history.fields.index("spike_left_distance")], 0)
        y = history.encode(replace(snapshot, tick=12, spike_dropped=None), [])
        self.assertEqual(y[history.fields.index("spike_on_ground")], 0)
        self.assertEqual(y[history.fields.index("spike_disappeared")], 1)
        self.assertEqual(y[history.fields.index("spike_last_row")], x[history.fields.index("spike_last_row")])
        self.assertAlmostEqual(y[history.fields.index("spike_last_seen_age")], .02)
        z = history.encode(replace(snapshot, tick=15, spike_dropped=None), [])
        self.assertAlmostEqual(z[history.fields.index("spike_disappeared_age")], .03)
        again = history.encode(replace(snapshot, tick=16, spike_dropped=position), [])
        self.assertEqual(again[history.fields.index("spike_disappeared")], 0)
        history.reset()
        empty = history.encode(snapshot, [])
        self.assertEqual(empty[history.fields.index("spike_ever_dropped")], 0)

    def test_dropped_spike_public_sensor_and_event_log(self):
        game = world(self.scenario)
        game.spike_pos = self.scenario.branches["c"][0]
        controller = ObserverController(self.scenario, SiteModel(len(FeatureHistory(self.scenario).fields)))
        controller.set_game(game)
        controller.prepare_team_tick()
        self.assertEqual(controller.frames[0]["spike_dropped"], game.spike_pos)
        self.assertEqual(controller.events[0]["type"], "spike_dropped")
        game.spike_pos, game.battle_tick = None, 1
        controller.prepare_team_tick()
        self.assertIn("spike_disappeared", [e["type"] for e in controller.events])

    def test_confirmation_and_last_change_time(self):
        gate = DecisionGate(.8, 3)
        for t in range(3):
            gate.update([.9, .1], t)
        self.assertEqual((gate.side, gate.tick), ("L", 2))
        gate.update([.9, .1], 3)
        self.assertEqual(gate.tick, 2)
        gate.update([.1, .9], 4)
        gate.update([.5, .5], 5)
        self.assertEqual(gate.side, "L")
        for t in (6, 7, 8):
            gate.update([.1, .9], t)
        self.assertEqual((gate.side, gate.tick, gate.changes), ("R", 8, 1))

    def test_best_prefers_correctness_over_lead_and_penalizes_abstention(self):
        old = dict(correct_all_plants=.8, accuracy=.8, mean_correct_lead=20.)
        fast_wrong = dict(correct_all_plants=.7, accuracy=.7, mean_correct_lead=80.)
        mostly_wait = dict(correct_all_plants=.2, accuracy=1., mean_correct_lead=90.)
        early_correct = {**old, "mean_correct_lead": 30.}
        self.assertLess(best_rank(fast_wrong), best_rank(old))
        self.assertLess(best_rank(mostly_wait), best_rank(old))
        self.assertGreater(best_rank(early_correct), best_rank(old))
        self.assertEqual(best_rank(dict(correct_all_plants=0., accuracy=None, mean_correct_lead=None)), (0., 0., -1.))

    def test_multi_seed_evaluation_pools_rounds_instead_of_seed_percentages(self):
        def result(site, correct):
            return dict(site=site, decision="L" if site else None, correct=correct,
                        correct_lead_ticks=20 if correct else None, defenders_alive=4,
                        defenders_at_plant=4 if site else None, changes=0)
        groups = [[result("L", True) for _ in range(12)],
                  [result("R", False)] + [result(None, None) for _ in range(11)],
                  [result("L", True) for _ in range(6)] + [result("R", False) for _ in range(6)]]
        log = Mock()
        args = parser().parse_args([])
        identity = dict(opponent="fnatic_v3", phase="eval", set=10, trained_rounds=120, replay_rounds=20)
        with patch("toruAI_v4.tv4_train_defender_analysis.play_block", side_effect=[(r, []) for r in groups]) as play:
            metrics = evaluate_seeds("fnatic_v3", self.scenario, Mock(), args, identity, [100, 200, 300], log)
        self.assertEqual(metrics["rounds"], 36)
        self.assertEqual(metrics["evaluation_seeds"], [100, 200, 300])
        self.assertAlmostEqual(metrics["accuracy"], 18 / 25)
        self.assertEqual([call.args[4] for call in play.call_args_list], [100, 200, 300])
        self.assertEqual(parser().parse_args([]).eval_every, 10)
        self.assertEqual(parser().parse_args([]).eval_seeds, 3)

    def test_legacy_best_is_compared_on_common_seeds_and_cached_without_mutation(self):
        args = parser().parse_args([])
        config = analysis_config(args)
        saved = dict(version=VERSION, opponent="fnatic_v3", scenario=self.scenario.signature,
                     fields=["x"], config=config, selection_rule="old_single_seed", evaluation_seed=100,
                     evaluation=dict(correct_all_plants=.1, accuracy=.1, mean_correct_lead=10))
        candidate = dict(phase="eval", plants=36, evaluation_seeds=[100, 200, 300],
                         correct_all_plants=.8, accuracy=.8, mean_correct_lead=30, trained_rounds=120)
        reference = {**candidate, "correct_all_plants": .9, "accuracy": .9}
        reevaluate = Mock(return_value=reference)
        path, logger, model = Mock(), Mock(), Mock()
        path.exists.return_value = True
        path.stat.return_value = SimpleNamespace(st_mtime_ns=1, st_size=2)
        cache = {}
        with patch("toruAI_v4.tv4_train_defender_analysis.torch.load", return_value=saved), patch("toruAI_v4.tv4_train_defender_analysis.torch.save") as save:
            for _ in range(2):
                self.assertFalse(consider_best(path, model, candidate, "fnatic_v3", self.scenario,
                                 ["x"], args, [100, 200, 300], logger, reevaluate=reevaluate, cache=cache))
            save.assert_not_called()
            improved = {**candidate, "correct_all_plants": .95, "accuracy": .95}
            self.assertTrue(consider_best(path, model, improved, "fnatic_v3", self.scenario,
                             ["x"], args, [100, 200, 300], logger, reevaluate=reevaluate, cache=cache))
            payload = save.call_args.args[0]
            self.assertEqual(payload["evaluation_seeds"], [100, 200, 300])
            self.assertEqual(payload["selection_rule"], BEST_RULE)
        reevaluate.assert_called_once_with(saved)
        self.assertEqual(saved["evaluation"]["accuracy"], .1)

    def test_analysis_output_defaults_are_fixed_directories(self):
        from toruAI_v4.tv4_train_defender_analysis import HERE
        args = parser().parse_args([])
        self.assertEqual(args.data_dir, HERE / "data" / "defender_analysis")
        self.assertEqual(args.log_dir, HERE / "logs" / "defender_analysis")

    def test_own_capabilities_change_features_without_player_name_dependence(self):
        snapshot = FrcPerceptionBuilder("D").build(world(self.scenario))
        history = FeatureHistory(self.scenario)
        original = history.encode(snapshot, [])
        renamed = replace(snapshot, allies=tuple(replace(a, name="renamed_" + str(a.slot)) for a in snapshot.allies))
        np.testing.assert_array_equal(original, FeatureHistory(self.scenario).encode(renamed, []))
        changed = replace(snapshot, allies=(replace(snapshot.allies[0], ability_name="ASH", accuracy=.123),) + snapshot.allies[1:])
        features = FeatureHistory(self.scenario).encode(changed, [])
        self.assertNotEqual(original[history.fields.index("ally_0_accuracy")], features[history.fields.index("ally_0_accuracy")])
        self.assertEqual(features[history.fields.index("ally_0_ability_ASH")], 1.)

    def test_new_conditions_replace_baseline_only_for_explicit_fresh_training(self):
        args = parser().parse_args([])
        saved = dict(version=VERSION, opponent="fnatic_v3", scenario=self.scenario.signature,
                     fields=["x"], config={"defender_preset": "Gorigons"})
        metrics = dict(phase="eval", plants=36, evaluation_seeds=[100, 200, 300],
                       correct_all_plants=.7, accuracy=.7, mean_correct_lead=30, trained_rounds=120)
        path = Mock()
        path.exists.return_value = True
        with patch("toruAI_v4.tv4_train_defender_analysis.torch.load", return_value=saved), \
             patch("toruAI_v4.tv4_train_defender_analysis.torch.save") as save:
            with self.assertRaisesRegex(ValueError, "conditions differ"):
                consider_best(path, Mock(), metrics, "fnatic_v3", self.scenario, ["x"], args,
                              [100, 200, 300], Mock())
            save.assert_not_called()
            self.assertTrue(consider_best(path, Mock(), metrics, "fnatic_v3", self.scenario, ["x"], args,
                                         [100, 200, 300], Mock(), allow_new_conditions=True))
            self.assertEqual(save.call_args.args[0]["completed_sets"], 10)

    def test_renamed_analysis_best_compares_existing_legacy_file(self):
        args = parser().parse_args([])
        config = analysis_config(args)
        metrics = dict(phase="eval", plants=36, evaluation_seeds=[100, 200, 300],
                       correct_all_plants=.9, accuracy=.9, mean_correct_lead=30, trained_rounds=120)
        saved = dict(version=VERSION, opponent="fnatic_v3", scenario=self.scenario.signature,
                     fields=["x"], config=config, selection_rule=BEST_RULE,
                     evaluation_seeds=[100, 200, 300], evaluation=metrics)
        path, legacy = Mock(), Mock()
        path.exists.return_value = False
        legacy.exists.return_value = True
        with patch("toruAI_v4.tv4_train_defender_analysis.torch.load", return_value=saved) as load, \
             patch("toruAI_v4.tv4_train_defender_analysis.torch.save") as save:
            self.assertFalse(consider_best(path, Mock(), {**metrics, "accuracy": .8, "correct_all_plants": .8},
                                          "fnatic_v3", self.scenario, ["x"], args, [100, 200, 300], Mock(), previous_path=legacy))
            self.assertEqual(load.call_args.args[0], legacy)
            save.assert_not_called()

    def test_scout_leaves_watch_post_for_cover_on_hide_cycle(self):
        game = world(self.scenario)
        game.battle_tick = 4  # slot zero is in the hidden half of the cycle.
        controller = ObserverController(self.scenario, SiteModel(len(FeatureHistory(self.scenario).fields)))
        controller.set_game(game)
        controller.prepare_team_tick()
        own = game.chars[0]
        destination, _ = controller.actions[own.name]
        self.assertNotEqual(tuple(destination), tuple(own.pos))
        self.assertEqual(abs(destination[0] - own.pos[0]) + abs(destination[1] - own.pos[1]), 1)
        self.assertIn({"tick": 4, "type": "hide", "role": "left_scout"}, controller.events)
        # The other calls within this tick must use the same frozen plan.
        own.pos = list(destination)
        controller.decide_move(game.chars[1], {})
        self.assertEqual(len(controller.frames), 1)

    def test_unjudged_and_no_plant_rounds_do_not_inflate_accuracy(self):
        def record(site, decision, correct, lead):
            return dict(site=site, decision=decision, correct=correct, correct_lead_ticks=lead,
                        defenders_alive=3, defenders_at_plant=4 if site else None, changes=0)
        metrics = summarize([record("L", "L", True, 20), record("R", "L", False, None),
                             record("L", None, None, None), record(None, "R", None, None)])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertEqual(metrics["mean_correct_lead"], 20)

    def test_rotation_keeps_outer_scouts_and_postplant_does_not_predict(self):
        game = world(self.scenario)
        model = SiteModel(len(FeatureHistory(self.scenario).fields))
        controller = ObserverController(self.scenario, model, rotate=True)
        controller.set_game(game)
        controller.gate.side, controller.gate.tick = "R", 0
        controller.prepare_team_tick()
        self.assertEqual(tuple(controller.actions[game.chars[0].name][0]), tuple(game.chars[0].pos))
        game.is_planted, game.planted_pos, game.battle_tick = True, (5, 40), 1
        controller.prepare_team_tick()
        self.assertEqual(len(controller.frames), 1)

    def test_survivor_and_preplant_kill_averages_use_the_correct_rounds(self):
        game = world(self.scenario)
        game.chars[0].is_alive = False
        game.chars[5].is_alive = False
        game.chars[6].is_alive = False
        game.chars[1].round_kills = 1
        counts = preplant_counts(game)
        self.assertEqual(counts, {"defenders": 4, "attackers": 3, "defender_kills": 1})
        base = dict(site="L", decision="L", correct=True, correct_lead_ticks=20, changes=0,
                    defenders_at_plant=4, attackers_at_plant=3, defenders_alive=2, attackers_alive=1,
                    preplant_defenders_alive=4, preplant_attackers_alive=3,
                    preplant_defender_losses=1, preplant_attackers_eliminated=2, preplant_defender_kills=1)
        no_plant = {**base, "site": None, "correct": None, "correct_lead_ticks": None,
                    "defenders_at_plant": None, "attackers_at_plant": None, "defenders_alive": 5,
                    "attackers_alive": 0, "preplant_defender_kills": 5, "preplant_defender_losses": 0}
        metrics = summarize([base, no_plant])
        self.assertEqual(metrics["mean_alive_at_plant"], 4)
        self.assertEqual(metrics["mean_enemy_alive_at_plant"], 3)
        self.assertEqual(metrics["mean_alive"], 3.5)
        self.assertEqual(metrics["mean_enemy_alive"], .5)
        self.assertEqual(metrics["mean_preplant_defender_kills"], 3)
        self.assertEqual(metrics["mean_preplant_defender_losses"], .5)

    def test_reset_round_clears_enemy_memory_but_preserves_previous_round_results(self):
        model = SiteModel(len(FeatureHistory(self.scenario).fields))
        controller = ObserverController(self.scenario, model)
        controller.previous_rounds = [{"site": "L", "winner": "A"}]
        controller.history.tracks[0] = ((14, 17), 5, (0., 0.))
        controller.reset_round()
        self.assertEqual(controller.history.tracks, {})
        self.assertEqual(controller.previous_rounds, [{"site": "L", "winner": "A"}])

    def test_learning_updates_weights_and_checkpoint_roundtrip(self):
        model = SiteModel(4)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        x = np.array([1, 0, 0, 0], dtype=np.float16)
        replay = [{"features": np.stack([x, x]), "label": 0}]
        before = model.probabilities(x)[0]
        loss = optimize(model, optimizer, replay, np.random.default_rng(1), 10, 8)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(model.probabilities(x)[0], before)
        buffer = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        copy = SiteModel(4)
        copy.load_state_dict(loaded["model"])
        np.testing.assert_array_equal(copy.probabilities(x), model.probabilities(x))


if __name__ == "__main__":
    unittest.main()
