"""Success metrics and opponent-specific selection of frozen routes."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_attacker_selection import METRIC, load_selection, route_candidates, defending_opponent
from concon_v1.co1_attacker_controller import ConconRoundAttackerController
with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1.evaluate_co1_attacker import summarize_attack_success
    from concon_v1.evaluate_co1_attacker_selection import evaluate_selection


def report_fixture(rates):
    return dict(version=1, metric=METRIC, threshold=.8, results=[
        dict(opponent="gc_v1", map_name=name, model_sha256="hash", rounds=50,
             attack_success_rate=rate)
        for name, rate in zip(("A1", "A2", "A3", "A4"), rates)
    ])


class AttackerSelectionTests(unittest.TestCase):
    def test_metric_counts_plants_even_after_defuse_and_avoids_double_counting(self):
        result = summarize_attack_success([
            dict(planted=True, end_reason="defused"),
            dict(planted=True, end_reason="defender_eliminated"),
            dict(planted=False, end_reason="defender_eliminated"),
            dict(planted=False, end_reason="attacker_eliminated"),
            dict(planted=False, end_reason="time_expired"),
        ])
        self.assertEqual(result, dict(attack_successes=3, preplant_defender_eliminations=1, attack_success_rate=.6))

    def test_threshold_is_inclusive_and_fallback_uses_best_ties(self):
        names = ("A1", "A2", "A3", "A4")
        hashes = dict.fromkeys(names, "hash")
        self.assertEqual(route_candidates(report_fixture((.79, .8, .95, .1)), "gc_v1", names, hashes), ("A2", "A3"))
        self.assertEqual(route_candidates(report_fixture((.7, .7, .6, .1)), "gc_v1", names, hashes), ("A1", "A2"))

    def test_unknown_incomplete_or_updated_models_use_all_candidates(self):
        names = ("A1", "A2", "A3", "A4")
        hashes = dict.fromkeys(names, "hash")
        report = report_fixture((.79, .8, .95, .1))
        self.assertEqual(route_candidates(report, "unknown", names, hashes), names)
        self.assertEqual(route_candidates(None, "gc_v1", names, hashes), names)
        self.assertEqual(route_candidates(report, "gc_v1", names, {**hashes, "A1": "updated"}), names)
        report["results"].pop()
        self.assertEqual(route_candidates(report, "gc_v1", names, hashes), names)

    def test_invalid_reports_do_not_break_matches(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temporary) / "report.json"
            self.assertIsNone(load_selection(path))
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(load_selection(path))
            report = report_fixture((.5, .8, .9, .7))
            for field, value in (("threshold", float("nan")), ("metric", "plant_only"), ("results", {})):
                path.write_text(json.dumps({**report, field: value}), encoding="utf-8")
                self.assertIsNone(load_selection(path))
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(load_selection(path), report)

    def test_runtime_selects_once_per_round_and_rechecks_current_defender(self):
        data = b"fixed weights"
        report = report_fixture((.79, .8, .95, .1))
        for row in report["results"]:
            row["model_sha256"] = hashlib.sha256(data).hexdigest()
        with patch("concon_v1.co1_attacker_controller.load_selection", return_value=report), \
             patch("concon_v1.co1_attacker_controller.get_scenario",
                   return_value=SimpleNamespace(model_path=Mock(is_file=Mock(return_value=True), read_bytes=Mock(return_value=data)))), \
             patch("concon_v1.co1_attacker_controller.ConconAttackerController", side_effect=lambda **kw: Mock()) as factory:
            controller = ConconRoundAttackerController(seed=4)
        game = SimpleNamespace(current_defender_team_ai=SimpleNamespace(name="Ghost Champions v1"))
        controller.set_game(game)
        with patch.object(controller.rng, "choice", return_value="A2") as choice:
            controller.reset_round()
            for _ in range(5):
                controller.decide_move(Mock(), {})
            choice.assert_called_once_with(("A2", "A3"))
            game.current_defender_team_ai.name = "Unseen AI"
            controller.reset_round()
            choice.assert_called_with(("A1", "A2", "A3", "A4"))
        self.assertEqual(factory.call_count, 4)
        self.assertTrue(all(call.kwargs["checkpoint_bytes"] == data for call in factory.call_args_list))
        self.assertIsNone(defending_opponent(SimpleNamespace()))

    def test_evaluation_publishes_all_24_pairs_only_after_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "model.pt"
            model.write_bytes(b"frozen")
            output = directory / "selection.json"

            def evaluate(opponent, rounds, seed, frozen_checkpoint, map_name, on_trial):
                self.assertFalse(output.exists())
                self.assertEqual(frozen_checkpoint, b"frozen")
                return dict(map_name=map_name, opponent=opponent, model_sha256="hash", rounds=rounds,
                            attack_successes=rounds, attack_success_rate=1., plants=rounds,
                            preplant_defender_eliminations=0)

            with contextlib.redirect_stdout(io.StringIO()), \
                 patch("concon_v1.evaluate_co1_attacker_selection.get_scenario", return_value=SimpleNamespace(model_path=model)), \
                 patch("concon_v1.evaluate_co1_attacker_selection.torch.load", return_value={}), \
                 patch("concon_v1.evaluate_co1_attacker_selection.validate_checkpoint_scenario"), \
                 patch("concon_v1.evaluate_co1_attacker_selection.evaluate", side_effect=evaluate) as evaluator:
                report = evaluate_selection(rounds=36, output=output)
            self.assertEqual(evaluator.call_count, 24)
            self.assertEqual(len(report["results"]), 24)
            self.assertEqual(load_selection(output), report)
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_unwritable_destination_is_rejected_before_any_matches(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch("concon_v1.evaluate_co1_attacker_selection.Path.open", side_effect=PermissionError("unwritable")), \
             patch("concon_v1.evaluate_co1_attacker_selection.evaluate") as evaluator:
            with self.assertRaises(PermissionError):
                evaluate_selection(output=Path(temporary) / "selection.json")
            evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
