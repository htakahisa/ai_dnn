"""Prior mixing, one-time reads and completed-report export; no training."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR.parent))
sys.path.insert(0, str(AI_DIR))

import ov1_attacker_carry_route_priors as priors
import ov1_learning_attacker_carry as carry
import ov1_summarize_attacker_carry_routes as exporter


class RoutePriorTests(unittest.TestCase):
    def setUp(self):
        self.waypoints = {"left": [(0, 0), (0, 1)], "right": [(0, 2), (0, 3)]}
        self.routes = [("left", 0), ("left", 1), ("right", 0), ("right", 1)]
        self.game = SimpleNamespace(current_defender_team_ai=SimpleNamespace(name="Fnatic v3"),
                                    defender_roster=["P"])
        self.cache = object.__new__(priors.RoutePriorCache)
        self.cache.opponents = {"fnatic_v3": {"players": ("P",), "win_rates": dict(zip(
            [priors.route_id(site, self.waypoints[site][index]) for site, index in self.routes],
            [.4, .3, .2, .1],
        ))}}

    def test_fifty_percent_mix_and_win_rate_normalization(self):
        expected = [.325, .275, .225, .175]
        for value, target in zip(self.cache.probabilities(self.game, self.routes, self.waypoints, .5), expected):
            self.assertAlmostEqual(value, target)
        self.cache.opponents["fnatic_v3"]["win_rates"] = {
            key: value * 2 for key, value in self.cache.opponents["fnatic_v3"]["win_rates"].items()}
        for value, target in zip(self.cache.probabilities(self.game, self.routes, self.waypoints, .5), expected):
            self.assertAlmostEqual(value, target)

    def test_new_route_has_neutral_probability_and_unknown_opponent_falls_back(self):
        self.waypoints["right"].append((0, 4))
        self.routes.append(("right", 2))
        probabilities = self.cache.probabilities(self.game, self.routes, self.waypoints, .5)
        self.assertAlmostEqual(probabilities[-1], .2)
        self.assertAlmostEqual(sum(probabilities), 1)
        self.game.current_defender_team_ai.name = "Unknown AI"
        self.assertIsNone(self.cache.probabilities(self.game, self.routes, self.waypoints, .5))

    def test_roster_mismatch_and_all_zero_rates(self):
        self.game.defender_roster = ["Other"]
        self.assertIsNone(self.cache.probabilities(self.game, self.routes, self.waypoints, .5))
        self.game.defender_roster = ["P"]
        self.cache.opponents["fnatic_v3"]["win_rates"] = {
            key: 0 for key in self.cache.opponents["fnatic_v3"]["win_rates"]}
        self.assertEqual(self.cache.probabilities(self.game, self.routes, self.waypoints, .5), [.25] * 4)

    def test_summary_is_read_once_and_model_map_changes_disable_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "omoko_v1/data/carry.pt"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"model")
            summary = root / "summary.json"
            data = {"schema_version": 1, "map_signature": priors.map_signature("00000"),
                    "wall_los_version": priors.WALL_LOS_VERSION,
                    "model_hashes": {"omoko_v1/data/carry.pt": hashlib.sha256(b"model").hexdigest()},
                    "opponents": self.cache.opponents}
            summary.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(priors, "PROJECT_DIR", root):
                cache = priors.RoutePriorCache(summary, "67000", model)
                self.assertEqual(cache.status, "loaded")
                with patch.object(Path, "open", side_effect=AssertionError("unexpected file read")):
                    for _ in range(100):
                        self.assertIsNotNone(cache.probabilities(self.game, self.routes, self.waypoints, .5))
                self.assertFalse(priors.RoutePriorCache(summary, "00100", model).opponents)
                model.write_bytes(b"changed model")
                self.assertFalse(priors.RoutePriorCache(summary, "00000", model).opponents)

    def test_only_warmup_uses_priors_and_later_rounds_keep_match_evidence(self):
        controller = object.__new__(carry.Ov1LearningAttackerCarryController)
        controller.waypoint_cells_by_site = self.waypoints
        controller._real_game = self.game
        controller._route_prior_cache = self.cache
        controller._priority_cells_by_site = {"left": [(1, 1)], "right": [(1, 2)]}
        controller._plant_cells_by_site = {}
        controller._route_stats = {("left", 1): {"wins": 5, "games": 5}}
        with patch.object(carry.random, "choices", return_value=[("right", 0)]) as choose:
            self.assertEqual(controller._choose_route(1), ("right", 0))
            for value, target in zip(choose.call_args.kwargs["weights"], [.325, .275, .225, .175]):
                self.assertAlmostEqual(value, target)
        with patch.object(carry.random, "random", return_value=1), \
                patch.object(carry.random, "choices", side_effect=AssertionError("warmup ended")):
            self.assertEqual(controller._choose_route(7), ("left", 1))


class SummaryTests(unittest.TestCase):
    def test_completed_report_exports_lightweight_rates_and_table(self):
        sources = {name: hashlib.sha256((exporter.PROJECT_DIR / name).read_bytes()).hexdigest()
                   for name in ("omoko_v1/ov1_map_data_carry.py", "map_data.py", "party_presets.py", "grid_lines.py")}
        report = {"schema_version": 2, "rounds_per_set": 12,
                  "evaluation": "12round_attack_set_full_engine_fixed_carry_route_v2",
                  "scenario": {"source_hashes": sources, "routes": [{"id": "left:4,11"}],
                               "wall_los_version": priors.WALL_LOS_VERSION}, "model_hashes": {},
                  "opponents": {"fnatic_v3": {"status": "complete", "roster": "Fnatic2023",
                      "target_rounds_per_route": 96, "routes": {"left:4,11": {"rounds": 96, "wins": 48}}}}}
        summary, table = exporter.summarize(report)
        self.assertEqual(summary["opponents"]["fnatic_v3"]["win_rates"], {"left:4,11": .5})
        self.assertIn("50.0%", table)
        self.assertNotIn("sets", json.dumps(summary))
        report["opponents"]["fnatic_v3"]["status"] = "running"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            exporter.summarize(report)


if __name__ == "__main__":
    unittest.main()
