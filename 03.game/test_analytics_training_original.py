import io
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analytics import analysis_branching as branching
from analytics import train_analysis_models as training
from analytics.match_calculation import calculate_match_data_from_original


def original_match():
    return {
        "team1": "Alpha",
        "team2": "Bravo",
        "team1_score": 1,
        "team2_score": 0,
        "maps": [{
            "initial_attacker": "Alpha",
            "player_stats": [
                {"name": "a", "team": "Alpha", "kills": 2, "deaths": 1},
                {"name": "b", "team": "Bravo", "kills": 1, "deaths": 2},
            ],
            "round_records": [{
                "round_number": 1,
                "winner": "attacker",
                "reason": "detonated",
                "planted": True,
                "tactic": {"attacker_strategy": "split", "final_attack_site": "A"},
                "players": {},
            }],
            "assist_events": [{"assister": "a", "method": "flash"}],
        }],
    }


class StubClassifier:
    """Records model inputs without running XGBoost or any learning."""

    def __init__(self, **kwargs):
        self.options = kwargs

    def fit(self, matrix, labels):
        self.matrix = matrix
        self.labels = labels
        self.classes_ = sorted(set(labels))

    def predict_proba(self, matrix):
        return [[0.1, 0.9] for row in matrix]


class OriginalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(Path.cwd().resolve()))
        self.series = self.root / "series_data"
        folder = self.series / "sample"
        folder.mkdir(parents=True)
        self.original_path = folder / "sample_original.json"
        self.original_path.write_text(json.dumps(original_match()), encoding="utf-8")
        self.labels_path = self.root / "labels.jsonl"

    def write_labels(self, records):
        self.labels_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def test_rebuilds_from_original_and_ignores_stored_features(self):
        self.write_labels([
            {
                "match_path": "sample/sample_inference.json",
                "team_name": "Alpha",
                "features": {"map_aggregate": {"ability_assists": 999}},
                "labels": {"ability_usage": "good"},
            },
            {
                "match_path": "sample/sample_original.json",
                "team_name": "Bravo",
                "labels": {"micro_precision": "poor"},
            },
        ])
        records = training.load_records(self.labels_path, self.series)
        alpha = records[0]["features"]
        self.assertEqual(records[0]["match_path"], records[1]["match_path"])
        self.assertEqual(alpha["map_aggregate"]["ability_assists"], 1)
        self.assertEqual(alpha["map_aggregate"]["ability_data_available"], 1)
        self.assertEqual(alpha["map_aggregate"]["ability_use_data_available"], 0)
        self.assertEqual([p["name"] for p in alpha["player_stats"]], ["a"])
        self.assertEqual(records[1]["features"]["map_aggregate"]["ability_assists"], 0)
        row = branching._feature_vector(alpha, ["map_ability_uses", "map_ability_assists"])[0]
        self.assertTrue(math.isnan(row[0]))
        self.assertEqual(row[1], 1)
        self.assertEqual(json.loads(self.original_path.read_text(encoding="utf-8")), original_match())

    def test_missing_original_does_not_fall_back_to_embedded_features(self):
        self.write_labels([{
            "match_path": "sample/missing_inference.json",
            "team_name": "Alpha",
            "features": {"map_aggregate": {}},
            "labels": {"ability_usage": "good"},
        }])
        with self.assertRaisesRegex(ValueError, "original ファイルがありません"):
            training.load_records(self.labels_path, self.series)

    def test_training_skips_missing_labels_and_preserves_enum_mapping(self):
        self.write_labels([
            {
                "match_path": "sample/sample_original.json",
                "team_name": team,
                "labels": {"ability_usage": label},
            }
            for team, label in (("Alpha", "good"), ("Bravo", "poor"))
        ])
        rules = [r for r in branching.AI_ANALYSIS_RULES if r["variable"] in {
            "ability_usage", "micro_precision",
        }]
        args = SimpleNamespace(
            labels=self.labels_path, series_data=self.series,
            output=self.root / "stub.pkl", test_size=0.2, seed=42,
            n_estimators=1, max_depth=1, learning_rate=0.1, n_jobs=1,
        )
        with patch.dict("sys.modules", {"xgboost": SimpleNamespace(XGBClassifier=StubClassifier)}), patch.object(
            training, "AI_ANALYSIS_RULES", rules
        ), patch("sys.stdout", new_callable=io.StringIO):
            training.train(args)
        with args.output.open("rb") as file:
            bundle = pickle.load(file)
        self.assertEqual(set(bundle["models"]), {"ability_usage"})
        self.assertIn("micro_precision", bundle["skipped_models"])
        self.assertEqual(bundle["class_values"]["ability_usage"], ["good", "poor"])
        self.assertEqual(bundle["models"]["ability_usage"].labels, [0, 1])
        self.assertEqual(bundle["model_record_counts"]["ability_usage"]["train"], 2)


class AnalysisDataTests(unittest.TestCase):
    def test_map_player_metrics_fill_missing_round_fields_without_double_counting(self):
        original = original_match()
        original["maps"][0]["player_stats"][0].update(
            kills=99, preaim_angle_sum=60.0, preaim_angle_count=2
        )
        original["maps"][0]["round_records"][0]["players"] = {
            "a": {"team": "Alpha", "side": "attacker", "kills": 3, "deaths": 1}
        }
        aggregate = calculate_match_data_from_original(original)["map_aggregate"]["Alpha"]
        self.assertEqual(aggregate["kills"], 3)
        self.assertEqual(aggregate["preaim_error_mean"], 30)
        self.assertEqual(aggregate["preaim_data_available"], 1)
        self.assertEqual(aggregate["first_death_data_available"], 0)
        self.assertEqual(aggregate["duel_data_available"], 0)

    def test_replay_side_ids_missing_charges_and_round_resets(self):
        original = original_match()
        original["maps"][0].pop("assist_events")
        original["maps"][0]["replay_frames"] = [
            {"round": 1, "chars": [{"name": "a", "team": "A", "ability_charges": 2}]},
            {"round": 1, "chars": [{"name": "a", "team": "A", "ability_charges": 1}]},
            {"round": 1, "chars": [{"name": "a", "team": "A"}]},
            {"round": 2, "chars": [{"name": "a", "team": "D", "ability_charges": 0}]},
        ]
        result = calculate_match_data_from_original(original)
        alpha, bravo = (result["map_aggregate"][team] for team in ("Alpha", "Bravo"))
        self.assertEqual(alpha["ability_uses"], 1)
        self.assertEqual(alpha["ability_data_available"], 1)
        self.assertEqual(bravo["ability_data_available"], 0)

    def test_zero_assists_is_observed_but_absent_log_is_missing(self):
        original = original_match()
        original["maps"][0]["assist_events"] = []
        aggregate = calculate_match_data_from_original(original)["map_aggregate"]["Alpha"]
        self.assertEqual(aggregate["ability_assists"], 0)
        self.assertEqual(aggregate["ability_data_available"], 1)
        original["maps"][0].pop("assist_events")
        aggregate = calculate_match_data_from_original(original)["map_aggregate"]["Alpha"]
        self.assertEqual(aggregate["ability_data_available"], 0)

    def test_eligibility_is_specific_to_labels_and_data(self):
        rule = next(r for r in branching.AI_ANALYSIS_RULES if r["variable"] == "ability_usage")
        records = [
            {"labels": labels, "features": {"map_aggregate": {"ability_data_available": flag}}}
            for labels, flag in (({}, 1), ({"ability_usage": "good"}, 0), ({"ability_usage": "good"}, 1))
        ]
        self.assertEqual(training.eligible_record_indices(records, rule), [2])

    def test_holdout_keeps_matches_together_and_all_classes_in_training(self):
        records = [{"match_path": path} for path in ("a", "a", "b", "b", "c", "c")]
        labels = [0, 1, 0, 1, 2, 2]
        train, test = training.group_split(records, 0.5, 42, labels)
        self.assertEqual({labels[i] for i in train}, {0, 1, 2})
        self.assertTrue(test)
        self.assertFalse({records[i]["match_path"] for i in train} & {records[i]["match_path"] for i in test})
        self.assertEqual(training.group_split(records[:2], 0.5, 42, labels[:2]), ([0, 1], []))

    def test_partial_model_decodes_observed_enums_and_omits_missing_data(self):
        model = StubClassifier()
        model.classes_ = [0, 1]
        bundle = {
            "feature_names": [],
            "models": {"ability_usage": model, "micro_precision": model},
            "class_values": {"ability_usage": ["good", "poor"], "micro_precision": ["excellent", "good"]},
        }
        data = {"map_aggregate": {"ability_data_available": 0, "micro_data_available": 1}}
        with patch.object(branching, "MODEL_PATH") as model_path:
            model_path.exists.return_value = True
            model_path.open.return_value = io.BytesIO(pickle.dumps(bundle))
            results = branching.analyze_match_data(data, "Alpha")
        self.assertEqual([(r["variable"], r["value"]) for r in results], [("micro_precision", "good")])


if __name__ == "__main__":
    unittest.main()
