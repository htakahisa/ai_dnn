import copy
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analytics import analysis_branching, match_output
from analytics.app import app
from analytics.match_calculation import calculate_match_data_from_original
from analytics.migrate_map_labels import migrate, migrate_records
from analytics.train_analysis_models import load_records, group_split
from test_analytics_training_original import original_match


def two_map_series():
    original = original_match()
    first = original["maps"][0]
    first.update(number=1, seed=11, score1=13, score2=4,
                 replay_frames=[{"round": 1, "tick": 0, "chars": [], "marker": "first-only"}])
    second = copy.deepcopy(first)
    second.update(number=2, seed=22,
                  replay_frames=[{"round": 1, "tick": 0, "chars": [], "marker": "second-only"}])
    second["player_stats"][0]["kills"] = 20
    second["assist_events"] = []
    second["round_records"][0]["winner"] = "defender"
    original["maps"].append(second)
    original["team1_score"] = 2
    return original


class MapScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assertTrue(self.root.resolve().is_relative_to(Path.cwd().resolve()))
        self.series = self.root / "series_data"
        directory = self.series / "sample"
        directory.mkdir(parents=True)
        self.path = "sample/sample_original.json"
        self.original_path = self.series / self.path
        self.original = two_map_series()
        self.original_path.write_text(json.dumps(self.original), encoding="utf-8")
        self.labels = self.root / "labels.jsonl"
        self.pending = self.root / "pending.jsonl"
        self.favorites = self.root / "favorites.json"
        for name, value in (
            ("SERIES_DATA_DIR", self.series), ("TRAINING_LABELS_PATH", self.labels),
            ("PENDING_LABELS_PATH", self.pending), ("FAVORITES_PATH", self.favorites),
        ):
            patcher = patch.object(match_output, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client = app.test_client()

    def write_records(self, records):
        self.labels.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    def test_stats_rounds_and_replay_are_isolated(self):
        with patch.object(match_output, "analyze_match_data", return_value=[]) as ai:
            response = self.client.get(f"/match/{self.path}/maps/1")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"second-only", response.data)
        self.assertNotIn(b"first-only", response.data)
        alpha = ai.call_args_list[0].args[0]
        self.assertEqual(alpha["map_aggregate"]["kills"], 20)
        self.assertEqual(alpha["map_aggregate"]["ability_assists"], 0)
        self.assertEqual(alpha["match_metadata"]["team1_score"], 13)
        self.assertEqual(alpha["match_metadata"]["map_index"], 1)
        self.assertEqual(len(alpha["round_features"]), 1)
        self.assertEqual(alpha["round_features"][0]["round_number"], 1)
        self.assertEqual(json.loads(self.original_path.read_text(encoding="utf-8")), self.original)

    def test_old_series_url_selects_maps_without_series_analysis(self):
        with patch.object(match_output, "analyze_match_data") as ai:
            response = self.client.get(f"/match/{self.path}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(f"/match/{self.path}/maps/0".encode(), response.data)
        self.assertIn(f"/match/{self.path}/maps/1".encode(), response.data)
        ai.assert_not_called()
        listing = match_output.get_all_matches()
        self.assertEqual([r["map_index"] for r in listing], [0, 1])

    def test_saving_map_labels_preserves_existing_labels_and_other_map(self):
        self.write_records([
            {"match_path": self.path, "map_index": 0, "team_name": "Alpha", "labels": {"ability_usage": "poor"}},
            {"match_path": self.path, "map_index": 1, "team_name": "Alpha", "labels": {"micro_precision": "excellent"}},
        ])
        response = self.client.post(f"/match/{self.path}/maps/1/label", data={
            "Alpha__ability_usage__present": "1", "Alpha__ability_usage": "good",
            "Alpha__first_deaths_issue__present": "1",
        })
        self.assertEqual(response.status_code, 302)
        records = [json.loads(line) for line in self.labels.read_text(encoding="utf-8").splitlines()]
        saved = records[-2]
        self.assertEqual(saved["map_index"], 1)
        self.assertEqual(saved["scope"], "map")
        self.assertEqual(saved["labels"], {
            "micro_precision": "excellent", "ability_usage": "good", "first_deaths_issue": False,
        })
        self.assertEqual(saved["features"]["map_aggregate"]["kills"], 20)
        self.assertEqual(records[0]["labels"]["ability_usage"], "poor")
        self.assertEqual(self.client.post(f"/match/{self.path}/label").status_code, 400)

    def test_loader_does_not_spread_series_labels_across_maps(self):
        self.write_records([
            {"match_path": self.path, "team_name": "Alpha", "labels": {"ability_usage": "poor"}},
            {"match_path": self.path, "map_index": 1, "team_name": "Alpha", "labels": {"ability_usage": "good"}},
        ])
        with patch("sys.stdout", new_callable=io.StringIO):
            records = load_records(self.labels, self.series)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["map_index"], 1)
        self.assertEqual(records[0]["features"]["map_aggregate"]["kills"], 20)

    def test_migration_backs_up_labels_and_preserves_ambiguous_series(self):
        single = copy.deepcopy(self.original)
        single["maps"] = single["maps"][:1]
        (self.series / "sample" / "single_original.json").write_text(json.dumps(single), encoding="utf-8")
        rows = [
            {"match_path": "sample/single_inference.json", "team_name": "Alpha", "labels": {"ability_usage": "good"}},
            {"match_path": self.path, "team_name": "Alpha", "labels": {"ability_usage": "poor"}},
        ]
        self.write_records(rows)
        original_bytes = self.labels.read_bytes()
        with patch("sys.stdout", new_callable=io.StringIO):
            active, pending = migrate(self.labels, self.series, self.pending, apply=True)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["map_index"], 0)
        self.assertEqual(active[0]["features"]["match_metadata"]["team1_score"], 13)
        self.assertEqual(pending[0]["labels"], rows[1]["labels"])
        self.assertNotIn("map_index", pending[0])
        backups = list(self.root.glob("labels.series_backup_*.jsonl"))
        self.assertEqual(backups[0].read_bytes(), original_bytes)
        with patch("sys.stdout", new_callable=io.StringIO):
            migrate(self.labels, self.series, self.pending, apply=True)
        self.assertEqual(len(list(self.root.glob("labels.series_backup_*.jsonl"))), 1)

    def test_replay_recovery_uses_map_identity_even_with_equal_scores(self):
        source = copy.deepcopy(self.original)
        for data in source["maps"]:
            data.pop("replay_frames")
        results = self.root / "competition_results"
        results.mkdir()
        result_path = results / "series_Alpha_vs_Bravo_2-0_20260927.json"
        result_path.write_text(json.dumps(self.original), encoding="utf-8")
        with patch.object(match_output, "COMPETITION_RESULTS_DIR", results):
            frames = match_output._load_replay_frames(source, 1)
        self.assertEqual(frames[0]["marker"], "second-only")

    def test_favorites_are_per_map_and_old_series_favorites_are_preserved(self):
        self.favorites.write_text(json.dumps([self.path]), encoding="utf-8")
        response = self.client.post(f"/match/{self.path}/maps/1/favorite")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(match_output.get_favorites(), [self.path + "#map=0"])

    def test_series_models_are_not_applied_to_map_features(self):
        data = calculate_match_data_from_original(self.original, 0)
        with patch.object(analysis_branching, "MODEL_PATH") as model_path:
            model_path.exists.return_value = True
            model_path.open.return_value = io.BytesIO(pickle.dumps({"version": 2}))
            with self.assertRaisesRegex(analysis_branching.AnalysisUnavailableError, "再学習"):
                analysis_branching.analyze_match_data(data, "Alpha")

    def test_invalid_map_or_path_is_rejected(self):
        self.assertEqual(self.client.get(f"/match/{self.path}/maps/9").status_code, 400)
        self.assertEqual(self.client.get("/match/../outside_original.json/maps/0").status_code, 400)

    def test_same_series_maps_remain_in_same_holdout_group(self):
        records = [
            {"match_path": path, "map_index": index}
            for path in ("a_original.json", "b_original.json") for index in (0, 1)
        ]
        train, test = group_split(records, 0.5, 42, [0, 1, 0, 1])
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        self.assertFalse({records[i]["match_path"] for i in train} & {records[i]["match_path"] for i in test})


if __name__ == "__main__":
    unittest.main()
