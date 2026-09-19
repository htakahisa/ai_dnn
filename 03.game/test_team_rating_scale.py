import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_competition_manager import TeamRatingStore


class TeamRatingScaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "team_ratings.json"

    def test_new_parties_start_at_1500_and_updates_preserve_mean(self):
        store = TeamRatingStore(["A", "B"], self.path)
        self.assertEqual(store.get("A"), 1500)
        self.assertEqual(store.get("new party"), 1500)
        series = SimpleNamespace(
            team1="A", team2="B", winner="A", team1_wins=2, team2_wins=0
        )
        store.update_series(series, "test", "test")
        self.assertGreater(store.get("A"), 1500)
        self.assertAlmostEqual((store.get("A") + store.get("B")) / 2, 1500)

    def test_2000_is_overwhelming_against_average_party(self):
        self.assertGreater(TeamRatingStore.expected_score(2000, 1500), 0.94)
        self.assertLess(TeamRatingStore.expected_score(2000, 1500), 0.95)

    def test_legacy_ratings_and_history_migrate_once_with_backup(self):
        data = {
            "version": 1,
            "default_rating": 2500,
            "ratings": {"A": 2550, "B": 2450},
            "history": [
                {"type": "series", "before": {"A": 2500, "B": 2500},
                 "after": {"A": 2532, "B": 2468},
                 "delta": {"A": 32, "B": -32}, "expected": {"A": 0.5, "B": 0.5}},
                {"type": "manual", "team": "A", "before": 2532,
                 "after": 2550, "delta": 18},
            ],
        }
        self.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.path.read_bytes()
        store = TeamRatingStore(["A", "B", "C"], self.path)
        self.assertEqual(store.default_rating, 1500)
        self.assertEqual(store.get("A"), 1550)
        self.assertEqual(store.get("B"), 1450)
        self.assertEqual(store.get("C"), 1500)
        self.assertEqual(store.history[0]["before"], {"A": 1500, "B": 1500})
        self.assertEqual(store.history[0]["after"], {"A": 1532, "B": 1468})
        self.assertEqual(store.history[0]["delta"], data["history"][0]["delta"])
        self.assertEqual(store.history[0]["expected"], data["history"][0]["expected"])
        self.assertEqual(store.history[1]["before"], 1532)
        self.assertEqual(store.team_history("A")[-1]["rating"], 1550)
        self.assertEqual(self.path.with_suffix(".2500.bak").read_bytes(), original)
        reloaded = TeamRatingStore(["A", "B", "C"], self.path)
        self.assertEqual(reloaded.ratings, store.ratings)
        self.assertEqual(reloaded.history, store.history)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))["version"], 2)


if __name__ == "__main__":
    unittest.main()
