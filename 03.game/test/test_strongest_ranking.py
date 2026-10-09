import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from strongest_ranking import aggregate_summaries, load_rankings, summarize_record


def sample_map(seed=1):
    return {
        "team1": "A", "team2": "B", "seed": seed, "score1": 13, "score2": 7,
        "mvp1": {"name": "same"}, "mvp2": {"name": "same"},
        "player_stats": [
            {"name": "same", "team": team, "role": role, "kills": kills,
             "deaths": deaths, "assists": 3, "covers": 2,
             "one_v_one_won": 4, "one_v_one_lost": 2}
            for team, role, kills, deaths in (("A", "フラッシュ", 10, 5), ("B", "スモーカー", 5, 10))
        ],
    }


class RankingTests(unittest.TestCase):
    def test_same_name_different_teams_and_weighted_averages(self):
        first, second = sample_map(), sample_map(2)
        second["score2"] = 2
        second["player_stats"][0].update(kills=20, deaths=10, role="シーカー")
        rows = aggregate_summaries(summarize_record({"maps": [first, second]}, "test"))
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual((a["maps"], a["rounds"], a["mvps"]), (2, 35, 2))
        self.assertEqual(a["kd"], 2)
        self.assertAlmostEqual(a["kills_per_round"], 30 / 35)
        self.assertEqual(a["role"], "フラッシュ")
        self.assertEqual((a["one_v_one_won"], a["one_v_one_lost"]), (8, 4))
        self.assertEqual(b["kills"], 10)

    def test_missing_fields_and_zero_deaths(self):
        game = sample_map()
        del game["player_stats"][0]["covers"]
        game["player_stats"][0]["deaths"] = 0
        row = aggregate_summaries(summarize_record({"maps": [game]}, "test"))[0]
        self.assertIsNone(row["covers"])
        self.assertEqual(row["kd"], float("inf"))

    def test_cache_deduplicates_exports_updates_and_removes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            originals = base / "series_data" / "match"
            originals.mkdir(parents=True)
            competitions = base / "competition_results"
            competitions.mkdir()
            original = originals / "match_original.json"
            exported = competitions / "league.json"
            first = {"maps": [sample_map()]}
            original.write_text(json.dumps(first), encoding="utf-8")
            exported.write_text(json.dumps({"rounds": [{"matches": [first]}]}), encoding="utf-8")
            result = load_rankings(base)
            self.assertEqual((result["maps"], result["read_files"]), (1, 2))
            self.assertFalse(result["ranking_cached"])
            with patch("strongest_ranking.summarize_record", side_effect=AssertionError("cached file parsed")), \
                    patch("strongest_ranking.aggregate_summaries", side_effect=AssertionError("cached ranking recalculated")):
                cached = load_rankings(base)
            self.assertTrue(cached["ranking_cached"])
            self.assertEqual(cached["rows"], result["rows"])
            self.assertEqual((cached["read_files"], cached["cached_files"]), (0, 2))
            second = sample_map(2)
            original.write_text(json.dumps({"maps": [sample_map(), second]}), encoding="utf-8")
            updated = load_rankings(base)
            self.assertFalse(updated["ranking_cached"])
            self.assertEqual((updated["maps"], updated["read_files"], updated["cached_files"]), (2, 1, 1))
            original.unlink()
            self.assertEqual(load_rankings(base)["maps"], 1)
            exported.write_text("{unfinished", encoding="utf-8")
            failed = load_rankings(base)
            self.assertEqual(failed["maps"], 0)
            self.assertEqual(len(failed["errors"]), 1)
            exported.write_text(json.dumps(first), encoding="utf-8")
            self.assertEqual(load_rankings(base)["maps"], 1)

    def test_final_ranking_cache_invalidates_on_addition_and_version_change(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertEqual(load_rankings(base)["maps"], 0)
            self.assertTrue(load_rankings(base)["ranking_cached"])
            competitions = base / "competition_results"
            competitions.mkdir()
            (competitions / "series.json").write_text(
                json.dumps({"maps": [sample_map()]}), encoding="utf-8")
            added = load_rankings(base)
            self.assertEqual(added["maps"], 1)
            self.assertFalse(added["ranking_cached"])
            self.assertTrue(load_rankings(base)["ranking_cached"])
            with patch("strongest_ranking.CACHE_VERSION", 2):
                updated = load_rankings(base)
            self.assertFalse(updated["ranking_cached"])
            self.assertEqual(updated["read_files"], 1)


class RankingUITests(unittest.TestCase):
    def test_filter_all_columns_sort_and_sort_preservation(self):
        import tkinter as tk
        from tkinter import ttk
        from run_competition_manager import CompetitionApp

        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = CompetitionApp.__new__(CompetitionApp)
        app.root = root
        app.notebook = ttk.Notebook(root)
        app.strongest_ranking_tab = tk.Frame(app.notebook)
        app.notebook.add(app.strongest_ranking_tab, text="最強ランキング")
        app.team_power_sort_states = {}
        app.team_power_sort_values = {}
        app.team_power_heading_labels = {}
        app._build_strongest_ranking_tab()
        app.strongest_ranking_loaded = True
        app.strongest_ranking_rows = aggregate_summaries(summarize_record({"maps": [sample_map()]}, "test"))
        missing = copy.deepcopy(app.strongest_ranking_rows[0])
        missing.update(name="missing", kd=None, covers=None, one_v_one_won=0, one_v_one_lost=0)
        app.strongest_ranking_rows.append(missing)
        app._render_strongest_ranking()
        tree = app.strongest_ranking_tree
        self.assertEqual(tree.heading("one_v_one", "text"), "1v1 (WinRate)")
        for item in tree.get_children():
            self.assertEqual(tree.set(item, "one_v_one"), "—" if tree.set(item, "name") == "missing" else "66.7%")
        for column in tree["columns"][1:]:
            for descending in (True, False):
                app._sort_team_power_rows(tree, column, descending)
                raw = app.team_power_sort_values[tree]
                values = [raw[item][column] for item in tree.get_children()]
                valid = [value for value in values if value is not None]
                self.assertEqual(values, sorted(valid, reverse=descending) + [None] * (len(values) - len(valid)))
        app._sort_team_power_rows(tree, "kd", True)
        app.strongest_role_var.set("スモーカー")
        app._render_strongest_ranking()
        self.assertEqual(len(tree.get_children()), 1)
        self.assertEqual(tree.set(tree.get_children()[0], "team"), "B")
        app.strongest_role_var.set("すべて")
        app._render_strongest_ranking()
        self.assertEqual(len(tree.get_children()), 3)
        self.assertEqual(tree.set(tree.get_children()[0], "team"), "A")
        self.assertEqual([int(tree.set(item, "no")) for item in tree.get_children()], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
