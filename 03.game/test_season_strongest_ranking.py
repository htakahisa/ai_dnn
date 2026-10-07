import json
from pathlib import Path
import tempfile
import unittest

from season_strongest_ranking import load_season_rankings, summarize_season_record
from season_strongest_ranking_ui import display_values, ranked_rows, sort_value
from strongest_ranking import aggregate_summaries


def record(seed=1, detailed=False):
    request = {"own": {"players": [{"name": "Leo", "role": "シーカー"}]},
               "opponent": {"players": [{"name": "Derke", "role": "フラッシュ"}]}}
    result = {"status": "completed", "own_team": "自チーム", "opponent_team": "相手",
              "own_score": 13, "opponent_score": 7, "seed": seed,
              "player_stats": {"Leo": {"kills": 20, "deaths": 10}, "Derke": {"kills": 10, "deaths": 20}}}
    if detailed:
        for stats in result["player_stats"].values():
            stats.update(assists=3, covers=2, one_v_one_won=4, one_v_one_lost=2)
    return request, result


class SeasonRankingTests(unittest.TestCase):
    def write_record(self, directory, seed=1, detailed=False):
        directory.mkdir(parents=True)
        request, result = record(seed, detailed)
        for name, data in (("request", request), ("result", result)):
            (directory / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
        return directory / "result.json"

    def test_scope_is_active_save_only_and_supports_series(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            active, other = base / "active", base / "other"
            self.write_record(active / "scrims" / "one")
            self.write_record(other / "scrims" / "one", 2)
            request, first = record(3, True)
            _, second = record(4, True)
            folder = active / "tournaments" / "cup" / "two"
            folder.mkdir(parents=True)
            (folder / "request.json").write_text(json.dumps(request), encoding="utf-8")
            (folder / "result.json").write_text(json.dumps({"status": "completed", "maps": [first, second]}), encoding="utf-8")
            result = load_season_rankings(active / "save.json")
            self.assertEqual(result["maps"], 3)
            self.assertEqual(result["errors"], [])
            leo = next(row for row in result["rows"] if row["name"] == "Leo")
            self.assertEqual((leo["kills"], leo["rounds"], leo["maps"], leo["mvps"]), (60, 60, 3, 3))
            self.assertEqual(leo["role"], "シーカー")
            self.assertEqual(leo["kd"], 2)
            self.assertIsNone(leo["covers"])

    def test_reload_reflects_updates_and_deletions_and_reports_errors(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            base = Path(temporary)
            path = self.write_record(base / "scrims" / "one")
            bad = base / "tournaments" / "cup" / "bad"
            bad.mkdir(parents=True)
            (bad / "result.json").write_text("{broken", encoding="utf-8")
            result = load_season_rankings(base / "save.json")
            self.assertEqual(result["maps"], 1)
            self.assertEqual(len(result["errors"]), 1)
            path.write_text(json.dumps({"status": "cancelled"}), encoding="utf-8")
            self.assertEqual(load_season_rankings(base / "save.json")["maps"], 0)
            path.unlink()
            self.assertEqual(load_season_rankings(base / "save.json")["rows"], [])

    def test_display_sort_filter_and_missing_fields(self):
        request, result = record(detailed=True)
        rows = aggregate_summaries(summarize_season_record(result, request, "test"))
        missing = dict(rows[0], name="missing", role="記録なし", kd=None, kills=None, covers=None, one_v_one_won=0, one_v_one_lost=0)
        rows.append(missing)
        for column in ("name", "team", "role", "maps", "kd", "kda", "kills_per_round", "covers", "mvps", "one_v_one"):
            for descending in (True, False):
                sorted_rows = ranked_rows(rows, column=column, descending=descending)
                values = [sort_value(row, column) for row in sorted_rows]
                known = [value for value in values if value is not None]
                self.assertEqual(values, sorted(known, reverse=descending) + [None] * (len(values) - len(known)))
        self.assertEqual(ranked_rows(rows)[-1]["name"], "missing")
        self.assertEqual(ranked_rows(rows, descending=False)[-1]["name"], "missing")
        self.assertEqual(len(ranked_rows(rows, role="フラッシュ")), 1)
        leo = next(row for row in rows if row["name"] == "Leo")
        self.assertEqual(display_values(leo, 1)[-1], "66.7%")
        self.assertEqual(display_values(missing, 3)[-1], "—")
        self.assertEqual(display_values(missing, 3)[5], "—")

    def test_round_roles_and_cancelled_series(self):
        request, result = record()
        result["round_records"] = [{"players": {"Leo": {"team": "自チーム", "role": "スモーカー"}}}]
        rows = aggregate_summaries(summarize_season_record(result, request, "test"))
        leo = next(row for row in rows if row["name"] == "Leo")
        self.assertEqual(leo["role"], "スモーカー")
        self.assertEqual(leo["rounds"], 1)
        self.assertEqual(summarize_season_record({"status": "cancelled", "maps": [result]}, request, "test"), [])


if __name__ == "__main__":
    unittest.main()
