"""Cash accounting, full timelines, transfer records and export recovery."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_teams as teams
import realtime_season_training as training
import realtime_season_world_levels as levels
from realtime_season import SeasonSaveError, SeasonStore, new_season, FORCED_OFFER_LOYALTY
from season_competitions import SeriesScore, next_match


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex")
CUP = dict(id="cup", name="Cup", start_date="2026-01-02", team_count=2,
           format="single_elimination", prizes={1: 1_000_000, 2: 500_000},
           normal_maps_to_win=1, grand_final_maps_to_win=1)


class SeasonHistoryTest(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        contexts = [patch.dict(character_stats.CHARACTER_TABLE, fixture),
                    patch.object(teams, "SEASON_TEAMS", [dict(name="Rival", players=list(RIVAL))]),
                    patch.object(calendar, "START_DATE", "2026-01-01"),
                    patch.object(calendar, "TOURNAMENTS", []),
                    patch.object(levels, "WORLD_LEVELS", [
                        {"レベル": 1, "上位%": 100, "敵倍率": 1, "スポンサー資金": 7_500_000},
                        {"レベル": 2, "上位%": 50, "敵倍率": 1.5, "スポンサー資金": 10_000_000}]),
                    patch.object(training, "TRAINING", {
                        title: {"上昇量": growth, "費用": [100_000 * level for level in range(1, 31)]}
                        for title, growth in (("研究", 5), ("エイムラボ", 1))})]
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "custom.json")

    def state(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        state = state.with_selected_team(state.teams[0].id)
        return replace(state, money=10_000_000, history=())._record_history("シーズン開始")

    def export(self, state):
        self.store.save(state)
        self.assertIsNone(self.store.history_export_error)
        return json.loads(self.store.history_path.read_text(encoding="utf-8"))

    def assert_balanced(self, entries):
        for previous, entry in zip(entries, entries[1:]):
            self.assertEqual(entry["資金"] - previous["資金"],
                             sum(i["金額"] for i in entry["収入"]) - sum(i["金額"] for i in entry["支出"]))

    def test_daily_metrics_monthly_details_training_and_repeated_saves(self):
        state = self.state().with_trained_player("Leo", "research").with_trained_player("Leo", "aim_lab")
        state = state.advance_days(29)
        data = self.export(state)
        self.assertEqual(self.store.history_path.name, "custom_history.json")
        days = [e["ゲーム内日付"] for e in data["履歴"] if e["種別"] == "日付進行"]
        self.assertEqual(len(days), 31)
        self.assertEqual((days[0], days[-1]), ("2026-01-02", "2026-02-01"))
        self.assertEqual(data["収入の内訳"], {"スポンサー収入": 7_500_000})
        self.assertEqual(data["支出の内訳"], {"研究": 100_000, "エイムラボ": 100_000, "給与": 500_000})
        settlement = next(e for e in data["履歴"] if e["種別"] == "月次決算")
        self.assertEqual({i["選手"] for i in settlement["支出"]}, set(OWN))
        self.assertEqual(data["履歴"][-1]["資金"], state.money)
        self.assertEqual(data["履歴"][-1]["月給総額"], state.monthly_payroll)
        self.assert_balanced(data["履歴"])
        self.assertEqual(self.export(state), data)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.history, state.history)
        self.assertEqual(self.export(loaded), data)

    def test_incoming_transfer_and_signing_bonus_are_separate_expenses(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        data = self.export(state)
        self.assertEqual(data["支出の内訳"], {"契約金": 300_000, "移籍金": 1_200_000})
        log = next(i for i in data["移籍ログ"] if i["選手"] == "Aspas")
        self.assertEqual((log["移籍元"], log["移籍先"]), ("Rival", state.team_name))
        self.assertEqual(data["履歴"][-1]["月給総額"], 600_000)
        self.assert_balanced(data["履歴"])
        renamed = state.with_team_name("New Name")
        self.assertEqual(renamed.history[-1]["移籍ログ"], [])

    def test_pending_rejected_forced_offers_and_transfer_income(self):
        state = self.state()
        state = replace(state, opponent_teams=tuple(replace(c, regular_members=c.members) for c in state.opponent_teams))
        state = state.with_scouted_player("Aspas", "year1").advance_months()
        offer = state.transfer_offer("Aspas")
        self.assertIsNotNone(offer)
        state = state.with_transfer_response(offer.id, False).with_team_loyalty("Aspas", FORCED_OFFER_LOYALTY - 1)
        data = self.export(state)
        kinds = {log["種別"] for log in data["移籍ログ"] if log["選手"] == "Aspas"}
        self.assertTrue({"オファー到着", "オファー拒否", "強制移籍成立", "移籍"}.issubset(kinds))
        self.assertEqual(data["収入の内訳"]["移籍金"], offer.fee)
        self.assertEqual(state.opponent_owner("Aspas").name, "Rival")
        self.assert_balanced(data["履歴"])

    def test_tournament_rank_prize_rating_and_world_level_transitions(self):
        with patch.object(calendar, "TOURNAMENTS", [CUP]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        self.assertIsNone(state.history[-1]["大会順位"][0]["順位"])
        state = state.advance_days()
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        state = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right, 1, 0))
        data = self.export(state)
        current = data["履歴"][-1]
        self.assertEqual(current["大会順位"][0]["順位"], 1)
        self.assertEqual(current["大会順位"][0]["順位表"][0], state.team_name)
        self.assertEqual(current["レート"], 1532)
        self.assertEqual(current["世界レベル"], 2)
        self.assertEqual(data["収入の内訳"], {"大会賞金": 1_000_000})
        self.assert_balanced(data["履歴"])

    def test_export_failure_does_not_replay_paid_action_and_restart_repairs_file(self):
        before = self.state()
        self.export(before)
        original = self.store.history_path.read_bytes()
        candidate = before.with_trained_player("Leo", "research")
        with patch("realtime_season.export_history", side_effect=OSError("locked history file")):
            self.store.save(candidate)
        self.assertIn("locked", self.store.history_export_error)
        self.assertEqual(self.store.history_path.read_bytes(), original)
        restarted = SeasonStore(self.store.path)
        loaded = restarted.load_or_create()
        self.assertEqual(loaded.history, candidate.history)
        self.assertEqual(loaded.money, before.money - 100_000)
        repaired = json.loads(restarted.history_path.read_text(encoding="utf-8"))
        self.assertEqual(repaired["支出の内訳"], {"研究": 100_000})
        self.assertEqual(sum(e["種別"] == "研究" for e in repaired["履歴"]), 1)
        self.assertIsNone(restarted.history_export_error)

    def test_legacy_save_starts_from_current_state_without_inventing_past_income(self):
        state = self.state().advance_days(31)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 17
        data.pop("history")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(len(loaded.history), 1)
        self.assertEqual(loaded.history[0]["資金"], state.money)
        self.assertEqual(self.store.path.read_bytes(), original)
        exported = json.loads(self.store.history_path.read_text(encoding="utf-8"))
        self.assertEqual((exported["収入累計"], exported["支出累計"]), (0, 0))

    def test_invalid_history_and_failed_save_cannot_change_export_or_saved_game(self):
        state = self.state()
        self.export(state)
        save_before, export_before = self.store.path.read_bytes(), self.store.history_path.read_bytes()
        bad = deepcopy(state.history)
        bad[0]["資金"] = "invalid"
        with self.assertRaises(SeasonSaveError):
            self.store.save(replace(state, history=bad))
        with patch("realtime_season.os.replace", side_effect=OSError("save failed")):
            with self.assertRaises(OSError):
                self.store.save(state.with_trained_player("Leo", "research"))
        self.assertEqual(self.store.path.read_bytes(), save_before)
        self.assertEqual(self.store.history_path.read_bytes(), export_before)


if __name__ == "__main__":
    unittest.main()
