"""Performance salaries, static compatibility, month boundaries and startup UI."""

from dataclasses import replace
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import patch

from season import season_salary_config as config
import realtime_season_world_levels
from realtime_season import SalaryMode, SeasonSaveError, new_season
from season.season_salary import (CompetitionTotals, SalarySettings, calculate_salaries, clamp_change,
                           invalidate_salary_cache, rounded_salary, salary_from_percentile,
                           salary_records, smoothed_kd, _read_result)
from test_season_competitions import (CompetitionScreenTest, SeasonCompetitionTest, OWN,
                                      config as competition_config, definition)


class SalaryRulesTest(unittest.TestCase):
    def settings(self, **kwargs):
        return SalarySettings("unused", **kwargs)

    def test_reference_interpolation_and_half_up_rounding(self):
        for p, value, rounded in ((0, 90, 90), (.25, 116.189500, 116), (.5, 150, 150),
                                  (.75, 212.132034, 212), (1, 300, 300)):
            with self.subTest(p=p):
                self.assertAlmostEqual(salary_from_percentile(p), value, places=5)
                self.assertEqual(rounded_salary(p), rounded * 10_000)
        self.assertEqual(salary_from_percentile(-1), 90)
        self.assertEqual(salary_from_percentile(2), 300)
        with patch("season.season_salary.salary_from_percentile", return_value=116.5):
            self.assertEqual(rounded_salary(.25), 1_170_000)

    def test_smoothing_reference_and_zero_deaths(self):
        self.assertEqual(smoothed_kd(10, 2), 1.25)
        self.assertEqual(smoothed_kd(0, 0), 1)
        self.assertEqual(smoothed_kd(30, 0), 2)

    def test_games_threshold_and_single_eligible_player(self):
        totals = (CompetitionTotals("A", 100, 0, 9), CompetitionTotals("B", 10, 2, 10))
        a, b = calculate_salaries({"A": 123_457, "B": 100_000}, totals, self.settings())
        self.assertEqual((a.monthly_salary, a.percentile), (123_457, None))
        self.assertEqual((b.monthly_salary, b.kd, b.percentile), (1_500_000, 1.25, .5))

    def test_equal_kd_uses_average_rank(self):
        totals = tuple(CompetitionTotals(name, kills, deaths, 10)
                       for name, kills, deaths in (("A", 0, 100), ("B", 20, 20), ("C", 40, 40), ("D", 200, 10)))
        records = calculate_salaries({name: 100_000 for name in "ABCD"}, totals, self.settings())
        self.assertEqual([r.percentile for r in records], [0, .5, .5, 1])
        self.assertEqual([r.monthly_salary for r in records], [900_000, 1_500_000, 1_500_000, 3_000_000])
        equal = calculate_salaries({name: 100_000 for name in "ABCD"},
                                  tuple(CompetitionTotals(name, 0, 0, 10) for name in "ABCD"), self.settings())
        self.assertTrue(all(r.percentile == .5 and r.monthly_salary == 1_500_000 for r in equal))

    def test_fixed_salary_overrides_insufficient_games_and_change_limit(self):
        record, = calculate_salaries({"A": 100_000}, (), self.settings(fixed_salaries=(("A", 10_000_000),)), {"A": 100_000})
        self.assertEqual((record.monthly_salary, record.fixed, record.percentile), (10_000_000, True, None))

    def test_change_limit_both_directions_and_initial_unlimited(self):
        self.assertEqual(clamp_change(4_000_000, None), 4_000_000)
        self.assertEqual(clamp_change(4_000_000, 1_000_000), 1_200_000)
        self.assertEqual(clamp_change(110_000, 1_000_000), 800_000)
        self.assertEqual(clamp_change(1_500_000, 1_000_000, .1), 1_100_000)
        self.assertEqual(clamp_change(999, 123_457, 0), 123_457)
        self.assertEqual(clamp_change(0, 123_457), 98_766)
        self.assertEqual(clamp_change(9_999_999, 123_457), 148_148)

    def test_invalid_config_is_rejected(self):
        for kwargs in ({"a": 0}, {"a": float("nan")}, {"min_games": True}, {"min_games": 0},
                       {"max_change": 1.1}, {"fixed_salaries": (("A", -1),)},
                       {"fixed_salaries": (("A", 1), ("A", 2))}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.settings(**kwargs).validate()


class SalaryStateTest(SeasonCompetitionTest):
    def setUp(self):
        super().setUp()
        self.results = self.path.parent / "results"
        self.results.mkdir()
        self.configure_salary()
        self.write_stats()

    def configure_salary(self):
        context = patch.object(realtime_season_world_levels, "WORLD_LEVELS",
                               [{"レベル": 1, "必要レート": 0, "敵倍率": 1, "スポンサー資金": 7_500_000}])
        context.start()
        self.addCleanup(context.stop)
        for key, value in (("COMPETITION_RESULTS_DIR", self.results), ("A", 30), ("MIN_GAMES", 10),
                           ("MAX_MONTHLY_CHANGE", .2), ("FIXED_MONTHLY_SALARIES", {}), ("EXCLUDED_RESULT_FILES", ())):
            context = patch.object(config, key, value)
            context.start()
            self.addCleanup(context.stop)
        logger = logging.getLogger("season.season_salary")
        before = logger.level
        logger.setLevel(logging.CRITICAL)
        self.addCleanup(logger.setLevel, before)
        invalidate_salary_cache()
        self.addCleanup(invalidate_salary_cache)

    def write_stats(self, **changes):
        rows = [{"player": "Leo", "team": "A", "kills": 0, "deaths": 100, "maps": 10},
                {"player": "Boaster", "team": "A", "kills": 50, "deaths": 50, "maps": 10},
                {"player": "Derke", "team": "A", "kills": 200, "deaths": 10, "maps": 10}]
        for row in rows:
            row.update(changes.get(row["player"], {}))
        (self.results / "series.json").write_text(json.dumps({"mode": "series", "player_leaderboards": {"all_players": rows}}), encoding="utf-8")

    def dynamic_state(self):
        return new_season(salary_mode=SalaryMode.KD_DYNAMIC).with_initial_selection(OWN)

    def test_static_mode_matches_original_even_with_broken_result_data(self):
        (self.results / "series.json").write_text("broken", encoding="utf-8")
        with patch("season.season_salary._file_signature", side_effect=AssertionError("STATIC must not read results")):
            state = new_season().with_initial_selection(OWN)
            self.assertEqual(state.salary_mode, SalaryMode.STATIC)
            self.assertTrue(all(p.monthly_salary == 100_000 for p in state.owned_players))
            self.assertTrue(all(c.monthly_salary == 100_000 for c in state.contracts))
            self.assertEqual(state.contract_terms(state.player("Leo"), "year2").monthly_salary, 90_000)
            self.assertEqual(state.contract_terms(state.player("Leo"), "year3").monthly_salary, 80_000)
            self.assertEqual(state.advance_months().money, state.money + 7_500_000 - 500_000)
            self.store.save(state)
            self.assertEqual(self.store.load_or_create(), state)

    def test_initial_calculation_without_clamp_and_mode_locked_after_start(self):
        state = self.dynamic_state()
        self.assertEqual([state.player(name).monthly_salary for name in OWN], [900_000, 1_500_000, 3_000_000, 100_000, 100_000])
        self.assertEqual(state.monthly_payroll, 5_600_000)
        from realtime_season import INITIAL_MONEY
        self.assertEqual(state.money, INITIAL_MONEY)
        for mode in SalaryMode:
            with self.assertRaisesRegex(SeasonSaveError, "開始前"):
                state.with_salary_mode(mode)
        self.store.save(state)
        with patch.object(config, "A", 99), patch.object(config, "FIXED_MONTHLY_SALARIES", {"Leo": 99_000_000}):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(loaded.salary_settings.a, 30)

    def test_result_changes_invalidate_cache_but_wages_wait_for_month_and_old_wages_are_paid(self):
        state = self.dynamic_state()
        self.write_stats(Leo={"kills": 1000, "deaths": 0}, Derke={"kills": 0, "deaths": 200})
        settings = state.salary_settings
        fresh = salary_records({"Leo": 100_000}, settings)
        self.assertEqual(fresh[0].monthly_salary, 3_000_000)
        january = state.advance_days(30)
        self.assertEqual(january.contract("Leo").monthly_salary, 900_000)
        with self.assertLogs("season.season_salary", level="INFO") as logs:
            february = january.advance_days()
        self.assertEqual(february.money, state.money + 7_500_000 - 5_600_000)
        self.assertEqual((february.player("Leo").monthly_salary, february.contract("Leo").monthly_salary), (1_080_000, 1_080_000))
        self.assertEqual(february.contract("Derke").monthly_salary, 2_400_000)
        self.assertTrue(any("Leo" in row and "900000 -> 1080000" in row and "kd=" in row and "p=" in row for row in logs.output))
        self.store.save(february)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.with_updated_salaries(), loaded)
        self.assertEqual(loaded.advance_days().contract("Leo"), february.contract("Leo"))
        self.assertEqual(state.advance_months(2), february.advance_months())

    def test_contract_discounts_update_monthly_and_fixed_contracts_override_discount(self):
        state = self.dynamic_state()
        contracts = tuple(replace(c, kind="year2", duration_months=24, monthly_salary=810_000)
                          if c.player_name == "Leo" else c for c in state.contracts)
        state = replace(state, contracts=contracts)
        self.assertEqual(state.contract_terms(state.player("Leo"), "year2").monthly_salary, 810_000)
        self.write_stats(Leo={"kills": 1000, "deaths": 0})
        next_month = state.advance_months()
        self.assertEqual(next_month.contract("Leo").monthly_salary, 972_000)
        with patch.object(config, "FIXED_MONTHLY_SALARIES", {"Meiy": 10_000_000}), \
                patch("realtime_season.with_randomized_clubs", side_effect=lambda state: state):
            state = replace(self.dynamic_state(), money=500_000_000)
        terms = state.contract_terms(next(p for p in state.scout_players if p.name == "Meiy"), "year3")
        self.assertEqual(terms.monthly_salary, 10_000_000)
        signed = state.with_scouted_player("Meiy", "year3")
        self.assertEqual((signed.contract("Meiy").monthly_salary, signed.money), (10_000_000, 470_000_000))
        self.assertEqual(signed.advance_months().contract("Meiy").monthly_salary, 10_000_000)

    def test_npc_contracts_and_lft_use_dynamic_salary_and_transfer_fee(self):
        # Salary assertions require a known NPC/LFT assignment.
        with patch.object(config, "FIXED_MONTHLY_SALARIES", {"Aspas": 1_000_000, "Meiy": 2_000_000}), \
                patch("realtime_season.with_randomized_clubs", side_effect=lambda state: state):
            state = self.dynamic_state()
        rival = next(c for c in state.opponent_teams if "Aspas" in c.members)
        self.assertEqual(next(c.monthly_salary for c in rival.contracts if c.player_name == "Aspas"), 1_000_000)
        self.assertEqual(state.transfer_fee("Aspas"), 12_000_000)
        self.assertEqual(next(p.monthly_salary for p in state.lft_players if p.name == "Meiy"), 2_000_000)
        stripped = replace(rival, players=rival.players[1:], contracts=tuple(
            replace(c, end_reason="released") if c.player_name == "Aspas" else c for c in rival.contracts), igl=None, carrier=None)
        state = replace(state, opponent_teams=tuple(stripped if c.id == rival.id else c for c in state.opponent_teams))
        following = state.advance_months()
        for club in following.opponent_teams:
            for p in club.players:
                self.assertEqual(p.monthly_salary, following.salary_player(p).monthly_salary)
                contract = next(c for c in club.contracts if c.player_name == p.name)
                if following.contract_active(contract):
                    self.assertEqual(contract.monthly_salary, p.monthly_salary)

    def test_sum_by_name_across_teams_and_legacy_map_stats_without_double_counting(self):
        other = {"mode": "series", "maps": [{"player_stats": [
            {"name": "Leo", "team": "Other", "kills": 30, "deaths": 20},
            {"name": "Leo", "team": "Clone", "kills": 10, "deaths": 0}]}]}
        (self.results / "older.json").write_text(json.dumps(other), encoding="utf-8")
        (self.results / "team_ratings.json").write_text("irrelevant", encoding="utf-8")
        state = self.dynamic_state()
        leo = next(r for r in state.salary_records if r.name == "Leo")
        self.assertEqual((leo.kills, leo.deaths, leo.games), (40, 120, 12))
        # The all_players totals are authoritative, even if maps are also present.
        data = json.loads((self.results / "series.json").read_text(encoding="utf-8"))
        data["maps"] = other["maps"]
        (self.results / "series.json").write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(next(r for r in self.dynamic_state().salary_records if r.name == "Leo").games, 12)

    def test_old_save_defaults_static_and_malformed_data_never_overwrites_save(self):
        state = self.state()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 13
        for key in ("salary_mode", "salary_settings", "salary_records", "salary_updated_month"):
            data.pop(key)
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        self.assertEqual(self.store.load_or_create(), state)
        self.assertEqual(self.path.read_bytes(), before)
        (self.results / "series.json").write_text("broken", encoding="utf-8")
        with self.assertRaises(SeasonSaveError):
            new_season().with_salary_mode(SalaryMode.KD_DYNAMIC)
        self.assertEqual(self.path.read_bytes(), before)

    def test_competition_save_explicitly_invalidates_salary_cache(self):
        from run_competition_manager import save_json
        with patch("run_competition_manager.RESULT_DIR", self.results), patch("season.season_salary.invalidate_salary_cache") as invalidate:
            save_json("new", {"mode": "series", "maps": []})
        invalidate.assert_called_once_with()

    def test_current_large_results_read_only_leaderboard_and_legacy_compact_fallback(self):
        data = json.loads((self.results / "series.json").read_text(encoding="utf-8"))
        data["unrelated_logs"] = "x" * 300_000
        # Preserve the manager's ordering: the leaderboard follows the logs.
        data = {"mode": data["mode"], "maps": [], "logs": data["unrelated_logs"],
                "player_leaderboards": data["player_leaderboards"], "rating_updates": []}
        path = self.results / "large.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.assertEqual(_read_result(path), {"player_leaderboards": data["player_leaderboards"], "rating_updates": []})
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(_read_result(path), data)

    def test_evaluation_exclusion_and_empty_sources(self):
        (self.results / "evaluation.json").write_text(json.dumps({"evaluation": "full_engine_gc_matchup", "maps": 20}), encoding="utf-8")
        bad = self.results / "incomplete.json"
        bad.write_text("broken", encoding="utf-8")
        with patch.object(config, "EXCLUDED_RESULT_FILES", (bad.name,)):
            state = self.dynamic_state()
        self.assertEqual(state.salary_settings.excluded_files, (bad.name,))
        self.assertEqual(next(r.games for r in state.salary_records if r.name == "Leo"), 10)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        for path in (self.results / "series.json", bad):
            path.unlink()
        empty = self.dynamic_state()
        self.assertTrue(all(r.monthly_salary == r.static_salary and r.percentile is None for r in empty.salary_records))
        with patch.object(config, "COMPETITION_RESULTS_DIR", self.results / "missing"):
            self.assertTrue(all(r.monthly_salary == r.static_salary for r in self.dynamic_state().salary_records))

    def test_round_robin_and_bracket_legacy_stats_and_new_files_invalidate_cache(self):
        state = self.dynamic_state()
        legacy = {"matches": [{"maps": [{"player_stats": [{"name": "Leo", "kills": 3, "deaths": 2}]}]}],
                  "bracket_matches": [{"maps": [{"player_stats": [{"name": "Leo", "kills": 1, "deaths": 4}]}]}]}
        path = self.results / "new_legacy.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        records = salary_records({"Leo": 100_000}, state.salary_settings)
        self.assertEqual((records[0].kills, records[0].deaths, records[0].games), (4, 106, 12))
        path.unlink()
        self.assertEqual(salary_records({"Leo": 100_000}, state.salary_settings)[0].games, 10)

    def test_invalid_saved_salary_fields_are_rejected_and_preserved(self):
        self.store.save(self.dynamic_state())
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in (("salary_mode", "UNKNOWN"), ("salary_records", []), ("salary_updated_month", -1)):
            invalid = {**data, key: value}
            self.path.write_text(json.dumps(invalid), encoding="utf-8")
            before = self.path.read_bytes()
            with self.subTest(key=key), self.assertRaises(SeasonSaveError):
                self.store.load_or_create()
            self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_monthly_results_leave_source_state_and_save_unchanged(self):
        state = self.dynamic_state()
        self.store.save(state)
        before = self.path.read_bytes()
        self.write_stats(Leo={"kills": -1})
        with self.assertRaises(SeasonSaveError):
            state.advance_months()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.load_or_create(), state)

    def test_skipped_npc_series_updates_dynamic_contracts_on_month_boundary_once(self):
        with patch.object(competition_config, "TOURNAMENTS", [definition(
                start_date="2026-01-31", format="single_elimination", team_count=3,
                prizes={1: 5_000_000, 2: 2_000_000, 3: 1_000_000}, allow_player_entry=False)]):
            state = self.dynamic_state().with_tournament_entry("cup").advance_days(30)
        self.write_stats(Leo={"kills": 1000, "deaths": 0}, Derke={"kills": 0, "deaths": 200})
        following = state.advance_days()
        self.assertEqual(following.game_date, "2026-02-01")
        self.assertEqual(following.salary_updated_month, 1)
        self.assertEqual(following.contract("Leo").monthly_salary, 1_080_000)
        self.assertEqual(following.contract("Derke").monthly_salary, 2_400_000)
        self.assertEqual(following.money, state.money + 7_500_000 - 5_600_000)
        finished = following.advance_days()
        self.assertEqual(finished.game_date, "2026-02-02")
        self.assertEqual(finished.tournament("cup").completed_date, "2026-02-01")
        self.assertEqual(finished.money, following.money)
        self.assertEqual(len([e for e in finished.monthly_events if e.kind == "month_completed"]), 1)
        self.store.save(finished)
        self.assertEqual(self.store.load_or_create(), finished)


class SalaryScreenTest(CompetitionScreenTest):
    def setUp(self):
        super().setUp()
        self.results = self.path.parent / "results"
        self.results.mkdir()
        SalaryStateTest.configure_salary(self)
        SalaryStateTest.write_stats(self)
        self.app.commit(new_season(), "new")
        self.app.show_screen("starter")

    def test_mode_selection_saves_preview_and_start_locks_mode(self):
        app = self.app
        self.assertEqual(app.starter_salary_mode.get(), "静的月給（従来）")
        app.starter_salary_mode.set("成績連動月給（K/D）")
        app.starter_salary_menu.event_generate("<<ComboboxSelected>>")
        self.assertEqual(self.store.load_or_create().salary_mode, SalaryMode.KD_DYNAMIC)
        self.assertEqual(app.starter_players.set("Leo", "salary"), "900,000")
        app.commit(app.state.with_starter_selection(OWN), "chosen")
        app.confirm_starters()
        self.assertEqual(app.state.monthly_payroll, 5_600_000)
        self.assertEqual(str(app.starter_salary_menu["state"]), "disabled")
        self.assertIn("成績連動", app.home_summary.get())
        self.assertIn("-5,600,000", app.home_payroll.get())
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_pending_mode_restores_on_reload_and_hint_uses_saved_threshold(self):
        app = self.app
        with patch.object(config, "MIN_GAMES", 12):
            app.starter_salary_mode.set("成績連動月給（K/D）")
            app.change_starter_salary_mode()
        app.state = self.store.load_or_create()
        app.refresh()
        self.assertEqual(app.starter_salary_mode.get(), "成績連動月給（K/D）")
        self.assertIn("12マップ", app.starter_salary_hint.get())
        self.assertEqual(app.starter_players.set("Leo", "salary"), "100,000")
        app.root.update_idletasks()
        self.assertLessEqual(app.starter_host.winfo_reqheight(), 800)

    def test_pending_mode_can_revert_to_static_and_save_failure_restores_selection(self):
        app = self.app
        app.starter_salary_mode.set("成績連動月給（K/D）")
        app.change_starter_salary_mode()
        before = app.state
        with patch.object(self.store, "save", side_effect=OSError("test failure")), patch("run_realtime_season.messagebox.showerror"):
            app.starter_salary_mode.set("静的月給（従来）")
            app.change_starter_salary_mode()
        self.assertEqual(app.state, before)
        self.assertEqual(app.starter_salary_mode.get(), "成績連動月給（K/D）")
        app.starter_salary_mode.set("静的月給（従来）")
        app.change_starter_salary_mode()
        self.assertEqual(app.state.salary_mode, SalaryMode.STATIC)
        self.assertTrue(all(p.monthly_salary == 100_000 for p in app.state.starter_candidates))
        self.assertTrue(all(p.monthly_salary == 100_000 for c in app.state.opponent_teams for p in c.players))
        self.assertEqual(self.store.load_or_create(), app.state)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for cls in (SalaryRulesTest, SalaryStateTest, SalaryScreenTest):
        suite.addTests(cls(name) for name in cls.__dict__ if name.startswith("test_"))
    return suite


if __name__ == "__main__":
    unittest.main()
