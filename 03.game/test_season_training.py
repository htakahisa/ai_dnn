"""Paid player development and one calendar day for completed scrims."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_teams as teams
import realtime_season_training as training
import realtime_season_world_levels as world
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season_scrim import build_scrim_request
from season_series import build_series_request


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex")
TRAINING = {title: {"上昇量": growth, "費用": [100_000 * level for level in range(1, 11)]}
            for title, growth in (("研究", 5), ("エイムラボ", 1))}
CUP = dict(id="cup", name="Cup", start_date="2026-01-31", team_count=2,
           format="single_elimination", prizes={}, normal_maps_to_win=1)


class TrainingTest(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        contexts = [patch.dict(character_stats.CHARACTER_TABLE, fixture),
                    patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
                    patch.object(teams, "SEASON_TEAMS", [dict(name="Rival", players=list(RIVAL))]),
                    patch.object(calendar, "START_DATE", "2026-01-01"),
                    patch.object(calendar, "TOURNAMENTS", []),
                    patch.object(world, "WORLD_LEVELS", [{"レベル": 1, "上位%": 100, "敵倍率": 1, "スポンサー資金": 7_500_000}]),
                    patch.object(training, "TRAINING", TRAINING)]
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        return state.with_selected_team(state.teams[0].id)

    def app(self, state):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        self.store.save(state)
        return RealtimeSeasonApp(root, self.store, state)

    def test_research_changes_only_selected_player_iq_and_level_and_cost_increases(self):
        state = self.state()
        first = state.with_trained_player("Leo", "research")
        self.assertEqual(first.player("Leo"), replace(state.player("Leo"), iq=155, research_level=1))
        self.assertEqual(first.money, state.money - 100_000)
        self.assertEqual(first.date, state.date)
        self.assertEqual(first.owned_players[1:], state.owned_players[1:])
        self.assertEqual(first.opponent_teams, state.opponent_teams)
        self.assertEqual(first.contracts, state.contracts)
        self.assertEqual(first.training_terms("Leo", "research").cost, 200_000)
        second = first.with_trained_player("Leo", "research")
        self.assertEqual(second.player("Leo").iq, 160)
        self.assertEqual(second.player("Leo").research_level, 2)
        self.assertEqual(second.money, state.money - 300_000)

    def test_research_cap_ten_is_persisted_without_extra_charges(self):
        state = self.state()
        before = state
        for level in range(1, 11):
            state = state.with_trained_player("Leo", "research")
            self.assertEqual(state.player("Leo").research_level, level)
        self.assertEqual(state.player("Leo").iq, before.player("Leo").iq + 50)
        self.assertEqual(state.money, before.money - 5_500_000)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        with self.assertRaisesRegex(SeasonSaveError, "最大レベル10"):
            loaded.with_trained_player("Leo", "research")

    def test_aim_lab_uses_percentage_points_and_has_independent_levels(self):
        state = self.state()
        for _ in range(10):
            state = state.with_trained_player("Leo", "aim_lab")
        self.assertEqual(state.player("Leo").hit_pct, .93)
        self.assertEqual(state.player("Leo").aim_lab_level, 10)
        self.assertEqual(state.player("Leo").research_level, 0)
        self.assertEqual(state.player("Leo").iq, 150)
        with self.assertRaisesRegex(SeasonSaveError, "最大レベル10"):
            state.with_trained_player("Leo", "aim_lab")
        state = state.with_trained_player("Leo", "research")
        self.assertEqual((state.player("Leo").research_level, state.player("Leo").aim_lab_level), (1, 10))

    def test_aim_lab_caps_at_one_hundred_percent_and_cannot_charge_after_cap(self):
        state = self.state()
        state = replace(state, owned_players=(replace(state.owned_players[0], hit_pct=.995), *state.owned_players[1:]))
        state = state.with_trained_player("Leo", "aim_lab")
        self.assertEqual(state.player("Leo").hit_pct, 1)
        self.assertEqual(state.player("Leo").aim_lab_level, 1)
        with self.assertRaisesRegex(SeasonSaveError, "100%"):
            state.with_trained_player("Leo", "aim_lab")

    def test_invalid_player_insufficient_funds_and_bad_config_leave_state_unchanged(self):
        state = replace(self.state(), money=99_999)
        self.store.save(state)
        before = self.store.path.read_bytes()
        for name, kind in (("Leo", "research"), ("Aspas", "research"), ("missing", "aim_lab"), ("Leo", "invalid")):
            with self.subTest(name=name, kind=kind), self.assertRaises(SeasonSaveError):
                state.with_trained_player(name, kind)
        with patch.object(training, "TRAINING", {"研究": {"上昇量": 5, "費用": [1] * 10}}):
            with self.assertRaises(SeasonSaveError):
                state.with_trained_player("Leo", "research")
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(state.player("Leo").research_level, 0)

    def test_training_reaches_next_scrim_and_current_tournament_without_changing_lineup(self):
        with patch.object(calendar, "TOURNAMENTS", [{**CUP, "start_date": "2026-01-01"}]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        snapshot = state.tournament("cup").entrants
        state = state.with_trained_player("Leo", "research").with_trained_player("Leo", "aim_lab")
        request = build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id)
        self.assertEqual(request["own"]["players"][0]["iq"], 155)
        self.assertEqual(request["own"]["players"][0]["hit_pct"], .84)
        request = build_series_request(state, "cup")
        self.assertEqual(request["own"]["players"][0]["iq"], 155)
        self.assertEqual(state.tournament("cup").entrants, snapshot)

    def test_training_survives_release_lft_reacquisition_and_opponent_transfer(self):
        state = new_season(OWN).with_trained_player("Leo", "research")
        trained = state.player("Leo")
        released = state.without_player("Leo")
        self.assertEqual(next(p for p in released.lft_players if p.name == "Leo"), trained)
        self.store.save(released)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.with_scouted_player("Leo", "short", 1).player("Leo"), trained)
        self.assertEqual(loaded.with_added_players(("Leo",)).player("Leo"), trained)
        # Move a trained opponent back to its original team through an offer.
        state = replace(self.state(), opponent_teams=tuple(replace(c, regular_members=c.members) for c in self.state().opponent_teams))
        state = state.with_scouted_player("Aspas", "year1").with_trained_player("Aspas", "research").advance_months()
        trained = state.player("Aspas")
        state = state.with_transfer_response(state.transfer_offer("Aspas").id, True)
        self.assertEqual(next(p for p in state.opponent_owner("Aspas").players if p.name == "Aspas"), trained)

    def test_legacy_save_migrates_levels_to_zero_without_changing_abilities_or_money(self):
        state = self.state()
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 15
        data.pop("developed_players")
        for player in [*data["owned_players"], *(p for c in data["opponent_teams"] for p in c["players"])]:
            player.pop("research_level")
            player.pop("aim_lab_level")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_invalid_level_or_missing_training_data_cannot_overwrite_save(self):
        state = self.state().with_trained_player("Leo", "research")
        self.store.save(state)
        before = self.store.path.read_bytes()
        for level in (-1, 11, 1.5, True):
            bad = replace(state, owned_players=(replace(state.owned_players[0], research_level=level), *state.owned_players[1:]))
            with self.subTest(level=level), self.assertRaises(SeasonSaveError):
                self.store.save(bad)
            self.assertEqual(self.store.path.read_bytes(), before)
        data = json.loads(before)
        data["owned_players"][0].pop("research_level")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(SeasonSaveError, "育成レベル"):
            self.store.load_or_create()

    def test_training_screens_select_upgrade_refresh_save_and_disable_at_limits(self):
        app = self.app(self.state())
        app.show_screen("research")
        tree = app.training_players["research"]
        tree.selection_set("Leo")
        app.refresh_training_offer("research")
        self.assertIn("100,000", app.training_summaries["research"].get())
        app.training_buttons["research"].invoke()
        self.assertEqual(app.state.player("Leo").iq, 155)
        self.assertIn("200,000", app.training_summaries["research"].get())
        app.show_screen("aim_lab")
        app.training_players["aim_lab"].selection_set("Leo")
        app.refresh_training_offer("aim_lab")
        app.training_buttons["aim_lab"].invoke()
        self.assertEqual(app.state.player("Leo").hit_pct, .84)
        self.assertEqual(self.store.load_or_create(), app.state)
        with patch.object(type(app), "match_running", new_callable=unittest.mock.PropertyMock, return_value=True):
            app.refresh_training_offer("aim_lab")
            self.assertEqual(str(app.training_buttons["aim_lab"].cget("state")), "disabled")
            before = app.state
            app.train_selected_player("aim_lab")
            self.assertEqual(app.state, before)
        for screen in ("research", "aim_lab", "home"):
            app.show_screen(screen)
            app.root.update_idletasks()
            host = app.home_host if screen == "home" else app.training_hosts[screen]
            self.assertLessEqual(host.winfo_reqheight(), 800)

    def test_scrim_result_spends_exactly_one_day_and_is_idempotent_after_restart(self):
        state = self.state()
        done = state.with_scrim_result("scrim:one", state.selected_team_id, state.opponent_teams[0].id, 1, 0)
        self.assertEqual(done.date, state.date + timedelta(days=1))
        self.assertEqual(done.rating(done.club_id), 1532)
        self.assertEqual(done.with_scrim_result("scrim:one", state.selected_team_id, state.opponent_teams[0].id, 1, 0), done)
        self.store.save(done)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.with_scrim_result("scrim:one", state.selected_team_id, state.opponent_teams[0].id, 1, 0), done)

    def test_scrim_month_boundary_runs_normal_sponsor_payroll_and_events(self):
        state = self.state().advance_days(30)
        done = state.with_scrim_result("scrim:month", state.selected_team_id, state.opponent_teams[0].id, 0, 1)
        self.assertEqual(done.game_date, "2026-02-01")
        self.assertEqual(done.money, state.money + 7_500_000 - state.monthly_payroll)
        self.assertEqual(done.monthly_events_through, 1)
        self.assertEqual(sum(e.kind == "month_completed" for e in done.monthly_events), 1)
        self.assertEqual(done, state.with_rated_result("scrim:month", state.selected_team_id, state.opponent_teams[0].id, 0, 1).advance_days())

    def test_scrim_spends_day_even_with_an_unplayed_registered_tournament(self):
        with patch.object(calendar, "TOURNAMENTS", [CUP]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(30)
        self.assertEqual(state.advance_days(), state)
        done = state.with_scrim_result("scrim:busy", state.selected_team_id, state.opponent_teams[0].id, 1, 0)
        self.assertEqual(done.game_date, "2026-02-01")
        self.assertEqual(done.tournament("cup").results, ())

    def test_scrim_ui_cancel_and_error_do_not_spend_days_and_failed_save_can_retry(self):
        app = self.app(self.state())
        before = app.state
        for status in ("cancelled", "error"):
            app.scrim_job = SimpleNamespace(poll=lambda: dict(status=status))
            app._scrim_rating_context = ("scrim:failed", app.state.selected_team_id, app.state.opponent_teams[0].id)
            app.poll_scrim()
            self.assertEqual(app.state, before)
        result = dict(status="completed", own_team=app.state.team_name, opponent_team="Rival", own_score=13,
                      opponent_score=9, winner=app.state.team_name)
        app.scrim_job = SimpleNamespace(poll=lambda: result)
        app._scrim_rating_context = ("scrim:retry", app.state.selected_team_id, app.state.opponent_teams[0].id)
        with patch.object(app, "commit", return_value=False):
            app.poll_scrim()
        self.assertEqual(app.state, before)
        app.root.after_cancel(app._scrim_after_id)
        app.poll_scrim()
        self.assertEqual(app.state.date, before.date + timedelta(days=1))
        self.assertEqual(self.store.load_or_create(), app.state)
        app.poll_scrim()
        self.assertEqual(app.state.date, before.date + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
