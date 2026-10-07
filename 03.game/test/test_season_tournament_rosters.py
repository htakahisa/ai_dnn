from dataclasses import replace
import tkinter as tk
import time
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
from realtime_season import SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import SeriesScore, next_match
from season.season_competition_ui import CURRENT_ROSTER
from season.season_series import build_series_request
from season.season_scrim import ScrimJob
from season.season_tournament_rosters import friend_player
from test_season_competitions import OWN, definition, finish
import test_season_tournament_contracts as fixtures


class TournamentRosterTests(unittest.TestCase):
    def setUp(self):
        catalog = {name: replace(p, debut_chapter=1) for name, p in character_stats.CHARACTER_TABLE.items()}
        context = patch.dict(character_stats.CHARACTER_TABLE, catalog)
        context.start()
        self.addCleanup(context.stop)
        fixtures.TournamentContractTests.setUp(self)

    state = fixtures.TournamentContractTests.state

    def entered(self):
        state = self.state()
        # Keep a contracted reserve available for roster changes during the cup.
        state = replace(state, contracts=tuple(replace(c, kind="year1", duration_months=12)
                        if c.player_name == "Meiy" else c for c in state.contracts))
        return state.with_tournament_entry("cup", state.selected_team_id)

    def app(self, state):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        self.store.save(state)
        app = RealtimeSeasonApp(root, self.store, state)
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.preview_competition()
        return app

    def test_cancel_before_start_preserves_date_money_and_allows_reregistration(self):
        state = self.entered()
        cancelled = state.with_cancelled_tournament_entry("cup")
        self.assertIsNone(cancelled.tournament("cup"))
        self.assertEqual((cancelled.date, cancelled.money), (state.date, state.money))
        self.store.save(cancelled)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, cancelled)
        registered = loaded.with_tournament_entry("cup", loaded.selected_team_id)
        self.assertIsNotNone(registered.tournament("cup"))
        registered.validate()

    def test_cancellation_is_rejected_on_start_day_and_for_npc_tournaments(self):
        state = self.entered().advance_days(28)
        with self.assertRaises(SeasonSaveError):
            state.with_cancelled_tournament_entry("cup")
        with self.assertRaises(SeasonSaveError):
            self.state().with_cancelled_tournament_entry("cup")
        ended = finish(state)
        with self.assertRaises(SeasonSaveError):
            ended.with_cancelled_tournament_entry("cup")

    def test_expired_and_departed_registered_players_are_replaced_without_new_contracts(self):
        with patch.object(calendar, "TOURNAMENTS", [definition(start_date="2026-03-03", format="single_elimination")]):
            state = self.entered()
        state = state.advance_days(61)
        self.assertIsNone(state.player("Leo"))
        own = state.tournament_team("cup")
        self.assertEqual(len(own.players), 5)
        self.assertIn(friend_player(1), own.players)
        request = build_series_request(state, "cup", render=False)
        team = request["own"] if request["left_id"] == state.club_id else request["opponent"]
        self.assertEqual(len(team["players"]), 5)
        self.assertNotIn("Leo", [p["name"] for p in team["players"]])
        self.assertIsNone(state.player("友達"))
        self.assertIsNone(state.contract("友達"))
        self.assertEqual(state.monthly_payroll, 500_000)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        ended = finish(state)
        self.assertTrue(ended.tournament("cup").completed)

    def test_all_five_substitutes_have_unique_names_and_initial_abilities(self):
        state = self.entered().with_tournament_roster("cup", ())
        players = state.tournament_team("cup").players
        self.assertEqual(len({p.name for p in players}), 5)
        for index, player in enumerate(players, 1):
            self.assertEqual(player, friend_player(index))
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)

    def test_entry_with_expired_preset_members_fills_only_unplayable_slots(self):
        with patch.object(calendar, "TOURNAMENTS", [definition(start_date="2026-02-03")]):
            state = self.state().with_preset_settings(igl="Leo", carrier="Leo", ai="fnatic_v3")
        state = state.advance_days(31)
        self.assertFalse(state.can_play("Leo"))
        contracts, money, date = state.contracts, state.money, state.date
        entered = state.with_tournament_entry("cup", state.selected_team_id)
        team = entered.tournament_team("cup")
        self.assertEqual(tuple(p.name for p in team.players), (*OWN[1:], "友達"))
        self.assertEqual(team.ai, "fnatic_v3")
        self.assertIn(team.igl, OWN[1:])
        self.assertEqual(team.carrier, OWN[1])
        self.assertEqual((entered.contracts, entered.money, entered.date), (contracts, money, date))
        self.assertFalse(entered.can_play("Leo"))
        self.store.save(entered)
        self.assertEqual(self.store.load_or_create(), entered)
        self.assertTrue(finish(entered.advance_days(2)).tournament("cup").completed)

    def test_draft_entry_with_zero_to_four_players_survives_reload_and_finishes(self):
        for count in (0, 1, 4):
            with self.subTest(count=count):
                state = replace(self.state(), teams=(), selected_team_id=None,
                                editing_team_id=None).with_roster(OWN[:count])
                entered = state.with_tournament_entry("cup")
                team = entered.tournament_team("cup")
                self.assertEqual(team.players[:count], tuple(state.player(n) for n in OWN[:count]))
                self.assertEqual(team.players[count:], tuple(friend_player(n) for n in range(1, 6 - count)))
                self.assertEqual(entered.teams, ())
                self.assertEqual((entered.contracts, entered.money, entered.date),
                                 (state.contracts, state.money, state.date))
                self.store.save(entered)
                loaded = self.store.load_or_create()
                self.assertEqual(loaded, entered)
                self.assertTrue(finish(loaded.advance_days(28)).tournament("cup").completed)

    def test_saved_preset_with_no_active_players_enters_with_five_friends(self):
        with patch.object(calendar, "TOURNAMENTS", [definition(start_date="2026-02-03")]):
            state = self.state()
        state = replace(state, contracts=tuple(
            replace(c, kind="short", duration_months=1, team_loyalty=50) for c in state.contracts))
        state = state.advance_days(31)
        self.assertFalse(any(state.can_play(p.name) for p in state.owned_players))
        entered = state.with_tournament_entry("cup", state.selected_team_id)
        self.assertEqual(entered.tournament_team("cup").players,
                         tuple(friend_player(n) for n in range(1, 6)))
        self.store.save(entered)
        self.assertEqual(self.store.load_or_create(), entered)
        self.assertTrue(finish(entered.advance_days(2)).tournament("cup").completed)

    def test_ui_can_register_an_incomplete_draft_without_saved_presets(self):
        state = replace(self.state(), teams=(), selected_team_id=None,
                        editing_team_id=None).with_roster(OWN[:2])
        app = self.app(state)
        self.assertEqual(app.competition_team.get(), CURRENT_ROSTER)
        app.competition_enter_button.invoke()
        self.assertIsNotNone(app.state.tournament("cup"))
        self.assertIn("友達3", app.competition_roster_summary.get())
        updated = app.state.with_roster(OWN[1:2]).with_preset_settings(ai="fnatic_v3")
        app.commit(updated, "Draft changed")
        app.competition_roster_preset_button.invoke()
        team = app.state.tournament_team("cup")
        self.assertEqual(tuple(p.name for p in team.players), (OWN[1], "友達", "友達2", "友達3", "友達4"))
        self.assertEqual(team.ai, "fnatic_v3")
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_five_friends_complete_a_real_tournament_series(self):
        state = self.state().with_new_team().with_tournament_entry("cup").advance_days(28)
        request = build_series_request(state, "cup", render=False, tick_time_ms=15)
        job = ScrimJob(request, self.store.path.parent / "friend_match")
        try:
            deadline = time.monotonic() + 120
            result = None
            while result is None and time.monotonic() < deadline:
                result = job.poll()
                if result is None:
                    time.sleep(.05)
            self.assertIsNotNone(result, f"Series timed out. Log: {job.log_path}")
            self.assertEqual(result["status"], "completed", result)
            participants = result["maps"][0]["player_stats"]
            self.assertEqual(len(participants), 10)
            self.assertTrue(all(friend_player(n).name in participants for n in range(1, 6)))
        finally:
            job.cancel()
            job.process.wait(timeout=10)

    def test_roster_changes_before_start_and_between_matches_preserve_progress(self):
        state = self.entered()
        original = state.tournament("cup")
        names = (*OWN[1:], "Meiy")
        state = state.with_tournament_roster("cup", names)
        self.assertEqual(tuple(p.name for p in state.tournament_team("cup").players), names)
        self.assertEqual(state.date.isoformat(), "2026-01-01")
        state = state.advance_days(28)
        run = state.tournament("cup")
        match, _ = next_match(state.tournament_definition("cup"), run)
        request = build_series_request(state, "cup", render=False)
        own = request["own"] if request["left_id"] == state.club_id else request["opponent"]
        self.assertEqual(tuple(p["name"] for p in own["players"]), names)
        state = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right, match.maps_to_win, 0))
        results = state.tournament("cup").results
        date, money = state.date, state.money
        state = state.with_tournament_roster("cup", OWN, igl="Boaster", carrier="Derke")
        self.assertEqual(state.tournament("cup").results, results)
        self.assertEqual(state.tournament("cup").seed, original.seed)
        self.assertEqual((state.date, state.money), (date, money))
        self.assertEqual(state.tournament_team("cup").igl, "Boaster")
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        self.assertTrue(finish(state).tournament("cup").completed)

    def test_duplicate_unowned_or_six_player_rosters_are_rejected(self):
        state = self.entered()
        for names in (("Leo", "Leo"), ("Unknown",), (*OWN, "Meiy")):
            with self.subTest(names=names), self.assertRaises(SeasonSaveError):
                state.with_tournament_roster("cup", names)

    def test_saved_preset_can_be_changed_after_registration(self):
        state = self.entered().with_roster((*OWN[1:], "Meiy")).with_confirmed_team()
        state.validate()
        preset = state.selected_team
        changed = state.with_tournament_roster("cup", preset.roster, preset_id=preset.id,
                                               ai=preset.ai, igl=preset.igl, carrier=preset.carrier)
        self.assertEqual(tuple(p.name for p in changed.tournament_team("cup").players), preset.roster)

    def test_ui_shows_roster_cancels_entry_and_applies_selection_during_tournament(self):
        app = self.app(self.entered())
        self.assertIn("Leo", app.competition_roster_summary.get())
        self.assertEqual(str(app.competition_entry_cancel_button["state"]), "normal")
        app.competition_entry_cancel_button.invoke()
        self.assertIsNone(app.state.tournament("cup"))
        app.competition_enter_button.invoke()
        self.assertIsNotNone(app.state.tournament("cup"))
        app.competition_roster_players.selection_set(("Meiy", "Boaster"))
        app.competition_roster_apply_button.invoke()
        names = tuple(p.name for p in app.state.tournament_team("cup").players)
        self.assertEqual(len(names), 5)
        self.assertIn("Meiy", names)
        self.assertIn("友達3", names)
        self.assertIn("友達", app.competition_roster_summary.get())
        app.commit(app.state.advance_days(28), "Cup started")
        self.assertEqual(str(app.competition_entry_cancel_button["state"]), "disabled")
        self.assertEqual(str(app.competition_roster_apply_button["state"]), "normal")
        app.competition_roster_players.selection_set(OWN)
        app.competition_roster_apply_button.invoke()
        self.assertEqual(tuple(p.name for p in app.state.tournament_team("cup").players), OWN)
        with patch.object(type(app), "match_running", new_callable=unittest.mock.PropertyMock, return_value=True):
            app.preview_competition()
            self.assertEqual(str(app.competition_roster_apply_button["state"]), "disabled")
        app.root.update_idletasks()
        self.assertLessEqual(app.competition_host.winfo_reqheight(), 800)
        self.assertEqual(self.store.load_or_create(), app.state)


if __name__ == "__main__":
    unittest.main()
