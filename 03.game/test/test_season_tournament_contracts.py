"""Contract endings wait for registered tournament players, not reserves."""

from dataclasses import replace
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_rival_economy as economy
import realtime_season_teams as teams
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import SeriesScore, next_match
from season.season_rival_economy import make_offer
from season.season_series import build_series_request
from test_season_competitions import definition, finish, OWN, RIVALS


class TournamentContractTests(unittest.TestCase):
    def setUp(self):
        fixture = {n: replace(p, monthly_salary=100_000, loyalty=10, debut_chapter=1)
                   for n, p in character_stats.CHARACTER_TABLE.items()}
        self.cup = definition(start_date="2026-01-29", format="double_elimination")
        for context in (
            patch.dict(character_stats.CHARACTER_TABLE, fixture),
            patch.object(calendar, "START_DATE", "2026-01-01"),
            patch.object(calendar, "TOURNAMENTS", [self.cup]),
            patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
            patch.object(economy, "NON_REGULAR_OFFER_CHANCE", 0),
            patch.object(teams, "SEASON_TEAMS", [dict(name=f"Rival{i}", players=list(names), initial_money=100_000_000)
                                                for i, names in enumerate(RIVALS)]),
        ):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self, *, short_loyalty=50):
        state = new_season((*OWN, "Meiy")).with_roster(OWN).with_confirmed_team()
        state = state.with_selected_team(state.teams[0].id)
        return replace(state, money=100_000_000, contracts=tuple(
            replace(c, kind="short", duration_months=1, team_loyalty=short_loyalty) if c.player_name == "Leo"
            else replace(c, kind="short", duration_months=1) if c.player_name == "Meiy" else c
            for c in state.contracts))

    def in_progress(self, state=None):
        state = self.state() if state is None else state
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(28)
        while state.game_date < "2026-02-01":
            run = state.tournament("cup")
            match, _ = next_match(state.tournament_definition("cup"), run)
            winner = run.own_team_id if run.own_team_id in (match.left, match.right) else match.left
            old_date = state.date
            state = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right,
                match.maps_to_win if winner == match.left else 0, match.maps_to_win if winner == match.right else 0))
            if state.date == old_date:
                state = state.advance_days()
        self.assertFalse(state.tournament("cup").completed)
        return state

    def test_expired_starter_continues_until_finish_and_can_then_renew(self):
        state = self.in_progress()
        self.assertFalse(state.contract_active(state.contract("Leo")))
        self.assertTrue(state.contract_end_deferred(state.contract("Leo")))
        self.assertTrue(state.can_play("Leo"))
        self.assertFalse(state.can_play("Meiy"))
        with self.assertRaisesRegex(SeasonSaveError, "契約延期中"):
            state.with_renewed_contract("Leo", "year1")
        while state.tournament("cup").last_match_date == state.game_date:
            state = state.advance_days()
        # Continue using the actual registered lineup even after its nominal expiry.
        run = state.tournament("cup")
        match, _ = next_match(state.tournament_definition("cup"), run)
        while run.own_team_id not in (match.left, match.right):
            state = state.with_tournament_rating_result("cup")
            run = state.tournament("cup")
            match, _ = next_match(state.tournament_definition("cup"), run)
        request = build_series_request(state, "cup", render=False)
        self.assertIn("Leo", [p["name"] for p in request["own"]["players"] + request["opponent"]["players"]])
        ended = finish(state)
        self.assertFalse(ended.contract_end_deferred(ended.contract("Leo")))
        self.assertFalse(ended.can_play("Leo"))
        self.assertIsNotNone(ended.player("Leo"))
        self.assertTrue(ended.with_renewed_contract("Leo", "year1").can_play("Leo"))

    def test_extension_keeps_payroll_and_daily_iq_until_completion(self):
        state = self.in_progress()
        self.assertEqual(state.monthly_payroll, 500_000)
        run = state.tournament("cup")
        match, _ = next_match(state.tournament_definition("cup"), run)
        advanced = state.with_tournament_result("cup", SeriesScore(
            match.id, match.left, match.right, match.maps_to_win, 0))
        if advanced.date == state.date:
            advanced = advanced.advance_days()
        self.assertAlmostEqual(advanced.player("Leo").iq, state.player("Leo").iq + 0.1)
        self.assertEqual(advanced.player("Meiy").iq, state.player("Meiy").iq)
        ended = finish(advanced)
        self.assertEqual(ended.monthly_payroll, 500_000)
        self.assertEqual(ended.advance_days().player("Leo").iq, ended.player("Leo").iq)

    def test_zero_loyalty_departure_is_delayed_until_tournament_finishes(self):
        state = self.in_progress(self.state(short_loyalty=-100))
        self.assertIsNotNone(state.player("Leo"))
        self.assertTrue(state.can_play("Leo"))
        ended = finish(state)
        self.assertIsNone(ended.player("Leo"))
        self.assertEqual(ended.contract("Leo").end_reason, "left")

    def test_expiry_before_start_does_not_revive_contract(self):
        with patch.object(calendar, "TOURNAMENTS", [definition(start_date="2026-02-03", format="single_elimination")]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(33)
        self.assertFalse(state.contract_end_deferred(state.contract("Leo")))
        self.assertFalse(state.can_play("Leo"))
        request = build_series_request(state, "cup")
        own = request["own"] if request["left_id"] == state.club_id else request["opponent"]
        self.assertNotIn("Leo", [p["name"] for p in own["players"]])
        self.assertIn("友達", [p["name"] for p in own["players"]])

    def test_rival_expiry_settles_immediately_on_finish_not_next_month(self):
        state = self.state()
        rival = state.opponent_teams[0]
        rival = replace(rival, contracts=tuple(replace(c, kind="short", duration_months=1) if c.player_name in ("Aspas", "Sato") else c
                                               for c in rival.contracts))
        state = replace(state, opponent_teams=(rival, *state.opponent_teams[1:]))
        during = self.in_progress(state)
        club = during.opponent_teams[0]
        asp = next(c for c in club.contracts if c.player_name == "Aspas")
        bench = next(c for c in club.contracts if c.player_name == "Sato")
        self.assertEqual(asp.end_month, 1)
        self.assertTrue(during.contract_end_deferred(asp, club.id))
        self.assertGreater(bench.end_month, 1)
        ended = finish(during)
        renewed = next(c for c in ended.opponent_teams[0].contracts if c.player_name == "Aspas")
        self.assertEqual(renewed.start_month, ended.game_month)
        self.assertTrue(ended.contract_active(renewed))
        self.assertTrue(any(e.kind == "renewal" and e.player_name == "Aspas" and "大会終了" in e.message for e in ended.monthly_events))

    def test_save_reload_and_contract_status_with_offer(self):
        state = self.in_progress()
        buyer = state.opponent_teams[0]
        state = make_offer(state, buyer, state.player("Leo"), lambda *args: None)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertTrue(loaded.can_play("Leo"))
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, loaded)
        app.show_screen("contracts")
        self.assertEqual(app.contracts_players.item("Leo", "values")[1], "出場中につき契約延期中")
        app.contracts_players.selection_set("Leo")
        app.refresh_offer("contracts")
        self.assertEqual(str(app.offer_buttons["contracts"]["state"]), "disabled")
        self.assertIn("延期中", app.offer_summary["contracts"].get())
        ended = finish(state)
        app.state = ended
        app.refresh_management()
        self.assertNotIn("延期中", app.contracts_players.item("Leo", "values")[1])

    def test_poor_rival_gets_one_month_grace_after_tournament_completion(self):
        self.cup["prizes"] = {}
        state = self.state()
        rival = state.opponent_teams[0]
        rival = replace(rival, money=0, sponsor_active=False, contracts=tuple(
            replace(c, kind="short", duration_months=1) if c.player_name == "Aspas" else c
            for c in rival.contracts))
        state = replace(state, opponent_teams=(rival, *state.opponent_teams[1:]))
        during = self.in_progress(state)
        self.assertIn("Aspas", during.opponent_teams[0].members)
        ended = finish(during)
        self.assertIn("Aspas", ended.opponent_teams[0].members)
        contract = next(c for c in ended.opponent_teams[0].contracts if c.player_name == "Aspas")
        self.assertIsNone(contract.end_reason)
        self.assertEqual(contract.expired_on, ended.game_date)
        self.store.save(ended)
        self.assertEqual(self.store.load_or_create(), ended)


if __name__ == "__main__":
    unittest.main()
