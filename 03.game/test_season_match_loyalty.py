"""Loyalty depends on the actual five participants, once per series."""

from dataclasses import replace
import tkinter as tk
import unittest
from unittest.mock import patch

import realtime_season_competitions as calendar
import test_season_economy as fixtures
from realtime_season import SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
from season_competitions import next_match, SeriesScore
from season_rival_economy import make_offer


class MatchLoyaltyTests(unittest.TestCase):
    setUp = fixtures.SeasonEconomyTest.setUp

    def state(self):
        return fixtures.SeasonEconomyTest.state(self).with_scouted_player("Meiy", "year1")

    def test_benches_lose_three_on_wins_or_losses_and_result_is_applied_once(self):
        state = self.state()
        rival = state.opponent_teams[0]
        for won in (True, False):
            with self.subTest(won=won):
                result = state.with_rated_result("series", state.selected_team_id, rival.id, 3 if won else 2, 2 if won else 3)
                self.assertEqual(result.team_loyalty("Meiy"), 27)
                self.assertEqual(result.team_loyalty("Sato", rival.id), 27)
                self.assertEqual(result.team_loyalty("Leo"), 21 if won else 19.5)
                self.assertEqual(result.team_loyalty("Aspas", rival.id), 29.5 if won else 31)
                self.assertEqual(result.with_rated_result("series", state.selected_team_id, rival.id, 3, 2), result)
                self.assertEqual(result.player("Meiy").loyalty, state.player("Meiy").loyalty)
                self.store.save(result)
                self.assertEqual(self.store.load_or_create(), result)

    def test_expired_players_are_exempt_but_maximum_trait_bench_still_loses_three(self):
        state = self.state()
        state = replace(state, owned_players=tuple(replace(p, loyalty=10) if p.name == "Meiy" else p for p in state.owned_players))
        rival = state.opponent_teams[0]
        result = state.with_rated_result("win", state.club_id, rival.id, 1, 0)
        self.assertEqual(result.team_loyalty("Meiy"), 27)
        expired = replace(state, contracts=tuple(replace(c, start_month=0, kind="short", duration_months=1)
                                                if c.player_name == "Meiy" else c for c in state.contracts),
                          game_month=1, game_date="2026-02-01")
        result = expired.with_rated_result("win", expired.club_id, rival.id, 1, 0)
        self.assertEqual(result.team_loyalty("Meiy"), 30)

    def test_the_played_preset_is_used_even_if_another_preset_is_selected(self):
        state = self.state()
        first = state.selected_team_id
        state = state.with_new_team().with_roster(("Meiy", *fixtures.OWN[:4])).with_confirmed_team()
        played = state.teams[-1].id
        state = state.with_selected_team(first)
        result = state.with_rated_result("other-preset", played, state.opponent_teams[0].id, 1, 0)
        self.assertEqual(result.team_loyalty("Meiy"), 31)
        self.assertEqual(result.team_loyalty("Alfajer"), 17)

    def test_tournament_uses_registered_snapshot_even_if_rival_order_changes(self):
        with patch.object(calendar, "TOURNAMENTS", [fixtures.CUP]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        rival = state.opponent_teams[0]
        rival = replace(rival, players=(rival.players[-1], *rival.players[:-1]))
        state = replace(state, opponent_teams=(rival,))
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        score = SeriesScore(match.id, match.left, match.right, int(match.left == state.club_id), int(match.right == state.club_id))
        result = state.with_tournament_result("cup", score)
        self.assertEqual(result.team_loyalty("Sato", rival.id), 27)
        self.assertEqual(result.team_loyalty("tex", rival.id), 29.5)
        self.assertEqual(result.team_loyalty("Meiy"), 27)

    def test_scrim_snapshot_and_bench_penalty_can_force_transfer(self):
        state = self.state()
        rival = state.opponent_teams[0]
        from realtime_season import FORCED_OFFER_LOYALTY
        state = state.with_team_loyalty("Meiy", FORCED_OFFER_LOYALTY + 1)
        state = make_offer(state, rival, state.player("Meiy"), lambda *args: None)
        result = state.with_scrim_result("scrim", state.selected_team_id, rival.id, 1, 0,
                                         participants={state.club_id: fixtures.OWN, rival.id: rival.roster})
        self.assertIsNone(result.player("Meiy"))
        self.assertEqual(result.transfer_offers[-1].status, "forced")
        self.assertEqual(result.team_loyalty("Meiy"), FORCED_OFFER_LOYALTY - 2)

    def test_missing_or_duplicate_participants_are_rejected(self):
        state = self.state()
        rival = state.opponent_teams[0]
        for participants in ({state.club_id: fixtures.OWN},
                             {state.club_id: ("Leo", "Leo"), rival.id: rival.roster}):
            with self.assertRaises(SeasonSaveError):
                state.with_rated_result("invalid", state.club_id, rival.id, 1, 0, participants=participants)

    def test_editor_and_scout_details_use_the_same_level_thirty_cap(self):
        state = self.state()
        state = replace(state, owned_players=tuple(replace(p, research_level=23, aim_lab_level=30)
                                                 if p.name == "Leo" else p for p in state.owned_players))
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        app.players.selection_set(str(next(i for i, p in enumerate(state.owned_players) if p.name == "Leo")))
        app.select_player()
        self.assertIn("研究Lv: 23 / 30", app.details.get())
        self.assertIn("エイムラボLv: 30 / 30", app.details.get())
        app.refresh_scout_details(state.player("Leo"))
        self.assertIn("研究Lv: 23 / 30", app.scout_details.get())
        self.assertIn("エイムラボLv: 30 / 30", app.scout_details.get())


if __name__ == "__main__":
    unittest.main()
