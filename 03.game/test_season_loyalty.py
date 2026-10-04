"""Persistent player-to-club loyalty across transfers, LFT and old saves."""

from dataclasses import replace
import json
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season
import realtime_season_teams as teams
import test_season_rival_economy as fixtures
from realtime_season import SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
from season_rival_economy import make_offer
from season_transfers import restore_regular_members


class TeamLoyaltyTests(unittest.TestCase):
    setUp = fixtures.RivalEconomyTests.setUp

    def state(self):
        return replace(fixtures.RivalEconomyTests.state(self), money=100_000_000)

    def transfer(self, state, name, club_id):
        club = next(c for c in state.opponent_teams if c.id == club_id)
        offered = make_offer(state, club, state.player(name), lambda *args: None)
        return offered.with_transfer_response(offered.transfer_offer(name).id, True)

    def test_every_player_has_default_for_every_team_even_before_joining(self):
        state = self.state()
        ids = {state.club_id, *(c.id for c in state.opponent_teams)}
        self.assertTrue(set(character_stats.CHARACTER_TABLE).issubset(state.team_loyalties))
        for name in character_stats.CHARACTER_TABLE:
            self.assertEqual(set(state.team_loyalties[name]), ids)
            for identifier, value in state.team_loyalties[name].items():
                current = state.club_id if state.player(name) else state.opponent_owner(name).id if state.opponent_owner(name) else None
                self.assertEqual(value, 30 if identifier == current else 50)

    def test_transfers_and_rejoins_preserve_all_previous_club_values(self):
        state = self.state()
        a, b = (c.id for c in state.opponent_teams)
        state = state.with_team_loyalty("Leo", 72).with_team_loyalty("Leo", 64, a).with_team_loyalty("Leo", 38, b)
        state = self.transfer(state, "Leo", a)
        self.assertEqual(state.team_loyalty("Leo", a), 30)
        self.assertEqual(state.team_loyalty("Leo"), 72)
        state = state.with_team_loyalty("Leo", 81, a)
        state = state.with_scouted_player("Leo", "year1")
        self.assertEqual(state.contract("Leo").team_loyalty, 82)
        state = self.transfer(state, "Leo", b)
        self.assertEqual(state.team_loyalty("Leo", b), 30)
        state = state.with_scouted_player("Leo", "year1")
        # The existing offer system permits one proposal per player/club/month.
        state = state.advance_months()
        state = self.transfer(state, "Leo", a)
        self.assertEqual(state.team_loyalty("Leo", a), 91)
        self.assertEqual(state.team_loyalty("Leo", b), 30)
        self.assertEqual(state.team_loyalty("Leo"), 91.5)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.team_loyalties, state.team_loyalties)
        self.assertEqual(loaded, state)

    def test_lft_freezes_previous_value_and_recruitment_restores_it(self):
        state = self.state().with_scouted_player("Aspas", "year1").with_team_loyalty("Aspas", 27)
        rival_id = state.opponent_teams[0].id
        state = state.without_player("Aspas").advance_days(2)
        self.assertIsNone(state.opponent_owner("Aspas"))
        self.assertEqual(state.team_loyalty("Aspas"), 27)
        self.assertEqual(state.team_loyalty("Aspas", rival_id), 30)
        state = state.with_scouted_player("Aspas", "year1")
        self.assertEqual(state.contract("Aspas").team_loyalty, 37)

    def test_regular_ai_return_from_another_rival_restores_target_history(self):
        state = self.state()
        a, b = (c.id for c in state.opponent_teams)
        state = state.with_team_loyalty("Aspas", 83, a).with_team_loyalty("Aspas", 62, b)
        state = state.with_scouted_player("Aspas", "year1")
        state = self.transfer(state, "Aspas", b)
        returned = restore_regular_members(state, set(), lambda *args: None)
        self.assertEqual(returned.opponent_owner("Aspas").id, a)
        self.assertEqual(returned.team_loyalty("Aspas", a), 93)
        self.assertEqual(returned.team_loyalty("Aspas", b), 30)
        returned.validate()

    def test_forced_transfer_preserves_source_and_restores_destination(self):
        state = self.state()
        buyer = state.opponent_teams[0]
        state = state.with_team_loyalty("Leo", 91, buyer.id)
        state = make_offer(state, state.opponent_teams[0], state.player("Leo"), lambda *args: None)
        threshold = realtime_season.FORCED_OFFER_LOYALTY
        state = state.with_team_loyalty("Leo", threshold - .1)
        self.assertEqual(state.transfer_offers[-1].status, "forced")
        self.assertEqual(state.team_loyalty("Leo"), threshold - .1)
        self.assertEqual(state.team_loyalty("Leo", buyer.id), 30)

    def test_ai_departure_to_lft_keeps_values_and_other_club_recruits_with_history(self):
        state = self.state()
        a, b = (c.id for c in state.opponent_teams)
        state = state.with_team_loyalty("Aspas", .4, a).with_team_loyalty("Aspas", 73, b)
        club = state.opponent_teams[0]
        club = replace(club, contracts=tuple(replace(c, kind="short", duration_months=6) if c.player_name == "Aspas" else c
                                            for c in club.contracts))
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        with patch("season_monthly_events.choose_recruit", return_value=None):
            state = state.advance_months(2)
        self.assertIsNone(state.opponent_owner("Aspas"))
        self.assertAlmostEqual(state.team_loyalty("Aspas", a), -.1)
        self.assertEqual(state.team_loyalty("Aspas", b), 73)
        club = state.opponent_teams[1]
        club = replace(club, regular_members=("Aspas", *club.regular_members[:4]))
        former = replace(state.opponent_teams[0], regular_members=(), regular_igl=None, regular_carrier=None)
        state = replace(state, opponent_teams=(former, club))
        state = restore_regular_members(state, set(), lambda *args: None)
        self.assertEqual(state.opponent_owner("Aspas").id, b)
        self.assertEqual(state.team_loyalty("Aspas", b), 30)
        self.assertAlmostEqual(state.team_loyalty("Aspas", a), -.1)
        state.validate()

    def test_match_and_month_change_only_current_club_relationship(self):
        state = self.state()
        a, b = (c.id for c in state.opponent_teams)
        state = state.with_team_loyalty("Leo", 64, a).with_team_loyalty("Leo", 38, b)
        win = state.with_rated_result("win", state.club_id, a, 1, 0)
        self.assertEqual(win.team_loyalty("Leo"), 31)
        self.assertEqual(win.team_loyalties["Leo"][state.club_id], 31)
        month = win.advance_months()
        self.assertEqual(month.team_loyalty("Leo"), 30.5)
        self.assertEqual(month.team_loyalty("Leo", a), 64)
        self.assertEqual(month.team_loyalty("Leo", b), 38)

    def test_negative_loyalty_prevents_return_to_that_club_only(self):
        state = self.state()
        a, b = (c.id for c in state.opponent_teams)
        state = state.with_team_loyalty("Leo", -.1, a)
        refused = make_offer(state, state.opponent_teams[0], state.player("Leo"), lambda *args: None)
        self.assertFalse(refused.pending_transfer_offers)
        state = self.transfer(state, "Leo", b).with_team_loyalty("Leo", -.1)
        with self.assertRaisesRegex(SeasonSaveError, "忠誠"):
            state.with_scouted_player("Leo", "year1")
        self.assertEqual(state.team_loyalty("Leo", b), 30)

    def test_removed_team_and_reintroduced_team_keep_the_same_relationship(self):
        state = self.state()
        club_id = state.opponent_teams[0].id
        state = state.with_team_loyalty("Aspas", 79, club_id)
        with patch.object(teams, "SEASON_TEAMS", self.config[1:]):
            state = self.store.import_season_teams(state)
        self.assertEqual(state.team_loyalty("Aspas", club_id), 79)
        state = self.store.import_season_teams(state)
        self.assertEqual(state.opponent_owner("Aspas").id, club_id)
        self.assertEqual(state.team_loyalty("Aspas", club_id), 89)

    def test_legacy_save_recovers_retained_current_and_ended_contracts(self):
        state = self.state()
        state = state.with_team_loyalty("Aspas", 77, state.opponent_teams[0].id)
        state = state.with_scouted_player("Aspas", "year1").with_team_loyalty("Aspas", 63).without_player("Aspas")
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 24
        del data["team_loyalties"]
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.team_loyalty("Aspas"), 63)
        self.assertEqual(loaded.team_loyalty("Aspas", loaded.opponent_teams[0].id), 77)
        self.assertEqual(loaded.team_loyalty("Aspas", loaded.opponent_teams[1].id), 50)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_bad_memory_cannot_overwrite_save(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        for values in ([], {"Leo": []}, {"Leo": {"id": True}}, {"Leo": {"id": float("nan")}}):
            with self.subTest(values=values), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, team_loyalties=values))
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_ui_lists_all_team_values_and_scout_uses_current_club_value(self):
        state = self.state()
        a = state.opponent_teams[0].id
        state = state.with_team_loyalty("Leo", 72).with_team_loyalty("Leo", 64, a)
        state = self.transfer(state, "Leo", a)
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        window = app.show_player_loyalties("Leo")
        rows = {key: window.loyalty_table.item(key, "values") for key in window.loyalty_table.get_children()}
        self.assertEqual(rows[state.club_id][1], "72")
        self.assertEqual(rows[a][1:], ("30", "● 所属中"))
        player = next(p for p in state.scout_players if p.name == "Leo")
        app.refresh_scout_details(player)
        self.assertIn("所属チームへの忠誠: 30", app.scout_details.get())


if __name__ == "__main__":
    unittest.main()
