"""Roster mixing, monthly returns and player-controlled outgoing transfers."""

from dataclasses import replace
import json
from pathlib import Path
from random import Random
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_config as config
import realtime_season_teams as teams
import realtime_season_competitions as calendar
from realtime_season import SeasonStore, SeasonSaveError, new_season
from season_monthly_events import process_monthly_events
from season_transfers import with_randomized_clubs
from run_realtime_season import RealtimeSeasonApp


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = (("Aspas", "valyn", "trent", "leaf", "tex"), ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"))


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.configured = [dict(name=f"Rival{i}", players=list(names), igl=names[0], carrier=names[1])
                           for i, names in enumerate(RIVALS)]
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5) for name, p in character_stats.CHARACTER_TABLE.items()}
        patches = [patch.dict(character_stats.CHARACTER_TABLE, fixture),
                   patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
                   patch.object(teams, "SEASON_TEAMS", self.configured),
                   patch.object(calendar, "START_DATE", "2026-01-01"),
                   patch.object(calendar, "TOURNAMENTS", [])]
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self, seed=1):
        with patch("realtime_season.with_randomized_clubs", side_effect=lambda s: with_randomized_clubs(s, Random(seed))):
            return new_season().with_initial_selection(OWN).with_roster(OWN).with_confirmed_team()

    def offer_state(self):
        # Start with a fixed world, but retain the original team membership.
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        state = replace(state, opponent_teams=tuple(replace(c, regular_members=c.members,
                        regular_igl=c.igl, regular_carrier=c.carrier) for c in state.opponent_teams))
        return state.with_scouted_player("Aspas", "year1").advance_months()

    def test_initial_mixing_is_unique_and_averages_two_missing_starters(self):
        counts = []
        cross_team, lft = False, False
        for seed in range(40):
            state = self.state(seed)
            state.validate()
            names = [p.name for p in state.owned_players] + [p.name for c in state.opponent_teams for p in c.players]
            self.assertEqual(len(names), len(set(names)))
            for club, original in zip(state.opponent_teams, RIVALS):
                count = len(set(original) - set(club.roster))
                self.assertIn(count, (1, 2, 3))
                counts.append(count)
                self.assertEqual(club.regular_members, original)
                for name in set(original) - set(club.members):
                    owner = state.opponent_owner(name)
                    cross_team |= owner is not None and owner.id != club.id
                    lft |= owner is None
        self.assertAlmostEqual(sum(counts) / len(counts), 2, delta=.25)
        self.assertTrue(cross_team and lft)

    def test_initial_selection_and_restart_do_not_repeat_mixing(self):
        state = self.state()
        self.store.save(state)
        with patch("realtime_season.with_randomized_clubs", side_effect=AssertionError("mixed twice")):
            self.assertEqual(self.store.load_or_create(), state)
        with self.assertRaises(SeasonSaveError):
            state.with_initial_selection(OWN)

    def test_monthly_event_restores_original_lineups_and_roles(self):
        state = self.state()
        self.assertEqual(state.advance_days(30).opponent_teams, state.opponent_teams)
        month = state.advance_months()
        self.assertEqual(tuple(c.members for c in month.opponent_teams), RIVALS)
        for club, original in zip(month.opponent_teams, RIVALS):
            self.assertEqual((club.igl, club.carrier), original[:2])
        self.assertTrue(any(e.kind == "roster_return" for e in month.monthly_events))
        self.assertEqual(process_monthly_events(month), month)
        self.store.save(month)
        self.assertEqual(self.store.load_or_create(), month)

    def test_monthly_offer_is_persisted_without_duplicate_pending_requests(self):
        state = self.offer_state()
        offer = state.transfer_offer("Aspas")
        self.assertIsNotNone(offer)
        self.assertEqual(offer.fee, 1_200_000)
        self.assertIsNotNone(state.player("Aspas"))
        following = state.advance_months()
        self.assertEqual(following.pending_transfer_offers, (offer,))
        self.assertEqual(sum(e.kind == "transfer_offer" for e in following.monthly_events), 1)
        self.store.save(following)
        self.assertEqual(self.store.load_or_create(), following)

    def test_accept_moves_snapshot_pays_fee_and_removes_invalid_presets(self):
        state = self.offer_state()
        state = state.with_roster(("Aspas", *OWN[:4])).with_confirmed_team()
        state = state.with_selected_team(state.teams[-1].id)
        offer = state.transfer_offer("Aspas")
        snapshot = state.player("Aspas")
        transferred = state.with_transfer_response(offer.id, True)
        self.assertIsNone(transferred.player("Aspas"))
        self.assertEqual(transferred.opponent_owner("Aspas").id, offer.team_id)
        self.assertEqual(next(p for p in transferred.opponent_owner("Aspas").players if p.name == "Aspas"), snapshot)
        self.assertEqual(transferred.money, state.money + offer.fee)
        self.assertIsNone(transferred.selected_team_id)
        self.assertNotIn("Aspas", transferred.roster)
        self.assertFalse(any("Aspas" in team.roster for team in transferred.teams))
        self.assertEqual(transferred.transfer_offers[-1].status, "accepted")
        self.assertEqual(transferred.advance_months().opponent_teams[0].roster, RIVALS[0])
        with self.assertRaises(SeasonSaveError):
            transferred.with_transfer_response(offer.id, True)

    def test_decline_keeps_player_and_allows_another_offer_next_month(self):
        state = self.offer_state()
        offer = state.transfer_offer("Aspas")
        declined = state.with_transfer_response(offer.id, False)
        self.assertIsNotNone(declined.player("Aspas"))
        self.assertEqual(declined.money, state.money)
        self.assertFalse(declined.pending_transfer_offers)
        self.assertEqual(declined.transfer_offers[-1].status, "rejected")
        following = declined.advance_months()
        self.assertNotEqual(following.transfer_offer("Aspas").id, offer.id)

    def test_threshold_is_strict_and_forces_transfer_immediately(self):
        state = self.offer_state().with_team_loyalty("Aspas", 30)
        self.assertIsNotNone(state.player("Aspas"))
        forced = state.with_team_loyalty("Aspas", 29.99)
        self.assertIsNone(forced.player("Aspas"))
        self.assertEqual(forced.transfer_offers[-1].status, "forced")
        self.assertEqual(forced.money, state.money + state.transfer_offer("Aspas").fee)

    def test_match_loss_and_monthly_decay_force_pending_transfers(self):
        state = self.offer_state().with_team_loyalty("Aspas", 30)
        lost = state.with_rated_result("loss", state.club_id, state.opponent_teams[0].id, 0, 1)
        self.assertIsNone(lost.player("Aspas"))
        self.assertEqual(lost.transfer_offers[-1].status, "forced")
        month = state.advance_months()
        self.assertIsNone(month.player("Aspas"))
        self.assertEqual(month.transfer_offers[-1].status, "forced")

    def test_new_offer_already_below_threshold_forces_transfer(self):
        state = new_season(OWN)
        state = replace(state, opponent_teams=tuple(replace(c, regular_members=c.members) for c in state.opponent_teams))
        state = state.with_scouted_player("Aspas", "year1")
        state = state.with_team_loyalty("Aspas", 29)
        month = state.advance_months()
        self.assertIsNone(month.player("Aspas"))
        self.assertEqual(month.transfer_offers[-1].status, "forced")

    def test_declined_offer_still_forces_transfer_when_loyalty_falls(self):
        state = self.offer_state().with_team_loyalty("Aspas", 30)
        offer = state.transfer_offer("Aspas")
        state = state.with_transfer_response(offer.id, False)
        self.assertFalse(state.pending_transfer_offers)
        self.assertIsNotNone(state.player("Aspas"))
        forced = state.with_team_loyalty("Aspas", 29.9)
        self.assertIsNone(forced.player("Aspas"))
        self.assertEqual(forced.transfer_offers[-1].status, "forced")
        self.assertEqual(forced.money, state.money + offer.fee)

    def test_daily_and_bulk_advances_have_identical_results(self):
        state = self.state()
        bulk = state.advance_months(3)
        daily = state
        for _ in range((bulk.date - state.date).days):
            daily = daily.advance_days()
        self.assertEqual(daily, bulk)

    def test_import_preserves_mixed_lineups_pending_offers_and_returns(self):
        state = self.state()
        self.assertEqual(self.store.import_season_teams(state), state)
        state = self.offer_state()
        self.assertEqual(self.store.import_season_teams(state), state)

    def test_invalid_offer_data_does_not_overwrite_save(self):
        state = self.offer_state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        for changes in ({"fee": -1}, {"team_id": "missing"}, {"player_name": "unknown"}, {"status": "invalid"}):
            with self.subTest(changes=changes), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, transfer_offers=(replace(state.transfer_offers[0], **changes),)))
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_releasing_player_cancels_offer(self):
        state = self.offer_state().without_player("Aspas")
        self.assertFalse(state.pending_transfer_offers)
        self.assertEqual(state.transfer_offers[-1].status, "cancelled")

    def test_version_fourteen_keeps_rosters_and_enables_future_offers(self):
        state = new_season(OWN).with_scouted_player("Aspas", "year1")
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 14
        data.pop("transfer_offers")
        for club in data["opponent_teams"]:
            for field in ("regular_members", "regular_igl", "regular_carrier"):
                club.pop(field)
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.owned_players, state.owned_players)
        self.assertEqual(loaded.opponent_teams[0].players, state.opponent_teams[0].players)
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertIsNotNone(loaded.advance_months().transfer_offer("Aspas"))

    def test_tournament_snapshots_survive_forced_transfer_and_defer_npc_returns(self):
        cup = dict(id="cup", name="Cup", start_date="2026-03-01", team_count=2,
                   format="single_elimination", prizes={}, normal_maps_to_win=1)
        with patch.object(calendar, "TOURNAMENTS", [cup]):
            state = self.offer_state()
        state = state.with_roster(("Aspas", *OWN[:4])).with_confirmed_team()
        state = state.with_selected_team(state.teams[-1].id)
        state = state.with_tournament_entry("cup", state.selected_team_id)
        snapshot = state.tournament("cup").entrants
        forced = state.with_team_loyalty("Aspas", 29)
        self.assertIsNone(forced.player("Aspas"))
        self.assertEqual(forced.tournament("cup").entrants, snapshot)
        forced.validate()

        with patch.object(calendar, "TOURNAMENTS", [cup]):
            mixed = self.state()
        mixed = mixed.with_tournament_entry("cup", mixed.teams[0].id)
        snapshot = mixed.tournament("cup").entrants
        month = mixed.advance_months()
        participating_ids = {t.id for t in snapshot}
        self.assertEqual(tuple(c.members for c in month.opponent_teams if c.id in participating_ids),
                         tuple(c.members for c in mixed.opponent_teams if c.id in participating_ids))
        self.assertEqual(month.tournament("cup").entrants, snapshot)

    def test_home_banner_contract_mark_and_response_buttons(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = self.offer_state()
        self.store.save(state)
        app = RealtimeSeasonApp(root, self.store, state)
        self.assertEqual(app.transfer_banner.winfo_manager(), "pack")
        app.transfer_banner.invoke()
        self.assertEqual(app.current_screen, "contracts")
        self.assertEqual(app.contracts_players.selection(), ("Aspas",))
        self.assertIn("incoming_offer", app.contracts_players.item("Aspas", "tags"))
        self.assertIn("●", app.contracts_players.item("Aspas", "values")[0])
        self.assertIn("Rival0", app.incoming_offer_summary.get())
        with patch.object(type(app), "match_running", new_callable=unittest.mock.PropertyMock, return_value=True):
            app.refresh_offer("contracts")
            self.assertEqual(str(app.accept_transfer_button.cget("state")), "disabled")
        app.refresh_offer("contracts")
        app.accept_transfer_button.invoke()
        self.assertIsNone(app.state.player("Aspas"))
        self.assertEqual(app.transfer_banner.winfo_manager(), "")
        self.assertEqual(self.store.load_or_create(), app.state)


class ConfiguredWorldTests(unittest.TestCase):
    def test_full_world_starts_unique_and_restores_regular_members(self):
        with patch.object(calendar, "TOURNAMENTS", []):
            for seed in range(5):
                with patch("realtime_season.with_randomized_clubs", side_effect=lambda s: with_randomized_clubs(s, Random(seed))):
                    pending = new_season()
                    state = pending.with_initial_selection(tuple(p.name for p in pending.starter_candidates[:5]))
                state.validate()
                month = state.advance_months()
                self.assertTrue(all(club.members == club.regular_members for club in month.opponent_teams))


if __name__ == "__main__":
    unittest.main()
