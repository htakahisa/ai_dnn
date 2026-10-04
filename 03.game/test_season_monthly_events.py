"""Month-boundary NPC signings, renewals, departures and saved event UI."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_config
import realtime_season_teams
import realtime_season_competitions as calendar
import realtime_season_rival_economy as rival_economy
from character_stats import get_by_name
from realtime_season import SeasonClub, SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season_monthly_events import choose_recruit, process_monthly_events
from season_scrim import build_scrim_request


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = (("Aspas", "valyn", "trent", "leaf", "tex"), ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"))


class MonthlyEventsTest(unittest.TestCase):
    def setUp(self):
        self.config = [dict(name=f"Rival{i}", players=list(names)) for i, names in enumerate(RIVALS)]
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5) for name, p in character_stats.CHARACTER_TABLE.items()}
        context = patch.dict(character_stats.CHARACTER_TABLE, fixture)
        context.start()
        self.addCleanup(context.stop)
        for module, field, value in ((realtime_season_config, "INITIAL_OWNED_PLAYERS", OWN),
                                     (rival_economy, "NON_REGULAR_OFFER_CHANCE", 0),
                                     (realtime_season_teams, "SEASON_TEAMS", self.config),
                                     (calendar, "START_DATE", "2026-01-01"), (calendar, "TOURNAMENTS", [])):
            context = patch.object(module, field, value)
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        # These tests exercise vacancy filling with a fixed NPC lineup.
        with patch("realtime_season.with_randomized_clubs", side_effect=lambda state: state):
            state = new_season().with_initial_selection(OWN).with_roster(OWN).with_confirmed_team()
        return replace(state.with_selected_team(state.teams[0].id), money=10_000_000)

    def test_recruits_missing_role_at_boundary_and_can_scrim_again(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        old = state.opponent_teams[0]
        self.assertEqual(len(old.players), 4)
        self.assertEqual(state.advance_days(29).opponent_teams[0], old)
        month = state.advance_days(30)
        club = month.opponent_teams[0]
        self.assertEqual(len(club.players), 5)
        self.assertEqual(club.players[-1].role, get_by_name("Aspas").role)
        player = club.players[-1]
        self.assertNotEqual(player.name, "Aspas")
        self.assertNotIn(player.name, {p.name for p in month.lft_players})
        contract = next(c for c in club.contracts if c.player_name == player.name)
        self.assertEqual((contract.kind, contract.start_month, contract.duration_months), ("year1", 1, 12))
        self.assertEqual(contract.monthly_salary, player.monthly_salary)
        self.assertEqual([e.kind for e in month.monthly_events], ["recruitment", "month_completed"])
        self.assertEqual(month.money, state.money + state.monthly_sponsor_income - sum(c.monthly_salary for c in state.contracts))
        build_scrim_request(month, month.selected_team_id, club.id)

    def test_multiple_clubs_and_slots_share_a_unique_lft_pool(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        state = state.with_scouted_player("valyn", "short", 1).with_scouted_player("Demon1", "short", 1)
        month = state.advance_months()
        names = [p.name for p in month.owned_players] + [p.name for c in month.opponent_teams for p in c.players]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([len(c.players) for c in month.opponent_teams], [5, 5])
        self.assertEqual(sum(e.kind == "recruitment" for e in month.monthly_events), 3)

    def test_selection_prefers_role_then_suitable_strength(self):
        base = get_by_name("Leo")
        squad = tuple(replace(base, name=f"P{i}", role=role) for i, role in enumerate(("シーカー", "フラッシュ", "エンジニア", "スモーカー")))
        club = SeasonClub("test", "Test", squad, preferred_roles=("シーカー", "フラッシュ", "エンジニア", "スモーカー", "タイガー"))
        close = replace(base, name="Close", role="タイガー")
        much_stronger = replace(close, name="Star", iq=close.iq + 300)
        wrong = replace(close, name="Wrong", role="スモーカー")
        self.assertEqual(choose_recruit(club, (wrong, much_stronger, close), 1500), close)
        self.assertEqual(choose_recruit(club, (wrong,), 1500), wrong)
        self.assertIsNone(choose_recruit(club, (), 1500))

    def test_no_lft_records_unfilled_and_keeps_save_valid(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        with patch("season_monthly_events.all_characters", return_value=list(state.owned_players)):
            month = state.advance_months()
        self.assertEqual(len(month.opponent_teams[0].players), 4)
        self.assertEqual([e.kind for e in month.monthly_events], ["recruitment_unfilled", "month_completed"])
        self.store.save(month)
        self.assertEqual(self.store.load_or_create(), month)

    def test_day_bulk_restart_and_repeat_processing_do_not_duplicate_events(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        bulk = state.advance_months(3)
        daily = state
        for _ in range((bulk.date - state.date).days):
            daily = daily.advance_days()
        self.assertEqual(daily, bulk)
        self.assertEqual(sum(e.kind == "recruitment" for e in bulk.monthly_events), 1)
        self.assertEqual(process_monthly_events(bulk), bulk)
        self.store.save(bulk)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, bulk)
        self.assertEqual(loaded.advance_days().monthly_events, bulk.monthly_events)

    def test_zero_loyalty_trait_uses_short_contract(self):
        state = self.state().with_scouted_player("Aspas", "short", 1)
        candidate = replace(get_by_name("Meiy"), loyalty=0)
        with patch("season_monthly_events.all_characters", return_value=[candidate]):
            month = state.advance_months()
        contract = next(c for c in month.opponent_teams[0].contracts if c.player_name == candidate.name)
        self.assertEqual((contract.kind, contract.duration_months), ("short", 6))

    def test_auto_contract_renewal_and_departure_follow_loyalty(self):
        state = self.state()
        club = state.opponent_teams[0]
        club = replace(club, contracts=tuple(replace(c, kind="short", duration_months=1,
                               team_loyalty=0 if c.player_name == "Aspas" else 17) if c.player_name in ("Aspas", "valyn") else c
                                            for c in club.contracts))
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        month = state.advance_months()
        self.assertNotIn("Aspas", month.opponent_teams[0].members)
        renewal = next(c for c in month.opponent_teams[0].contracts if c.player_name == "valyn")
        self.assertEqual(renewal.start_month, 1)
        self.assertEqual(renewal.team_loyalty, 16.5)
        self.assertEqual({e.kind for e in month.monthly_events}, {"departure", "renewal", "recruitment", "month_completed"})
        imported = self.store.import_season_teams(month)
        self.assertNotIn("Aspas", imported.opponent_teams[0].members)
        self.assertEqual(imported.opponent_teams[0].members, month.opponent_teams[0].members)

    def test_registered_tournament_members_are_not_removed_at_month_boundary(self):
        cup = dict(id="cup", name="Cup", start_date="2026-02-02", end_date="2026-02-04", team_count=2,
                   format="single_elimination", prizes={}, normal_maps_to_win=1, lower_final_maps_to_win=1,
                   grand_final_maps_to_win=1)
        with patch.object(calendar, "TOURNAMENTS", [cup]):
            state = self.state()
        club = state.opponent_teams[0]
        club = replace(club, contracts=tuple(replace(c, kind="short", duration_months=1, team_loyalty=0)
                                             if c.player_name == "Aspas" else c for c in club.contracts))
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        state = state.with_tournament_entry("cup", state.selected_team_id)
        month = state.advance_months()
        self.assertIn("Aspas", month.opponent_teams[0].members)
        self.assertFalse(any(e.kind == "departure" and e.player_name == "Aspas" for e in month.monthly_events))
        self.assertEqual(month.tournament("cup").entrants, state.tournament("cup").entrants)

    def test_maximum_trait_preserves_npc_loyalty_on_renewal(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Aspas": replace(get_by_name("Aspas"), loyalty=10)}):
            state = self.state()
        club = state.opponent_teams[0]
        club = replace(club, contracts=tuple(replace(c, kind="short", duration_months=1, team_loyalty=17)
                                             if c.player_name == "Aspas" else c for c in club.contracts))
        month = replace(state, opponent_teams=(club, state.opponent_teams[1])).advance_months()
        self.assertEqual(next(c.team_loyalty for c in month.opponent_teams[0].contracts if c.player_name == "Aspas"), 17)

    def test_import_preserves_autorecruits_contracts_and_rating(self):
        state = self.state().with_scouted_player("Aspas", "short", 1).advance_months()
        self.config[0]["transfer_multiplier"] = 20
        imported = self.store.import_season_teams(state)
        self.assertEqual(imported.opponent_teams[0].players, state.opponent_teams[0].players)
        self.assertEqual(imported.opponent_teams[0].contracts, state.opponent_teams[0].contracts)
        self.assertEqual(imported.opponent_teams[0].transfer_multiplier, 20)
        self.assertEqual(imported.ratings, state.ratings)

    def test_version_eight_migrates_without_processing_past_months(self):
        state = self.state().advance_months(2)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 8
        for key in ("monthly_events", "monthly_events_through"):
            data.pop(key)
        for club in data["opponent_teams"]:
            club.pop("contracts")
            club.pop("preferred_roles")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.game_month, 2)
        self.assertEqual(loaded.monthly_events_through, 2)
        self.assertEqual(loaded.monthly_events, ())
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(loaded.owned_players, state.owned_players)
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(loaded.advance_days().monthly_events, ())
        self.assertEqual(len(loaded.advance_months().monthly_events), 1)

    def test_corrupt_events_and_failed_save_do_not_overwrite_progress(self):
        state = self.state().advance_months()
        self.store.save(state)
        before = self.store.path.read_bytes()
        bad = replace(state, monthly_events=(replace(state.monthly_events[0], date="2099-01-01"),))
        with self.assertRaises(SeasonSaveError):
            self.store.save(bad)
        with patch("realtime_season.os.replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            self.store.save(state.advance_months())
        self.assertEqual(self.store.path.read_bytes(), before)


class MonthlyScreenTest(MonthlyEventsTest):
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        state = self.state().with_scouted_player("Aspas", "short", 1)
        self.store.save(state)
        self.app = RealtimeSeasonApp(self.root, self.store, state)

    def test_advance_from_contracts_shows_persisted_event_and_affiliation(self):
        app = self.app
        app.advance_game_month()
        event = next(e for e in app.state.monthly_events if e.kind == "recruitment")
        app.show_screen("monthly")
        self.assertTrue(app.monthly_table.exists(event.id))
        app.monthly_table.selection_set(event.id)
        app.preview_monthly_event()
        self.assertIn(event.player_name, app.monthly_detail.get())
        self.assertIn("月次イベント", app.status.get())
        self.assertEqual(self.store.load_or_create(), app.state)
        app.show_screen("scout")
        self.assertFalse(app.scout_players.exists(event.player_name))
        app.scout_filter.set("全選手")
        self.assertEqual(app.scout_players.item(event.player_name, "values")[-2], "Rival0")
        app.show_screen("monthly")
        self.root.update_idletasks()
        self.assertLessEqual(app.monthly_host.winfo_reqheight(), 800)

    def test_daily_home_advancement_runs_same_monthly_pipeline(self):
        app = self.app
        app.advance_calendar(31)
        self.assertEqual(app.state.monthly_events_through, 1)
        self.assertEqual(len(app.state.opponent_teams[0].players), 5)
        self.assertIn("月次イベント", app.status.get())


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(MonthlyEventsTest)
    suite.addTest(MonthlyScreenTest("test_advance_from_contracts_shows_persisted_event_and_affiliation"))
    suite.addTest(MonthlyScreenTest("test_daily_home_advancement_runs_same_monthly_pipeline"))
    return suite
