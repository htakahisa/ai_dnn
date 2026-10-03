"""Ratings, monthly sponsorship, paid transfers, persistence and Tk workflows."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config
import realtime_season_teams
from character_stats import get_by_name
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season_competitions import SeriesScore, next_match
from season_ratings import SeasonRating, series_ratings
from season_scrim import build_scrim_request


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex", "Sato")
CUP = dict(id="cup", name="Cup", start_date="2026-01-02", end_date="2026-01-03",
           team_count=2, format="single_elimination", prizes={}, normal_maps_to_win=1,
           lower_final_maps_to_win=1, grand_final_maps_to_win=1)


class SeasonEconomyTest(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5) for name, p in character_stats.CHARACTER_TABLE.items()}
        context = patch.dict(character_stats.CHARACTER_TABLE, fixture)
        context.start()
        self.addCleanup(context.stop)
        self.club = dict(name="Rival", players=list(RIVAL), igl="Aspas", carrier="Aspas", transfer_multiplier=12)
        for module, key, value in ((realtime_season_config, "INITIAL_OWNED_PLAYERS", OWN),
                                  (realtime_season_teams, "SEASON_TEAMS", [self.club]),
                                  (calendar, "START_DATE", "2026-01-01"), (calendar, "TOURNAMENTS", [])):
            context = patch.object(module, key, value)
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        state = new_season().with_initial_selection(OWN).with_roster(OWN).with_confirmed_team()
        return state.with_selected_team(state.teams[0].id)

    def test_exact_rating_parity_with_competition_manager(self):
        from run_competition_manager import SeriesResult, TeamRatingStore
        for a, b in ((1500, 1500), (2000, 1500), (100, 0), (0, 0)):
            for wins1, wins2 in ((1, 0), (2, 1), (0, 3), (2, 3)):
                with self.subTest(ratings=(a, b), score=(wins1, wins2)):
                    store = TeamRatingStore.__new__(TeamRatingStore)
                    store.ratings = {"A": a, "B": b}
                    store.history = []
                    store.default_rating = 1500
                    store.save = Mock()
                    result = SeriesResult("A", "B", max(wins1, wins2), wins1, wins2,
                                          "A" if wins1 > wins2 else "B", "B" if wins1 > wins2 else "A", [])
                    store.update_series(result, "test", "test")
                    self.assertEqual(series_ratings(a, b, wins1, wins2), (store.get("A"), store.get("B")))

    def test_ratings_are_local_persisted_and_duplicate_results_are_ignored(self):
        state = self.state()
        own, rival = state.teams[0], state.opponent_teams[0]
        self.assertEqual(len(state.rating_ranking), 2)
        won = state.with_rated_result("scrim:1", own.id, rival.id, 1, 0)
        self.assertEqual(won.rating(own.id), 1532)
        self.assertEqual(won.rating(rival.id), 1468)
        self.assertEqual(won.with_rated_result("scrim:1", own.id, rival.id, 1, 0), won)
        self.store.save(won)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, won)
        self.assertEqual(loaded.with_rated_result("scrim:1", own.id, rival.id, 1, 0), loaded)
        self.assertEqual(self.state().rating(self.state().opponent_teams[0].id), 1500)
        renamed = won.with_editing_team(own.id).with_team_name("Renamed").with_confirmed_team()
        self.assertEqual(renamed.rating(own.id), 1532)
        self.assertEqual(next(r.team_name for r in renamed.rating_ranking if r.team_id == own.id), "Renamed")

    def test_tournament_results_update_both_ratings_once(self):
        with patch.object(calendar, "TOURNAMENTS", [CUP]):
            state = self.state().advance_days()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        score = SeriesScore(match.id, match.left, match.right,
                            int(match.left == state.selected_team_id), int(match.right == state.selected_team_id))
        with self.assertRaisesRegex(SeasonSaveError, "大会参加中"):
            state.with_scouted_player("Aspas", "year1")
        # Reserves are not in the event snapshot, so their affiliation may change.
        reserved = state.with_scouted_player("Sato", "year1")
        self.assertIsNone(reserved.opponent_owner("Sato"))
        won = state.with_tournament_result("cup", score)
        self.assertEqual(won.rating(state.selected_team_id), 1532)
        self.assertEqual(won.rating(state.opponent_teams[0].id), 1468)
        self.assertEqual(len(won.rated_results), 1)
        self.store.save(won)
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create().with_tournament_result("cup", score)

    def test_zero_trait_and_free_transfer_multiplier_still_obey_contract_rules(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Sato": replace(get_by_name("Sato"), loyalty=0)}):
            state = self.state()
        state = replace(state, opponent_teams=(replace(state.opponent_teams[0], transfer_multiplier=0),))
        self.assertEqual(state.transfer_fee("Sato"), 0)
        with self.assertRaisesRegex(SeasonSaveError, "短期契約"):
            state.with_scouted_player("Sato", "year3")
        self.assertEqual(state.with_scouted_player("Sato", "short", 1).money, state.money)

    def test_monthly_income_and_payroll_share_a_single_calendar_boundary(self):
        state = self.state()
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)
        self.assertEqual(state.advance_days(30).money, state.money)
        feb = state.advance_days(31)
        self.assertEqual(feb.money, 17_000_000)
        self.assertEqual(feb.advance_days().money, feb.money)
        bulk = state.advance_months(3)
        daily = state
        for _ in range((bulk.date - state.date).days):
            daily = daily.advance_days()
        self.assertEqual(bulk, daily)
        self.assertEqual(bulk.money, 31_000_000)
        self.store.save(feb)
        self.assertEqual(self.store.load_or_create().advance_days().money, feb.money)
        disabled = state.with_sponsor_contract(False).advance_months()
        self.assertEqual(disabled.money, 9_500_000)
        self.assertEqual(new_season().monthly_sponsor_income, 0)

    def test_sponsor_uses_one_selected_team_and_current_rate(self):
        state = self.state()
        own = state.teams[0]
        state = state.with_rated_result("win", own.id, state.opponent_teams[0].id, 1, 0)
        self.assertEqual(state.monthly_sponsor_income, 7_660_000)
        names = tuple(p.name for p in state.lft_players[:5])
        state = state.with_added_players(names).with_new_team().with_roster(names).with_confirmed_team()
        self.assertEqual(state.monthly_sponsor_income, 7_660_000)
        state = state.with_selected_team(state.teams[1].id)
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)
        state = replace(state, ratings=tuple(replace(r, value=1500.00019) if r.team_id == state.selected_team_id else r for r in state.ratings))
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)

    def test_transfer_charges_fee_only_and_preserves_snapshot_and_unique_affiliation(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Aspas": replace(get_by_name("Aspas"), monthly_salary=123_457)}):
            state = self.state()
        state = replace(state, opponent_teams=(replace(state.opponent_teams[0], transfer_multiplier=1.25),))
        fee = state.transfer_fee("Aspas")
        self.assertEqual(fee, 154_322)
        required = fee + 123_457 * 12
        state = replace(state, money=required)
        before_player = state.opponent_teams[0].players[0]
        with self.assertRaises(SeasonSaveError):
            replace(state, money=required - 1).with_scouted_player("Aspas", "year1")
        with patch.dict(character_stats.CHARACTER_TABLE, {"Aspas": replace(get_by_name("Aspas"), monthly_salary=999_999)}):
            signed = state.with_scouted_player("Aspas", "year1")
        self.assertEqual(signed.money, required - fee)
        self.assertEqual(signed.player("Aspas"), before_player)
        self.assertIsNone(signed.opponent_owner("Aspas"))
        self.assertEqual(signed.opponent_teams[0].roster, RIVAL[1:])
        self.assertIn(signed.opponent_teams[0].effective_igl, RIVAL[1:])
        self.assertIn(signed.opponent_teams[0].effective_carrier, RIVAL[1:])
        self.store.save(signed)
        self.assertEqual(self.store.load_or_create(), signed)
        self.club["transfer_multiplier"] = 20
        imported = self.store.import_season_teams(signed)
        self.assertIsNone(imported.opponent_owner("Aspas"))
        self.assertEqual(imported.opponent_teams[0].transfer_multiplier, 20)
        self.assertEqual(imported.player("Aspas"), before_player)

    def test_reserve_transfers_lft_and_owned_affiliations(self):
        state = self.state()
        signed = state.with_scouted_player("Sato", "short", 1)
        self.assertEqual(signed.money, state.money - 1_200_000)
        self.assertEqual(signed.opponent_teams[0].roster, RIVAL[:5])
        self.assertEqual(signed.opponent_teams[0].igl, "Aspas")
        self.assertEqual(state.player_affiliation("Leo"), state.teams[0].name)
        self.assertEqual(state.player_affiliation("Aspas"), "Rival")
        self.assertEqual(state.player_affiliation("Meiy"), "LFT")
        self.assertEqual(state.with_scouted_player("Meiy", "year1").money, state.money)
        with self.assertRaises(SeasonSaveError):
            state.with_scouted_player("Leo", "year1")

    def test_depleted_club_is_saved_but_cannot_play_until_replenished(self):
        state = self.state().with_scouted_player("Sato", "short", 1).with_scouted_player("Aspas", "short", 1)
        self.assertEqual(len(state.opponent_teams[0].players), 4)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        with self.assertRaisesRegex(SeasonSaveError, "5人未満"):
            build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id)
        replacement = state.lft_players[0].name
        self.club["players"].append(replacement)
        replenished = self.store.import_season_teams(state)
        self.assertEqual(len(replenished.opponent_teams[0].players), 5)
        build_scrim_request(replenished, replenished.selected_team_id, replenished.opponent_teams[0].id)

    def test_invalid_multipliers_and_new_save_fields_are_rejected_atomically(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        for value in (-1, True, float("inf"), "12"):
            with self.subTest(value=value), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, opponent_teams=(replace(state.opponent_teams[0], transfer_multiplier=value),)))
            self.assertEqual(self.store.path.read_bytes(), before)
        for bad in (replace(state, sponsor_active=1), replace(state, rated_results=("x", "x")),
                    replace(state, ratings=(replace(state.ratings[0], value=float("nan")),)),
                    replace(state, transferred_players=("Aspas", "Aspas"))):
            with self.assertRaises(SeasonSaveError):
                self.store.save(bad)
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_missing_saved_rating_is_not_silently_reset(self):
        state = self.state()
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["ratings"] = []
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        with self.assertRaisesRegex(SeasonSaveError, "レーティング"):
            self.store.load_or_create()
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_version_seven_adds_local_ratings_without_resetting_money_loyalty_or_contracts(self):
        state = self.state()
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 7
        for key in ("ratings", "rated_results", "sponsor_active", "transferred_players"):
            data.pop(key)
        data["opponent_teams"][0].pop("transfer_multiplier")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(self.store.path.read_bytes(), before)


class SeasonEconomyScreenTest(SeasonEconomyTest):
    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        state = self.state()
        self.store.save(state)
        self.app = RealtimeSeasonApp(self.root, self.store, state)

    def test_scout_filter_membership_fee_and_transfer_workflow(self):
        app = self.app
        app.show_screen("scout")
        self.assertFalse(app.scout_players.exists("Sato"))
        app.scout_filter.set("全選手")
        self.assertTrue(app.scout_players.exists("Sato"))
        self.assertEqual(app.scout_players.item("Sato", "values")[-2:], ("Rival", "1,200,000"))
        app.scout_players.selection_set("Sato")
        app.refresh_offer("scout")
        self.assertIn("2,400,000", app.offer_summary["scout"].get())
        app.sign_selected_contract("scout")
        self.assertIsNotNone(app.state.player("Sato"))
        self.assertEqual(app.state.money, 8_800_000)
        app.scout_players.selection_set("Sato")
        app.refresh_offer("scout")
        self.assertEqual(str(app.offer_buttons["scout"]["state"]), "disabled")
        app.scout_filter.set("LFTのみ")
        self.assertFalse(app.scout_players.exists("Sato"))
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_rating_screen_sponsor_toggle_and_scrim_poll_are_saved(self):
        app = self.app
        app.show_screen("ratings")
        self.assertIn("7,500,000", app.sponsor_summary.get())
        app.sponsor_enabled.set(False)
        app.change_sponsor_contract()
        self.assertEqual(app.state.monthly_sponsor_income, 0)
        self.assertFalse(self.store.load_or_create().sponsor_active)
        app.sponsor_enabled.set(True)
        app.change_sponsor_contract()
        own, rival = app.state.teams[0], app.state.opponent_teams[0]
        result = dict(status="completed", own_team=own.name, opponent_team=rival.name,
                      own_score=13, opponent_score=8, winner=own.name)
        app.scrim_job = SimpleNamespace(poll=lambda: result)
        app._scrim_rating_context = ("scrim:example", own.id, rival.id)
        app.poll_scrim()
        self.assertEqual(app.state.rating(own.id), 1532)
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertIn("7,660,000", app.home_summary.get())

    def test_new_screens_fit_default_height(self):
        app = self.app
        for screen, host in (("home", app.home_host), ("ratings", app.ratings_host),
                             ("scout", app.scout_host), ("contracts", app.contracts_host)):
            app.show_screen(screen)
            if screen in ("scout", "contracts"):
                tree = getattr(app, f"{screen}_players")
                tree.selection_set(tree.get_children()[0])
                app.refresh_offer(screen)
            self.root.update_idletasks()
            with self.subTest(screen=screen):
                self.assertLessEqual(host.winfo_reqheight(), 800)


def load_tests(loader, tests, pattern):
    suite = loader.loadTestsFromTestCase(SeasonEconomyTest)
    suite.addTest(SeasonEconomyScreenTest("test_scout_filter_membership_fee_and_transfer_workflow"))
    suite.addTest(SeasonEconomyScreenTest("test_rating_screen_sponsor_toggle_and_scrim_poll_are_saved"))
    suite.addTest(SeasonEconomyScreenTest("test_new_screens_fit_default_height"))
    return suite
