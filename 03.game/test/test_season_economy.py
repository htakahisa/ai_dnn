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
import realtime_season_world_levels
import realtime_season_rival_economy
from character_stats import get_by_name
from realtime_season import SeasonSaveError, SeasonStore, new_season
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import SeriesScore, next_match
from season.season_ratings import SeasonRating, series_ratings
from season.season_scrim import build_scrim_request


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
                                  (realtime_season_rival_economy, "NON_REGULAR_OFFER_CHANCE", 0),
                                  (realtime_season_teams, "SEASON_TEAMS", [self.club]),
                                  (calendar, "START_DATE", "2026-01-01"), (calendar, "TOURNAMENTS", []),
                                  (realtime_season_world_levels, "WORLD_LEVELS",
                                   [{"レベル": 1, "必要レート": 0, "敵倍率": 1, "スポンサー資金": 7_500_000}])):
            context = patch.object(module, key, value)
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        # Isolate economy calculations from the random starting lineups.
        with patch("realtime_season.with_randomized_clubs", side_effect=lambda state: state):
            state = new_season().with_initial_selection(OWN).with_roster(OWN).with_confirmed_team()
        return replace(state.with_selected_team(state.teams[0].id), money=10_000_000)

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
        self.assertEqual(won.contract("Leo").team_loyalty, 21)
        self.assertEqual(won.opponent_teams[0].contracts[-1].team_loyalty, 27)
        self.assertEqual(won.with_rated_result("scrim:1", own.id, rival.id, 1, 0), won)
        self.store.save(won)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, won)
        self.assertEqual(loaded.with_rated_result("scrim:1", own.id, rival.id, 1, 0), loaded)
        self.assertEqual(self.state().rating(self.state().opponent_teams[0].id), 1500)
        renamed = won.with_editing_team(own.id).with_team_name("Renamed").with_confirmed_team()
        self.assertEqual(renamed.rating(own.id), 1532)
        self.assertEqual(next(r.team_name for r in renamed.rating_ranking if r.team_id == renamed.club_id), "Renamed")

    def test_match_loyalty_uses_saved_traits_for_both_sides_and_includes_reserves(self):
        traits = {name: replace(get_by_name(name), loyalty=value)
                  for name, value in zip((*OWN, *RIVAL), (0, 1, 5, 9, 10, 0, 1, 5, 9, 10, 5))}
        with patch.dict(character_stats.CHARACTER_TABLE, traits):
            state = self.state().with_scouted_player("Meiy", "year1")
        # A second preset is still the same club and must not multiply changes.
        state = state.with_new_team().with_roster(OWN).with_confirmed_team()
        own_id, rival_id = state.teams[-1].id, state.opponent_teams[0].id
        for left_id, right_id in ((own_id, rival_id), (rival_id, own_id)):
            for own_won in (True, False):
                with self.subTest(left=left_id, own_won=own_won):
                    left_won = own_won if left_id == own_id else not own_won
                    result = state.with_rated_result("result", left_id, right_id,
                                                     2 if left_won else 1, 1 if left_won else 2)
                    for player in state.owned_players:
                        delta = (1 if own_won else -(10 - player.loyalty) / 10) if player.name in OWN else -3
                        self.assertAlmostEqual(result.contract(player.name).team_loyalty, state.contract(player.name).team_loyalty + delta)
                    club = result.opponent_teams[0]
                    for player, contract in zip(club.players, club.contracts):
                        delta = (-(10 - player.loyalty) / 10 if own_won else 1) if player.name in RIVAL[:5] else -3
                        self.assertAlmostEqual(contract.team_loyalty, state.team_loyalty(player.name, rival_id) + delta)
                    self.assertEqual(result.player("Leo").loyalty, 0)
                    self.assertEqual(result.money, state.money)
                    self.assertEqual(result.with_rated_result("result", left_id, right_id, 2, 1), result)

    def test_npc_match_changes_only_the_two_clubs(self):
        state = self.state()
        other_names = tuple(p.name for p in state.lft_players[:5])
        with patch.object(realtime_season_teams, "SEASON_TEAMS",
                          [self.club, dict(name="Other", players=other_names)]):
            state = self.store.import_season_teams(state)
        left, right = state.opponent_teams
        result = state.with_rated_result("npc", left.id, right.id, 0, 3)
        self.assertEqual(result.contracts, state.contracts)
        self.assertTrue(all(c.team_loyalty == (29.5 if c.player_name in RIVAL[:5] else 27)
                            for c in result.opponent_teams[0].contracts))
        self.assertTrue(all(c.team_loyalty == 31 for c in result.opponent_teams[1].contracts))

    def test_match_loyalty_preserves_inactive_contracts_and_controls_departure_and_renewal(self):
        # This scenario checks an existing long contract; starters now have
        # short contracts and would leave early when team loyalty reaches zero.
        state = new_season(OWN).with_team_loyalty("Leo", .5)
        rival_id = state.opponent_teams[0].id
        lost = state.with_rated_result("loss", state.club_id, rival_id, 0, 1)
        self.assertEqual(lost.contract("Leo").team_loyalty, 0)
        recovered = lost.with_rated_result("win", state.club_id, rival_id, 1, 0)
        self.assertEqual(recovered.contract("Leo").team_loyalty, 1)
        expired = lost.advance_months(12, pay_salaries=False)
        with self.assertRaisesRegex(SeasonSaveError, "忠誠"):
            expired.with_renewed_contract("Leo", "year1")
        after = expired.with_rated_result("after-expiry", expired.club_id, rival_id, 1, 0)
        self.assertEqual(after.contracts, expired.contracts)
        released = new_season(OWN).with_team_loyalty("Leo", 17).without_player("Leo")
        after = released.with_rated_result("after-release", released.club_id, rival_id, 1, 0)
        self.assertEqual(after.contract("Leo"), released.contract("Leo"))
        short = new_season(()).with_scouted_player("Leo", "short", 6).with_team_loyalty("Leo", .1)
        lost = short.with_rated_result("short-loss", short.club_id, short.opponent_teams[0].id, 0, 1)
        self.assertLess(lost.contract("Leo").team_loyalty, 0)
        self.assertIsNone(lost.advance_months(pay_salaries=False).player("Leo"))

    def test_tournament_results_update_both_ratings_once(self):
        with patch.object(calendar, "TOURNAMENTS", [CUP]):
            state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days()
        match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
        score = SeriesScore(match.id, match.left, match.right,
                            int(match.left == state.club_id), int(match.right == state.club_id))
        with self.assertRaisesRegex(SeasonSaveError, "出場中"):
            state.with_scouted_player("Aspas", "year1")
        # A scouting action spends a day, including signing an opponent's bench.
        with self.assertRaisesRegex(SeasonSaveError, "出場中"):
            state.with_scouted_player("Sato", "year1")
        won = state.with_tournament_result("cup", score)
        self.assertEqual(won.rating(state.selected_team_id), 1532)
        self.assertEqual(won.rating(state.opponent_teams[0].id), 1468)
        self.assertEqual(len(won.rated_results), 1)
        self.assertEqual(won.contract("Leo").team_loyalty, 21)
        self.assertTrue(all(c.team_loyalty == (29.5 if c.player_name in RIVAL[:5] else 27)
                            for c in won.opponent_teams[0].contracts))
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
        self.assertEqual(state.with_scouted_player("Sato", "short", 1).money, state.money - 300_000)

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

    def test_sponsor_is_world_level_amount_at_low_ratings_and_respects_disabled_state(self):
        base = self.state()
        for rating in (0, 400, 999.9999, 1000, 1000.00019, 1000.0002, 1500):
            with self.subTest(rating=rating):
                state = replace(base, ratings=tuple(replace(r, value=rating) if r.team_id == base.club_id else r for r in base.ratings))
                self.assertEqual(state.monthly_sponsor_income, 7_500_000)
                self.assertEqual(state.with_sponsor_contract(False).monthly_sponsor_income, 0)
        self.assertEqual(new_season().monthly_sponsor_income, 0)

    def test_low_rating_world_sponsor_is_paid_once_and_saved_alongside_payroll(self):
        state = self.state()
        state = replace(state, ratings=tuple(replace(r, value=400) if r.team_id == state.club_id else r for r in state.ratings))
        self.assertEqual(state.advance_days(30).money, state.money)
        feb = state.advance_days(31)
        self.assertEqual(feb.money, state.money + 7_500_000 - state.monthly_payroll)
        self.store.save(feb)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.advance_days().money, feb.money)
        self.assertEqual(loaded.monthly_sponsor_income, 7_500_000)
        self.assertEqual(state.advance_months(2), feb.advance_months())

    def test_sponsor_uses_actual_club_world_level_across_presets(self):
        state = self.state()
        own = state.teams[0]
        state = state.with_rated_result("win", own.id, state.opponent_teams[0].id, 1, 0)
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)
        names = tuple(p.name for p in state.lft_players[:5])
        state = state.with_added_players(names).with_new_team().with_roster(names).with_confirmed_team()
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)
        state = state.with_selected_team(state.teams[1].id)
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)
        state = replace(state, ratings=tuple(replace(r, value=1500.00019) if r.team_id == state.club_id else r for r in state.ratings))
        self.assertEqual(state.monthly_sponsor_income, 7_500_000)

    def test_transfer_charges_fee_and_signing_bonus_and_preserves_snapshot_and_affiliation(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Aspas": replace(get_by_name("Aspas"), monthly_salary=123_457)}):
            state = self.state()
        state = replace(state, opponent_teams=(replace(state.opponent_teams[0], transfer_multiplier=1.25),))
        fee = state.transfer_fee("Aspas")
        self.assertEqual(fee, 154_322)
        required = fee + 123_457 * 15
        state = replace(state, money=required)
        before_player = state.opponent_teams[0].players[0]
        with self.assertRaises(SeasonSaveError):
            replace(state, money=required - 1).with_scouted_player("Aspas", "year1")
        with patch.dict(character_stats.CHARACTER_TABLE, {"Aspas": replace(get_by_name("Aspas"), monthly_salary=999_999)}):
            signed = state.with_scouted_player("Aspas", "year1")
        self.assertEqual(signed.money, required - fee - 123_457 * 3)
        self.assertEqual(signed.player("Aspas"), replace(before_player, iq=before_player.iq + .1))
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
        self.assertEqual(imported.player("Aspas"), signed.player("Aspas"))

    def test_reserve_transfers_lft_and_owned_affiliations(self):
        state = self.state()
        signed = state.with_scouted_player("Sato", "short", 1)
        self.assertEqual(signed.money, state.money - 1_200_000 - 300_000)
        self.assertEqual(signed.opponent_teams[0].roster, RIVAL[:5])
        self.assertEqual(signed.opponent_teams[0].igl, "Aspas")
        self.assertEqual(state.player_affiliation("Leo"), state.team_name)
        self.assertEqual(state.player_affiliation("Aspas"), "Rival")
        self.assertEqual(state.player_affiliation("Meiy"), "LFT")
        self.assertEqual(state.with_scouted_player("Meiy", "year1").money, state.money - 300_000)
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
        self.assertIn("2,700,000", app.offer_summary["scout"].get())
        self.assertIn("契約金: 300,000", app.offer_summary["scout"].get())
        app.sign_selected_contract("scout")
        self.assertIsNotNone(app.state.player("Sato"))
        self.assertEqual(app.state.money, 8_500_000)
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
        self.assertEqual(app.state.contract("Leo").team_loyalty, 21)
        self.assertTrue(all(c.team_loyalty == (29.5 if c.player_name in RIVAL[:5] else 27)
                            for c in app.state.opponent_teams[0].contracts))
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertIn("7,500,000", app.home_summary.get())

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
