"""Contract affordability, LFT affiliation, save migration, and real Tk workflows."""

from dataclasses import fields, replace
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import character_stats
import realtime_season_config
import realtime_season_competitions
import realtime_season_teams
from character_stats import CharacterStats, get_by_name
from realtime_season import INITIAL_MONEY, SeasonSaveError, SeasonStore, contract_terms, new_season
from run_realtime_season import RealtimeSeasonApp
from season_scrim import build_scrim_request


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVAL = ("Aspas", "valyn", "trent", "leaf", "tex", "Sato")


def patch_economy_defaults(test):
    fixture = {name: replace(p, monthly_salary=100_000, loyalty=5) for name, p in character_stats.CHARACTER_TABLE.items()}
    context = patch.dict(character_stats.CHARACTER_TABLE, fixture)
    context.start()
    test.addCleanup(context.stop)
    # Contract-only tests must reach month boundaries regardless of the user's
    # configured tournament dates, where normal calendar advancement stops.
    for key, value in (("START_DATE", "2026-01-01"), ("TOURNAMENTS", [])):
        context = patch.object(realtime_season_competitions, key, value)
        context.start()
        test.addCleanup(context.stop)


class ContractRulesTest(unittest.TestCase):
    def setUp(self):
        patch_economy_defaults(self)
        for module, field, value in ((realtime_season_config, "INITIAL_OWNED_PLAYERS", []),
                                     (realtime_season_teams, "SEASON_TEAMS", [])):
            config = patch.object(module, field, value)
            config.start()
            self.addCleanup(config.stop)

    def test_initial_money_and_new_stats_are_last_and_backward_compatible(self):
        self.assertEqual(new_season(()).money, 10_000_000)
        self.assertEqual([f.name for f in fields(CharacterStats)][-4:],
                         ["monthly_salary", "loyalty", "research_level", "aim_lab_level"])
        old_definition = CharacterStats("Example", .3, .2, 100, .7, 100, "フラッシュ", 50, 5, 5)
        self.assertEqual(old_definition.monthly_salary, 100_000)
        self.assertEqual(old_definition.loyalty, 5)
        self.assertEqual((old_definition.research_level, old_definition.aim_lab_level), (0, 0))

    def test_lft_excludes_owned_players_and_opponent_reserves(self):
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "Rival", "players": RIVAL}]):
            state = new_season(OWN)
        names = {p.name for p in state.lft_players}
        self.assertTrue(names.isdisjoint((*OWN, *RIVAL)))
        self.assertIn("Meiy", names)
        for name in ("Leo",):
            with self.subTest(name=name), self.assertRaises(SeasonSaveError):
                state.with_scouted_player(name, "year1")
        for name in ("Aspas", "Sato"):
            signed = state.with_scouted_player(name, "year1")
            self.assertIsNone(signed.opponent_owner(name))
            self.assertEqual(signed.money, state.money - state.transfer_fee(name) - signed.contract(name).monthly_salary * 3)

    def test_every_term_pays_three_month_signing_bonus_and_checks_remaining_funds(self):
        player = replace(get_by_name("Meiy"), monthly_salary=123_457)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Meiy": player}):
            for kind, duration, wage in (("short", 3, 123_457), ("year1", 12, 123_457),
                                         ("year2", 24, 111_112), ("year3", 36, 98_766)):
                with self.subTest(kind=kind):
                    funds = wage * (duration + 3)
                    state = replace(new_season(()), money=funds)
                    signed = state.with_scouted_player("Meiy", kind, 3)
                    self.assertEqual(signed.money, funds - wage * 3)
                    self.assertEqual(signed.contract("Meiy").monthly_salary, wage)
                    self.assertEqual(signed.contract("Meiy").duration_months, duration)
                    self.assertEqual(signed.contract("Meiy").team_loyalty, 50)
                    self.assertNotIn("Meiy", {p.name for p in signed.lft_players})
                    with self.assertRaises(SeasonSaveError):
                        replace(state, money=funds - 1).with_scouted_player("Meiy", kind, 3)
                    self.assertIsNone(state.player("Meiy"))

    def test_short_periods_one_through_six_and_invalid_inputs(self):
        player = get_by_name("Leo")
        for month in range(1, 7):
            self.assertEqual(contract_terms(player, "short", month).months, month)
        for month in (0, 7, True, 1.5, "3"):
            with self.subTest(month=month), self.assertRaises(SeasonSaveError):
                contract_terms(player, "short", month)
        for kind in ("missing", None, []):
            with self.subTest(kind=kind), self.assertRaises(SeasonSaveError):
                contract_terms(player, kind)

    def test_renewal_requires_bonus_and_remaining_wages_and_cannot_charge_twice(self):
        expired = new_season(("Leo",)).advance_months(12, pay_salaries=False)
        for kind in ("short", "year1", "year2", "year3"):
            with self.subTest(kind=kind):
                terms = contract_terms(expired.player("Leo"), kind, 1)
                self.assertEqual(terms.signing_bonus, terms.monthly_salary * 3)
                ready = replace(expired, money=terms.total_required_funds)
                with self.assertRaises(SeasonSaveError):
                    replace(ready, money=ready.money - 1).with_renewed_contract("Leo", kind, 1)
                renewed = ready.with_renewed_contract("Leo", kind, 1)
                self.assertEqual(renewed.money, terms.required_funds)
                self.assertEqual(renewed.contract("Leo").team_loyalty, ready.contract("Leo").team_loyalty)
                with self.assertRaises(SeasonSaveError):
                    renewed.with_renewed_contract("Leo", kind, 1)
                self.assertEqual(ready.money, terms.total_required_funds)

    def test_monthly_payroll_and_expiry_use_simulation_months(self):
        state = new_season(()).with_scouted_player("Leo", "short", 2)
        state = state.with_scouted_player("Meiy", "year2")
        next_month = state.with_sponsor_contract(False).advance_months()
        payroll = state.contract("Leo").monthly_salary + state.contract("Meiy").monthly_salary
        self.assertEqual(next_month.money, state.money - payroll)
        self.assertEqual(next_month.game_month, 1)
        self.assertTrue(next_month.can_play("Leo"))
        expiry = next_month.advance_months()
        self.assertFalse(expiry.can_play("Leo"))
        self.assertIsNotNone(expiry.player("Leo"))
        self.assertNotIn("Leo", {p.name for p in expiry.lft_players})
        after_expiry = expiry.advance_months()
        self.assertEqual(expiry.money - after_expiry.money, state.contract("Meiy").monthly_salary)
        self.assertEqual(expiry.contract("Leo").elapsed(expiry.game_month), 2)
        self.assertEqual(expiry.contract("Leo").remaining(expiry.game_month), 0)

    def test_renewal_keeps_loyalty_and_discounts_base_salary_once(self):
        state = new_season(()).with_scouted_player("Leo", "year2").with_team_loyalty("Leo", 17)
        with self.assertRaises(SeasonSaveError):
            state.with_renewed_contract("Leo", "year3")
        state = state.advance_months(24, pay_salaries=False)
        renewed = state.with_renewed_contract("Leo", "year3")
        self.assertEqual(renewed.contract("Leo").monthly_salary, 80_000)
        self.assertEqual(renewed.contract("Leo").start_month, 24)
        self.assertEqual(renewed.contract("Leo").end_month, 60)
        self.assertEqual(renewed.contract("Leo").team_loyalty, state.contract("Leo").team_loyalty)
        self.assertEqual(renewed.contract("Leo").team_loyalty, 5)
        self.assertEqual(renewed.money, state.money - 240_000)
        self.assertEqual(len(renewed.owned_players), 1)

    def test_zero_or_negative_team_loyalty_blocks_renewal_but_preserves_long_contract(self):
        for loyalty in (0, -5):
            with self.subTest(loyalty=loyalty):
                state = new_season(()).with_scouted_player("Leo", "year1").with_team_loyalty("Leo", loyalty)
                self.assertTrue(state.advance_months().can_play("Leo"))
                expired = state.advance_months(12, pay_salaries=False)
                with self.assertRaisesRegex(SeasonSaveError, "忠誠"):
                    expired.with_renewed_contract("Leo", "year1")
                released = expired.without_player("Leo")
                with self.assertRaisesRegex(SeasonSaveError, "忠誠"):
                    released.with_scouted_player("Leo", "short", 1)

    def test_player_loyalty_trait_is_independent_of_team_loyalty(self):
        player = replace(get_by_name("Leo"), loyalty=0)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
            state = new_season(()).with_scouted_player("Leo", "short", 1)
        self.assertEqual(state.contract("Leo").team_loyalty, 50)
        state = state.advance_months()
        with self.assertRaisesRegex(SeasonSaveError, "短期契約"):
            state.with_renewed_contract("Leo", "year1")
        state = state.with_renewed_contract("Leo", "short", 6)
        state = state.with_team_loyalty("Leo", -1)
        self.assertEqual(state.player("Leo").loyalty, 0)

    def test_short_departure_releases_affiliation_and_repairs_saved_roster(self):
        state = new_season(OWN[1:]).with_scouted_player("Leo", "short", 6)
        state = state.with_roster(OWN).with_confirmed_team()
        state = state.with_selected_team(state.teams[0].id).with_team_loyalty("Leo", 0)
        next_month = state.advance_months()
        self.assertIsNone(next_month.player("Leo"))
        self.assertIn("Leo", {p.name for p in next_month.lft_players})
        self.assertEqual(next_month.contract("Leo").end_reason, "left")
        self.assertEqual(next_month.roster, OWN[1:])
        self.assertEqual(next_month.teams, ())
        self.assertIsNone(next_month.selected_team_id)
        self.assertIsNone(next_month.editing_team_id)
        self.assertEqual(next_month.money - state.money, state.monthly_sponsor_income - 400_000)
        next_month.validate()

    def test_zero_trait_rejects_every_long_contract_including_initial_setup(self):
        player = replace(get_by_name("Leo"), loyalty=0)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
            state = new_season(())
            for kind in ("year1", "year2", "year3"):
                with self.subTest(kind=kind), self.assertRaisesRegex(SeasonSaveError, "短期契約"):
                    state.with_scouted_player("Leo", kind)
            self.assertEqual(new_season(("Leo",)).contract("Leo").kind, "short")
            self.assertEqual(new_season(("Leo",)).contract("Leo").duration_months, 6)
            with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", OWN):
                selected = new_season().with_initial_selection(OWN)
            self.assertEqual(selected.contract("Leo").kind, "short")

    def test_trait_controls_monthly_loss_and_daily_advances_do_not_repeat_it(self):
        for trait, loss in ((0, 1), (1, .9), (5, .5), (9, .1), (10, 0)):
            with self.subTest(trait=trait):
                player = replace(get_by_name("Leo"), loyalty=trait)
                with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": player}):
                    state = new_season(()).with_scouted_player("Leo", "short", 6)
                before_boundary = state.advance_days(30)
                self.assertEqual(before_boundary.contract("Leo").team_loyalty, 50)
                crossed = before_boundary.advance_days()
                self.assertAlmostEqual(crossed.contract("Leo").team_loyalty, 50 - loss)
                self.assertEqual(crossed.advance_days(10).contract("Leo"), crossed.contract("Leo"))
                bulk = state.advance_months(3, pay_salaries=False)
                daily = state
                for _ in range((bulk.date - state.date).days):
                    daily = daily.advance_days(pay_salaries=False, stop_for_tournaments=False)
                self.assertEqual(bulk, daily)
                self.assertAlmostEqual(bulk.contract("Leo").team_loyalty, 50 - loss * 3)

    def test_maximum_trait_blocks_decreases_and_allows_increases(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(get_by_name("Leo"), loyalty=10)}):
            state = new_season(("Leo",))
        self.assertEqual(state.with_team_loyalty("Leo", -1), state)
        increased = state.with_team_loyalty("Leo", 75.123456789012)
        self.assertEqual(increased.advance_months(12, pay_salaries=False).contract("Leo").team_loyalty,
                         increased.contract("Leo").team_loyalty)

    def test_monthly_loss_can_trigger_departure_or_prevent_renewal(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(get_by_name("Leo"), loyalty=0)}):
            state = new_season(()).with_scouted_player("Leo", "short", 6).with_team_loyalty("Leo", .5)
        departed = state.advance_months(pay_salaries=False)
        self.assertIsNone(departed.player("Leo"))
        self.assertEqual(departed.contract("Leo").team_loyalty, -.5)
        long = new_season(()).with_scouted_player("Leo", "year1").with_team_loyalty("Leo", 1)
        self.assertTrue(long.advance_months(2, pay_salaries=False).can_play("Leo"))
        expired = long.advance_months(12, pay_salaries=False)
        with self.assertRaisesRegex(SeasonSaveError, "忠誠"):
            expired.with_renewed_contract("Leo", "year1")

    def test_expired_contract_stops_losing_loyalty(self):
        state = new_season(()).with_scouted_player("Leo", "short", 1)
        expired = state.advance_months(pay_salaries=False)
        self.assertEqual(expired.contract("Leo").team_loyalty, 49.5)
        later = expired.advance_months(3, pay_salaries=False)
        self.assertEqual(later.contract("Leo"), expired.contract("Leo"))

    def test_out_of_range_traits_are_rejected(self):
        for value in (-1, 10.1, 100, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(SeasonSaveError):
                contract_terms(replace(get_by_name("Leo"), loyalty=value), "short", 1)

    def test_expired_contract_cannot_start_scrim(self):
        with patch.object(realtime_season_teams, "SEASON_TEAMS", [{"name": "Rival", "players": RIVAL}]):
            state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        state = state.advance_months(12, pay_salaries=False)
        with self.assertRaisesRegex(SeasonSaveError, "契約が終了"):
            build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id)
        for player in OWN:
            state = state.with_renewed_contract(player, "year1")
        self.assertEqual(len(build_scrim_request(state, state.teams[0].id, state.opponent_teams[0].id)["own"]["players"]), 5)

    def test_payroll_can_produce_deficit_and_contracts_still_require_funds(self):
        state = replace(new_season(("Leo",)), money=0, sponsor_active=False)
        state = state.advance_months()
        self.assertEqual(state.money, -100_000)
        with self.assertRaises(SeasonSaveError):
            state.with_scouted_player("Meiy", "short", 1)
        state.validate()


class ContractPersistenceTest(unittest.TestCase):
    def setUp(self):
        patch_economy_defaults(self)
        config = patch.object(realtime_season_teams, "SEASON_TEAMS", [])
        config.start()
        self.addCleanup(config.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "save.json"
        self.store = SeasonStore(self.path)

    def test_money_contracts_expiry_and_loyalty_survive_restart(self):
        state = new_season(()).with_scouted_player("Leo", "short", 1).with_team_loyalty("Leo", -1)
        state = state.with_scouted_player("Meiy", "year3").advance_months(2)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, state)
        self.assertEqual(loaded.contract("Leo").end_reason, "left")
        self.assertEqual(loaded.contract("Meiy").remaining(loaded.game_month), 34)

    def test_old_version_three_migrates_money_and_stats_without_resetting_rosters(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 3
        for key in ("money", "game_month", "contracts"):
            data.pop(key)
        for player in data["owned_players"]:
            player.pop("monthly_salary")
            player.pop("loyalty")
        data["owned_players"][0]["iq"] = 180
        self.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.path.read_bytes()
        updated = replace(get_by_name("Leo"), monthly_salary=345_000, loyalty=7)
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": updated}):
            loaded = self.store.load_or_create()
        self.assertEqual(loaded.money, INITIAL_MONEY)
        self.assertEqual(loaded.contract("Leo").monthly_salary, 345_000)
        self.assertEqual(loaded.player("Leo").loyalty, 7)
        self.assertEqual(loaded.player("Leo").iq, 180)
        self.assertEqual(loaded.teams, state.teams)
        self.assertEqual(self.path.read_bytes(), original)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_invalid_money_and_contract_data_do_not_overwrite_save(self):
        state = new_season(("Leo",))
        self.store.save(state)
        original = self.path.read_bytes()
        c = state.contract("Leo")
        bad_states = [replace(state, money=True), replace(state, game_month=-1), replace(state, contracts=()),
                      replace(state, contracts=(c, c)), replace(state, contracts=(replace(c, duration_months=13),)),
                      replace(state, contracts=(replace(c, team_loyalty=float("nan")),)),
                      replace(state, owned_players=(replace(state.player("Leo"), monthly_salary=1.5),)),
                      replace(state, owned_players=(replace(state.player("Leo"), loyalty=11),))]
        for bad in bad_states:
            with self.subTest(bad=bad), self.assertRaises(SeasonSaveError):
                self.store.save(bad)
            self.assertEqual(self.path.read_bytes(), original)
        data = json.loads(original)
        data["contracts"] = []
        self.path.write_text(json.dumps(data), encoding="utf-8")
        corrupted = self.path.read_bytes()
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()
        self.assertEqual(self.path.read_bytes(), corrupted)

    def test_version_six_converts_owned_candidates_and_rival_reserves_once(self):
        with patch.object(realtime_season_config, "INITIAL_OWNED_PLAYERS", OWN), patch.object(
                realtime_season_teams, "SEASON_TEAMS", [{"name": "Rival", "players": RIVAL}]):
            state = new_season().with_initial_selection(OWN)
        self.store.save(state)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 6
        snapshots = [*data["owned_players"], *data["starter_candidates"],
                     *(p for club in data["opponent_teams"] for p in club["players"])]
        for player in snapshots:
            player["loyalty"] *= 10
        data["owned_players"][0]["loyalty"] = 100
        data["starter_candidates"][0]["loyalty"] = 100
        data["opponent_teams"][0]["players"][-1]["loyalty"] = 0
        self.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.player("Leo").loyalty, 10)
        self.assertEqual(loaded.starter_candidates[0].loyalty, 10)
        self.assertEqual(loaded.player("Boaster").loyalty, 5)
        self.assertEqual(loaded.opponent_teams[0].players[-1].loyalty, 0)
        self.assertEqual(loaded.contracts, state.contracts)
        self.assertEqual(loaded.money, state.money)
        self.assertEqual(self.path.read_bytes(), before)
        self.store.save(loaded)
        self.assertEqual(self.store.load_or_create(), loaded)

    def test_invalid_legacy_trait_preserves_file(self):
        self.store.save(new_season(("Leo",)))
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["version"] = 6
        for value in (101, -1, True, "50"):
            data["owned_players"][0]["loyalty"] = value
            self.path.write_text(json.dumps(data), encoding="utf-8")
            before = self.path.read_bytes()
            with self.subTest(value=value), self.assertRaises(SeasonSaveError):
                self.store.load_or_create()
            self.assertEqual(self.path.read_bytes(), before)


class ContractScreenTest(unittest.TestCase):
    def setUp(self):
        patch_economy_defaults(self)
        for module, key in ((realtime_season_config, "INITIAL_OWNED_PLAYERS"), (realtime_season_teams, "SEASON_TEAMS")):
            config = patch.object(module, key, [])
            config.start()
            self.addCleanup(config.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")
        self.store.save(new_season(()))
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = RealtimeSeasonApp(self.root, self.store, self.store.load_or_create())

    def select(self, screen, name):
        getattr(self.app, f"{screen}_players").selection_set(name)
        self.app.refresh_offer(screen)

    def test_signing_bonus_updates_scout_colors_and_charges_on_renewal(self):
        app = self.app
        app.commit(replace(app.state, money=1_200_000), "Only the wage reserve")
        app.show_screen("scout")
        self.select("scout", "Leo")
        self.assertIn("contract_unavailable", app.scout_players.item("Leo", "tags"))
        self.assertEqual(str(app.offer_buttons["scout"]["state"]), "disabled")
        before = app.state
        app.sign_selected_contract("scout")
        self.assertEqual(app.state, before)
        app.commit(replace(app.state, money=1_500_000), "Reserve plus signing bonus")
        self.assertIn("contract_available", app.scout_players.item("Leo", "tags"))
        self.assertIn("契約金: 300,000", app.offer_summary["scout"].get())
        app.sign_selected_contract("scout")
        self.assertEqual(app.state.money, 1_200_000)
        self.assertIn("契約金: 300,000", app.status.get())
        self.assertEqual(self.store.load_or_create().money, app.state.money)
        expired = app.state.advance_months(12, pay_salaries=False)
        app.commit(replace(expired, money=1_500_000), "Renewal funds")
        app.show_screen("contracts")
        self.select("contracts", "Leo")
        self.assertIn("契約金: 300,000", app.offer_summary["contracts"].get())
        app.sign_selected_contract("contracts")
        self.assertEqual(app.state.money, 1_200_000)
        before = app.state
        app.sign_selected_contract("contracts")
        self.assertEqual(app.state, before)
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_scout_to_inventory_and_contract_view_with_restart(self):
        app = self.app
        app.commit(app.state.with_sponsor_contract(False), "Payroll-only scenario")
        app.show_screen("scout")
        self.select("scout", "Leo")
        app.offer_kind["scout"].set("2年契約")
        app.refresh_offer("scout")
        self.assertIn("2,160,000", app.offer_summary["scout"].get())
        app.sign_selected_contract("scout")
        self.assertFalse(app.scout_players.exists("Leo"))
        self.assertEqual(app.state.money, INITIAL_MONEY - 270_000)
        self.assertEqual(app.state.contract("Leo").monthly_salary, 90_000)
        self.assertIn("9,730,000", app.home_summary.get())
        app.show_screen("contracts")
        self.assertEqual(app.contracts_players.item("Leo", "values")[1], "契約中")
        app.advance_game_month()
        self.assertEqual(app.state.money, INITIAL_MONEY - 270_000 - 90_000)
        self.assertEqual(app.contracts_players.item("Leo", "values")[4], "1 / 24月")
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_expiry_renewal_and_zero_loyalty_controls(self):
        app = self.app
        app.commit(app.state.with_scouted_player("Leo", "short", 1), "signed")
        app.show_screen("contracts")
        self.select("contracts", "Leo")
        self.assertEqual(str(app.offer_buttons["contracts"]["state"]), "disabled")
        app.advance_game_month()
        self.select("contracts", "Leo")
        self.assertEqual(str(app.offer_buttons["contracts"]["state"]), "normal")
        app.commit(app.state.with_team_loyalty("Leo", 0), "loyalty")
        self.select("contracts", "Leo")
        self.assertEqual(str(app.offer_buttons["contracts"]["state"]), "disabled")
        self.assertIn("忠誠", app.offer_summary["contracts"].get())
        before = app.state
        app.sign_selected_contract("contracts")
        self.assertEqual(app.state, before)
        app.commit(app.state.with_team_loyalty("Leo", 1), "loyalty")
        self.select("contracts", "Leo")
        app.sign_selected_contract("contracts")
        self.assertTrue(app.state.can_play("Leo"))

    def test_affordability_search_and_short_period_selection(self):
        app = self.app
        app.commit(replace(app.state, money=400_000), "balance")
        app.show_screen("scout")
        app.scout_search.set("Leo")
        self.assertTrue(app.scout_players.exists("Leo"))
        self.assertFalse(app.scout_players.exists("Meiy"))
        self.select("scout", "Leo")
        self.assertEqual(str(app.offer_buttons["scout"]["state"]), "disabled")
        app.offer_kind["scout"].set("短期契約")
        app.offer_months["scout"].set("1")
        app.refresh_offer("scout")
        self.assertEqual(str(app.offer_buttons["scout"]["state"]), "normal")
        app.sign_selected_contract("scout")
        self.assertEqual(app.state.contract("Leo").duration_months, 1)

    def test_month_cannot_advance_during_scrim(self):
        self.app.scrim_job = Mock()
        before = self.app.state
        self.app.advance_game_month()
        self.assertEqual(self.app.state, before)
        self.app.scrim_job = None

    def test_zero_trait_only_offers_short_contract_for_scout_and_renewal(self):
        app = self.app
        with patch.dict(character_stats.CHARACTER_TABLE, {"Leo": replace(get_by_name("Leo"), loyalty=0)}):
            app.refresh_management()
            self.select("scout", "Leo")
            self.assertEqual(tuple(app.scout_kind_menu["values"]), ("短期契約",))
            self.assertEqual(app.offer_kind["scout"].get(), "短期契約")
            app.offer_months["scout"].set("1")
            app.sign_selected_contract("scout")
        self.assertEqual(app.state.contract("Leo").kind, "short")
        app.advance_game_month()
        self.select("contracts", "Leo")
        self.assertEqual(tuple(app.contracts_kind_menu["values"]), ("短期契約",))
        self.assertEqual(app.offer_kind["contracts"].get(), "短期契約")
        app.sign_selected_contract("contracts")
        self.assertTrue(app.state.can_play("Leo"))
        self.select("scout", "Meiy")
        self.assertEqual(len(app.scout_kind_menu["values"]), 4)


if __name__ == "__main__":
    unittest.main()
