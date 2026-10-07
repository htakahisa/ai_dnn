"""Term-dependent loyalty, permanent refusal and calendar renewal grace."""

from dataclasses import replace
import json
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import test_season_rival_economy as fixtures
import test_season_tournament_contracts as tournaments
from realtime_season import SeasonSaveError, new_season
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import add_months, parse_date
from season.season_loyalty import CONTRACT_LOYALTY
from season.season_transfers import restore_regular_members


class ContractLifecycleTests(unittest.TestCase):
    setUp = fixtures.RivalEconomyTests.setUp

    def state(self):
        return replace(fixtures.RivalEconomyTests.state(self), money=100_000_000)

    def test_first_contract_loyalty_for_every_duration(self):
        for kind, (initial, _) in CONTRACT_LOYALTY.items():
            with self.subTest(kind=kind):
                state = self.state().with_scouted_player("Meiy", kind, 1)
                self.assertEqual(state.team_loyalty("Meiy"), initial)
                self.assertEqual(state.contract("Meiy").signing_number, 1)
                self.assertEqual(state.contract_signings["Meiy"][state.club_id], 1)

    def test_second_and_third_contract_recover_loyalty_for_new_duration(self):
        for kind, (_, recovery) in CONTRACT_LOYALTY.items():
            with self.subTest(kind=kind):
                state = self.state().with_scouted_player("Meiy", "short", 1).advance_months()
                for count in (2, 3):
                    before = state.team_loyalty("Meiy")
                    state = state.with_renewed_contract("Meiy", kind, 1)
                    self.assertEqual(state.team_loyalty("Meiy"), before + recovery)
                    self.assertEqual(state.contract("Meiy").signing_number, count)
                    days = state.contract("Meiy").remaining(state.date, state.start_date)
                    state = state.advance_days(days, pay_salaries=False)

    def test_expired_negative_player_leaves_and_refuses_forever_only_that_club(self):
        state = self.state().with_team_loyalty("Leo", -.1)
        self.assertTrue(state.advance_months().can_play("Leo"))
        state = state.advance_months(12, pay_salaries=False)
        self.assertIsNone(state.player("Leo"))
        self.assertIn("Leo", {p.name for p in state.lft_players})
        self.assertIn(("Leo", state.club_id), state.contract_bans)
        state = state.with_team_loyalty("Leo", 99)
        with self.assertRaisesRegex(SeasonSaveError, "永久"):
            state.with_scouted_player("Leo", "year3")
        with self.assertRaisesRegex(SeasonSaveError, "永久"):
            state.with_added_players(("Leo",))
        club = state.opponent_teams[0]
        club = replace(club, regular_members=("Leo", *club.regular_members[:4]))
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        state = restore_regular_members(state, set(), lambda *args: None)
        self.assertEqual(state.opponent_owner("Leo").id, club.id)
        self.assertEqual(state.team_loyalty("Leo", club.id), 30)
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.contract_bans, state.contract_bans)
        self.assertTrue(loaded.contract_refused("Leo"))

    def test_zero_is_renewable_and_expired_negative_change_leaves_immediately(self):
        state = self.state().with_scouted_player("Meiy", "short", 1).advance_months()
        zero = state.with_team_loyalty("Meiy", 0)
        self.assertEqual(zero.with_renewed_contract("Meiy", "year1").team_loyalty("Meiy"), 10)
        negative = state.with_team_loyalty("Meiy", -.01)
        self.assertIsNone(negative.player("Meiy"))
        self.assertTrue(negative.contract_refused("Meiy"))

    def test_one_calendar_month_grace_penalty_once_and_rescout_recovery(self):
        state = self.state().with_scouted_player("Meiy", "short", 1).advance_months()
        contract = state.contract("Meiy")
        loyalty = contract.team_loyalty
        deadline = add_months(parse_date(contract.expired_on), 1)
        state = state.advance_days((deadline - state.date).days - 1)
        self.assertIsNotNone(state.player("Meiy"))
        self.assertFalse(state.can_play("Meiy"))
        self.assertEqual(state.team_loyalty("Meiy"), loyalty)
        state = state.advance_days()
        self.assertIsNone(state.player("Meiy"))
        self.assertEqual(state.team_loyalty("Meiy"), loyalty - 5)
        state = state.advance_days(5)
        self.assertEqual(state.team_loyalty("Meiy"), loyalty - 5)
        self.store.save(state)
        state = self.store.load_or_create().with_scouted_player("Meiy", "year1")
        self.assertEqual(state.team_loyalty("Meiy"), loyalty + 5)
        self.assertEqual(state.contract("Meiy").signing_number, 2)

    def test_timeout_penalty_applies_even_to_maximum_loyalty_trait(self):
        with patch.dict(character_stats.CHARACTER_TABLE, {"Meiy": replace(character_stats.get_by_name("Meiy"), loyalty=10)}):
            state = self.state().with_scouted_player("Meiy", "short", 1).advance_months(2)
        self.assertEqual(state.team_loyalty("Meiy"), 15)
        self.assertIsNone(state.player("Meiy"))

    def test_timeout_that_makes_loyalty_negative_also_records_permanent_refusal(self):
        state = self.state().with_scouted_player("Meiy", "short", 1).advance_months()
        state = state.with_team_loyalty("Meiy", 4).advance_months()
        self.assertEqual(state.team_loyalty("Meiy"), -1)
        self.assertTrue(state.contract_refused("Meiy"))
        self.assertIn(("Meiy", state.club_id), state.contract_bans)

    def test_invalid_contract_counter_or_expiry_cannot_overwrite_save(self):
        state = self.state()
        self.store.save(state)
        before = self.store.path.read_bytes()
        invalid = (
            replace(state, contract_signings={"Leo": {state.club_id: True}}),
            replace(state, contract_signings={"Leo": {state.club_id: -1}}),
            replace(state, contract_bans=(("Leo",),)),
            replace(state, contracts=(replace(state.contracts[0], signing_number=0), *state.contracts[1:])),
            replace(state, contracts=(replace(state.contracts[0], expired_on="invalid"), *state.contracts[1:])),
        )
        for candidate in invalid:
            with self.assertRaises(SeasonSaveError):
                self.store.save(candidate)
            self.assertEqual(self.store.path.read_bytes(), before)

    def test_rival_auto_renewal_recovers_and_poor_rival_has_grace(self):
        state = self.state()
        club = state.opponent_teams[0]
        club = replace(club, contracts=tuple(replace(c, kind="short", duration_months=1)
                                            if c.player_name == "Aspas" else c for c in club.contracts))
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        renewed = state.advance_months()
        self.assertEqual(renewed.team_loyalty("Aspas", club.id), 40)
        self.assertEqual(renewed.contract_signings["Aspas"][club.id], 2)
        poor = replace(state, opponent_teams=(replace(club, money=0, sponsor_active=False), state.opponent_teams[1]))
        month = poor.advance_months()
        self.assertEqual(month.opponent_owner("Aspas").id, club.id)
        self.assertEqual(month.team_loyalty("Aspas", club.id), 30)
        with patch("season.season_monthly_events.choose_recruit", return_value=None):
            late = month.advance_months()
        self.assertIsNone(late.opponent_owner("Aspas"))
        self.assertEqual(late.team_loyalty("Aspas", club.id), 25)

    def test_bulk_daily_save_and_legacy_migration_preserve_counters(self):
        state = self.state().with_scouted_player("Meiy", "short", 1)
        bulk = state.advance_months(2)
        daily = state
        while daily.date < bulk.date:
            daily = daily.advance_days()
        self.assertEqual(bulk, daily)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 25
        del data["contract_signings"], data["contract_bans"]
        for contracts in [data["contracts"], *(c["contracts"] for c in data["opponent_teams"])]:
            for contract in contracts:
                del contract["signing_number"], contract["expired_on"]
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        before = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.team_loyalty("Meiy"), state.team_loyalty("Meiy"))
        self.assertEqual(loaded.contract_signings["Meiy"][loaded.club_id], 1)
        self.assertEqual(self.store.path.read_bytes(), before)
        loaded = loaded.advance_months().with_renewed_contract("Meiy", "year2")
        self.assertEqual(loaded.contract("Meiy").signing_number, 2)

    def test_ui_shows_deadline_recovery_and_zero_loyalty_enabled(self):
        state = self.state().with_scouted_player("Meiy", "short", 1).advance_months().with_team_loyalty("Meiy", 0)
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = RealtimeSeasonApp(root, self.store, state)
        app.contracts_players.selection_set("Meiy")
        app.offer_kind["contracts"].set("730日契約")
        app.refresh_offer("contracts")
        self.assertIn("契約後の忠誠: 15", app.offer_summary["contracts"].get())
        self.assertIn("2026/02/28", app.offer_summary["contracts"].get())
        self.assertEqual(str(app.offer_buttons["contracts"].cget("state")), "normal")


class DeferredGraceTests(unittest.TestCase):
    setUp = tournaments.TournamentContractTests.setUp
    state = tournaments.TournamentContractTests.state
    in_progress = tournaments.TournamentContractTests.in_progress

    def test_tournament_completion_starts_a_full_month_of_renewal_grace(self):
        state = tournaments.finish(self.in_progress())
        contract = state.contract("Leo")
        self.assertEqual(contract.expired_on, state.game_date)
        self.assertIsNotNone(state.player("Leo"))
        deadline = add_months(state.date, 1)
        state = state.advance_days((deadline - state.date).days - 1, stop_for_tournaments=False)
        self.assertIsNotNone(state.player("Leo"))
        state = state.advance_days(stop_for_tournaments=False)
        self.assertIsNone(state.player("Leo"))


if __name__ == "__main__":
    unittest.main()
