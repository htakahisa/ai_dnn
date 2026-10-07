"""Fixed-day contract boundaries, signing dates, migration, and Tk display."""

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import realtime_season_teams as teams
import test_season_contracts as fixtures
from realtime_season import SeasonSaveError, SeasonStore, contract_terms
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import month_index, parse_date
from season.season_rival_economy import signed_contract


class ContractDaysTests(unittest.TestCase):
    def setUp(self):
        fixtures.ContractRulesTest.setUp(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def on_date(self, value):
        state = fixtures.new_season(())
        month = month_index(parse_date(state.start_date), parse_date(value))
        return replace(state, game_date=value, game_month=month, monthly_events_through=month)

    def test_fixed_lengths_and_exact_signing_date(self):
        state = self.on_date("2028-02-29")
        for kind, months, days in (("short", 1, 30), ("short", 2, 60), ("short", 6, 180),
                                  ("year1", 12, 365), ("year2", 24, 730), ("year3", 36, 1095)):
            with self.subTest(kind=kind, months=months):
                terms = contract_terms(fixtures.get_by_name("Leo"), kind, months if kind == "short" else 6)
                signed = state.with_scouted_player("Leo", kind, months if kind == "short" else 6)
                contract = signed.contract("Leo")
                self.assertEqual((terms.days, contract.duration_days), (days, days))
                self.assertEqual(contract.signed_on, "2028-02-29")
                self.assertEqual(contract.ends_on(signed.start_date), date(2028, 2, 29) + timedelta(days=days))
                self.assertEqual(contract.remaining(signed.date, signed.start_date), days - 1)
                self.assertTrue(contract.active(contract.ends_on(signed.start_date) - timedelta(days=1), signed.start_date))
                self.assertFalse(contract.active(contract.ends_on(signed.start_date), signed.start_date))
                self.assertEqual(self.store_round_trip(signed), signed)

    def store_round_trip(self, state):
        self.store.save(state)
        return self.store.load_or_create()

    def test_month_end_and_leap_day_expire_on_exact_day(self):
        for start, months, end in (("2026-01-31", 1, "2026-03-02"),
                                   ("2026-02-28", 2, "2026-04-29"),
                                   ("2028-02-29", 1, "2028-03-30")):
            with self.subTest(start=start):
                signed = self.on_date(start).with_scouted_player("Leo", "short", months)
                before = signed.advance_days((parse_date(end) - signed.date).days - 1)
                self.assertTrue(before.can_play("Leo"))
                self.assertEqual(before.contract("Leo").remaining(before.date, before.start_date), 1)
                ended = before.advance_days()
                self.assertEqual(ended.game_date, end)
                self.assertFalse(ended.can_play("Leo"))
                self.assertEqual(ended.contract("Leo").expired_on, end)
                self.assertIsNotNone(ended.player("Leo"))

    def test_renewal_starts_on_renewal_day_and_batch_matches_daily(self):
        signed = self.on_date("2026-02-28").with_scouted_player("Leo", "short", 1)
        end = signed.advance_days(29)
        self.assertEqual(end.game_date, "2026-03-30")
        with self.assertRaisesRegex(SeasonSaveError, "終了してから"):
            signed.with_renewed_contract("Leo", "short", 2)
        ready = end.advance_days(4)
        renewed = ready.with_renewed_contract("Leo", "short", 2)
        self.assertEqual(renewed.contract("Leo").signed_on, "2026-04-03")
        self.assertEqual(renewed.contract("Leo").ends_on(renewed.start_date), date(2026, 6, 2))
        daily = renewed
        for _ in range(60):
            daily = daily.advance_days()
        batch = renewed.advance_days(60)
        self.assertEqual(batch, daily)
        self.assertFalse(batch.can_play("Leo"))

    def test_rival_signing_and_import_use_current_day(self):
        state = self.on_date("2026-01-31")
        with patch.object(teams, "SEASON_TEAMS", [{"name": "Rival", "players": fixtures.RIVAL}]):
            state = self.store.import_season_teams(state)
        club = state.opponent_teams[0]
        self.assertEqual({c.signed_on for c in club.contracts}, {"2026-01-31"})
        player = club.players[0]
        contract = signed_contract(player, state.contract_terms(player, "short", 2), state.game_month,
                                   state=state, team_id=club.id)
        self.assertEqual(contract.ends_on(state.start_date), date(2026, 4, 1))
        state = replace(state, opponent_teams=(replace(club, contracts=tuple(
            contract if c.player_name == player.name else c for c in club.contracts)),))
        before = state.advance_days(59)
        self.assertTrue(before.contract_usable(before.opponent_teams[0].contracts[0], club.id))
        self.assertEqual(self.store_round_trip(before), before)

    def test_legacy_save_uses_saved_start_month_without_rewriting_save(self):
        with patch.object(teams, "SEASON_TEAMS", [{"name": "Rival", "players": fixtures.RIVAL}]):
            state = self.on_date("2026-02-18").with_scouted_player("Leo", "short", 2)
        self.store.save(state)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 26
        for contract in data["contracts"]:
            contract.pop("signed_on")
        for club in data["opponent_teams"]:
            for contract in club["contracts"]:
                contract.pop("signed_on")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        original = self.store.path.read_bytes()
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.contract("Leo").starts_on(loaded.start_date), date(2026, 2, 1))
        self.assertEqual(loaded.contract("Leo").ends_on(loaded.start_date), date(2026, 4, 2))
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertEqual(self.store_round_trip(loaded), loaded)

    def test_invalid_saved_signing_dates_are_rejected(self):
        state = self.on_date("2026-02-18").with_scouted_player("Leo", "short", 2)
        for signed_on in ("bad", "2026-02-20", "2026-01-31", "2025-12-31", 12):
            with self.subTest(signed_on=signed_on), self.assertRaises((SeasonSaveError, TypeError)):
                replace(state, contracts=(replace(state.contract("Leo"), signed_on=signed_on),)).validate()

    def test_contract_screen_and_scout_use_days(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = self.on_date("2026-01-31").with_scouted_player("Leo", "short", 1)
        app = RealtimeSeasonApp(root, self.store, state)
        app.show_screen("contracts")
        row = app.contracts_players.item("Leo", "values")
        self.assertEqual(row[4:7], ("1 / 30日", "29日", "2026/03/02"))
        self.assertEqual(app.contracts_players.heading("end", "text"), "終了日")
        app.show_screen("scout")
        self.assertEqual(tuple(app.scout_duration_menu["values"]), ("30", "60", "90", "120", "150", "180"))
        self.assertIn("365日契約", app.scout_kind_menu["values"])
        app.scout_players.selection_set("Meiy")
        app.offer_kind["scout"].set("短期契約")
        app.offer_months["scout"].set("60")
        app.refresh_offer("scout")
        self.assertIn("60日間", app.offer_summary["scout"].get())
        app.sign_selected_contract("scout")
        self.assertEqual(app.state.contract("Meiy").signed_on, "2026-02-01")
        self.assertEqual(app.state.contract("Meiy").ends_on(app.state.start_date), date(2026, 4, 2))


if __name__ == "__main__":
    unittest.main()
