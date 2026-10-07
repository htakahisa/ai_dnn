"""Monthly wages remain payable after a contract expires or a player leaves."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import test_season_contracts as fixtures
from realtime_season import SeasonSaveError, SeasonStore


class MonthlySalaryDueTests(unittest.TestCase):
    def setUp(self):
        fixtures.ContractRulesTest.setUp(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def signed(self, date="2026-01-01"):
        state = replace(fixtures.new_season(()), game_date=date)
        return state.with_sponsor_contract(False).with_scouted_player("Leo", "short", 1)

    def advance_to(self, state, date):
        from season.season_competitions import parse_date
        return state.advance_days((parse_date(date) - state.date).days)

    def wages(self, state):
        return [item for entry in state.history if entry["種別"] == "月次決算"
                for item in entry["支出"] if item["内訳"] == "給与"]

    def test_january_31_expiry_pays_january_salary_and_only_once(self):
        expired = self.advance_to(self.signed(), "2026-01-31")
        self.assertFalse(expired.can_play("Leo"))
        self.assertEqual(expired.monthly_payroll, 100_000)
        settled = expired.advance_days()
        self.assertEqual(settled.money, expired.money - 100_000)
        self.assertEqual(settled.monthly_salary_due, {})
        self.assertEqual(settled.monthly_payroll, 0)
        self.assertEqual(self.wages(settled), [{"内訳": "給与", "金額": 100_000, "選手": "Leo"}])
        march = self.advance_to(settled, "2026-03-01")
        self.assertEqual(march.money, settled.money)
        self.assertEqual(self.wages(march), self.wages(settled))

    def test_month_midway_expiry_keeps_full_monthly_salary(self):
        state = self.signed("2026-01-15")
        expired = self.advance_to(state, "2026-02-14")
        self.assertEqual(expired.money, state.money - 100_000)
        self.assertEqual(expired.monthly_payroll, 100_000)
        settled = self.advance_to(expired, "2026-03-01")
        self.assertEqual(settled.money, expired.money - 100_000)
        self.assertEqual(len(self.wages(settled)), 2)

    def test_release_does_not_cancel_earned_wages(self):
        state = self.signed().without_player("Leo")
        self.assertIsNone(state.player("Leo"))
        self.assertEqual(state.monthly_payroll, 100_000)
        settled = self.advance_to(state, "2026-02-01")
        self.assertEqual(settled.money, state.money - 100_000)
        self.assertEqual(len(self.wages(settled)), 1)

    def test_departure_after_expiry_does_not_cancel_earned_wages(self):
        state = self.advance_to(self.signed(), "2026-01-30")
        state = replace(state, contracts=(replace(state.contract("Leo"), team_loyalty=-1),))
        expired = state.advance_days()
        self.assertIsNone(expired.player("Leo"))
        settled = expired.advance_days()
        self.assertEqual(settled.money, expired.money - 100_000)
        self.assertEqual(len(self.wages(settled)), 1)

    def test_renewal_in_same_month_pays_once_at_current_rate(self):
        expired = self.advance_to(self.signed("2026-01-15"), "2026-02-14")
        renewed = expired.with_renewed_contract("Leo", "year2")
        self.assertEqual(renewed.monthly_payroll, 90_000)
        settled = self.advance_to(renewed, "2026-03-01")
        self.assertEqual(settled.money, renewed.money - 90_000)
        self.assertEqual([item["金額"] for item in self.wages(settled)], [100_000, 90_000])

    def test_restart_keeps_wages_for_released_player(self):
        released = self.signed().without_player("Leo")
        self.store.save(released)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, released)
        settled = self.advance_to(loaded, "2026-02-01")
        self.assertEqual(settled.money, released.money - 100_000)
        self.store.save(settled)
        later = self.advance_to(self.store.load_or_create(), "2026-03-01")
        self.assertEqual(later.money, settled.money)

    def test_legacy_save_recovers_current_month_expired_contract_wages(self):
        expired = self.advance_to(self.signed(), "2026-01-31")
        self.store.save(expired)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 31
        data.pop("monthly_salary_due")
        original = json.dumps(data, ensure_ascii=False)
        self.store.path.write_text(original, encoding="utf-8")
        loaded = self.store.load_or_create()
        self.assertEqual(self.store.path.read_text(encoding="utf-8"), original)
        self.assertEqual(loaded.monthly_payroll, 100_000)
        self.assertEqual(loaded.advance_days().money, expired.money - 100_000)

    def test_legacy_save_does_not_charge_preceding_month(self):
        settled = self.advance_to(self.signed(), "2026-02-01")
        self.store.save(settled)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 31
        data.pop("monthly_salary_due")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.monthly_payroll, 0)
        self.assertEqual(self.advance_to(loaded, "2026-03-01").money, settled.money)

    def test_invalid_or_missing_ledger_is_rejected(self):
        self.store.save(self.signed())
        original = self.store.path.read_text(encoding="utf-8")
        for ledger in (None, [], {"Leo": -1}, {"Leo": True}, {"": 100_000}):
            with self.subTest(ledger=ledger):
                data = json.loads(original)
                data["monthly_salary_due"] = ledger
                self.store.path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(SeasonSaveError):
                    self.store.load_or_create()
        data = json.loads(original)
        data.pop("monthly_salary_due")
        self.store.path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(SeasonSaveError):
            self.store.load_or_create()

    def test_daily_and_batch_settlement_match(self):
        state = self.signed()
        daily = state
        for _ in range(59):
            daily = daily.advance_days()
        batch = state.advance_days(59)
        self.assertEqual(batch, daily)
        self.assertEqual(batch.history, daily.history)
        self.assertEqual(batch.money, state.money - 100_000)

    def test_skip_salary_clears_ledger_at_month_boundary(self):
        expired = self.advance_to(self.signed(), "2026-01-31")
        skipped = expired.advance_days(pay_salaries=False)
        self.assertEqual(skipped.money, expired.money)
        self.assertEqual(skipped.monthly_salary_due, {})
        self.assertEqual(skipped.monthly_payroll, 0)


if __name__ == "__main__":
    unittest.main()
