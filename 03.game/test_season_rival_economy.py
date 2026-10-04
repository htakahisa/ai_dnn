"""Rival budgets, double-entry transfers, contract costs and save migration."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_rival_economy as settings
import realtime_season_teams as teams
import realtime_season_world_levels as levels
from realtime_season import SeasonSaveError, SeasonStore, new_season, FORCED_OFFER_LOYALTY
from season_competitions import SeriesScore, next_match
from season_monthly_events import process_monthly_events
from season_ratings import SeasonRating
from season_rival_economy import make_offer, monthly_settlement, reserved_funds, signing_terms
from season_transfers import restore_regular_members


OWN = ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")
RIVALS = (("Aspas", "valyn", "trent", "leaf", "tex"), ("Boostio", "Ethan", "jawgemo", "C0M", "Demon1"))
CUP = dict(id="cup", name="Cup", start_date="2026-01-02", team_count=2,
           format="single_elimination", prizes={1: 1_000_000, 2: 500_000}, normal_maps_to_win=1, grand_final_maps_to_win=1)


class RivalEconomyTests(unittest.TestCase):
    def setUp(self):
        fixture = {name: replace(p, monthly_salary=100_000, loyalty=5)
                   for name, p in character_stats.CHARACTER_TABLE.items()}
        self.config = [dict(name=f"Rival{i}", players=list(names), initial_money=20_000_000)
                       for i, names in enumerate(RIVALS)]
        contexts = [patch.dict(character_stats.CHARACTER_TABLE, fixture),
                    patch.object(teams, "SEASON_TEAMS", self.config),
                    patch.object(calendar, "START_DATE", "2026-01-01"),
                    patch.object(calendar, "TOURNAMENTS", []),
                    patch.object(settings, "NON_REGULAR_OFFER_CHANCE", 0),
                    patch.object(levels, "WORLD_LEVELS", [
                        {"レベル": 1, "上位%": 100, "敵倍率": 1, "スポンサー資金": 7_500_000},
                        {"レベル": 2, "上位%": 50, "敵倍率": 1.5, "スポンサー資金": 10_000_000}])]
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        return replace(state, opponent_teams=tuple(replace(c, regular_members=c.members) for c in state.opponent_teams))

    def offer(self, state, name="Leo", money=None):
        buyer = state.opponent_teams[0]
        if money is not None:
            buyer = replace(buyer, money=money)
            state = replace(state, opponent_teams=(buyer, state.opponent_teams[1]))
        return make_offer(state, buyer, state.player(name), lambda *args: None)

    def test_monthly_sponsorship_uses_rival_rank_and_pays_active_contracts(self):
        state = self.state()
        rivals = state.opponent_teams
        state = replace(state, ratings=(SeasonRating(state.club_id, state.team_name, 1500),
                        SeasonRating(rivals[0].id, rivals[0].name, 1800),
                        SeasonRating(rivals[1].id, rivals[1].name, 1300)))
        month = state.advance_months()
        self.assertEqual([c.money for c in month.opponent_teams], [29_500_000, 27_000_000])
        self.assertEqual(process_monthly_events(month), month)
        self.assertEqual(state.advance_days(30).opponent_teams, state.opponent_teams)
        stopped = replace(state, opponent_teams=(replace(rivals[0], sponsor_active=False), rivals[1]))
        self.assertEqual(monthly_settlement(stopped).opponent_teams[0].money, 19_500_000)
        self.assertEqual(monthly_settlement(stopped, pay_salaries=False).opponent_teams[0].money, 20_000_000)

    def test_player_scout_pays_the_source_rival(self):
        # Give this transfer scenario sufficient funds regardless of the starting budget.
        state = replace(self.state(), money=3_000_000)
        fee = state.transfer_fee("Aspas")
        signed = state.with_scouted_player("Aspas", "year1")
        self.assertEqual(signed.opponent_teams[0].money, state.opponent_teams[0].money + fee)
        self.assertEqual(signed.money, state.money - fee - 300_000)

    def test_offer_needs_fee_bonus_and_full_term_wages_and_debits_only_on_accept(self):
        state = self.state()
        required = 5_000_000 + signing_terms(state, state.player("Leo")).total_required_funds
        self.assertFalse(self.offer(state, money=required - 1).pending_transfer_offers)
        proposed = self.offer(state, money=required)
        offer = proposed.transfer_offer("Leo")
        self.assertEqual(proposed.transfer_multiplier, 50)
        self.assertEqual(offer.fee, 5_000_000)
        self.assertEqual(proposed.opponent_teams[0].money, required)
        self.assertEqual(reserved_funds(proposed, proposed.opponent_teams[0]), required)
        accepted = proposed.with_transfer_response(offer.id, True)
        self.assertEqual(accepted.opponent_teams[0].money, 1_200_000)
        self.assertEqual(accepted.money, proposed.money + offer.fee)
        self.assertEqual(accepted.opponent_owner("Leo").id, offer.team_id)
        self.assertEqual(accepted.transfer_offers[0].status, "accepted")
        self.assertEqual(reserved_funds(accepted, accepted.opponent_teams[0]), 0)
        contract = next(c for c in accepted.opponent_teams[0].contracts if c.player_name == "Leo")
        self.assertEqual((contract.kind, contract.duration_months, contract.monthly_salary), ("year1", 12, 100_000))
        with self.assertRaises(SeasonSaveError):
            accepted.with_transfer_response(offer.id, True)

    def test_decline_then_low_loyalty_forces_exactly_one_payment(self):
        state = self.offer(self.state())
        offer = state.transfer_offer("Leo")
        rejected = state.with_transfer_response(offer.id, False)
        self.assertEqual(rejected.opponent_teams[0].money, state.opponent_teams[0].money)
        self.assertEqual(reserved_funds(rejected, rejected.opponent_teams[0]), 6_500_000)
        forced = rejected.with_team_loyalty("Leo", FORCED_OFFER_LOYALTY - .1)
        self.assertIsNone(forced.player("Leo"))
        self.assertEqual(forced.opponent_teams[0].money, 14_700_000)
        self.assertEqual(forced.transfer_offers[0].status, "forced")
        self.assertEqual(forced.with_resolved_transfer_offers(), forced)
        self.store.save(forced)
        self.assertEqual(self.store.load_or_create(), forced)

    def test_buyer_losing_funds_cancels_without_creating_money_or_losing_player(self):
        state = self.offer(self.state())
        state = replace(state, opponent_teams=(replace(state.opponent_teams[0], money=1), state.opponent_teams[1]))
        before_money = state.money
        candidate = state.with_transfer_response(state.transfer_offer("Leo").id, True)
        self.assertIsNotNone(candidate.player("Leo"))
        self.assertEqual(candidate.money, before_money)
        self.assertEqual(candidate.opponent_teams[0].money, 1)
        self.assertEqual(candidate.transfer_offers[0].status, "cancelled")

    def test_multiple_pending_offers_cannot_overcommit_and_poor_club_cannot_fill_vacancy(self):
        state = self.offer(self.state(), money=6_500_000)
        candidate = self.offer(state, "Boaster")
        self.assertEqual(len(candidate.pending_transfer_offers), 1)
        club = candidate.opponent_teams[0]
        # A vacancy cannot consume money reserved for the outgoing proposal.
        club = replace(club, players=club.players[:-1], contracts=tuple(
            replace(c, end_reason="released") if c.player_name == "tex" else c for c in club.contracts),
            regular_members=(), sponsor_active=False)
        candidate = replace(candidate, opponent_teams=(club, candidate.opponent_teams[1]), game_month=1, game_date="2026-02-01")
        processed = process_monthly_events(candidate)
        self.assertEqual(len(processed.opponent_teams[0].players), 4)
        self.assertEqual(processed.opponent_teams[0].money, 6_500_000)
        self.assertTrue(any(e.kind == "recruitment_unfilled" for e in processed.monthly_events))

    def test_non_regular_monthly_offer_keeps_acquired_player_after_next_month(self):
        state = self.state()
        state = replace(state, owned_players=tuple(replace(p, iq=900) if p.name == "Leo" else p for p in state.owned_players))
        with patch.object(settings, "NON_REGULAR_OFFER_CHANCE", 1):
            state = state.advance_months()
        offer = state.transfer_offer("Leo")
        self.assertIsNotNone(offer)
        self.assertNotIn("Leo", state.opponent_teams[0].regular_members)
        self.assertEqual(len([o for o in state.pending_transfer_offers if o.team_id == state.opponent_teams[0].id]), 1)
        before = state.opponent_teams[0].money
        state = state.with_transfer_response(offer.id, True)
        self.assertEqual(state.opponent_teams[0].money, before - 5_300_000)
        following = state.advance_months()
        self.assertIn("Leo", following.opponent_teams[0].members)
        self.assertIn("Leo", following.opponent_teams[0].roster)
        self.assertIn("Leo", following.opponent_teams[0].acquired_members)

    def test_cross_rival_returns_pay_source_fee_and_signing_bonuses(self):
        state = self.state()
        left, right = state.opponent_teams
        left = replace(left, regular_members=("Boostio", *RIVALS[0][1:]))
        right = replace(right, regular_members=("Aspas", *RIVALS[1][1:]))
        state = replace(state, opponent_teams=(left, right))
        restored = restore_regular_members(state, set(), lambda *args: None)
        self.assertEqual(restored.opponent_teams[0].members, left.regular_members)
        self.assertEqual(restored.opponent_teams[1].members, right.regular_members)
        self.assertEqual([c.money for c in restored.opponent_teams], [19_700_000, 19_700_000])
        restored.validate()
        broke = replace(state, opponent_teams=tuple(replace(c, money=0) for c in state.opponent_teams))
        unchanged = restore_regular_members(broke, set(), lambda *args: None)
        self.assertEqual({c.id: set(c.members) for c in unchanged.opponent_teams},
                         {c.id: set(c.members) for c in broke.opponent_teams})
        self.assertEqual([c.money for c in unchanged.opponent_teams], [0, 0])

    def test_import_preserves_paid_recruit_even_without_regular_member_metadata(self):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        state = self.offer(state)
        offer = state.transfer_offer("Leo")
        state = state.with_transfer_response(offer.id, True)
        self.assertIn("Leo", state.opponent_teams[0].acquired_members)
        imported = self.store.import_season_teams(state)
        self.assertEqual(imported.opponent_teams[0], state.opponent_teams[0])
        self.assertEqual(self.store.load_or_create(), imported)

    def test_renewals_charge_bonus_and_require_term_funds(self):
        state = self.state()
        club = state.opponent_teams[0]
        contracts = tuple(replace(c, kind="short", duration_months=1) if c.player_name == "Aspas" else c for c in club.contracts)
        club = replace(club, contracts=contracts, regular_members=(), sponsor_active=False)
        state = replace(state, opponent_teams=(club, state.opponent_teams[1]))
        rich = state.advance_months()
        self.assertEqual(rich.opponent_teams[0].money, 19_300_000)
        self.assertTrue(any(e.kind == "renewal" for e in rich.monthly_events))
        poor = replace(state, opponent_teams=(replace(club, money=500_000), state.opponent_teams[1])).advance_months()
        self.assertIn("Aspas", poor.opponent_teams[0].members)
        self.assertEqual(poor.opponent_teams[0].money, 100_000)
        self.assertNotIn("Aspas", poor.advance_months().opponent_teams[0].members)

    def test_all_finalists_receive_prizes_once_including_npc_only_cup(self):
        for own_entry in (True, False):
            with self.subTest(own_entry=own_entry), patch.object(calendar, "TOURNAMENTS", [dict(CUP, allow_player_entry=own_entry)]):
                state = self.state()
                state = state.with_tournament_entry("cup", state.teams[0].id if own_entry else None)
                before = {c.id: c.money for c in state.opponent_teams}
                match, _ = next_match(state.tournament_definition("cup"), state.tournament("cup"))
                score = SeriesScore(match.id, match.left, match.right, 1, 0)
                state = state.advance_days()
                finished = state.with_tournament_result("cup", score) if own_entry else state
                rankings = finished.tournament("cup").ranking
                for club in finished.opponent_teams:
                    prize = CUP["prizes"].get(rankings.index(club.id) + 1, 0) if club.id in rankings else 0
                    self.assertEqual(club.money, before[club.id] + prize)
                self.assertEqual(finished.money, state.money + (1_000_000 if own_entry else 0))
                with self.assertRaises(SeasonSaveError):
                    finished.with_tournament_result("cup", score)
                self.store.save(finished)
                self.assertEqual(self.store.load_or_create(), finished)

    def test_rival_world_level_stays_fixed_during_cup(self):
        with patch.object(calendar, "TOURNAMENTS", [CUP]):
            state = self.state()
        first = state.opponent_teams[0]
        state = replace(state, ratings=tuple(replace(r, value=2000 if r.team_id == first.id else 1000) for r in state.ratings))
        state = state.with_tournament_entry("cup", state.teams[0].id).advance_days()
        self.assertEqual(state.opponent_teams[0].world_level_lock.sponsor_funds, 10_000_000)
        changed = replace(state, ratings=tuple(replace(r, value=1 if r.team_id == first.id else 3000) for r in state.ratings))
        self.assertEqual(monthly_settlement(changed).opponent_teams[0].money, 29_500_000)

    def test_save_import_and_legacy_migration_preserve_budgets_without_replaying_income(self):
        state = self.offer(self.state())
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        self.config[0]["initial_money"] = 99_000_000
        imported = self.store.import_season_teams(state)
        self.assertEqual(imported.opponent_teams[0].money, state.opponent_teams[0].money)
        self.assertEqual(imported.transfer_offers, state.transfer_offers)
        data = json.loads(self.store.path.read_text(encoding="utf-8"))
        data["version"] = 18
        for club in data["opponent_teams"]:
            for field in ("money", "sponsor_active", "acquired_members", "world_level_lock"):
                club.pop(field)
        for offer in data["transfer_offers"]:
            offer["fee"] = 1_200_000
            for field in ("contract_kind", "contract_months", "monthly_salary"):
                offer.pop(field)
        self.store.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        original = self.store.path.read_bytes()
        migrated = self.store.load_or_create()
        self.assertEqual(migrated.opponent_teams[0].money, 99_000_000)
        self.assertEqual(migrated.transfer_offer("Leo").fee, 5_000_000)
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertEqual(self.store.load_or_create(), migrated)

    def test_invalid_budget_and_offer_terms_are_rejected(self):
        state = self.offer(self.state())
        for bad_money in (1.5, True, "100"):
            with self.subTest(bad_money=bad_money), self.assertRaises(SeasonSaveError):
                self.store.save(replace(state, opponent_teams=(replace(state.opponent_teams[0], money=bad_money), state.opponent_teams[1])))
        with self.assertRaises(SeasonSaveError):
            self.store.save(replace(state, transfer_offers=(replace(state.transfer_offers[0], monthly_salary=None),)))


if __name__ == "__main__":
    unittest.main()
