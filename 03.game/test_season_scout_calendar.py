"""Scouting allowances and automatic rival tournaments share the daily calendar."""

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

import character_stats
import realtime_season_competitions as calendar
import realtime_season_config as config
import realtime_season_rival_economy as economy
import realtime_season_teams as teams
from realtime_season import SeasonSaveError, SeasonStore, new_season
from season_competitions import SeriesScore, next_match
from season_rival_economy import make_offer
from season_monthly_events import process_monthly_events
from season_scrim import build_scrim_request
from season_scouting import season_window
from test_season_competitions import definition, OWN, RIVALS


class ScoutCalendarTests(unittest.TestCase):
    def setUp(self):
        fixture = {n: replace(p, monthly_salary=100_000, loyalty=10)
                   for n, p in character_stats.CHARACTER_TABLE.items()}
        others = [n for n in fixture if n not in OWN and not any(n in r for r in RIVALS)][:5]
        for context in (
            patch.dict(character_stats.CHARACTER_TABLE, fixture),
            patch.object(config, "INITIAL_OWNED_PLAYERS", OWN),
            patch.object(calendar, "START_DATE", "2026-01-01"),
            patch.object(calendar, "IN_SEASON_PERIODS", [("03-01", "11-30")]),
            patch.object(calendar, "TOURNAMENTS", []),
            patch.object(economy, "NON_REGULAR_OFFER_CHANCE", 0),
            patch.object(teams, "SEASON_TEAMS", [dict(name=f"Rival{i}", players=list(names), initial_money=100_000_000)
                                                for i, names in enumerate((*RIVALS, others))]),
        ):
            context.start()
            self.addCleanup(context.stop)
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        self.store = SeasonStore(Path(directory.name) / "save.json")

    def state(self, **changes):
        state = new_season(OWN).with_roster(OWN).with_confirmed_team()
        return replace(state.with_selected_team(state.teams[0].id), money=100_000_000, **changes)

    def free_names(self, state):
        return [p.name for p in state.lft_players if p.name not in state.owned_players][:4]

    def cup(self, **changes):
        from season_competitions import definition_from_dict
        return definition_from_dict(definition(**{"start_date": "2026-01-05", **changes}))

    def test_scout_spends_day_updates_iq_and_familiarity(self):
        state = self.state()
        name = self.free_names(state)[0]
        player = next(p for p in state.scout_players if p.name == name)
        updated = state.with_scouted_player(name, "year1")
        self.assertEqual((updated.date - state.date).days, 1)
        self.assertAlmostEqual(updated.player(name).iq, player.iq + .1)
        self.assertEqual(updated.pair_days[tuple(sorted((name, OWN[0])))], 1)
        self.assertEqual(updated.scout_uses[-1], ("2026-01-01", state.club_id))

    def test_scout_on_month_boundary_settles_salary_and_sponsor_once(self):
        state = self.state().advance_days(30)
        name = self.free_names(state)[0]
        player = next(p for p in state.scout_players if p.name == name)
        terms = state.contract_terms(player, "year1")
        updated = state.with_scouted_player(name, "year1")
        self.assertEqual(updated.game_date, "2026-02-01")
        self.assertEqual(updated.money, state.money - terms.signing_bonus
                         + state.monthly_sponsor_income - state.monthly_payroll - terms.monthly_salary)
        self.assertEqual(sum(e.kind == "month_completed" for e in updated.monthly_events), 1)
        self.assertEqual(updated.pair_days[tuple(sorted((name, OWN[0])))], 1)

    def test_in_season_limit_and_failed_attempt_are_atomic(self):
        state = self.state().advance_days(59)
        names = self.free_names(state)
        self.assertEqual(state.scout_remaining(), 2)
        for name in names[:2]:
            state = state.with_scouted_player(name, "year1")
        self.assertEqual(state.scout_remaining(), 0)
        self.assertIn("残り0回", state.scout_allowance_text)
        with self.assertRaisesRegex(SeasonSaveError, "スカウト上限"):
            state.with_scouted_player(names[2], "year1")
        self.assertEqual(state.game_date, "2026-03-03")
        self.assertEqual(len(state.scout_uses), 2)

    def test_off_season_is_unlimited_and_new_season_resets(self):
        state = self.state()
        for name in self.free_names(state)[:3]:
            state = state.with_scouted_player(name, "year1")
        self.assertIsNone(state.scout_remaining())
        state = state.advance_days(56)
        self.assertEqual(state.game_date, "2026-03-01")
        self.assertEqual(state.scout_remaining(), 2)
        state = state._with_scout_use(state.club_id)._with_scout_use(state.club_id)
        state = state.advance_days(275)
        self.assertEqual(state.game_date, "2026-12-01")
        self.assertIsNone(state.scout_remaining())
        state = state.advance_days(90)
        self.assertEqual(state.game_date, "2027-03-01")
        self.assertEqual(state.scout_remaining(), 2)

    def test_wraparound_window_keeps_same_counter_across_new_year(self):
        state = self.state(in_season_periods=(("11-01", "02-28"),))
        state = state._with_scout_use(state.club_id).advance_days(30)
        self.assertEqual(state.scout_remaining(), 1)
        self.assertEqual(season_window(state.date, state.in_season_periods)[0].isoformat(), "2025-11-01")

    def test_failed_finance_or_missing_target_spends_neither_day_nor_slot(self):
        state = self.state()
        for target, money in (("does-not-exist", state.money), (self.free_names(state)[0], 0)):
            with self.assertRaises(SeasonSaveError):
                replace(state, money=money).with_scouted_player(target, "year1")
        self.assertEqual(state.scout_uses, ())
        self.assertEqual(state.game_date, "2026-01-01")

    def test_renewal_does_not_consume_scout_slot_or_day(self):
        state = self.state()
        state = replace(state, contracts=tuple(replace(c, kind="short", duration_months=2) for c in state.contracts))
        state = state.advance_days(60)
        updated = state.with_renewed_contract(OWN[0], "year1")
        self.assertEqual(updated.date, state.date)
        self.assertEqual(updated.scout_remaining(), 2)

    def test_rival_acquisitions_and_pending_offers_share_two_slots(self):
        state = self.state().advance_days(59)
        club = state.opponent_teams[0]
        for name in OWN[:2]:
            state = make_offer(state, club, state.player(name), lambda *args: None)
            state = state.with_transfer_response(state.transfer_offer(name).id, True)
        self.assertEqual(state.scout_remaining(club.id), 0)
        self.assertIs(make_offer(state, club, state.player(OWN[2]), lambda *args: None), state)

    def test_rival_reserves_slots_for_outstanding_proposals(self):
        state = self.state().advance_days(59)
        club = state.opponent_teams[0]
        for name in OWN[:2]:
            state = make_offer(state, club, state.player(name), lambda *args: None)
        self.assertEqual(len(state.pending_transfer_offers), 2)
        self.assertIs(make_offer(state, club, state.player(OWN[2]), lambda *args: None), state)

    def test_regular_returns_and_lft_recruitment_cannot_exceed_rival_quota(self):
        state = self.state().advance_days(59)
        club = state.opponent_teams[0]
        retained = club.players[2:]
        removed = {p.name for p in club.players[:2]}
        club = replace(club, players=retained, regular_members=tuple(p.name for p in club.players),
                       contracts=tuple(replace(c, end_reason="released") if c.player_name in removed else c
                                       for c in club.contracts))
        state = replace(state, opponent_teams=(club, *state.opponent_teams[1:]), monthly_events_through=1,
                        monthly_events=tuple(e for e in state.monthly_events if e.date < state.game_date))
        state = state._with_scout_use(club.id)
        updated = process_monthly_events(state)
        self.assertEqual(updated.scout_remaining(club.id), 0)
        # Exactly one return fits; the other return and vacancy fill wait.
        self.assertEqual(len(updated.opponent_teams[0].players), len(retained) + 1)
        self.assertEqual(sum(c == club.id for _, c in updated.scout_uses), 2)

    def test_offer_settlement_rechecks_quota_before_charging_buyer(self):
        state = self.state().advance_days(59)
        buyer = state.opponent_teams[0]
        state = make_offer(state, buyer, state.player(OWN[0]), lambda *args: None)
        state = state._with_scout_use(buyer.id)._with_scout_use(buyer.id)
        offer = state.transfer_offer(OWN[0])
        updated = state.with_transfer_response(offer.id, True)
        self.assertIsNotNone(updated.player(OWN[0]))
        self.assertEqual(updated.money, state.money)
        self.assertEqual(updated.opponent_teams[0].money, buyer.money)
        self.assertEqual(next(o for o in updated.transfer_offers if o.id == offer.id).status, "cancelled")

    def test_registered_tournament_blocks_every_day_action_from_start(self):
        state = self.state(tournament_definitions=(self.cup(),))
        state = state.with_tournament_entry("cup", state.selected_team_id)
        state = state.advance_days(4)
        self.assertEqual(state.game_date, "2026-01-05")
        name = self.free_names(state)[0]
        actions = (
            lambda: state.with_scouted_player(name, "year1"),
            lambda: state.with_trained_player(OWN[0], "research"),
            lambda: state.with_trained_player(OWN[0], "aim_lab"),
            lambda: build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id),
            lambda: state.with_scrim_result("scrim", state.club_id, state.opponent_teams[0].id, 1, 0),
        )
        for action in actions:
            with self.assertRaisesRegex(SeasonSaveError, "出場中"):
                action()
        self.assertEqual(state.advance_days(), state)
        match, _ = next_match(self.cup(), state.tournament("cup"))
        updated = state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right, match.maps_to_win, 0))
        self.assertTrue(updated.day_action_blocked)

    def test_scout_day_can_arrive_at_registered_tournament_start(self):
        state = self.state(tournament_definitions=(self.cup(),))
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(3)
        state = state.with_scouted_player(self.free_names(state)[0], "year1")
        self.assertEqual(state.game_date, "2026-01-05")
        self.assertTrue(state.day_action_blocked)

    def test_unregistered_cup_keeps_start_date_open_then_runs_automatically(self):
        state = self.state(tournament_definitions=(self.cup(),)).advance_days(4)
        self.assertIsNone(state.tournament("cup"))
        self.assertEqual(state.entry_deadline_tournaments, (self.cup(),))
        state = state.advance_days()
        run = state.tournament("cup")
        self.assertIsNone(run.own_team_id)
        self.assertEqual(len(run.results), 2)
        self.assertEqual(run.last_match_date, "2026-01-06")
        with self.assertRaisesRegex(SeasonSaveError, "締切"):
            state.with_tournament_entry("cup", state.selected_team_id)
        state = state.with_trained_player(OWN[0], "research")
        self.assertEqual(len(state.tournament("cup").results), 3)
        state = state.with_scouted_player(self.free_names(state)[0], "year1")
        self.assertEqual(len(state.tournament("cup").results), 4)
        build_scrim_request(state, state.selected_team_id, state.opponent_teams[0].id, seed=1)

    def test_player_entry_conditions_do_not_cancel_rival_calendar(self):
        cup = self.cup(appearance_conditions={"min_money": 1_000_000_000})
        state = self.state(tournament_definitions=(cup,))
        self.assertNotIn(cup, state.visible_tournaments)
        state = state.advance_days(4)
        self.assertIsNone(state.tournament("cup").own_team_id)
        self.assertEqual(len(state.tournament("cup").results), 1)

    def test_explicit_decline_and_mandatory_missed_registration_both_run_npc_only(self):
        state = self.state(tournament_definitions=(self.cup(),)).with_declined_tournament("cup")
        state = state.advance_days(4)
        self.assertTrue(state.tournament("cup").declined)
        state = state.advance_days()
        self.assertFalse(state.tournament("cup").declined)
        self.assertIsNone(state.tournament("cup").own_team_id)
        mandatory = self.state(tournament_definitions=(self.cup(participation_optional=False),)).advance_days(5)
        self.assertIsNone(mandatory.tournament("cup").own_team_id)

    def test_automatic_tournament_prizes_and_reload_never_replay_results(self):
        state = self.state(tournament_definitions=(self.cup(),)).advance_days(4)
        self.store.save(state)
        reloaded = self.store.load_or_create()
        self.assertEqual(reloaded, state)
        updated = reloaded.advance_days(5)
        run = updated.tournament("cup")
        self.assertTrue(run.completed)
        self.assertEqual(run.completed_date, "2026-01-10")
        self.assertEqual(updated.money, state.money)
        self.assertEqual(sum(c.money for c in updated.opponent_teams) - sum(c.money for c in state.opponent_teams),
                         sum(self.cup().prizes.values()))
        self.assertEqual(len(run.results), 6)
        self.assertEqual(updated.advance_days().tournament("cup"), run)

    def test_batched_calendar_matches_daily_calls_with_concurrent_events(self):
        state = self.state(tournament_definitions=(self.cup(), self.cup(id="cup2", start_date="2026-01-07")))
        daily = state
        for _ in range(12):
            daily = daily.advance_days()
        together = state.advance_days(12)
        self.assertEqual(together, daily)
        self.assertEqual(together.history, daily.history)

    def test_large_automatic_field_uses_only_real_clubs(self):
        state = self.state(tournament_definitions=(self.cup(team_count=16),)).advance_days(5)
        self.assertEqual(len(state.tournament("cup").entrants), 4)
        state.validate()

    def test_old_save_on_start_date_still_plays_exactly_one_series_per_date(self):
        state = self.state(tournament_definitions=(self.cup(),))
        state = replace(state, game_date="2026-01-05")
        updated = state.with_trained_player(OWN[0], "research")
        run = updated.tournament("cup")
        self.assertEqual(len(run.results), 2)
        self.assertEqual(run.last_match_date, "2026-01-06")
        registration = next(e for e in updated.history if e["種別"] == "大会自動開催（自チーム不参加）")
        self.assertEqual(len(registration["ペア練度"]), len(state.opponent_teams) + 1)

    def test_scout_counter_roundtrip_old_save_and_malformed_data(self):
        state = self.state().advance_days(59)
        state = state.with_scouted_player(self.free_names(state)[0], "year1")
        self.store.save(state)
        self.assertEqual(self.store.load_or_create().scout_remaining(), 1)
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        payload["version"] = 21
        payload.pop("scout_uses")
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(self.store.load_or_create().scout_remaining(), 2)
        payload["version"] = 22
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SeasonSaveError, "スカウト実績"):
            self.store.load_or_create()

    def test_home_remaining_count_and_tournament_controls(self):
        from run_realtime_season import RealtimeSeasonApp
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = self.state(tournament_definitions=(self.cup(),))
        app = RealtimeSeasonApp(root, self.store, state)
        self.assertIn("無制限（オフシーズン）", app.home_summary.get())
        state = state.with_tournament_entry("cup", state.selected_team_id).advance_days(4)
        app.state = state
        app.refresh()
        app.show_screen("research")
        app.training_players["research"].selection_set(OWN[0])
        app.refresh_training_offer("research")
        self.assertEqual(str(app.training_buttons["research"]["state"]), "disabled")
        self.assertIn("出場中", app.training_summaries["research"].get())
        app.show_screen("competitions")
        app.competition_list.selection_set("cup")
        app.preview_competition()
        self.assertEqual(str(app.competition_enter_button["state"]), "disabled")
        app.state = self.state().advance_days(59)._with_scout_use(state.club_id)
        app.refresh()
        self.assertIn("残り1回", app.home_summary.get())


if __name__ == "__main__":
    unittest.main()
