"""Entry deadline prompts preserve completed actions and their pending calendar day."""

from dataclasses import replace
from types import SimpleNamespace
import json
import tkinter as tk
import unittest
from unittest.mock import patch

from realtime_season import SeasonSaveError
from run_realtime_season import RealtimeSeasonApp
from season.season_competitions import SeriesScore, next_match
import test_season_scout_calendar as scout_fixtures
from test_season_competitions import OWN


class EntryWarningTests(unittest.TestCase):
    setUp = scout_fixtures.ScoutCalendarTests.setUp
    state = scout_fixtures.ScoutCalendarTests.state
    cup = scout_fixtures.ScoutCalendarTests.cup
    free_names = scout_fixtures.ScoutCalendarTests.free_names

    def app(self, state=None):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        state = state or self.deadline_state()
        self.store.save(state)
        return RealtimeSeasonApp(root, self.store, state)

    def deadline_state(self):
        return self.state(tournament_definitions=(self.cup(),)).advance_days(4)

    def test_skip_no_opens_and_selects_joinable_cup_without_changing_date(self):
        app = self.app()
        before = app.state
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=False) as prompt:
            app.advance_calendar(1)
        prompt.assert_called_once()
        self.assertIn("エントリー期限が今日までの不参加の大会がありますが、明日に進んでよろしいですか？", prompt.call_args.args[1])
        self.assertEqual(app.state, before)
        self.assertEqual(app.current_screen, "competitions")
        self.assertEqual(app.competition_list.selection(), ("cup",))
        self.assertEqual(str(app.competition_enter_button["state"]), "normal")
        app.enter_competition()
        self.assertEqual(app.state.tournament("cup").own_team_id, app.state.club_id)
        self.assertTrue(app.state.day_action_blocked)
        self.assertEqual(self.store.load_or_create(), app.state)

    def test_skip_yes_closes_entry_and_runs_one_series_per_date(self):
        app = self.app()
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=True) as prompt:
            app.advance_calendar(1)
        prompt.assert_called_once()
        run = app.state.tournament("cup")
        self.assertIsNone(run.own_team_id)
        self.assertEqual(app.state.game_date, "2026-01-06")
        self.assertEqual(len(run.results), 2)
        self.assertEqual(run.last_match_date, "2026-01-06")
        with self.assertRaisesRegex(SeasonSaveError, "締切"):
            app.state.with_tournament_entry("cup", app.state.selected_team_id)
        with patch("season.season_competition_ui.messagebox.askyesno") as prompt:
            app.advance_calendar(1)
        prompt.assert_not_called()
        self.assertEqual(len(app.state.tournament("cup").results), 3)

    def test_batch_and_month_advance_stop_at_intermediate_deadline(self):
        for month in (False, True):
            with self.subTest(month=month):
                app = self.app(self.state(tournament_definitions=(self.cup(),)))
                with patch("season.season_competition_ui.messagebox.askyesno", return_value=False):
                    app.advance_game_month() if month else app.advance_calendar(12)
                self.assertEqual(app.state.game_date, "2026-01-05")
                self.assertIsNone(app.state.tournament("cup"))
                self.assertEqual(self.store.load_or_create(), app.state)

    def test_same_day_events_share_one_prompt_and_second_deadline_is_not_skipped(self):
        state = self.state(tournament_definitions=(self.cup(), self.cup(id="cup2", name="Second"),
                                                  self.cup(id="cup3", name="Third", start_date="2026-01-07")))
        app = self.app(state)
        with patch("season.season_competition_ui.messagebox.askyesno", side_effect=(True, False)) as prompt:
            app.advance_calendar(12)
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("Second", prompt.call_args_list[0].args[1])
        self.assertEqual(app.state.game_date, "2026-01-07")
        self.assertIsNotNone(app.state.tournament("cup"))
        self.assertIsNotNone(app.state.tournament("cup2"))
        self.assertIsNone(app.state.tournament("cup3"))
        self.assertEqual(app.competition_list.selection(), ("cup3",))

    def test_declined_event_can_be_joined_after_no(self):
        state = self.state(tournament_definitions=(self.cup(),)).with_declined_tournament("cup").advance_days(4)
        app = self.app(state)
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=False):
            app.advance_calendar(1)
        app.enter_competition()
        run = app.state.tournament("cup")
        self.assertFalse(run.declined)
        self.assertEqual(run.own_team_id, app.state.club_id)
        self.assertEqual(len(app.state.tournaments), 1)

    def test_registered_or_unavailable_events_do_not_warn(self):
        entered = self.deadline_state()
        entered = entered.with_tournament_entry("cup", entered.selected_team_id)
        for state in (entered,
                      self.state(tournament_definitions=(self.cup(allow_player_entry=False),)).advance_days(4),
                      self.state(tournament_definitions=(self.cup(appearance_conditions={"min_money": 10**12}),)).advance_days(4)):
            app = self.app(state)
            with patch("season.season_competition_ui.messagebox.askyesno") as prompt:
                app.advance_calendar(1)
            prompt.assert_not_called()

    def test_scout_is_saved_before_prompt_and_no_does_not_undo_contract(self):
        app = self.app()
        before = app.state
        name = self.free_names(before)[0]
        terms = before.contract_terms(next(p for p in before.scout_players if p.name == name), "year1")
        app.scout_players.selection_set(name)
        def refuse(*args, **kwargs):
            saved = self.store.load_or_create()
            self.assertIsNotNone(saved.player(name))
            self.assertEqual(saved.money, before.money - terms.signing_bonus)
            self.assertEqual(saved.game_date, before.game_date)
            self.assertTrue(saved.day_advance_pending)
            self.assertEqual(len(saved.scout_uses), 1)
            return False
        with patch("season.season_competition_ui.messagebox.askyesno", side_effect=refuse) as prompt:
            app.sign_selected_contract("scout")
        prompt.assert_called_once()
        self.assertEqual(app.state.game_date, before.game_date)
        self.assertIsNotNone(app.state.player(name))
        reloaded = self.store.load_or_create()
        self.assertEqual(reloaded, app.state)
        self.assertTrue(reloaded.day_action_blocked)
        with self.assertRaisesRegex(SeasonSaveError, "1日進行"):
            reloaded.with_scouted_player(self.free_names(reloaded)[0], "year1")
        with self.assertRaisesRegex(SeasonSaveError, "1日進行"):
            reloaded.with_trained_player(OWN[0], "research")
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=True):
            app.advance_calendar(1)
        self.assertFalse(app.state.day_advance_pending)
        self.assertEqual(len(app.state.scout_uses), 1)
        self.assertEqual(app.state.money, before.money - terms.signing_bonus)
        self.assertAlmostEqual(app.state.player(name).iq, before.displayed_player(next(p for p in before.scout_players if p.name == name)).iq + .1)

    def test_training_both_kinds_saved_before_prompt_and_yes_spends_exactly_one_day(self):
        for kind in ("research", "aim_lab"):
            with self.subTest(kind=kind):
                app = self.app()
                before = app.state
                terms = before.training_terms(OWN[0], kind)
                app.training_players[kind].selection_set(OWN[0])
                def accept(*args, **kwargs):
                    saved = self.store.load_or_create()
                    self.assertEqual(saved.game_date, before.game_date)
                    self.assertEqual(getattr(saved.player(OWN[0]), terms.level_field), 1)
                    self.assertEqual(saved.money, before.money - terms.cost)
                    return True
                with patch("season.season_competition_ui.messagebox.askyesno", side_effect=accept) as prompt:
                    app.train_selected_player(kind)
                prompt.assert_called_once()
                self.assertEqual(app.state.game_date, "2026-01-06")
                self.assertFalse(app.state.day_advance_pending)
                self.assertEqual(getattr(app.state.player(OWN[0]), terms.level_field), 1)
                self.assertEqual(app.state.money, before.money - terms.cost)

    def test_scout_no_then_join_and_play_preserves_paid_action(self):
        app = self.app()
        name = self.free_names(app.state)[0]
        app.scout_players.selection_set(name)
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=False):
            app.sign_selected_contract("scout")
        app.enter_competition()
        self.assertTrue(app.state.day_advance_pending)
        match, _ = next_match(self.cup(), app.state.tournament("cup"))
        state = app.state.with_tournament_result("cup", SeriesScore(match.id, match.left, match.right, match.maps_to_win, 0))
        app.commit(state, "試合結果")
        app._after_competition_result("cup")
        self.assertEqual(app.state.game_date, "2026-01-06")
        self.assertFalse(app.state.day_advance_pending)
        self.assertIsNotNone(app.state.player(name))
        self.assertEqual(len(app.state.scout_uses), 1)

    def test_scrim_results_saved_and_job_finished_before_prompt(self):
        app = self.app()
        before = app.state
        rival = before.opponent_teams[0]
        result = dict(status="completed", own_team=before.team_name, opponent_team=rival.name,
                      own_score=13, opponent_score=8, winner=before.team_name)
        app.scrim_job = SimpleNamespace(poll=lambda: result)
        app._scrim_rating_context = ("scrim:deadline", before.club_id, rival.id)
        def refuse(*args, **kwargs):
            self.assertIsNone(app.scrim_job)
            saved = self.store.load_or_create()
            self.assertIn("scrim:deadline", saved.rated_results)
            self.assertEqual(saved.game_date, before.game_date)
            self.assertNotEqual(saved.rating(before.club_id), before.rating(before.club_id))
            return False
        with patch("season.season_competition_ui.messagebox.askyesno", side_effect=refuse) as prompt:
            app.poll_scrim()
        prompt.assert_called_once()
        self.assertTrue(app.state.day_advance_pending)
        self.assertEqual(app.current_screen, "competitions")
        self.assertEqual(app.state.with_scrim_result("scrim:deadline", before.club_id, rival.id, 1, 0), app.state)
        with patch("season.season_competition_ui.messagebox.askyesno") as prompt:
            app.poll_scrim()
        prompt.assert_not_called()

    def test_pair_contract_stops_after_first_preserved_acquisition_when_no(self):
        app = self.app(replace(self.deadline_state(), money=10**12))
        first, second = app.state.opponent_teams[0].members[:2]
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=False) as prompt:
            app.sign_player_pair(first, second)
        prompt.assert_called_once()
        self.assertIsNotNone(app.state.player(first))
        self.assertIsNone(app.state.player(second))
        self.assertEqual(len(app.state.scout_uses), 1)
        self.assertTrue(app.state.day_advance_pending)

    def test_failed_action_and_failed_save_do_not_prompt(self):
        app = self.app(replace(self.deadline_state(), money=0))
        app.training_players["research"].selection_set(OWN[0])
        with patch("season.season_competition_ui.messagebox.askyesno") as prompt:
            app.train_selected_player("research")
        prompt.assert_not_called()
        app.state = self.deadline_state()
        with patch.object(app, "commit", return_value=False), patch("season.season_competition_ui.messagebox.askyesno") as prompt:
            app.train_selected_player("research")
        prompt.assert_not_called()
        self.assertFalse(app.state.day_advance_pending)

    def test_pending_roundtrip_old_save_defaults_and_invalid_flag_rejected(self):
        state = self.deadline_state().with_trained_player(OWN[0], "research", advance_day=False)
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)
        payload = json.loads(self.store.path.read_text(encoding="utf-8"))
        payload["day_advance_pending"] = "yes"
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(SeasonSaveError, "日付進行待ち"):
            self.store.load_or_create()
        payload["version"] = 23
        payload.pop("day_advance_pending")
        self.store.path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertFalse(self.store.load_or_create().day_advance_pending)

    def test_month_end_no_postpones_settlement_until_pending_day_is_consumed_once(self):
        state = self.state(tournament_definitions=(self.cup(start_date="2026-01-31"),)).advance_days(30)
        app = self.app(state)
        terms = state.training_terms(OWN[0], "research")
        app.training_players["research"].selection_set(OWN[0])
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=False):
            app.train_selected_player("research")
        self.assertEqual(app.state.game_date, "2026-01-31")
        self.assertEqual(app.state.money, state.money - terms.cost)
        self.assertEqual(app.state.game_month, 0)
        restarted = self.app(self.store.load_or_create())
        self.assertIn("1日進行待ち", restarted.home_summary.get())
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=True):
            restarted.advance_calendar(1)
        self.assertEqual(restarted.state.game_date, "2026-02-01")
        self.assertEqual(restarted.state.money, state.money - terms.cost
                         + state.monthly_sponsor_income - state.monthly_payroll)
        self.assertEqual(sum(e.kind == "month_completed" for e in restarted.state.monthly_events), 1)
        money = restarted.state.money
        restarted.advance_calendar(1)
        self.assertEqual(restarted.state.money, money)
        self.assertEqual(restarted.state.player(OWN[0]).research_level, 1)

    def test_failed_day_save_retains_completed_action_and_retry_does_not_charge_again(self):
        app = self.app()
        before = app.state
        terms = before.training_terms(OWN[0], "research")
        app.training_players["research"].selection_set(OWN[0])
        save = self.store.save
        def save_action_only(candidate):
            if candidate.date > before.date:
                raise OSError("calendar save failure")
            save(candidate)
        with patch.object(self.store, "save", side_effect=save_action_only), \
                patch("run_realtime_season.messagebox.showerror"), \
                patch("season.season_competition_ui.messagebox.askyesno", return_value=True):
            app.train_selected_player("research")
        self.assertTrue(app.state.day_advance_pending)
        self.assertEqual(app.state.game_date, before.game_date)
        self.assertEqual(app.state.player(OWN[0]).research_level, 1)
        self.assertEqual(self.store.load_or_create(), app.state)
        with patch("season.season_competition_ui.messagebox.askyesno", return_value=True):
            app.advance_calendar(1)
        self.assertFalse(app.state.day_advance_pending)
        self.assertEqual(app.state.money, before.money - terms.cost)
        self.assertEqual(app.state.player(OWN[0]).research_level, 1)


if __name__ == "__main__":
    unittest.main()
