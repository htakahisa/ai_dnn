"""NPC season series use Elo draws even while the player's club survives."""

from dataclasses import replace
from datetime import timedelta
import unittest
from unittest.mock import patch

from realtime_season import SeasonSaveError
from season_competitions import next_match, player_eliminated
from season_ratings import expected_score, series_ratings
from test_season_competitions import (
    CompetitionScreenTest, SeasonCompetitionTest, config, definition, record_next,
)


class EloProbabilityTest(unittest.TestCase):
    def test_equal_ratings_gap_symmetry_and_extreme_ratings(self):
        self.assertEqual(expected_score(1500, 1500), .5)
        self.assertAlmostEqual(expected_score(1900, 1500), 10 / 11)
        self.assertAlmostEqual(expected_score(1500, 1900), 1 / 11)
        self.assertEqual(expected_score(0, 200000), 0)
        self.assertEqual(expected_score(200000, 0), 1)
        left, right = series_ratings(1900, 1500, 1, 0)
        self.assertAlmostEqual(left, 1900 + 64 * (1 - 10 / 11))
        self.assertAlmostEqual(right, 1500 - 64 * (1 - 10 / 11))


class NpcSeriesTest(SeasonCompetitionTest):
    def npc_day(self):
        return record_next(self.entered()).advance_days()

    def test_draw_threshold_uses_current_ratings_without_map_probability_conversion(self):
        state = self.npc_day()
        event, run = state.tournament_definition("cup"), state.tournament("cup")
        match, _ = next_match(event, run)
        state = replace(state, ratings=tuple(replace(r, value=1900 if r.team_id == match.left else 1500)
                                            for r in state.ratings))
        probability = 10 / 11
        for draw, left_wins in ((probability - 1e-9, True), (probability + 1e-9, False)):
            with self.subTest(draw=draw), patch("realtime_season.Random") as rng:
                rng.return_value.random.return_value = draw
                result = state.with_tournament_rating_result("cup")
                rng.assert_called_once_with((run.seed + len(run.results) * 1000) % (2**31))
                rng.return_value.random.assert_called_once_with()
            score = result.tournament("cup").results[-1]
            self.assertEqual(score.left_wins > score.right_wins, left_wins)
            self.assertTrue(score.decided_by_rating)
            self.assertFalse(player_eliminated(event, result.tournament("cup")))
            self.assertEqual((result.rating(match.left), result.rating(match.right)),
                             series_ratings(1900, 1500, score.left_wins, score.right_wins))
            self.assertEqual(result.money, state.money)
            self.assertEqual(result.contract("Leo"), state.contract("Leo"))

    def test_equal_rating_coin_draw_has_no_seed_order_tie_break(self):
        state = self.npc_day()
        for draw, left_wins in ((.49, True), (.5, False), (.51, False)):
            with self.subTest(draw=draw), patch("realtime_season.Random") as rng:
                rng.return_value.random.return_value = draw
                score = state.with_tournament_rating_result("cup").tournament("cup").results[-1]
                self.assertEqual(score.left_wins > score.right_wins, left_wins)

    def test_reload_before_draw_reproduces_result_and_saves_next_day_atomically(self):
        state = self.npc_day()
        expected = state.with_tournament_rating_result("cup")
        self.store.save(state)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded.with_tournament_rating_result("cup"), expected)
        self.store.save(expected)
        loaded = self.store.load_or_create()
        self.assertEqual(loaded, expected)
        self.assertEqual(loaded.date, state.date + timedelta(days=1))
        self.assertEqual(loaded.tournament("cup").last_match_date, state.game_date)
        next_day = loaded.with_tournament_rating_result("cup")
        self.assertEqual(next_day.date, state.date + timedelta(days=2))
        self.assertEqual(len(next_day.tournament("cup").results), len(state.tournament("cup").results) + 2)

    def test_player_series_and_invalid_or_early_event_cannot_be_drawn(self):
        state = self.entered()
        with self.assertRaisesRegex(SeasonSaveError, "自チーム"):
            state.with_tournament_rating_result("cup")
        with self.assertRaises(SeasonSaveError):
            state.with_tournament_rating_result("missing")
        state = self.state()
        state = state.with_tournament_entry("cup", state.selected_team_id)
        with self.assertRaises(SeasonSaveError):
            state.with_tournament_rating_result("cup")

    def test_opponent_only_event_draws_daily_and_never_pays_player_prize(self):
        with patch.object(config, "TOURNAMENTS", [definition(format="single_elimination", team_count=3,
                                                            allow_player_entry=False, prizes={1: 5000000, 2: 2000000, 3: 1000000})]):
            state = self.state().advance_days(33).with_tournament_entry("cup")
        balance = state.money
        while not state.tournament("cup").completed:
            if state.tournament("cup").last_match_date == state.game_date:
                state = state.advance_days()
            state = state.with_tournament_rating_result("cup")
        run = state.tournament("cup")
        self.assertTrue(all(s.decided_by_rating for s in run.results))
        self.assertEqual(run.completed_date, "2026-02-04")
        self.assertEqual(state.game_date, "2026-02-05")
        self.assertEqual((run.prize_paid, state.money), (0, balance))
        self.store.save(state)
        self.assertEqual(self.store.load_or_create(), state)


class NpcSeriesScreenTest(CompetitionScreenTest):
    def setUp(self):
        super().setUp()
        self.addCleanup(self.cancel_pending_callback)

    def cancel_pending_callback(self):
        if self.app._competition_after_id is not None:
            self.root.after_cancel(self.app._competition_after_id)

    def select_npc_day(self):
        state = record_next(self.entered()).advance_days()
        self.app.commit(state, "fixture")
        self.app.show_screen("competitions")
        self.app.competition_list.selection_set("cup")
        self.app.preview_competition()
        return state

    def test_npc_result_never_starts_engine_even_with_render_enabled_and_invalid_tick_ms(self):
        state = self.select_npc_day()
        app = self.app
        self.assertIn("レート抽選", app.competition_play_button["text"])
        self.assertIn("50.0%", app.competition_info.get())
        app.competition_render.set(True)
        app.competition_tick_ms.set("invalid")
        with patch("season_competition_ui.ScrimJob") as worker, patch("season_competition_ui.build_series_request") as request:
            app.start_competition_series()
            worker.assert_not_called()
            request.assert_not_called()
        self.assertEqual(len(app.state.tournament("cup").results), 2)
        self.assertTrue(app.state.tournament("cup").results[-1].decided_by_rating)
        self.assertEqual(app.state.date, state.date + timedelta(days=1))
        self.assertIsNone(app.competition_job)
        self.assertEqual(self.store.load_or_create(), app.state)
        self.assertEqual(str(app.competition_play_button["state"]), "normal")

    def test_failed_npc_result_save_preserves_state_and_retry_draw(self):
        state = self.select_npc_day()
        app = self.app
        expected = state.with_tournament_rating_result("cup")
        with patch.object(self.store, "save", side_effect=OSError("test failure")), patch("run_realtime_season.messagebox.showerror"):
            app.start_competition_series()
        self.assertEqual(app.state, state)
        self.assertEqual(self.store.load_or_create(), state)
        app.start_competition_series()
        self.assertEqual(app.state, expected)

    def test_auto_npc_followed_by_player_series_uses_worker_only_for_player(self):
        self.select_npc_day()
        app = self.app
        app.competition_auto.set(True)
        with patch("season_competition_ui.ScrimJob") as worker:
            app.start_competition_series()
            worker.assert_not_called()
            self.assertEqual(app.state.game_date, "2026-02-05")
            self.root.after_cancel(app._competition_after_id)
            app._start_next_competition_series("cup")
            worker.assert_not_called()
            self.assertEqual(app.state.game_date, "2026-02-06")
            self.root.after_cancel(app._competition_after_id)
            app._start_next_competition_series("cup")
            worker.assert_called_once()
            request = worker.call_args.args[0]
            self.assertIn(app.state.club_id, (request["left_id"], request["right_id"]))
            self.assertTrue(request["render"])
        self.root.after_cancel(app._competition_after_id)
        app._competition_after_id = None
        app.competition_job = None

    def test_cancel_auto_between_instant_results_stops_scheduled_next_match(self):
        self.select_npc_day()
        app = self.app
        app.competition_auto.set(True)
        app.start_competition_series()
        state = app.state
        self.assertIsNotNone(app._competition_after_id)
        app.cancel_competition_series()
        self.assertFalse(app.competition_auto.get())
        self.assertIsNone(app._competition_after_id)
        self.root.update()
        self.assertEqual(app.state, state)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for case in (EloProbabilityTest, NpcSeriesTest, NpcSeriesScreenTest):
        suite.addTests(case(name) for name in case.__dict__ if name.startswith("test_"))
    return suite


if __name__ == "__main__":
    unittest.main()
