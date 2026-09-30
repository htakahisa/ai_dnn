"""Tick timing reaches the game scheduler without changing battle tick rules."""

import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from game_core import TICK_TIME, validate_tick_time_ms
from run_game import VisualFPSBattle
from run_competition_manager import CompetitionApp


class GameTickTimeTests(unittest.TestCase):
    def game(self):
        game = VisualFPSBattle.__new__(VisualFPSBattle)
        game.root = SimpleNamespace(after=Mock(), mainloop=Mock())
        game.headless = False
        game.draw = Mock()
        game.match_over = game.round_over = False
        game.defender_setup_phase = SimpleNamespace(active=True)
        game._run_defender_setup_tick = Mock()
        game._record_replay_frame = Mock()
        return game

    def test_tick_time_accepts_positive_integer_milliseconds(self):
        self.assertEqual(validate_tick_time_ms(" 250 "), 250)
        for value in ("", "abc", "0", "-1", "1.5", 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_tick_time_ms(value)

    def test_initial_and_setup_ticks_use_current_setting(self):
        game = self.game()
        timing = SimpleNamespace(value=250)
        game.tick_time_ms = lambda: timing.value
        game.run()
        game.root.after.assert_called_with(250, game.loop)
        timing.value = 40
        game.loop()
        game.root.after.assert_called_with(40, game.loop)
        game._run_defender_setup_tick.assert_called_once()

    def test_battle_ticks_use_the_same_delay(self):
        game = self.game()
        game.tick_time_ms = 350
        game.defender_setup_phase.active = False
        game.chars = []
        game._build_occupancy_counts = Mock()
        game._clear_occupancy_counts = Mock()
        game._move_order = lambda: []
        game.process_battle = Mock()
        game._advance_combo_announcement = Mock()
        game.loop()
        game.process_battle.assert_called_once()
        game.root.after.assert_called_with(350, game.loop)

    def test_round_transition_uses_setting_and_updates_animation_time(self):
        game = self.game()
        game.tick_time_ms = 250
        game.round_transition_ticks_left = 2
        game.explosion_effect = None
        game.special_round_banner = {"effect": {}, "animation_elapsed_ms": 100}
        game._advance_round_transition()
        game.root.after.assert_called_with(250, game._advance_round_transition)
        self.assertEqual(game.special_round_banner["animation_elapsed_ms"], 350)
        self.assertEqual(game.round_transition_ticks_left, 1)

    def test_default_timing_and_headless_execution(self):
        game = self.game()
        self.assertEqual(game._tick_delay_ms(), TICK_TIME)
        game.headless = True
        game.tick_time_ms = 2000
        game.run_headless_loop = Mock()
        game.run()
        game.run_headless_loop.assert_called_once()
        game.root.after.assert_not_called()

    def test_manager_ignores_incomplete_input_and_applies_valid_edits(self):
        app = CompetitionApp.__new__(CompetitionApp)
        app.tick_time_ms = 100
        app.tick_time_var = SimpleNamespace(get=lambda: "")
        app._on_tick_time_change()
        self.assertEqual(app.tick_time_ms, 100)
        app.tick_time_var = SimpleNamespace(get=lambda: "275")
        app._on_tick_time_change()
        self.assertEqual(app.tick_time_ms, 275)

    def test_render_request_passes_a_live_tick_setting_to_game(self):
        app = CompetitionApp.__new__(CompetitionApp)
        app.tick_time_ms = 250
        app.render_requests = queue.Queue()
        request = {key: None for key in (
            "team1", "team2", "map_number", "seed", "team1_controller_key",
            "team2_controller_key", "team1_series_wins", "team2_series_wins",
            "series_maps_to_win", "mental_fatigue_state")}
        request["done"] = threading.Event()
        app.render_requests.put(request)
        with patch("run_competition_manager.play_map", return_value="result") as play:
            app._process_render_requests()
            tick_setting = play.call_args.kwargs["tick_time_ms"]
            self.assertEqual(tick_setting(), 250)
            app.tick_time_ms = 60
            self.assertEqual(tick_setting(), 60)
        self.assertEqual(request["result"], "result")
        self.assertTrue(request["done"].is_set())


if __name__ == "__main__":
    unittest.main()
