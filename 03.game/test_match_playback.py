"""Playback controls stop at real round/side/map boundaries without losing state."""

import contextlib
import io
import random
import tkinter as tk
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from map_data import NEW_MAZE_STR
from run_game import VisualFPSBattle, _build_team_ai


class Scheduler:
    def __init__(self):
        self.pending = {}
        self.serial = 0

    def after(self, delay, callback):
        self.serial += 1
        self.pending[self.serial] = (delay, callback)
        return self.serial

    def after_cancel(self, handle):
        self.pending.pop(handle, None)

    def advance(self):
        handle = min(self.pending)
        _, callback = self.pending.pop(handle)
        callback()


class MatchPlaybackTests(unittest.TestCase):
    def game(self, *, rendered=False, round_number=1, overtime=False):
        random.seed(17)
        np.random.seed(17)
        with contextlib.redirect_stdout(io.StringIO()):
            game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"), headless=True,
                attacker_roster=["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"],
                defender_roster=["Aspas", "valyn", "trent", "leaf", "tex"],
                attacker_team_name="Own", defender_team_name="Rival",
            )
        game.record_replay = False
        if round_number != 1:
            game.current_round = round_number
            game.overtime = overtime
            game.attacker_wins, game.defender_wins = (12, 12) if overtime else (6, 5)
            game.init_round()
        if rendered:
            game.headless = False
            game.root = Scheduler()
            game.draw = Mock()
            game.label = Mock()
        return game

    @staticmethod
    def snapshot(game):
        return (game.current_round, game.attacker_wins, game.defender_wins,
                game.attacker_team_name, game._side_swap_count, game.match_stats,
                [(c.name, tuple(c.pos), c.hp, c.ultimate_points) for c in game.chars])

    def test_fast_round_side_overtime_and_map_match_headless_simulation(self):
        for scope, start_round, overtime in (("round", 1, False), ("side", 12, False), ("side", 25, True), ("map", 1, False)):
            with self.subTest(scope=scope, start=start_round), patch("battle_logic.WINNING_ROUNDS", 2 if scope == "map" else 13), contextlib.redirect_stdout(io.StringIO()):
                baseline = self.game(round_number=start_round, overtime=overtime)
                start_side = baseline._side_swap_count
                for _ in range(20000):
                    if baseline.match_over or (scope == "round" and baseline.current_round > start_round) or (scope == "side" and baseline._side_swap_count != start_side):
                        break
                    baseline._simulate_tick()
                else:
                    self.fail("Baseline did not reach its boundary")
                expected = self.snapshot(baseline)
                game = self.game(rendered=True, round_number=start_round, overtime=overtime)
                game.start_fast_forward(scope)
                for _ in range(10000):
                    if game.fast_forward_mode is None:
                        break
                    game.root.advance()
                else:
                    self.fail("Fast forward did not stop")
                self.assertFalse(game.headless)
                self.assertEqual(self.snapshot(game), expected)
                game.draw.assert_called_once()
                self.assertLessEqual(len(game.root.pending), 1)

    def test_pause_stops_ticks_and_resume_keeps_single_timer(self):
        game = self.game(rendered=True)
        game._simulate_tick = Mock()
        game._schedule_match_callback(game.loop)
        game.toggle_pause()
        self.assertTrue(game.paused)
        self.assertFalse(game.root.pending)
        game.loop()
        game._simulate_tick.assert_not_called()
        game.toggle_pause()
        self.assertFalse(game.paused)
        self.assertEqual(len(game.root.pending), 1)
        game.root.advance()
        game._simulate_tick.assert_called_once()
        self.assertEqual(len(game.root.pending), 1)

    def test_tick_change_immediately_replaces_pending_battle_or_transition_timer(self):
        game = self.game(rendered=True)
        for callback in (game.loop, game._advance_round_transition):
            game._schedule_match_callback(callback)
            old_handle = game._match_after_id
            game.set_tick_time_ms("35")
            self.assertNotIn(old_handle, game.root.pending)
            self.assertEqual(list(game.root.pending.values()), [(35, callback)])
        game.toggle_pause()
        game.set_tick_time_ms(12)
        self.assertFalse(game.root.pending)
        game.toggle_pause()
        self.assertEqual(list(game.root.pending.values()), [(12, game._advance_round_transition)])
        with self.assertRaises(ValueError):
            game.set_tick_time_ms("0")
        self.assertEqual(game.tick_time_ms, 12)

    def test_pause_during_round_transition_or_fast_forward_preserves_callback(self):
        game = self.game(rendered=True)
        game.round_transition_ticks_left = 3
        game._schedule_match_callback(game._advance_round_transition)
        game.toggle_pause()
        game._advance_round_transition()
        self.assertEqual(game.round_transition_ticks_left, 3)
        game.toggle_pause()
        self.assertEqual(next(iter(game.root.pending.values()))[1], game._advance_round_transition)
        game.start_fast_forward("map")
        game.toggle_pause()
        game._run_fast_forward_chunk()
        self.assertEqual(game.current_round, 1)
        self.assertEqual(game.fast_forward_mode, "map")
        game.toggle_pause()
        self.assertEqual(next(iter(game.root.pending.values()))[1], game._run_fast_forward_chunk)

    def test_skip_during_transition_returns_at_next_round_without_simulating_it(self):
        game = self.game(rendered=True)
        game.current_round = 2
        game.round_over = True
        game._simulate_tick = Mock()
        game.start_fast_forward("round")
        game.root.advance()
        game._simulate_tick.assert_not_called()
        self.assertEqual(game.current_round, 2)
        self.assertFalse(game.round_over)
        self.assertIsNone(game.fast_forward_mode)

    def test_actual_tk_match_window_controls_fast_forward_and_finish(self):
        root = tk.Tk()
        root.withdraw()
        game = None
        try:
            with patch("run_game.tk.Tk", return_value=root), patch("battle_logic.WINNING_ROUNDS", 2), contextlib.redirect_stdout(io.StringIO()):
                game = VisualFPSBattle(
                    NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"), headless=False,
                    attacker_roster=["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"],
                    defender_roster=["Aspas", "valyn", "trent", "leaf", "tex"],
                )
                game.record_replay = False
                game.playback_tick_var.set("17")
                game.apply_tick_time()
                self.assertEqual(game.tick_time_ms, 17)
                game.pause_button.invoke()
                self.assertTrue(game.paused)
                game.skip_buttons[2].invoke()
                self.assertFalse(game.paused)
                deadline = time.monotonic() + 30
                while not game.match_over and time.monotonic() < deadline:
                    root.update()
                self.assertTrue(game.match_over)
                self.assertIsNone(game.fast_forward_mode)
                self.assertFalse(game.headless)
                self.assertEqual(game.playback_status.get(), "試合終了")
                self.assertEqual(str(game.pause_button["state"]), "disabled")
        finally:
            if game is not None:
                game.stop_playback()
            root.destroy()


if __name__ == "__main__":
    unittest.main()
