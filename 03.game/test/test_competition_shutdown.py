"""Closing a competition cancels work and cleans up match timers."""

import gc
import queue
import threading
import tkinter as tk
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import run_competition_manager as manager
from battle_logic import BattleLogicMixin


class CompetitionShutdownTests(unittest.TestCase):
    def app(self):
        app = manager.CompetitionApp.__new__(manager.CompetitionApp)
        app.root = Mock()
        app.root.tk.splitlist.return_value = ()
        app.stop_event = threading.Event()
        app._closing = False
        app._active_game = None
        app.render_requests = queue.Queue()
        app.events = queue.Queue()
        app.worker = None
        app.live_render_enabled = True
        return app

    def test_close_releases_queued_worker_and_is_repeatable(self):
        app = self.app()
        errors = []

        def work():
            try:
                app._render_controller.play_map(None, None, 1, 42,
                                                "default", "default", 0, 0, 1)
            except manager.CompetitionCancelled:
                errors.append("cancelled")

        worker = threading.Thread(target=work)
        worker.start()
        request = app.render_requests.get(timeout=2)
        app.render_requests.put(request)
        app.close()
        app.close()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, ["cancelled"])
        self.assertTrue(request["done"].is_set())
        self.assertTrue(app.stop_event.is_set())
        app.root.quit.assert_called_once()
        self.assertTrue(app.events.empty())

    def test_headless_loop_stops_at_next_tick(self):
        stop = threading.Event()
        game = SimpleNamespace(match_over=False, round_over=False)

        def tick():
            stop.set()
            return True

        game.step_tick = Mock(side_effect=tick)
        BattleLogicMixin.run_headless_loop(game, stop_event=stop)
        game.step_tick.assert_called_once()
        self.assertFalse(game.match_over)

    def test_render_wait_stops_even_if_request_arrives_after_close(self):
        app = self.app()
        app.render_requests = SimpleNamespace(put=lambda request: app.stop_event.set())
        with self.assertRaises(manager.CompetitionCancelled):
            app._render_controller.play_map(None, None, 1, 42,
                                            "default", "default", 0, 0, 1)

    def test_close_during_render_skips_ui_updates_and_timer_reschedule(self):
        app = self.app()
        app._process_render_requests = Mock(side_effect=app.close)
        app._poll_events()
        app.root.after.assert_not_called()
        self.assertTrue(app.events.empty())

    def test_cancelled_job_does_not_save_results_or_report_error(self):
        app = self.app()
        with patch.object(manager, "run_series_core", side_effect=manager.CompetitionCancelled), \
                patch.object(manager, "save_json") as save:
            app._execute_job(("series", ["one", "two"]), 1, 1, 1, {}, {},
                             "fixed", 42, app._render_controller)
        save.assert_not_called()
        self.assertTrue(app.events.empty())

    def test_match_close_cancels_timers_and_destroys_window(self):
        app = self.app()
        game = Mock(match_over=False)
        handlers = {}
        game.root.protocol.side_effect = lambda name, callback: handlers.update({name: callback})
        game.root.after.return_value = "watch-timer"
        game.run.side_effect = lambda **kwargs: handlers["WM_DELETE_WINDOW"]()
        team = SimpleNamespace(name="one", players=["player"],
                               spike_holder="player", igl="player")
        with patch.object(manager, "VisualFPSBattle", return_value=game), \
                patch.object(manager, "_build_team_ai"), \
                patch.object(manager, "seed_all"), \
                patch.object(manager, "cpu_inference", return_value=nullcontext()), \
                patch.object(manager, "original_scores") as scores:
            with self.assertRaises(manager.CompetitionCancelled):
                manager.play_map(team, team, 1, 42, True,
                                 stop_event=app.stop_event, on_close=app.close,
                                 on_game_ready=app._set_active_game)
        self.assertTrue(app._closing)
        game.stop_playback.assert_called()
        game.root.after_cancel.assert_called_with("watch-timer")
        game.root.destroy.assert_called_once()
        scores.assert_not_called()

    def test_run_waits_for_worker_before_destroying_manager(self):
        app = self.app()
        order = []
        app.worker = Mock()
        app.worker.join.side_effect = lambda: order.append("joined")
        app.root.destroy.side_effect = lambda: order.append("destroyed")
        app.run()
        self.assertTrue(app.stop_event.is_set())
        self.assertEqual(order, ["joined", "destroyed"])

    def test_actual_tk_close_unwinds_nested_match_and_stops_worker(self):
        for close_target in ("manager", "match", "headless"):
            with self.subTest(close_target=close_target):
                app = self.app()
                try:
                    app.root = tk.Tk()
                except tk.TclError as exc:
                    self.skipTest(f"Tk unavailable: {exc}")
                app.root.withdraw()
                app.root.update_idletasks()
                errors, outcome, games = [], [], []
                app.root.report_callback_exception = lambda *args: errors.append(args)
                app.root.protocol("WM_DELETE_WINDOW", app.close)
                app.live_render_enabled = close_target != "headless"
                app.tick_time_ms = 50
                teams = [SimpleNamespace(name=name, players=players,
                                         spike_holder=players[0], igl=players[0])
                         for name, players in (
                             ("one", ["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"]),
                             ("two", ["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"]))]

                def hidden_root():
                    root = original_tk()
                    root.withdraw()
                    root.report_callback_exception = lambda *args: errors.append(args)
                    return root

                def ready(game):
                    app._active_game = game
                    games.append((game, game.root))
                    target = app.root if close_target == "manager" else game.root
                    game.root.after(10, lambda: target.tk.call(target.protocol("WM_DELETE_WINDOW")))

                def work():
                    try:
                        app._render_controller.play_map(*teams, 1, 42,
                                                        "default", "default", 0, 0, 1)
                    except manager.CompetitionCancelled:
                        outcome.append("cancelled")
                    except BaseException as exc:
                        errors.append(exc)

                def timeout():
                    errors.append("shutdown timed out")
                    app.close()

                original_tk = tk.Tk
                with patch("run_game.tk.Tk", side_effect=hidden_root), \
                        patch.object(app, "_set_active_game", side_effect=ready):
                    app.worker = threading.Thread(target=work, daemon=True)
                    app.worker.start()
                    app.root.after(0, app._poll_events)
                    app.root.after(5000, timeout)
                    if close_target == "headless":
                        app.root.after(30, app.close)
                    app.run()
                self.assertFalse(app.worker.is_alive())
                self.assertEqual(outcome, ["cancelled"])
                self.assertEqual(errors, [])
                for game, root in games:
                    with self.assertRaises(tk.TclError):
                        root.winfo_exists()
                    self.assertIsNone(game.root)
                # Collect test fixtures with Tcl interpreters on the Tk thread.
                app.worker = None
                gc.collect()

    def test_actual_tk_completed_maps_leave_manager_responsive(self):
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        root.withdraw()
        root.update_idletasks()
        original_tk = tk.Tk
        games, errors = [], []

        def hidden_root():
            window = original_tk()
            window.withdraw()
            window.report_callback_exception = lambda *args: errors.append(args)
            return window

        def ready(game):
            games.append(game)
            # Exercise automatic completion independently of match duration.
            game.match_over = True

        teams = [SimpleNamespace(name=name, players=players,
                                 spike_holder=players[0], igl=players[0])
                 for name, players in (
                     ("one", ["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"]),
                     ("two", ["Demon1", "jawgemo", "Ethan", "Boostio", "C0M"]))]
        try:
            with patch("run_game.tk.Tk", side_effect=hidden_root):
                for number in (1, 2):
                    result = manager.play_map(*teams, number, 42, True,
                                              "default", "default", on_game_ready=ready)
                    self.assertEqual(result.number, number)
                    root.update()
            self.assertEqual(errors, [])
            self.assertEqual(root.tk.call("after", "info"), "")
            self.assertTrue(root.winfo_exists())
            self.assertTrue(all(game.root is None for game in games))
        finally:
            root.destroy()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
