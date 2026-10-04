"""The manager shares season bracket styling while preserving real match data."""

from pathlib import Path
import queue
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import run_competition_manager as manager
from competition_bracket import project_bracket


def setup_for(teams):
    slots, _ = manager.build_seeded_bracket_slots(teams, len(teams), "qualifier", [], {})
    return {"slots": slots, "normal_maps_to_win": 2,
            "lower_final_maps_to_win": 3, "grand_final_maps_to_win": 3}


class BracketProjectionTests(unittest.TestCase):
    def test_all_counts_follow_worker_results_and_hide_future_participants(self):
        for count in range(4, manager.MAX_TEAM_SLOTS + 1):
            teams = [f"Team{i}" for i in range(count)]
            for prefer_right in (True, False):
                with self.subTest(count=count, prefer_right=prefer_right):
                    calls, snapshots = [], []
                    def play(**kw):
                        left, right, need = kw["team1_name"], kw["team2_name"], kw["maps_to_win"]
                        choose_right = prefer_right != (len(calls) % 2 == 0)
                        calls.append((left, right, need))
                        winner, loser = (right, left) if choose_right else (left, right)
                        return manager.SeriesResult(left, right, need, 0 if choose_right else need,
                                                    need if choose_right else 0, winner, loser, [])
                    with patch.object(manager, "validate_preset"), \
                            patch.object(manager, "run_series_core", side_effect=play), \
                            patch.object(manager, "save_json", return_value=Path("unused.json")):
                        result = manager.run_double_elimination(teams, 2, 3, 3,
                            {"seed_count": count, "seed_method": "qualifier"}, "fixed", 42, False, snapshots.append)
                    start = next(event[1] for event in snapshots if event[0] == "bracket_start")
                    results = {}
                    initial = project_bracket(start, results)
                    topology = [(m["id"], m["source1"], m["source2"], m["source1_outcome"], m["source2_outcome"])
                                for m in initial]
                    for event in snapshots:
                        if event[0] != "bracket_match":
                            continue
                        actual = event[1]
                        results[actual["id"]] = actual
                        cards = project_bracket(start, results)
                        self.assertEqual([(m["id"], m["source1"], m["source2"], m["source1_outcome"], m["source2_outcome"])
                                          for m in cards], topology)
                        for card in cards:
                            if card["id"] in results:
                                for key, value in results[card["id"]].items():
                                    self.assertEqual(card[key], value)
                            else:
                                for row in (1, 2):
                                    source = card[f"source{row}"]
                                    if not source.startswith("SLOT-") and results.get(source, {}).get("status") not in ("finished", "bye"):
                                        self.assertIsNone(card[f"team{row}"])
                    self.assertEqual(len(calls), 2 * count - 2)
                    final = project_bracket(start, results)
                    self.assertEqual(final[-1]["winner"], result["champion"])
                    self.assertTrue(all(c["status"] in ("finished", "bye") for c in final))
                    self.assertEqual([c["id"] for c in final], [c["id"] for c in result["bracket_matches"]])


class ManagerBracketUITests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = app = manager.CompetitionApp.__new__(manager.CompetitionApp)
        app.root = self.root
        app.visual_mode = "swiss"
        app.visual_bracket_setup = setup_for(["TeamA", "TeamB", "TeamC", "TeamD"])
        app.visual_bracket_matches = {}
        app.visual_bracket_cards = {}
        app.visual_window = None
        app.visual_frame = tk.Frame(self.root)
        app.visual_window_button = tk.Button(self.root)
        app.series_score_var = tk.StringVar(self.root)
        app.events = queue.Queue()
        app._process_render_requests = Mock()
        app._build_visual_canvas(app.visual_frame)

    def texts(self, tag="all"):
        canvas = self.app.visual_canvas
        return [canvas.itemcget(i, "text") for i in canvas.find_withtag(tag) if canvas.type(i) == "text"]

    def test_complete_cards_place_connectors_under_text_and_final_after_lower(self):
        app = self.app
        self.assertEqual(len(app.visual_bracket_cards), 6)
        self.assertTrue(any("勝者" in t for t in self.texts()))
        self.assertTrue(any("敗者" in t for t in self.texts()))
        self.assertTrue(any("次の試合" in t for t in self.texts("match:W1M1")))
        canvas = app.visual_canvas
        self.assertTrue(canvas.bind("<MouseWheel>"))
        self.assertTrue(canvas.bind("<Shift-MouseWheel>"))
        connectors = canvas.find_withtag("connector")
        rectangles = [i for i in canvas.find_all() if canvas.type(i) == "rectangle"]
        self.assertLess(max(connectors), min(rectangles))
        final_x = canvas.coords(canvas.find_withtag("match:GRAND_FINAL")[0])[0]
        lower_x = canvas.coords(canvas.find_withtag("match:LOWER_FINAL")[0])[0]
        self.assertGreater(final_x, lower_x)

    def test_worker_events_update_scores_and_winner_colors(self):
        app = self.app
        card = app.visual_bracket_cards["W1M1"]
        playing = {**card, "status": "playing", "team1_wins": 0, "team2_wins": 0}
        app.events.put(("bracket_start", app.visual_bracket_setup))
        app.events.put(("bracket_match", playing))
        app.events.put(("series_score", card["team1"], 1, 0, card["team2"], "WINNERS R1 M1"))
        with patch.object(self.root, "after"):
            app._poll_events()
        self.assertIn("試合中", " ".join(self.texts("match:W1M1")))
        self.assertEqual(app.visual_bracket_matches["W1M1"]["team1_wins"], 1)
        finished = {**playing, "status": "finished", "team1_wins": 2,
                    "winner": card["team1"], "loser": card["team2"]}
        app.events.put(("bracket_match", finished))
        with patch.object(self.root, "after"):
            app._poll_events()
        fills = [app.visual_canvas.itemcget(i, "fill") for i in app.visual_canvas.find_withtag("match:W1M1")
                 if app.visual_canvas.type(i) == "rectangle"]
        self.assertIn("#14532d", fills)
        self.assertIn("#343a46", fills)
        self.assertEqual(app.visual_bracket_cards["W2M1"]["team1"], card["team1"])
        self.assertIsNone(app.visual_bracket_cards["W2M1"]["team2"])

    def test_future_and_real_cards_open_details_and_popout_keeps_drawing(self):
        app = self.app
        original_toplevel = tk.Toplevel
        def hidden_window(*args, **kw):
            window = original_toplevel(*args, **kw)
            window.withdraw()
            return window
        with patch.object(manager.tk, "Toplevel", side_effect=hidden_window):
            app._show_bracket_match_details("LOWER_FINAL")
            details = next(child for child in self.root.winfo_children() if isinstance(child, original_toplevel))
            label = next(child for child in details.winfo_children() if isinstance(child, tk.Label))
            self.assertIn("敗者", label["text"])
            self.assertIn("勝者", label["text"])
            details.destroy()
            app.toggle_visual_window()
            self.assertIsNotNone(app.visual_window)
            self.assertEqual(len(app.visual_bracket_cards), 6)
            self.assertTrue(app.visual_canvas.find_withtag("connector"))
            app._close_visual_window()
        self.assertIsNone(app.visual_window)
        self.assertTrue(app.visual_canvas.find_withtag("match:GRAND_FINAL"))


if __name__ == "__main__":
    unittest.main()
