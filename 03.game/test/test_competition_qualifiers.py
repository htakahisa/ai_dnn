import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tkinter as tk
from tkinter import ttk

import run_competition_manager as manager


class QualifierImportTests(unittest.TestCase):
    def setUp(self):
        self.names = manager.all_preset_names()
        self.ranking = list(reversed(self.names[:12]))
        self.data = {
            "mode": "round_robin", "ranking": self.ranking,
            "team_controllers": {name: "fnatic_v3" for name in self.ranking},
        }

    def test_saved_ranking_controls_selection_and_controllers(self):
        # 保存済み順位を使い、レートやチーム一覧の並びで再ソートしない。
        teams, controllers = manager.prepare_round_robin_qualifiers(self.data, 9, self.names)
        self.assertEqual(teams, self.ranking[:9])
        self.assertEqual(controllers, {name: "fnatic_v3" for name in teams})

    def test_legacy_names_and_missing_controllers(self):
        self.data["ranking"] = ["日本代表"] + [
            name for name in self.ranking if name != "Japan All-Stars"
        ]
        self.data["team_controllers"] = {"日本代表": "fnatic_v1"}
        teams, controllers = manager.prepare_round_robin_qualifiers(self.data, 4, self.names)
        self.assertEqual(teams[0], "Japan All-Stars")
        self.assertEqual(controllers[teams[0]], "fnatic_v1")
        self.assertEqual(controllers[teams[1]], manager.CONTROLLER_OPTIONS[manager.DEFAULT_CONTROLLER_DISPLAY])

    def test_invalid_results_and_counts_are_rejected(self):
        invalid = [
            ([], 4), ({**self.data, "mode": "double_elimination"}, 4),
            ({**self.data, "ranking": None}, 4),
            ({**self.data, "ranking": [123, *self.ranking]}, 4),
            ({**self.data, "ranking": [self.ranking[0], *self.ranking]}, 4),
            ({**self.data, "ranking": ["unknown", *self.ranking]}, 4),
            ({**self.data, "team_controllers": []}, 4),
            ({**self.data, "team_controllers": {self.ranking[0]: "unknown_ai"}}, 4),
            (self.data, 3), (self.data, 31), (self.data, 13),
        ]
        for data, count in invalid:
            with self.subTest(data=data, count=count), self.assertRaises(ValueError):
                manager.prepare_round_robin_qualifiers(data, count, self.names)


class QualifierBracketTests(unittest.TestCase):
    def test_all_supported_counts_seed_every_team_and_give_top_seeds_byes(self):
        for count in range(4, manager.MAX_TEAM_SLOTS + 1):
            teams = [f"Team {rank}" for rank in range(1, count + 1)]
            slots, seeds = manager.build_seeded_bracket_slots(teams, count, "qualifier", [], {})
            with self.subTest(count=count):
                self.assertEqual(seeds, teams)
                self.assertCountEqual([team for team in slots if team], teams)
                self.assertEqual(len(slots), manager._next_power_of_two(count))
                bye_teams = []
                for left, right in zip(slots[::2], slots[1::2]):
                    self.assertTrue(left or right)
                    if left is None or right is None:
                        bye_teams.append(left or right)
                self.assertCountEqual(bye_teams, teams[:len(slots) - count])
                self.assertLess(slots.index(teams[0]), len(slots) // 2)
                self.assertGreaterEqual(slots.index(teams[1]), len(slots) // 2)

    def test_eight_team_standard_matchups(self):
        teams = [str(rank) for rank in range(1, 9)]
        slots, _ = manager.build_seeded_bracket_slots(teams, 8, "qualifier", [], {})
        self.assertEqual(slots, ["1", "8", "4", "5", "2", "7", "3", "6"])

    def test_existing_manual_seeding_still_assigns_byes(self):
        teams = [str(rank) for rank in range(1, 13)]
        slots, seeds = manager.build_seeded_bracket_slots(teams, 4, "manual", teams[:4], {})
        self.assertEqual(seeds, teams[:4])
        self.assertCountEqual([team for team in slots if team], teams)
        for team in seeds:
            self.assertIsNone(slots[slots.index(team) ^ 1])

    def test_seeded_tournaments_finish_with_and_without_byes(self):
        for count in range(4, manager.MAX_TEAM_SLOTS + 1):
            teams = [f"Team {rank}" for rank in range(1, count + 1)]

            def play(**kwargs):
                left, right = kwargs["team1_name"], kwargs["team2_name"]
                winner, loser = sorted((left, right), key=teams.index)
                need = kwargs["maps_to_win"]
                return manager.SeriesResult(
                    left, right, need, need if winner == left else 0,
                    need if winner == right else 0, winner, loser, [],
                )

            with self.subTest(count=count), patch.object(manager, "validate_preset"), \
                    patch.object(manager, "run_series_core", side_effect=play) as runner, \
                    patch.object(manager, "save_json", return_value=Path("unused.json")):
                result = manager.run_double_elimination(
                    teams, 2, 3, 3, {"seed_count": count, "seed_method": "qualifier"},
                    "fixed", 42, False, lambda _event: None,
                )
                self.assertEqual(result["champion"], teams[0])
                self.assertEqual(result["runner_up"], teams[1])
                self.assertEqual(runner.call_count, 2 * count - 2)
                self.assertEqual(result["tournament_seeding"]["resolved_seeds"], teams)
                for team in teams[2:]:
                    self.assertEqual(result["records"][team]["losses"], 2)


class QualifierUITests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = manager.CompetitionApp.__new__(manager.CompetitionApp)
        self.app.root = self.root
        self.app.worker = None
        self.app.names = manager.all_preset_names()
        self.app.notebook = ttk.Notebook(self.root)
        self.app.swiss_tab = tk.Frame(self.app.notebook)
        self.app.notebook.add(self.app.swiss_tab, text="DE")
        self.app.swiss_slots = manager.TeamSlotEditor(self.app.swiss_tab, self.app.names, "Teams")
        self.app.tournament_seed_editor = manager.TournamentSeedEditor(
            self.app.swiss_tab, self.app.names,
            Mock(ranking=Mock(return_value=[]), get=Mock(return_value=1500.0)),
        )
        self.app.qualifier_count_var = tk.StringVar(master=self.root, value="12")
        self.app.qualifier_status_var = tk.StringVar(master=self.root)
        self.app.status_var = tk.StringVar(master=self.root)

    def tearDown(self):
        self.root.destroy()

    def test_file_import_updates_slots_ai_and_seeds(self):
        ranking = list(reversed(self.app.names[:12]))
        data = {
            "mode": "round_robin", "ranking": ranking,
            "team_controllers": {name: "fnatic_v3" for name in ranking},
        }
        with patch.object(manager.filedialog, "askopenfilename", return_value="qualifier.json"), \
                patch.object(Path, "open", return_value=io.StringIO(json.dumps(data))):
            self.app.import_round_robin_qualifiers()
        self.assertEqual(self.app.swiss_slots.selected_teams(), ranking)
        self.assertEqual(self.app.swiss_slots.selected_team_controllers(), data["team_controllers"])
        editor = self.app.tournament_seed_editor
        self.assertEqual(editor.get_config(ranking)["seed_count"], 12)
        self.assertEqual(editor.get_config(ranking)["seed_method"], "qualifier")
        editor.set_enabled(False)
        self.assertEqual(str(editor.count_box.cget("state")), "disabled")
        editor.set_enabled(True)
        self.assertEqual(str(editor.count_box.cget("state")), "disabled")
        editor.seed_method_var.set("manual")
        editor._refresh()
        self.assertEqual(editor.seed_count_var.get(), "4")
        self.assertEqual(str(editor.count_box.cget("state")), "readonly")

    def test_cancel_and_invalid_file_leave_settings_intact(self):
        before = self.app.swiss_slots.selected_teams()
        for path, content in (("", ""), ("bad.json", "{}"), ("bad.json", "{")):
            with self.subTest(path=path, content=content), \
                    patch.object(manager.filedialog, "askopenfilename", return_value=path), \
                    patch.object(Path, "open", return_value=io.StringIO(content)), \
                    patch.object(manager.messagebox, "showerror") as error:
                self.app.import_round_robin_qualifiers()
                self.assertEqual(self.app.swiss_slots.selected_teams(), before)
                self.assertEqual(self.app.tournament_seed_editor.seed_method_var.get(), "rating")
                self.assertEqual(error.call_count, int(bool(path)))


if __name__ == "__main__":
    unittest.main()
