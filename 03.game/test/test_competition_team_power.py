import unittest
from types import SimpleNamespace
from unittest.mock import patch

import game_core
import run_competition_manager as manager


class TeamPowerTests(unittest.TestCase):
    def test_report_includes_chained_combos_and_all_displayed_stats(self):
        names = ("a", "b", "c", "d", "e")
        stats = dict(
            hs_rate=0.8, accuracy=0.9, dodge_rate=0.8, reaction=100,
            iq=190, influence=7, form_variance=3, mental=18, role="フラッシュ",
        )
        combos = [
            dict(name="first", players=("a", "b"),
                 bonuses={"hs_rate": 0.4, "accuracy": 0.3, "mental": 5,
                          "form_variance": -5, "iq": 50},
                 player_bonuses={"a": {"reaction": 20}}, renames={"a": "renamed"}),
            dict(name="second", players=("renamed", "c"), bonuses={"dodge_rate": 0.5}),
            dict(name="inactive", players=("a", "missing"), bonuses={"iq": 100}),
        ]
        with patch.object(manager, "get_preset", return_value=SimpleNamespace(players=names)), \
             patch.object(manager, "validate_preset"), \
             patch.object(manager, "get_character_combat_stats", return_value=stats), \
             patch.object(manager, "PLAYER_COMBOS", combos):
            report = manager.build_team_combo_power_report("test")
        self.assertEqual(report["active_combos"], ["first", "second"])
        self.assertEqual(len(report["players"]), 5)
        first = report["players"][0]
        self.assertEqual(first["base"], stats)
        expected = SimpleNamespace(**stats)
        for key, value in dict(
            hs_rate=0.4, accuracy=0.3, mental=5, form_variance=-5,
            iq=50, reaction=20, dodge_rate=0.5,
        ).items():
            game_core._apply_combo_bonus(expected, key, value)
        for key in stats:
            self.assertEqual(first["after"][key], getattr(expected, key))
        self.assertAlmostEqual(
            first["combo_power"],
            manager.calculate_combat_power_index(1.2, 1.0, 240, 1.2, 120),
        )
        self.assertAlmostEqual(report["base_total"], sum(r["base_power"] for r in report["players"]))
        self.assertAlmostEqual(report["combo_total"], sum(r["combo_power"] for r in report["players"]))

    def test_all_registered_teams_have_five_player_reports(self):
        for name in manager.all_preset_names():
            with self.subTest(team=name):
                report = manager.build_team_combo_power_report(name)
                self.assertEqual(len(report["players"]), 5)
                self.assertTrue(all("role" in row["after"] for row in report["players"]))

    def test_team_selection_and_refresh_show_matching_five_players(self):
        root = manager.tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = manager.CompetitionApp.__new__(manager.CompetitionApp)
        app.names = manager.all_preset_names()
        app.team_power_tab = manager.tk.Frame(root)
        app.team_power_tree = None
        app.team_power_players_tree = None
        app.team_power_reports = {}
        app.team_power_detail_var = manager.tk.StringVar(root)
        app.team_power_combos_var = manager.tk.StringVar(root)
        app._build_team_power_tab()
        root.update()
        self.assertEqual(len(app.team_power_tree.get_children()), len(app.names))
        self.assertEqual(app.team_power_tree["columns"][0], "no")
        self.assertFalse(app.team_power_tree.heading("no", "command"))
        self.assertEqual(
            [int(app.team_power_tree.set(item, "no")) for item in app.team_power_tree.get_children()],
            list(range(1, len(app.names) + 1)),
        )
        for name in (app.names[0], app.names[-1]):
            app.team_power_tree.selection_set(name)
            root.update()
            report = app.team_power_reports[name]
            team_row = app.team_power_tree.item(name, "values")
            self.assertEqual(team_row[2], f'{report["combo_total"]:.1f} ({report["base_total"]:.1f})')
            rows = [app.team_power_players_tree.item(i, "values")
                    for i in app.team_power_players_tree.get_children()]
            self.assertEqual(len(rows), 5)
            self.assertEqual([r[0] for r in rows], list(manager.get_preset(name).players))
            first = report["players"][0]
            self.assertEqual(rows[0][1], f'{first["after"]["hs_rate"] * 100:.1f}% ({first["base"]["hs_rate"] * 100:.1f}%)')
            self.assertEqual(rows[0][-1], f'{first["combo_power"]:.1f} ({first["base_power"]:.1f})')
        with patch.object(manager, "reload_game_data") as reload_data:
            app.team_power_refresh_button.invoke()
            reload_data.assert_called_once_with()
        root.update()
        self.assertEqual(app.team_power_tree.selection(), (app.names[-1],))
        self.assertEqual(len(app.team_power_players_tree.get_children()), 5)
        for tree in (app.team_power_tree, app.team_power_players_tree):
            for column in tree["columns"]:
                if column == "no":
                    continue
                for descending in (True, False):
                    with self.subTest(table=str(tree), column=column, descending=descending):
                        root.tk.call(tree.heading(column, "command"))
                        rows = tree.get_children()
                        raw = app.team_power_sort_values[tree]
                        values = [raw[item][column] for item in rows]
                        self.assertEqual(values, sorted(values, reverse=descending))
                        self.assertTrue(tree.heading(column, "text").endswith("▼" if descending else "▲"))
                        self.assertEqual(app.team_power_tree.selection(), (app.names[-1],))
                        self.assertEqual(
                            [int(app.team_power_tree.set(item, "no")) for item in app.team_power_tree.get_children()],
                            list(range(1, len(app.names) + 1)),
                        )
        app._refresh_team_power()
        root.update()
        for tree in (app.team_power_tree, app.team_power_players_tree):
            raw = app.team_power_sort_values[tree]
            values = [raw[item]["power"] for item in tree.get_children()]
            self.assertEqual(values, sorted(values))
        app.team_power_tree.selection_set(app.names[0])
        root.update()
        raw = app.team_power_sort_values[app.team_power_players_tree]
        values = [raw[item]["power"] for item in app.team_power_players_tree.get_children()]
        self.assertEqual(values, sorted(values))

        app.player_power_tab = manager.tk.Frame(root)
        app.player_power_status_var = manager.tk.StringVar(root)
        app._build_player_power_tab()
        root.update()
        tree = app.player_power_tree
        expected_players = {
            (team, name) for team in app.names for name in manager.get_preset(team).players
        }
        self.assertEqual(len(tree.get_children()), len(expected_players))
        self.assertEqual(
            {(tree.set(i, "team"), tree.set(i, "name")) for i in tree.get_children()},
            expected_players,
        )
        for item in tree.get_children():
            team, name = tree.set(item, "team"), tree.set(item, "name")
            row = next(r for r in app.team_power_reports[team]["players"] if r["name"] == name)
            self.assertEqual(tree.set(item, "power"), f'{row["combo_power"]:.1f} ({row["base_power"]:.1f})')
            self.assertEqual(tree.set(item, "mental"), f'{row["after"]["mental"]:g} ({row["base"]["mental"]:g})')
        self.assertFalse(tree.heading("no", "command"))
        for column in tree["columns"][1:]:
            for descending in (True, False):
                root.tk.call(tree.heading(column, "command"))
                raw = app.team_power_sort_values[tree]
                values = [raw[i][column] for i in tree.get_children()]
                self.assertEqual(values, sorted(values, reverse=descending))
                self.assertEqual(
                    [int(tree.set(i, "no")) for i in tree.get_children()],
                    list(range(1, len(expected_players) + 1)),
                )
        selected = tree.get_children()[0]
        selected_name = tree.set(selected, "name")
        selected_team = tree.set(selected, "team")
        tree.selection_set(selected)
        with patch.object(manager, "reload_game_data"):
            app.player_power_refresh_button.invoke()
        self.assertEqual(tree.set(tree.selection()[0], "name"), selected_name)
        self.assertEqual(tree.set(tree.selection()[0], "team"), selected_team)
        raw = app.team_power_sort_values[tree]
        values = [raw[i]["power"] for i in tree.get_children()]
        self.assertEqual(values, sorted(values))


if __name__ == "__main__":
    unittest.main()
