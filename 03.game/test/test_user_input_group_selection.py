import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from controllers import UserInputController
from battle_logic import BattleLogicMixin
from game_core import COMBO_BANNER_HEIGHT
from rendering_ui import RenderingUIMixin


def character(name, pos, team="A", alive=True):
    return SimpleNamespace(name=name, pos=list(pos), team=team, is_alive=alive)


class GroupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = UserInputController()
        self.grid = np.zeros((8, 8), dtype=int)
        self.chars = [character("one", (1, 1)), character("two", (2, 2)),
                      character("enemy", (1, 2), "D"), character("dead", (2, 1), alive=False)]

    def test_group_move_assigns_distinct_floor_cells_and_keeps_selection(self):
        self.ctrl.select_characters(["one", "two", "dead", "enemy"])
        self.grid[4, 5] = 1
        self.ctrl.issue_move(4, 4, self.grid, self.chars, "A")
        self.assertEqual(set(self.ctrl.targets), {"one", "two"})
        self.assertEqual(len(set(self.ctrl.targets.values())), 2)
        self.assertIn((4, 4), self.ctrl.targets.values())
        self.assertTrue(all(self.grid[pos] != 1 for pos in self.ctrl.targets.values()))
        self.assertEqual(self.ctrl.selected_char, "one")
        self.ctrl.issue_move(6, 6, self.grid, self.chars, "A")
        self.assertIn((6, 6), self.ctrl.targets.values())

    def test_single_move_keeps_selection_for_more_orders_and_round_reset_clears_it(self):
        self.ctrl.handle_click(1, 1, self.grid, self.chars, "A")
        self.ctrl.handle_click(4, 4, self.grid, self.chars, "A")
        self.assertEqual(self.ctrl.targets, {"one": (4, 4)})
        self.assertEqual(self.ctrl.selected_chars, ["one"])
        self.assertEqual(self.ctrl.selected_char, "one")
        self.ctrl.handle_click(6, 6, self.grid, self.chars, "A")
        self.assertEqual(self.ctrl.targets, {"one": (6, 6)})
        ui = self.make_ui()
        ui.active_user_team = "A"
        ui.on_canvas_right_click(self.event(190, 30))
        self.assertEqual(self.ctrl.facings, {"one": "E"})
        self.assertEqual(self.ctrl.selected_chars, ["one"])
        self.ctrl.select_characters(["one", "two"])
        self.ctrl.selected_char = "two"
        self.assertEqual(self.ctrl.selected_chars, ["two"])
        self.ctrl.reset_round()
        self.assertEqual(self.ctrl.selected_chars, [])
        self.assertEqual(self.ctrl.targets, {})
        self.assertEqual(self.ctrl.facings, {})

    def test_left_click_moves_group_and_wall_click_clears_selection(self):
        self.ctrl.select_characters(["one", "two"])
        self.ctrl.handle_click(4, 4, self.grid, self.chars, "A")
        self.assertEqual(len(self.ctrl.targets), 2)
        self.assertEqual(self.ctrl.selected_chars, ["one", "two"])
        self.grid[5, 5] = 1
        previous = dict(self.ctrl.targets)
        self.ctrl.handle_click(5, 5, self.grid, self.chars, "A")
        self.assertEqual(self.ctrl.targets, previous)
        self.assertEqual(self.ctrl.selected_chars, [])

    def test_group_avoids_occupied_cells_and_disconnected_destination(self):
        self.ctrl.select_characters(["one", "two"])
        self.ctrl.issue_move(1, 2, self.grid, self.chars, "A")
        self.assertNotIn((1, 2), self.ctrl.targets.values())
        self.ctrl.targets.clear()
        self.grid[3, :] = 1
        self.ctrl.issue_move(6, 6, self.grid, self.chars, "A")
        self.assertEqual(self.ctrl.targets, {})

    def make_ui(self):
        ui = RenderingUIMixin()
        ui.attacker_controller = self.ctrl
        ui.defender_controller = object()
        ui.chars, ui.grid = self.chars, self.grid
        ui.map_offset_x, ui.map_pixel_width, ui.map_pixel_height = 100, 160, 160
        ui.cell_size = 20
        ui.height = ui.width = 8
        ui._facing_towards = BattleLogicMixin()._facing_towards
        ui.ability_mode = ui.ultimate_mode = None
        ui.canvas, ui.draw = Mock(), Mock()
        ui.on_canvas_click = Mock()
        return ui

    def event(self, x, y):
        return SimpleNamespace(x=x, y=y + COMBO_BANNER_HEIGHT)

    def test_reverse_drag_selects_only_living_allies_then_left_click_moves(self):
        ui = self.make_ui()
        ui.on_canvas_press(self.event(160, 60))
        ui.on_canvas_drag(self.event(120, 20))
        ui.on_canvas_release(self.event(120, 20))
        self.assertEqual(self.ctrl.selected_chars, ["one", "two"])
        ui.on_canvas_click.assert_not_called()
        self.assertEqual(self.ctrl.targets, {})
        self.assertTrue(ui._is_character_selected(self.chars[1]))
        self.ctrl.handle_click(4, 4, self.grid, self.chars, "A")
        self.assertEqual(len(self.ctrl.targets), 2)
        self.assertIn((4, 4), self.ctrl.targets.values())

    def test_small_motion_is_click_and_ability_targeting_is_immediate(self):
        ui = self.make_ui()
        ui.on_canvas_press(self.event(130, 30))
        ui.on_canvas_release(self.event(132, 31))
        ui.on_canvas_click.assert_called_once()
        ui.on_canvas_click.reset_mock()
        ui.ability_mode = ("SMOKE", "A", "one")
        ui.on_canvas_press(self.event(130, 30))
        ui.on_canvas_release(self.event(160, 60))
        ui.on_canvas_click.assert_called_once()

    def test_empty_drag_clears_selection_and_off_map_release_is_clipped(self):
        ui = self.make_ui()
        self.ctrl.selected_char = "one"
        ui.on_canvas_press(self.event(200, 100))
        ui.on_canvas_release(self.event(400, 300))
        self.assertEqual(self.ctrl.selected_chars, [])
        ui.on_canvas_click.assert_not_called()

    def test_drag_keeps_active_team_when_both_teams_are_manual(self):
        ui = self.make_ui()
        ui.defender_controller = UserInputController()
        ui.defender_controller.selected_char = "enemy"
        ui.active_user_team = "A"
        ui.on_canvas_press(self.event(120, 20))
        ui.on_canvas_release(self.event(160, 60))
        self.assertEqual(self.ctrl.selected_chars, ["one", "two"])
        self.assertEqual(ui.defender_controller.selected_chars, [])
        ui.active_user_team = "D"
        ui.on_canvas_press(self.event(120, 20))
        ui.on_canvas_release(self.event(160, 60))
        self.assertEqual(ui.defender_controller.selected_chars, ["enemy"])
        self.assertEqual(self.ctrl.selected_chars, [])

    def test_right_click_faces_selected_allies_without_changing_move_orders(self):
        ui = self.make_ui()
        ui.active_user_team = "A"
        self.ctrl.select_characters(["one", "two", "dead", "enemy"])
        self.ctrl.targets = {"one": (6, 6), "two": (7, 7)}
        self.grid[1, 4] = 1  # A wall is still a valid facing target.
        ui.on_canvas_right_click(self.event(190, 30))
        self.assertEqual(self.ctrl.facings, {"one": "E", "two": "NE"})
        self.assertEqual(self.ctrl.targets, {"one": (6, 6), "two": (7, 7)})
        self.assertEqual(self.chars[0].facing, "E")
        self.assertEqual(self.chars[1].facing, "NE")
        ui.on_canvas_click.assert_not_called()

    def test_same_cell_and_off_map_right_click_keep_facing(self):
        ui = self.make_ui()
        ui.active_user_team = "A"
        self.ctrl.selected_char = "one"
        self.ctrl.facings["one"] = "W"
        ui.on_canvas_right_click(self.event(130, 30))
        ui.on_canvas_right_click(self.event(0, 0))
        self.assertEqual(self.ctrl.facings, {"one": "W"})

    def test_manual_facing_survives_movement_but_respects_forced_turn(self):
        from test_ultimate_system import UltimateTestGame, make_character

        for side in ("A", "D"):
            with self.subTest(side=side):
                game = UltimateTestGame()
                actor = make_character("Xdll", side, (4, 4), 99)
                ctrl = UserInputController()
                ctrl.facings[actor.name] = "W"
                ctrl.targets[actor.name] = (4, 7)
                game.chars = [actor]
                if side == "A":
                    game.attacker_controller = ctrl
                else:
                    game.defender_controller = ctrl
                game.move_character(actor)
                self.assertEqual(actor.pos, [4, 5])
                self.assertEqual(actor.facing, "W")
                actor.forced_facing_next_tick = "N"
                game.move_character(actor)
                self.assertEqual(actor.facing, "N")
                game.move_character(actor)
                self.assertEqual(actor.facing, "W")
                ctrl.reset_round()
                self.assertEqual(ctrl.facings, {})


if __name__ == "__main__":
    unittest.main()
