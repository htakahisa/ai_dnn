import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

from rendering_ui import RenderingUIMixin
from game_core import COMBO_BANNER_HEIGHT


class UserMapAbilityIconTests(unittest.TestCase):
    def make_game(self, attacker="user", defender="default"):
        from map_data import NEW_MAZE_STR
        from run_game import VisualFPSBattle, _build_team_ai

        with redirect_stdout(StringIO()):
            game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai(attacker),
                                   _build_team_ai(defender), headless=True)
        game.canvas = Mock()
        game.canvas.bbox.return_value = None
        game.headless = False
        game.is_visible_to_team = lambda char, team: True
        game._draw_compact_ability_icon = Mock()
        return game

    def map_icons(self, game):
        return [call.args for call in game._draw_compact_ability_icon.call_args_list
                if game.map_offset_x <= call.args[1] < game.map_offset_x + game.map_pixel_width]

    def test_only_manual_team_gets_icons_without_selection(self):
        for attacker, defender, manual_team in (("user", "default", "A"),
                                                 ("default", "user", "D")):
            with self.subTest(team=manual_team):
                game = self.make_game(attacker, defender)
                game.draw()
                expected = [ch.ability_name for ch in game.chars if ch.team == manual_team and ch.is_alive]
                self.assertEqual([args[0] for args in self.map_icons(game)], expected)

    def test_ai_only_game_has_no_map_icons(self):
        game = self.make_game("default", "default")
        game.draw()
        self.assertEqual(self.map_icons(game), [])

    def test_hover_hides_name_panel_and_leave_restores_it(self):
        game = self.make_game()
        owner = next(ch for ch in game.chars if ch.team == "A")
        actor = next(ch for ch in game.chars if ch.team == "A" and ch is not owner)
        owner.pos, actor.pos = [4, 4], [10, 10]
        game.chars = [owner, actor]
        game.attacker_controller.selected_char = actor.name
        game.draw()
        x1, y1, x2, y2 = game._name_panel_bounds[owner.name]
        event = SimpleNamespace(x=(x1 + x2) / 2,
                                y=(y1 + y2) / 2 + COMBO_BANNER_HEIGHT)
        game.canvas.reset_mock()
        game.on_canvas_hover(event)
        self.assertEqual(game._hidden_name_panels, {owner.name})
        self.assertFalse(any(call.kwargs.get("text") == owner.display_name
                             and game.map_offset_x <= call.args[0] < game.map_offset_x + game.map_pixel_width
                             for call in game.canvas.create_text.call_args_list))
        self.assertTrue(game.canvas.create_oval.called)
        game.canvas.reset_mock()
        game.on_canvas_hover(event)
        game.canvas.delete.assert_not_called()
        game.on_canvas_leave(event)
        self.assertEqual(game._hidden_name_panels, set())
        self.assertTrue(any(call.kwargs.get("text") == owner.display_name
                            and game.map_offset_x <= call.args[0] < game.map_offset_x + game.map_pixel_width
                            for call in game.canvas.create_text.call_args_list))

        game.attacker_controller.selected_char = None
        game.on_canvas_hover(event)
        game.draw()
        self.assertEqual(game._hidden_name_panels, set())
        game.attacker_controller.selected_char = owner.name
        game.draw()
        self.assertEqual(game._hidden_name_panels, set())
        game.attacker_controller.select_characters([owner.name, actor.name])
        game.draw()
        self.assertEqual(game._hidden_name_panels, set())
        game.attacker_controller.selected_char = actor.name
        game.draw()
        self.assertEqual(game._hidden_name_panels, {owner.name})
        actor.is_alive = False
        game.draw()
        self.assertEqual(game._hidden_name_panels, set())

    def test_remaining_charge_changes_map_availability_on_redraw(self):
        game = self.make_game()
        owner = next(ch for ch in game.chars if ch.team == "A")
        owner.ability_name = "SMOKE"
        owner.smoke_charges = 1
        game.chars = [owner]
        game.draw()
        self.assertTrue(self.map_icons(game)[0][3])
        owner.smoke_charges = 0
        game._draw_compact_ability_icon.reset_mock()
        game.draw()
        self.assertFalse(self.map_icons(game)[0][3])

    def test_all_consumables_and_passive_share_panel_availability(self):
        for ability in ("SMOKE", "FLASH", "RECON", "RAMP", "DANCE", "ASH"):
            char = SimpleNamespace(ability_name=ability, is_alive=True)
            for charges in (2, 1, 0):
                with self.subTest(ability=ability, charges=charges):
                    setattr(char, ability.lower() + "_charges", charges)
                    self.assertEqual(RenderingUIMixin._ability_display_state(char), (charges, charges > 0))
        passive = SimpleNamespace(ability_name="HUNT", is_alive=True)
        self.assertEqual(RenderingUIMixin._ability_display_state(passive), (1, True))
        passive.is_alive = False
        self.assertEqual(RenderingUIMixin._ability_display_state(passive), (1, False))

    def test_depleted_icon_uses_gray_drawing_colors(self):
        ui = RenderingUIMixin()
        for ability in ("SMOKE", "FLASH", "RECON", "RAMP", "DANCE", "ASH", "HUNT"):
            with self.subTest(ability=ability):
                ui.canvas = Mock()
                ui._draw_compact_ability_icon(ability, 20, 20, False)
                colors = [color for call in ui.canvas.method_calls for key, color in call.kwargs.items()
                          if key in ("fill", "outline")]
                self.assertTrue(colors)
                self.assertTrue(all(color in ("#59616c", "#7f8c8d") for color in colors))


if __name__ == "__main__":
    unittest.main()
