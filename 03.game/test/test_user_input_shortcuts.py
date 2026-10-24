import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from game_core import COMBO_BANNER_HEIGHT
from rendering_ui import RenderingUIMixin


class UserShortcutTests(unittest.TestCase):
    def setUp(self):
        self.ui = ui = RenderingUIMixin()
        self.owner = SimpleNamespace(name="owner", team="A", is_alive=True, pos=[1, 1],
                                     ability_name="SMOKE", ultimate_name="ESCAPE",
                                     ultimate_points=9, ultimate_cost=9,
                                     ability_seal_remaining=0, smoke_charges=2,
                                     flash_charges=2, recon_charges=2, dance_charges=2,
                                     ash_charges=2)
        ui.chars = [self.owner]
        ui.grid = np.zeros((8, 8), dtype=int)
        ui.height = ui.width = 8
        ui.cell_size, ui.map_offset_x, ui.map_pixel_width = 20, 100, 160
        ui.ability_mode = ui.ultimate_mode = None
        ui.draw = Mock()
        ui.execute_ai_ability = Mock(return_value=True)
        ui.execute_ai_ultimate = Mock(return_value=True)
        ui._selected_user_character = lambda: self.owner
        ui._handle_team_panel_click = lambda x, y: False
        ui._plant_button_bounds = ui._orb_button_bounds = lambda: None
        ui._ability_button_bounds = lambda ability: (100, 180, 170, 248)
        ui._ultimate_button_bounds = lambda: (180, 180, 250, 248)
        ui.smokes, ui.flash_projectiles, ui.recon_projectiles = [], [], []
        ui._projectile_path = lambda start, target: [start, target]

    def event(self, col=4, row=4):
        return SimpleNamespace(x=100 + col * 20 + 10, y=COMBO_BANNER_HEIGHT + row * 20 + 10)

    def press_a(self, event=None):
        event = event or self.event()
        self.ui.canvas = Mock()
        self.ui.canvas.winfo_pointerx.return_value = event.x + 30
        self.ui.canvas.winfo_pointery.return_value = event.y + 50
        self.ui.canvas.winfo_rootx.return_value = 30
        self.ui.canvas.winfo_rooty.return_value = 50
        self.ui.on_ability_shortcut_release(None)
        self.assertEqual(self.ui.on_ability_shortcut(None), "break")

    def test_a_key_casts_smoke_at_cursor_without_moving(self):
        self.press_a()
        self.assertEqual(self.ui.smokes[0]["center"], (4, 4))
        self.assertEqual(self.owner.smoke_charges, 1)
        self.assertIsNone(self.ui.ability_mode)
        self.ui.on_ability_shortcut(None)
        self.assertEqual(self.owner.smoke_charges, 1)

    def test_a_key_uses_direction_for_flash_and_recon(self):
        for ability, projectiles in (("FLASH", self.ui.flash_projectiles),
                                     ("RECON", self.ui.recon_projectiles)):
            with self.subTest(ability=ability):
                self.owner.ability_name = ability
                self.press_a()
                self.assertEqual(projectiles[0]["path"], [(1, 1), (4, 4)])
                self.assertEqual(getattr(self.owner, ability.lower() + "_charges"), 1)

    def test_a_key_dispatches_ramp_ash_and_dance(self):
        self.owner.ability_name = "RAMP"
        self.press_a()
        self.ui.execute_ai_ability.assert_called_with(self.owner, {"ability": "RAMP"})
        self.owner.ability_name = "ASH"
        self.press_a()
        self.ui.execute_ai_ability.assert_called_with(self.owner, {"ability": "ASH", "target": (4, 4)})
        self.owner.ability_name = "DANCE"
        self.ui.chars.append(SimpleNamespace(name="ally", team="A", is_alive=True, pos=[4, 4]))
        self.press_a()
        self.ui.execute_ai_ability.assert_called_with(self.owner, {"ability": "DANCE", "target_name": "ally"})

    def test_invalid_a_key_does_not_consume_ability(self):
        for case in ("dead", "sealed", "empty", "wall", "outside", "passive"):
            with self.subTest(case=case):
                self.setUp()
                event = self.event()
                if case == "dead":
                    self.owner.is_alive = False
                elif case == "sealed":
                    self.owner.ability_seal_remaining = 3
                elif case == "empty":
                    self.owner.smoke_charges = 0
                elif case == "wall":
                    self.ui.grid[4, 4] = 1
                elif case == "outside":
                    event = self.event(col=-1)
                else:
                    self.owner.ability_name = "HUNT"
                self.press_a(event)
                self.assertEqual(self.ui.smokes, [])
                self.ui.execute_ai_ability.assert_not_called()

    def test_space_arms_targeted_ultimate_and_left_click_casts(self):
        self.assertEqual(self.ui.on_ultimate_shortcut(None), "break")
        self.assertEqual(self.ui.ultimate_mode, ("ESCAPE", "A", "owner"))
        self.ui.on_ultimate_shortcut(None)  # Ignore key repeat.
        self.assertIsNotNone(self.ui.ultimate_mode)
        self.ui.on_canvas_click(self.event())
        self.ui.execute_ai_ultimate.assert_called_once_with(
            self.owner, {"ultimate": "ESCAPE", "target": (4, 4)})

    def test_space_release_allows_cancel_and_instant_ultimate(self):
        self.ui.on_ultimate_shortcut(None)
        self.ui.on_ultimate_shortcut_release(None)
        self.ui.on_ultimate_shortcut(None)
        self.assertIsNone(self.ui.ultimate_mode)
        self.ui.on_ultimate_shortcut_release(None)
        self.owner.ultimate_name = "NEON"
        self.ui.on_ultimate_shortcut(None)
        self.assertEqual(self.ui.ultimate_mode, ("NEON", "A", "owner"))
        self.ui.on_ultimate_shortcut_release(None)
        self.owner.ultimate_name = "RAID"
        self.ui.on_ultimate_shortcut(None)
        self.ui.execute_ai_ultimate.assert_called_once_with(self.owner, {"ultimate": "RAID"})

    def test_space_obeys_points_and_seal_and_missing_selection(self):
        self.owner.ultimate_points = 0
        self.ui.on_ultimate_shortcut(None)
        self.assertIsNone(self.ui.ultimate_mode)
        self.ui.on_ultimate_shortcut_release(None)
        self.owner.ultimate_points = 9
        self.owner.ability_seal_remaining = 3
        self.ui.on_ultimate_shortcut(None)
        self.assertIsNone(self.ui.ultimate_mode)
        self.ui.on_ultimate_shortcut_release(None)
        self.ui._ultimate_button_bounds = lambda: None
        self.ui.on_ultimate_shortcut(None)
        self.ui.execute_ai_ultimate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
