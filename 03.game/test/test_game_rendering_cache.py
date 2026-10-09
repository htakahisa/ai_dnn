"""Check retained rendering with a real Tk canvas (requires Tcl/Tk)."""

import unittest
import tkinter as tk
from contextlib import redirect_stdout
from io import StringIO

with redirect_stdout(StringIO()):
    from run_game import VisualFPSBattle, _build_team_ai
    from map_data import NEW_MAZE_STR


class RenderingCacheTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        with redirect_stdout(StringIO()):
            self.game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("default"),
                _build_team_ai("default"), headless=True,
            )
        self.game.root = self.root
        self.game.canvas = tk.Canvas(self.root)
        self.game.headless = False
        self.game.smokes = [{"cells": [(10, 10)], "remaining_ticks": 10}]
        self.game.recon_bursts = [{"cells": {(10, 10)}, "remaining_ticks": 4}]

    def items(self, tag):
        return self.game.canvas.find_withtag(tag)

    def test_unchanged_layers_reuse_items_without_coordinate_drift(self):
        self.game.draw()
        ids = {tag: self.items(tag) for tag in
               ("render_map", "render_smoke", "render_recon")}
        coordinates = {item: self.game.canvas.coords(item)
                       for items in ids.values() for item in items}
        count = len(self.game.canvas.find_all())
        for _ in range(4):
            self.game.battle_tick += 1
            self.game.smokes[0]["remaining_ticks"] -= 1
            self.game.draw()
            self.assertEqual(count, len(self.game.canvas.find_all()))
            for tag, items in ids.items():
                self.assertEqual(items, self.items(tag))
            for item, coords in coordinates.items():
                self.assertEqual(coords, self.game.canvas.coords(item))

    def test_smoke_blinks_and_effects_disappear(self):
        self.game.smokes[0]["remaining_ticks"] = 3
        self.game.battle_tick = 1
        self.game.draw()
        self.assertEqual(5, len(self.items("render_smoke")))
        self.game.battle_tick = 2
        self.game.draw()
        self.assertFalse(self.items("render_smoke"))
        self.game.battle_tick = 3
        self.game.draw()
        self.assertEqual(5, len(self.items("render_smoke")))
        self.game.smokes.clear()
        self.game.recon_bursts.clear()
        self.game.draw()
        self.assertFalse(self.items("render_smoke"))
        self.assertFalse(self.items("render_recon"))

    def test_effect_geometry_changes_in_place(self):
        self.game.draw()
        old_smoke = self.items("render_smoke")
        old_recon = self.items("render_recon")
        self.game.smokes[0]["cells"][:] = [(11, 12)]
        self.game.recon_bursts[0]["cells"].add((11, 12))
        self.game.draw()
        self.assertNotEqual(old_smoke, self.items("render_smoke"))
        self.assertNotEqual(old_recon, self.items("render_recon"))
        self.assertEqual(5, len(self.items("render_smoke")))
        self.assertEqual(2, len(self.items("render_recon")))

    def test_map_and_display_geometry_invalidate_cache(self):
        self.game.draw()
        old_map = self.items("render_map")
        self.game.grid[10, 10] = 1
        self.game.draw()
        self.assertNotEqual(old_map, self.items("render_map"))
        item = self.items("render_map")[10 * self.game.width + 10]
        self.assertEqual("#34495e", self.game.canvas.itemcget(item, "fill"))
        old_smoke = self.items("render_smoke")
        self.game.cell_size += 1
        self.game.map_offset_x += 5
        self.game.draw()
        self.assertNotEqual(old_smoke, self.items("render_smoke"))

    def test_layer_order_is_restored_on_repeated_draws(self):
        for _ in range(3):
            self.game.draw()
            order = self.game.canvas.find_all()
            self.assertLess(order.index(self.items("render_map")[-1]),
                            order.index(self.items("render_smoke")[0]))
            self.assertLess(order.index(self.items("render_smoke")[-1]),
                            order.index(self.items("render_recon")[0]))
            self.assertLess(order.index(self.items("render_recon")[-1]),
                            len(order) - 1)


if __name__ == "__main__":
    unittest.main()
