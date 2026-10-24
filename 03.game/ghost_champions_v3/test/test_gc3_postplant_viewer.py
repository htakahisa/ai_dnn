"""Scene selection and retry checks for the post-plant viewer."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ghost_champions_v3.tools.gc3_watch_attacker_postplant import (
    NoPostplantEntry, SceneViewer, layout_matches, scene_layout, scene_signature,
)


class PostplantViewerTests(unittest.TestCase):
    def test_layout_ignores_dead_players_and_keeps_the_two_teams_separate(self):
        chars = [SimpleNamespace(name=name, team=team, pos=pos, is_alive=alive)
                 for name, team, pos, alive in (
                     ("a1", "A", [4, 3], True), ("a2", "A", [7, 8], True),
                     ("dead", "A", [2, 40], False),
                     ("d1", "D", [2, 8], True), ("d2", "D", [2, 10], True),
                 )]
        game = SimpleNamespace(chars=chars, width=44, planted_pos=[9, 3])
        layout = scene_layout(game)
        self.assertEqual(layout, dict(site="A", attacker_span=8, defender_span=2))
        self.assertTrue(layout_matches(layout, "A", "attacker"))
        self.assertFalse(layout_matches(layout, "B", "attacker"))
        self.assertFalse(layout_matches(layout, "A", "both"))
        before = scene_signature(game)
        game.planted_pos = [8, 42]
        self.assertEqual(scene_layout(game)["site"], "B")
        self.assertNotEqual(before, scene_signature(game))
        self.assertEqual(chars[0].pos, [4, 3])

    def test_a_single_survivor_is_not_a_dispersed_team(self):
        game = SimpleNamespace(width=44, planted_pos=[8, 42], chars=[
            SimpleNamespace(name="solo", team="A", pos=[3, 40], is_alive=True)])
        self.assertFalse(layout_matches(scene_layout(game), spread="attacker"))

    def test_failed_search_keeps_the_window_and_retry_draws_a_new_seed(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        viewer = SceneViewer(None, {}, seed=11, root=root)

        def unavailable(*args, **kwargs):
            if False:
                yield
            raise NoPostplantEntry("No matching actual plant")

        try:
            with patch("ghost_champions_v3.tools.gc3_watch_attacker_postplant.scene_steps", unavailable):
                viewer.next_scene()
                first_seed = viewer.request_seed
                root.after_cancel(viewer.after_id)
                viewer._prepare_next()
                self.assertIsNone(viewer.steps)
                self.assertEqual(str(viewer.next_button["state"]), "normal")
                self.assertTrue(root.winfo_exists())
                viewer.next_scene()
                self.assertNotEqual(viewer.request_seed, first_seed)
        finally:
            viewer.close()


if __name__ == "__main__":
    unittest.main()
