import unittest
from analytics.replay_viewer import ReplayViewer


class ReplayVisibilityTests(unittest.TestCase):
    def test_team_view_uses_per_team_visibility(self):
        viewer = ReplayViewer.__new__(ReplayViewer)
        class Mode:
            def get(self):
                return "A"
        viewer.view_mode = Mode()
        self.assertTrue(viewer._char_visible({"team": "A"}))
        self.assertTrue(viewer._char_visible({"team": "D", "visible_to": ["A"]}))
        self.assertFalse(viewer._char_visible({"team": "D", "visible_to": ["D"]}))

    def test_old_frames_fall_back_to_global_reveal(self):
        viewer = ReplayViewer.__new__(ReplayViewer)
        class Mode:
            def get(self):
                return "D"
        viewer.view_mode = Mode()
        self.assertTrue(viewer._char_visible({"team": "A", "revealed": True}))
        self.assertFalse(viewer._char_visible({"team": "A", "revealed": False}))


if __name__ == "__main__":
    unittest.main()
