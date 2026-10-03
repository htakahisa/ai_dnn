import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
from train_guard_reposition_gc import GuardProgress, route_action, selection_score, navigation_teacher
from training_opponent_pool_gc import opponent_for_episode, OPPONENT_SPECS


class GuardRepositionTests(unittest.TestCase):
    def test_guard_overlay_uses_the_same_banner_offset_as_the_map(self):
        from gc_v1.watch_guard_reposition_gc import GuardWatchBattle, VisualFPSBattle
        from game_core import COMBO_BANNER_HEIGHT
        game = GuardWatchBattle.__new__(GuardWatchBattle)
        game.headless = False
        game.is_planted = True
        game.cell_size = 24
        game.map_offset_x = 250
        game.canvas = NS(create_rectangle=Mock(), create_text=Mock())
        game.chars = [NS(name="guard", team="A", is_alive=True),
                      NS(name="dead", team="A", is_alive=False)]
        game.attacker_controller = NS(inner_controller=NS(guard=NS(
            _assigned_guard_positions={"guard": (10, 2), "dead": (6, 3)})))
        with patch.object(VisualFPSBattle, "draw"):
            game.draw()
        game.canvas.create_rectangle.assert_called_once_with(
            game._map_x(2 * 24) + 1, COMBO_BANNER_HEIGHT + 10 * 24 + 1,
            game._map_x(3 * 24) - 1, COMBO_BANNER_HEIGHT + 11 * 24 - 1,
            outline="#00dfff", width=2)
        self.assertEqual(game.canvas.create_text.call_args.args[:2],
                         (game._map_x(2 * 24) + 12, COMBO_BANNER_HEIGHT + 10 * 24 + 12))

    def test_rotation_keeps_each_dedicated_ai_with_its_roster(self):
        from party_presets import get_preset
        rotation = [opponent_for_episode(i) for i in range(10)]
        self.assertEqual([(o.name, o.ai_key) for o in rotation[:5]], list(OPPONENT_SPECS))
        self.assertEqual(rotation[:5], rotation[5:])
        for opponent in rotation:
            preset = get_preset(opponent.name)
            self.assertEqual(opponent.players, tuple(preset.players))
            self.assertEqual(opponent.igl, preset.igl)
            self.assertEqual(opponent.spike_holder, preset.spike_holder)

    def test_route_can_detour_away_from_goal_around_a_teammate(self):
        grid = np.ones((5, 5), dtype=int)
        grid[1:4, 1:4] = 0
        grid[1, 2] = 1
        char = NS(name="guard", pos=[2, 1], is_alive=True)
        ally = NS(name="ally", pos=[2, 2], is_alive=True)
        action = route_action(grid, char.pos, (2, 3), [char, ally], char.name, np.ones(12, bool))
        self.assertEqual(action, 4)  # Move down first, temporarily farther away.
        grid[3, 1] = 1
        self.assertIsNone(route_action(grid, char.pos, (2, 3), [char, ally], char.name, np.ones(12, bool)))

    def test_public_hidden_defuse_alert_overrides_guard_goal(self):
        char = NS(name="guard", pos=[2, 2], is_alive=True, team="A")
        controller = NS(_active_defuse_info=lambda state: {"name": "hidden"},
                        _assigned_guard_positions={"guard": (2, 0)})
        state = {"grid": np.zeros((5, 5), int), "chars": [char], "planted_pos": (2, 4)}
        self.assertEqual(navigation_teacher(controller, char, state, np.ones(12, bool)), 8)

    def test_no_enemy_information_is_needed_to_return_from_mid(self):
        char = NS(name="guard", pos=[2, 4], is_alive=True, team="A")
        controller = NS(_active_defuse_info=lambda state: None,
                        _assigned_guard_positions={"guard": (2, 0)})
        state = {"grid": np.zeros((5, 5), int), "chars": [char], "planted_pos": (2, 1)}
        self.assertEqual(navigation_teacher(controller, char, state, np.ones(12, bool)), 6)

    def test_progress_and_arrival_cannot_be_farmed_by_oscillation(self):
        progress = GuardProgress()
        reward = lambda before, after: progress.reward("p", (2, 3), before, after, False, False, True)
        self.assertGreater(reward(3, 2), 0)
        self.assertEqual(reward(2, 3), 0)
        self.assertEqual(reward(3, 2), 0)
        arrival = reward(2, 0)
        reward(0, 2)
        self.assertGreater(arrival, reward(2, 0))

    def test_blocked_or_engaged_guard_is_not_penalized_for_waiting(self):
        progress = GuardProgress()
        self.assertLess(progress.reward("p", (2, 3), 5, 5, False, False, True), 0)
        self.assertEqual(progress.reward("p", (2, 3), 5, 5, False, False, False), 0)
        self.assertEqual(progress.reward("p", (2, 3), 5, 5, True, False, True), 0)

    def test_arrival_improvement_cannot_hide_one_opponents_defuse_regression(self):
        previous = {"postplant_episodes": 25, "postplant_win_rate": .5, "defuse_loss_rate": .2}
        baseline = {**previous, "far_arrival_rate": .1, "far_quiet_stall_rate": .5,
                    "arrival_rate": .2, "by_opponent_name": {"SUPES": previous}}
        candidate = {**baseline, "far_arrival_rate": .9,
                     "by_opponent_name": {"SUPES": {**previous, "defuse_loss_rate": .4}}}
        self.assertEqual(selection_score(candidate, baseline)[0], 0)
        candidate["by_opponent_name"]["SUPES"]["defuse_loss_rate"] = .2
        self.assertEqual(selection_score(candidate, baseline)[0], 1)


if __name__ == "__main__":
    unittest.main()
