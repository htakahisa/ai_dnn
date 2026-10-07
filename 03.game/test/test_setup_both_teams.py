"""Setup movement applies the same map and action rules to both teams."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from battle_logic import BattleLogicMixin
from defender_setup_phase import DefenderSetupPhase


class SetupGame(BattleLogicMixin):
    def __init__(self):
        self.grid = np.zeros((2, 4), dtype=int)
        self.height, self.width = self.grid.shape
        self.chars = [self._char("A", 0), self._char("D", 1)]
        self.attacker_controller = Mock()
        self.defender_controller = Mock()
        self.spike_pos = None
        self.target_plant_pos = (0, 3)
        self.detonate_timer = 0
        self.round_timer = 30
        self.battle_tick = 0
        self.headless = True
        self.defender_setup_phase = DefenderSetupPhase(setup_ticks=2)
        self.defender_setup_phase.start()
        self._prepare_team_controllers_tick = Mock()
        self._build_occupancy_counts = Mock()
        self._clear_occupancy_counts = Mock()
        self._update_occupancy_after_move = Mock()

    @staticmethod
    def _char(team, row):
        return SimpleNamespace(
            team=team, name=team, pos=[row, 0], is_alive=True,
            facing=(0, 1), is_planting=True, plant_timer=1, defuse_timer=1,
        )

    def _is_position_occupied(self, char, target, old_pos):
        return any(other is not char and tuple(other.pos) == target
                   for other in self.chars)

    @staticmethod
    def _facing_from_delta(dr, dc, old_facing):
        return (dr, dc) if dr or dc else old_facing


class SetupBothTeamsTests(unittest.TestCase):
    def test_phase_allows_both_teams_on_the_same_mask(self):
        phase = DefenderSetupPhase(setup_ticks=1)
        phase.start()
        with patch("defender_setup_phase.is_setup_position_allowed",
                   side_effect=lambda row, col: col == 1):
            for team in ("A", "D"):
                self.assertTrue(phase.team_can_move(team))
                self.assertTrue(phase.can_move_to(team, 0, 1))
                self.assertFalse(phase.can_move_to(team, 0, 2))
            self.assertFalse(phase.attacker_is_frozen())
            self.assertFalse(phase.can_move_to("other", 0, 1))
        self.assertTrue(phase.advance_tick())
        self.assertTrue(phase.can_move_to("A", 0, 2))

    def test_tick_moves_both_teams_without_running_actions(self):
        game = SetupGame()
        a, d = game.chars
        game.attacker_controller.decide_move.return_value = (
            (0, 1), {"ability": "SMOKE", "target": (1, 1)})
        game.defender_controller.decide_move.return_value = (
            (1, 1), {"ability": "FLASH", "target": (0, 1)})
        with patch("battle_logic.build_team_position_view",
                   side_effect=lambda game, team: SimpleNamespace(chars=game.chars)), \
             patch("defender_setup_phase.is_setup_position_allowed",
                   side_effect=lambda row, col: col < 2):
            game._run_defender_setup_tick()
            self.assertEqual(tuple(a.pos), (0, 1))
            self.assertEqual(tuple(d.pos), (1, 1))
            self.assertEqual(game.defender_setup_phase.ticks_remaining, 1)
            game.attacker_controller.decide_move.return_value = (0, 2)
            game.defender_controller.decide_move.return_value = (1, 2)
            game._run_defender_setup_tick()

        self.assertEqual(tuple(a.pos), (0, 1))
        self.assertEqual(tuple(d.pos), (1, 1))
        self.assertEqual(game.attacker_controller.decide_move.call_count, 2)
        self.assertEqual(game.defender_controller.decide_move.call_count, 2)
        self.assertFalse(game.defender_setup_phase.active)
        self.assertEqual(game.round_timer, 30)
        self.assertEqual(game.battle_tick, 0)
        for char in (a, d):
            self.assertFalse(char.is_planting)
            self.assertEqual(char.plant_timer, 0)
            self.assertEqual(char.defuse_timer, 0)


if __name__ == "__main__":
    unittest.main()
