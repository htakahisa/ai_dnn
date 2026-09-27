"""Fnatic v3 runtime auto-aim and movement-facing regressions."""

import unittest
from types import SimpleNamespace as NS

from controllers import DefaultAttackerController, UserInputController
from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import parse_grid
from game_core import SHOOT_INTERVAL_TICKS
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_ultimate_system import UltimateTestGame, make_character


class FnaticV3FacingTests(unittest.TestCase):
    def fixture(self, side='A'):
        game = UltimateTestGame()
        char = make_character('Boaster', side, (4, 4))
        char.facing = 'W'
        ctrl = FnaticV3AttackerController() if side == 'A' else FnaticV3DefenderController()
        wrapper = IQAwareController(ctrl)
        wrapper.set_game(game)
        if side == 'A':
            game.attacker_controller = wrapper
        else:
            game.defender_controller = wrapper
        enemy = make_character('Derke', 'D' if side == 'A' else 'A', (4, 7))
        game.chars = [char, enemy]
        return game, char, enemy

    def test_attacker_and_defender_turn_to_clear_shot_enemy_behind_them(self):
        for side in ('A', 'D'):
            with self.subTest(side=side):
                game, char, _ = self.fixture(side)
                game._force_ai_facing_visible_enemy()
                self.assertEqual(char.facing, 'E')

    def test_nearest_enemy_is_selected_and_dead_enemy_ignored(self):
        game, char, _ = self.fixture()
        near = make_character('Leo', 'D', (2, 4))
        dead = make_character('Alfajer', 'D', (4, 3))
        dead.is_alive = False
        game.chars.extend([near, dead])
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'N')

    def test_wall_smoke_and_player_blocked_enemy_does_not_turn_actor(self):
        for obstruction in ('wall', 'smoke', 'player'):
            with self.subTest(obstruction=obstruction):
                game, char, _ = self.fixture()
                if obstruction == 'wall':
                    game.grid[4, 5] = 1
                elif obstruction == 'smoke':
                    game._smoke_cells = lambda: {(4, 5), (4, 6)}
                else:
                    game.chars.append(make_character('Leo', 'A', (4, 5)))
                game._force_ai_facing_visible_enemy()
                self.assertEqual(char.facing, 'W')

    def test_blocked_near_enemy_is_skipped_for_clear_far_enemy(self):
        game, char, _ = self.fixture()
        near = make_character('Leo', 'D', (2, 4))
        game.chars.append(near)
        game.grid[3, 4] = 1
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'E')

    def test_revealed_enemy_can_be_aimed_through_smoke_but_not_wall(self):
        game, char, enemy = self.fixture()
        enemy.reveal_remaining = 3
        game._smoke_cells = lambda: {(4, 5), (4, 6)}
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'E')
        char.facing = 'W'
        game.grid[4, 5] = 1
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'W')

    def test_clear_shot_defuser_has_priority_over_nearer_enemy(self):
        game, char, enemy = self.fixture()
        game.is_planted = True
        enemy.defuse_timer = 1
        game.chars.append(make_character('Leo', 'D', (2, 4)))
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'E')

    def test_other_controllers_and_manual_input_are_unchanged(self):
        for controller in (DefaultAttackerController(), UserInputController()):
            with self.subTest(controller=type(controller).__name__):
                game, char, _ = self.fixture()
                game.attacker_controller = NS(inner=controller)
                game._force_ai_facing_visible_enemy()
                self.assertEqual(char.facing, 'W')

    def test_gc_auto_aim_still_works(self):
        class GhostChampionsV1TestController:
            pass
        game, char, _ = self.fixture()
        game.attacker_controller = NS(inner=GhostChampionsV1TestController())
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, 'E')

    def test_runtime_auto_aim_allows_shooting_enemy_behind_actor(self):
        game, char, enemy = self.fixture()
        game.battle_tick = SHOOT_INTERVAL_TICKS
        game._force_ai_facing_visible_enemy()
        game._resolve_all_shots()
        self.assertTrue(any(shot['shooter'] is char and shot['target'] is enemy
                            for shot in game.last_shots))

    def test_real_guard_wait_and_movement_face_the_spike_again(self):
        game, char, _ = self.fixture()
        game.grid = parse_grid(NEW_MAZE_STR)
        game.height, game.width = game.grid.shape
        game.chars = [char]
        game.is_planted = True
        ctrl = FnaticV3AttackerController()
        game.attacker_controller = ctrl
        ctrl.set_game(game)
        anchor = ctrl.positions.plants[7][0]
        target = (2, 37)
        game.planted_pos = anchor
        ctrl.guard_anchor = anchor
        ctrl.guard_targets = {char.name: target}
        char.pos = list(target)
        game.move_character(char)
        game._force_ai_facing_visible_enemy()
        self.assertEqual(char.facing, game._facing_towards(char.pos, anchor))
        char.pos = [2, 36]
        game.move_character(char)
        self.assertEqual(char.pos, list(target))
        self.assertEqual(char.facing, game._facing_towards([2, 36], anchor))


if __name__ == '__main__':
    unittest.main()
