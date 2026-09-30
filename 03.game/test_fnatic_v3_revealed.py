"""Revealed enemy targeting through actual recon, blast and shot geometry."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.positions import TacticalPositions
from game_core import SHOOT_INTERVAL_TICKS
from iq_controller_adapter import IQAwareController
from iq_perception import PerceivedCharacter
from test_fnatic_v3_retake import retake_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticRevealedTests(unittest.TestCase):
    def fixture(self, side='D', neon=False, wall=False, use_iq=False):
        game = UltimateTestGame(10, 24)
        if wall:
            game.grid[:, 8] = 1
        game.grid[1, 1] = game.grid[1, 22] = 2
        char = make_character('Alfajer' if neon else 'Derke', side, (4, 4))
        char.ability_name = 'NONE'
        char.ultimate_name = 'NEON' if neon else 'NONE'
        char.ultimate_points = char.ultimate_cost if neon else 0
        char.iq = char.effective_iq = 200
        game.chars = [char]
        marked = np.where(game.grid == 1, 1, 0)
        marked[1, 1], marked[1, 22] = 5, 6
        text = '\n'.join(''.join(map(str, row)) for row in marked)
        ctrl = (FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
                if side == 'A' else FnaticV3DefenderController('', '', engineer_map=''))
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        if side == 'A':
            game.attacker_controller = adapter
        else:
            game.defender_controller = adapter
        return game, ctrl, char

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    detonate_timer=game.detonate_timer)

    @staticmethod
    def enemy(game, cell, revealed=True, name='Enemy'):
        enemy = make_character(name, 'A' if game.chars[0].team == 'D' else 'D', cell)
        enemy.reveal_remaining = 5 if revealed else 0
        game.chars.append(enemy)
        return enemy

    def test_recon_triggers_neon_on_both_sides_without_entry_or_wall_los(self):
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, _, char = self.fixture(side, neon=True, wall=True, use_iq=use_iq)
                    enemy = self.enemy(game, (4, 14), revealed=False)
                    game._explode_recon(dict(team=side, owner='Leo', path=[(4, 14)], progress=0))
                    self.assertGreater(enemy.reveal_remaining, 0)
                    self.assertFalse(game.check_line_of_sight(char, enemy))
                    game.move_character(char)
                    self.assertEqual(char.ultimate_points, 0)
                    self.assertEqual(game.neon_bursts[0]['pos'], tuple(enemy.pos))
                    for _ in range(11):
                        game._advance_engineer_effects()
                    self.assertLess(enemy.hp, enemy.max_hp)

    def test_neon_uses_midpoint_to_hit_all_three_instead_of_enemy_centers(self):
        game, _, char = self.fixture(neon=True, wall=True)
        enemies = [self.enemy(game, cell, False, 'Enemy' + str(i))
                   for i, cell in enumerate(((4, 11), (4, 17), (5, 15)))]
        game._explode_recon(dict(team='D', owner='Leo', path=[(4, 14)], progress=0))
        self.enemy(game, (4, 23), False, 'Unrevealed')
        dead = self.enemy(game, (1, 14), True, 'Dead')
        dead.is_alive = False
        game.move_character(char)
        burst = game.neon_bursts[0]
        self.assertTrue(all(tuple(c.pos) in burst['cells'] for c in enemies))
        self.assertNotIn(burst['pos'], [tuple(c.pos) for c in enemies])

    def test_neon_chooses_legal_center_if_best_midpoint_is_a_wall(self):
        game, _, char = self.fixture(neon=True)
        game.grid[4, 14] = 1
        enemies = [self.enemy(game, p, name=str(i)) for i, p in enumerate(((4, 11), (4, 17)))]
        game.move_character(char)
        burst = game.neon_bursts[0]
        self.assertNotEqual(game.grid[burst['pos']], 1)
        self.assertTrue(all(tuple(c.pos) in burst['cells'] for c in enemies))

    def test_neon_requires_points_and_timed_reveal_and_uses_only_perceived_positions(self):
        game, ctrl, char = self.fixture(neon=True)
        enemy = self.enemy(game, (4, 14), False)
        enemy.los_revealed = True
        self.assertIsNone(ctrl.ultimates.revealed_neon(ctrl, char, self.state(game)))
        enemy.reveal_remaining = 5
        char.ultimate_points -= 1
        self.assertIsNone(ctrl.ultimates.revealed_neon(ctrl, char, self.state(game)))
        char.ultimate_points = char.ultimate_cost
        state = self.state(game)
        state['chars'] = [char, PerceivedCharacter(enemy, pos=[8, 20])]
        result = ctrl.ultimates.revealed_neon(ctrl, char, state)
        self.assertEqual(result[1]['target'], (8, 20))

    def test_neon_can_cast_stationary_while_retake_group_is_gathering(self):
        game, _, char = self.fixture(neon=True, wall=True)
        char.pos = [6, 3]
        self.enemy(game, (4, 14))
        game.is_planted, game.planted_pos = True, (1, 1)
        ctrl = FnaticV3DefenderController('', retake_map(game.grid, ((6, 3), (6, 4), (6, 5)), ()),
                                          engineer_map='')
        ctrl.set_game(game)
        game.defender_controller = ctrl
        game.move_character(char)
        self.assertTrue(ctrl.retake.gathering)
        self.assertEqual(char.pos, [6, 3])
        self.assertEqual(char.ultimate_points, 0)

    def test_peek_round_corner_then_stay_and_shoot_when_clear_with_iq(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                game, _, char = self.fixture(use_iq=use_iq)
                game.grid[4, 5] = 1
                enemy = self.enemy(game, (4, 10))
                self.assertFalse(game.check_shot_line_of_sight(char, enemy))
                game.move_character(char)
                self.assertNotEqual(char.pos, [4, 4])
                self.assertTrue(game.check_shot_line_of_sight(char, enemy))
                arrived = list(char.pos)
                game.battle_tick += 1
                game.move_character(char)
                self.assertEqual(char.pos, arrived)
                enemy.facing = 'E'
                game.battle_tick = SHOOT_INTERVAL_TICKS
                game._resolve_all_shots()
                self.assertTrue(any(shot['shooter'] is char and shot['target'] is enemy
                                    for shot in game.last_shots))

    def corridor(self, steps):
        game, ctrl, char = self.fixture()
        game.grid[:] = 1
        game.grid[1, 1] = 2
        game.grid[6, 5 - steps:6] = 0
        game.grid[2:7, 5] = 0
        char.pos = [6, 5 - steps]
        enemy = self.enemy(game, (2, 5))
        return game, ctrl, char, enemy

    def test_exactly_three_walking_steps_peak_but_four_steps_do_not(self):
        game, _, char, enemy = self.corridor(3)
        for expected in ((6, 3), (6, 4), (6, 5), (6, 5)):
            game.move_character(char)
            game.battle_tick += 1
            self.assertEqual(tuple(char.pos), expected)
        self.assertTrue(game.check_shot_line_of_sight(char, enemy))
        game, ctrl, char, _ = self.corridor(4)
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))
        game.move_character(char)
        self.assertEqual(char.pos, [6, 1])

    def test_peak_handles_ally_on_shot_line_and_does_not_move_into_occupied_cells(self):
        game, _, char = self.fixture()
        enemy = self.enemy(game, (4, 10))
        ally = make_character('Leo', 'D', (4, 6))
        game.chars.append(ally)
        self.assertFalse(game.check_shot_line_of_sight(char, enemy))
        game.move_character(char)
        self.assertNotEqual(char.pos, ally.pos)
        self.assertTrue(game.check_shot_line_of_sight(char, enemy))
        game, ctrl, char, _ = self.corridor(3)
        game.chars.append(make_character('Leo', 'D', (6, 3)))
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))

    def test_recon_target_is_shootable_through_smoke_and_stays_without_casting(self):
        game, _, char = self.fixture()
        char.ability_name, char.recon_charges = 'RECON', 2
        enemy = self.enemy(game, (4, 10))
        game.smokes.append(dict(cells={(4, 6), (4, 7)}, team='A', owner='Enemy', remaining_ticks=10))
        self.assertTrue(game.check_shot_line_of_sight(char, enemy))
        game.move_character(char)
        self.assertEqual(char.pos, [4, 4])
        self.assertEqual(char.recon_charges, 2)

    def test_los_revealed_enemies_can_be_peaked_but_do_not_ignore_smoke(self):
        game, ctrl, char = self.fixture()
        game.grid[4, 5] = 1
        enemy = self.enemy(game, (4, 10), False)
        enemy.los_revealed = True
        result = ctrl.reveal_peek.result(ctrl, char, self.state(game))
        self.assertNotEqual(result[0], char.pos)
        game.smokes.append(dict(cells=set(zip(*np.where(game.grid != 1))), team='A',
                                owner='Enemy', remaining_ticks=10))
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))

    def test_expired_dead_and_unrevealed_enemies_and_attackers_do_not_trigger_peak(self):
        game, ctrl, char = self.fixture()
        game.grid[4, 5] = 1
        enemy = self.enemy(game, (4, 10), False)
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))
        enemy.reveal_remaining = 3
        enemy.is_alive = False
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))
        game, ctrl, char = self.fixture('A')
        self.enemy(game, (4, 10))
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, self.state(game)))

    def test_setup_and_defuse_and_urgent_retake_keep_their_priority(self):
        game, ctrl, char = self.fixture(neon=True)
        enemy = self.enemy(game, (4, 10))
        state = self.state(game)
        state['defender_setup_active'] = True
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, state))
        char.pos = [4, 9]
        game.is_planted, game.planted_pos = True, (4, 8)
        game.move_character(char)
        self.assertEqual(char.defuse_timer, 1)
        self.assertEqual(char.ultimate_points, char.ultimate_cost)
        game, ctrl, char = self.fixture()
        enemy = self.enemy(game, (4, 10))
        game.is_planted, game.planted_pos, game.detonate_timer = True, (1, 1), 19
        state = self.state(game)
        state['detonate_timer'] = 100
        self.assertIsNone(ctrl.reveal_peek.result(ctrl, char, state))


if __name__ == '__main__':
    unittest.main()
