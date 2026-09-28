"""Fnatic response to the public defuse tap, movement and sight checks."""

import unittest

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.positions import distances, parse_grid
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_fnatic_v3_engineer import lamp_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticDefuseResponseTests(unittest.TestCase):
    def fixture(self, players=(('Leo', (11, 6)), ('Derke', (15, 6))), use_iq=False, engineer_map=''):
        grid = parse_grid(NEW_MAZE_STR)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        game.is_planted = True
        game.planted_pos = (9, 3)
        allies = [make_character(name, 'A', cell) for name, cell in players]
        for char in allies:
            char.iq = char.effective_iq = 200
        enemy = make_character('Demon1', 'D', (10, 3))
        enemy.defuse_timer = 1
        game.chars = allies + [enemy]
        ctrl = FnaticV3AttackerController(engineer_map=engineer_map)
        controller = IQAwareController(ctrl) if use_iq else ctrl
        controller.set_game(game)
        game.attacker_controller = controller
        ctrl.guard_anchor = game.planted_pos
        ctrl.guard_targets = {c.name: tuple(c.pos) for c in allies}
        return game, ctrl, allies, enemy

    def state(self, game, chars=None, notify=True):
        return dict(grid=game.grid, chars=game.chars if chars is None else chars,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    defender_defuse_info=({c.name: (c.defuse_timer, 6) for c in game.chars
                                          if c.team == 'D' and c.is_alive} if notify else {}))

    def test_notification_routes_to_spike_even_when_enemy_is_unobserved(self):
        game, ctrl, allies, _ = self.fixture()
        char = allies[0]
        state = self.state(game, chars=allies)
        result = ctrl.decide_move(char, state)
        self.assertEqual(ctrl.defuse_responder, 'Leo')
        self.assertEqual(result[0], [11, 5])
        self.assertNotIn('ability', result[1])
        self.assertEqual(char.recon_charges, 2)
        # Seeing neither the enemy nor a tap keeps the mapped guard position.
        result = ctrl.decide_move(char, self.state(game, chars=allies, notify=False))
        self.assertEqual(result[0], char.pos)
        self.assertIsNone(ctrl.defuse_responder)

    def test_runner_stops_as_soon_as_defuser_is_visible_and_can_be_shot(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                game, ctrl, allies, enemy = self.fixture(use_iq=use_iq)
                char = allies[0]
                lengths = distances(game.planted_pos, game.grid)
                moved = False
                for _ in range(10):
                    seen = game.check_shot_line_of_sight(char, enemy)
                    before = tuple(char.pos)
                    game.move_character(char)
                    game.battle_tick += 1
                    if seen:
                        self.assertEqual(tuple(char.pos), before)
                        break
                    moved |= tuple(char.pos) != before
                    self.assertLess(lengths[tuple(char.pos)], lengths[before])
                else:
                    self.fail('Runner never reached a view of the defuser')
                self.assertTrue(moved)
                self.assertEqual(ctrl.defuse_responder, char.name)
                self.assertEqual(char.recon_charges, 2)
                stopped = tuple(char.pos)
                for _ in range(3):
                    game.move_character(char)
                    game.battle_tick += 1
                    self.assertEqual(tuple(char.pos), stopped)
                self.assertEqual(char.facing, game._facing_towards(char.pos, enemy.pos))

    def test_selects_one_nearest_by_walking_distance_and_keeps_assignment(self):
        game, ctrl, allies, _ = self.fixture((('Derke', (14, 3)), ('Chronicle', (9, 6))))
        nearest, other = allies
        # Chronicle is geometrically closer but has the longer walking route.
        lengths = distances(game.planted_pos, game.grid)
        self.assertLess(lengths[tuple(nearest.pos)], lengths[tuple(other.pos)])
        ctrl.decide_move(other, self.state(game))  # Decision order cannot choose the runner.
        self.assertEqual(ctrl.defuse_responder, 'Derke')
        other.pos = [10, 2]  # Someone moving closer does not create a second runner.
        other.flash_charges = 0
        result = ctrl.decide_move(other, self.state(game))
        self.assertEqual(ctrl.defuse_responder, 'Derke')
        self.assertNotEqual(result[0], list(game.planted_pos))

    def test_dead_runner_is_replaced_by_next_living_ally(self):
        game, ctrl, allies, _ = self.fixture()
        ctrl.decide_move(allies[0], self.state(game))
        self.assertEqual(ctrl.defuse_responder, 'Leo')
        allies[0].is_alive = False
        result = ctrl.decide_move(allies[1], self.state(game))
        self.assertEqual(ctrl.defuse_responder, 'Derke')
        self.assertNotEqual(result[0], allies[1].pos)

    def test_smoke_requires_approach_until_adjacent_but_recon_can_reveal_defuser(self):
        for reveal, sees_through_smoke in ((0, False), (5, False), (0, True)):
            with self.subTest(reveal=reveal, sees_through_smoke=sees_through_smoke):
                game, ctrl, allies, enemy = self.fixture((('Leo', (12, 3)),))
                char = allies[0]
                char.sees_through_smoke = sees_through_smoke
                game.smokes = [{'cells': {(11, 3)}, 'remaining_ticks': 10, 'owner': enemy.name}]
                enemy.reveal_remaining = reveal
                before = list(char.pos)
                game.move_character(char)
                if reveal or sees_through_smoke:
                    self.assertEqual(char.pos, before)
                else:
                    self.assertEqual(char.pos, [11, 3])
                    game.battle_tick += 1
                    game.move_character(char)
                    self.assertEqual(char.pos, [11, 3])
                self.assertEqual(char.recon_charges, 2)

    def test_visible_other_enemy_does_not_stop_the_runner(self):
        game, ctrl, allies, _ = self.fixture()
        decoy = make_character('Alfajer', 'D', (11, 7))
        game.chars.append(decoy)
        self.assertTrue(game.check_shot_line_of_sight(allies[0], decoy))
        game.move_character(allies[0])
        self.assertEqual(allies[0].pos, [11, 5])

    def test_recon_reveal_through_wall_does_not_stop_approach(self):
        game, ctrl, allies, enemy = self.fixture((('Leo', (10, 6)),))
        enemy.reveal_remaining = 5
        self.assertFalse(game.check_shot_line_of_sight(allies[0], enemy))
        game.move_character(allies[0])
        self.assertEqual(allies[0].pos, [11, 6])

    def test_defuse_interrupts_nearest_engineers_lamp_mission_then_resumes_it(self):
        grid = parse_grid(NEW_MAZE_STR)
        game, ctrl, allies, enemy = self.fixture((('Alfajer', (11, 6)), ('Derke', (15, 6))),
                                                 engineer_map=lamp_map(grid, ((11, 6), (12, 6))))
        engineer = allies[0]
        ctrl.engineer.target = (11, 6)
        game.move_character(engineer)
        self.assertEqual(ctrl.defuse_responder, 'Alfajer')
        self.assertEqual(engineer.pos, [11, 5])
        self.assertEqual(engineer.ramp_charges, 2)
        enemy.defuse_timer = 0
        game.battle_tick += 1
        game.move_character(engineer)
        self.assertIsNone(ctrl.defuse_responder)
        self.assertEqual(engineer.pos, [11, 6])
        game.battle_tick += 1
        game.move_character(engineer)
        self.assertEqual(engineer.ramp_charges, 1)

    def test_cancelled_tap_and_round_reset_clear_response(self):
        game, ctrl, allies, enemy = self.fixture()
        char = allies[0]
        char.recon_charges = 0  # Isolate guard resumption from normal utility casts.
        game.move_character(char)
        self.assertNotEqual(tuple(char.pos), ctrl.guard_targets[char.name])
        enemy.defuse_timer = 0
        game.battle_tick += 1
        game.move_character(char)
        self.assertIsNone(ctrl.defuse_responder)
        self.assertEqual(ctrl.defuse_names, ())
        self.assertEqual(tuple(char.pos), ctrl.guard_targets[char.name])
        enemy.defuse_timer = 1
        game.battle_tick += 1
        game.move_character(char)
        self.assertEqual(ctrl.defuse_responder, char.name)
        ctrl.reset_round()
        self.assertIsNone(ctrl.defuse_responder)
        self.assertEqual(ctrl.defuse_names, ())


if __name__ == '__main__':
    unittest.main()
