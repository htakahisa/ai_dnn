"""Fnatic front-player selection, main-group cohesion and Mid separation."""

import unittest
from unittest.mock import patch

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.formation import MID_CONTROL_POSITIONS, MAIN_GROUP_RADIUS
from fnatic_v3.positions import distances, parse_grid
from map_data import NEW_MAZE_STR
from test_fnatic_v3_rules import actor
from test_ultimate_system import UltimateTestGame, make_character
from iq_controller_adapter import IQAwareController


class FnaticV3FormationTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = FnaticV3AttackerController()
        self.grid = np.zeros((9, 24), dtype=np.int32)
        self.ctrl.target = (4, 22)
        self.carrier = actor('carrier', (4, 4))
        self.carrier.has_spike = True

    def decide(self, char, allies):
        alive = [c for c in allies if c.is_alive]
        blocked = {tuple(c.pos) for c in alive if c.name != char.name}
        return self.ctrl.formation.result(self.ctrl, char, self.carrier,
                                          alive, self.grid, blocked, [])

    def test_front_priority_uses_nearby_living_players_only(self):
        derke = actor('Derke', (4, 12))  # Eight cells away; no front assignment.
        chronicle = actor('Chronicle', (4, 6))
        leo = actor('Leo', (3, 6))
        allies = [self.carrier, derke, chronicle, leo]
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Chronicle')
        derke.pos = [4, 11]  # Exactly seven cells is eligible.
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Derke')
        derke.is_alive = False
        chronicle.is_alive = False
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.front_name, 'Leo')

    def test_all_five_front_priorities_in_order(self):
        for expected, names in (
            ('Derke', ('Derke', 'Chronicle', 'Leo', 'Boaster', 'Alfajer')),
            ('Chronicle', ('Chronicle', 'Leo', 'Boaster', 'Alfajer')),
            ('Leo', ('Leo', 'Boaster', 'Alfajer')),
            ('Boaster', ('Boaster', 'Alfajer')),
            ('Alfajer', ('Alfajer',)),
        ):
            with self.subTest(expected=expected):
                self.ctrl.reset_round()
                self.ctrl.target = (4, 22)
                allies = [self.carrier] + [actor(name, (3, 4 + i)) for i, name in enumerate(names)]
                # A unit already serving in a separate group is excluded;
                # here suppress Mid so the final two priorities can be tested.
                self.ctrl.formation.mid_name = None
                with patch('fnatic_v3.formation.MID_PRIORITY', ()):
                    self.decide(self.carrier, allies)
                self.assertEqual(self.ctrl.formation.front_name, expected)

    def test_mid_uses_alfajer_then_boaster_and_never_carrier(self):
        alfajer = actor('Alfajer', (3, 4))
        boaster = actor('Boaster', (5, 4))
        allies = [self.carrier, alfajer, boaster]
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
        alfajer.is_alive = False
        self.decide(self.carrier, allies)
        self.assertEqual(self.ctrl.formation.mid_name, 'Boaster')
        self.carrier.name = 'Boaster'
        self.decide(self.carrier, [self.carrier])
        self.assertIsNone(self.ctrl.formation.mid_name)

    def test_separate_mid_is_not_recalled_as_front_even_when_nearby(self):
        alfajer = actor('Alfajer', (4, 6))
        self.decide(self.carrier, [self.carrier, alfajer])
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
        self.assertIsNone(self.ctrl.formation.front_name)

    def test_carrier_waits_until_nearby_front_gets_ahead(self):
        derke = actor('Derke', (4, 3))
        self.assertEqual(self.decide(self.carrier, [self.carrier, derke])[0], self.carrier.pos)
        for _ in range(10):
            derke.pos = self.decide(derke, [self.carrier, derke])[0]
            lengths = distances(self.ctrl.target, self.grid)
            if lengths[tuple(derke.pos)] <= lengths[tuple(self.carrier.pos)] - 2:
                break
        self.assertLessEqual(lengths[tuple(derke.pos)], lengths[tuple(self.carrier.pos)] - 2)
        self.assertEqual(self.decide(self.carrier, [self.carrier, derke])[0], [4, 5])

    def test_main_group_is_brought_near_carrier_instead_of_rushing_alone(self):
        derke = actor('Derke', (4, 6))
        chronicle = actor('Chronicle', (1, 12))
        allies = [self.carrier, derke, chronicle]
        self.assertEqual(self.decide(self.carrier, allies)[0], self.carrier.pos)
        before = distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)]
        chronicle.pos = self.decide(chronicle, allies)[0]
        after = distances(tuple(self.carrier.pos), self.grid)[tuple(chronicle.pos)]
        self.assertLess(after, before)

    def test_mid_stays_independent_in_both_rush_and_default(self):
        grid = parse_grid(NEW_MAZE_STR)
        for strategy in ('RUSH', 'DEFAULT'):
            with self.subTest(strategy=strategy):
                self.ctrl.reset_round()
                self.ctrl.strategy = strategy
                carrier = actor('Leo', (7, 3))
                carrier.has_spike = True
                alfajer = actor('Alfajer', (13, 23))
                state = dict(grid=grid, chars=[carrier, alfajer], round_timer=100)
                self.ctrl.decide_move(alfajer, state)
                self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
                self.assertEqual(self.ctrl.formation.mid_target, (13, 23))
                self.assertEqual(self.ctrl.decide_move(alfajer, state)[0], alfajer.pos)
                self.assertIsNone(self.ctrl.formation.front_name)

    def test_mid_moves_to_registered_guard_after_plant(self):
        grid = parse_grid(NEW_MAZE_STR)
        carrier = actor('Leo', (7, 3))
        carrier.has_spike = True
        alfajer = actor('Alfajer', (13, 23))
        state = dict(grid=grid, chars=[carrier, alfajer], round_timer=100)
        self.ctrl.decide_move(alfajer, state)
        carrier.has_spike = False
        state.update(is_planted=True, planted_pos=(7, 3))
        self.ctrl.decide_move(alfajer, state)
        self.assertIn(self.ctrl.guard_targets['Alfajer'], self.ctrl.positions.guards[6])

    def test_dropped_spike_is_recovered_by_main_group_without_recalling_mid(self):
        grid = parse_grid(NEW_MAZE_STR)
        derke = actor('Derke', (10, 3))
        alfajer = actor('Alfajer', (13, 23))
        self.ctrl.formation.mid_name = 'Alfajer'
        self.ctrl.formation.mid_target = (13, 23)
        state = dict(grid=grid, chars=[derke, alfajer], round_timer=100, spike_pos=(9, 3))
        self.assertEqual(self.ctrl.decide_move(derke, state)[0], [9, 3])
        self.assertEqual(self.ctrl.retriever, 'Derke')
        self.assertEqual(self.ctrl.decide_move(alfajer, state)[0], alfajer.pos)
        self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')

    def test_main_group_keeps_recon_utility_when_holding_formation(self):
        grid = parse_grid(NEW_MAZE_STR)
        carrier = actor('Leo', (12, 3))
        carrier.has_spike = True
        front = actor('Derke', (10, 3))
        support = actor('Chronicle', (13, 3), recon_charges=1)
        support.ability_name = 'RECON'
        state = dict(grid=grid, chars=[carrier, front, support], round_timer=100)
        result = self.ctrl.decide_move(support, state)
        self.assertEqual(result[1]['ability'], 'RECON')
        self.assertEqual(result[0], support.pos)

    def test_real_team_reaches_plant_while_mid_remains_separate(self):
        for strategy, use_iq, marker in (('RUSH', False, 5), ('DEFAULT', False, 5),
                                         ('RUSH', True, 5), ('RUSH', False, 7),
                                         ('RUSH', False, 8)):
            with self.subTest(strategy=strategy, use_iq=use_iq, marker=marker):
                self.ctrl.reset_round()
                self.ctrl.strategy = strategy
                self.ctrl.target = self.ctrl.positions.plants[marker][0]
                grid = parse_grid(NEW_MAZE_STR)
                game = UltimateTestGame(*grid.shape)
                game.grid = grid
                names = ('Leo', 'Boaster', 'Derke', 'Chronicle', 'Alfajer')
                chars = [make_character(name, 'A', (23, 17 + i)) for i, name in enumerate(names)]
                carrier = chars[0]
                carrier.has_spike = True
                game.chars = chars
                game.attacker_controller = self.ctrl
                self.ctrl.set_game(game)
                if use_iq:
                    wrapper = IQAwareController(self.ctrl)
                    wrapper.set_game(game)
                    game.attacker_controller = wrapper
                separated = False
                for _ in range(160):
                    for c in chars:
                        old_carrier = tuple(carrier.pos)
                        game.move_character(c)
                        if c is carrier and tuple(carrier.pos) != old_carrier:
                            remaining = distances(self.ctrl.target, grid)
                            if remaining[old_carrier] > 2:
                                self.assertTrue(any(
                                    remaining[tuple(teammate.pos)] < remaining[tuple(carrier.pos)]
                                    for teammate in chars if teammate is not carrier
                                    and teammate.name != self.ctrl.formation.mid_name))
                    game.battle_tick += 1
                    self.assertEqual(len({tuple(c.pos) for c in chars}), 5)
                    if self.ctrl.formation.mid_target is not None:
                        separated |= tuple(chars[-1].pos) == self.ctrl.formation.mid_target
                    if game.is_planted:
                        break
                self.assertTrue(game.is_planted, [(c.name, c.pos) for c in chars])
                self.assertEqual(self.ctrl.formation.mid_name, 'Alfajer')
                self.assertTrue(separated)
                lengths = distances(tuple(carrier.pos), grid)
                self.assertTrue(all(lengths[tuple(c.pos)] <= MAIN_GROUP_RADIUS + 1
                                    for c in chars[1:-1]))


if __name__ == '__main__':
    unittest.main()
