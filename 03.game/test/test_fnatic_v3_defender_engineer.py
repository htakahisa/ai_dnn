"""Defending engineer finishes mapped lamps before taking its guard slot."""

import unittest
from unittest.mock import patch

import numpy as np

from defender_setup_phase import DefenderSetupPhase
from fnatic_v3.controller import FnaticV3DefenderController
from fnatic_v3.map_data_defender_ability_fnatic import ENGINEER_LAMP_STR
from fnatic_v3.positions import distances, parse_grid
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_fnatic_v3_defender_positions import placement_map
from test_fnatic_v3_engineer import lamp_map
from test_fnatic_v3_retake import retake_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticDefenderEngineerTests(unittest.TestCase):
    def fixture(self, use_iq=False, candidates=((2, 22), (2, 24), (2, 26))):
        grid = parse_grid(NEW_MAZE_STR)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        game.chars = [make_character('Alfajer', 'D', (1, 18)),
                      make_character('Leo', 'D', (1, 19))]
        ctrl = FnaticV3DefenderController(
            placement_map(grid, {3: ((2, 17), (2, 18))}), retake_map='',
            engineer_map=lamp_map(grid, candidates))
        ctrl.defender_positions.targets = {'Alfajer': (2, 17), 'Leo': (2, 18)}
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        game.defender_controller = adapter
        return game, ctrl, game.chars[0]

    @staticmethod
    def state(game, setup=False):
        return dict(grid=game.grid, chars=game.chars, is_planted=game.is_planted,
                    planted_pos=game.planted_pos, defender_setup_active=setup,
                    detonate_timer=game.detonate_timer, round_timer=100)

    def test_two_distinct_lamps_then_guard_with_real_effects_and_iq(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                game, ctrl, char = self.fixture(use_iq)
                placed = []
                for _ in range(80):
                    for c in game.chars:
                        before = char.ramp_charges
                        game.move_character(c)
                        if char.ramp_charges < before:
                            placed.append(tuple(char.pos))
                        if c is char and char.ramp_charges > 0:
                            self.assertIn(ctrl.engineer.target, ctrl.engineer.candidates)
                    game.battle_tick += 1
                    if char.ramp_charges == 0 and tuple(char.pos) == ctrl.defender_positions.targets[char.name]:
                        break
                self.assertEqual(len(set(placed)), 2)
                self.assertTrue(set(placed) <= set(ctrl.engineer.candidates))
                self.assertEqual(char.ramp_charges, 0)
                self.assertEqual(len(game.ramp_traps), 2)
                self.assertTrue(all(trap['team'] == 'D' for trap in game.ramp_traps))
                self.assertEqual(tuple(char.pos), ctrl.defender_positions.targets[char.name])

    def test_setup_moves_to_first_lamp_and_waits_then_live_places_before_guard(self):
        game, ctrl, char = self.fixture()
        game.defender_setup_phase = DefenderSetupPhase()
        game.defender_setup_phase.start()
        for _ in range(20):
            game._move_character_during_defender_setup(char)
        first = ctrl.engineer.target
        self.assertIn(first, ctrl.engineer.candidates)
        self.assertEqual(tuple(char.pos), first)
        self.assertNotEqual(first, ctrl.defender_positions.targets[char.name])
        self.assertEqual(char.ramp_charges, 2)
        self.assertIsNone(ctrl.engineer.pending)
        game.move_character(char)
        self.assertEqual(char.ramp_charges, 1)
        self.assertEqual(tuple(game.ramp_traps[0]['pos']), first)
        for _ in range(50):
            game.move_character(char)
            game.battle_tick += 1
            if char.ramp_charges == 0 and tuple(char.pos) == ctrl.defender_positions.targets[char.name]:
                break
        self.assertEqual(char.ramp_charges, 0)
        self.assertEqual(tuple(char.pos), ctrl.defender_positions.targets[char.name])

    def test_forbidden_setup_lamp_waits_until_live_round_instead_of_taking_guard(self):
        game, ctrl, char = self.fixture(candidates=((2, 22),))
        with patch('fnatic_v1_rules.is_setup_position_allowed', lambda r, c: c < 22):
            result = ctrl.decide_move(char, self.state(game, setup=True))
        self.assertEqual(result[0], char.pos)
        self.assertIsNone(ctrl.engineer.target)
        self.assertIsNone(ctrl.engineer.pending)
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.engineer.target, (2, 22))

    def test_retries_rejected_cast_and_only_counts_actual_charge_consumption(self):
        game, ctrl, char = self.fixture()
        char.pos = [2, 22]
        ctrl.engineer.target = (2, 22)
        for _ in range(2):
            result = ctrl.decide_move(char, self.state(game))
            self.assertEqual(result, ([2, 22], {'ability': 'RAMP'}))
            self.assertEqual(ctrl.engineer.completed, set())
        self.assertTrue(game.execute_ai_ability(char, result[1]))
        ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.engineer.completed, {(2, 22)})
        self.assertNotEqual(ctrl.engineer.target, (2, 22))

    def test_one_candidate_finishes_without_repeating_and_returns_to_guard(self):
        game, ctrl, char = self.fixture(candidates=((2, 22),))
        char.pos = [2, 22]
        game.move_character(char)
        result = ctrl.decide_move(char, self.state(game))
        self.assertEqual(char.ramp_charges, 1)
        self.assertNotEqual(result[1], {'ability': 'RAMP'})
        goal = ctrl.defender_positions.targets[char.name]
        lengths = distances(goal, game.grid)
        self.assertLess(lengths[tuple(result[0])], lengths[tuple(char.pos)])

    def test_occupied_candidate_is_retained_without_spending_a_charge(self):
        game, ctrl, char = self.fixture()
        char.pos = [2, 21]
        game.chars[1].pos = [2, 22]
        ctrl.engineer.target = (2, 22)
        result = ctrl.decide_move(char, self.state(game))
        self.assertEqual(result[0], char.pos)
        self.assertEqual(char.ramp_charges, 2)
        self.assertEqual(ctrl.engineer.target, (2, 22))
        game.chars[1].pos = [2, 18]
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], [2, 22])

    def test_contact_and_urgent_retake_take_priority_over_remaining_lamps(self):
        game, ctrl, char = self.fixture()
        game.chars.append(make_character('Enemy', 'A', (2, 18)))
        self.assertEqual(ctrl.decide_move(char, self.state(game))[0], char.pos)
        self.assertEqual(char.ramp_charges, 2)
        game.chars.pop()
        ctrl.retake = type(ctrl.retake)(retake_map(game.grid, ((10, 3), (12, 3), (13, 3)), ()))
        game.is_planted = True
        game.planted_pos = (9, 3)
        game.detonate_timer = 19
        result = ctrl.decide_move(char, self.state(game))
        self.assertFalse(ctrl.retake.preparing)
        self.assertNotEqual(result[0], char.pos)
        self.assertTrue(ctrl.retake.launched)
        self.assertEqual(char.ramp_charges, 2)
        self.assertIsNone(ctrl.engineer.target)

    def test_guard_on_a_lamp_candidate_moves_away_before_engineer_placement(self):
        game, ctrl, char = self.fixture(candidates=((2, 22), (2, 24)))
        ctrl.defender_positions = type(ctrl.defender_positions)(
            placement_map(game.grid, {3: ((2, 17), (2, 18), (2, 22))}))
        ctrl.defender_positions.targets = {'Alfajer': (2, 17), 'Leo': (2, 22)}
        other = game.chars[1]
        other.pos = [2, 22]
        ctrl.engineer.target = (2, 22)
        game.move_character(other)
        self.assertEqual(ctrl.defender_positions.targets[other.name], (2, 18))
        self.assertNotEqual(other.pos, [2, 22])
        for _ in range(50):
            for c in (other, char):
                game.move_character(c)
            game.battle_tick += 1
            if char.ramp_charges == 0 and tuple(char.pos) == ctrl.defender_positions.targets[char.name]:
                break
        self.assertEqual(char.ramp_charges, 0)
        self.assertEqual(len(game.ramp_traps), 2)
        self.assertEqual(tuple(char.pos), ctrl.defender_positions.targets[char.name])
        self.assertEqual(ctrl.defender_positions.reserved, set())

    def test_blank_map_is_normal_guard_and_round_reset_clears_placement_state(self):
        game, ctrl, char = self.fixture()
        ctrl.engineer.target = (2, 22)
        ctrl.engineer.completed.add((2, 24))
        ctrl.reset_round()
        self.assertEqual(ctrl.engineer.completed, set())
        self.assertIsNone(ctrl.engineer.target)
        self.assertIsNone(ctrl.engineer.pending)
        ctrl.engineer = type(ctrl.engineer)('')
        result = ctrl.decide_move(char, self.state(game))
        self.assertIn(ctrl.defender_positions.targets.get(char.name), ((2, 17), (2, 18)))
        self.assertNotEqual(result[1], {'ability': 'RAMP'})

    def test_settled_guard_clears_a_single_lane_for_lamp_mission(self):
        game = UltimateTestGame(7, 12)
        game.grid[:] = 1
        game.grid[4, 1:11] = 0
        game.grid[4, 1] = 2
        char = make_character('Alfajer', 'D', (4, 3))
        other = make_character('Leo', 'D', (4, 4))
        game.chars = [char, other]
        ctrl = FnaticV3DefenderController(
            placement_map(game.grid, {3: ((4, 2), (4, 4), (4, 10))}), retake_map='',
            engineer_map=lamp_map(game.grid, ((4, 6), (4, 8))))
        ctrl.defender_positions.targets = {'Alfajer': (4, 2), 'Leo': (4, 4)}
        ctrl.engineer.target = (4, 8)
        ctrl.set_game(game)
        game.defender_controller = ctrl
        for _ in range(40):
            for c in (other, char):
                game.move_character(c)
            game.battle_tick += 1
            if char.ramp_charges == 0 and tuple(char.pos) == (4, 2):
                break
        self.assertEqual(char.ramp_charges, 0)
        self.assertEqual(len(game.ramp_traps), 2)
        self.assertEqual(tuple(char.pos), (4, 2))
        self.assertIn(tuple(other.pos), ((4, 4), (4, 10)))
        self.assertNotEqual(tuple(other.pos), tuple(char.pos))

    def test_default_uses_defender_map_and_bad_shape_or_walls_are_rejected(self):
        game, _, char = self.fixture()
        ctrl = FnaticV3DefenderController()
        mapped = parse_grid(ENGINEER_LAMP_STR)
        expected = set(tuple(map(int, p)) for p in zip(*np.where(mapped == 3)))
        self.assertEqual(set(ctrl.engineer.candidates), expected)
        ctrl.set_game(game)
        ctrl.decide_move(char, self.state(game))
        self.assertIn(ctrl.engineer.target, expected)
        for text in ('030\n000', lamp_map(game.grid, ((0, 0),))):
            bad = FnaticV3DefenderController('', '', engineer_map=text)
            with self.assertRaises(ValueError):
                bad.decide_move(char, self.state(game))


if __name__ == '__main__':
    unittest.main()
