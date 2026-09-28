"""Fnatic engineer placement, real charge consumption, and regrouping."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController
from fnatic_v3.positions import distances, parse_grid
from iq_controller_adapter import IQAwareController
from map_data import NEW_MAZE_STR
from test_ultimate_system import UltimateTestGame, make_character


def lamp_map(grid, candidates):
    marked = np.where(grid == 1, 1, 0)
    for cell in candidates:
        marked[cell] = 3
    return '\n'.join(''.join(str(int(value)) for value in row) for row in marked)


class FnaticEngineerTests(unittest.TestCase):
    def fixture(self, candidates=((22, 21), (21, 22), (22, 25)), use_iq=False):
        grid = parse_grid(NEW_MAZE_STR)
        game = UltimateTestGame(*grid.shape)
        game.grid = grid
        names = ('Leo', 'Derke', 'Chronicle', 'Boaster', 'Alfajer')
        game.chars = [make_character(name, 'A', (23, 17 + i)) for i, name in enumerate(names)]
        game.chars[0].has_spike = True
        ctrl = FnaticV3AttackerController(engineer_map=lamp_map(grid, candidates))
        controller = IQAwareController(ctrl) if use_iq else ctrl
        controller.set_game(game)
        game.attacker_controller = controller
        return game, ctrl, game.chars[-1]

    def state(self, game):
        return dict(grid=game.grid, chars=game.chars, round_timer=100,
                    is_planted=game.is_planted, planted_pos=game.planted_pos)

    def test_two_distinct_candidate_placements_then_join_main_and_guards(self):
        for use_iq in (False, True):
            with self.subTest(use_iq=use_iq):
                game, ctrl, engineer = self.fixture(use_iq=use_iq)
                placed = []
                joined_main = False
                for _ in range(200):
                    for char in game._move_order():
                        before = engineer.ramp_charges
                        game.move_character(char)
                        if engineer.ramp_charges < before:
                            placed.append(tuple(engineer.pos))
                    game.battle_tick += 1
                    # A fast entry can plant before the detached engineer
                    # finishes. He must rejoin the main team/guards either way.
                    if engineer.ramp_charges == 0:
                        joined_main |= (engineer.name not in ctrl.formation.detached_names
                                        and ctrl.formation.mid_name == 'Boaster')
                    if game.is_planted and all(
                            ctrl.guard_targets.get(c.name) == tuple(c.pos) for c in game.chars):
                        break
                self.assertEqual(len(placed), 2)
                self.assertEqual(len(set(placed)), 2)
                self.assertTrue(set(placed).issubset(ctrl.engineer.candidates))
                self.assertEqual(len({trap['pos'] for trap in game.ramp_traps}), 2)
                self.assertEqual(engineer.ramp_charges, 0)
                self.assertTrue(joined_main)
                self.assertTrue(game.is_planted)
                self.assertEqual(ctrl.guard_targets.get(engineer.name), tuple(engineer.pos))

    def test_cast_is_retried_until_game_spends_a_charge(self):
        game, ctrl, engineer = self.fixture()
        engineer.pos = [22, 21]
        ctrl.engineer.target = (22, 21)
        for _ in range(2):
            result = ctrl.decide_move(engineer, self.state(game))
            self.assertEqual(result, ([22, 21], {'ability': 'RAMP'}))
            self.assertEqual(ctrl.engineer.completed, set())
            self.assertEqual(engineer.ramp_charges, 2)
        self.assertTrue(game.execute_ai_ability(engineer, result[1]))
        result = ctrl.decide_move(engineer, self.state(game))
        self.assertEqual(ctrl.engineer.completed, {(22, 21)})
        self.assertIn(ctrl.engineer.target, {(21, 22), (22, 25)})
        self.assertNotEqual(result[0], engineer.pos)

    def test_existing_team_trap_and_completed_cell_are_not_reused(self):
        game, ctrl, engineer = self.fixture()
        engineer.pos = [22, 21]
        game.execute_ai_ability(engineer, {'ability': 'RAMP'})
        ctrl.decide_move(engineer, self.state(game))
        self.assertIn(ctrl.engineer.target, {(21, 22), (22, 25)})
        selected = ctrl.engineer.target
        ctrl.engineer.completed.add((22, 21))
        game.ramp_traps = []  # Placement history survives removal of a trap.
        ctrl.decide_move(engineer, self.state(game))
        self.assertEqual(ctrl.engineer.target, selected)

    def test_occupied_candidate_waits_without_spending_or_abandoning(self):
        game, ctrl, engineer = self.fixture(candidates=((22, 21),))
        engineer.pos = [22, 20]
        carrier = game.chars[0]
        carrier.pos = [22, 21]
        result = ctrl.decide_move(engineer, self.state(game))
        self.assertEqual(result[0], engineer.pos)
        self.assertEqual(engineer.ramp_charges, 2)
        self.assertEqual(ctrl.engineer.target, (22, 21))
        self.assertIn('Alfajer', ctrl.formation.detached_names)
        carrier.pos = [23, 17]
        self.assertEqual(ctrl.decide_move(engineer, self.state(game))[0], [22, 21])

    def test_fewer_candidates_than_charges_finishes_without_duplicate(self):
        game, ctrl, engineer = self.fixture(candidates=((22, 21),))
        engineer.pos = [22, 21]
        game.move_character(engineer)
        result = ctrl.decide_move(engineer, self.state(game))
        self.assertEqual(engineer.ramp_charges, 1)
        self.assertNotEqual(result[1], {'ability': 'RAMP'})
        self.assertNotIn('Alfajer', ctrl.formation.detached_names)
        self.assertEqual(ctrl.formation.mid_name, 'Boaster')

    def test_empty_map_retains_existing_mid_rule(self):
        game, _, engineer = self.fixture()
        ctrl = FnaticV3AttackerController(engineer_map='')
        ctrl.set_game(game)
        result = ctrl.decide_move(engineer, self.state(game))
        self.assertEqual(ctrl.formation.mid_name, 'Alfajer')
        self.assertNotEqual(result[1], {'ability': 'RAMP'})

    def test_finished_engineer_joins_at_twelve_steps_but_goes_mid_at_thirteen(self):
        for col, join in ((28, True), (29, False)):
            with self.subTest(col=col, join=join):
                game, ctrl, engineer = self.fixture()
                engineer.ramp_charges = 0
                engineer.pos = [22, col]
                before = distances(tuple(game.chars[0].pos), game.grid)[tuple(engineer.pos)]
                self.assertEqual(before, 12 if join else 13)
                result = ctrl.decide_move(engineer, self.state(game))
                self.assertEqual(engineer.name not in ctrl.formation.detached_names, join)
                self.assertEqual(ctrl.formation.mid_name, 'Boaster')
                if join:
                    self.assertIsNone(ctrl.engineer.mid_target)
                    self.assertLess(distances(tuple(game.chars[0].pos), game.grid)[tuple(result[0])], before)
                else:
                    self.assertIsNotNone(ctrl.engineer.mid_target)

    def test_finished_engineer_uses_wall_aware_distance_despite_short_grid_distance(self):
        game, ctrl, engineer = self.fixture(use_iq=True)
        carrier = game.chars[0]
        carrier.pos = [13, 6]
        engineer.pos = [13, 16]
        engineer.ramp_charges = 0
        self.assertEqual(distances(tuple(carrier.pos), game.grid)[tuple(engineer.pos)], 18)
        game.move_character(engineer)
        self.assertIn(engineer.name, ctrl.formation.detached_names)
        self.assertEqual(ctrl.engineer.mid_target, (13, 16))

    def test_far_engineer_does_not_hold_carrier_and_rejoins_when_close(self):
        game, ctrl, engineer = self.fixture()
        carrier, derke, chronicle, boaster, _ = game.chars
        carrier.pos = [13, 3]
        carrier.recon_charges = 0
        derke.pos = [10, 3]
        chronicle.pos = [11, 3]
        boaster.pos = [13, 23]
        engineer.pos = [22, 38]
        engineer.ramp_charges = 0
        ctrl.target = (9, 3)
        result = ctrl.decide_move(carrier, self.state(game))
        self.assertIn(engineer.name, ctrl.formation.detached_names)
        self.assertNotEqual(result[0], carrier.pos)
        ctrl.decide_move(engineer, self.state(game))
        self.assertIsNotNone(ctrl.engineer.mid_target)
        engineer.pos = [13, 6]
        ctrl.decide_move(carrier, self.state(game))
        self.assertNotIn(engineer.name, ctrl.formation.detached_names)
        self.assertIsNone(ctrl.engineer.mid_target)

    def test_finished_engineer_and_regular_mid_player_hold_different_mid_slots(self):
        game, ctrl, engineer = self.fixture()
        game.chars[0].pos = [13, 3]
        engineer.ramp_charges = 0
        engineer.pos = [14, 23]
        boaster = next(c for c in game.chars if c.name == 'Boaster')
        boaster.pos = [13, 23]
        for _ in range(10):
            for char in (engineer, boaster):
                game.move_character(char)
            game.battle_tick += 1
        self.assertEqual(tuple(engineer.pos), ctrl.engineer.mid_target)
        self.assertEqual(tuple(boaster.pos), ctrl.formation.mid_target)
        self.assertNotEqual(ctrl.engineer.mid_target, ctrl.formation.mid_target)
        self.assertIn(engineer.name, ctrl.formation.detached_names)

    def test_postplant_finished_far_engineer_returns_to_guard(self):
        game, ctrl, engineer = self.fixture()
        engineer.ramp_charges = 0
        engineer.pos = [22, 38]
        game.chars[0].pos = [13, 3]
        ctrl.decide_move(engineer, self.state(game))
        self.assertIn(engineer.name, ctrl.formation.detached_names)
        game.is_planted = True
        game.planted_pos = (9, 3)
        game.chars[0].has_spike = False
        ctrl.decide_move(engineer, self.state(game))
        self.assertNotIn(engineer.name, ctrl.formation.detached_names)
        self.assertIsNone(ctrl.engineer.mid_target)
        self.assertIn(engineer.name, ctrl.guard_targets)

    def test_finished_lone_engineer_recovers_dropped_spike_instead_of_holding_mid(self):
        game, ctrl, engineer = self.fixture()
        for char in game.chars[:-1]:
            char.has_spike = False
            char.is_alive = False
        engineer.ramp_charges = 0
        engineer.pos = [22, 20]
        state = self.state(game)
        state['spike_pos'] = (22, 21)
        result = ctrl.decide_move(engineer, state)
        self.assertEqual(result[0], [22, 21])
        self.assertEqual(ctrl.retriever, engineer.name)
        self.assertNotIn(engineer.name, ctrl.formation.detached_names)
        self.assertIsNone(ctrl.engineer.mid_target)

    def test_postplant_finishes_placements_before_guard_assignment(self):
        game, ctrl, engineer = self.fixture(candidates=((22, 21), (22, 22)))
        game.is_planted = True
        game.planted_pos = ctrl.positions.plants[5][0]
        game.chars[0].has_spike = False
        engineer.pos = [22, 21]
        game.move_character(engineer)
        ctrl.decide_move(game.chars[0], self.state(game))
        self.assertNotIn('Alfajer', ctrl.guard_targets)
        for _ in range(5):
            game.move_character(engineer)
            game.battle_tick += 1
        self.assertEqual(engineer.ramp_charges, 0)
        self.assertEqual(len(game.ramp_traps), 2)
        self.assertIn('Alfajer', ctrl.guard_targets)

    def test_round_reset_clears_placement_history(self):
        game, ctrl, engineer = self.fixture()
        engineer.pos = [22, 21]
        game.move_character(engineer)
        ctrl.decide_move(engineer, self.state(game))
        ctrl.reset_round()
        self.assertEqual(ctrl.engineer.completed, set())
        self.assertIsNone(ctrl.engineer.target)
        self.assertIsNone(ctrl.engineer.pending)

    def test_invalid_size_or_walls_are_rejected(self):
        game, _, engineer = self.fixture()
        maps = ('000\n030', lamp_map(game.grid, ((0, 0),)))
        for text in maps:
            with self.subTest(text=text[:10]):
                ctrl = FnaticV3AttackerController(engineer_map=text)
                with self.assertRaises(ValueError):
                    ctrl.decide_move(engineer, self.state(game))


if __name__ == '__main__':
    unittest.main()
