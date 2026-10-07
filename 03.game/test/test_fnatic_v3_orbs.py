"""Fnatic orb priorities through real collection and shared IQ movement."""

import unittest

import numpy as np

from fnatic_v3.controller import FnaticV3AttackerController, FnaticV3DefenderController
from fnatic_v3.orb_collection import ORB_PRIORITY
from fnatic_v3.positions import TacticalPositions
from game_core import ORB_COLLECT_REQUIRED_TICKS, ORB_ULTIMATE_POINTS
from iq_controller_adapter import IQAwareController
from test_fnatic_v3_retake import retake_map
from test_ultimate_system import UltimateTestGame, make_character


class FnaticOrbTests(unittest.TestCase):
    def fixture(self, side='D', use_iq=False):
        game = UltimateTestGame(12, 24)
        game.grid[1, 1] = game.grid[1, 22] = 2
        positions = ((3, 10), (4, 9), (4, 11), (6, 9), (6, 11))
        game.chars = [make_character(name, side, cell) for name, cell in zip(ORB_PRIORITY, positions)]
        for char in game.chars:
            char.ability_name = 'NONE'
            char.ultimate_name = 'NONE'
        marked = np.zeros_like(game.grid)
        marked[1, 1] = 5
        marked[1, 22] = 6
        text = '\n'.join(''.join(map(str, row)) for row in marked)
        ctrl = (FnaticV3AttackerController(TacticalPositions(text, text), engineer_map='')
                if side == 'A' else FnaticV3DefenderController('', '', engineer_map=''))
        if side == 'A':
            game.chars[2].has_spike = True
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        if side == 'A':
            game.attacker_controller = adapter
        else:
            game.defender_controller = adapter
        game.available_orbs = {(5, 10)}
        return game, ctrl

    @staticmethod
    def state(game):
        return dict(grid=game.grid, chars=game.chars, available_orbs=game.available_orbs,
                    is_planted=game.is_planted, planted_pos=game.planted_pos,
                    round_timer=100, detonate_timer=game.detonate_timer)

    def test_all_priorities_on_both_sides_do_not_choose_closest_over_priority(self):
        for side in ('A', 'D'):
            for index, expected in enumerate(ORB_PRIORITY):
                with self.subTest(side=side, expected=expected):
                    game, ctrl = self.fixture(side)
                    for char in game.chars[:index]:
                        char.ultimate_points = char.ultimate_cost
                    game.chars[-1].pos = [5, 11]  # Derke is nearer than the higher priorities.
                    ctrl.decide_move(game.chars[-1], self.state(game))
                    self.assertEqual(ctrl.orbs.targets, {expected: (5, 10)})

    def test_dead_full_and_more_than_five_steps_away_are_ignored(self):
        game, ctrl = self.fixture()
        game.chars[0].is_alive = False
        game.chars[1].ultimate_points = game.chars[1].ultimate_cost
        game.chars[2].pos = [5, 16]
        ctrl.decide_move(game.chars[-1], self.state(game))
        self.assertEqual(ctrl.orbs.targets, {'Boaster': (5, 10)})

    def test_five_step_boundary_includes_wall_detours(self):
        for col, wall, expected in ((15, False, True), (16, False, False), (15, True, False)):
            with self.subTest(col=col, wall=wall):
                game, ctrl = self.fixture()
                game.chars = [game.chars[-1]]
                char = game.chars[0]
                char.pos = [5, 10]
                game.available_orbs = {(5, col)}
                if wall:
                    game.grid[5, 12] = 1
                ctrl.decide_move(char, self.state(game))
                self.assertEqual(char.name in ctrl.orbs.targets, expected)

    def test_both_sides_reach_collect_gain_points_and_return_to_normal_with_iq(self):
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, ctrl = self.fixture(side, use_iq)
                    collector = game.chars[0]
                    collector.ultimate_points = collector.ultimate_cost - 1
                    before = collector.ultimate_points
                    for _ in range(12):
                        game.move_character(collector)
                        game.battle_tick += 1
                        if not game.available_orbs:
                            break
                    self.assertEqual(game.available_orbs, set())
                    self.assertEqual(collector.ultimate_points,
                                     min(collector.ultimate_cost, before + ORB_ULTIMATE_POINTS))
                    self.assertEqual(collector.orb_collect_timer, 0)
                    result = ctrl.decide_move(collector, self.state(game))
                    self.assertNotEqual(result[1] if len(result) > 1 else None, 'COLLECT_ORB')
                    self.assertEqual(ctrl.orbs.targets, {})

    def test_collection_is_repeated_until_the_real_game_completes_it(self):
        game, ctrl = self.fixture()
        char = game.chars[0]
        char.pos = [5, 10]
        for tick in range(ORB_COLLECT_REQUIRED_TICKS):
            game.move_character(char)
            if tick < ORB_COLLECT_REQUIRED_TICKS - 1:
                self.assertEqual(char.orb_collect_timer, tick + 1)
                self.assertIn((5, 10), game.available_orbs)
        self.assertEqual(char.ultimate_points, ORB_ULTIMATE_POINTS)
        self.assertEqual(game.available_orbs, set())

    def test_multiple_orbs_receive_distinct_collectors_and_stable_assignments(self):
        game, ctrl = self.fixture()
        game.available_orbs.add((5, 8))
        for char in reversed(game.chars):
            ctrl.decide_move(char, self.state(game))
        targets = dict(ctrl.orbs.targets)
        self.assertEqual(set(targets), {'Alfajer', 'Chronicle'})
        self.assertEqual(set(targets.values()), game.available_orbs)
        for char in game.chars:
            ctrl.decide_move(char, self.state(game))
        self.assertEqual(ctrl.orbs.targets, targets)

    def test_collector_death_or_charging_ultimate_reassigns_remaining_orb(self):
        for full in (False, True):
            game, ctrl = self.fixture()
            char = game.chars[0]
            ctrl.decide_move(char, self.state(game))
            if full:
                char.ultimate_points = char.ultimate_cost
            else:
                char.is_alive = False
            ctrl.decide_move(game.chars[1], self.state(game))
            self.assertEqual(ctrl.orbs.targets, {'Chronicle': (5, 10)})
            ctrl.reset_round()
            self.assertEqual(ctrl.orbs.targets, {})

    def test_combat_interrupts_collection_and_can_resume_when_enemy_is_gone(self):
        game, ctrl = self.fixture()
        char = game.chars[0]
        char.pos = [5, 10]
        game.move_character(char)
        self.assertEqual(char.orb_collect_timer, 1)
        enemy = make_character('Enemy', 'A', (5, 12))
        game.chars.append(enemy)
        game.move_character(char)
        self.assertEqual(char.orb_collect_timer, 0)
        self.assertIn((5, 10), game.available_orbs)
        game.chars.remove(enemy)
        game.move_character(char)
        self.assertEqual(char.orb_collect_timer, 1)

    def test_retakes_and_urgent_group_launch_override_nearby_orbs(self):
        for remaining in (55, 19):
            game, _ = self.fixture()
            ctrl = FnaticV3DefenderController('', retake_map(game.grid, ((2, 5), (2, 6), (2, 7)), ()), engineer_map='')
            ctrl.set_game(game)
            game.defender_controller = ctrl
            game.is_planted = True
            game.planted_pos = (1, 1)
            game.detonate_timer = remaining
            char = game.chars[0]
            game.available_orbs = {tuple(char.pos)}
            game.move_character(char)
            self.assertFalse(char.is_collecting_orb)
            self.assertEqual(ctrl.retake.launched, remaining < 20)
            self.assertEqual(ctrl.orbs.targets, {})

    def test_planting_and_defusing_override_orb_collection(self):
        for side, action in (('A', 'PLANT'), ('D', 'DEFUSE')):
            game, ctrl = self.fixture(side)
            char = game.chars[2]
            char.pos = [1, 1]
            game.available_orbs = {(1, 1)}
            if side == 'A':
                ctrl.target = (1, 1)
            else:
                game.is_planted = True
                game.planted_pos = (1, 1)
            self.assertEqual(ctrl.decide_move(char, self.state(game))[1], action)
            self.assertEqual(ctrl.orbs.targets, {})

    def test_defuse_notification_response_overrides_orb_collection(self):
        game, ctrl = self.fixture('A')
        char = game.chars[0]
        char.pos = [3, 1]
        game.available_orbs = {tuple(char.pos)}
        game.is_planted = True
        game.planted_pos = (1, 1)
        state = self.state(game)
        state['defender_defuse_info'] = {'Enemy': (1, 6)}
        result = ctrl.decide_move(char, state)
        self.assertEqual(ctrl.defuse_responder, char.name)
        self.assertEqual(result[0], [2, 1])
        self.assertEqual(ctrl.orbs.targets, {})


if __name__ == '__main__':
    unittest.main()
