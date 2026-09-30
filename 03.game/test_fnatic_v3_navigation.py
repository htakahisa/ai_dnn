"""Friendly traffic must make progress without interrupting stationary actions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fnatic_v1_rules import FnaticRulesController, pos
from fnatic_v3.controller import FnaticV3FacingMixin
from iq_controller_adapter import IQAwareController
from test_ultimate_system import UltimateTestGame, make_character


class RouteController(FnaticV3FacingMixin, FnaticRulesController):
    """Exercise the shared navigation with explicit movement/hold intentions."""

    def __init__(self, side, goals, clamp=False):
        self.goals, self.clamp = goals, clamp
        self.actions = {}
        super().__init__(side)

    def _decide_tactics(self, char, state):
        goal = self.goals[char.name]
        blocked = {pos(c) for c in state['chars'] if c.is_alive and c.name != char.name}
        step, length, _ = self._route(pos(char), (goal,), state['grid'],
                                      setup=state.get('defender_setup_active', False))
        if step in blocked:
            step, detour_length, _ = self._route(pos(char), (goal,), state['grid'], blocked,
                                                 setup=state.get('defender_setup_active', False))
            if self.clamp and detour_length > length + 1:
                step = pos(char)
        action = self.actions.get(char.name)
        return (list(step), action) if action is not None else self._result(char, step)


class FnaticNavigationTests(unittest.TestCase):
    def fixture(self, cells=None, side='A', starts=((3, 1), (3, 3)),
                goals=((3, 8), (3, 3)), clamp=False, use_iq=False):
        game = UltimateTestGame(8, 12)
        if cells is not None:
            game.grid[:] = 1
            for cell in cells:
                game.grid[cell] = 0
        game.chars = [make_character(name, side, cell)
                      for name, cell in zip(('Derke', 'Leo'), starts)]
        ctrl = RouteController(side, dict(zip(('Derke', 'Leo'), goals)), clamp)
        adapter = IQAwareController(ctrl) if use_iq else ctrl
        adapter.set_game(game)
        if side == 'A':
            game.attacker_controller = adapter
        else:
            game.defender_controller = adapter
        return game, ctrl

    @staticmethod
    def state(game, setup=False):
        return dict(grid=game.grid, chars=game.chars, defender_setup_active=setup)

    def advance(self, game, ticks=30):
        for _ in range(ticks):
            for char in game.chars:
                game.move_character(char)
                self.assertEqual(len({pos(c) for c in game.chars}), len(game.chars))
            game.battle_tick += 1

    def test_detour_on_both_sides_even_when_old_leash_rejects_it(self):
        for side in ('A', 'D'):
            for clamp in (False, True):
                with self.subTest(side=side, clamp=clamp):
                    game, _ = self.fixture(side=side, clamp=clamp)
                    self.advance(game)
                    self.assertEqual(pos(game.chars[0]), (3, 8))

    def test_parked_guard_yields_into_side_pocket_and_requester_passes(self):
        cells = [(3, c) for c in range(1, 10)] + [(2, 3)]
        for side in ('A', 'D'):
            for use_iq in (False, True):
                with self.subTest(side=side, use_iq=use_iq):
                    game, _ = self.fixture(cells, side=side, use_iq=use_iq)
                    self.advance(game)
                    self.assertEqual(pos(game.chars[0]), (3, 8))
                    self.assertEqual(pos(game.chars[1]), (3, 3))

    def test_distant_blocker_does_not_hold_up_the_open_part_of_lane(self):
        cells = [(3, c) for c in range(1, 10)] + [(2, 6)]
        game, _ = self.fixture(cells, starts=((3, 1), (3, 6)), goals=((3, 8), (3, 6)))
        game.move_character(game.chars[0])
        self.assertEqual(pos(game.chars[0]), (3, 2))
        self.advance(game)
        self.assertEqual(pos(game.chars[0]), (3, 8))

    def test_no_side_pocket_blocker_moves_forward_to_make_space(self):
        game, _ = self.fixture([(3, c) for c in range(1, 10)])
        self.advance(game, 10)
        self.assertGreater(pos(game.chars[0])[1], 3)

    def test_plant_goal_can_be_crossed_to_clear_a_narrow_entry(self):
        game, ctrl = self.fixture([(3, c) for c in range(1, 9)],
                                  starts=((3, 4), (3, 5)), goals=((3, 6), (3, 5)))
        ctrl.target = (3, 6)
        for _ in range(10):
            self.advance(game, 1)
            if pos(game.chars[0]) == ctrl.target:
                break
        self.assertEqual(pos(game.chars[0]), ctrl.target)
        self.assertNotEqual(pos(game.chars[1]), ctrl.target)

    def test_head_on_players_pass_in_alcove(self):
        cells = [(3, c) for c in range(1, 10)] + [(2, 5)]
        game, _ = self.fixture(cells, starts=((3, 2), (3, 8)), goals=((3, 8), (3, 2)))
        self.advance(game, 50)
        self.assertEqual([pos(c) for c in game.chars], [(3, 8), (3, 2)])

    def test_channelled_actions_and_ability_payloads_are_preserved(self):
        for action in ('PLANT', 'DEFUSE', 'COLLECT_ORB', {'ability': 'RECON'}, {'ultimate': 'MONITOR'}):
            with self.subTest(action=action):
                game, ctrl = self.fixture(starts=((3, 2), (3, 3)), clamp=True)
                actor = game.chars[0]
                ctrl.actions[actor.name] = action
                result = ctrl.decide_move(actor, self.state(game))
                self.assertEqual(result, ([3, 2], action))

    def test_combat_hold_does_not_yield_to_a_teammate(self):
        game, ctrl = self.fixture()
        mover, guard = game.chars
        ctrl.decide_move(mover, self.state(game))
        game.chars.append(make_character('Enemy', 'D', (3, 6)))
        result = ctrl.decide_move(guard, self.state(game))
        self.assertEqual(tuple(result[0]), pos(guard))

    def test_setup_detour_uses_only_permitted_cells(self):
        game, ctrl = self.fixture(side='D', starts=((3, 2), (3, 3)), clamp=True)
        state = self.state(game, setup=True)
        allowed = lambda r, c: r >= 3
        with patch('fnatic_v1_rules.is_setup_position_allowed', allowed), \
                patch('fnatic_v3.navigation.is_setup_position_allowed', allowed):
            step, _ = ctrl.decide_move(game.chars[0], state)
        self.assertGreaterEqual(step[0], 3)
        self.assertNotEqual(tuple(step), pos(game.chars[0]))
        self.assertNotEqual(tuple(step), pos(game.chars[1]))

    def test_intentional_carrier_wait_is_preserved_and_reset_clears_traffic(self):
        game, ctrl = self.fixture(starts=((3, 2), (3, 3)), clamp=True)
        ctrl.formation = SimpleNamespace(navigation_waiting={'Derke'}, passage_waits={})
        result = ctrl.decide_move(game.chars[0], self.state(game))
        self.assertEqual(tuple(result[0]), pos(game.chars[0]))
        self.assertTrue(ctrl.navigation.yields)
        ctrl.reset_round()
        self.assertFalse(ctrl.navigation.yields)


if __name__ == '__main__':
    unittest.main()
