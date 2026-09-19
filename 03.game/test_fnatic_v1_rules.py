import unittest
from types import SimpleNamespace
import numpy as np
from fnatic_v1_rules import FnaticV1AttackerController, FnaticV1DefenderController


def character(name, team, position, **extra):
    return SimpleNamespace(name=name, team=team, pos=list(position),
                           is_alive=True, facing='N', ability_name='NONE', **extra)


class FnaticRulesTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((9, 12), dtype=int)
        self.grid[2, 2] = self.grid[2, 10] = 2

    def state(self, chars, **extra):
        return dict(grid=self.grid, chars=chars, round_timer=100, **extra)

    def test_carrier_plants_immediately(self):
        c = character('carrier', 'A', (2, 2), has_spike=True)
        result = FnaticV1AttackerController().decide_move(c, self.state([c]))
        self.assertEqual(result, ([2, 2], 'PLANT'))

    def test_deadline_ignores_remote_target(self):
        c = character('carrier', 'A', (3, 2), has_spike=True)
        ctrl = FnaticV1AttackerController()
        ctrl.target = (2, 10)
        state = self.state([c])
        state['round_timer'] = 8
        self.assertEqual(ctrl.decide_move(c, state)[0], [2, 2])

    def test_dropped_spike_retrieved_directly(self):
        c = character('one', 'A', (3, 2), has_spike=False)
        result = FnaticV1AttackerController().decide_move(c, self.state([c], spike_pos=(2, 2)))
        self.assertEqual(result[0], [2, 2])

    def test_adjacent_defuse(self):
        c = character('one', 'D', (3, 2))
        result = FnaticV1DefenderController().decide_move(c, self.state([c], is_planted=True, planted_pos=(2, 2)))
        self.assertEqual(result, ([3, 2], 'DEFUSE'))

    def test_defender_stops_when_fighting(self):
        c = character('one', 'D', (3, 2))
        enemy = character('enemy', 'A', (3, 6), has_spike=True)
        self.assertEqual(FnaticV1DefenderController().decide_move(c, self.state([c, enemy]))[0], [3, 2])

    def test_blocked_route_does_not_random_walk(self):
        ctrl = FnaticV1AttackerController()
        self.grid[4, :] = 1
        self.assertEqual(ctrl._route((5, 2), [(2, 2)], self.grid)[0], (5, 2))

    def test_smoke_before_defuse(self):
        c = character('one', 'D', (3, 2), smoke_charges=1)
        c.ability_name = 'SMOKE'
        result = FnaticV1DefenderController().decide_move(c, self.state([c], is_planted=True, planted_pos=(2, 2)))
        self.assertEqual(result[1], {'ability': 'SMOKE', 'target': [2, 2]})


if __name__ == '__main__':
    unittest.main()
