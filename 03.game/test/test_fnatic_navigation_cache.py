"""Navigation reuse must preserve paths when the match's obstacles change."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import fnatic_v1_rules
import map_data_defender_setup
from fnatic_v1_rules import FnaticRulesController
from fnatic_v3.positions import _cached_distances, _walking_distances, distances


class DistanceCacheTests(unittest.TestCase):
    def setUp(self):
        _cached_distances.cache_clear()
        self.addCleanup(_cached_distances.cache_clear)

    def test_matches_original_bfs_with_walls_and_moving_occupants(self):
        rng = np.random.default_rng(47)
        for _ in range(30):
            grid = (rng.random((7, 9)) < .25).astype(np.int32)
            start = (3, 4)
            grid[start] = 0
            blocked = {tuple(map(int, cell)) for cell in rng.integers((0, 0), (7, 9), (4, 2))}
            expected = _walking_distances(start, grid, blocked)
            self.assertEqual(distances(start, grid, blocked), expected)
            self.assertEqual(distances(start, grid, reversed(sorted(blocked))), expected)
        self.assertEqual(_cached_distances.cache_info().hits, 30)

    def test_mutating_return_value_cannot_corrupt_cache(self):
        grid = np.zeros((3, 5), dtype=int)
        first = distances((1, 0), grid)
        first[(1, 4)] = -100
        first.clear()
        self.assertEqual(distances((1, 0), grid)[(1, 4)], 4)
        self.assertEqual(_cached_distances.cache_info().hits, 1)

    def test_wall_occupancy_and_shape_changes_cannot_reuse_stale_distances(self):
        grid = np.zeros((1, 5), dtype=int)
        self.assertEqual(distances((0, 0), grid)[(0, 4)], 4)
        grid[0, 2] = 1
        self.assertNotIn((0, 4), distances((0, 0), grid))
        grid[0, 2] = 2  # Plant markers do not change walking distances.
        self.assertEqual(distances((0, 0), grid)[(0, 4)], 4)
        self.assertNotIn((0, 4), distances((0, 0), grid, {(0, 2)}))
        self.assertEqual(distances((0, 0), grid.reshape(5, 1))[(4, 0)], 4)
        self.assertEqual(_cached_distances.cache_info().hits, 1)

    def test_cache_is_bounded_across_many_map_layouts(self):
        for layout in range(1030):
            grid = np.zeros((4, 4), dtype=int)
            for bit in range(11):
                grid.flat[bit + 1] = (layout >> bit) & 1
            distances((0, 0), grid)
        self.assertEqual(_cached_distances.cache_info().currsize, 1024)


class RouteCacheTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = FnaticRulesController('A')

    def test_matches_original_tie_breaking_multigoal_and_unreachable_paths(self):
        rng = np.random.default_rng(18)
        for _ in range(40):
            grid = (rng.random((7, 9)) < .3).astype(int)
            start = (3, 4)
            grid[start] = 0
            goals = tuple(tuple(map(int, cell)) for cell in rng.integers((0, 0), (7, 9), (3, 2)))
            blocked = {tuple(map(int, cell)) for cell in rng.integers((0, 0), (7, 9), (4, 2))}
            expected = self.ctrl._route_uncached(start, goals, grid, blocked)
            self.assertEqual(self.ctrl._route(start, iter(goals), grid, iter(blocked)), expected)
            with patch.object(self.ctrl, '_route_uncached', side_effect=AssertionError('Repeated path was recomputed')):
                self.assertEqual(self.ctrl._route(start, reversed(goals), grid, blocked | {start}), expected)

    def test_changed_walls_occupants_and_goals_are_recomputed(self):
        grid = np.zeros((3, 5), dtype=int)
        start, goal = (1, 0), (1, 4)
        self.assertEqual(self.ctrl._route(start, (goal,), grid)[0], (1, 1))
        for blocked, goals in (({(1, 1)}, (goal,)), ((), ((0, 0),)), ((), ())):
            expected = self.ctrl._route_uncached(start, goals, grid, blocked)
            self.assertEqual(self.ctrl._route(start, goals, grid, blocked), expected)
        grid[1, 1] = 1
        self.assertNotEqual(self.ctrl._route(start, (goal,), grid)[0], (1, 1))
        grid[1, 1] = 0
        self.assertEqual(self.ctrl._route(start, (goal,), grid)[0], (1, 1))

    def test_setup_mask_changes_and_custom_setup_rules_are_respected(self):
        grid = np.zeros((3, 4), dtype=int)
        start, goals = (1, 0), ((1, 3),)
        with patch.object(map_data_defender_setup, 'DEFENDER_SETUP_MASK_STR', '0000\n0000\n0000'):
            self.assertEqual(self.ctrl._route(start, goals, grid, setup=True)[1], 3)
        with patch.object(map_data_defender_setup, 'DEFENDER_SETUP_MASK_STR', '1111\n0100\n1111'):
            self.assertEqual(self.ctrl._route(start, goals, grid, setup=True)[1], float('inf'))
            self.assertEqual(self.ctrl._route(start, goals, grid, setup=False)[1], 3)
        allowed = {'value': True}
        with patch.object(fnatic_v1_rules, 'is_setup_position_allowed', lambda r, c: allowed['value']):
            self.assertEqual(self.ctrl._route(start, goals, grid, setup=True)[1], 3)
            allowed['value'] = False
            self.assertEqual(self.ctrl._route(start, goals, grid, setup=True)[1], float('inf'))

    def test_risk_routes_recalculate_exposure_when_smoke_changes(self):
        grid = np.zeros((3, 5), dtype=int)
        game = SimpleNamespace(smoked=False)
        game.check_cell_line_of_sight = lambda a, b, block_smoke=True: a[0] == 1 and not game.smoked
        self.ctrl.set_game(game)
        args = ((1, 0), ((1, 4),), grid)
        exposed = self.ctrl._route(*args, risks=((1, 4),))
        game.smoked = True
        covered = self.ctrl._route(*args, risks=((1, 4),))
        self.assertNotEqual(exposed, covered)
        self.assertEqual(covered, self.ctrl._route_uncached(*args, risks=((1, 4),)))
        self.assertFalse(self.ctrl._route_cache)

    def test_cache_evicts_old_entries_and_reuses_paths_across_rounds(self):
        grid = np.zeros((3, 5), dtype=int)
        start = (1, 0)
        with patch.object(fnatic_v1_rules, '_ROUTE_CACHE_SIZE', 2), \
                patch.object(self.ctrl, '_route_uncached', wraps=self.ctrl._route_uncached) as search:
            for goal in ((1, 1), (1, 2), (1, 3)):
                self.ctrl._route(start, (goal,), grid)
            self.assertEqual(len(self.ctrl._route_cache), 2)
            self.ctrl.reset_round()
            self.ctrl._route(start, ((1, 3),), grid)
            self.assertEqual(search.call_count, 3)
            self.ctrl._route(start, ((1, 1),), grid)
            self.assertEqual(search.call_count, 4)


if __name__ == '__main__':
    unittest.main()
