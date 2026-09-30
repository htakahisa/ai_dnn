import random
import unittest

from concon_v1.co1_train_attacker_A1 import (
    GRID,
    ACTION_PLANT,
    LEFT_PLANT_CELLS,
    OBS_DIM,
    SPLIT_PATTERNS,
    WAYPOINT_POINTS,
    RouteProgress,
    build_observation,
    build_action_mask,
    choose_split_assignment,
    parse_game_grid,
    parse_strategy_points,
    select_nearest_candidate,
)


class ConconAttackerA1Tests(unittest.TestCase):
    def test_strategy_markers_are_parsed_separately_from_terrain(self):
        points = parse_strategy_points("abcd\n1234\n")
        self.assertEqual(points["a"], [(0, 0)])
        self.assertEqual(points["b"], [(0, 1)])
        self.assertEqual(points["c"], [(0, 2)])
        self.assertEqual(points["d"], [(0, 3)])
        with self.assertRaises(ValueError):
            parse_game_grid("10a\n")
        self.assertEqual(WAYPOINT_POINTS["a"], [(14, 21), (18, 9)])
        self.assertEqual(WAYPOINT_POINTS["d"], [(11, 1), (11, 6)])
        self.assertTrue(all(GRID[row, col] != 1 for cells in WAYPOINT_POINTS.values() for row, col in cells))

    def test_bfs_selects_a_reachable_left_plant_candidate(self):
        goal, index, distance = select_nearest_candidate(GRID, (11, 1), LEFT_PLANT_CELLS)
        self.assertEqual(goal, (9, 3))
        self.assertEqual(index, LEFT_PLANT_CELLS.index(goal))
        self.assertEqual(distance, 4)
        self.assertEqual(int(GRID[goal]), 2)
        self.assertLess(goal[1], GRID.shape[1] // 2)

    def test_split_patterns_guarantee_exact_group_sizes_and_a_goal(self):
        for pattern_index, expected_counts in enumerate(SPLIT_PATTERNS):
            selected, groups = choose_split_assignment(random.Random(7), 5, pattern_index)
            self.assertEqual(selected, pattern_index)
            self.assertEqual((groups.count(0), groups.count(1)), expected_counts)
            routes = [RouteProgress(group, selected, spawn) for group, spawn in zip(groups, [(23, 18), (23, 19), (23, 20), (23, 21), (23, 22)])]
            for group, route in zip(groups, routes):
                self.assertEqual(route.goal, WAYPOINT_POINTS["a"][group])

    def test_observation_encodes_split_group_and_target(self):
        route = RouteProgress(1, 2, (23, 22))
        observation = build_observation(route, (23, 22), True, [], 0, 0)
        self.assertEqual(observation.shape, (OBS_DIM,))
        self.assertEqual(observation[3], 1.0)
        self.assertEqual(observation[6], 1.0)
        self.assertEqual(observation[8], 1.0)
        self.assertEqual(observation[14], 1.0)
        self.assertEqual(observation[21], 1.0)

    def test_plant_is_only_available_on_the_selected_left_site_goal(self):
        goal = LEFT_PLANT_CELLS[0]
        other_site_cell = LEFT_PLANT_CELLS[1]
        mask = build_action_mask(GRID, goal, [], True, True, goal)
        self.assertTrue(mask[ACTION_PLANT])
        other_mask = build_action_mask(GRID, other_site_cell, [], True, True, goal)
        self.assertFalse(other_mask[ACTION_PLANT])


if __name__ == "__main__":
    unittest.main()