import random
import unittest

import numpy as np

from concon_v1.co1_train_attacker import (
    GRID,
    ACTION_PLANT,
    ACTION_WAIT,
    EPSILON_DECAY_RATIO,
    EPSILON_END,
    EPSILON_START,
    LEFT_PLANT_CELLS,
    MAX_CANDIDATE_BFS_DISTANCE,
    OBS_DIM,
    SPLIT_PATTERNS,
    WAYPOINT_POINTS,
    RouteProgress,
    build_observation,
    build_action_mask,
    bfs_distance_map,
    choose_split_assignment,
    epsilon_by_episode,
    parse_game_grid,
    parse_strategy_points,
    select_nearest_candidate,
)


class ConconAttackerA1Tests(unittest.TestCase):
    def test_epsilon_decays_from_configured_start_to_floor(self):
        self.assertEqual(EPSILON_START, 1.0)
        self.assertEqual(EPSILON_END, 0.05)
        self.assertEqual(EPSILON_DECAY_RATIO, 0.7)
        self.assertAlmostEqual(epsilon_by_episode(0, 5000), 1.0)
        self.assertAlmostEqual(epsilon_by_episode(1750, 5000), 0.525)
        self.assertAlmostEqual(epsilon_by_episode(3500, 5000), 0.05)
        self.assertAlmostEqual(epsilon_by_episode(5000, 5000), 0.05)

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

    def test_candidate_selection_rejects_points_beyond_bfs_limit(self):
        grid = np.zeros((1, 20), dtype=np.int32)
        self.assertEqual(MAX_CANDIDATE_BFS_DISTANCE, 12)
        with self.assertRaisesRegex(ValueError, "within 12"):
            select_nearest_candidate(grid, (0, 0), [(0, 13), (0, 19)])

    def test_plant_goal_selection_allows_more_than_waypoint_distance_limit(self):
        route = RouteProgress(0, 0, (22, 12))
        route.set_stage(4, (22, 12))

        self.assertIn(route.goal, LEFT_PLANT_CELLS)
        self.assertGreater(int(route.distance_map[22, 12]), MAX_CANDIDATE_BFS_DISTANCE)

    def test_split_patterns_guarantee_exact_group_sizes_and_a_goal(self):
        for pattern_index, expected_counts in enumerate(SPLIT_PATTERNS):
            selected, groups = choose_split_assignment(random.Random(7), 5, pattern_index)
            self.assertEqual(selected, pattern_index)
            self.assertEqual((groups.count(0), groups.count(1)), expected_counts)
            routes = [RouteProgress(group, selected, spawn) for group, spawn in zip(groups, [(23, 18), (23, 19), (23, 20), (23, 21), (23, 22)])]
            for group, route in zip(groups, routes):
                self.assertEqual(route.goal, WAYPOINT_POINTS["a"][group])

    def test_route_waypoints_advance_for_the_group_or_team_on_one_arrival(self):
        from concon_v1.co1_train_attacker import RouteEnv

        env = RouteEnv(seed=7)
        env.pattern_index = 0
        groups = [0, 0, 1, 1, 1]
        env.routes = [
            RouteProgress(group, env.pattern_index, position)
            for group, position in zip(groups, env.positions)
        ]
        group_indices = {
            group: next(i for i, route in enumerate(env.routes) if route.group == group)
            for group in (0, 1)
        }

        first_index = group_indices[0]
        env.positions[first_index] = env.routes[first_index].goal
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 0 for route in env.routes))
        _, masks = env._collect()
        self.assertFalse(masks[first_index][:4].any())
        self.assertTrue(masks[first_index][ACTION_WAIT])

        second_index = group_indices[1]
        env.positions[second_index] = env.routes[second_index].goal
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 1 for route in env.routes))
        self.assertEqual(len({route.goal for route in env.routes}), 1)

        shared_waypoint = env.routes[0].goal
        env.positions[0] = shared_waypoint
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 2 for route in env.routes))
        self.assertEqual(len({route.goal for route in env.routes}), 1)

    def test_single_a_group_can_advance_without_waiting_for_absent_group(self):
        from concon_v1.co1_train_attacker import RouteEnv

        env = RouteEnv(seed=11)
        env._a_completed_groups.clear()
        env.routes = [
            RouteProgress(0, env.pattern_index, position)
            for position in env.positions
        ]
        env.positions[0] = env.routes[0].goal

        env._advance_routes_if_reached()

        self.assertTrue(all(route.stage == 1 for route in env.routes))

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

    def test_plant_remains_available_after_route_marks_goal_reached(self):
        route = RouteProgress(0, 0, WAYPOINT_POINTS["a"][0])
        for _ in "abcd":
            self.assertTrue(route.advance_if_reached(route.goal))
        self.assertTrue(route.at_plant_stage)
        plant_goal = route.goal
        self.assertTrue(route.advance_if_reached(plant_goal))
        self.assertTrue(route.at_plant_stage)
        mask = build_action_mask(GRID, plant_goal, [], True, route.at_plant_stage, plant_goal)
        self.assertTrue(mask[ACTION_PLANT])
        observation = build_observation(route, plant_goal, True, [], 0, 0)
        self.assertEqual(observation[12], 1.0)

    def test_waits_when_all_bfs_progress_cells_are_blocked_by_allies(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        goal = (0, 1)
        distances = bfs_distance_map(grid, goal)
        mask = build_action_mask(
            grid, (1, 1), [(0, 1)], False, False, goal, distances,
        )
        self.assertTrue(mask[ACTION_WAIT])
        self.assertFalse(mask[:4].any())

    def test_keeps_open_bfs_progress_move_available(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        goal = (0, 0)
        distances = bfs_distance_map(grid, goal)
        mask = build_action_mask(
            grid, (1, 1), [(0, 1)], False, False, goal, distances,
        )
        self.assertTrue(mask[2])


if __name__ == "__main__":
    unittest.main()
