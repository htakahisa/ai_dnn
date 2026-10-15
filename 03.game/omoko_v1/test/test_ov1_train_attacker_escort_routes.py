"""Regression checks for the scripted carrier's waypoint routes; no training."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ov1_train_attacker_escort as escort


class EscortCarryRoutesTest(unittest.TestCase):
    def test_every_current_route_reaches_same_site_plant_via_waypoint(self):
        env = escort.EscortEnv(seed=12)
        for site, waypoints in env.waypoint_cells_by_site.items():
            self.assertTrue(waypoints)
            for waypoint in waypoints:
                for start in env.attacker_spawns:
                    target = env.plant_cells_by_site[site][0]
                    path = escort._carry_route_path(env.grid, start, waypoint, target)
                    self.assertEqual(path[0], start)
                    self.assertEqual(path[-1], target)
                    self.assertTrue(any(escort._chebyshev(p, waypoint) <= 1 for p in path))
                    for a, b in zip(path, path[1:]):
                        self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
                        self.assertNotEqual(env.grid[b], 1)

    def test_adjacent_waypoint_switches_without_entering_marker(self):
        grid = np.zeros((5, 7), dtype=int)
        path = escort._carry_route_path(grid, (2, 0), (2, 4), (0, 3))
        self.assertEqual(path[:4], [(2, 0), (2, 1), (2, 2), (2, 3)])
        self.assertNotIn((2, 4), path)
        already_near = escort._carry_route_path(grid, (2, 3), (2, 4), (0, 3))
        self.assertEqual(already_near, [(2, 3), (1, 3), (0, 3)])

    def test_marker_counts_are_dynamic_and_missing_waypoints_allow_direct_route(self):
        grid = np.zeros((3, 8), dtype=int)
        waypoints, plants = escort._parse_carry_routes(
            "00000000\n77500666\n02000020", grid,
        )
        self.assertEqual(len(waypoints["left"]), 2)
        self.assertEqual(len(waypoints["right"]), 3)
        self.assertEqual(plants["left"], [(1, 2)])
        self.assertEqual(plants["right"], [(2, 6)])
        path = escort._carry_route_path(grid, (0, 0), None, (0, 2))
        self.assertEqual(path, [(0, 0), (0, 1), (0, 2)])

    def test_map_mismatch_and_unreachable_route_fail_explicitly(self):
        with self.assertRaises(ValueError):
            escort._parse_carry_routes("100\n000", np.zeros((2, 3)))
        grid = np.array([[0, 1, 0], [0, 1, 0]])
        with self.assertRaises(ValueError):
            escort._carry_route_path(grid, (0, 0), (0, 2), (1, 2))
        with self.assertRaises(ValueError):
            escort._carry_route_path(grid, (0, 0), None, (1, 2))

    def test_reset_samples_all_routes_and_sets_correct_path(self):
        env = escort.EscortEnv(seed=7)
        seen = set()
        for _ in range(80):
            obs = env.reset()
            seen.add((env.active_site, env.carry_waypoint))
            self.assertIn(env.goal_pos, env.plant_cells_by_site[env.active_site])
            self.assertEqual(env.carry_path[0], env.carry_pos)
            self.assertEqual(env.carry_path[-1], env.goal_pos)
            self.assertEqual(len(obs[0]), env._obs_dim())
        expected = {(site, cell) for site, cells in env.waypoint_cells_by_site.items() for cell in cells}
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
