"""Deployment regression checks; no training or checkpoint loading."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ov1_train_defender_search as training
import ov1_learning_defender_search as runtime
from ov1_defender_search_routes import deployment_step, setup_distance_map


class DeploymentTests(unittest.TestCase):
    def test_forbidden_goal_waits_at_frontier_then_continues(self):
        grid = np.zeros((1, 7), dtype=np.int32)
        restricted = grid.copy()
        restricted[0, 3] = 1
        normal = training.ov1_common_rl.bfs_distance_map(grid, (0, 6))
        setup = setup_distance_map(restricted, normal, (0, 0))
        pos = (0, 0)
        for _ in range(10):
            dr, dc = deployment_step(restricted, setup, pos)
            pos = (pos[0] + dr, pos[1] + dc)
        self.assertEqual(pos, (0, 2))
        for _ in range(10):
            dr, dc = deployment_step(grid, normal, pos)
            pos = (pos[0] + dr, pos[1] + dc)
        self.assertEqual(pos, (0, 6))

    def test_route_detours_around_teammate(self):
        grid = np.zeros((3, 5), dtype=np.int32)
        dist = training.ov1_common_rl.bfs_distance_map(grid, (1, 4))
        pos = (1, 0)
        for _ in range(10):
            dr, dc = deployment_step(grid, dist, pos, {(1, 1)})
            pos = (pos[0] + dr, pos[1] + dc)
            self.assertNotEqual(pos, (1, 1))
        self.assertEqual(pos, (1, 4))

    def test_queue_for_moving_ally_instead_of_long_detour(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        grid[1:6, 1] = 1
        grid[3, 1] = 0
        dist = training.ov1_common_rl.bfs_distance_map(grid, (3, 8))
        blocked = {(3, 1)}
        self.assertEqual(deployment_step(grid, dist, (3, 0), blocked, blocked), (0, 0))
        self.assertEqual(deployment_step(grid, dist, (3, 0)), (0, 1))
        # An ally holding its post must still be bypassed.
        self.assertNotEqual(deployment_step(grid, dist, (3, 0), blocked), (0, 0))

    def test_take_short_detour_even_when_blocker_is_moving(self):
        grid = np.zeros((3, 5), dtype=np.int32)
        dist = training.ov1_common_rl.bfs_distance_map(grid, (1, 4))
        move = deployment_step(grid, dist, (1, 0), {(1, 1)}, {(1, 1)})
        self.assertIn(move, ((-1, 0), (1, 0)))

    def test_current_map_all_five_arrive_after_setup(self):
        self.assertEqual(training.DEFENSE_ASSIGNMENT, runtime._FIXED_DEFENSE_ASSIGNMENT)
        env = training.SearchEnv()
        env.reset()
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller._assignment_done = False
        controller._assigned_positions = {}
        controller._assigned_dist_maps = {}
        controller._assigned_setup_dist_maps = {}
        controller._prev_defense_bfs_dist = {}
        controller._ensure_defense_assignment(env.defenders[0], training.GRID, env.defenders)
        for index, defender in enumerate(env.defenders):
            np.testing.assert_array_equal(
                controller._assigned_setup_dist_maps[defender.name],
                training.SETUP_DEFENSE_POS_DIST_MAPS[index],
            )
        actions = {d.name: training.encode_action((0, 0), False, d.facing)
                   for d in env.defenders}
        for _ in range(training.DEFENDER_SETUP_TICKS):
            env.step(actions)
            for d in env.defenders:
                self.assertEqual(training.SETUP_WALK_GRID[tuple(d.pos)], 0)
        self.assertFalse(env.in_setup_phase)
        # No enemies or dead-carrier spike information in this deployment test.
        env.attackers = []
        env.idle_orb_outing.phase = "done"
        # Exercise actual normal-phase movement without combat or training.
        for _ in range(120):
            env.step(actions)
            if all(tuple(d.pos) == training.DEFENSE_ASSIGNMENT[d.name]
                   for d in env.defenders):
                break
        self.assertEqual([tuple(d.pos) for d in env.defenders], training.DEFENSE_POSITIONS)

    def test_real_map_deployment_does_not_repeatedly_backtrack(self):
        env = training.SearchEnv()
        env.reset()
        env.attackers = []
        env.idle_orb_outing.phase = "done"
        actions = {d.name: training.encode_action((0, 0), False, "S") for d in env.defenders}
        arrivals, backtracks = {}, {d.name: 0 for d in env.defenders}
        for tick in range(training.DEFENDER_SETUP_TICKS + 15):
            setup = env.in_setup_phase
            maps = {d.name: d.assigned_setup_dist_map if setup else d.assigned_defense_dist_map
                    for d in env.defenders}
            previous = {d.name: int(maps[d.name][tuple(d.pos)]) for d in env.defenders}
            env.step(actions)
            for d in env.defenders:
                backtracks[d.name] += int(maps[d.name][tuple(d.pos)] > previous[d.name])
                if tuple(d.pos) == training.DEFENSE_ASSIGNMENT[d.name]:
                    arrivals.setdefault(d.name, tick + 1)
            if len(arrivals) == len(env.defenders):
                break
        self.assertEqual(len(arrivals), 5)
        self.assertLessEqual(arrivals["おもこ"], training.DEFENDER_SETUP_TICKS + 7)
        self.assertLessEqual(arrivals["ねこさん"], training.DEFENDER_SETUP_TICKS + 12)
        self.assertTrue(all(count <= 2 for count in backtracks.values()), backtracks)


if __name__ == "__main__":
    unittest.main()
