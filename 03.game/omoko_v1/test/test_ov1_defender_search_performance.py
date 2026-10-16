"""Compare search optimizations with original geometry; never train/save weights."""

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ov1_train_defender_search as search
import ov1_defender_search_support as support
from ov1_defender_search_geometry import SearchGeometryCache


def original_projectile_impact(grid, start, aim, role):
    start, aim = tuple(map(int, start)), tuple(map(int, aim))
    if start == aim:
        return None
    scale = max(grid.shape) * 3
    far = (start[0] + (aim[0] - start[0]) * scale,
           start[1] + (aim[1] - start[1]) * scale)
    path = [start]
    for r, c in support.line_cells(start, far)[1:]:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            break
        if grid[r, c] == 1:
            break
        path.append((r, c))
    if len(path) <= 1:
        return None
    if role == 'FLASH':
        return path[min(len(path) - 1,
                        support.FLASH_SPEED_CELLS_PER_TICK * support.FLASH_MAX_FLIGHT_TICKS)]
    return path[-1]


def rollout(optimized, episodes=3):
    """Fixed-seed simulation only: no optimizer, replay buffer or checkpoint I/O."""
    random.seed(42)
    torch.manual_seed(42)
    net = search.DuelingQNet(search.OBS_DIM, search.ACTION_DIM)
    trace = []
    projectile = support.projectile_impact if optimized else original_projectile_impact
    with patch.object(search, 'GEOMETRY_CACHE_ENABLED', optimized), \
            patch.object(support, 'projectile_impact', projectile):
        env = search.SearchEnv()
        for _ in range(episodes):
            observations, masks = env.reset()
            for _ in range(search.MAX_TICKS + search.DEFENDER_SETUP_TICKS):
                actions = {name: search.select_action(net, obs, masks[name], 0.5)
                           for name, obs in observations.items()}
                observations, masks, rewards, done, actual = env.step(actions)
                trace.append((
                    tuple((name, obs.tobytes(), masks[name].tobytes())
                          for name, obs in observations.items()),
                    actions, actual, rewards, done, env.match_over_reason,
                    tuple((u.name, tuple(u.pos), u.hp, u.kills)
                          for u in env.defenders + env.attackers),
                    random.getstate(),
                ))
                if done or not observations:
                    break
    return trace


class SearchPerformanceTests(unittest.TestCase):
    def test_los_matches_original_with_smoke_and_in_place_wall_edits(self):
        rng = np.random.default_rng(214)
        cache = SearchGeometryCache(los_capacity=16)
        for _ in range(30):
            grid = (rng.random((8, 8)) < 0.2).astype(np.int32)
            for _ in range(30):
                start, end = [tuple(map(int, rng.integers(0, 8, size=2))) for _ in range(2)]
                for smoke in (set(), {start}, {end}, {(3, 3), (4, 4)}):
                    self.assertEqual(cache.has_los(grid, start, end, smoke),
                                     search.ov1_common_rl.has_los(grid, start, end, smoke))
                grid[start] = 1 - grid[start]
                self.assertEqual(cache.has_los(grid, start, end),
                                 search.ov1_common_rl.has_los(grid, start, end))
            self.assertLessEqual(len(cache._lines), 16)

    def test_distance_reuse_bounded_capacity_and_map_invalidation(self):
        grid = np.zeros((6, 6), dtype=np.int32)
        cache = SearchGeometryCache(distance_capacity=2)
        first = cache.distance_map(grid, (0, 0))
        self.assertFalse(first.flags.writeable)
        with patch.object(search.ov1_common_rl, 'bfs_distance_map',
                          side_effect=AssertionError('Repeated goal must reuse BFS')):
            self.assertIs(first, cache.distance_map(grid, (0, 0)))
        for goal in ((1, 1), (2, 2), (3, 3)):
            np.testing.assert_array_equal(cache.distance_map(grid, goal),
                                          search.ov1_common_rl.bfs_distance_map(grid, goal))
        self.assertEqual(len(cache._distances), 2)
        grid[2, 2] = 1
        np.testing.assert_array_equal(cache.distance_map(grid, (2, 2)),
                                      search.ov1_common_rl.bfs_distance_map(grid, (2, 2)))
        self.assertEqual(len(cache._distances), 1)

    def test_streamed_projectile_matches_original_for_both_roles(self):
        rng = np.random.default_rng(33)
        for _ in range(1000):
            grid = (rng.random((8, 8)) < 0.2).astype(np.int32)
            start, aim = [tuple(map(int, rng.integers(0, 8, size=2))) for _ in range(2)]
            for role in ('FLASH', 'RECON'):
                self.assertEqual(support.projectile_impact(grid, start, aim, role),
                                 original_projectile_impact(grid, start, aim, role))

    def test_rollouts_match_observations_masks_actions_rewards_and_random_state(self):
        self.assertEqual(rollout(False), rollout(True))


if __name__ == '__main__':
    unittest.main()
