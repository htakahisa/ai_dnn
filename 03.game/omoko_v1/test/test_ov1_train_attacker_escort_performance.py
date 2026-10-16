"""Check speed optimizations against original decisions; never train/save weights."""
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ov1_train_attacker_escort as escort
from ov1_common_rl import select_action


class EscortPerformanceTest(unittest.TestCase):
    def test_single_wall_scan_matches_previous_los_with_dynamic_smoke(self):
        rng = np.random.default_rng(214)
        for _ in range(1000):
            grid = (rng.random((8, 8)) < 0.2).astype(int)
            start, end = [tuple(map(int, rng.integers(0, 8, size=2))) for _ in range(2)]
            smoke = {tuple(map(int, rng.integers(0, 8, size=2))) for _ in range(5)}
            if rng.random() < 0.5:
                smoke.clear()
            wall_clear = all(grid[r, c] != 1 for r, c in escort.wall_line_cells(start, end))
            expected = wall_clear and escort.ov1_common_rl.has_los(grid, start, end, smoke)
            self.assertEqual(escort._has_los(grid, smoke, start, end), expected)

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_actions_and_exploration_random_state_match_original(self):
        rng = np.random.default_rng(19)
        torch.manual_seed(19)
        net = escort.DuelingQNet(73, escort.EscortEnv.N_ACTIONS)
        for epsilon in (0.0, 0.25, 0.75, 1.0):
            for seed in range(20):
                observations = rng.normal(size=(4, 73)).astype(np.float32)
                masks = [rng.random(escort.EscortEnv.N_ACTIONS) > 0.4 for _ in range(4)]
                if seed % 2:
                    masks[1] = None
                    masks[2][:] = False
                random.seed(seed)
                expected = [
                    select_action(net, observations[i], mask, epsilon,
                                  device='cpu', fallback_action=escort.EscortEnv.ACTION_STAY)
                    if mask is not None else None
                    for i, mask in enumerate(masks)
                ]
                expected_rng = random.getstate()
                random.seed(seed)
                actual = escort._select_escort_actions(net, observations, masks, 'cpu', epsilon)
                self.assertEqual(actual, expected)
                self.assertEqual(random.getstate(), expected_rng)

    def test_evaluation_does_not_consume_randomness_and_batches_forward(self):
        torch.manual_seed(9)
        net = escort.DuelingQNet(73, escort.EscortEnv.N_ACTIONS)
        obs = np.random.default_rng(9).normal(size=(4, 73)).astype(np.float32)
        masks = [np.ones(escort.EscortEnv.N_ACTIONS, dtype=bool) for _ in range(4)]
        with torch.no_grad():
            expected = [int(net(torch.as_tensor(row).unsqueeze(0)).argmax()) for row in obs]
        state = random.getstate()
        with patch.object(net, 'forward', wraps=net.forward) as forward:
            self.assertEqual(escort._select_escort_actions(net, obs, masks, 'cpu'), expected)
            self.assertEqual(forward.call_count, 1)
        self.assertEqual(random.getstate(), state)

    def test_near_tie_uses_original_single_row_decision(self):
        class NearTie(torch.nn.Module):
            def forward(self, states):
                q = torch.zeros((len(states), 3))
                q[:, 1 if len(states) > 1 else 0] = 1e-7
                return q
        masks = [np.ones(3, dtype=bool)] * 4
        with patch.object(escort, 'BATCH_ACTION_INFERENCE', True):
            self.assertEqual(escort._select_escort_actions(
                NearTie(), np.zeros((4, 2), dtype=np.float32), masks, 'cpu'), [0] * 4)

    def test_distance_cache_matches_bfs_and_invalidates_on_wall_changes(self):
        env = escort.EscortEnv(seed=5)
        for source in env.attacker_spawns[:3]:
            expected = escort._build_distance_map_walls_only(env.grid, [source])
            first = env._wall_distance_map(source)
            np.testing.assert_array_equal(first, expected)
            with patch.object(escort, '_build_distance_map_walls_only',
                              side_effect=AssertionError('BFS should be reused')):
                self.assertIs(env._wall_distance_map(source), first)
            self.assertFalse(first.flags.writeable)
        source = env.attacker_spawns[0]
        env.grid[source] = 1
        np.testing.assert_array_equal(env._wall_distance_map(source),
                                      escort._build_distance_map_walls_only(env.grid, [source]))

    def test_cached_orb_steps_keep_dynamic_occupancy(self):
        grid = np.zeros((5, 5), dtype=int)
        goal = (2, 4)
        distances = escort._build_distance_map_walls_only(grid, [goal])
        for blocked in ((), ((2, 1),), ((2, 1), (1, 0), (3, 0))):
            self.assertEqual(escort._bfs_next_step(grid, (2, 0), goal, blocked),
                             escort._bfs_next_step(grid, (2, 0), goal, blocked, distances))

    def test_seeded_rollouts_match_observations_actions_rewards_and_rng(self):
        torch.manual_seed(8)
        net = escort.DuelingQNet(73, escort.EscortEnv.N_ACTIONS)

        def rollout(optimized, epsilon):
            random.seed(42)
            env = escort.EscortEnv(seed=42, max_ticks=18)
            trace = []
            with patch.object(escort, 'CACHE_WALL_DISTANCE_MAPS', optimized):
                for _ in range(2):
                    obs = env.reset()
                    done = False
                    while not done:
                        masks = [env.get_action_mask(i) if env.escort_alive[i] else None
                                 for i in range(env.n_escorts)]
                        if optimized:
                            actions = escort._select_escort_actions(net, obs, masks, 'cpu', epsilon)
                        else:
                            actions = [select_action(net, obs[i], mask, epsilon, device='cpu',
                                       fallback_action=escort.EscortEnv.ACTION_STAY)
                                       if mask is not None else None for i, mask in enumerate(masks)]
                        obs, rewards, done, info = env.step(actions)
                        trace.append((np.asarray(obs).tobytes(), actions, rewards, done, info,
                                      random.getstate(), env.rng.getstate()))
            return trace

        for epsilon in (0.0, 0.5, 1.0):
            self.assertEqual(rollout(False, epsilon), rollout(True, epsilon))


if __name__ == '__main__':
    unittest.main()
