"""Exploration and greedy actions respect the assigned route movement mask."""

import random
import unittest

import numpy as np
import torch

from concon_v1.co1_attacker_common import (
    ACTION_PLANT, ACTION_WAIT, SPIKE_CARRIER_INDEX, SharedRouteDQN,
    _choose_action,
)
from concon_v1.co1_train_attacker import RouteEnv


class RouteExplorationTests(unittest.TestCase):
    def test_exploration_reaches_plant_on_a2_and_a3_within_round_limit(self):
        for map_name in ("A2", "A3"):
            for seed in range(5):
                with self.subTest(map=map_name, seed=seed):
                    env = RouteEnv(seed, map_name=map_name)
                    rng = random.Random(seed)
                    observations, masks = env._collect()
                    while not env.done:
                        actions = [
                            _choose_action(None, obs, mask, 1.0, rng,
                                           route=route, position=pos)
                            for obs, mask, route, pos in zip(
                                observations, masks, env.routes, env.positions)
                        ]
                        _, _, _, observations, masks, _ = env.step(actions)
                    self.assertTrue(env.success)
                    self.assertLessEqual(env.elapsed_ticks, 100)

    def test_greedy_policy_cannot_choose_an_ordinary_retreat(self):
        env = RouteEnv(0, map_name="A3")
        observations, masks = env._collect()
        route, pos, mask = env.routes[0], env.positions[0], masks[0]
        # A high Q value must not bypass the restored movement mask.
        from concon_v1.co1_attacker_common import CARDINAL_MOVES
        retreat = next(a for a, (dr, dc) in enumerate(CARDINAL_MOVES)
                       if 0 <= pos[0] + dr < env.scenario.grid.shape[0]
                       and 0 <= pos[1] + dc < env.scenario.grid.shape[1]
                       and env.scenario.grid[pos[0] + dr, pos[1] + dc] != 1
                       and route.distance_map[pos[0] + dr, pos[1] + dc]
                       > route.distance_map[pos])
        model = SharedRouteDQN(env.scenario.obs_dim)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.advantage[-1].bias[retreat] = 10.0
        self.assertFalse(mask[retreat])
        self.assertNotEqual(_choose_action(model, observations[0], mask, 0.0,
                                       random.Random(0), route=route, position=pos), retreat)

    def test_plant_continues_despite_stale_yield_state(self):
        env = RouteEnv(0, map_name="A3")
        carrier = SPIKE_CARRIER_INDEX
        goal = env.scenario.plant_cells[0]
        env.positions[carrier] = goal
        route = env.routes[carrier]
        route.set_stage(len(env.scenario.waypoint_order), goal, goal=goal, goal_index=0)
        route.yield_for = 0
        route.yield_origin = env.routes[0].goal
        route.yield_priority_goal = env.routes[0].goal
        observations, masks = env._collect()
        self.assertEqual(np.flatnonzero(masks[carrier]).tolist(), [ACTION_PLANT])
        self.assertIsNone(route.yield_for)
        for _ in range(4):
            observations, masks = env._collect()
            action = _choose_action(None, observations[carrier], masks[carrier], 1.0,
                                    random.Random(0), route=route, position=goal)
            self.assertEqual(action, ACTION_PLANT)
            actions = [ACTION_WAIT] * 5
            actions[carrier] = action
            env.step(actions)
        self.assertTrue(env.success)


if __name__ == "__main__":
    unittest.main()
