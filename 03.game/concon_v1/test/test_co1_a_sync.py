import unittest
from types import SimpleNamespace

import torch
import numpy as np

from concon_v1.co1_learn_attacker_A1 import ConconAttackerA1Controller
from concon_v1.co1_train_attacker_A1 import (
    ACTION_DIM, ACTION_PLANT, ATTACKER_SPAWNS, GORIGONS, GRID,
    LEFT_PLANT_CELLS, SPIKE_CARRIER_INDEX, RouteEnv, RouteProgress,
    WAYPOINT_POINTS, bfs_distance_map, build_action_mask,
    plant_stage_action_mask, select_nearest_candidate,
)


class ZeroModel(torch.nn.Module):
    def forward(self, observations):
        return torch.zeros((observations.shape[0], ACTION_DIM))


class ASynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.chars = [
            SimpleNamespace(name=f"A{i}", team="A", pos=pos, is_alive=True,
                            has_spike=i == 0, plant_timer=0)
            for i, pos in enumerate(ATTACKER_SPAWNS)
        ]
        self.groups = [0, 0, 1, 1, 1]
        self.controller = ConconAttackerA1Controller.__new__(ConconAttackerA1Controller)
        self.controller._pattern_index = 0
        self.controller._groups = {c.name: group for c, group in zip(self.chars, self.groups)}
        self.controller._routes = {
            c.name: RouteProgress(group, 0, c.pos)
            for c, group in zip(self.chars, self.groups)
        }
        self.controller._a_completed_groups = set()
        self.controller.model = ZeroModel()
        self.state = {"chars": self.chars, "grid": GRID, "battle_tick": 1}

    def test_first_arrival_waits_until_other_group_reaches_a(self):
        first, second = self.chars[0], self.chars[2]
        first.pos = WAYPOINT_POINTS["a"][0]
        self.assertEqual(self.controller.decide_move(first, self.state), list(first.pos))
        self.assertTrue(all(route.stage == 0 for route in self.controller._routes.values()))

        second.pos = WAYPOINT_POINTS["a"][1]
        self.controller.decide_move(second, self.state)
        self.assertTrue(all(route.stage == 1 for route in self.controller._routes.values()))

    def test_arrival_continues_when_other_group_is_eliminated(self):
        first = self.chars[0]
        first.pos = WAYPOINT_POINTS["a"][0]
        for char in self.chars[2:]:
            char.is_alive = False
        self.controller.decide_move(first, self.state)
        self.assertTrue(all(self.controller._routes[c.name].stage == 1
                            for c in self.chars[:2]))

    def test_training_route_releases_when_other_group_is_eliminated(self):
        env = RouteEnv(seed=1)
        env.routes = [RouteProgress(group, 0, pos)
                      for group, pos in zip(self.groups, env.positions)]
        env.alive = [True, True, False, False, False]
        env.positions[0] = WAYPOINT_POINTS["a"][0]
        env._advance_routes_if_reached()
        self.assertTrue(all(env.routes[i].stage == 1 for i in (0, 1)))

    def test_b_goal_uses_a_arrival_instead_of_another_players_position(self):
        env = RouteEnv(seed=1)
        env.routes = [RouteProgress(group, 0, pos)
                      for group, pos in zip(self.groups, env.positions)]
        env.positions[0] = (21, 18)
        env.positions[1] = WAYPOINT_POINTS["a"][0]
        env.positions[3] = WAYPOINT_POINTS["a"][1]
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 1 for route in env.routes))

    def test_first_a_and_final_plant_leg_have_no_twelve_step_limit(self):
        route = RouteProgress(1, 0, ATTACKER_SPAWNS[0])
        self.assertEqual(route.goal, WAYPOINT_POINTS["a"][1])
        self.assertGreater(int(route.distance_map[ATTACKER_SPAWNS[0]]), 12)
        grid = np.zeros((1, 20), dtype=np.int32)
        self.assertEqual(select_nearest_candidate(
            grid, (0, 0), [(0, 13)], max_distance=None
        ), ((0, 13), 0, 13))
        route.set_stage(4, (22, 12))
        self.assertIn(route.goal, LEFT_PLANT_CELLS)
        self.assertGreater(int(bfs_distance_map(GRID, route.goal)[22, 12]), 12)

    def test_gorigons_gonta_is_the_training_carrier(self):
        env = RouteEnv(seed=1)
        self.assertEqual(GORIGONS.players[SPIKE_CARRIER_INDEX], "ごんた")
        observations, _ = env._collect()
        self.assertEqual([int(obs[21]) for obs in observations], [0, 0, 1, 0, 0])
        for index, route in enumerate(env.routes):
            route.set_stage(4, env.positions[index])
        env.positions[SPIKE_CARRIER_INDEX] = env.routes[SPIKE_CARRIER_INDEX].goal
        _, masks = env._collect()
        self.assertTrue(masks[SPIKE_CARRIER_INDEX][ACTION_PLANT])
        self.assertFalse(any(mask[ACTION_PLANT] for i, mask in enumerate(masks)
                             if i != SPIKE_CARRIER_INDEX))

    def test_route_mask_rejects_moves_away_from_current_goal(self):
        grid = np.zeros((3, 3), dtype=np.int32)
        distances = bfs_distance_map(grid, (0, 0))
        mask = build_action_mask(grid, (1, 1), [], False, False, (0, 0), distances)
        self.assertTrue(mask[0])
        self.assertTrue(mask[2])
        self.assertFalse(mask[1])
        self.assertFalse(mask[3])

    def test_only_carrier_heads_to_plant_after_d(self):
        env = RouteEnv(seed=1)
        for i, route in enumerate(env.routes):
            route.set_stage(3, env.positions[i], goal=WAYPOINT_POINTS["d"][1], goal_index=1)
        env.positions[0] = WAYPOINT_POINTS["d"][1]
        env._a_completed_groups = {0, 1}
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 4 for route in env.routes))
        self.assertIn(env.routes[SPIKE_CARRIER_INDEX].goal, LEFT_PLANT_CELLS)
        self.assertEqual(env.routes[0].goal, env.positions[0])
        _, masks = env._collect()
        self.assertFalse(masks[0][ACTION_PLANT])

    def test_escort_steps_off_carrier_route_instead_of_blocking(self):
        grid = np.zeros((3, 4), dtype=np.int32)
        grid[0, 0] = 2
        mask = plant_stage_action_mask(grid, (0, 2), [(0, 3)], (0, 3), (0, 0))
        self.assertTrue(mask[1])
        self.assertFalse(mask[4])


if __name__ == "__main__":
    unittest.main()
