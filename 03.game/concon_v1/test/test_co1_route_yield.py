"""Assigned waypoint arrivals advance stages, and blocked corridors allow yielding."""

import unittest

import numpy as np

from concon_v1.co1_attacker_common import (
    ACTION_WAIT, CARDINAL_MOVES, plant_stage_action_mask,
)
from concon_v1.co1_train_attacker import RouteEnv
from concon_v1.test.test_co1_single_waypoint_split import make_scenario, arrive


class RouteArrivalAndYieldTests(unittest.TestCase):
    def make_env(self):
        env = RouteEnv(0, map_name=make_scenario({
            "a": [(3, 2)], "b": [(1, 6), (5, 6)], "c": [(3, 9)],
            "d": [(1, 11), (5, 11)], "e": [(1, 13), (5, 13)],
        }))
        arrive(env)
        return env

    def test_one_assigned_arrival_advances_to_a_single_next_point(self):
        env = self.make_env()
        arrive(env, 0)
        self.assertTrue(all(route.stage == 2 and route.goal == (3, 9) for route in env.routes))

    def test_unassigned_out_of_range_point_is_not_reassigned_within_stage(self):
        env = RouteEnv(0, map_name=make_scenario({
            "a": [(3, 4)], "b": [(3, 6), (3, 18)], "c": [(3, 12)],
        }, limit=7))
        arrive(env)
        self.assertTrue(all(route.goal == (3, 6) for route in env.routes))
        arrive(env)
        self.assertTrue(all(route.stage == 2 for route in env.routes))
        self.assertTrue(all(route.goal == (3, 12) for route in env.routes))

    def test_opposite_unassigned_point_does_not_count_as_goal_arrival(self):
        env = self.make_env()
        # Visit the opposite branch, with neither assigned member arriving.
        env.positions[0] = (5, 6)
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 1 for route in env.routes))
        env.positions[0] = (3, 7)
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 1 for route in env.routes))
        self.assertEqual(env.routes[0].goal, (1, 6))
        env.positions[1] = (1, 6)
        env._advance_routes_if_reached()
        self.assertTrue(all(route.stage == 2 for route in env.routes))

    def test_each_marker_stage_is_visited_in_order_before_planting(self):
        env = self.make_env()
        arrive(env, 0)
        arrive(env, 4)  # One actor visits the single c point.
        self.assertTrue(all(route.stage == 3 for route in env.routes))
        arrive(env, 0)
        self.assertTrue(all(route.stage == 4 for route in env.routes))
        arrive(env, 0)
        self.assertTrue(all(route.at_plant_stage for route in env.routes))

    def test_remaining_actor_advances_without_revisiting_eliminated_branch(self):
        env = self.make_env()
        env.alive = [True, False, False, False, False]
        arrive(env, 0)
        self.assertEqual(env.routes[0].stage, 2)
        self.assertEqual(env.routes[0].goal, (3, 9))

    def test_pasted_head_on_block_has_an_actual_yield_move(self):
        env = RouteEnv(0, map_name="A3")
        env._a_completed_groups = {0}
        env.positions = [(7, 37), (14, 40), (13, 31), (11, 31), (12, 31)]
        goals = [(7, 36), (12, 40), (7, 36), (12, 40), (7, 36)]
        for index, route in enumerate(env.routes):
            route.set_stage(3, env.positions[index], goal=goals[index], goal_index=index % 2)
        _, masks = env._collect()
        self.assertFalse(masks[4][ACTION_WAIT])
        self.assertTrue(masks[4][2])  # Kurimaru steps left into the passing space.
        env.step([ACTION_WAIT] * 4 + [2])
        self.assertEqual(env.positions[4], (12, 30))
        _, masks = env._collect()
        self.assertTrue(masks[2][0])  # Carrier can now enter the vacated cell.
        env.step([ACTION_WAIT, ACTION_WAIT, 0, ACTION_WAIT, ACTION_WAIT])
        self.assertEqual(env.positions[2], (12, 31))

    def test_escort_uses_two_moves_to_reach_a_pullout_in_a_narrow_corridor(self):
        grid = np.ones((5, 5), dtype=np.int32)
        grid[1:5, 2] = 0
        grid[1, 3] = 0
        grid[0, 2] = 2
        first = plant_stage_action_mask(grid, (3, 2), [(4, 2)], (4, 2), (0, 2))
        self.assertFalse(first[ACTION_WAIT])
        self.assertTrue(first[0])
        second = plant_stage_action_mask(grid, (2, 2), [(4, 2)], (4, 2), (0, 2))
        self.assertTrue(second[0])
        third = plant_stage_action_mask(grid, (1, 2), [(4, 2)], (4, 2), (0, 2))
        self.assertTrue(third[3])
        parked = plant_stage_action_mask(grid, (1, 3), [(4, 2)], (4, 2), (0, 2))
        self.assertEqual(np.flatnonzero(parked).tolist(), [ACTION_WAIT])

    def test_normal_route_finishes_retreat_and_waits_for_priority_actor_to_pass(self):
        env = RouteEnv(0, map_name="A3")
        env.positions = [(7, 37), (14, 34), (14, 40), (13, 40), (17, 31)]
        goals = [(7, 36), (12, 40), (10, 40), (14, 40), (7, 36)]
        for i, route in enumerate(env.routes):
            route.set_stage(3, env.positions[i], goal=goals[i], goal_index=0)
        _, masks = env._collect()
        self.assertEqual(np.flatnonzero(masks[3]).tolist(), [0])
        env.positions[3] = (12, 40)
        _, masks = env._collect()
        self.assertEqual(np.flatnonzero(masks[3]).tolist(), [2])
        env.positions[3] = (12, 39)
        _, masks = env._collect()
        self.assertEqual(np.flatnonzero(masks[3]).tolist(), [ACTION_WAIT])
        env.positions[2] = (13, 40)
        _, masks = env._collect()
        self.assertIsNone(env.routes[3].yield_for)
        self.assertTrue(masks[3][:4].any())

    def test_escort_can_clear_a_plant_cell_at_the_carriers_destination(self):
        grid = np.zeros((3, 4), dtype=np.int32)
        grid[0, :3] = 2
        mask = plant_stage_action_mask(grid, (0, 2), [(0, 3)], (0, 3), (0, 2))
        self.assertFalse(mask[ACTION_WAIT])
        self.assertTrue(mask[:4].any())
        action = int(np.flatnonzero(mask[:4])[0])
        destination = (CARDINAL_MOVES[action][0], 2 + CARDINAL_MOVES[action][1])
        self.assertNotEqual(destination, (0, 2))


if __name__ == "__main__":
    unittest.main()
