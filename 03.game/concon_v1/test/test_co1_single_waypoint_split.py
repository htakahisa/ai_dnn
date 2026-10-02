"""Single waypoints support the whole team and split toward in-range candidates."""

from collections import Counter
import contextlib
import io
import random
import unittest
from types import SimpleNamespace

from concon_v1.co1_attacker_common import (
    GORIGONS, RouteProgress, SharedRouteDQN, SPLIT_PATTERNS, choose_split_assignment,
)
from concon_v1.co1_attacker_scenarios import build_scenario, get_scenario
from concon_v1.co1_learn_attacker import ConconAttackerRouteController
from concon_v1.co1_train_attacker import RouteEnv


def make_scenario(points, limit=12, walls=()):
    terrain = [list("0" * 20) for _ in range(7)]
    terrain[0][18] = "2"
    terrain[6][1:6] = list("33333")
    for row, col in walls:
        terrain[row][col] = "1"
    strategy = [row.copy() for row in terrain]
    for marker, cells in points.items():
        for row, col in cells:
            strategy[row][col] = marker
    return build_scenario("SINGLE_TEST", "\n".join(map("".join, strategy)), "right",
                          game_map="\n".join(map("".join, terrain)),
                          waypoint_order="".join(points), max_candidate_bfs_distance=limit)


def arrive(env, index=0):
    env.positions[index] = env.routes[index].goal
    env._advance_routes_if_reached()


class SingleWaypointSplitTests(unittest.TestCase):
    def test_single_a_always_assigns_everyone_to_its_only_point(self):
        scenario = make_scenario({"a": [(3, 4)], "b": [(3, 6)]})
        for seed in range(8):
            env = RouteEnv(seed, map_name=scenario)
            self.assertEqual([route.goal for route in env.routes], [(3, 4)] * 5)
            self.assertEqual([route.group for route in env.routes], [0] * 5)
            self.assertEqual(SPLIT_PATTERNS[env.pattern_index], (5, 0))
        route = RouteProgress(1, 0, scenario.attacker_spawns[0], scenario=scenario)
        self.assertEqual((route.group, route.goal_index, route.goal), (0, 0, (3, 4)))
        rng = random.Random(7)
        state = rng.getstate()
        self.assertEqual(choose_split_assignment(rng, a_point_count=1), (3, [0] * 5))
        self.assertEqual(rng.getstate(), state)

    def test_single_a_splits_five_across_two_three_or_five_b_candidates(self):
        for goals, counts in (
            ([(1, 6), (5, 6)], [2, 3]),
            ([(1, 6), (3, 8), (5, 6)], [1, 2, 2]),
            ([(r, 6) for r in range(1, 6)], [1] * 5),
        ):
            with self.subTest(candidates=len(goals)):
                env = RouteEnv(7, map_name=make_scenario({"a": [(3, 4)], "b": goals}))
                env._advance_routes_if_reached()
                self.assertTrue(all(route.stage == 0 for route in env.routes))
                arrive(env)
                assigned = Counter(route.goal for route in env.routes)
                self.assertEqual(set(assigned), set(goals))
                self.assertEqual(sorted(assigned.values()), counts)

    def test_single_later_waypoints_also_split_toward_multiple_candidates(self):
        for goals, counts in (([(1, 6), (5, 6)], [2, 3]),
                              ([(1, 6), (3, 8), (5, 6)], [1, 2, 2])):
            with self.subTest(candidates=len(goals)):
                env = RouteEnv(7, map_name=make_scenario({
                    "a": [(3, 2)], "b": [(3, 4)], "c": goals,
                }))
                arrive(env)
                arrive(env)
                self.assertTrue(all(route.stage == 2 for route in env.routes))
                self.assertEqual(sorted(Counter(r.goal for r in env.routes).values()), counts)

    def test_split_origins_choose_next_candidates_from_their_own_waypoint(self):
        env = RouteEnv(7, map_name=make_scenario({
            "a": [(3, 4)], "b": [(1, 6), (5, 6)], "c": [(1, 9), (5, 9)],
        }))
        arrive(env)
        previous = [route.goal for route in env.routes]
        arrive(env)
        for source, route in zip(previous, env.routes):
            self.assertEqual(route.goal, (source[0], 9))
        self.assertTrue(all(route.stage == 2 for route in env.routes))

    def test_only_reachable_candidates_within_the_limit_receive_players(self):
        for goals, limit, walls in (
            ([(3, 6), (3, 9), (3, 18)], 5, ()),
            ([(3, 6), (3, 9), (3, 12)], 12, [(r, 10) for r in range(7)]),
        ):
            with self.subTest(limit=limit):
                env = RouteEnv(7, map_name=make_scenario({"a": [(3, 4)], "b": goals},
                                                       limit=limit, walls=walls))
                arrive(env)
                self.assertEqual(Counter(r.goal for r in env.routes), {(3, 6): 3, (3, 9): 2})

    def test_one_eligible_candidate_gets_everyone(self):
        env = RouteEnv(7, map_name=make_scenario({"a": [(3, 4)], "b": [(3, 6), (3, 18)]}))
        arrive(env)
        self.assertEqual(Counter(r.goal for r in env.routes), {(3, 6): 5})

    def test_only_living_players_are_split(self):
        env = RouteEnv(7, map_name=make_scenario({"a": [(3, 4)], "b": [(1, 6), (5, 6)]}))
        env.alive = [True, False, True, False, True]
        arrive(env)
        self.assertEqual(sorted(Counter(env.routes[i].goal for i in (0, 2, 4)).values()), [1, 2])
        self.assertEqual([env.routes[i].stage for i in (1, 3)], [0, 0])

    def test_no_eligible_candidate_keeps_the_existing_layout_error(self):
        env = RouteEnv(7, map_name=make_scenario({"a": [(3, 4)], "b": [(3, 6)]}, limit=1))
        with self.assertRaisesRegex(ValueError, "within 1 BFS steps"):
            arrive(env)

    def test_production_controller_and_route_training_use_the_same_three_way_split(self):
        scenario = make_scenario({"a": [(3, 4)], "b": [(1, 6), (3, 8), (5, 6)]})
        env = RouteEnv(7, map_name=scenario)
        controller = ConconAttackerRouteController(model=SharedRouteDQN(scenario.obs_dim),
                                                  seed=7, map_name=scenario)
        chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                 has_spike=name == GORIGONS.spike_holder)
                 for name, pos in zip(GORIGONS.players, scenario.attacker_spawns)]
        controller._prepare_round(chars)
        self.assertEqual(set(controller._groups.values()), {0})
        chars[0].pos = [3, 4]
        controller._prepare_route(chars[0], {"chars": chars, "grid": scenario.grid})
        arrive(env)
        self.assertEqual([route.goal for route in env.routes],
                         [controller._routes[char.name].goal for char in chars])
        self.assertEqual(sorted(Counter(r.goal for r in env.routes).values()), [1, 2, 2])
        controller.reset_round()
        controller._prepare_round(chars)
        self.assertTrue(all(route.stage == 0 and route.goal == (3, 4)
                            for route in controller._routes.values()))

    def test_real_battle_training_accepts_one_a_without_changing_observation_size(self):
        from concon_v1.co1_battle_training import BattleRouteEnv

        original = get_scenario("A1")
        scenario = build_scenario("SINGLE_A", original.strategy_map.replace("a", "0", 1), "left")
        with contextlib.redirect_stdout(io.StringIO()):
            env = BattleRouteEnv(7, opponents=["omoko_v1"], map_name=scenario)
            observations, _ = env._collect()
            self.assertTrue(all(route.goal == scenario.waypoint_points["a"][0] for route in env.routes))
            self.assertTrue(all(obs.shape == (original.obs_dim,) for obs in observations))
            env.step(epsilon=0)
        self.assertEqual(env.elapsed_ticks, 1)


if __name__ == "__main__":
    unittest.main()
