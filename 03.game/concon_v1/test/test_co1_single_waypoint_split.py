"""Single waypoints support the whole team and split toward in-range candidates."""

from collections import Counter
import contextlib
import io
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


def arrive(env, index=0, *, randomize=False):
    env.positions[index] = env.routes[index].goal
    if randomize:
        env._advance_routes_if_reached()
    else:
        # Topology tests use a fixed 3:2 assignment; randomness is tested below.
        with patch.object(env.rng, "randrange", return_value=1), \
                patch.object(env.rng, "shuffle", return_value=None):
            env._advance_routes_if_reached()


class SingleWaypointSplitTests(unittest.TestCase):
    def test_two_candidates_use_all_four_random_splits_at_b_and_later_stages(self):
        for points in (
            {"a": [(3, 2)], "b": [(1, 6), (5, 6)]},
            {"a": [(3, 2)], "b": [(3, 4)], "c": [(1, 6), (5, 6)]},
        ):
            for pattern_index, expected in enumerate(SPLIT_PATTERNS):
                with self.subTest(stage=list(points)[-1], pattern=expected):
                    env = RouteEnv(0, map_name=make_scenario(points))
                    if "c" in points:
                        arrive(env)
                    with patch.object(env.rng, "randrange", return_value=pattern_index) as sample:
                        arrive(env, randomize=True)
                    goals = points[list(points)[-1]]
                    counts = Counter(route.goal for route in env.routes)
                    self.assertEqual(tuple(counts[goal] for goal in goals), expected)
                    sample.assert_called_once_with(4)
                    assigned = [route.goal for route in env.routes]
                    rng_state = env.rng.getstate()
                    env._advance_routes_if_reached()
                    self.assertEqual([route.goal for route in env.routes], assigned)
                    self.assertEqual(env.rng.getstate(), rng_state)

    def test_seeded_two_candidate_assignment_matches_production(self):
        scenario = make_scenario({"a": [(3, 2)], "b": [(1, 6), (5, 6)]})
        patterns = set()
        for seed in range(20):
            env = RouteEnv(seed, map_name=scenario)
            controller = ConconAttackerRouteController(
                model=SharedRouteDQN(scenario.obs_dim), seed=seed, map_name=scenario)
            chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                     has_spike=name == GORIGONS.spike_holder)
                     for name, pos in zip(GORIGONS.players, env.positions)]
            controller._prepare_round(chars)
            chars[0].pos = list(scenario.waypoint_points["a"][0])
            controller._prepare_route(chars[0], {"chars": chars, "grid": scenario.grid})
            arrive(env, randomize=True)
            self.assertEqual([route.goal for route in env.routes],
                             [controller._routes[char.name].goal for char in chars])
            counts = Counter(route.goal for route in env.routes)
            patterns.add(tuple(counts[goal] for goal in scenario.waypoint_points["b"]))
        self.assertEqual(patterns, set(SPLIT_PATTERNS))

    def test_multiple_origins_choose_the_nearest_candidate_independently(self):
        scenario = make_scenario({"a": [(1, 4), (5, 4)], "b": [(1, 6), (5, 6)]})
        env = RouteEnv(0, map_name=scenario)
        groups = [0, 0, 1, 1, 1]
        env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                      for group, pos in zip(groups, env.positions)]
        env.positions[0] = scenario.waypoint_points["a"][0]
        env.positions[2] = scenario.waypoint_points["a"][1]
        with patch.object(env.rng, "randrange") as sample:
            env._advance_routes_if_reached()
        sample.assert_not_called()
        self.assertEqual([route.goal for route in env.routes],
                         [(1, 6), (1, 6), (5, 6), (5, 6), (5, 6)])

    def test_multiple_origins_can_choose_the_same_nearest_candidate(self):
        scenario = make_scenario({"a": [(1, 4), (5, 4)], "b": [(3, 6), (3, 14)]})
        env = RouteEnv(0, map_name=scenario)
        groups = [0, 0, 1, 1, 1]
        env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                      for group, pos in zip(groups, env.positions)]
        env.positions[0], env.positions[2] = scenario.waypoint_points["a"]
        env._advance_routes_if_reached()
        self.assertEqual([route.goal for route in env.routes], [(3, 6)] * 5)

    def test_multiple_origins_exclude_out_of_range_candidates(self):
        scenario = make_scenario({"a": [(1, 4), (5, 4)], "b": [(1, 6), (5, 6)]}, limit=3)
        env = RouteEnv(0, map_name=scenario)
        groups = [0, 0, 1, 1, 1]
        env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                      for group, pos in zip(groups, env.positions)]
        env.positions[0], env.positions[2] = scenario.waypoint_points["a"]
        env._advance_routes_if_reached()
        self.assertEqual([route.goal for route in env.routes],
                         [(1, 6), (1, 6), (5, 6), (5, 6), (5, 6)])

    def test_multiple_origin_selection_matches_production(self):
        scenario = make_scenario({"a": [(1, 4), (5, 4)], "b": [(1, 6), (5, 6)]})
        env = RouteEnv(0, map_name=scenario)
        groups = [0, 0, 1, 1, 1]
        env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                      for group, pos in zip(groups, env.positions)]
        env.positions[0], env.positions[2] = scenario.waypoint_points["a"]
        controller = ConconAttackerRouteController(
            model=SharedRouteDQN(scenario.obs_dim), seed=0, map_name=scenario)
        chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                 has_spike=name == GORIGONS.spike_holder)
                 for name, pos in zip(GORIGONS.players, env.positions)]
        controller._pattern_index = 0
        controller._groups = dict(zip(GORIGONS.players, groups))
        controller._routes = {
            char.name: RouteProgress(group, 0, char.pos, scenario=scenario)
            for char, group in zip(chars, groups)
        }
        controller._prepare_route(chars[0], {"chars": chars, "grid": scenario.grid})
        env._advance_routes_if_reached()
        self.assertEqual([route.goal for route in env.routes],
                         [controller._routes[char.name].goal for char in chars])

    def _make_two_to_one_merge(self):
        scenario = make_scenario({
            "a": [(1, 4), (5, 4)], "b": [(1, 6), (5, 6)],
            "c": [(3, 9)], "d": [(1, 11), (5, 11)],
        })
        env = RouteEnv(7, map_name=scenario)
        groups = [0, 0, 1, 1, 1]
        env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                      for group, pos in zip(groups, env.positions)]
        env._a_completed_groups = {0, 1}
        for index, route in enumerate(env.routes):
            route.set_stage(1, env.positions[index], goal=scenario.waypoint_points["b"][groups[index]],
                            goal_index=groups[index])
        # One arrival advances the team; c is a single team goal.
        arrive(env, 0)
        self.assertTrue(all(route.stage == 2 and route.goal == (3, 9) for route in env.routes))
        return env

    def test_any_of_five_players_completes_two_to_one_merge_in_route_training(self):
        for arriving_index in range(5):
            with self.subTest(arriving_index=arriving_index):
                env = self._make_two_to_one_merge()
                other_positions = [pos for i, pos in enumerate(env.positions) if i != arriving_index]
                arrive(env, arriving_index)
                self.assertTrue(all(route.stage == 3 for route in env.routes))
                self.assertEqual([pos for i, pos in enumerate(env.positions) if i != arriving_index],
                                 other_positions)
                self.assertEqual(sum(pos == (3, 9) for pos in env.positions), 1)

    def test_any_of_five_players_completes_two_to_one_merge_in_production(self):
        for arriving_index in range(5):
            with self.subTest(arriving_index=arriving_index):
                env = self._make_two_to_one_merge()
                controller = ConconAttackerRouteController(
                    model=SharedRouteDQN(env.scenario.obs_dim), map_name=env.scenario,
                )
                chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                         has_spike=name == GORIGONS.spike_holder)
                         for name, pos in zip(GORIGONS.players, env.positions)]
                controller._pattern_index = 0
                controller._groups = {char.name: route.group for char, route in zip(chars, env.routes)}
                controller._routes = dict(zip(GORIGONS.players, env.routes))
                controller._a_completed_groups = {0, 1}
                chars[arriving_index].pos = [3, 9]
                controller._prepare_route(chars[arriving_index], {"chars": chars, "grid": env.scenario.grid})
                self.assertTrue(all(route.stage == 3 for route in controller._routes.values()))
                self.assertEqual(sum(char.pos == [3, 9] for char in chars), 1)

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
        }, limit=5))
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
        arrive(env, randomize=True)
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
