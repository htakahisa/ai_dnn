"""Every marker uses its letter case to select the allowed split patterns."""

from collections import Counter
import string
import unittest

from concon_v1.co1_attacker_common import GORIGONS, RouteProgress, SharedRouteDQN, SPLIT_PATTERNS
from concon_v1.co1_attacker_scenarios import parse_strategy_points, validate_checkpoint_scenario
from concon_v1.co1_learn_attacker import ConconAttackerRouteController
from concon_v1.co1_train_attacker import RouteEnv
from concon_v1.test.test_co1_single_waypoint_split import make_scenario
from types import SimpleNamespace


class MarkerCaseTests(unittest.TestCase):
    def test_parser_normalizes_order_and_records_uppercase_from_the_map(self):
        self.assertEqual(parse_strategy_points("aBBCC", "aBc"),
                         {"a": [(0, 0)], "b": [(0, 1), (0, 2)], "c": [(0, 3), (0, 4)]})
        for text in ("abB", "AaBB"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "mix uppercase"):
                parse_strategy_points(text, "ab")

    def test_all_later_letters_follow_the_case_rule(self):
        for letter in string.ascii_lowercase[1:].replace("s", "").replace("u", ""):
            for uppercase in (False, True):
                marker = letter.upper() if uppercase else letter
                scenario = make_scenario({"a": [(3, 2)], marker: [(1, 6), (5, 6)]})
                observed = set()
                for seed in range(20):
                    env = RouteEnv(seed, map_name=scenario)
                    env.positions[0] = env.routes[0].goal
                    env._advance_routes_if_reached()
                    counts = Counter(route.goal for route in env.routes)
                    observed.add(tuple(counts[goal] for goal in scenario.waypoint_points[letter]))
                with self.subTest(marker=marker):
                    self.assertEqual(observed, set(SPLIT_PATTERNS[:2] if uppercase else SPLIT_PATTERNS))

    def test_initial_a_also_follows_the_case_rule_and_matches_production(self):
        for marker in ("a", "A"):
            scenario = make_scenario({marker: [(1, 2), (5, 2)], "b": [(3, 6)]})
            observed = set()
            for seed in range(20):
                env = RouteEnv(seed, map_name=scenario)
                controller = ConconAttackerRouteController(
                    model=SharedRouteDQN(scenario.obs_dim), seed=seed, map_name=scenario)
                chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                         has_spike=name == GORIGONS.spike_holder)
                         for name, pos in zip(GORIGONS.players, env.positions)]
                controller._prepare_round(chars)
                self.assertEqual([route.goal for route in env.routes],
                                 [controller._routes[char.name].goal for char in chars])
                observed.add(SPLIT_PATTERNS[env.pattern_index])
            self.assertEqual(observed, set(SPLIT_PATTERNS[:2] if marker.isupper() else SPLIT_PATTERNS))

    def test_uppercase_later_stage_chooses_the_nearest_goal_per_origin(self):
        scenario = make_scenario({"a": [(1, 2), (5, 2)], "Z": [(1, 6), (5, 6)]})
        for seed in range(20):
            env = RouteEnv(seed, map_name=scenario)
            groups = [0, 0, 1, 1, 1]
            env.routes = [RouteProgress(group, 0, pos, scenario=scenario)
                          for group, pos in zip(groups, env.positions)]
            env.positions[0], env.positions[2] = scenario.waypoint_points["a"]
            env._advance_routes_if_reached()
            self.assertEqual([route.goal for route in env.routes],
                             [(1, 6), (1, 6), (5, 6), (5, 6), (5, 6)])

    def test_uppercase_later_assignment_matches_production(self):
        scenario = make_scenario({"a": [(3, 2)], "Z": [(1, 6), (5, 6)]})
        for seed in range(20):
            env = RouteEnv(seed, map_name=scenario)
            controller = ConconAttackerRouteController(
                model=SharedRouteDQN(scenario.obs_dim), seed=seed, map_name=scenario)
            chars = [SimpleNamespace(name=name, team="A", pos=list(pos), is_alive=True,
                                     has_spike=name == GORIGONS.spike_holder)
                     for name, pos in zip(GORIGONS.players, env.positions)]
            chars[0].pos = list(scenario.waypoint_points["a"][0])
            env.positions[0] = tuple(chars[0].pos)
            env._advance_routes_if_reached()
            controller._prepare_route(chars[0], {"chars": chars, "grid": scenario.grid})
            self.assertEqual([route.goal for route in env.routes],
                             [controller._routes[char.name].goal for char in chars])

    def test_case_change_is_rejected_by_checkpoint_signature(self):
        lower = make_scenario({"a": [(3, 2)], "b": [(1, 6), (5, 6)]})
        upper = make_scenario({"a": [(3, 2)], "B": [(1, 6), (5, 6)]})
        self.assertEqual(lower.waypoint_points, upper.waypoint_points)
        self.assertEqual(lower.obs_dim, upper.obs_dim)
        self.assertEqual(upper.uppercase_markers, frozenset({"b"}))
        checkpoint = {"map_name": lower.map_name, "waypoint_order": lower.waypoint_order,
                      "scenario_signature": lower.signature}
        validate_checkpoint_scenario(checkpoint, lower)
        with self.assertRaisesRegex(ValueError, "route/terrain settings"):
            validate_checkpoint_scenario(checkpoint, upper)


if __name__ == "__main__":
    unittest.main()
