"""Diagnostics distinguish actual blocking from parking and planting progress."""

import contextlib
import io
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from concon_v1.check_co1_stuck import StopTracker, main, run_trial, snapshot, waypoint_checks
from concon_v1.co1_attacker_common import SPIKE_CARRIER_INDEX, SharedRouteDQN
from concon_v1.co1_attacker_scenarios import get_scenario
from concon_v1 import co1_train_attacker as training


def args(**overrides):
    values = dict(mode="route", policy="move-first", max_ticks=300,
                  stuck_ticks=10, epsilon=0.0)
    values.update(overrides)
    return SimpleNamespace(**values)


class StuckDiagnosticsTests(unittest.TestCase):
    def test_plain_cli_uses_progress_policy_and_prints_timeout_state(self):
        timeout = run_trial(get_scenario("A3"), args(max_ticks=2), 0, None)
        output = io.StringIO()
        with patch.object(sys, "argv", ["check_co1_stuck.py", "-map", "A3", "--rounds", "1"]), \
                patch("concon_v1.check_co1_stuck.run_trial", return_value=timeout) as trial, \
                contextlib.redirect_stdout(output):
            main()
        self.assertEqual(trial.call_args.args[1].policy, "move-first")
        self.assertIn("TIMEOUT", output.getvalue())
        self.assertIn("stage=a", output.getvalue())
        self.assertIn("assigned_goals=", output.getvalue())
        self.assertIn("arrived=", output.getvalue())
        self.assertNotIn("remaining=", output.getvalue())
        self.assertIn("carrier=True", output.getvalue())

    def test_chokepoint_describes_blocker_and_available_yield(self):
        env = training.RouteEnv(0, map_name="A3")
        env.positions = [(4, 42), (14, 34), (7, 39), (7, 40), (7, 35)]
        for index, route in enumerate(env.routes):
            route.set_stage(len(env.scenario.waypoint_order), env.positions[index],
                            goal=(7, 40) if index == SPIKE_CARRIER_INDEX else env.positions[index],
                            goal_index=0)
        _, masks = env._collect()
        actors = snapshot(env, masks, "route", [4] * 5)
        carrier = actors[SPIKE_CARRIER_INDEX]
        self.assertEqual(carrier["constraint"], "ally_blocked")
        self.assertEqual(carrier["allowed"], ["WAIT"])
        self.assertEqual(carrier["blockers"], [actors[3]["name"]])
        self.assertEqual(actors[3]["constraint"], "escort_holding")
        self.assertEqual(actors[3]["allowed"], ["UP"])
        tracker = StopTracker(10)
        for tick in range(1, 32):
            tracker.update(tick, actors)
        self.assertEqual(len(tracker.events), 1)
        self.assertEqual(tracker.events[0]["stopped_ticks"], 30)
        self.assertFalse(tracker.events[0]["resolved"])
        env.positions[SPIKE_CARRIER_INDEX] = (6, 40)
        _, masks = env._collect()
        tracker.update(32, snapshot(env, masks, "route", [4] * 5))
        self.assertTrue(tracker.events[0]["resolved"])

    def test_snapshot_reports_assigned_goal_without_obsolete_visit_requirements(self):
        env = training.RouteEnv(0, map_name="A1")
        env.positions[0] = env.routes[0].goal
        _, masks = env._collect()
        actors = snapshot(env, masks, "route", [4] * 5)
        self.assertTrue(actors[0]["goal_reached"])
        for actor in actors:
            self.assertEqual(actor["goal_reached"], actor["pos"] == actor["goal"])
            self.assertNotIn("required_points", actor)
            self.assertNotIn("visited_points", actor)

    def test_stage_change_is_reported_without_claiming_the_actor_moved(self):
        tracker = StopTracker(2)
        actor = dict(name="carrier", constraint="policy_wait", pos=[0, 0], stage="d",
                     goal=[0, 10], plant_progress=0, allowed=["RIGHT", "WAIT"], blockers=[])
        for tick in range(4):
            tracker.update(tick, [actor])
        tracker.update(4, [{**actor, "stage": "e"}])
        self.assertFalse(tracker.events[0]["resolved"])
        self.assertEqual(tracker.events[0]["ended_by"], "stage_changed")

    def test_available_plant_is_mandatory_and_progress_is_not_a_stop(self):
        env = training.RouteEnv(0, map_name="A3")
        env.positions[SPIKE_CARRIER_INDEX] = (6, 40)
        env.routes[SPIKE_CARRIER_INDEX].set_stage(5, (6, 40), goal=(6, 40), goal_index=0)
        _, masks = env._collect()
        actors = snapshot(env, masks, "route", [4] * 5)
        carrier = actors[SPIKE_CARRIER_INDEX]
        self.assertEqual(carrier["constraint"], "policy_wait")
        self.assertEqual(carrier["allowed"], ["PLANT"])
        tracker = StopTracker(3)
        for tick in range(1, 6):
            tracker.update(tick, [carrier])
        self.assertEqual(tracker.events[0]["reason"], "policy_wait")
        tracker = StopTracker(2)
        for tick in range(1, 6):
            tracker.update(tick, [{**carrier, "plant_progress": tick}])
        self.assertEqual(tracker.events, [])

    def test_moving_actor_does_not_generate_stop_events(self):
        tracker = StopTracker(2)
        actor = dict(name="carrier", constraint="policy_wait", pos=[0, 0], stage="a",
                     goal=[0, 10], plant_progress=0, allowed=["RIGHT", "WAIT"], blockers=[])
        for tick in range(10):
            tracker.update(tick, [{**actor, "pos": [0, tick]}])
        self.assertEqual(tracker.events, [])

    def test_candidate_check_reports_bfs_distance_over_limit(self):
        # The wall forces four steps despite a Manhattan distance of two.
        grid = np.zeros((3, 5), dtype=np.int32)
        grid[:2, 2] = 1
        scenario = SimpleNamespace(grid=grid, waypoint_order="ab",
                                   waypoint_points={"a": [(1, 1)], "b": [(1, 3), (2, 1)]},
                                   max_candidate_bfs_distance=3)
        candidates = waypoint_checks(scenario)[0]["candidates"]
        self.assertEqual(candidates, [
            {"pos": [1, 3], "distance": 4, "eligible": False},
            {"pos": [2, 1], "distance": 1, "eligible": True},
        ])
        json.dumps(candidates)

    def test_route_checks_all_maps_and_restores_training_limit(self):
        original = training.MAX_TICKS
        for map_name in ("A1", "A2", "A3"):
            with self.subTest(map=map_name):
                result = run_trial(get_scenario(map_name), args(), 0, None)
                self.assertLessEqual(result["ticks"], 300)
                self.assertEqual(training.MAX_TICKS, original)
                self.assertEqual(len(result["final"]), 5)
                self.assertIn(result["end_reason"], ("planted", "time_expired"))
                json.dumps(result)

    def test_battle_diagnostic_obeys_tick_limit_without_a_model_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_trial(get_scenario("A1"), args(mode="battle", max_ticks=2),
                               0, None, "gc_v1")
        self.assertEqual(result["ticks"], 2)
        self.assertEqual(result["end_reason"], "time_expired")
        self.assertFalse(result["planted"])

    def test_battle_model_uses_production_policy_decisions(self):
        scenario = get_scenario("A1")
        model = SharedRouteDQN(scenario.obs_dim)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_trial(scenario, args(mode="battle", policy="model", max_ticks=2,
                                              epsilon=0.5), 0, model, "gc_v1")
        self.assertEqual(result["ticks"], 2)
        self.assertGreater(sum(result["action_counts"].values()), 0)
        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
