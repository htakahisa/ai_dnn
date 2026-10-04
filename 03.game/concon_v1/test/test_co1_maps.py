"""Map selection, model compatibility, and shared A1/A2 route behavior."""

import contextlib
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from concon_v1 import co1_train_attacker as training
from concon_v1.co1_attacker_common import (
    ACTION_DIM, ACTION_PLANT, GORIGONS, OBS_DIM, RouteProgress, SharedRouteDQN,
    SPIKE_CARRIER_INDEX, build_action_mask, build_observation,
)
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, ScenarioSettings, build_scenario, get_scenario, validate_checkpoint_scenario,
)
from concon_v1.co1_learn_attacker import ConconAttackerRouteController

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1 import evaluate_co1_attacker as evaluation
    from concon_v1.co1_battle_training import BattleRouteEnv


def checkpoint_bytes(map_name=None):
    scenario = get_scenario(map_name or "A1")
    model = SharedRouteDQN(scenario.obs_dim)
    checkpoint = {
        "model_state_dict": model.state_dict(), "obs_dim": scenario.obs_dim,
        "waypoint_order": scenario.waypoint_order,
        "n_actions": ACTION_DIM, "training_roster": GORIGONS.players,
        "spike_carrier": GORIGONS.spike_holder,
    }
    if map_name is not None:
        scenario = get_scenario(map_name)
        checkpoint.update(map_name=map_name, scenario_signature=scenario.signature)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


class MapSelectionTests(unittest.TestCase):
    def test_maps_use_distinct_waypoints_sites_and_model_paths(self):
        a1, a2 = get_scenario("A1"), get_scenario("A2")
        np.testing.assert_array_equal(a1.grid, a2.grid)
        self.assertNotEqual(a1.waypoint_points, a2.waypoint_points)
        self.assertTrue(all(col < a1.grid.shape[1] // 2 for _, col in a1.plant_cells))
        self.assertTrue(all(col >= a2.grid.shape[1] // 2 for _, col in a2.plant_cells))
        self.assertEqual(a1.model_path.name, "co1_attacker_A1_best.pt")
        self.assertEqual(a2.model_path.parent.name, "attacker_A2_data")
        self.assertEqual(a2.checkpoint_filename("latest"), "co1_attacker_A2_latest.pt")

    def test_unknown_map_and_unsupported_waypoint_layout_fail_early(self):
        with self.assertRaisesRegex(ValueError, "unknown map"):
            get_scenario("A99")
        scenario = get_scenario("A1")
        with self.assertRaisesRegex(ValueError, "one or two points"):
            build_scenario("TEST", scenario.strategy_map.replace("33333", "a3333"), "left")

    def test_a3_can_be_registered_without_changing_shared_behavior(self):
        from concon_v1.co1_attacker_scenarios import _load_scenario
        _load_scenario.cache_clear()
        with patch.dict(SCENARIOS, {"A3": ScenarioSettings(
            "co1_map_attacker_A2", "right", max_candidate_bfs_distance=16,
            waypoint_order=get_scenario("A2").waypoint_order,
        )}):
            try:
                a3 = get_scenario("A3")
                self.assertEqual(a3.waypoint_points, get_scenario("A2").waypoint_points)
                self.assertEqual(a3.model_path.name, "co1_attacker_A3_best.pt")
                self.assertEqual(training.RouteEnv(0, map_name=a3).scenario.map_name, "A3")
            finally:
                # The temporary registration must not survive in the loader cache.
                from concon_v1.co1_attacker_scenarios import _load_scenario
                _load_scenario.cache_clear()

    def test_training_cli_passes_single_dash_map_to_training(self):
        for map_name in ("A1", "A2"):
            with self.subTest(map_name=map_name), patch.object(sys, "argv", [
                "co1_train_attacker.py", "-map", map_name, "--episodes", "123",
            ]), patch.object(training, "train") as train:
                training.main()
                self.assertEqual(train.call_args.kwargs["map_name"], map_name)
                self.assertEqual(train.call_args.args[:2], (123, None))

    def test_cli_rejects_unregistered_map_before_work_starts(self):
        for module in (training, evaluation):
            with self.subTest(module=module.__name__), patch.object(sys, "argv", [
                "script.py", "-map", "A99",
            ]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    module.main()
                self.assertEqual(error.exception.code, 2)

    def test_evaluation_cli_selects_map_default_model_and_freezes_it(self):
        with patch.object(sys, "argv", [
            "evaluate_co1_attacker.py", "-map", "A2", "--rounds", "1",
            "--opponents", "gc_v1",
        ]), patch.object(evaluation.Path, "read_bytes", autospec=True,
                         return_value=b"A2-weights") as read, \
                patch.object(evaluation, "evaluate", side_effect=RuntimeError("captured")) as evaluate:
            with self.assertRaisesRegex(RuntimeError, "captured"):
                evaluation.main()
            self.assertEqual(read.call_args.args[0], get_scenario("A2").model_path)
            self.assertEqual(evaluate.call_args.kwargs["map_name"].map_name, "A2")
            self.assertEqual(evaluate.call_args.kwargs["frozen_checkpoint"], b"A2-weights")

    def test_a1_and_a2_controllers_coexist_without_changing_global_map(self):
        controllers = [ConconAttackerRouteController(model=SharedRouteDQN(), map_name=name)
                       for name in ("A1", "A2")]
        for controller in controllers:
            scenario = controller.scenario
            chars = [SimpleNamespace(name=name, team="A", pos=pos)
                     for name, pos in zip(GORIGONS.players, scenario.attacker_spawns)]
            controller._prepare_round(chars)
            self.assertTrue(all(route.scenario is scenario
                                and route.goal in scenario.waypoint_points["a"]
                                for route in controller._routes.values()))
        self.assertEqual(controllers[0].scenario.map_name, "A1")

    def test_a2_team_advances_to_right_plant_cells(self):
        env = training.RouteEnv(seed=0, map_name="A2")
        scenario = env.scenario
        for marker in scenario.waypoint_order:
            self.assertTrue(all(route.goal in scenario.waypoint_points[marker]
                                for route in env.routes))
            env.positions = [route.goal for route in env.routes]
            env._advance_routes_if_reached()
        carrier_route = env.routes[SPIKE_CARRIER_INDEX]
        self.assertIn(carrier_route.goal, scenario.plant_cells)
        self.assertNotIn(carrier_route.goal, get_scenario("A1").plant_cells)
        env.positions[SPIKE_CARRIER_INDEX] = carrier_route.goal
        _, masks = env._collect()
        self.assertTrue(masks[SPIKE_CARRIER_INDEX][ACTION_PLANT])

    def test_all_a2_plant_indices_leave_coordinate_and_status_features_intact(self):
        scenario = get_scenario("A2")
        height, width = scenario.grid.shape
        route = RouteProgress(0, 0, scenario.attacker_spawns[0], scenario=scenario)
        for index, goal in enumerate(scenario.plant_cells):
            with self.subTest(index=index):
                route.set_stage(len(scenario.waypoint_order), goal, goal=goal, goal_index=index)
                observation = build_observation(route, goal, True, [], 0, 0)
                self.assertEqual(observation.shape, (scenario.obs_dim,))
                goal_offset = 9 + len(scenario.waypoint_order)
                status_offset = goal_offset + 5
                self.assertEqual(observation[goal_offset:status_offset].sum(), 1.0)
                np.testing.assert_allclose(observation[status_offset:status_offset + 4], [
                    goal[0] / (height - 1), goal[1] / (width - 1), 0.0, 1.0,
                ])
                self.assertTrue(build_action_mask(
                    scenario.grid, goal, [], True, True, goal,
                )[ACTION_PLANT])

    def test_a2_distance_limit_covers_every_existing_b_to_c_route(self):
        scenario = get_scenario("A2")
        self.assertEqual(get_scenario("A1").max_candidate_bfs_distance, 12)
        self.assertEqual(scenario.max_candidate_bfs_distance, SCENARIOS['A2'].max_candidate_bfs_distance)
        route = RouteProgress(0, 0, scenario.attacker_spawns[0], scenario=scenario)
        for point in scenario.waypoint_points["b"]:
            with self.subTest(point=point):
                route.set_stage(2, point)
                self.assertIn(route.goal, scenario.waypoint_points["c"])
                self.assertLessEqual(route.distance_map[point], scenario.max_candidate_bfs_distance)

    def test_old_a1_weights_load_and_cross_map_models_are_rejected(self):
        legacy = checkpoint_bytes()
        controller = ConconAttackerRouteController(checkpoint_bytes=legacy)
        self.assertEqual(controller.scenario.map_name, "A1")
        for blob, requested in ((legacy, "A2"), (checkpoint_bytes("A2"), "A1")):
            with self.subTest(requested=requested), self.assertRaisesRegex(ValueError, "checkpoint map"):
                ConconAttackerRouteController(checkpoint_bytes=blob, map_name=requested)
        self.assertEqual(ConconAttackerRouteController(
            checkpoint_bytes=checkpoint_bytes("A2"), map_name="A2",
        ).scenario.map_name, "A2")

    def test_changed_waypoints_or_signature_are_rejected(self):
        for metadata in ({"map_name": "A2", "scenario_signature": "old-map"},
                         {"map_name": "A2", "waypoint_points": get_scenario("A1").waypoint_points}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_checkpoint_scenario(metadata, get_scenario("A2"))

    def test_frozen_evaluation_preserves_selected_map_for_every_opponent(self):
        checkpoint = {"map_name": "A2", "training_mode": "battle",
                      "waypoint_order": get_scenario("A2").waypoint_order,
                      "opponents": ("gc_v1", "omoko_v1")}
        with patch.object(evaluation, "evaluate", return_value={
            "plants": 1, "rounds": 2, "plant_success_rate": 0.5,
        }) as evaluate, contextlib.redirect_stdout(io.StringIO()):
            training.evaluate_checkpoint(checkpoint, rounds=2)
        self.assertEqual(evaluate.call_count, 2)
        self.assertTrue(all(call.kwargs["map_name"] == "A2" for call in evaluate.call_args_list))

    def test_training_saves_map_metadata_and_uses_map_specific_default_paths(self):
        # Exercise checkpoint scheduling with finished stub rounds; no policy
        # steps, optimization, games, or real model files are produced.
        finished = SimpleNamespace(
            opponents=("gc_v1",), opponent="gc_v1", done=True, success=True,
            had_spike_drop=False, spike_recovered=False, elapsed_ticks=0,
            reset=lambda: ([], []),
        )
        for map_name in ("A1", "A2"):
            scenario = get_scenario(map_name)
            with self.subTest(map_name=map_name), \
                    patch("concon_v1.co1_battle_training.BattleRouteEnv", return_value=finished) as env, \
                    patch.object(training.Path, "mkdir"), \
                    patch.object(training.Path, "is_file", return_value=False), \
                    patch.object(training, "evaluate_checkpoint", return_value={
                        "success_rate": 1.0, "min_team_plant_rate": 1.0,
                    }), patch.object(training.torch, "save") as save, \
                    contextlib.redirect_stdout(io.StringIO()):
                training.train(episodes=1, map_name=map_name, force_save=False)
            self.assertIs(env.call_args.kwargs["map_name"], scenario)
            self.assertEqual([call.args[1] for call in save.call_args_list], [
                scenario.save_dir / scenario.checkpoint_filename("latest"),
                scenario.model_path,
            ])
            saved = save.call_args.args[0]
            self.assertEqual(saved["map_name"], map_name)
            self.assertEqual(saved["waypoint_points"], scenario.waypoint_points)
            self.assertEqual(saved["plant_cells"], scenario.plant_cells)
            self.assertEqual(saved["scenario_signature"], scenario.signature)


if __name__ == "__main__":
    unittest.main()
