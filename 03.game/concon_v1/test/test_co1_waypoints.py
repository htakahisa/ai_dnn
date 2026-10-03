"""Variable waypoint sequences share training and production route behavior."""

import contextlib
import hashlib
import io
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from concon_v1 import co1_train_attacker as training
from concon_v1.co1_attacker_common import (
    ACTION_DIM, ACTION_PLANT, GORIGONS, RouteProgress, SPIKE_CARRIER_INDEX,
    SharedRouteDQN, build_observation,
)
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, ScenarioSettings, _load_scenario, _rows, get_scenario,
    parse_strategy_points, validate_checkpoint_scenario,
)
from concon_v1.co1_learn_attacker import ConconAttackerRouteController

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1 import evaluate_co1_attacker as evaluation
    from concon_v1.co1_battle_training import BattleRouteEnv


@contextlib.contextmanager
def variable_scenario(order):
    rows = [list(row) for row in _rows(get_scenario("A1").strategy_map)]
    for row in rows:
        for index, marker in enumerate(row):
            if marker in "abcd" and marker not in order:
                row[index] = "0"
    if "e" in order:
        rows[10][3] = "e"
    strategy_map = "\n".join("".join(row) for row in rows)
    settings = ScenarioSettings("co1_map_attacker_test", "left", waypoint_order=order)
    with patch.dict(SCENARIOS, {"TEST": settings}), patch(
        "concon_v1.co1_attacker_scenarios.import_module",
        return_value=SimpleNamespace(MAZE_STR=strategy_map),
    ):
        _load_scenario.cache_clear()
        try:
            yield get_scenario("TEST")
        finally:
            _load_scenario.cache_clear()


def make_checkpoint(scenario, model=None):
    model = model if model is not None else SharedRouteDQN(obs_dim=scenario.obs_dim)
    return {
        "model_state_dict": model.state_dict(), "obs_dim": scenario.obs_dim,
        "n_actions": ACTION_DIM, "training_roster": GORIGONS.players,
        "spike_carrier": GORIGONS.spike_holder, "map_name": scenario.map_name,
        "waypoint_order": scenario.waypoint_order, "scenario_signature": scenario.signature,
    }


def serialize(checkpoint):
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


class VariableWaypointTests(unittest.TestCase):
    def test_parser_accepts_custom_sequence_and_rejects_missing_or_extra_markers(self):
        self.assertEqual(tuple(parse_strategy_points("abc\n000", "abc")), tuple("abc"))
        self.assertEqual(tuple(parse_strategy_points("abcde", "abcde")), tuple("abcde"))
        for text, order in (("abcd", "abc"), ("abcd", "abcde"), ("abc", "aac"),
                            ("abc", "bca"), ("abc", "aAbc"), ("abc", "")):
            with self.subTest(text=text, order=order), self.assertRaises(ValueError):
                parse_strategy_points(text, order)

    def test_team_reaches_plant_stage_after_the_configured_last_waypoint(self):
        for order in ("a", "abc", "abcd", "abcde"):
            with self.subTest(order=order), variable_scenario(order) as scenario:
                env = training.RouteEnv(seed=0, map_name=scenario)
                for stage, marker in enumerate(order):
                    self.assertTrue(all(route.stage == stage and not route.at_plant_stage
                                        and route.goal in scenario.waypoint_points[marker]
                                        for route in env.routes))
                    env.positions = [route.goal for route in env.routes]
                    env._advance_routes_if_reached()
                self.assertTrue(all(route.stage == len(order) and route.at_plant_stage
                                    for route in env.routes))
                route = env.routes[SPIKE_CARRIER_INDEX]
                self.assertIn(route.goal, scenario.plant_cells)
                env.positions[SPIKE_CARRIER_INDEX] = route.goal
                observations, masks = env._collect()
                self.assertTrue(masks[SPIKE_CARRIER_INDEX][ACTION_PLANT])
                self.assertTrue(all(obs.shape == (24 + len(order),) for obs in observations))

    def test_observation_stage_and_goal_features_do_not_overlap_for_variable_lengths(self):
        for order in ("abc", "abcde"):
            with self.subTest(order=order), variable_scenario(order) as scenario:
                route = RouteProgress(0, 0, scenario.attacker_spawns[0], scenario=scenario)
                status_offset = 14 + len(order)
                goal = scenario.plant_cells[2]
                route.set_stage(len(order), goal, goal=goal, goal_index=2)
                observation = build_observation(route, goal, True, [], 2, 50)
                self.assertEqual(observation[8:9 + len(order)].sum(), 1)
                self.assertEqual(observation[8 + len(order)], 1)
                self.assertEqual(observation[9 + len(order):status_offset].sum(), 1)
                np.testing.assert_allclose(observation[status_offset:status_offset + 4], [
                    goal[0] / 25, goal[1] / 43, 0, 1,
                ])
                np.testing.assert_allclose(observation[-2:], [0.5, 0.5])
                self.assertEqual(SharedRouteDQN(obs_dim=scenario.obs_dim)(
                    torch.as_tensor(observation).unsqueeze(0),
                ).shape, (1, ACTION_DIM))

    def test_variable_length_models_load_and_old_layout_is_rejected(self):
        for order in ("abc", "abcde"):
            with self.subTest(order=order), variable_scenario(order) as scenario:
                checkpoint = make_checkpoint(scenario)
                controller = ConconAttackerRouteController(
                    checkpoint_bytes=serialize(checkpoint), map_name=scenario,
                )
                self.assertEqual(controller.model.features[0].in_features, scenario.obs_dim)
                wrong_dimension = {**checkpoint, "obs_dim": 28}
                with self.assertRaisesRegex(ValueError, "observation dimensions"):
                    validate_checkpoint_scenario(wrong_dimension, scenario)
                wrong_order = {**checkpoint, "waypoint_order": "abcd"}
                with self.assertRaisesRegex(ValueError, "waypoint order"):
                    validate_checkpoint_scenario(wrong_order, scenario)

    def test_existing_abcd_signatures_and_observation_layout_are_unchanged(self):
        for map_name in ("A1", "A2"):
            scenario = get_scenario(map_name)
            contents = "\n".join(_rows(scenario.game_map) + _rows(scenario.strategy_map)
                                 + [scenario.plant_side, str(scenario.max_candidate_bfs_distance)])
            self.assertEqual(scenario.signature, hashlib.sha256(contents.encode("utf-8")).hexdigest())
            route = RouteProgress(1, 2, scenario.attacker_spawns[0], scenario=scenario)
            route.set_stage(4, (7, 3), goal=(7, 3), goal_index=2)
            observation = build_observation(route, (7, 3), True, [(7, 4)], 2, 50)
            expected = np.zeros(28, dtype=np.float32)
            expected[:2] = (7 / 25, 3 / 43)
            expected[[3, 6, 12, 15, 21, 25]] = 1
            expected[18:20] = (7 / 25, 3 / 43)
            expected[26:] = 0.5
            np.testing.assert_array_equal(observation, expected)

    def test_training_checkpoint_uses_variable_dimensions_without_training_steps(self):
        finished = SimpleNamespace(
            opponents=("gc_v1",), opponent="gc_v1", done=True, success=True,
            had_spike_drop=False, spike_recovered=False, elapsed_ticks=0,
            reset=lambda: ([], []),
        )
        for order in ("abc", "abcde"):
            with self.subTest(order=order), variable_scenario(order) as scenario, \
                    patch("concon_v1.co1_battle_training.BattleRouteEnv", return_value=finished), \
                    patch.object(training.Path, "mkdir"), patch.object(training.Path, "is_file", return_value=False), \
                    patch.object(training, "evaluate_checkpoint", return_value={
                        "success_rate": 1.0, "min_team_plant_rate": 1.0,
                    }), patch.object(training.torch, "save") as save, \
                    contextlib.redirect_stdout(io.StringIO()):
                training.train(episodes=1, map_name=scenario)
                checkpoint = save.call_args.args[0]
                self.assertEqual(checkpoint["obs_dim"], scenario.obs_dim)
                self.assertEqual(checkpoint["waypoint_order"], order)
                self.assertEqual(checkpoint["model_state_dict"]["features.0.weight"].shape[1], scenario.obs_dim)

    def test_real_training_and_evaluation_agree_for_variable_waypoint_lengths(self):
        # Deterministic frozen policy rollouts, with no optimization or model writes.
        for order in ("abc", "abcde"):
            with self.subTest(order=order), variable_scenario(order) as scenario, \
                    contextlib.redirect_stdout(io.StringIO()):
                model = SharedRouteDQN(obs_dim=scenario.obs_dim)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                    model.advantage[-1].bias[:4] = 1.0
                    model.advantage[-1].bias[ACTION_PLANT] = 2.0
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                env = BattleRouteEnv(seed=0, opponents=["gc_v1"], model=model, map_name=scenario)
                observations, _ = env._collect()
                self.assertTrue(all(obs.shape == (scenario.obs_dim,) for obs in observations))
                training_frames = []
                for _ in range(250):
                    if env.done:
                        break
                    env.step(epsilon=0)
                    training_frames.append(self.frame(env.game))
                self.assertTrue(env.done)
                evaluation_frames = []

                class TracedBattle(evaluation.LimitedRoundBattle):
                    def step_tick(game):
                        advanced = super().step_tick()
                        if advanced and game.battle_tick > 0:
                            evaluation_frames.append(self.frame(game))
                        return advanced

                with patch.object(evaluation, "LimitedRoundBattle", TracedBattle):
                    evaluation.evaluate("gc_v1", rounds=1, seed=0, map_name=scenario,
                                        frozen_checkpoint=serialize(make_checkpoint(scenario, model)))
                self.assertEqual(training_frames, evaluation_frames)

    @staticmethod
    def frame(game):
        routes = game.attacker_controller.route_controller._routes
        return (
            game.battle_tick, game.is_planted, game.round_over,
            tuple((char.name, tuple(char.pos), char.hp, char.is_alive, char.has_spike)
                  for char in game.chars),
            tuple((name, route.stage, route.goal) for name, route in routes.items()),
        )


if __name__ == "__main__":
    unittest.main()
