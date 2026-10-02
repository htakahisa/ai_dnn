"""Compare frozen training rollouts with fresh production evaluation games."""

import contextlib
import io
from itertools import product
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

with contextlib.redirect_stdout(io.StringIO()):
    from battle_logic import BattleLogicMixin
    from concon_v1.co1_battle_training import BattleRouteEnv, OPPONENTS
    from concon_v1.co1_attacker_scenarios import get_scenario
    from concon_v1 import evaluate_co1_attacker as evaluation
    from concon_v1.co1_train_attacker import (
        ACTION_DIM, ACTION_PLANT, ACTION_WAIT, GORIGONS, SharedRouteDQN,
    )


def game_frame(game):
    route = game.attacker_controller.route_controller
    return (
        game.current_round, game.battle_tick, game.round_timer,
        game.is_planted, game.is_defused, game.round_over,
        game.attacker_wins, game.defender_wins,
        tuple(game.spike_pos) if game.spike_pos is not None else None,
        tuple((char.name, tuple(char.pos), char.hp, char.is_alive,
               char.has_spike, char.facing, char.plant_timer, char.ultimate_points)
              for char in game.chars),
        tuple((name, progress.stage, progress.goal)
              for name, progress in route._routes.items()),
    )


class EnvironmentParityTests(unittest.TestCase):
    def test_training_and_evaluation_match_until_plant_or_round_end_for_all_opponents(self):
        # Prefer planting when legal, then a BFS-progress move, then waiting.
        # This creates a reproducible policy without relying on a trained artifact.
        for map_name, opponent in product(("A1", "A2"), OPPONENTS):
            scenario = get_scenario(map_name)
            model = SharedRouteDQN(obs_dim=scenario.obs_dim)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.advantage[-1].bias[:4] = 1.0
                model.advantage[-1].bias[ACTION_PLANT] = 2.0
            checkpoint = {
                "model_state_dict": model.state_dict(), "obs_dim": scenario.obs_dim,
                "n_actions": ACTION_DIM, "training_roster": GORIGONS.players,
                "spike_carrier": GORIGONS.spike_holder, "map_name": map_name,
                "waypoint_order": scenario.waypoint_order,
                "scenario_signature": scenario.signature,
            }
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            frozen = buffer.getvalue()
            with self.subTest(map_name=map_name, opponent=opponent), contextlib.redirect_stdout(io.StringIO()):
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                env = BattleRouteEnv(seed=0, opponents=[opponent], model=model, map_name=map_name)
                training_frames = []
                for _ in range(250):
                    if env.done:
                        break
                    env.step(epsilon=0.0)
                    training_frames.append(game_frame(env.game))
                self.assertTrue(env.done)
                self.assertEqual(env.game.current_round, 1)
                evaluation_frames = []

                class TracedBattle(evaluation.LimitedRoundBattle):
                    def step_tick(self):
                        advanced = super().step_tick()
                        if advanced and self.battle_tick > 0:
                            evaluation_frames.append(game_frame(self))
                        return advanced

                with patch.object(evaluation, "LimitedRoundBattle", TracedBattle):
                    result = evaluation.evaluate(opponent, rounds=1, seed=0,
                                                 frozen_checkpoint=frozen, map_name=map_name)
                # Evaluation plays the full round; training stops at the plant.
                if env.success:
                    self.assertGreaterEqual(len(evaluation_frames), len(training_frames))
                    self.assertTrue(training_frames[-1][3])  # Plant completed on the last tick.
                    self.assertFalse(any(frame[3] for frame in training_frames[:-1]))
                else:
                    self.assertEqual(len(training_frames), len(evaluation_frames))
                for tick, (training_frame, evaluation_frame) in enumerate(
                        zip(training_frames, evaluation_frames), 1):
                    self.assertEqual(training_frame, evaluation_frame,
                                     f"{opponent}: different state at tick {tick}")
                self.assertEqual(result["plants"], int(env.success))
                if not env.success:
                    self.assertEqual(result["attacker_wins"], env.game.attacker_wins)

    def test_bootstrap_collection_does_not_mutate_live_policy_or_perception(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        perception = env.game.current_attacker_team_ai.perception_engine
        rng_state = env.route_rng.getstate()
        cache = dict(perception._cache)
        self.assertEqual(env.route_controller._routes, {})
        env._collect()
        env._collect()
        self.assertEqual(env.route_controller._routes, {})
        self.assertEqual(env.route_rng.getstate(), rng_state)
        self.assertEqual(perception._cache, cache)
        env.step(epsilon=0.0)
        routes = {name: (route.stage, route.goal)
                  for name, route in env.route_controller._routes.items()}
        live_maps = {name: route.distance_map
                     for name, route in env.route_controller._routes.items()}
        saved_maps = {name: distances.copy() for name, distances in live_maps.items()}
        cache = dict(perception._cache)
        rng_state = env.route_rng.getstate()
        env._collect()
        self.assertEqual({name: (route.stage, route.goal)
                          for name, route in env.route_controller._routes.items()}, routes)
        self.assertEqual(env.route_rng.getstate(), rng_state)
        self.assertEqual(perception._cache, cache)
        for name, live_route in env.route_controller._routes.items():
            self.assertIs(live_route.distance_map, live_maps[name])
            np.testing.assert_array_equal(live_route.distance_map, saved_maps[name])
            self.assertIsNot(env._preview_routes[name], live_route)
            self.assertIs(env._preview_routes[name].scenario, live_route.scenario)

    def test_policy_replay_uses_inputs_at_the_actual_decision(self):
        with contextlib.redirect_stdout(io.StringIO()):
            env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        recorded = {}
        choose = env.route_controller._choose_policy_action

        def capture(char, observation, mask):
            recorded[env.attacker_indices[char.name]] = observation.copy(), mask.copy()
            return choose(char, observation, mask)

        with patch.object(env.route_controller, "_choose_policy_action", side_effect=capture):
            old_obs, old_masks, *_ = env.step(epsilon=0.0)
        self.assertTrue(recorded)
        for index, (observation, mask) in recorded.items():
            np.testing.assert_array_equal(old_obs[index], observation)
            np.testing.assert_array_equal(old_masks[index], mask)
            self.assertTrue(env.policy_action_applied[index])
            self.assertTrue(mask[env.actions[index]])

    def test_shared_tick_applies_hp_effect_and_clears_occupancy_before_combat(self):
        events = []
        char = SimpleNamespace(is_alive=True, hp=2, carnal_lust_syndicate_active=True)
        game = SimpleNamespace(
            round_over=False, match_over=False, headless=True, chars=[char],
            defender_setup_phase=SimpleNamespace(active=False),
            _prepare_team_controllers_tick=lambda: events.append("prepare"),
            _build_occupancy_counts=lambda: events.append("build"),
            _move_order=lambda: [char],
            move_character=lambda actor: events.append(("move", actor.hp)),
            _clear_occupancy_counts=lambda: events.append("clear"),
            process_battle=lambda: events.append("combat"),
            _advance_combo_announcement=lambda: events.append("announcement"),
            _record_replay_frame=lambda: events.append("replay"),
        )
        self.assertTrue(BattleLogicMixin.step_tick(game))
        self.assertEqual(events, ["prepare", "build", ("move", 1), "clear", "combat",
                                  "announcement", "replay"])
        self.assertEqual(char.hp, 1)


if __name__ == "__main__":
    unittest.main()
