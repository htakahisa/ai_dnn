import contextlib
import unittest
from unittest.mock import patch

import numpy as np

from iq_controller_adapter import IQAwareController
from iq_perception import PerceivedCharacter, PerceivedGameView
from concon_v1.co1_battle_training import OPPONENTS, BattleRouteEnv, plant_advantage_reward
from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_train_attacker import (
    ACTION_WAIT, MAX_TICKS, PLANT_REQUIRED_TICKS, format_team_plants, summarize_team_plants,
    format_team_round_metric, summarize_team_rounds,
)


class BattleTrainingTests(unittest.TestCase):
    def c_boundary_env(self, leader_index):
        env = BattleRouteEnv(seed=0, opponents=["touyama_v2"], map_name="A3")
        positions = [(22, 35), (22, 36), (19, 40), (18, 40), (18, 41)]
        positions[leader_index], positions[4] = positions[4], positions[leader_index]
        for char, pos in zip(env.attackers, positions):
            char.pos = list(pos)
            env.route_controller._routes[char.name].set_stage(
                2, pos, env.scenario.grid, goal=(18, 42), goal_index=0,
            )
        env.route_controller._a_completed_groups.update(route.group for route in env.routes)
        env.game.current_attacker_team_ai.perception_engine.clear_cache()
        return env

    def route_step_without_combat(self, env, actions):
        # Isolate waypoint rewards from ability delays and combat outcomes.
        def advance_clock_without_combat():
            env.game.battle_tick += 1
            env.game.round_timer -= 1

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("concon_v1.co1_learn_attacker.preplant_contact_action",
                                      return_value=None))
            stack.enter_context(patch.object(env.game, "process_battle",
                                              side_effect=advance_clock_without_combat))
            stack.enter_context(patch.object(env.game, "_move_order",
                                              return_value=list(env.game.chars)))
            for plan in (env.controller.fixed_smokes, env.controller.fixed_flashes,
                         env.controller.fixed_recons):
                stack.enter_context(patch.object(plan, "choose", return_value=None))
            return env.step(actions)

    def test_last_actor_gets_c_progress_reward_on_arrival_not_on_next_tick(self):
        env = self.c_boundary_env(4)
        actions = [ACTION_WAIT] * 5
        actions[4] = 3  # RIGHT, from (18, 41) onto c.
        transition = self.route_step_without_combat(env, actions)
        self.assertEqual(env.positions[4], (18, 42))
        self.assertEqual(env.route_controller._routes[env.attackers[4].name].stage, 2)
        self.assertEqual(np.argmax(transition[3][4][8:13]), 3)
        self.assertAlmostEqual(transition[2][4], -0.005 + 0.25)
        transition = self.route_step_without_combat(env, [ACTION_WAIT] * 5)
        self.assertEqual(np.argmax(transition[0][4][8:13]), 3)
        self.assertAlmostEqual(transition[2][4], -0.005)

    def test_c_progress_reward_does_not_depend_on_actor_order(self):
        for leader in (0, 4):
            with self.subTest(leader=leader):
                env = self.c_boundary_env(leader)
                actions = [ACTION_WAIT] * 5
                actions[leader] = 3
                transition = self.route_step_without_combat(env, actions)
                self.assertEqual(env.positions[leader], (18, 42))
                self.assertEqual(np.argmax(transition[3][leader][8:13]), 3)
                self.assertAlmostEqual(transition[2][leader], -0.005 + 0.25)

    def test_waiting_before_c_gets_no_stage_progress_reward(self):
        env = self.c_boundary_env(4)
        transition = self.route_step_without_combat(env, [ACTION_WAIT] * 5)
        self.assertEqual(env.positions[4], (18, 41))
        self.assertEqual(np.argmax(transition[3][4][8:13]), 2)
        self.assertAlmostEqual(transition[2][4], -0.005)

    def test_team_round_metrics_use_each_opponents_round_count(self):
        opponents = ("omoko_v1", "gc_v1", "fnatic_v3")
        results = [("omoko_v1", 40, False), ("gc_v1", 100, True),
                   ("omoko_v1", 80, False)]
        summary = summarize_team_rounds(opponents, results)
        recent = summarize_team_rounds(opponents, results[-2:])
        self.assertEqual(summary["omoko_v1"]["avg_ticks"], 60)
        self.assertEqual(summary["omoko_v1"]["timeouts"], 0)
        self.assertEqual(summary["gc_v1"]["avg_ticks"], 100)
        self.assertEqual(summary["gc_v1"]["timeouts"], 1)
        self.assertIsNone(summary["fnatic_v3"]["avg_ticks"])
        self.assertIn("omoko_v1=60.00(recent100=80.00)",
                      format_team_round_metric(summary, recent, "avg_ticks"))
        self.assertIn("fnatic_v3=-(recent100=-)",
                      format_team_round_metric(summary, recent, "avg_ticks"))
        self.assertIn("gc_v1=1(recent100=1)",
                      format_team_round_metric(summary, recent, "timeouts"))

    def test_team_plant_rates_use_each_opponents_played_episode_count(self):
        summary = summarize_team_plants(
            ("omoko_v1", "gc_v1", "fnatic_v3"),
            (("omoko_v1", True), ("gc_v1", False), ("omoko_v1", False)),
        )
        self.assertEqual(summary["omoko_v1"], {
            "plants": 1, "episodes": 2, "plant_rate": 0.5,
        })
        self.assertEqual(summary["gc_v1"], {
            "plants": 0, "episodes": 1, "plant_rate": 0.0,
        })
        self.assertEqual(summary["fnatic_v3"], {
            "plants": 0, "episodes": 0, "plant_rate": None,
        })
        self.assertIn("fnatic_v3=0/0(-)", format_team_plants(summary))

    def test_no_plant_gives_no_team_advantage_reward(self):
        self.assertEqual(plant_advantage_reward(5, 0), 0.0)
        self.assertEqual(plant_advantage_reward(1, 5), 0.0)

    def test_plant_reward_reflects_the_actual_team_gap(self):
        self.assertEqual(plant_advantage_reward(3, 1, planted=True), 1.0)
        self.assertEqual(plant_advantage_reward(5, 4, planted=True), 0.5)
        self.assertEqual(plant_advantage_reward(3, 5, planted=True), -1.0)

    def test_training_environment_runs_a_real_five_vs_five_tick(self):
        env = BattleRouteEnv(seed=5, opponents=["omoko_v1"])
        observations, masks = env._collect()
        self.assertEqual(len(observations), 5)
        self.assertEqual(sum(c.is_alive for c in env.game.chars if c.team == "D"), 5)
        actions = [int(np.flatnonzero(mask)[0]) for mask in masks]
        _, _, rewards, next_observations, _, done = env.step(actions)
        self.assertEqual(env.elapsed_ticks, 1)
        self.assertEqual(len(rewards), 5)
        self.assertEqual(len(next_observations), 5)
        self.assertFalse(done)
        self.assertTrue(any(env.policy_action_applied))

    def test_training_contact_decisions_receive_iq_game_views(self):
        env = BattleRouteEnv(seed=5, opponents=["omoko_v1"])
        self.assertIsInstance(env.game.attacker_controller, IQAwareController)
        seen_views = []

        def capture_contact(char, game_state, game, *args):
            seen_views.append((char, game_state, game))
            return list(char.pos)

        with patch("concon_v1.co1_learn_attacker.preplant_contact_action",
                   side_effect=capture_contact):
            env.step([ACTION_WAIT] * len(env.attackers))
        self.assertTrue(seen_views)
        self.assertTrue(all(isinstance(char, PerceivedCharacter)
                            and isinstance(view, PerceivedGameView)
                            and view.real_game is env.game
                            and game_state["chars"] is view.chars
                            for char, game_state, view in seen_views))
        self.assertFalse(any(env.policy_action_applied))

    def test_training_observation_uses_iq_ally_position(self):
        env = BattleRouteEnv(seed=5, opponents=["omoko_v1"])
        viewer, ally = env.attackers[:2]
        perception = env.game.current_attacker_team_ai.perception_engine
        original_blur = perception._blur_pos

        def move_ally_in_view(game, observer, value, maximum, channel, subject):
            if channel == "ally" and observer is viewer:
                if subject == ally.name:
                    return [viewer.pos[0] - 1, viewer.pos[1]]
                return [viewer.pos[0], viewer.pos[1] + 3]
            return original_blur(game, observer, value, maximum, channel, subject)

        with patch.object(perception, "_blur_pos", side_effect=move_ally_in_view):
            perception.clear_cache()
            observations, _ = env._collect()
        self.assertEqual(observations[0][22], 1.0)  # North has the perceived ally.
        self.assertNotEqual(tuple(ally.pos), (viewer.pos[0] - 1, viewer.pos[1]))

    def test_step_uses_the_observation_that_selected_its_actions(self):
        env = BattleRouteEnv(seed=5, opponents=["omoko_v1"])
        observations, masks = env._collect()
        with patch.object(env, "_collect", wraps=env._collect) as collect:
            old_observations, old_masks, *_ = env.step(
                [ACTION_WAIT] * len(env.attackers),
                current=(observations, masks),
            )
        self.assertIs(old_observations, observations)
        self.assertIs(old_masks, masks)
        self.assertEqual(collect.call_count, 1)  # Only the next tick is collected.

    def test_battle_episode_terminates_without_a_carrier_action(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])
        for _ in range(MAX_TICKS):
            if env.done:
                break
            env.step([ACTION_WAIT] * 5)
        self.assertTrue(env.done)
        self.assertLessEqual(env.elapsed_ticks, MAX_TICKS)

    def test_elimination_win_without_plant_is_not_plant_success(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])

        def finish_by_elimination():
            env.game.attacker_wins += 1
            env.game.round_over = True

        with (patch.object(env.game, "process_battle", side_effect=finish_by_elimination),
              patch("concon_v1.co1_learn_attacker.preplant_contact_action",
                    side_effect=lambda char, *args, **kwargs: list(char.pos))):
            _, _, rewards, _, _, _ = env.step([ACTION_WAIT] * len(env.attackers))
        self.assertTrue(env.done)
        self.assertFalse(env.game.is_planted)
        self.assertFalse(env.success)
        np.testing.assert_allclose(rewards, [7.0 - 0.005] * len(env.attackers))

    def test_failed_round_keeps_its_penalty(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])

        def finish_by_defeat():
            env.game.round_over = True

        with (patch.object(env.game, "process_battle", side_effect=finish_by_defeat),
              patch("concon_v1.co1_learn_attacker.preplant_contact_action",
                    side_effect=lambda char, *args, **kwargs: list(char.pos))):
            _, _, rewards, _, _, done = env.step([ACTION_WAIT] * len(env.attackers))
        self.assertTrue(done)
        self.assertFalse(env.success)
        np.testing.assert_allclose(rewards, [-3.0 - 0.005] * len(env.attackers))

    def test_completed_plant_ends_episode_before_round_result_and_reset_starts_next(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])
        carrier = next(char for char in env.attackers if char.has_spike)
        carrier.pos = list(env.scenario.plant_cells[0])
        move_character = env.game.move_character

        # Exercise real planting while keeping the other characters in place.
        with (patch.object(env.game.attacker_controller, "decide_move",
                           return_value=(carrier.pos, "PLANT")),
              patch.object(env.game, "move_character",
                           side_effect=lambda char: move_character(char) if char is carrier else None)):
            for _ in range(PLANT_REQUIRED_TICKS - 1):
                *_, done = env.step([ACTION_WAIT] * len(env.attackers))
                self.assertFalse(done)
                self.assertFalse(env.success)
            _, _, rewards, _, _, done = env.step([ACTION_WAIT] * len(env.attackers))

        self.assertTrue(done)
        self.assertTrue(env.success)
        self.assertTrue(env.game.is_planted)
        self.assertFalse(env.game.round_over)
        self.assertEqual(env.elapsed_ticks, PLANT_REQUIRED_TICKS)
        advantage = plant_advantage_reward(
            sum(env.alive), sum(char.is_alive for char in env.game.chars if char.team == "D"),
            planted=True,
        )
        np.testing.assert_allclose(rewards, [10.0 - 0.005 + advantage] * len(env.attackers))
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step([ACTION_WAIT] * len(env.attackers))

        previous_game = env.game
        env.reset()
        self.assertIsNot(env.game, previous_game)
        self.assertFalse(env.done)
        self.assertFalse(env.success)
        self.assertFalse(env.game.is_planted)
        self.assertEqual(env.elapsed_ticks, 0)

    def test_each_opponent_can_start_training_round(self):
        for opponent in OPPONENTS:
            with self.subTest(opponent=opponent):
                env = BattleRouteEnv(seed=2, opponents=[opponent])
                self.assertEqual(sum(env.alive), 5)
                self.assertEqual(sum(c.is_alive for c in env.game.chars
                                     if c.team == "D"), 5)
                env.step([ACTION_WAIT] * 5)
                self.assertEqual(env.elapsed_ticks, 1)

    def test_route_policy_is_not_forced_to_wait_by_exact_enemy_position(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        defender = next(c for c in env.game.chars if c.team == "D")
        defender.pos = [22, 18]
        env.game.current_attacker_team_ai.perception_engine.clear_cache()
        observations, masks = env._collect()
        self.assertEqual(observations[0][-1], 0.0)
        self.assertNotEqual(np.flatnonzero(masks[0]).tolist(), [ACTION_WAIT])

    def test_smoke_and_recon_target_the_same_engagement(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        enemy = next(c for c in env.game.chars if c.team == "D")
        enemy.pos = [22, 18]
        smoker = next(c for c in env.attackers if c.ability_name == "SMOKE")
        scout = next(c for c in env.attackers if c.ability_name == "RECON")
        smoke = choose_ability(smoker, env.game)
        self.assertEqual(smoke, {"ability": "SMOKE", "target": (22, 18)})
        self.assertTrue(env.game.execute_ai_ability(smoker, smoke))
        recon = choose_ability(scout, env.game)
        self.assertEqual(recon["ability"], "RECON")
        path = env.game._projectile_path(tuple(scout.pos), recon["target"])
        self.assertLessEqual(max(abs(path[-1][i] - (22, 18)[i]) for i in range(2)), 3)
        self.assertTrue(env.game.execute_ai_ability(scout, recon))

    def test_flash_supports_visible_ally_engagement(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        enemy = next(c for c in env.game.chars if c.team == "D")
        enemy.pos = [22, 18]
        flasher = next(c for c in env.attackers if c.ability_name == "FLASH")
        flash = choose_ability(flasher, env.game)
        self.assertEqual(flash["ability"], "FLASH")
        path = env.game._projectile_path(tuple(flasher.pos), flash["target"])
        from game_core import FLASH_MAX_FLIGHT_TICKS, FLASH_SPEED_CELLS_PER_TICK
        impact = path[min(len(path) - 1,
                          FLASH_MAX_FLIGHT_TICKS * FLASH_SPEED_CELLS_PER_TICK)]
        self.assertTrue(env.game.check_cell_line_of_sight(
            tuple(enemy.pos), impact, block_smoke=True))
        self.assertTrue(env.game.execute_ai_ability(flasher, flash))

    def test_ready_ultimate_is_used_during_contact(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        enemy = next(c for c in env.game.chars if c.team == "D")
        enemy.pos = [22, 18]
        scout = next(c for c in env.attackers if c.ultimate_name == "MONITOR")
        scout.recon_charges = 0
        scout.ultimate_points = scout.ultimate_cost
        self.assertEqual(choose_ability(scout, env.game), {"ultimate": "MONITOR"})


if __name__ == "__main__":
    unittest.main()
