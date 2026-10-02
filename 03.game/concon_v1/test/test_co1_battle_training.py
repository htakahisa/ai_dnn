import unittest
from unittest.mock import patch

import numpy as np

from iq_controller_adapter import IQAwareController
from iq_perception import PerceivedCharacter, PerceivedGameView
from concon_v1.co1_battle_training import OPPONENTS, BattleRouteEnv, plant_advantage_reward
from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_train_attacker_A1 import (
    ACTION_WAIT, MAX_TICKS, format_team_plants, summarize_team_plants,
)


class BattleTrainingTests(unittest.TestCase):
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

        with patch("concon_v1.co1_learn_attacker_A1.preplant_contact_action",
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

        with patch.object(env.game, "process_battle", side_effect=finish_by_elimination):
            env.step([ACTION_WAIT] * len(env.attackers))
        self.assertTrue(env.done)
        self.assertFalse(env.game.is_planted)
        self.assertFalse(env.success)

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
