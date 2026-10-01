import unittest

import numpy as np

from concon_v1.co1_battle_training import OPPONENTS, BattleRouteEnv, plant_advantage_reward
from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_train_attacker_A1 import ACTION_WAIT, MAX_TICKS


class BattleTrainingTests(unittest.TestCase):
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

    def test_battle_episode_terminates_without_a_carrier_action(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])
        for _ in range(MAX_TICKS):
            if env.done:
                break
            env.step([ACTION_WAIT] * 5)
        self.assertTrue(env.done)
        self.assertLessEqual(env.elapsed_ticks, MAX_TICKS)

    def test_each_opponent_can_start_training_round(self):
        for opponent in OPPONENTS:
            with self.subTest(opponent=opponent):
                env = BattleRouteEnv(seed=2, opponents=[opponent])
                self.assertEqual(sum(env.alive), 5)
                self.assertEqual(sum(c.is_alive for c in env.game.chars
                                     if c.team == "D"), 5)
                env.step([ACTION_WAIT] * 5)
                self.assertEqual(env.elapsed_ticks, 1)

    def test_visible_enemy_marks_the_forced_combat_wait(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        defender = next(c for c in env.game.chars if c.team == "D")
        defender.pos = [22, 18]
        observations, masks = env._collect()
        self.assertEqual(observations[0][-1], -1.0)
        self.assertEqual(np.flatnonzero(masks[0]).tolist(), [ACTION_WAIT])

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
