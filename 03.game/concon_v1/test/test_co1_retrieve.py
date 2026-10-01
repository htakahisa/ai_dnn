"""Regression checks for the shared ConCon retrieve phase."""

import unittest
from unittest.mock import patch

import numpy as np

from concon_v1.co1_attacker_retrieve import ConconAttackerRetrieveController
from concon_v1.co1_attacker_abilities import choose_ability
from concon_v1.co1_battle_training import BattleRouteEnv
from concon_v1.co1_train_attacker_A1 import (
    ACTION_PLANT, ACTION_WAIT, GRID, LEFT_PLANT_CELLS, PLANT_REQUIRED_TICKS,
)


class RetrieveTests(unittest.TestCase):
    def test_retriever_moves_to_spike_and_escort_follows(self):
        class Character:
            def __init__(self, name, pos):
                self.name, self.pos = name, list(pos)
                self.team, self.is_alive = "A", True

        grid = np.zeros((5, 5), dtype=np.int32)
        near = Character("near", (2, 1))
        far = Character("far", (4, 1))
        controller = ConconAttackerRetrieveController()
        state = {"grid": grid, "chars": [near, far], "spike_pos": (2, 3)}
        self.assertEqual(controller.decide_move(near, state), [2, 2])
        self.assertNotEqual(controller.decide_move(far, state), far.pos)

    def test_shootable_enemy_stops_retrieval(self):
        class Character:
            def __init__(self, name, team, pos):
                self.name, self.team, self.pos = name, team, list(pos)
                self.is_alive, self.hp, self.facing = True, 100, "N"

        attacker = Character("attacker", "A", (2, 1))
        defender = Character("defender", "D", (2, 3))
        controller = ConconAttackerRetrieveController()
        state = {"grid": np.zeros((5, 5), dtype=np.int32),
                 "chars": [attacker, defender], "spike_pos": (4, 4)}
        position, action = controller.decide_move(attacker, state)
        self.assertEqual(position, attacker.pos)
        self.assertIn("facing", action)

    def test_drop_is_intermediate_and_pickup_restores_route_phase(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])
        carrier = next(char for char in env.attackers if char.has_spike)
        carrier.has_spike = False
        spike = next((row, col) for row in range(GRID.shape[0])
                     for col in range(GRID.shape[1])
                     if GRID[row, col] != 1 and all(tuple(c.pos) != (row, col)
                                                     for c in env.game.chars))
        env.game.spike_pos = spike
        observations, masks = env._collect()
        self.assertTrue(env.retrieve_active)
        self.assertTrue(all(np.flatnonzero(mask).tolist() == [ACTION_WAIT]
                            for mask in masks))
        with patch.object(env.game, "process_battle"):
            env.step([ACTION_WAIT] * 5)
        self.assertFalse(env.done)
        self.assertFalse(env.route_active_before_step)
        self.assertTrue(env.had_spike_drop)
        self.assertFalse(env.spike_recovered)
        carrier.has_spike = True
        env.game.spike_pos = None
        env._collect()
        self.assertFalse(env.retrieve_active)

    def test_spike_pickup_in_real_battle_returns_to_route_phase(self):
        env = BattleRouteEnv(seed=7, opponents=["omoko_v1"])
        carrier = next(char for char in env.attackers if char.has_spike)
        carrier.has_spike = False
        env.game.spike_pos = tuple(carrier.pos)
        env.step([ACTION_WAIT] * 5)
        self.assertTrue(carrier.has_spike)
        self.assertIsNone(env.game.spike_pos)
        self.assertFalse(env.retrieve_active)
        self.assertFalse(env.done)
        self.assertTrue(env.had_spike_drop)
        self.assertTrue(env.spike_recovered)

        site = LEFT_PLANT_CELLS[0]
        carrier.pos = list(site)
        carrier.plant_timer = PLANT_REQUIRED_TICKS - 1
        for char, route in zip(env.attackers, env.routes):
            goal = site if char is carrier else tuple(char.pos)
            route.set_stage(4, char.pos, goal=goal, goal_index=0)
        env.game.current_attacker_team_ai.perception_engine.clear_cache()
        actions = [ACTION_WAIT] * len(env.attackers)
        carrier_index = env.attackers.index(carrier)
        self.assertTrue(env._collect()[1][carrier_index][ACTION_PLANT])
        actions[carrier_index] = ACTION_PLANT
        with patch.object(env.game, "process_battle"):
            env.step(actions)
        self.assertTrue(env.game.is_planted)
        self.assertTrue(env.success)
        self.assertTrue(env.done)
        self.assertTrue(env.had_spike_drop)
        self.assertTrue(env.spike_recovered)

    def test_recon_targets_team_sighting_without_prior_smoke(self):
        env = BattleRouteEnv(seed=2, opponents=["omoko_v1"])
        enemy = next(char for char in env.game.chars if char.team == "D")
        enemy.pos = [22, 18]
        scout = next(char for char in env.attackers if char.ability_name == "RECON")
        recon = choose_ability(scout, env.game)
        self.assertEqual(recon["ability"], "RECON")
        path = env.game._projectile_path(tuple(scout.pos), recon["target"])
        self.assertLessEqual(max(abs(path[-1][i] - enemy.pos[i]) for i in range(2)), 4)


if __name__ == "__main__":
    unittest.main()
