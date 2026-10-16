"""Deterministic HS thresholds for the escort training combat simulator."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

AI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_DIR.parent))
sys.path.insert(0, str(AI_DIR))
import ov1_train_attacker_escort as training
import ov1_train_attacker_carry as carry_training


class CarryCombatRuleTests(unittest.TestCase):
    def damage(self, *, moving=False, target_moving=False, blind=False, facing='E',
               roll=0.2, team='A'):
        env = carry_training.CarryEnv.__new__(carry_training.CarryEnv)
        shooter = carry_training.UnitStub('shooter', team, (2, 1), '', 'NONE', 1, 0, .5, 200)
        target = carry_training.UnitStub('target', 'D' if team == 'A' else 'A',
                                        (2, 4), '', 'NONE', 1, 0, 0, 0)
        shooter.facing, target.facing = facing, 'E'
        shooter.moved_this_tick, target.moved_this_tick = moving, target_moving
        shooter.blind_remaining = int(blind)
        env.attackers, env.defenders = [shooter], [target]
        env._smoke_cells = lambda: ()
        with patch.object(carry_training, 'has_los', return_value=True), \
                patch.object(carry_training.random, 'random', side_effect=[0., roll]):
            env._resolve_shots()
        return carry_training.MAX_HP - target.hp

    def test_moving_hs_threshold_for_both_teams(self):
        for team in ('A', 'D'):
            with self.subTest(team=team):
                self.assertEqual(self.damage(team=team), carry_training.MAX_HP)
                self.assertEqual(self.damage(team=team, moving=True), carry_training.BODY_DAMAGE)
                self.assertEqual(self.damage(team=team, moving=True, roll=.049), carry_training.MAX_HP)
                self.assertEqual(self.damage(team=team, moving=True, roll=.051), carry_training.BODY_DAMAGE)

    def test_facing_blind_and_target_movement_hs_modifiers(self):
        self.assertEqual(self.damage(facing='N', roll=.3), carry_training.BODY_DAMAGE)
        self.assertEqual(self.damage(blind=True, roll=.1), carry_training.BODY_DAMAGE)
        self.assertEqual(self.damage(target_moving=True, roll=.4), carry_training.BODY_DAMAGE)


class EscortCombatRuleTests(unittest.TestCase):
    def test_corner_touching_wall_blocks_training_los(self):
        grid = np.array([[0, 1], [0, 0]])
        self.assertFalse(training._has_los(grid, (), (0, 0), (1, 1)))
        self.assertFalse(training._has_los(grid, (), (1, 1), (0, 0)))

    def damage(self, kind, *, moving=False, target_moving=False, facing="E", blind=False, roll=0.2):
        env = training.EscortEnv(seed=1)
        env.reset()
        env.grid = np.zeros((8, 8), dtype=int)
        env.smokes = []
        env.carry_alive = False
        env.escort_alive = [False] * env.n_escorts
        env.enemy_alive = [False] * env.n_enemies
        env.enemy_alive[0] = True
        if kind == "enemy":
            env.escort_alive[0] = True
            env.escort_pos[0], env.enemy_pos[0] = (2, 4), (2, 1)
            env.escort_facing[0], env.enemy_facing[0] = "E", facing
            env.escort_dodge[0] = 0
            env.escort_moved[0], env.enemy_moved[0] = target_moving, moving
            env.enemy_hs_rate[0], env.enemy_accuracy[0], env.enemy_reaction[0] = 0.5, 1, 200
            env.enemy_blind_remaining[0] = int(blind)
        else:
            env.enemy_pos[0], env.enemy_facing[0], env.enemy_dodge[0] = (2, 4), "E", 0
            env.enemy_moved[0] = target_moving
            if kind == "carry":
                env.carry_alive, env.carry_pos, env.carry_facing = True, (2, 1), facing
                env.carry_moved, env.carry_hs_rate, env.carry_accuracy, env.carry_reaction = moving, 0.5, 1, 200
            else:
                env.escort_alive[0], env.escort_pos[0], env.escort_facing[0] = True, (2, 1), facing
                env.escort_moved[0], env.escort_hs_rate[0], env.escort_accuracy[0], env.escort_reaction[0] = moving, 0.5, 1, 200
        env.rng = Mock()
        # Guaranteed hit, then an HS roll; target faces away and cannot fire.
        env.rng.random.side_effect = [0.0, roll]
        env._resolve_combat()
        hp = env.escort_hp[0] if kind == "enemy" else env.enemy_hp[0]
        return 100 - hp

    def test_moving_shooter_penalty_applies_to_carry_escort_and_enemy(self):
        for kind in ("carry", "escort", "enemy"):
            with self.subTest(kind=kind):
                self.assertEqual(self.damage(kind), 100)  # lethal HS
                self.assertEqual(self.damage(kind, moving=True), training.BODY_DAMAGE)
                self.assertEqual(self.damage(kind, moving=True, roll=0.049), 100)
                self.assertEqual(self.damage(kind, moving=True, roll=0.051), training.BODY_DAMAGE)

    def test_facing_moving_target_and_blind_multiply_hs_rate(self):
        self.assertEqual(self.damage("escort", facing="N", roll=0.3), training.BODY_DAMAGE)
        self.assertEqual(self.damage("escort", target_moving=True, roll=0.4), training.BODY_DAMAGE)
        self.assertEqual(self.damage("enemy", blind=True, roll=0.1), training.BODY_DAMAGE)

    def test_recon_revealed_enemy_is_visible_through_training_smoke(self):
        env = training.EscortEnv(seed=1)
        env.reset()
        env.grid = np.zeros((8, 8), dtype=int)
        env.enemy_alive = [False] * env.n_enemies
        env.enemy_alive[0], env.enemy_pos[0] = True, (2, 5)
        env.smokes = [{"cells": {(2, 3)}, "remaining": 5}]
        self.assertIsNone(env._nearest_visible_enemy((2, 1))[0])
        env.enemy_reveal_remaining[0] = 3
        self.assertEqual(env._nearest_visible_enemy((2, 1))[0], 0)
        env.grid[2, 3] = 1
        self.assertIsNone(env._nearest_visible_enemy((2, 1))[0])


if __name__ == "__main__":
    unittest.main()
