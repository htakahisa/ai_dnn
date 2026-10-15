"""Learn conservation and IQ-based counters; preserve ordinary actions under seals."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from concon_v1.co1_guard_common import ACTION_DIM, WAIT_ACTION, ULTIMATE_ACTION, SELF_SMOKE_ACTION, bfs_distance_map
from concon_v1.co1_guard_utility_training import learning_masks, preference_loss
from concon_v1.co1_guard_battle_training import GuardBattleEnv


class GuardResourceLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_preference_gradient_lowers_quiet_cast_and_raises_useful_counter(self):
        values = torch.zeros((2,ACTION_DIM),requires_grad=True)
        masks = np.zeros((2,3,ACTION_DIM),dtype=bool)
        masks[:,0,WAIT_ACTION] = True
        masks[0,1,40] = True
        masks[1,2,64] = True
        preference_loss(values,masks).backward()
        self.assertGreater(float(values.grad[0,40]),0)
        self.assertLess(float(values.grad[1,64]),0)
        # Movement/cover Q values are not pushed down to teach these choices.
        self.assertEqual(float(values.grad[:,WAIT_ACTION].abs().sum()),0)
        empty = torch.zeros((1,ACTION_DIM),requires_grad=True)
        self.assertEqual(float(preference_loss(empty,np.zeros((1,3,ACTION_DIM),bool))),0)

    def test_counter_teacher_uses_only_recent_disclosed_sightings(self):
        char = NS(pos=[2,2])
        game = NS(_projectile_path=lambda a,b:[a,b],
                  check_cell_line_of_sight=lambda a,b,**kw:a==(2,4))
        controller = NS(game=game,sightings={})
        context = dict(position=(2,2),targets=((2,4),(4,2),None),ultimate_actions={},
            tap=False,fireable=False,impaired=True,blind=True,revealed=False,
            exposures={(2,2):1},tick=20)
        mask = np.zeros(ACTION_DIM,bool)
        mask[WAIT_ACTION] = mask[64] = True
        self.assertFalse(learning_masks(controller,char,mask,context)[2].any())
        controller.sightings = {'enemy':((2,4),19)}
        self.assertTrue(learning_masks(controller,char,mask,context)[2,64])
        controller.sightings = {'enemy':((2,4),0)}
        self.assertFalse(learning_masks(controller,char,mask,context)[2].any())

    def test_revealed_actor_is_not_taught_that_self_smoke_is_cover(self):
        char = NS(pos=[2,2])
        controller = NS(game=NS(),sightings={})
        context = dict(targets=((2,4),(4,2),None),ultimate_actions={},tap=False,
            fireable=False,impaired=True,blind=True,revealed=True,exposures={(2,2):1},tick=1)
        mask = np.zeros(ACTION_DIM,bool)
        mask[WAIT_ACTION] = mask[SELF_SMOKE_ACTION] = True
        self.assertFalse(learning_masks(controller,char,mask,context)[2].any())
        context['revealed'] = False
        self.assertTrue(learning_masks(controller,char,mask,context)[2,SELF_SMOKE_ACTION])

    def test_reserve_starts_with_all_survivor_charges_and_real_iq(self):
        for side in ('L','R'):
            env = GuardBattleEnv(seed=4,map_name=side,opponents=['omoko_v1'])
            env.reset('reserve',attacker_count=5,defender_count=5)
            distance = bfs_distance_map(env.game.grid,tuple(env.game.planted_pos))
            for char in env.attackers:
                if char.ability_name in ('SMOKE','FLASH','RECON'):
                    self.assertGreater(getattr(char,char.ability_name.lower()+'_charges'),0)
                self.assertLessEqual(distance[tuple(char.pos)],3)
            self.assertTrue(all(8<=distance[tuple(c.pos)]<=16 for c in env.defenders))
            env.step(epsilon=0)
            self.assertEqual(env.metrics['ultimate_ready_decisions'],env.metrics['ultimate_decisions'])

    def test_sealed_ready_ultimate_does_not_remove_legal_movement(self):
        env = GuardBattleEnv(seed=2,opponents=['omoko_v1'])
        env.reset('hold',attacker_count=5,defender_count=5)
        char = env.attackers[2]
        char.ultimate_points = char.ultimate_cost
        char.ability_seal_remaining = 2
        state = dict(grid=env.game.grid,chars=env.game.chars,is_planted=True,
                     planted_pos=env.game.planted_pos,battle_tick=0,detonate_timer=50)
        _,mask,context = env.controller.policy_inputs(char,state)
        self.assertTrue(mask[:40].any())
        self.assertFalse(mask[40:].any())
        self.assertFalse(context['ultimate_actions'])
