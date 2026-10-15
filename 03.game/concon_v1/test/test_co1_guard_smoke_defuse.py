"""Smoke defuse overrides work with existing models and IQ disclosures only."""

from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from concon_v1.co1_guard_battle_training import GuardBattleEnv
from concon_v1.co1_guard_common import WAIT_ACTION, ULTIMATE_ACTION, bfs_distance_map, aim_alignment


class SmokeDefuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self,site='L'):
        env = GuardBattleEnv(map_name=site,opponents=['omoko_v1'])
        env.reset('hold',attacker_count=5,defender_count=5)
        char = env.attackers[2]
        char.ultimate_points = 0
        char.smoke_charges = char.flash_charges = char.recon_charges = 0
        spike = tuple(env.game.planted_pos)
        distance = bfs_distance_map(env.game.grid,spike)
        char.pos = list(map(int,np.argwhere(distance==4)[0]))
        enemy = NS(name='hidden',team='D',pos=[-1,-1],is_alive=True,position_known=False)
        state = dict(grid=env.game.grid,chars=[char,enemy],is_planted=True,
            planted_pos=spike,battle_tick=10,detonate_timer=40,
            smoke_cells={spike},defender_defuse_info={'hidden':(1,6)})
        with torch.no_grad():
            for parameter in env.model.parameters():
                parameter.zero_()
            env.model.head[-1].bias[WAIT_ACTION:WAIT_ACTION+8] = 1000
        return env,char,state,distance

    def test_wait_preferring_existing_model_still_approaches_on_both_sites(self):
        for site in ('L','R'):
            env,char,state,distance = self.fixture(site)
            destination,_ = env.controller.decide_move(char,state)
            self.assertLess(distance[tuple(destination)],distance[tuple(char.pos)])
            _,mask,context = env.controller.policy_inputs(char,state)
            self.assertTrue(context['smoke_defuse_approach'])
            self.assertFalse(mask[32:].any())

    def test_adjacent_actor_faces_spike_instead_of_walking_away(self):
        env,char,state,distance = self.fixture()
        char.pos = list(map(int,np.argwhere(distance==1)[0]))
        destination,payload = env.controller.decide_move(char,state)
        self.assertEqual(tuple(destination),tuple(char.pos))
        self.assertGreater(aim_alignment(char.pos,state['planted_pos'],payload['facing']),.99)

    def test_no_tap_or_no_smoke_keeps_model_decision(self):
        for field,value in [('defender_defuse_info',{}),('smoke_cells',set())]:
            env,char,state,_ = self.fixture()
            state[field] = value
            destination,_ = env.controller.decide_move(char,state)
            self.assertEqual(tuple(destination),tuple(char.pos))

    def test_hidden_live_coordinates_do_not_change_approach(self):
        env,char,state,_ = self.fixture()
        _,first,_ = env.controller.policy_inputs(char,state)
        state['chars'][-1].pos = list(char.pos)
        _,second,_ = env.controller.policy_inputs(char,state)
        np.testing.assert_array_equal(first,second)

    def test_ready_ultimate_keeps_requested_ultimate_priority(self):
        env,char,state,_ = self.fixture()
        char.ultimate_points = char.ultimate_cost
        _,mask,_ = env.controller.policy_inputs(char,state)
        self.assertFalse(mask[:ULTIMATE_ACTION].any())
        self.assertTrue(mask[ULTIMATE_ACTION:].any())

    def test_visible_non_defusing_enemy_does_not_cancel_approach(self):
        env,char,state,_ = self.fixture()
        enemy = env.defenders[0]
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            cell = (char.pos[0]+dr,char.pos[1]+dc)
            if 0<=cell[0]<state['grid'].shape[0] and 0<=cell[1]<state['grid'].shape[1] and state['grid'][cell]!=1:
                enemy.pos = list(cell)
                break
        enemy.position_known = True
        state['chars'].append(enemy)
        _,mask,context = env.controller.policy_inputs(char,state)
        self.assertTrue(context['fireable'])
        self.assertFalse(context['defuser_fireable'])
        self.assertTrue(context.get('smoke_defuse_approach',False))
        self.assertFalse(mask[32:].any())
