"""A fixed planting plan, user candidates, arrival holds and frozen real rollouts."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from grid_paths import distance_map
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
from ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives import SimpleDirectiveLogic,directive_policy
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
from ghost_champions_v3.macro_v2.gc3_rollout_attacker_directives import play_directives
from ghost_champions_v3.macro_v2.gc3_training_attacker_directives import evaluation_rank
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_macro_orders as orders
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_directive_training as training_tests


class SimpleMacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        char,state,ctrl=orders.MacroOrdersTests().controller()
        candidates=MacroCandidates.configured(state['grid'])
        own=sorted((c for c in state['chars'] if c.team=='A'),key=lambda c:str(c.name))
        for unit in own:unit.has_spike=False
        own[0].has_spike=True
        return char,state,ctrl.history,candidates,own

    def test_sites_alternate_and_plant_and_destinations_stay_in_user_maps(self):
        char,state,history,candidates,own=self.fixture()
        for seed in (0,1):
            logic=SimpleDirectiveLogic(seed)
            plan,row=logic.select_macro(char,state,history,None,candidates)
            self.assertEqual(int(plan['plant'][1]>=state['grid'].shape[1]//2),seed)
            self.assertTrue(candidates.plant_mask[np.ravel_multi_index(plan['plant'],state['grid'].shape)])
            self.assertEqual(plan['destinations'][0],plan['plant'])
            self.assertEqual(len(set(plan['destinations'])),5)
            for target in plan['destinations']:
                self.assertTrue(candidates.destination_mask[np.ravel_multi_index(target,state['grid'].shape)])
            for _ in range(50):
                current,_=logic.select_macro(char,state,history,plan,candidates)
                self.assertEqual(current,plan)
                self.assertNotIn('RETREAT',current['instructions'])
            logic.reset_round()
            self.assertEqual(logic.select_macro(char,state,history,None,candidates)[0],plan)

    def test_arrived_escorts_wait_and_carrier_keeps_advance_for_planting(self):
        char,state,history,candidates,own=self.fixture()
        logic=SimpleDirectiveLogic(1)
        plan,_=logic.select_macro(char,state,history,None,candidates)
        for unit,target in zip(own,plan['destinations']):unit.pos=list(target)
        current,_=logic.select_macro(char,state,history,plan,candidates)
        self.assertEqual(current['instructions'],['ADVANCE','WAIT','WAIT','WAIT','WAIT'])
        self.assertEqual(current['plant'],plan['plant'])

    def test_new_carrier_receives_plant_destination_and_dead_slots_have_no_order(self):
        char,state,history,candidates,own=self.fixture()
        logic=SimpleDirectiveLogic(0)
        plan,_=logic.select_macro(char,state,history,None,candidates)
        own[0].has_spike=False;own[0].is_alive=False;own[2].has_spike=True
        current,row=logic.select_macro(char,state,history,plan,candidates)
        self.assertIsNone(current['destinations'][0]);self.assertIsNone(current['instructions'][0])
        self.assertEqual(current['destinations'][2],plan['plant']);self.assertEqual(current['instructions'][2],'ADVANCE')
        self.assertFalse(row['active'][0]);self.assertEqual(current['plant'],plan['plant'])

    def test_plant_plan_does_not_read_enemy_live_positions(self):
        char,state,history,candidates,own=self.fixture()
        class Enemy:
            team='D'
            @property
            def pos(self):raise AssertionError('Hidden enemy position accessed')
        state['chars']=[c for c in state['chars'] if c.team=='A']
        # The public encoder intentionally reads only allowed perceived enemy
        # fields. Test planning before it is called, using a harmless stand-in.
        from unittest.mock import patch
        state['chars'].append(Enemy())
        with patch('ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives.encode_public',return_value=np.zeros(12000)):
            plan,_=SimpleDirectiveLogic(0).select_macro(char,state,history,None,candidates)
        self.assertEqual(plan['destinations'][0],plan['plant'])

    def test_real_rollout_uses_simple_logic_and_keeps_weights_unchanged(self):
        _,state,_,candidates,_=self.fixture()
        policy=DirectedMicroPolicy.from_micro(fresh_micro_policy());guard=FreshPolicy.fresh('attacker',9)
        cfg=dict(training_tests.config(),MACRO_DIRECTIVE_MODE='simple',SIMPLE_RETREAT_TICKS=0)
        self.assertIsInstance(directive_policy(cfg,0),SimpleDirectiveLogic)
        before=module_digest([policy,guard])
        rows,stats=play_directives('FNC',70,policy=policy,guard=guard,candidates=candidates,config=cfg)
        self.assertTrue(rows);self.assertEqual(stats['macro_directive_mode'],'simple')
        self.assertNotIn('RETREAT',stats['instructions']);self.assertEqual(module_digest([policy,guard]),before)
        self.assertEqual(len({tuple(r['macro_directive']['plant']) for r in rows}),1)
        self.assertEqual(stats['target_site'],'A')
        logp,_,values=policy.evaluate_records(rows[:4])
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in rows[:4]]),atol=2e-5,rtol=2e-5)

    def test_simple_best_selection_preserves_win_and_plant_priority(self):
        cfg=dict(MACRO_DIRECTIVE_MODE='simple')
        rates={'WAIT':.9,'ADVANCE':.9,'RETREAT':0.}
        failed=dict(actual_round_win_rate=0.,plant_rate=0.,instruction_compliance_rates=rates,tested_instructions=['WAIT','ADVANCE'])
        planted=dict(failed,plant_rate=.5,instruction_compliance_rates={'WAIT':.5,'ADVANCE':.5,'RETREAT':0.})
        self.assertGreater(evaluation_rank(planted,cfg),evaluation_rank(failed,cfg))


if __name__=='__main__':unittest.main()
