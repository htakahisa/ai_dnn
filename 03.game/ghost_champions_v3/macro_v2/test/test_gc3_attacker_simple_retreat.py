"""Short, staggered spawn retreats in the simple planting test logic."""
import unittest
import numpy as np
import torch
from grid_paths import distance_map
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
from ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives import SimpleDirectiveLogic,directive_policy
from ghost_champions_v3.macro_v2.gc3_rollout_attacker_directives import play_directives
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_simple_macro as simple_tests
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_directive_training as training_tests


class SimpleRetreatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,after=2,duration=3,wait=2):
        char,state,history,candidates,own=simple_tests.SimpleMacroTests().fixture()
        logic=SimpleDirectiveLogic(0,retreat_ticks=duration,retreat_after_ticks=after,retreat_min_distance=2,retreat_wait_ticks=wait)
        initial,_=logic.select_macro(char,state,history,None,candidates)
        distance=distance_map(state['grid'],tuple(own[1].pos))
        # Put one actor outside spawn on a user candidate that is not a plant
        # cell or its original destination, leaving other actors in spawn.
        far=next(candidates.coordinate(i) for i in np.flatnonzero(candidates.destination_mask)
            if distance[candidates.coordinate(i)]>=2 and state['grid'][candidates.coordinate(i)]!=2
            and candidates.coordinate(i)!=initial['destinations'][1])
        spawn=tuple(own[1].pos);own[1].pos=list(far)
        return char,state,history,candidates,own,logic,initial,spawn

    def test_advance_then_one_retreat_wait_and_readvance_without_changing_plant(self):
        char,state,history,candidates,own,logic,initial,spawn=self.fixture()
        plans=[logic.select_macro(char,state,history,None,candidates)[0] for _ in range(8)]
        self.assertNotIn('RETREAT',plans[0]['instructions'])
        self.assertEqual([p['instructions'][1] for p in plans[1:7]],['RETREAT','RETREAT','RETREAT','WAIT','WAIT','ADVANCE'])
        for plan in plans:
            self.assertLessEqual(plan['instructions'].count('RETREAT'),1)
            self.assertEqual(plan['plant'],initial['plant'])
            self.assertEqual(plan['destinations'],initial['destinations'])
        self.assertEqual(logic.spawn_positions[str(own[1].name)],spawn)
        self.assertEqual(logic.retreat_events[0]['reason'],'timeout')
        self.assertEqual(logic.retreat_events[0]['end']-logic.retreat_events[0]['start'],3)
        for _ in range(20):logic.select_macro(char,state,history,None,candidates)
        self.assertEqual(len(logic.retreat_events),1)

    def test_reaching_spawn_ends_retreat_early_and_reset_clears_practice(self):
        char,state,history,candidates,own,logic,initial,spawn=self.fixture(after=1,duration=12)
        plan,_=logic.select_macro(char,state,history,None,candidates)
        self.assertEqual(plan['instructions'][1],'RETREAT')
        own[1].pos=list(spawn)
        plan,_=logic.select_macro(char,state,history,None,candidates)
        self.assertEqual(plan['instructions'][1],'WAIT')
        self.assertEqual(logic.retreat_events[0]['reason'],'reached_spawn')
        logic.reset_round()
        self.assertFalse(logic.retreated);self.assertFalse(logic.retreat_events);self.assertIsNone(logic.active_retreat)

    def test_setup_does_not_consume_battle_warmup_or_issue_retreat(self):
        char,state,history,candidates,own,logic,initial,spawn=self.fixture(after=3)
        old_age=logic.battle_age
        state['defender_setup_active']=True
        for _ in range(20):
            plan,_=logic.select_macro(char,state,history,None,candidates)
            self.assertNotIn('RETREAT',plan['instructions'])
        self.assertEqual(logic.battle_age,old_age)
        state['defender_setup_active']=False
        for _ in range(2):
            plan,_=logic.select_macro(char,state,history,None,candidates)
            self.assertNotIn('RETREAT',plan['instructions'])
        self.assertIn('RETREAT',logic.select_macro(char,state,history,None,candidates)[0]['instructions'])

    def test_objectives_and_planted_round_are_protected(self):
        char,state,history,candidates,own,logic,initial,spawn=self.fixture(after=1)
        own[1].plant_timer=1
        self.assertNotIn('RETREAT',logic.select_macro(char,state,history,None,candidates)[0]['instructions'])
        own[1].plant_timer=0
        self.assertEqual(logic.select_macro(char,state,history,None,candidates)[0]['instructions'][1],'RETREAT')
        own[1].orb_collect_timer=1
        self.assertNotEqual(logic.select_macro(char,state,history,None,candidates)[0]['instructions'][1],'RETREAT')
        self.assertEqual(logic.retreat_events[0]['reason'],'objective')
        state['is_planted']=True
        own[1].orb_collect_timer=0
        for _ in range(10):self.assertNotIn('RETREAT',logic.select_macro(char,state,history,None,candidates)[0]['instructions'])

    def test_real_episode_supplies_retreat_decisions_without_weight_updates(self):
        _,state,_,candidates,_=simple_tests.SimpleMacroTests().fixture()
        policy=DirectedMicroPolicy.from_micro(fresh_micro_policy());guard=FreshPolicy.fresh('attacker',9)
        cfg=dict(training_tests.config(),MACRO_DIRECTIVE_MODE='simple',SIMPLE_RETREAT_AFTER_TICKS=0,
            SIMPLE_RETREAT_MIN_DISTANCE=1,SIMPLE_RETREAT_TICKS=3,SIMPLE_RETREAT_WAIT_TICKS=1)
        before=module_digest([policy,guard])
        rows,stats=play_directives('FNC',70,policy=policy,guard=guard,candidates=candidates,config=cfg)
        self.assertGreater(stats['instructions'].get('RETREAT',0),0)
        self.assertTrue(stats['retreat_practice_events'])
        self.assertEqual(before,module_digest([policy,guard]))
        selected=[r for r in rows if r['macro_directive']['instruction']=='RETREAT'][:4]
        logp,_,_=policy.evaluate_records(selected)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in selected]),atol=2e-5,rtol=2e-5)
        for row in selected:self.assertAlmostEqual(row['reward'],sum(row['reward_components'].values()),places=5)

    def test_retreat_practice_requires_retreat_measurements_for_readiness(self):
        from unittest.mock import patch
        from ghost_champions_v3.macro_v2 import gc3_training_attacker_directives as training
        cfg=dict(training_tests.config(),MACRO_DIRECTIVE_MODE='simple',OPPONENT_CODES=('FNC',),
            EVALUATION_ROUNDS_PER_OPPONENT=1,SIMPLE_RETREAT_TICKS=12)
        stats=dict(success=True,planted=True,orb_collections=0,planned_plant_distance=0,
            instructions={'WAIT':10,'ADVANCE':10,'RETREAT':0},
            instruction_measures={'WAIT_applicable':10,'WAIT_compliant':10,'ADVANCE_applicable':10,'ADVANCE_compliant':10})
        with patch.object(training,'play_directives',return_value=([],stats)):
            report=training.evaluate(None,None,None,cfg)
            self.assertFalse(report['scenario_instruction_ready']);self.assertFalse(report['instruction_ready'])
            self.assertEqual(report['required_instructions'],['WAIT','ADVANCE','RETREAT'])
            disabled=training.evaluate(None,None,None,dict(cfg,SIMPLE_RETREAT_TICKS=0))
            self.assertTrue(disabled['scenario_instruction_ready']);self.assertFalse(disabled['instruction_ready'])
            stats['instruction_measures'].update(RETREAT_applicable=10,RETREAT_compliant=10)
            measured=training.evaluate(None,None,None,cfg)
            self.assertTrue(measured['scenario_instruction_ready']);self.assertTrue(measured['instruction_ready'])

    def test_bad_retreat_settings_fail_before_loading_models(self):
        for cfg in [dict(SIMPLE_RETREAT_TICKS=-1),dict(SIMPLE_RETREAT_MIN_DISTANCE=0),dict(SIMPLE_RETREAT_WAIT_TICKS=-1),
                    dict(SIMPLE_RETREAT_MIN_DISTANCE=6,SIMPLE_RETREAT_MAX_DISTANCE=5)]:
            with self.assertRaises(ValueError):directive_policy(dict(cfg,MACRO_DIRECTIVE_MODE='simple'),0)

    def test_only_short_reachable_returns_are_selected_and_logged(self):
        char,state,history,candidates,own= simple_tests.SimpleMacroTests().fixture()
        logic=SimpleDirectiveLogic(0,retreat_ticks=20,retreat_after_ticks=0,retreat_min_distance=6,
            retreat_max_distance=10)
        initial,_=logic.select_macro(char,state,history,None,candidates)
        distances=distance_map(state['grid'],tuple(own[1].pos))
        far=next(tuple(map(int,p)) for p in np.argwhere((distances>10)&(state['grid']!=1)&(state['grid']!=2)))
        own[1].pos=list(far)
        self.assertNotIn('RETREAT',logic.select_macro(char,state,history,None,candidates)[0]['instructions'])
        near=next(tuple(map(int,p)) for p in np.argwhere((distances>=6)&(distances<=10)&(state['grid']!=2)))
        own[1].pos=list(near)
        plan,_=logic.select_macro(char,state,history,None,candidates)
        self.assertEqual(plan['instructions'][1],'RETREAT')
        self.assertEqual(plan['plant'],initial['plant'])
        self.assertEqual(logic.retreat_events[0]['start_distance'],int(distances[near]))

    def test_new_defaults_expose_retreat_in_a_real_episode(self):
        _,state,_,candidates,_=simple_tests.SimpleMacroTests().fixture()
        policy=DirectedMicroPolicy.from_micro(fresh_micro_policy());guard=FreshPolicy.fresh('attacker',9)
        cfg=dict(training_tests.config(),MACRO_DIRECTIVE_MODE='simple')
        before=module_digest([policy,guard])
        rows,stats=play_directives('FNC',70,policy=policy,guard=guard,candidates=candidates,config=cfg)
        self.assertGreater(stats['instructions'].get('RETREAT',0),0)
        self.assertTrue(stats['retreat_practice_events'])
        for event in stats['retreat_practice_events']:
            self.assertGreaterEqual(event['start_distance'],6);self.assertLessEqual(event['start_distance'],10)
        self.assertEqual(before,module_digest([policy,guard]))


if __name__=='__main__':unittest.main()
