"""Warm-start lineage, frozen foundation, spawn retreat, credits and safe preflight."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards
from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import optimize_history,rng_state
from ghost_champions_v3.macro_v2 import gc3_train_attacker_micro_directives as settings
from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
from ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives import DirectiveSampler
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy,directive_features
from ghost_champions_v3.macro_v2.gc3_reward_attacker_directives import DirectiveRewards
from ghost_champions_v3.macro_v2.gc3_rollout_attacker_directives import play_directives
from ghost_champions_v3.macro_v2.gc3_training_attacker_directives import initialize,base_digest,train,trainable_directives
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_macro_orders as order_tests


def config():
    result={key:value for key,value in vars(settings).items() if key.isupper()}
    result.update(DIRECTIVE_CHECKPOINT=None,ORDER_WARMUP_TICKS=0,MACRO_DIRECTIVE_MODE='random')
    return result


class DirectiveTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_warm_start_keeps_completed_weights_and_uses_only_new_optimizer_params(self):
        char,state,ctrl=order_tests.MacroOrdersTests().controller()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);source=fresh_micro_policy();source.save(root/'source.pt',trained=True,stage='rl',step=100175)
            guard=FreshPolicy.fresh('attacker',2);guard.save(root/'guard.pt',trained=True)
            data=root/'data';data.mkdir()
            cfg=dict(config(),INITIAL_CHECKPOINT=root/'source.pt',POSTPLANT_CHECKPOINT=root/'guard.pt',RUN_PATH=data,LOG_PATH=root/'logs')
            model,anchor,optimizer,saved,inherited=initialize(cfg,data,ctrl.candidates)
            self.assertEqual(inherited,100175);self.assertIsNone(saved);self.assertFalse(optimizer.state)
            self.assertEqual(base_digest(model),base_digest(source));self.assertEqual(module_digest([source]),module_digest([anchor]))
            self.assertEqual({name for name,p in model.named_parameters() if p.requires_grad},
                {'directive.0.weight','directive.0.bias','directive.2.weight','directive.2.bias'})
            initial=torch.load(data/'initial.pt',weights_only=True)
            self.assertEqual(initial['step'],100175);self.assertEqual(initial['additional_step'],0)
            self.assertFalse(initial['directives_trained'])
            self.assertEqual(json.loads((data/'manifest.json').read_text())['inherited_step'],100175)
            model.save(data/'latest.pt',stage='directive_rl',trained=False,step=100175,inherited_step=100175,
                additional_step=0,optimizer=optimizer.state_dict(),**rng_state())
            continued=initialize(dict(cfg,RESUME=True,ADDITIONAL_TRAINING_STEPS=30000),data,ctrl.candidates)
            self.assertEqual(base_digest(continued[0]),base_digest(model))
            with self.assertRaisesRegex(RuntimeError,'source/settings/map/source models changed'):
                initialize(dict(cfg,RESUME=True,ORDER_DURATION_TICKS=12),data,ctrl.candidates)

    def test_sampler_holds_instructions_then_changes_them_and_keeps_plant_fixed(self):
        char,state,ctrl=order_tests.MacroOrdersTests().controller()
        sampler=DirectiveSampler(43,3,(.2,.5,.3));history=MatchHistory();history.begin_round()
        plans=[sampler.select_macro(char,state,history,None,ctrl.candidates)[0] for _ in range(9)]
        self.assertEqual(plans[0],plans[1]);self.assertEqual(plans[1],plans[2])
        self.assertNotEqual(plans[0]['instructions'],plans[3]['instructions'])
        self.assertEqual(len({p['plant'] for p in plans}),1)
        sampler.reset_round()
        self.assertEqual(plans[0],sampler.select_macro(char,state,history,None,ctrl.candidates)[0])

    def test_retreat_input_and_credit_use_spawn_instead_of_candidate(self):
        char,state,ctrl=order_tests.MacroOrdersTests().controller();ctrl.decide_macro(char,state)
        directive=ctrl.directive_for(char.name);directive['instruction']='RETREAT'
        from ghost_champions_v3.preplant.gc3_observation_attacker_history import encode_public
        obs=encode_public(char,state,ctrl.history)
        first=directive_features(obs,directive)
        self.assertAlmostEqual(first[1],char.pos[0]/state['grid'].shape[0])
        self.assertAlmostEqual(first[2],char.pos[1]/state['grid'].shape[1])
        altered=deepcopy(directive);altered['destination']=ctrl.candidates.coordinate(np.flatnonzero(ctrl.candidates.destination_mask)[1])
        np.testing.assert_array_equal(first,directive_features(obs,altered))
        del altered['retreat_target']
        with self.assertRaisesRegex(ValueError,'public own spawn'):directive_features(obs,altered)

    def reward_fixture(self,instruction='ADVANCE'):
        reward=DirectiveRewards.__new__(DirectiveRewards)
        reward.game=SimpleNamespace(grid=np.zeros((7,7),dtype=int))
        reward.config=config();reward.distances={};reward.order_best={};reward.order_arrivals=set();reward.measures=Counter()
        reward.own={'p':SimpleNamespace(pos=[3,1])}
        reward.before_positions={'p':(3,1)};reward.before_indices={}
        row=dict(macro_directive=dict(instruction=instruction,destination=(3,5),plant=(3,5),retreat_target=(3,0)),
            selections=dict(mode=0,position=22))
        reward.recorder=SimpleNamespace(latest={'p':row})
        return reward,row

    def test_wait_does_not_reward_a_failed_move_and_approach_cannot_repeat(self):
        reward,row=self.reward_fixture('WAIT')
        with patch.object(PreplantRewards,'finish_tick',return_value=({},{})):
            _,personal=reward.finish_tick(False,False)
            self.assertEqual(personal['p']['order_wait'],settings.WAIT_HOLD_REWARD)
            row['selections']['position']=23
            _,personal=reward.finish_tick(False,False)
            self.assertEqual(personal['p']['order_wait'],-settings.WAIT_MOVE_PENALTY)
            row['macro_directive']['instruction']='ADVANCE'
            reward.own['p'].pos=[3,2]
            _,first=reward.finish_tick(False,False)
            self.assertGreater(first['p']['order_advance'],0.)
            row['macro_directive']['instruction']='WAIT'
            reward.own['p'].pos=[3,1];reward.finish_tick(False,False)
            row['macro_directive']['instruction']='ADVANCE';reward.own['p'].pos=[3,2]
            _,again=reward.finish_tick(False,False)
            self.assertEqual(again['p']['order_advance'],0.)
            row['macro_directive']['instruction']='RETREAT';reward.own['p'].pos=[3,0]
            _,retreat=reward.finish_tick(False,False)
            self.assertGreater(retreat['p']['order_retreat'],0.)

    def test_synthetic_ppo_updates_adapter_but_not_base_or_anchor(self):
        char,state,ctrl=order_tests.MacroOrdersTests().controller()
        anchor=fresh_micro_policy().requires_grad_(False)
        model=DirectedMicroPolicy.from_micro(anchor)
        optimizer=torch.optim.Adam(trainable_directives(model),lr=.001)
        history=MatchHistory();history.begin_round();history.goal_sites[0]=0
        directive=dict(destination=ctrl.candidates.coordinate(np.flatnonzero(ctrl.candidates.destination_mask)[0]),
            plant=ctrl.candidates.coordinate(np.flatnonzero(ctrl.candidates.plant_mask)[0]),instruction='WAIT',retreat_target=tuple(char.pos))
        _,row=model.select_directed(char,state,history,directive)
        row.update(advantage=1.,**{'return':row['value']+1.})
        advance=dict(directive,instruction='ADVANCE')
        _,other=model.select_directed(char,state,history,advance)
        other.update(advantage=-1.,**{'return':other['value']-1.})
        old_base=base_digest(model);old_anchor=module_digest([anchor])
        old_directive=module_digest([model.directive])
        loss=optimize_history(model,anchor,optimizer,[row,other],dict(config(),PPO_EPOCHS=1,BATCH_SIZE=2))
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(old_base,base_digest(model));self.assertEqual(old_anchor,module_digest([anchor]))
        self.assertNotEqual(old_directive,module_digest([model.directive]))

    def test_blank_map_fails_before_creating_run_or_loading_source(self):
        _,state,ctrl=order_tests.MacroOrdersTests().controller()
        with TemporaryDirectory() as tmp:
            cfg=dict(config(),RUN_PATH=Path(tmp)/'run',LOG_PATH=Path(tmp)/'logs')
            with patch('ghost_champions_v3.macro_v2.gc3_training_attacker_directives.MacroCandidates.configured',
                    side_effect=ValueError('Set destination candidates')),patch(
                    'ghost_champions_v3.macro_v2.gc3_training_attacker_directives.load_preplant_policy',
                    side_effect=AssertionError('Source loaded before map validation')):
                with self.assertRaisesRegex(ValueError,'Set destination candidates'):train(cfg)
            self.assertFalse(cfg['RUN_PATH'].exists());self.assertFalse(cfg['LOG_PATH'].exists())

    def test_temporary_synthetic_training_counts_only_used_samples_and_finishes_graph(self):
        from ghost_champions_v3.macro_v2 import gc3_training_attacker_directives as training
        from ghost_champions_v3.preplant.gc3_observation_attacker_history import command_features
        from ghost_champions_v3.tools.gc3_metrics_monitor_data import snapshot
        char,state,ctrl=order_tests.MacroOrdersTests().controller()
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            fresh_micro_policy().save(root/'source.pt',trained=True,stage='rl',step=100175)
            FreshPolicy.fresh('attacker',44).save(root/'guard.pt',trained=True)
            cfg=dict(config(),INITIAL_CHECKPOINT=root/'source.pt',POSTPLANT_CHECKPOINT=root/'guard.pt',
                RUN_PATH=root/'run',LOG_PATH=root/'logs',ADDITIONAL_TRAINING_STEPS=2,ROLLOUT_DECISIONS=2,
                ROUNDS_PER_MATCH=1,PPO_EPOCHS=1,BATCH_SIZE=2,EVALUATE_EVERY_STEPS=2)
            def fake_episode(code,seed,*,policy,guard,candidates,config,history):
                history.begin_round();history.goal_sites[history.round_id]=0
                directive=dict(instruction='WAIT',destination=candidates.coordinate(np.flatnonzero(candidates.destination_mask)[0]),
                    plant=candidates.coordinate(np.flatnonzero(candidates.plant_mask)[0]),retreat_target=tuple(char.pos))
                rows=[]
                for i in range(4):
                    command,row=policy.select_directed(char,state,history,directive)
                    history.append(row['obs'],command_features(char,command,state),name=char.name,
                        command=dict(macro_directive=directive))
                    row.update(reward=.02,reward_components={'order_wait':.02},age=1,discount=.99,
                        advantage=float(i),**{'return':row['value']+1.})
                    rows.append(row)
                return rows,dict(decisions=4,success=False,planted=False,phase='directed_preplant')
            evaluation=dict(success_rate=.5,actual_round_win_rate=.5,plant_rate=.5)
            with patch.object(training.MacroCandidates,'configured',return_value=ctrl.candidates),patch.object(
                    training,'play_directives',side_effect=fake_episode),patch.object(training,'evaluate',return_value=evaluation):
                train(cfg)
            saved=torch.load(cfg['RUN_PATH']/'latest.pt',weights_only=True)
            self.assertEqual(saved['step'],100177);self.assertEqual(saved['additional_step'],2)
            self.assertTrue(saved['directives_trained'])
            data=snapshot(cfg['LOG_PATH'])
            self.assertTrue(data['complete']);self.assertEqual(data['step'],100177)
            self.assertEqual(data['episodes'][0]['decisions'],2)
            self.assertEqual(data['episodes'][0]['sampled_decisions'],4)
            self.assertEqual(sum(m['transitions'] for m in data['metrics']),2)

    def test_real_rollout_rewards_sum_and_replay_without_weight_updates(self):
        _,_,ctrl=order_tests.MacroOrdersTests().controller()
        policy=DirectedMicroPolicy.from_micro(fresh_micro_policy());guard=FreshPolicy.fresh('attacker',9)
        cfg=dict(config(),ORDER_PROBABILITIES=(0.,0.,1.),TORCH_THREADS=1)
        before=module_digest([policy,guard])
        rows,stats=play_directives('FNC',70,policy=policy,guard=guard,candidates=ctrl.candidates,config=cfg)
        self.assertTrue(rows);self.assertEqual(stats['instructions'].get('RETREAT'),len(rows))
        self.assertEqual(before,module_digest([policy,guard]))
        self.assertFalse(stats['macro_learning'])
        for row in rows:self.assertAlmostEqual(row['reward'],sum(row['reward_components'].values()),places=5)
        chosen=rows[::max(1,len(rows)//6)][:6]
        logp,_,values=policy.evaluate_records(chosen)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in chosen]),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in chosen]),atol=2e-5,rtol=2e-5)


if __name__=='__main__':unittest.main()
