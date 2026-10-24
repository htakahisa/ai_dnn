"""Regression checks for critic isolation, KL rollback and preserved BC recovery."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
from ghost_champions_v3.macro_v2.gc3_bc_attacker_macro import MacroTeacher,save_teacher
from ghost_champions_v3.macro_v2.gc3_policy_attacker_macro import MacroPolicy,MacroHistoryView
from ghost_champions_v3.macro_v2.gc3_optimize_attacker_macro import (
    actor_optimizer,value_optimizer,fit_value,optimize_macro,OPTIMIZER_VERSION,
)
from ghost_champions_v3.macro_v2.gc3_training_attacker_macro import initialize,train,select_round_rows
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_macro_training as fixtures
config=fixtures.config


def actor_state(policy):
    return {k:v.detach().clone() for k,v in policy.state_dict().items() if not k.startswith('critic.')}


class GuardedMacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        np.random.seed(19);torch.manual_seed(19)
        helper=fixtures.MacroTrainingTests();char,state,ctrl,policy,view=helper.fixture();rows=[];plan=None
        for i in range(4):
            plan,row=policy.select_macro(char,state,view,plan,ctrl.candidates);helper.append(view,row,plan)
            row.update(advantage=(-1.)**i,reward=0.,reward_components={'outcome':0.},age=1,discount=.99,
                **{'return':row['value']+5.})
            rows.append(row)
        return helper,char,state,ctrl,policy,view,rows

    def unchanged_actor(self,policy,before):
        for key,value in actor_state(policy).items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)

    def test_value_gradients_and_optimizer_cannot_update_actor(self):
        _,_,_,_,policy,_,rows=self.fixture();policy.zero_grad()
        policy.evaluate_records(rows)[2].sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.actor_parameters()))
        self.assertTrue(all(p.grad is not None for p in policy.value_parameters()))
        cfg=dict(config(),BATCH_SIZE=4,VALUE_LEARNING_RATE=.01)
        actor=actor_optimizer(policy,cfg);critic=value_optimizer(policy,cfg)
        self.assertFalse({id(p) for g in actor.param_groups for p in g['params']} & {id(p) for g in critic.param_groups for p in g['params']})
        before=actor_state(policy);stats=fit_value(policy,critic,rows,cfg,8)
        self.unchanged_actor(policy,before);self.assertLess(stats['value_mse_after'],stats['value_mse_before'])

    def test_oversized_actor_step_is_rolled_back_with_optimizer_state(self):
        _,_,_,_,policy,_,rows=self.fixture()
        cfg=dict(config(),LEARNING_RATE=.1,BATCH_SIZE=4,PPO_EPOCHS=2,TARGET_KL=1e-10,MAX_KL=2e-10)
        actor=actor_optimizer(policy,cfg);critic=value_optimizer(policy,cfg)
        # Existing Adam moments must also survive rejection, not just weights.
        for p in policy.actor_parameters():p.grad=torch.zeros_like(p)
        actor.step();actor.zero_grad();expected_optimizer=deepcopy(actor.state_dict());before=actor_state(policy)
        stats=optimize_macro(policy,actor,critic,rows,cfg)
        self.unchanged_actor(policy,before)
        restored=actor.state_dict();self.assertEqual(restored['param_groups'],expected_optimizer['param_groups'])
        self.assertEqual(restored['state'].keys(),expected_optimizer['state'].keys())
        for slot,values in restored['state'].items():
            for key,value in values.items():torch.testing.assert_close(value,expected_optimizer['state'][slot][key],rtol=0,atol=0)
        self.assertEqual(stats['actor_updates'],0);self.assertGreater(stats['rejected_actor_updates'],0)
        self.assertEqual(stats['stop_reason'],'max_kl_rollback');self.assertLessEqual(stats['joint_kl'],cfg['MAX_KL'])
        self.assertGreater(stats['value_updates'],0)

    def test_full_batch_guard_also_rolls_back_a_step_that_passed_monitor(self):
        _,_,_,_,policy,_,rows=self.fixture();cfg=dict(config(),BATCH_SIZE=4,PPO_EPOCHS=1)
        actor=actor_optimizer(policy,cfg);before=actor_state(policy)
        with patch('ghost_champions_v3.macro_v2.gc3_optimize_attacker_macro.joint_kl',side_effect=[.01,.05,0.]):
            stats=optimize_macro(policy,actor,value_optimizer(policy,cfg),rows,cfg)
        self.unchanged_actor(policy,before);self.assertEqual(actor.state_dict()['state'],{})
        self.assertEqual(stats['stop_reason'],'full_batch_max_kl_rollback');self.assertEqual(stats['actor_updates'],0)

    def test_auxiliary_bc_reuses_labels_and_full_batch_kl_is_bounded(self):
        helper,char,state,ctrl,policy,view,rows=self.fixture()
        teacher=MacroTeacher(policy,config(),17);plan,label=teacher.select_macro(char,state,view,None,ctrl.candidates)
        helper.append(view,label,plan)
        cfg=dict(config(),LEARNING_RATE=1e-6,BATCH_SIZE=4,PPO_EPOCHS=2,BC_REPLAY_WEIGHT=.05,BC_REPLAY_BATCH_SIZE=1)
        stats=optimize_macro(policy,actor_optimizer(policy,cfg),value_optimizer(policy,cfg),rows,cfg,teacher_rows=[label],warmup=True)
        self.assertGreater(stats['bc_replay_loss_mean'],0);self.assertLessEqual(stats['joint_kl'],cfg['MAX_KL'])
        self.assertTrue(stats['value_warmup']);self.assertEqual(stats['optimizer_version'],OPTIMIZER_VERSION)

    def test_sampling_retains_contributions_from_all_rounds(self):
        rows=[dict(round=i//100) for i in range(700)];stats=[dict(decisions=100) for _ in range(7)]
        selected=select_round_rows(rows,stats,512)
        self.assertEqual(len(selected),512);self.assertEqual(sum(s['decisions'] for s in stats),512)
        self.assertTrue(all(s['decisions']>0 and s['sampled_decisions']==100 for s in stats))
        self.assertEqual({r['round'] for r in selected},set(range(7)))

    def test_bc_best_recovery_and_both_optimizers_resume_without_collection(self):
        from ghost_champions_v3.macro_v2 import gc3_training_attacker_macro as training
        from ghost_champions_v3.gc3_policy import FreshPolicy
        from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
        from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
        helper,char,state,ctrl,policy,view,rows=self.fixture()
        with TemporaryDirectory() as temp:
            root=Path(temp);pre=DirectedMicroPolicy.from_micro(fresh_micro_policy());pre.directives_trained=True
            pre.save(root/'pre.pt',trained=True,stage='directive_rl',step=189327)
            FreshPolicy.fresh('attacker',3).save(root/'guard.pt',trained=True)
            lower={k:file_hash(root/f) for k,f in [('PREPLANT_CHECKPOINT','pre.pt'),('POSTPLANT_CHECKPOINT','guard.pt')]}
            policy.save(root/'bc_best.pt',trained=True,stage='macro_rl',step=0,bc_epoch=8,
                candidate_signature=ctrl.candidates.signature,lower_checkpoint_hashes=lower)
            teacher_root=root/'teacher';teacher_root.mkdir()
            teacher=MacroTeacher(policy,config(),17);plan,label=teacher.select_macro(char,state,view,None,ctrl.candidates)
            helper.append(view,label,plan);save_teacher(teacher_root/'match.pt',[label],view,ctrl.candidates)
            cfg=dict(config(),INITIAL_CHECKPOINT=root/'bc_best.pt',BC_REPLAY_PATH=teacher_root,BC_REPLAY_WEIGHT=.05,
                PREPLANT_CHECKPOINT=root/'pre.pt',POSTPLANT_CHECKPOINT=root/'guard.pt',RUN_PATH=root/'run',LOG_PATH=root/'logs',
                TRAINING_STEPS=2,ROLLOUT_DECISIONS=2,ROUNDS_PER_MATCH=1,PPO_EPOCHS=1,BATCH_SIZE=2,EVALUATE_EVERY_STEPS=2,
                TARGET_KL=.5,MAX_KL=1.)
            hashes={p:file_hash(p) for p in [root/'bc_best.pt',root/'pre.pt',root/'guard.pt',teacher_root/'match.pt']}
            init_dir=root/'init';init_dir.mkdir();inherited,_,_,_=initialize(cfg,init_dir,ctrl.candidates)
            self.assertEqual(module_digest([inherited]),module_digest([policy]))
            def fake(code,seed,*,policy,history,**kwargs):
                history.begin_round();view=MacroHistoryView(history);batch=[];plan=None
                for i in range(2):
                    plan,row=policy.select_macro(char,state,view,plan,ctrl.candidates);helper.append(view,row,plan)
                    row.update(advantage=(-1.)**i,reward=0.,reward_components={'outcome':0.},age=1,discount=.99,
                        **{'return':row['value']+2.});batch.append(row)
                return batch,dict(decisions=2,success=False,phase='macro_v2'),view
            report=dict(reward_mean=0.,actual_round_win_rate=0.,success_rate=0.,plant_rate=0.)
            with patch.object(training.MacroCandidates,'configured',return_value=ctrl.candidates),patch.object(
                    training,'play_macro',side_effect=fake),patch.object(training,'evaluate',return_value=report),patch.object(
                    training,'MacroTeacher',side_effect=AssertionError('Teacher collection must not run')):
                train(cfg)
                _,saved=MacroPolicy.load(cfg['RUN_PATH']/'latest.pt')
                self.assertEqual(saved['bc_epoch'],8);self.assertTrue(saved['optimizer']['state']);self.assertTrue(saved['value_optimizer']['state'])
                self.assertEqual(saved['optimizer_version'],OPTIMIZER_VERSION)
                train(dict(cfg,RESUME=True,TRAINING_STEPS=4))
            status=json.loads((cfg['LOG_PATH']/'status.json').read_text())
            self.assertEqual(status['step'],4);self.assertTrue(status['complete'])
            self.assertTrue(all(file_hash(path)==value for path,value in hashes.items()))


if __name__=='__main__':unittest.main()
