"""Teacher replay, all-head optimization, frozen lower models and stage accounting."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
import torch
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory,ACTION_WIDTH
from ghost_champions_v3.macro_v2 import gc3_train_attacker_macro as settings
from ghost_champions_v3.macro_v2.gc3_policy_attacker_macro import MacroPolicy,MacroHistoryView
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
from ghost_champions_v3.macro_v2.gc3_bc_attacker_macro import MacroTeacher,save_teacher,load_teacher,optimize_bc
from ghost_champions_v3.macro_v2.gc3_rollout_attacker_macro import PreplantMacroRecorder,play_macro
from ghost_champions_v3.macro_v2.gc3_training_attacker_macro import initialize,train,optimize
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_macro_orders as orders


def config():
    return dict({k:v for k,v in vars(settings).items() if k.isupper()},
        INITIAL_CHECKPOINT=None,BC_REPLAY_PATH=None,BC_REPLAY_WEIGHT=0.,VALUE_WARMUP_EPOCHS=1,VALUE_EPOCHS=1)


class MacroTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        char,state,ctrl=orders.MacroOrdersTests().controller()
        policy=MacroPolicy.fresh('attacker',13)
        history=MatchHistory();history.begin_round();view=MacroHistoryView(history)
        return char,state,ctrl,policy,view

    def append(self,view,row,plan):
        view.events.append(dict(obs=row['obs'].copy(),action=np.zeros(ACTION_WIDTH,dtype=np.float32),
            macro_plan=plan,round_id=view.round_id,name='macro',tick=0))

    def test_teacher_labels_replay_causally_and_saved_data_can_be_reused(self):
        char,state,ctrl,policy,view=self.fixture()
        teacher=MacroTeacher(policy,config(),17)
        plan,row=teacher.select_macro(char,state,view,None,ctrl.candidates)
        self.append(view,row,plan)
        before=policy.evaluate_records([row])
        with TemporaryDirectory() as temp:
            path=Path(temp)/'teacher.pt'
            save_teacher(path,[row],view,ctrl.candidates)
            replay=load_teacher(Path(temp),ctrl.candidates)
            after=policy.evaluate_records(replay)
            for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            # Own label/future macro event must not enter its own decision.
            replay[0]['history'].events[0]['obs'][:]=7.
            again=policy.evaluate_records(replay)
            for a,b in zip(after,again):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)

    def test_bc_and_ppo_update_all_three_heads_but_not_lower_models(self):
        char,state,ctrl,policy,view=self.fixture()
        teacher=MacroTeacher(policy,config(),17);rows=[];plan=None
        lower_before=module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy])
        for _ in range(2):
            plan,row=teacher.select_macro(char,state,view,plan,ctrl.candidates)
            self.append(view,row,plan);rows.append(row)
        heads=[policy.destination,policy.instruction,policy.plant]
        before=[module_digest([head]) for head in heads]
        optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
        loss=optimize_bc(policy,optimizer,rows,dict(config(),BATCH_SIZE=2))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(old!=module_digest([head]) for old,head in zip(before,heads)))
        with torch.no_grad():logp,_,values=policy.evaluate_records(rows)
        for i,row in enumerate(rows):row.update(logp=float(logp[i]),advantage=1. if i==0 else -1.,**{'return':float(values[i])+1.})
        before=[module_digest([head]) for head in heads]
        loss=optimize(policy,optimizer,rows,dict(config(),PPO_EPOCHS=1,BATCH_SIZE=2,TARGET_KL=5.,MAX_KL=10.))
        self.assertTrue(np.isfinite(loss));self.assertTrue(all(old!=module_digest([head]) for old,head in zip(before,heads)))
        self.assertEqual(lower_before,module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy]))

    def test_postplant_commands_are_not_ppo_actions_and_final_reward_is_retained(self):
        recorder=PreplantMacroRecorder()
        recorder.record('macro',dict(obs=np.array([0.,0.,0.]),value=0.))
        recorder.reward(10.,.99,False,components={'plant':10.})
        recorder.record('macro',dict(obs=np.array([0.,0.,1.]),value=0.))
        recorder.reward(30.,.99,True,components={'outcome':30.})
        rows=recorder.advantages(.95)
        self.assertEqual(len(rows),1);self.assertAlmostEqual(rows[0]['reward'],39.7)
        self.assertEqual(rows[0]['reward_components']['outcome'],29.7)

    def test_fresh_macro_resume_manifest_and_extended_steps(self):
        char,state,ctrl,policy,view=self.fixture()
        with TemporaryDirectory() as temp:
            root=Path(temp);(root/'data').mkdir()
            cfg=dict(config(),PREPLANT_CHECKPOINT=root/'pre.pt',POSTPLANT_CHECKPOINT=root/'guard.pt')
            cfg['PREPLANT_CHECKPOINT'].write_bytes(b'pre');cfg['POSTPLANT_CHECKPOINT'].write_bytes(b'guard')
            model,opt,saved,inputs=initialize(cfg,root/'data',ctrl.candidates)
            self.assertIsNone(saved)
            from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import rng_state
            model.save(root/'data/latest.pt',stage='macro_rl',step=2,trained=True,
                candidate_signature=ctrl.candidates.signature,lower_checkpoint_hashes=inputs,optimizer=opt.state_dict(),**rng_state())
            continued=initialize(dict(cfg,RESUME=True,TRAINING_STEPS=30000),root/'data',ctrl.candidates)
            self.assertEqual(module_digest([continued[0]]),module_digest([model]))
            with self.assertRaisesRegex(RuntimeError,'source/settings/map/lower'):
                initialize(dict(cfg,RESUME=True,ENTROPY_WEIGHT=.2),root/'data',ctrl.candidates)

    def test_synthetic_pipeline_finishes_exactly_and_keeps_lower_checkpoints(self):
        from ghost_champions_v3.macro_v2 import gc3_training_attacker_macro as training
        char,state,ctrl,_,_=self.fixture()
        with TemporaryDirectory() as temp:
            root=Path(temp);pre=DirectedMicroPolicy.from_micro(fresh_micro_policy());pre.directives_trained=True
            pre.save(root/'pre.pt',trained=True,stage='directive_rl',step=189327)
            FreshPolicy.fresh('attacker',3).save(root/'guard.pt',trained=True)
            cfg=dict(config(),PREPLANT_CHECKPOINT=root/'pre.pt',POSTPLANT_CHECKPOINT=root/'guard.pt',
                RUN_PATH=root/'run',LOG_PATH=root/'logs',TRAINING_STEPS=2,ROLLOUT_DECISIONS=2,
                BC_WARMUP_STEPS=0,ROUNDS_PER_MATCH=1,PPO_EPOCHS=1,BATCH_SIZE=2,EVALUATE_EVERY_STEPS=2)
            def fake_episode(code,seed,*,policy,history,**kwargs):
                history.begin_round();view=MacroHistoryView(history);rows=[];plan=None
                for i in range(4):
                    plan,row=policy.select_macro(char,state,view,plan,ctrl.candidates);self.append(view,row,plan)
                    row.update(reward=0.,reward_components={'outcome':0.},age=1,discount=.99,
                        advantage=float(i),**{'return':row['value']+1.});rows.append(row)
                return rows,dict(decisions=4,success=False,phase='macro_v2'),view
            report=dict(reward_mean=0.,actual_round_win_rate=0.,success_rate=0.,plant_rate=0.)
            from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
            hashes=[file_hash(cfg[k]) for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')]
            with patch.object(training.MacroCandidates,'configured',return_value=ctrl.candidates),patch.object(
                    training,'play_macro',side_effect=fake_episode),patch.object(training,'evaluate',return_value=report):train(cfg)
            status=json.loads((cfg['LOG_PATH']/'status.json').read_text());self.assertTrue(status['complete'])
            self.assertEqual(status['step'],2)
            episode=json.loads((cfg['LOG_PATH']/'episodes.jsonl').read_text())
            self.assertEqual(episode['decisions'],2);self.assertEqual(episode['sampled_decisions'],4)
            self.assertEqual(hashes,[file_hash(cfg[k]) for k in ('PREPLANT_CHECKPOINT','POSTPLANT_CHECKPOINT')])

    def test_real_neural_macro_round_has_exact_replay_and_frozen_lower(self):
        char,state,ctrl,policy,view=self.fixture()
        pre=ctrl.lower.preplant.policy;guard=ctrl.lower.postplant.policy
        before=module_digest([policy,pre,guard])
        rows,stats,view=play_macro('FNC',70,policy=policy,preplant=pre,guard=guard,candidates=ctrl.candidates)
        self.assertTrue(rows);self.assertTrue(stats['macro_only_learning'])
        self.assertEqual(before,module_digest([policy,pre,guard]))
        self.assertTrue(all(not row['obs'][2] for row in rows))
        selected=rows[::max(1,len(rows)//4)][:4]
        logp,_,values=policy.evaluate_records(selected)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in selected]),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in selected]),atol=2e-5,rtol=2e-5)
        for row in rows:self.assertAlmostEqual(row['reward'],sum(row['reward_components'].values()),places=5)
        self.assertEqual(set(stats['reward_component_sums'])-{'kill','death','plant','outcome','no_defuse_start'},set())

    def test_bc_warmup_pipeline_and_collection_resume_use_saved_teacher(self):
        from ghost_champions_v3.macro_v2 import gc3_training_attacker_macro as training
        char,state,ctrl,_,_=self.fixture()
        with TemporaryDirectory() as temp:
            root=Path(temp);pre=DirectedMicroPolicy.from_micro(fresh_micro_policy());pre.directives_trained=True
            pre.save(root/'pre.pt',trained=True,stage='directive_rl',step=189327)
            FreshPolicy.fresh('attacker',3).save(root/'guard.pt',trained=True)
            cfg=dict(config(),PREPLANT_CHECKPOINT=root/'pre.pt',POSTPLANT_CHECKPOINT=root/'guard.pt',
                RUN_PATH=root/'run',LOG_PATH=root/'logs',TRAINING_STEPS=2,ROLLOUT_DECISIONS=2,
                BC_WARMUP_STEPS=2,BC_WARMUP_EPOCHS=1,ROUNDS_PER_MATCH=1,PPO_EPOCHS=1,BATCH_SIZE=2,EVALUATE_EVERY_STEPS=2)
            calls=[]
            def fake(code,seed,*,policy,history,**kwargs):
                calls.append(type(policy).__name__);history.begin_round();view=MacroHistoryView(history);rows=[];plan=None
                for i in range(2):
                    plan,row=policy.select_macro(char,state,view,plan,ctrl.candidates);self.append(view,row,plan)
                    row.update(reward=0.,reward_components={'outcome':0.},age=1,discount=.99,advantage=float(i),
                        **{'return':row['value']+1.});rows.append(row)
                return rows,dict(decisions=2,success=False,phase='macro_v2'),view
            report=dict(reward_mean=0.,actual_round_win_rate=0.,success_rate=0.,plant_rate=0.)
            with patch.object(training.MacroCandidates,'configured',return_value=ctrl.candidates),patch.object(
                    training,'play_macro',side_effect=fake),patch.object(training,'evaluate',return_value=report):train(cfg)
            self.assertEqual(calls,['MacroTeacher','MacroPolicy'])
            _,saved=MacroPolicy.load(cfg['RUN_PATH']/'latest.pt');self.assertEqual(saved['bc_epoch'],1)
            self.assertTrue(json.loads((cfg['LOG_PATH']/'status.json').read_text())['complete'])
            teacher_rows=load_teacher(cfg['RUN_PATH']/'bc_teacher',ctrl.candidates);self.assertEqual(len(teacher_rows),2)
            # A checkpoint saved before collection can resume without reinitializing.
            data=root/'resume';data.mkdir()
            model,opt,_,_=initialize(cfg,data,ctrl.candidates)
            restored=initialize(dict(cfg,RESUME=True),data,ctrl.candidates)
            self.assertEqual(module_digest([model]),module_digest([restored[0]]))
            self.assertEqual(restored[2]['stage'],'macro_bc');self.assertEqual(restored[2]['bc_epoch'],0)


if __name__=='__main__':unittest.main()
