"""Public-tap inputs, unchanged warm start, reward caps, conditional KL and real handoff."""
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.gc3_reward_phase import potential_difference
from ghost_champions_v3.gc3_observation import OBS_DIM,encode,map_hints
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.test.test_gc3_reward_phase import Game
from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard as settings
from ghost_champions_v3.postplant_v2.gc3_policy_attacker_guard import GuardPolicy
from ghost_champions_v3.postplant_v2.gc3_features_attacker_guard import guard_features,tap_state
from ghost_champions_v3.postplant_v2.gc3_reward_attacker_guard import GuardRewards,DEFUSER_DAMAGE_PER_HP,DEFUSER_KILL_REWARD
from ghost_champions_v3.postplant_v2.gc3_optimize_attacker_guard import (
    conditional_kl,probabilities,joint_kl,optimize,actor_optimizer,value_optimizer,
)
from ghost_champions_v3.postplant_v2.gc3_rollout_attacker_guard import load_entry,play_guard
from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import initialize,train


def config():return {name:value for name,value in vars(settings).items() if name.isupper()}


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def scene(self):
        char,state=preplant_fixture();state['is_planted']=True
        state['planted_pos']=tuple(map(int,np.argwhere(state['grid']==2)[0]));state['detonate_timer']=55
        state['defender_defuse_info']={'enemy':(0,6)}
        char.ultimate_name='NEON';char.ability_name='FLASH'
        return char,state

    def test_public_tap_count_start_remaining_and_hidden_position(self):
        char,state=self.scene();extra=guard_features(char,state)
        self.assertEqual(extra[3],0);self.assertEqual(extra[6],0)
        state['defender_defuse_info']={'one':(2,6),'two':(0,6)}
        extra=guard_features(char,state,(False,0.))
        self.assertEqual(extra[3],1);self.assertAlmostEqual(extra[4],2/6);self.assertAlmostEqual(extra[5],4/6)
        self.assertEqual(extra[7],1);self.assertAlmostEqual(extra[6],.2)
        class Hidden:
            name='unknown';team='D';is_alive=True;position_known=False
            @property
            def pos(self):raise AssertionError('Hidden enemy position was queried')
        state['chars'].append(Hidden());np.testing.assert_array_equal(extra,guard_features(char,state,(False,0.)))

    def test_zero_residual_retains_original_context_distribution_and_weights(self):
        char,state=self.scene();base=FreshPolicy.fresh('attacker',13);before=module_digest([base])
        policy=GuardPolicy.from_guard(base);obs=encode(char,state,map_hints(char,state));extra=guard_features(char,state)
        torch.testing.assert_close(base.features(torch.from_numpy(obs)),policy.features(torch.from_numpy(np.concatenate((obs,extra)))),rtol=0,atol=0)
        torch.manual_seed(17);first,row_base=base.select(char,state)
        torch.manual_seed(17);second,row_new=policy.select(char,state)
        self.assertEqual(first,second);self.assertEqual(row_base['selections'],row_new['selections'])
        self.assertAlmostEqual(row_base['logp'],row_new['logp'],places=6);self.assertEqual(before,module_digest([base]))
        replay,_,_=policy.evaluate_records([row_new]);self.assertAlmostEqual(float(replay[0].detach()),row_new['logp'],places=5)

    def test_conditional_kl_matches_enumerated_joint_actions(self):
        old_mode=[.30,.20,.20,.15,.15,0,0,0];new_mode=[.20,.25,.15,.20,.20,0,0,0]
        old={key:torch.tensor([[.4,.6]],dtype=torch.float64) for key in ('position','watch','target','ultimate_target','ally')}
        new={key:torch.tensor([[.7,.3]],dtype=torch.float64) for key in old}
        old['mode']=torch.tensor([old_mode],dtype=torch.float64);new['mode']=torch.tensor([new_mode],dtype=torch.float64)
        for ability,ultimate in [('FLASH','RAID'),('DANCE','NEON'),('RAMP','MONITOR')]:
            total=0.
            for mode in range(5):
                heads={0:['position'],1:['watch'],2:['position','watch'],3:(['ally'] if ability=='DANCE' else [] if ability=='RAMP' else ['target']),
                    4:(['watch'] if ultimate=='RAID' else ['ultimate_target'] if ultimate=='NEON' else [])}[mode]
                for choices in itertools.product(range(2),repeat=len(heads)):
                    p,q=old_mode[mode],new_mode[mode]
                    for key,choice in zip(heads,choices):p*=float(old[key][0,choice]);q*=float(new[key][0,choice])
                    total+=p*math.log(p/q)
            measured=float(conditional_kl(old,new,[dict(ability=ability,ultimate=ultimate)])[0])
            self.assertAlmostEqual(measured,total,places=12)

    def test_value_gradient_isolated_and_large_step_is_rolled_back(self):
        char,state=self.scene();policy=GuardPolicy.from_guard(FreshPolicy.fresh('attacker',13));rows=[]
        for i in range(4):
            _,row=policy.select(char,state);row.update(advantage=(-1.)**i,**{'return':row['value']+5.});rows.append(row)
        policy.zero_grad();policy.evaluate_records(rows)[2].sum().backward()
        self.assertTrue(all(p.grad is None for p in policy.actor_parameters()))
        before={k:v.detach().clone() for k,v in policy.state_dict().items() if not k.startswith('critic.')}
        cfg=dict(config(),LEARNING_RATE=.1,TARGET_KL=1e-9,MAX_KL=2e-9,BATCH_SIZE=4,VALUE_EPOCHS=2,VALUE_WARMUP_EPOCHS=1)
        result=optimize(policy,actor_optimizer(policy,cfg),value_optimizer(policy,cfg),rows,cfg)
        self.assertEqual(result['actor_updates'],0);self.assertGreater(result['rejected_actor_updates'],0)
        for key,value in before.items():torch.testing.assert_close(policy.state_dict()[key],value,rtol=0,atol=0)
        self.assertLessEqual(result['joint_kl'],cfg['MAX_KL'])

    def reward_game(self):
        game=Game();game.is_planted=True;game.is_defused=False;game.move_character=lambda char:None
        return game

    def test_actual_defuser_damage_and_kill_and_no_tap_only_bonus(self):
        game=self.reward_game();own,_,enemy,_=game.chars
        with GuardRewards(game,.99) as rewards:
            rewards.begin_tick();enemy.defuse_timer=1;game.move_character(enemy)
            game.analytics_tracker.record_contribution(own,enemy,1,'damage')
            enemy.hp=0.;enemy.is_alive=False;own.kills+=1
            shared,personal=rewards.finish_tick(True,True)
        self.assertEqual(shared['outcome'],30.);self.assertTrue(rewards.starts)
        self.assertAlmostEqual(personal[own.name]['defuser_damage'],100*DEFUSER_DAMAGE_PER_HP)
        self.assertAlmostEqual(personal[own.name]['defuser_kill'],DEFUSER_KILL_REWARD)
        game=self.reward_game();own,_,enemy,_=game.chars
        with GuardRewards(game,.99) as rewards:
            rewards.begin_tick();enemy.defuse_timer=1;game.move_character(enemy);enemy.defuse_timer=0
            _,personal=rewards.finish_tick(False,False)
        self.assertNotIn('defuser_damage',personal[own.name]);self.assertNotIn('defuser_kill',personal[own.name])

    def test_heal_and_tap_cycles_cannot_farm_extra_damage_reward(self):
        game=self.reward_game();own,_,enemy,_=game.chars;paid=0.
        with GuardRewards(game,.99) as rewards:
            for i in range(4):
                enemy.hp=100.;enemy.defuse_timer=1;rewards.begin_tick()
                game.analytics_tracker.record_contribution(own,enemy,i,'damage');enemy.hp=50.
                _,personal=rewards.finish_tick(False,False);paid+=personal[own.name]['defuser_damage']
        self.assertAlmostEqual(paid,100*DEFUSER_DAMAGE_PER_HP)
        values=[.3,1.5,.3,1.5,.3]
        shaping=[potential_difference(values[i],values[i+1],.99,i==len(values)-2) for i in range(len(values)-1)]
        self.assertAlmostEqual(sum(.99**i*r for i,r in enumerate(shaping)),-values[0])

    def test_mixed_nondefuse_damage_only_credits_defuse_share(self):
        game=self.reward_game();own,ally,enemy,_=game.chars
        with GuardRewards(game,.99) as rewards:
            rewards.begin_tick();game.analytics_tracker.record_contribution(ally,enemy,1,'damage')
            enemy.defuse_timer=1;game.analytics_tracker.record_contribution(own,enemy,1,'damage');enemy.hp=40.
            _,personal=rewards.finish_tick(False,False)
        self.assertAlmostEqual(personal[own.name]['defuser_damage'],30*DEFUSER_DAMAGE_PER_HP)
        self.assertNotIn('defuser_damage',personal[ally.name])

    def test_real_latest_macro_entry_guard_handoff_replay_and_frozen_sources(self):
        cfg=config();entry=load_entry(cfg);before=module_digest([entry['macro'],entry['micro'],entry['guard']])
        policy=GuardPolicy.from_guard(entry['guard'])
        rows,stats,_=play_guard('FNC',20361014,policy=policy,entry=entry,config=cfg)
        self.assertTrue(rows);self.assertEqual(stats['entry_macro_step'],20000)
        self.assertEqual(before,module_digest([entry['macro'],entry['micro'],entry['guard']]))
        sampled=rows[::max(1,len(rows)//4)][:4];logp,_,values=policy.evaluate_records(sampled)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in sampled]),atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in sampled]),atol=2e-5,rtol=2e-5)
        self.assertTrue(all(r['obs'][2] for r in rows))
        for row in rows:self.assertAlmostEqual(sum(row['reward_components'].values()),row['reward'],places=5)

    def test_synthetic_pipeline_dual_optimizer_resume_and_exact_steps(self):
        from ghost_champions_v3.postplant_v2 import gc3_training_attacker_guard as training
        char,state=self.scene()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);base=FreshPolicy.fresh('attacker',12);base.save(root/'guard.pt',trained=True,step=150252)
            (root/'macro.pt').write_bytes(b'macro');(root/'micro.pt').write_bytes(b'micro')
            cfg=dict(config(),INITIAL_CHECKPOINT=root/'guard.pt',ENTRY_MACRO_CHECKPOINT=root/'macro.pt',ENTRY_MICRO_CHECKPOINT=root/'micro.pt',
                RUN_PATH=root/'data',LOG_PATH=root/'logs',TRAINING_STEPS=2,ROLLOUT_DECISIONS=2,BATCH_SIZE=2,ROUNDS_PER_MATCH=1,
                VALUE_WARMUP_EPOCHS=1,VALUE_EPOCHS=1,PPO_EPOCHS=1,TARGET_KL=.5,MAX_KL=1.)
            entry=dict(macro_step=20000,micro_step=189327)
            class View:
                def save(self,path):path.write_bytes(b'public ledger')
            def fake(code,seed,*,policy,**kwargs):
                rows=[]
                for i in range(4):
                    _,row=policy.select(char,state);row.update(reward=0.,reward_components={},age=1,discount=.99,
                        advantage=(-1.)**i,**{'return':row['value']+2.});rows.append(row)
                return rows,dict(decisions=4,success=False,phase='postplant'),View()
            report=dict(success_rate=.5,defuse_started_win_rate=.5,reward_mean=0.,defused_rate=.5)
            with patch.object(training,'load_entry',return_value=entry),patch.object(training,'play_guard',side_effect=fake),patch.object(training,'evaluate',return_value=report):
                train(cfg);train(dict(cfg,RESUME=True,TRAINING_STEPS=4))
            status=json.loads((root/'logs/status.json').read_text());self.assertEqual(status['step'],4);self.assertTrue(status['complete'])
            _,saved=GuardPolicy.load(root/'data/latest.pt');self.assertIn('value_optimizer',saved)
            self.assertTrue(list((root/'data/rollouts').glob('*.npz')))


if __name__=='__main__':unittest.main()
