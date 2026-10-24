"""Regression tests for a notified adjacent defuser and learned facing response."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_engine_actions import legal_actions,decode_command
from ghost_champions_v3.postplant_v2.gc3_features_attacker_guard import public_coverage,response_targets
from ghost_champions_v3.postplant_v2.gc3_policy_attacker_guard import GuardPolicy
from ghost_champions_v3.postplant_v2.gc3_optimize_attacker_guard import response_loss,response_probability,optimize,actor_optimizer,value_optimizer
from ghost_champions_v3.postplant_v2.gc3_training_attacker_guard import select_response_rows


def adjacent_scene():
    char,state=preplant_fixture();grid=state['grid']
    r,c=next((r,c) for r,c in np.argwhere(grid!=1) if 0<c<grid.shape[1]-1 and grid[r,c-1]!=1 and grid[r,c+1]!=1)
    char.pos=[int(r),int(c)];char.facing='W';char.smoke_charges=0
    enemy=deepcopy(char);enemy.name='defuser';enemy.team='D';enemy.pos=[int(r),int(c+1)];enemy.position_known=True
    state.update(chars=[char,enemy],is_planted=True,planted_pos=tuple(enemy.pos),detonate_timer=30,
                 defender_defuse_info={'defuser':(3,6)})
    return char,enemy,state


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_visible_adjacent_defuser_requires_turning_toward_it(self):
        char,enemy,state=adjacent_scene();masks,_=legal_actions(char,state)
        targets=response_targets(char,state,masks);origin=targets['positions'].index(targets['origin'])
        right=np.ravel_multi_index(tuple(enemy.pos),state['grid'].shape)
        left=np.ravel_multi_index((char.pos[0],char.pos[1]-1),state['grid'].shape)
        self.assertTrue(targets['watch'][origin,right]);self.assertFalse(targets['watch'][origin,left])
        self.assertEqual(decode_command(char,state,{'mode':1,'watch':int(right)})[1]['facing'],'E')
        self.assertFalse(targets['move'][origin])  # Holding while facing away is not a response.

    def test_occupied_defuser_endpoint_is_shootable_and_area_is_not_a_maximum(self):
        walls=np.zeros((12,12),dtype=bool)
        toward=public_coverage((5,5),'E',[(5,6)],walls,set(),{(5,6)})
        self.assertGreater(toward,.9)
        both=public_coverage((5,5),'E',[(5,6),(5,4)],walls,set(),set())
        self.assertAlmostEqual(both,toward/2)
        self.assertEqual(public_coverage((5,5),'E',[(5,7)],walls,set(),{(5,6)}),0.)

    def test_adjacent_smoke_uses_the_engine_shooting_rule_and_keeps_turn_labels(self):
        from grid_lines import line_cells
        from ghost_champions_v3.test.test_gc3_reward_phase import Game
        engine=Game();walls=np.zeros((12,12),dtype=bool)
        for target in ((5,6),(6,6),(5,7),(5,8)):
            smoke={target};cells=line_cells((5,5),target)
            expected=engine._smoke_allows_line(cells,smoke)
            facing='E' if target[0]==5 else 'SE'
            actual=public_coverage((5,5),facing,[target],walls,smoke,set())>0
            self.assertEqual(actual,expected)
        char,enemy,state=adjacent_scene();state['smoke_cells']={tuple(enemy.pos)}
        masks,_=legal_actions(char,state);labels=response_targets(char,state,masks)
        self.assertIsNotNone(labels)
        origin=labels['positions'].index(labels['origin'])
        aim=int(np.ravel_multi_index(tuple(enemy.pos),state['grid'].shape))
        self.assertTrue(labels['watch'][origin,aim])
        self.assertFalse(labels['move'][origin])

    def test_public_recon_bypasses_smoke_but_not_wall_or_intervening_body(self):
        walls=np.zeros((12,12),dtype=bool);target=(5,9);smoke={(5,7)}
        self.assertEqual(public_coverage((5,5),'E',[target],walls,smoke,set()),0.)
        self.assertGreater(public_coverage((5,5),'E',[target],walls,smoke,set(),revealed_targets={target}),0.)
        walls[5,7]=True
        self.assertEqual(public_coverage((5,5),'E',[target],walls,smoke,set(),revealed_targets={target}),0.)
        walls[5,7]=False
        self.assertEqual(public_coverage((5,5),'E',[target],walls,smoke,{(5,7)},revealed_targets={target}),0.)

    def test_v2_checkpoint_keeps_old_smoke_feature_while_v3_matches_engine(self):
        char,enemy,state=adjacent_scene();char.facing='E';state['smoke_cells']={tuple(enemy.pos)}
        policy=GuardPolicy.from_guard(FreshPolicy.fresh('attacker',13));policy.feature_version=2
        _,old=policy.select(char,state);self.assertEqual(old['obs'][-32+11],0.)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'v2.pt';policy.save(path,trained=True,step=1);restored,_=GuardPolicy.load(path)
            self.assertEqual(restored.feature_version,2)
            _,row=restored.select(char,state);np.testing.assert_array_equal(old['obs'],row['obs'])
        policy.feature_version=3
        _,current=policy.select(char,state);self.assertGreater(current['obs'][-32+11],0.)

    def test_hidden_position_is_never_read_and_forced_facing_is_not_supervised(self):
        char,enemy,state=adjacent_scene()
        class Hidden:
            name='defuser';team='D';is_alive=True;position_known=False
            @property
            def pos(self):raise AssertionError('Hidden position read')
        state['chars']=[char,Hidden()];masks,_=legal_actions(char,state)
        self.assertIsNotNone(response_targets(char,state,masks))
        char.facing_forced_this_tick=True
        self.assertIsNone(response_targets(char,state,masks))

    def test_response_loss_trains_direction_and_sampling_survives_reload(self):
        char,_,state=adjacent_scene();model=GuardPolicy.from_guard(FreshPolicy.fresh('attacker',13))
        # The actual FNC failure had smoke on the adjacent defuser's cell.
        state['smoke_cells']={tuple(state['chars'][1].pos)}
        model.collect_response_targets=True
        _,row=model.select(char,state)
        before=float(response_probability(model,[row]).detach())
        optimizer=torch.optim.Adam(model.actor_parameters(),lr=.003)
        for _ in range(25):
            optimizer.zero_grad();response_loss(model,[row]).backward();optimizer.step()
        after=float(response_probability(model,[row]).detach())
        self.assertGreater(after,.95);self.assertGreater(after,before+.3)
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'guard.pt';model.save(path,trained=True,step=25)
            loaded,_=GuardPolicy.load(path)
            self.assertAlmostEqual(float(response_probability(loaded,[row]).detach()),after,places=6)
            from battle_logic import BattleLogicMixin
            from types import SimpleNamespace
            from grid_lines import line_cells
            from ghost_champions_v3.test.test_gc3_reward_phase import Game
            engine=Game();enemy=state['chars'][1]
            torch.manual_seed(333);commands=[loaded.select(char,state)[0] for _ in range(32)]
            shootable=0
            for destination,command in commands:
                dr,dc=destination[0]-char.pos[0],destination[1]-char.pos[1]
                facing=command['facing'] if isinstance(command,dict) else ('N' if dr<0 else 'S') if dr else ('W' if dc<0 else 'E') if dc else char.facing
                actor=SimpleNamespace(pos=destination,facing=facing)
                angle=BattleLogicMixin._facing_angle_diff(None,actor,enemy)
                shootable+=angle<=90 and engine._smoke_allows_line(line_cells(destination,enemy.pos),state['smoke_cells'])
            self.assertGreater(shootable,28)

    def test_rare_alerts_survive_rollout_sampling_and_both_rounds_remain(self):
        rows=[dict(index=i,response_targets={} if i in (10,11,12,210,211,212) else None) for i in range(400)]
        stats=[dict(decisions=200),dict(decisions=200)]
        selected=select_response_rows(rows,stats,32,.5)
        self.assertEqual(len(selected),32)
        self.assertEqual(sum(r['response_targets'] is not None for r in selected),6)
        self.assertTrue(all(s['decisions']>0 for s in stats))
        self.assertEqual(sum(s['decisions'] for s in stats),32)

    def test_old_checkpoint_retains_its_original_observation_semantics(self):
        char,_,state=adjacent_scene();policy=GuardPolicy.from_guard(FreshPolicy.fresh('attacker',13));policy.feature_version=1
        with torch.no_grad():policy.features.objective[-1].weight.fill_(.01)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.pt';policy.save(path,trained=True,step=1)
            data=torch.load(path,weights_only=True);del data['feature_version'];torch.save(data,path)
            restored,_=GuardPolicy.load(path);self.assertEqual(restored.feature_version,1)
            torch.manual_seed(500);command,row=policy.select(char,state)
            torch.manual_seed(500);loaded,other=restored.select(char,state)
            self.assertEqual(command,loaded);np.testing.assert_array_equal(row['obs'],other['obs'])
            self.assertAlmostEqual(row['logp'],other['logp'],places=6)

    def test_real_optimizer_improves_response_within_kl_limit(self):
        from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard_response as settings
        char,_,state=adjacent_scene();policy=GuardPolicy.from_guard(FreshPolicy.fresh('attacker',13));policy.collect_response_targets=True
        anchor=deepcopy(policy).requires_grad_(False);config={k:v for k,v in vars(settings).items() if k.isupper()}
        config.update(BATCH_SIZE=8,VALUE_EPOCHS=1,VALUE_WARMUP_EPOCHS=1,RESPONSE_LOSS_WEIGHT=2.)
        actor=actor_optimizer(policy,config);critic=value_optimizer(policy,config)
        _,probe=policy.select(char,state);before=float(response_probability(policy,[probe]).detach())
        for _ in range(20):
            rows=[]
            for i in range(8):
                _,row=policy.select(char,state);row.update(advantage=0.,**{'return':row['value']});rows.append(row)
            stats=optimize(policy,actor,critic,rows,config,anchor=anchor)
            self.assertLessEqual(stats['joint_kl'],config['MAX_KL']+1e-6)
            self.assertEqual(stats['response_rows'],8)
        self.assertGreater(float(response_probability(policy,[probe]).detach()),before+.15)
        self.assertTrue(all(p.grad is None for p in anchor.parameters()))

    def test_response_run_stops_on_regression_and_resumes_both_optimizers(self):
        from ghost_champions_v3.postplant_v2 import gc3_train_attacker_guard_response as settings
        from ghost_champions_v3.postplant_v2 import gc3_training_attacker_guard as training
        char,_,state=adjacent_scene()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);base=FreshPolicy.fresh('attacker',13);base.save(root/'base.pt',trained=True,step=100)
            GuardPolicy.from_guard(base).save(root/'warm.pt',trained=True,step=12)
            (root/'macro.pt').write_bytes(b'macro');(root/'micro.pt').write_bytes(b'micro')
            config={k:v for k,v in vars(settings).items() if k.isupper()}
            config.update(INITIAL_CHECKPOINT=root/'base.pt',START_CHECKPOINT=root/'warm.pt',ENTRY_MACRO_CHECKPOINT=root/'macro.pt',
                ENTRY_MICRO_CHECKPOINT=root/'micro.pt',RUN_PATH=root/'data',LOG_PATH=root/'logs',RESUME=False,
                TRAINING_STEPS=6,ROLLOUT_DECISIONS=2,ROUNDS_PER_MATCH=1,BATCH_SIZE=2,EVALUATE_EVERY_STEPS=2,
                EVALUATION_PATIENCE=1,VALUE_WARMUP_EPOCHS=1,VALUE_EPOCHS=1,PPO_EPOCHS=1)
            class View:
                def save(self,path):path.write_bytes(b'ledger')
            def collect(code,seed,*,policy,**kwargs):
                rows=[]
                for i in range(4):
                    local=dict(state,defender_defuse_info={'defuser':(3,6)} if i%2 else {})
                    _,row=policy.select(char,local)
                    row.update(reward=0.,reward_components={},age=1,discount=.99,advantage=(-1.)**i,**{'return':row['value']+1.})
                    rows.append(row)
                return rows,dict(decisions=4,success=False,phase='postplant'),View()
            report=dict(success_rate=.5,defuse_started_win_rate=.5,reward_mean=0.,defused_rate=.5)
            with patch.object(training,'load_entry',return_value=dict(macro_step=20,micro_step=30)),patch.object(training,'play_guard',side_effect=collect),patch.object(training,'evaluate',return_value=report):
                training.train(config)
                status=json.loads((root/'logs/status.json').read_text(encoding='utf-8'))
                self.assertEqual(status['step'],2);self.assertTrue(status['stopped_early']);self.assertFalse(status['complete'])
                training.train(dict(config,RESUME=True,TRAINING_STEPS=4))
            _,saved=GuardPolicy.load(root/'data/latest.pt')
            self.assertEqual(saved['step'],4);self.assertEqual(saved['feature_version'],3)
            self.assertTrue(saved['optimizer']['state']);self.assertTrue(saved['value_optimizer']['state'])
            self.assertGreaterEqual(saved['evaluations_without_improvement'],2)


if __name__=='__main__':unittest.main()
