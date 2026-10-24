"""Instruction transfer, learnable obedience, physical blocking and safe credit."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_observation import ARENA_SHAPE
from ghost_champions_v3.gc3_rollout import module_digest
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.preplant.gc3_rollout_attacker_preplant import PreplantRewards
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy
from ghost_champions_v3.macro_v2.gc3_curriculum_attacker_directives import DirectiveSampler
from ghost_champions_v3.macro_v2.gc3_supervision_attacker_directives import instruction_options,instruction_loss,optimize_directives
from ghost_champions_v3.macro_v2.gc3_training_attacker_directives import initialize,trainable_directives,base_digest,evaluation_rank
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_directive_training as training_tests
from ghost_champions_v3.macro_v2.test import test_gc3_attacker_macro_orders as orders

config=training_tests.config


class DirectiveAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def learning_fixture(self):
        char,state,ctrl=orders.MacroOrdersTests().controller()
        for p in np.argwhere(state['grid']!=1):
            r,c=map(int,p)
            neighbours=[(r+dr,c+dc) for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                if 0<=r+dr<ARENA_SHAPE[0] and 0<=c+dc<ARENA_SHAPE[1] and state['grid'][r+dr,c+dc]!=1]
            if len(neighbours)>=2 and state['grid'][r,c]!=2:break
        char.pos=[r,c];char.has_spike=False
        state['chars']=[char];state['available_orbs']=[]
        anchor=fresh_micro_policy().requires_grad_(False).eval()
        model=DirectedMicroPolicy.from_micro(anchor)
        history=MatchHistory();history.begin_round();history.goal_sites[0]=0
        plant=ctrl.candidates.coordinate(np.flatnonzero(ctrl.candidates.plant_mask)[0])
        directive=dict(instruction='ADVANCE',destination=neighbours[0],plant=plant,retreat_target=neighbours[1])
        rows=[]
        for instruction in ('WAIT','ADVANCE','RETREAT'):
            _,row=model.select_directed(char,state,history,dict(directive,instruction=instruction))
            row.update(advantage=0.,**{'return':row['value']})
            rows.append(row)
        return model,anchor,rows,ctrl

    def test_completed_directive_weights_are_retained_and_source_is_unchanged(self):
        model,anchor,_,ctrl=self.learning_fixture()
        with torch.no_grad():model.directive[-1].bias.fill_(.123)
        model.directives_trained=True
        before=module_digest([model])
        with TemporaryDirectory() as tmp:
            p=Path(tmp);anchor.save(p/'source.pt',trained=True,stage='rl',step=100175)
            model.save(p/'directed.pt',trained=True,stage='directive_rl',step=118607)
            (p/'guard.pt').write_bytes(b'guard hash fixture');(p/'run').mkdir()
            cfg=dict(config(),INITIAL_CHECKPOINT=p/'source.pt',DIRECTIVE_CHECKPOINT=p/'directed.pt',POSTPLANT_CHECKPOINT=p/'guard.pt')
            loaded,_,optimizer,_,step=initialize(cfg,p/'run',ctrl.candidates)
            self.assertEqual(step,118607);self.assertEqual(module_digest([loaded]),before)
            self.assertTrue(loaded.directives_trained);self.assertFalse(optimizer.state)
            self.assertEqual(len([p for p in loaded.parameters() if p.requires_grad]),4)
            with torch.no_grad():model.move.bias.add_(1.)
            model.save(p/'directed.pt',trained=True,stage='directive_rl',step=118607)
            with self.assertRaisesRegex(ValueError,'different frozen base'):initialize(cfg,p/'run',ctrl.candidates)

    def test_supervision_learns_three_different_orders_without_changing_legal_actions_or_base(self):
        torch.manual_seed(73)
        model,anchor,rows,_=self.learning_fixture();cfg=config()
        before_base=base_digest(model);before_anchor=module_digest([anchor])
        masks=[deepcopy(r['masks']) for r in rows]
        before=instruction_loss(model,rows,model.contexts(rows),cfg)[1]
        optimizer=torch.optim.Adam(trainable_directives(model),lr=.003)
        # Synthetic observations only, not user checkpoints or game training.
        for _ in range(100):
            optimizer.zero_grad();loss,_=instruction_loss(model,rows,model.contexts(rows),cfg)
            loss.backward();optimizer.step()
        after=instruction_loss(model,rows,model.contexts(rows),cfg)[1]
        for order in ('WAIT','ADVANCE','RETREAT'):
            self.assertGreater(after[order+'_probability'],before[order+'_probability']+.20)
        self.assertEqual(base_digest(model),before_base);self.assertEqual(module_digest([anchor]),before_anchor)
        for row,mask in zip(rows,masks):
            for key in mask:np.testing.assert_array_equal(row['masks'][key],mask[key])
        context=model.contexts(rows)
        # Disobedient physical moves are still present in the sampled policy.
        for i,row in enumerate(rows):
            dist=model.distribution(context[i],row,'position')
            self.assertTrue(bool((dist.probs[torch.from_numpy(row['masks']['position'])]>0).all()))

    def test_blocked_approach_and_objective_opportunities_are_excluded(self):
        model,_,rows,_=self.learning_fixture()
        row=deepcopy(rows[1]);row.pop('_instruction_options',None)
        origin=np.ravel_multi_index(instruction_options(row)['origin'],ARENA_SHAPE)
        row['masks']['position'][:]=False;row['masks']['position'][origin]=True
        row.pop('_instruction_options',None)
        self.assertFalse(instruction_options(row)['eligible'])
        _,report=instruction_loss(model,[row],model.contexts([row]),config())
        self.assertEqual(report['ADVANCE_samples'],0)
        for mode in (5,7):
            row=deepcopy(rows[1]);row.pop('_instruction_options',None);row['masks']['mode'][mode]=True
            self.assertFalse(instruction_options(row)['eligible'])

    def test_combined_ppo_records_instruction_probabilities_and_preserves_base(self):
        model,anchor,rows,_=self.learning_fixture();old=base_digest(model)
        optimizer=torch.optim.Adam(trainable_directives(model),lr=.001)
        loss,reports=optimize_directives(model,anchor,optimizer,rows,dict(config(),PPO_EPOCHS=1,BATCH_SIZE=3))
        self.assertTrue(np.isfinite(loss));self.assertEqual(base_digest(model),old)
        self.assertEqual(len(reports),1);self.assertGreater(reports[0]['instruction_loss'],0.)
        self.assertEqual(reports[0]['RETREAT_samples'],1)

    def test_retreat_exposure_starts_after_advancing_and_no_spawn_stop_counts_as_approach(self):
        char,state,ctrl=orders.MacroOrdersTests().controller()
        sampler=DirectiveSampler(43,3,(0.,0.,1.),warmup_ticks=2)
        history=MatchHistory();history.begin_round()
        plans=[sampler.select_macro(char,state,history,None,ctrl.candidates)[0] for _ in range(5)]
        self.assertTrue(all(x=='ADVANCE' for x in plans[0]['instructions']))
        self.assertEqual(plans[0],plans[1]);self.assertTrue(all(x=='RETREAT' for x in plans[2]['instructions']))
        self.assertEqual(plans[2],plans[4])
        reward,row=training_tests.DirectiveTrainingTests().reward_fixture('RETREAT')
        reward.before_positions={'p':(3,0)};reward.own['p'].pos=[3,0]
        with patch.object(PreplantRewards,'finish_tick',return_value=({},{})):reward.finish_tick(False,False)
        self.assertEqual(reward.measures['RETREAT_applicable'],0)

    def test_away_penalty_arrival_once_and_obedience_over_win_rate_in_best_selection(self):
        reward,row=training_tests.DirectiveTrainingTests().reward_fixture()
        with patch.object(PreplantRewards,'finish_tick',side_effect=lambda *args:({},{})):
            reward.own['p'].pos=[3,0]
            _,parts=reward.finish_tick(False,False)
            self.assertLess(parts['p']['order_advance_away'],0.)
            reward.before_positions={'p':(3,4)};reward.own['p'].pos=[3,5]
            _,parts=reward.finish_tick(False,False)
            self.assertGreater(parts['p']['order_advance_arrival'],0.)
            _,again=reward.finish_tick(False,False)
            self.assertNotIn('order_advance_arrival',again['p'])
            self.assertEqual(again['p']['order_advance'],0.)
        strong=dict(instruction_compliance_rates={'WAIT':.8,'ADVANCE':.8,'RETREAT':.8},actual_round_win_rate=.2,plant_rate=.2)
        weak=dict(instruction_compliance_rates={'WAIT':1.,'ADVANCE':.9,'RETREAT':.1},actual_round_win_rate=1.,plant_rate=1.)
        self.assertGreater(evaluation_rank(strong),evaluation_rank(weak))


if __name__=='__main__':unittest.main()
