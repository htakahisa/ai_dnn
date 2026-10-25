"""Macro timing/causality, exact event rewards, frozen lower actions and resume."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory, command_features, ACTION_WIDTH
from ghost_champions_v3.preplant.gc3_policy_attacker_site import SiteHistoryPolicy
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import Recorder, module_digest
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.macro_v1.gc3_policy_attacker_macro import MacroPolicy, MacroHistoryView
from ghost_champions_v3.macro_v1.gc3_learn_attacker_macro import MacroAttackerController
from ghost_champions_v3.macro_v1.gc3_reward_attacker_macro import MacroRewards
from ghost_champions_v3.macro_v1.gc3_training_attacker_macro import initialize, optimize, rng_state


class MacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def rewards(self):
        a = SimpleNamespace(name="own", team="A", is_alive=True, kills=0)
        b = SimpleNamespace(name="ally", team="A", is_alive=True, kills=0)
        d = SimpleNamespace(name="enemy", team="D", is_alive=True, defuse_timer=0)
        game = SimpleNamespace(chars=[a,b,d], is_planted=False, move_character=lambda c:None)
        return game, a, b, d, MacroRewards(game)

    def test_exact_rewards_are_team_totals_without_extra_shaping(self):
        game, a, b, d, reward = self.rewards()
        a.kills=1; b.kills=1; b.is_alive=False; game.is_planted=True
        parts=reward.finish_tick(False, False)
        self.assertEqual(parts,{"kill":2.,"death":-1.,"plant":10.})
        parts=reward.finish_tick(True, True)
        self.assertEqual(parts,{"kill":0.,"death":0.,"outcome":30.,"no_defuse_start":10.})
        self.assertEqual(sum(reward.totals.values()),51.)
        self.assertEqual(sum(reward.finish_tick(True, True).values()),0.)

    def test_a_single_interrupted_defuse_start_disqualifies_bonus(self):
        game, a, b, d, reward = self.rewards()
        game.is_planted=True; game.move_character=lambda c:setattr(c,"defuse_timer",1)
        with reward:
            game.move_character(d)
            # Dies/cancels during later battle resolution of this same tick.
            d.is_alive=False; d.defuse_timer=0
            parts=reward.finish_tick(True, True)
        self.assertTrue(reward.defuse_ever_started)
        self.assertEqual(parts,{"kill":0.,"death":0.,"plant":10.,"outcome":30.})

    def test_no_plant_never_earns_no_defuse_bonus_even_on_elimination_win(self):
        for win, value in ((True,30.),(False,-30.)):
            game, a, b, d, reward=self.rewards()
            self.assertEqual(reward.finish_tick(True,win),{"kill":0.,"death":0.,"outcome":value})

    def test_defeat_never_earns_no_defuse_bonus_even_after_plant(self):
        game, a, b, d, reward=self.rewards();game.is_planted=True
        parts=reward.finish_tick(True,False)
        self.assertNotIn("no_defuse_start",parts)
        self.assertEqual(parts["outcome"],-30.)

    def make_controller(self):
        preplant=SiteHistoryPolicy.fresh("attacker",10)
        guard=FreshPolicy.fresh("attacker",11)
        macro=MacroPolicy.fresh("attacker",12)
        history=MatchHistory();history.begin_round()
        controller=MacroAttackerController(macro_policy=macro,preplant_policy=preplant,guard_policy=guard,history=history)
        return controller

    def test_goals_can_switch_A_B_A_at_consecutive_ticks_including_postplant(self):
        char,state=preplant_fixture();ctrl=self.make_controller()
        choices=iter([1,2,1,0])
        original=ctrl.policy.select_macro
        def choose(*args):
            _,row=original(*args);action=next(choices);row['selections']={'macro':action};return action,row
        with patch.object(ctrl.policy,"select_macro",side_effect=choose):
            for i,goal in enumerate((0,1,0,0)):
                state['battle_tick']=i
                state['is_planted']=i>=2
                ctrl.decide_macro(char,state)
                self.assertEqual(ctrl.current_goal,goal)
                self.assertEqual(ctrl.history.goal_sites[ctrl.history.round_id],goal)
        self.assertEqual(ctrl.switches,2)
        self.assertEqual(len(ctrl.history.events),0) # Frozen lower inputs are not polluted.
        self.assertEqual(len(ctrl.macro_history.events),4)
        self.assertFalse(any(p.requires_grad for p in ctrl.lower.preplant.policy.parameters()))
        self.assertFalse(any(p.requires_grad for p in ctrl.lower.postplant.policy.parameters()))

    def test_lower_site_head_is_not_sampled_when_macro_owns_goal(self):
        char,state=preplant_fixture();ctrl=self.make_controller()
        ctrl.history.goal_sites[ctrl.history.round_id]=1
        _,row=ctrl.lower.preplant.policy.select(char,state,ctrl.history)
        self.assertEqual(row['goal_site'],1)
        self.assertNotIn('site',row['selections'])

    def test_tick_cache_uses_setup_countdown_and_runs_before_lower_moves(self):
        ctrl=self.make_controller()
        # An already-decided frame never re-samples on the other four agents.
        ctrl.history.round_id=0
        game=SimpleNamespace(in_defender_setup_phase=True,defender_setup_phase=SimpleNamespace(ticks_remaining=20))
        ctrl._real_game=game;ctrl._token=(0,True,20)
        with patch.object(ctrl,'decide_macro',side_effect=AssertionError('Resampled same frame')):
            ctrl.prepare_team_tick()

    def test_sampled_macro_probabilities_and_values_replay_causally(self):
        char,state=preplant_fixture();ctrl=self.make_controller();records=[]
        for i in range(4):
            ctrl.macro_history.sync_public_events()
            action,row=ctrl.policy.select_macro(char,state,ctrl.macro_history,ctrl.current_goal,ctrl.goal_age)
            ctrl.current_goal=ctrl.current_goal if action==0 else action-1
            ctrl.macro_history.events.append(dict(obs=row['obs'],action=np.zeros(ACTION_WIDTH,dtype=np.float32),
                goal_site=ctrl.current_goal,macro_action=action))
            records.append(row)
        logp,entropy,values=ctrl.policy.evaluate_records(records)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in records]),atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in records]),atol=3e-6,rtol=3e-6)
        self.assertTrue(bool((entropy>0).all()))
        self.assertFalse(records[0]['masks']['macro'][0])
        initial=ctrl.policy.contexts([records[0]])
        ctrl.macro_history.events[-1]['obs']=np.zeros_like(records[0]['obs'])
        torch.testing.assert_close(initial,ctrl.policy.contexts([records[0]]),atol=0,rtol=0)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'macro.pt';ctrl.policy.save(path,trained=False,step=0)
            loaded,_=MacroPolicy.load(path)
            torch.testing.assert_close(ctrl.policy.contexts(records),loaded.contexts(records),atol=0,rtol=0)

    def test_macro_memory_copies_all_public_round_events_and_has_own_cache(self):
        char,state=preplant_fixture();history=MatchHistory();history.begin_round()
        from ghost_champions_v3.preplant.gc3_observation_attacker_history import encode_public
        for _ in range(4):
            history.append(encode_public(char,state,history),np.zeros(ACTION_WIDTH,dtype=np.float32))
        view=MacroHistoryView(history);view.sync_public_events()
        self.assertEqual(len(view.events),4)
        view._cache_policy='macro';history._cache_policy='lower'
        self.assertNotEqual(view._cache_policy,history._cache_policy)
        history.begin_round();view.sync_public_events()
        self.assertEqual(view.round_id,1)
        self.assertEqual(len(view.events),4)
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.npz';view.save(path)
            restored=MatchHistory.load(path)
            self.assertEqual(len(restored.events),4)
            self.assertEqual(restored.round_id,1)

    def test_macro_backprop_does_not_update_frozen_lower_models(self):
        char,state=preplant_fixture();ctrl=self.make_controller();rows=[]
        for i in range(4):
            action,row=ctrl.policy.select_macro(char,state,ctrl.macro_history,0,i)
            row.update(advantage=i-1.5,**{'return':float(i)})
            rows.append(row)
        before=module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy,ctrl.policy])
        optimizer=SimpleNamespace(zero_grad=ctrl.policy.zero_grad,step=lambda:None)
        config=dict(PPO_EPOCHS=1,BATCH_SIZE=4,PPO_CLIP=.2,ENTROPY_WEIGHT=.01)
        self.assertTrue(np.isfinite(optimize(ctrl.policy,optimizer,rows,config)))
        self.assertTrue(torch.isfinite(ctrl.policy.actor.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in ctrl.lower.preplant.policy.parameters()))
        self.assertEqual(before,module_digest([ctrl.lower.preplant.policy,ctrl.lower.postplant.policy,ctrl.policy]))

    def test_fresh_macro_and_resume_preserve_inputs_and_reject_changed_models(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);pre=root/'pre.pt';guard=root/'guard.pt';pre.write_bytes(b'pre');guard.write_bytes(b'guard')
            config=dict(RESUME=False,SEED=22,PREPLANT_CHECKPOINT=pre,POSTPLANT_CHECKPOINT=guard,LEARNING_RATE=.0001)
            run=root/'run';run.mkdir()
            policy,opt,saved=initialize(config,run)
            self.assertIsNone(saved);self.assertEqual(len(opt.state),0)
            policy.save(run/'latest.pt',trained=False,step=0,optimizer=opt.state_dict(),**rng_state())
            resume=dict(config,RESUME=True)
            loaded,opt2,saved=initialize(resume,run)
            self.assertEqual(module_digest([policy]),module_digest([loaded]))
            self.assertEqual(len(opt2.state),0)
            guard.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'changed'):
                initialize(resume,run)


if __name__=='__main__':
    unittest.main()
