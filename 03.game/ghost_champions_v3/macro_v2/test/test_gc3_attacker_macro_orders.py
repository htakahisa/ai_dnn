"""Spatial macro sampling, causal replay, directive inputs and real engine handoff."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import Recorder,module_digest,engine_directory
from ghost_champions_v3.gc3_observation import ARENA_SHAPE
from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
from ghost_champions_v3.preplant.gc3_observation_attacker_history import MatchHistory
from ghost_champions_v3.test.test_gc3_attacker_preplant import preplant_fixture
from ghost_champions_v3.macro_v2.gc3_candidates_attacker_macro import MacroCandidates
from ghost_champions_v3.macro_v2.gc3_directives_attacker_macro import DirectedMicroPolicy,INSTRUCTIONS,directive_features
from ghost_champions_v3.macro_v2.gc3_policy_attacker_macro import MacroPolicy,MacroHistoryView
from ghost_champions_v3.macro_v2.gc3_learn_attacker_macro import MacroAttackerController


def candidates_for(grid):
    marked=grid.copy()
    marked[np.isin(marked,(5,6))]=0
    points=[tuple(map(int,p)) for p in np.argwhere(grid!=1)[-4:]]
    for point in points:marked[point]=5
    text='\n'.join(''.join(str(v) for v in row) for row in marked)
    return MacroCandidates(text,grid),text,points


class MacroOrdersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def controller(self):
        char,state=preplant_fixture()
        for slot,point in enumerate(np.argwhere(state['grid']!=1)[4:8]):
            ally=deepcopy(char);ally.name=f'ally{slot}';ally.pos=list(map(int,point))
            state['chars'].append(ally)
        for unit in state['chars']:
            if unit.team=='A':unit.is_alive=True
        candidates,_,_=candidates_for(state['grid'])
        history=MatchHistory();history.begin_round()
        micro=DirectedMicroPolicy.fresh('attacker',2)
        controller=MacroAttackerController(candidates=candidates,macro_policy=MacroPolicy.fresh('attacker',3),
            preplant_policy=micro,guard_policy=FreshPolicy.fresh('attacker',4),history=history)
        return char,state,controller

    def test_map_five_only_and_all_real_plant_cells(self):
        _,state=preplant_fixture();candidates,text,points=candidates_for(state['grid'])
        self.assertEqual(int(candidates.destination_mask.sum()),len(points))
        np.testing.assert_array_equal(candidates.plant_mask,(state['grid']==2).reshape(-1))
        for point in points:self.assertTrue(candidates.destination_mask[np.ravel_multi_index(point,ARENA_SHAPE)])
        plant=tuple(map(int,np.argwhere(state['grid']==2)[-1]))
        restricted=MacroCandidates(text,state['grid'],(plant,))
        self.assertEqual(restricted.plant_mask.sum(),1)
        self.assertNotEqual(candidates.signature,restricted.signature)
        with self.assertRaisesRegex(ValueError,'real plantable'):MacroCandidates(text,state['grid'],(points[-1],))

    def test_blank_user_map_does_not_invent_destinations(self):
        from ghost_champions_v3.macro_v2.gc3_map_attacker_destinations import MACRO_DESTINATION_MAP_STR
        _,state=preplant_fixture()
        blank=MACRO_DESTINATION_MAP_STR.replace('5','0').replace('6','0')
        with self.assertRaisesRegex(ValueError,'Set destination candidates'):MacroCandidates(blank,state['grid'])
        _,text,_=candidates_for(state['grid'])
        broken='0'+text[1:]
        with self.assertRaisesRegex(ValueError,'walls/dimensions'):MacroCandidates(broken,state['grid'])

    def test_each_player_has_independent_destination_and_three_orders(self):
        char,state,ctrl=self.controller()
        points=np.flatnonzero(ctrl.candidates.destination_mask)
        planted=np.flatnonzero(ctrl.candidates.plant_mask)[0]
        with torch.no_grad():
            for head in (ctrl.policy.destination,ctrl.policy.instruction,ctrl.policy.plant):head.weight.zero_();head.bias.fill_(-40.)
            for slot in range(5):
                ctrl.policy.destination.bias[slot*len(ctrl.candidates.destination_mask)+points[slot%len(points)]]=40.
                ctrl.policy.instruction.bias[slot*3+slot%3]=40.
            ctrl.policy.plant.bias[planted]=40.
        command=ctrl.decide_macro(char,state)
        own=sorted((c for c in state['chars'] if c.team=='A'),key=lambda c:str(c.name))
        self.assertEqual(len(command['players']),5)
        for slot,unit in enumerate(own):
            self.assertEqual(command['players'][str(unit.name)]['instruction'],INSTRUCTIONS[slot%3])
            self.assertEqual(command['players'][str(unit.name)]['destination'],MacroCandidates.coordinate(points[slot%len(points)]))
        self.assertEqual(command['planned_plant'],MacroCandidates.coordinate(planted))

    def test_joint_probabilities_values_and_all_heads_replay_causally(self):
        char,state,ctrl=self.controller();recorder=Recorder();ctrl.recorder=recorder
        for tick in range(4):
            state['battle_tick']=tick
            ctrl.decide_macro(char,state)
        rows=recorder.rows
        logp,entropy,value=ctrl.policy.evaluate_records(rows)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in rows]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(value,torch.tensor([r['value'] for r in rows]),atol=1e-5,rtol=1e-5)
        self.assertTrue(torch.isfinite(entropy).all())
        again=ctrl.policy.evaluate_records(rows[:1])
        ctrl.macro_history.events[-1]['macro_plan']['plant']=(0,0)
        after=ctrl.policy.evaluate_records(rows[:1])
        for a,b in zip(again,after):torch.testing.assert_close(a,b)
        (-(logp.mean())+value.square().mean()).backward()
        for head in (ctrl.policy.destination,ctrl.policy.instruction,ctrl.policy.plant):
            self.assertGreater(float(head.weight.grad.abs().sum()),0.)

    def test_sampling_is_not_argmax_and_dead_slots_are_not_commands(self):
        char,state,ctrl=self.controller()
        own=sorted((c for c in state['chars'] if c.team=='A'),key=lambda c:str(c.name))
        own[-1].is_alive=False
        with torch.no_grad():
            for head in (ctrl.policy.destination,ctrl.policy.instruction,ctrl.policy.plant):head.weight.zero_();head.bias.zero_()
        found=set()
        for _ in range(24):
            plan,row=ctrl.policy.select_macro(char,state,ctrl.macro_history,None,ctrl.candidates)
            found.add((plan['destinations'][0],plan['instructions'][0]))
            self.assertIsNone(plan['destinations'][-1]);self.assertIsNone(plan['instructions'][-1])
            self.assertFalse(row['active'][-1])
        self.assertGreater(len(found),1)
        command=ctrl.decide_macro(char,state)
        self.assertNotIn(str(own[-1].name),command['players'])

    def test_player_identity_does_not_depend_on_state_list_order(self):
        char,state,ctrl=self.controller()
        rng=torch.get_rng_state()
        first,row1=ctrl.policy.select_macro(char,state,ctrl.macro_history,None,ctrl.candidates)
        torch.set_rng_state(rng)
        reversed_state=dict(state,chars=list(reversed(state['chars'])))
        second,row2=ctrl.policy.select_macro(char,reversed_state,ctrl.macro_history,None,ctrl.candidates)
        self.assertEqual(row1['roster'],row2['roster'])
        self.assertEqual(first,second)
        np.testing.assert_array_equal(row1['obs'],row2['obs'])

    def test_micro_receives_own_targets_and_orders_without_action_masks(self):
        char,state,ctrl=self.controller()
        macro=ctrl.decide_macro(char,state)
        controller=ctrl.lower.preplant
        rec=Recorder();controller.recorder=rec
        controller.decide_move(char,state)
        row=rec.rows[-1];directive=macro['players'][str(char.name)]
        self.assertEqual(row['macro_directive'],directive)
        self.assertEqual(ctrl.history.events[-1]['command']['macro_directive'],directive)
        from ghost_champions_v3.preplant.gc3_policy_attacker_history import preplant_legal_actions
        masks,_=preplant_legal_actions(char,state)
        for key,mask in masks.items():np.testing.assert_array_equal(row['masks'][key],mask)
        feature=directive_features(row['obs'],directive)
        self.assertEqual(feature.shape,(32,));self.assertEqual(feature[-4:-1].sum(),1.)
        old=deepcopy(row['macro_directive'])
        ctrl.player_orders[str(char.name)]['instruction']='RETREAT'
        self.assertEqual(row['macro_directive'],old)

    def test_directed_micro_log_probs_replay_and_orders_have_a_learnable_effect(self):
        char,state,ctrl=self.controller();ctrl.decide_macro(char,state)
        model=ctrl.lower.preplant.policy
        with torch.no_grad():torch.nn.init.normal_(model.directive[-1].weight,std=.2)
        rec=Recorder();ctrl.lower.preplant.recorder=rec
        for tick in range(3):
            state['battle_tick']=tick;ctrl.lower.preplant.decide_move(char,state)
        logp,_,values=model.evaluate_records(rec.rows)
        torch.testing.assert_close(logp,torch.tensor([r['logp'] for r in rec.rows]),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(values,torch.tensor([r['value'] for r in rec.rows]),atol=1e-5,rtol=1e-5)
        a,b=deepcopy(rec.rows[0]),deepcopy(rec.rows[0])
        a['macro_directive']['instruction']='WAIT';b['macro_directive']['instruction']='ADVANCE'
        context=model.contexts([a,b])
        self.assertFalse(torch.allclose(context[0],context[1]))

    def test_zero_new_branch_preserves_an_explicitly_converted_micro(self):
        char,state=preplant_fixture();base=fresh_micro_policy();directed=DirectedMicroPolicy.from_micro(base)
        history=MatchHistory();history.begin_round()
        _,row=base.select(char,state,history)
        a=base.evaluate_records([row]);b=directed.evaluate_records([row])
        for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
        self.assertFalse(directed.directives_trained)

    def test_plan_changes_any_tick_and_round_reset_clears_all_orders(self):
        char,state,ctrl=self.controller()
        a=next(tuple(map(int,p)) for p in np.argwhere(state['grid']==2) if p[1]<ARENA_SHAPE[1]//2)
        b=next(tuple(map(int,p)) for p in np.argwhere(state['grid']==2) if p[1]>=ARENA_SHAPE[1]//2)
        original=ctrl.policy.select_macro
        chosen=iter((a,b,a))
        def force(*args):
            plan,row=original(*args);plan['plant']=next(chosen);return plan,row
        with patch.object(ctrl.policy,'select_macro',side_effect=force):
            for tick,plant in enumerate((a,b,a)):
                state['battle_tick']=tick;self.assertEqual(ctrl.decide_macro(char,state)['planned_plant'],plant)
        self.assertEqual(ctrl.switches,2)
        ctrl.reset_macro();self.assertIsNone(ctrl.current_plan);self.assertFalse(ctrl.player_orders)

    def test_postplant_uses_existing_guard_and_no_new_micro_inputs(self):
        char,state,ctrl=self.controller();ctrl.decide_macro(char,state)
        guard=ctrl.lower.postplant
        state['is_planted']=True;state['planted_pos']=tuple(map(int,np.argwhere(state['grid']==2)[0]))
        previous=ctrl.history.goal_sites.copy()
        ctrl.decide_macro(char,state)
        self.assertEqual(ctrl.history.goal_sites,previous)
        with patch.object(guard,'decide_move',return_value=(list(char.pos),'HOLD')) as called,patch.object(
                ctrl.lower.preplant.policy,'select_directed',side_effect=AssertionError('Postplant micro used')):
            ctrl.lower.decide_move(char,state)
        called.assert_called_once()
        self.assertIsNone(ctrl.history.events[-1]['command']['macro_directive'])

    def test_checkpoints_do_not_accept_v1_and_keep_directive_history(self):
        char,state,ctrl=self.controller();ctrl.decide_macro(char,state);ctrl.lower.preplant.decide_move(char,state)
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            ctrl.policy.save(root/'macro.pt',trained=False,candidate_signature=ctrl.candidates.signature)
            loaded,_=MacroPolicy.load(root/'macro.pt')
            self.assertEqual(module_digest([ctrl.policy]),module_digest([loaded]))
            ctrl.lower.preplant.policy.save(root/'micro.pt',trained=False)
            micro,_=DirectedMicroPolicy.load(root/'micro.pt')
            self.assertEqual(module_digest([micro]),module_digest([ctrl.lower.preplant.policy]))
            from ghost_champions_v3.macro_v1.gc3_policy_attacker_macro import MacroPolicy as OldMacro
            OldMacro.fresh('attacker',4).save(root/'old.pt',trained=True)
            with self.assertRaisesRegex(ValueError,'v2 checkpoint mismatch'):MacroPolicy.load(root/'old.pt')
            ctrl.macro_history.save(root/'history.npz')
            history=MatchHistory.load(root/'history.npz')
            self.assertTrue(any(e.get('macro_plan') is not None for e in history.events))
            self.assertTrue(any(isinstance(e.get('command'),dict) and e['command'].get('macro_directive') for e in history.events))

    def test_trained_macro_requires_directive_trained_lower_and_identical_map(self):
        _,_,ctrl=self.controller()
        with TemporaryDirectory() as tmp:
            root=Path(tmp);micro_path=root/'micro.pt';guard_path=root/'guard.pt';macro_path=root/'macro.pt'
            ctrl.lower.preplant.policy.save(micro_path,trained=True)
            ctrl.lower.postplant.policy.save(guard_path,trained=True)
            from ghost_champions_v3.preplant.gc3_data_attacker_preplant import file_hash
            hashes=dict(PREPLANT_CHECKPOINT=file_hash(micro_path),POSTPLANT_CHECKPOINT=file_hash(guard_path))
            ctrl.policy.save(macro_path,trained=True,candidate_signature=ctrl.candidates.signature,lower_checkpoint_hashes=hashes)
            with self.assertRaisesRegex(ValueError,'Train the micro directive inputs'):
                MacroAttackerController(candidates=ctrl.candidates,macro_checkpoint=macro_path,
                    preplant_checkpoint=micro_path,guard_checkpoint=guard_path)
            ctrl.policy.save(macro_path,trained=True,candidate_signature='changed',lower_checkpoint_hashes=hashes)
            with self.assertRaisesRegex(ValueError,'candidate map changed'):
                MacroAttackerController(candidates=ctrl.candidates,macro_checkpoint=macro_path,
                    preplant_checkpoint=micro_path,guard_checkpoint=guard_path)

    def test_real_setup_changes_plan_once_per_tick_for_all_five_players(self):
        from functools import partial
        with engine_directory():
            from run_game import VisualFPSBattle,_build_team_ai
            from map_data import NEW_MAZE_STR
            from party_presets import get_preset
            from team_ai import DualRoleTeamAI
            from controllers import DefaultDefenderController
            grid=np.array([[int(v) for v in row] for row in NEW_MAZE_STR.strip().splitlines()])
            candidates,_,_=candidates_for(grid)
            micro=DirectedMicroPolicy.fresh('attacker',6);guard=FreshPolicy.fresh('attacker',7)
            macro=MacroPolicy.fresh('attacker',8)
            raw=MacroAttackerController(candidates=candidates,macro_policy=macro,preplant_policy=micro,guard_policy=guard,history=MatchHistory())
            raw.lower.preplant.recorder=Recorder()
            own,enemy=get_preset('Ghost Champions'),get_preset('Fnatic2023')
            game=VisualFPSBattle(NEW_MAZE_STR,DualRoleTeamAI('GC macro v2',lambda:raw,DefaultDefenderController),
                _build_team_ai('default',device='cpu'),headless=True,disable_side_swap=True,
                attacker_roster=list(own.players),defender_roster=list(enemy.players),spike_holder_name=own.spike_holder)
            before=module_digest([macro,micro,guard])
            with torch.no_grad():
                for _ in range(3):game._simulate_tick()
            self.assertEqual(raw.macro_decisions,3)
            self.assertEqual(len(raw.player_orders),5)
            self.assertEqual(before,module_digest([macro,micro,guard]))
            self.assertTrue(raw.lower.preplant.recorder.rows)
            self.assertTrue(all(r.get('macro_directive') is not None for r in raw.lower.preplant.recorder.rows))


if __name__=='__main__':unittest.main()
