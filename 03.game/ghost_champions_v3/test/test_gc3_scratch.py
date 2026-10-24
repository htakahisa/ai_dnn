"""Behavior checks for the user-specified map and scratch-policy contract."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from map_data_defender_setup import get_setup_mask
from ghost_champions_v3 import AI_ROOT
from ghost_champions_v3.gc3_engine_actions import legal_actions, MODES, decode_command
from ghost_champions_v3.gc3_observation import encode, map_hints
from ghost_champions_v3.gc3_policy import FreshPolicy, ScratchController
from ghost_champions_v3.gc3_rollout import Recorder
from ghost_champions_v3.gc3_spatial_policy import parse_map, hint_fields, sampled_distribution, phase_settings, SpatialCandidatePolicy
from ghost_champions_v3 import gc3_map_attacker_postplant as postplant

def fixture(team="A", setup=False):
    grid = np.array([[int(ch) for ch in row] for row in NEW_MAZE_STR.strip().splitlines()])
    permitted = np.asarray(get_setup_mask()) == 0 if setup else grid != 1
    point = next(tuple(p) for p in np.argwhere((grid != 1)&permitted)
                 if sum(0 <= p[0]+dr < grid.shape[0] and 0 <= p[1]+dc < grid.shape[1]
                        and grid[p[0]+dr, p[1]+dc] != 1 for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))) >= 2)
    char = SimpleNamespace(name="own", team=team, pos=list(point), is_alive=True, hp=100, max_hp=100,
                           facing="S", ability_name="SMOKE", smoke_charges=2,
                           ultimate_name="ESCAPE", ultimate_points=0, ultimate_cost=6, has_spike=False)
    state = dict(grid=grid, chars=[char], is_planted=not setup, planted_pos=(8,40),
                 defender_setup_active=setup, public_effects=[], smoke_cells=set(), available_orbs=set())
    return char, state

class ScratchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_decorative_numbers_are_ignored(self):
        a,b = "12340\n00560", "10000\n00560"
        np.testing.assert_array_equal(parse_map(a),parse_map(b))
        for left,right in zip(hint_fields(a,1.5), hint_fields(b,1.5)):
            np.testing.assert_array_equal(left,right)
        char,state=fixture()
        hints=map_hints(char,state)
        original=encode(char,state,hints)
        decorative=postplant.NEW_MAZE_STR.translate(str.maketrans("234","000"))
        with patch.object(postplant,"NEW_MAZE_STR",decorative):
            np.testing.assert_array_equal(map_hints(char,state),hints)
            np.testing.assert_array_equal(encode(char,state,map_hints(char,state)),original)

    def test_invalid_map_rejected(self):
        for text in ("", "00\n0", "007"):
            with self.assertRaises(ValueError):
                parse_map(text)

    def test_candidates_are_soft_and_can_be_overridden_by_policy(self):
        spatial=SpatialCandidatePolicy(2,4)
        with torch.no_grad():
            spatial.position.weight.zero_()
            spatial.position.bias.zero_()
        hints=np.array([[1,0,0,0],[0,1,0,0]],dtype=np.float32)
        d=spatial.distribution(torch.zeros(2),np.ones(4,bool),kind="position",phase="postplant",hints=hints)
        self.assertTrue(bool((d.probs>0).all()))
        self.assertGreater(float(d.probs[0].detach()),float(d.probs[1].detach()))
        with torch.no_grad():
            spatial.position.bias[2]=2
        d=spatial.distribution(torch.zeros(2),np.ones(4,bool),kind="position",phase="postplant",hints=hints)
        self.assertGreater(float(d.probs[2].detach()),float(d.probs[0].detach()))
        draws=d.sample((200,)).tolist()
        self.assertGreater(len(set(draws)),1)

    def test_defender_initial_keeps_more_randomness(self):
        logits=torch.tensor([8.,0.,-8.])
        td,md,*_=phase_settings("defender")
        tp,mp,*_=phase_settings("postplant")
        d=sampled_distribution(logits,[True]*3,temperature=td,uniform_mix=md)
        p=sampled_distribution(logits,[True]*3,temperature=tp,uniform_mix=mp)
        self.assertGreater(float(d.entropy()),float(p.entropy()))
        self.assertTrue(bool((d.probs>=md/3-1e-7).all()))

    def test_only_physical_movement_restrictions(self):
        char,state=fixture()
        masks,_=legal_actions(char,state)
        grid=state["grid"]; start=np.array(char.pos)
        for i in np.flatnonzero(masks["position"]):
            self.assertLessEqual(int(np.abs(np.unravel_index(i,grid.shape)-start).sum()),1)
        np.testing.assert_array_equal(masks["watch"],(grid!=1).ravel())
        self.assertTrue(masks["mode"][MODES.index("ABILITY")])
        # Incoming threats and smoke rays do not install an automatic dodge or cast veto.
        altered=dict(state, public_effects=[SimpleNamespace(phase="warning",position=tuple(char.pos),cells=[])],
                     smoke_cells={tuple(char.pos)})
        for kind,mask in masks.items():
            np.testing.assert_array_equal(mask,legal_actions(char,altered)[0][kind])
        self.assertFalse(np.array_equal(encode(char,state,map_hints(char,state)),
                                       encode(char,altered,map_hints(char,altered))))

    def test_physical_seal_and_root_are_respected(self):
        char,state=fixture()
        char.ability_seal_remaining=2
        char.fate_loom_remaining=2
        masks,_=legal_actions(char,state)
        self.assertFalse(masks["mode"][MODES.index("ABILITY")])
        self.assertEqual(int(masks["position"].sum()),1)
        self.assertTrue(masks["mode"][MODES.index("WATCH")])

    def test_no_private_enemy_fields_in_initial_setup(self):
        char,state=fixture("D",True)
        enemy=SimpleNamespace(name="enemy",team="A",pos=[1,18],is_alive=True,hp=100,position_known=True)
        state["chars"].append(enemy)
        before=encode(char,state,map_hints(char,state))
        # Even a raw Setup state cannot expose live enemy positions or HP.
        enemy.pos=[10,20];enemy.hp=1;enemy.is_alive=False
        np.testing.assert_array_equal(before,encode(char,state,map_hints(char,state)))

    def test_recorded_likelihood_matches_batched_likelihood(self):
        char,state=fixture()
        policy=FreshPolicy.fresh("attacker",37)
        records=[policy.select(char,state)[1] for _ in range(40)]
        logp,entropy,value=policy.evaluate_records(records)
        np.testing.assert_allclose(logp.detach(),[r["logp"] for r in records],atol=2e-5)
        for index,record in enumerate(records[:4]):
            single=policy.evaluate_record(record)
            self.assertAlmostEqual(float(single[0].detach()),float(logp[index].detach()),places=5)
        self.assertTrue(bool(torch.isfinite(entropy).all() and torch.isfinite(value).all()))
        # Backpropagation validates trainability without applying any optimizer update.
        before={key:val.clone() for key,val in policy.state_dict().items()}
        (logp.sum()+value.sum()).backward()
        for parameter in policy.parameters():
            if parameter.grad is not None:
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
        self.assertGreater(float(policy.mode.weight.grad.abs().sum()),0)
        self.assertGreater(float(policy.spatial.position.weight.grad.abs().sum()),0)
        self.assertGreater(float(policy.spatial.watch.weight.grad.abs().sum()),0)
        self.assertGreater(float(policy.spatial.target.weight.grad.abs().sum()),0)
        for key,val in policy.state_dict().items():
            self.assertTrue(torch.equal(before[key],val))

    def test_initial_setup_decisions_are_recorded_for_learning(self):
        char,state=fixture("D",True)
        recorder=Recorder()
        controller=ScratchController(role="defender",seed=73,recorder=recorder,record_phase="search")
        controller.decide_move(char,state)
        self.assertEqual(len(recorder.rows),1)
        self.assertEqual(recorder.rows[0]["phase"],"defender")

    def test_terminal_reward_credits_setup_and_dead_agents(self):
        recorder=Recorder()
        recorder.record("living",dict(value=0.,phase="defender"))
        recorder.record("dead",dict(value=0.,phase="defender"))
        recorder.reward(0.,1.,False)
        recorder.record("living",dict(value=0.,phase="standard"))
        recorder.reward(1.,1.,True)
        rows=recorder.advantages(1.)
        self.assertEqual([r["return"] for r in rows],[1.,1.,1.])
        self.assertEqual([r["reward"] for r in rows],[0.,1.,1.])

    def test_fresh_initialization_and_checkpoint_reject_inheritance(self):
        torch.manual_seed(22);rng=torch.get_rng_state().clone()
        policy=FreshPolicy.fresh("attacker",19)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        other=FreshPolicy.fresh("attacker",19)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(policy.parameters(),other.parameters())))
        with TemporaryDirectory() as temp:
            path=Path(temp)/"initial.pt"
            policy.save(path,trained=False,step=0)
            restored,meta=FreshPolicy.load(path,role="attacker")
            self.assertFalse(meta["trained"])
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            with self.assertRaises(ValueError):
                FreshPolicy.load(path,role="defender")
            torch.save(dict(policy_type="legacy_gc_model"),path)
            with self.assertRaises(ValueError):
                FreshPolicy.load(path)

    def test_runtime_has_no_previous_gc_dependency(self):
        for path in AI_ROOT.glob("*.py"):
            tree=ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertFalse((node.module or "").startswith(("ghost_champions_v2","gc_v1")),path.name)
                if isinstance(node,ast.Import):
                    self.assertFalse(any(alias.name.startswith(("ghost_champions_v2","gc_v1")) for alias in node.names),path.name)

if __name__=="__main__":
    unittest.main()
