import contextlib
import copy
import io
import json
import random
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.rl.observation import ObservationEncoder,OBS_DIM,BLOCKS,OBSERVATION_SCHEMA,SCHEMA_HASH
from ghost_champions_v2.rl.actions import ACTION_DIM,candidates
from ghost_champions_v2.rl.policy import ResidualPolicy
from ghost_champions_v2.rl.rollout import potential,pfsp_probabilities,shaping_coefficient,play_round,freeze_modules
from ghost_champions_v2.rl.training import bc_loss,optimize_ppo
import ghost_champions_v2.rl.training as training


class RLTests(unittest.TestCase):
    def setUp(self):
        self.config=load_config()
        self.tactics=AttackerTactics(self.config)
        self.char=NS(name="self",team="A",pos=[10,10],is_alive=True,hp=100,
                     ability_name="HUNT",has_spike=False,plant_timer=0)
        self.state=dict(grid=np.zeros((26,44),int),chars=[self.char],battle_tick=1,
                        is_planted=False,enemy_roster=[],smoke_cells=set())
        self.proposal=([10,11],"MOVE",{"move_step_limit":1})
        self.tactics.observe(self.char,self.state)

    def test_hidden_enemy_coordinates_and_hp_do_not_change_observation(self):
        hidden=NS(name="hidden",team="D",pos=[-1,-1],hp=100,is_alive=True,position_known=False)
        self.state["chars"].append(hidden)
        first=ObservationEncoder().encode(self.char,self.state,self.tactics,self.proposal)
        hidden.pos,hidden.hp=[1,1],1
        hidden.facing,hidden.blind_remaining="S",10
        second=ObservationEncoder().encode(self.char,self.state,self.tactics,self.proposal)
        np.testing.assert_array_equal(first,second)
        self.assertEqual(first.shape,(OBS_DIM,))

    def test_public_defuse_and_own_status_are_inputs_without_revealing_the_defuser(self):
        encoder=ObservationEncoder()
        self.state.update(is_planted=True,planted_pos=(10,10),defender_defuse_info={"hidden":(0,6)})
        before=encoder.encode(self.char,self.state,self.tactics,self.proposal)
        self.state["defender_defuse_info"]={"hidden":(3,6)}
        self.char.facing,self.char.blind_remaining="W",10
        after=encoder.encode(self.char,self.state,self.tactics,self.proposal)
        offset=0
        blocks={}
        for name,size in BLOCKS:
            blocks[name]=slice(offset,offset+size)
            offset+=size
        self.assertEqual(OBS_DIM,448)
        self.assertEqual(OBSERVATION_SCHEMA["version"],2)
        np.testing.assert_allclose(before[blocks["public_defuse"]],[0,0])
        np.testing.assert_allclose(after[blocks["public_defuse"]],[.2,.5])
        self.assertFalse(np.array_equal(before[blocks["self_status"]],after[blocks["self_status"]]))
        np.testing.assert_array_equal(before[blocks["units"]],after[blocks["units"]])
        np.testing.assert_array_equal(before[blocks["enemy_roster"]],after[blocks["enemy_roster"]])

    def test_flash_candidates_ignore_zero_progress_but_respond_to_a_hidden_defuse_tap(self):
        self.char.ability_name,self.char.flash_charges="FLASH",1
        self.state.update(is_planted=True,planted_pos=(10,10),defender_defuse_info={"hidden":(0,6)})
        self.tactics.observe(self.char,self.state)
        _,idle_mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertFalse(idle_mask[19])
        self.state["defender_defuse_info"]={"hidden":(1,6)}
        choices,active_mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertTrue(active_mask[19])
        self.assertEqual(choices[19][1]["ability"],"FLASH")

    def test_resume_checks_pending_guardrail_before_another_training_round(self):
        config=json.loads((Path(__file__).resolve().parents[1]/"rl/training_config.json").read_text())
        config.update(steps=20,evaluate_every_steps=10)
        policy=ResidualPolicy()
        optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
        good=dict.fromkeys(("TYG","OMG","FRC","FNC","GG","SPS"),.9)
        bad=dict(good,TYG=.7)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data_dir=root/"data"/"resume_guard"
            data_dir.mkdir(parents=True)
            np.savez_compressed(data_dir/"demonstrations.npz",obs=np.zeros((1,OBS_DIM),np.float32),
                mask=np.ones((1,ACTION_DIM),np.bool_),teacher=np.array([0]),validation=np.array([False]))
            (data_dir/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,frozen_inputs={})))
            policy.save(data_dir/"bc.pt")
            policy.save(data_dir/"latest.pt",step=10,episode=1,rates=good,last_evaluated_step=0,
                optimizer_state_dict=optimizer.state_dict(),numpy_rng=np.random.default_rng(1).bit_generator.state,
                python_rng=random.getstate(),torch_rng=torch.get_rng_state())
            pending=data_dir/"evaluated_0000010.pt"
            policy.save(pending,step=10,verified=False)
            previous_bytes=pending.read_bytes()
            with patch.object(training,"AI_ROOT",root),patch.object(training,"fingerprint",return_value={}), \
                 patch.object(training,"evaluate",side_effect=[(good,{"complete":True}),(bad,{"complete":True}),(bad,{"complete":True})]) as evaluation, \
                 patch.object(training,"play_round") as rollout:
                training.train("resume_guard",config,root/"snapshot",resume=True)
            rollout.assert_not_called()
            self.assertEqual(evaluation.call_count,3)
            self.assertEqual(evaluation.call_args.kwargs["opponents"],["TYG"])
            self.assertEqual(evaluation.call_args.kwargs["series_count"],10)
            self.assertEqual(pending.read_bytes(),previous_bytes)
            status=json.loads((root/"logs/resume_guard/status.json").read_text())
            self.assertEqual(status["phase"],"stopped")
            self.assertEqual(status["step"],10)
            self.assertEqual(status["stop_reason"],"TYG_regression")

    def test_evaluation_change_rejects_policy_or_learning_changes(self):
        parent=dict(schema_hash=SCHEMA_HASH,verification=False,
                    config=dict(evaluation_series=10,gamma=.99),
                    frozen_inputs={str(training.AI_ROOT/"rl/training.py"):"old","policy.py":"fixed"})
        config=dict(evaluation_series=2,initial_evaluation_series=10,final_evaluation_series=10,gamma=.99)
        frozen={str(training.AI_ROOT/"rl/training.py"):"new","policy.py":"fixed"}
        training.validate_evaluation_change(parent,config,frozen)
        with self.assertRaises(ValueError):
            training.validate_evaluation_change(parent,dict(config,gamma=.95),frozen)
        with self.assertRaises(ValueError):
            training.validate_evaluation_change(parent,config,dict(frozen,**{"policy.py":"changed"}))

    def test_opponent_counts_and_plant_elapsed_are_model_inputs(self):
        encoder=ObservationEncoder()
        first=encoder.encode(self.char,self.state,self.tactics,self.proposal)
        self.tactics.opponent="FRC"
        self.tactics.observed_defenders_by_axis["A"]=2
        self.state.update(is_planted=True,planted_pos=(8,3))
        second=encoder.encode(self.char,self.state,self.tactics,self.proposal)
        self.state["battle_tick"]=11
        third=encoder.encode(self.char,self.state,self.tactics,self.proposal)
        self.assertFalse(np.array_equal(first,second))
        self.assertFalse(np.array_equal(second,third))
        self.assertEqual(second[2],1)
        self.assertAlmostEqual(second[8],.4)

    def test_checkpoint_schema_and_loading_do_not_change_match_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"policy.pt"
            ResidualPolicy().save(path)
            before=torch.get_rng_state().clone()
            ResidualPolicy.load(path)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            payload=torch.load(path,weights_only=True)
            payload["schema_hash"]="old"
            torch.save(payload,path)
            with self.assertRaises(ValueError):
                ResidualPolicy.load(path)

    def test_active_plant_and_tyg_have_only_the_rule_action(self):
        self.char.plant_timer=1
        _,mask=candidates(self.char,self.state,self.tactics,([10,10],"PLANT"))
        self.assertEqual(np.flatnonzero(mask).tolist(),[3])
        self.char.plant_timer=0
        self.tactics.opponent="TYG"
        _,mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertEqual(np.flatnonzero(mask).tolist(),[0])

    def test_potential_telescopes_and_vanishes_at_terminal(self):
        gamma=.99
        values=[potential(5,5,False),potential(4,5,False),potential(4,5,True),potential(0,5,False,True)]
        total=sum(gamma**i*(gamma*values[i+1]-values[i]) for i in range(3))
        self.assertAlmostEqual(total,-values[0])
        self.assertEqual(shaping_coefficient(300000),0)
        self.assertEqual(shaping_coefficient(400000),0)

    def test_pfsp_prefers_low_win_rate_and_handles_all_wins(self):
        probabilities=pfsp_probabilities(dict(TYG=.8,OMG=.2))
        self.assertAlmostEqual(probabilities[1]/probabilities[0],16)
        np.testing.assert_allclose(pfsp_probabilities(dict.fromkeys(("TYG","OMG","FRC","FNC","GG","SPS"),1)),np.full(6,1/6))

    def test_ability_bc_gradient_remains_after_movement_bc_decays(self):
        policy=ResidualPolicy()
        data=dict(obs=np.zeros((2,OBS_DIM),np.float32),mask=np.ones((2,ACTION_DIM),np.bool_),teacher=np.array([2,0]))
        config=json.loads((Path(__file__).resolve().parents[1]/"rl/training_config.json").read_text())
        loss=bc_loss(policy,data,np.array([0]),300000,config)
        loss.backward()
        self.assertGreater(abs(policy.actor.bias.grad[2].item()),0)
        self.assertEqual(bc_loss(policy,data,np.array([1]),300000,config).item(),0)

    def test_real_round_collects_fresh_bc_and_ppo_updates_only_new_policy(self):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            rows,stats=play_round("FRC",20261010)
        self.assertGreater(len(rows),0)
        self.assertEqual(stats["round_records"],1)
        self.assertTrue(stats["opponent_model_digest"])
        self.assertGreater(stats["opponent_modules"],0)
        self.assertLessEqual(stats["recon_rewarded_new"],stats["recon_new"])
        data=dict(obs=np.stack([r["obs"] for r in rows]),mask=np.stack([r["mask"] for r in rows]),
                  teacher=np.array([r["teacher"] for r in rows]),validation=np.zeros(len(rows),np.bool_))
        policy=ResidualPolicy()
        with torch.no_grad():
            dist,value=policy(torch.from_numpy(data["obs"]),torch.from_numpy(data["mask"]))
            logp=dist.log_prob(torch.tensor([r["action"] for r in rows]))
        for i,row in enumerate(rows):
            row.update(logp=logp[i].item(),value=value[i].item())
        config=json.loads((Path(__file__).resolve().parents[1]/"rl/training_config.json").read_text())
        before=policy.actor.weight.detach().clone()
        loss=optimize_ppo(policy,torch.optim.Adam(policy.parameters(),lr=.001),rows,data,100,config,np.random.default_rng(1))
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before,policy.actor.weight))
        with contextlib.redirect_stdout(io.StringIO()):
            next_rows,_=play_round("GG",20261011,policy=policy,mode="stochastic")
        self.assertGreater(len(next_rows),0)
        self.assertTrue(all(p.requires_grad for p in policy.parameters()))

    def test_freezing_opponent_does_not_follow_world_references_to_the_attacker(self):
        learner,opponent=ResidualPolicy(),ResidualPolicy()
        world=NS(policy=learner)
        controller=NS(real_game=world,model=opponent,other_world_reference=world)
        modules=freeze_modules(controller,excluded=(world,learner))
        self.assertEqual(modules,[opponent])
        self.assertTrue(all(p.requires_grad for p in learner.parameters()))
        self.assertTrue(all(not p.requires_grad for p in opponent.parameters()))


if __name__=="__main__":unittest.main()
