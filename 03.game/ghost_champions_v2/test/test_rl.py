import contextlib
import copy
import io
import json
import os
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
        self.assertEqual(OBS_DIM,811+dict(BLOCKS)["player_abilities_status"]+dict(BLOCKS)["self_capabilities"])
        self.assertEqual(OBSERVATION_SCHEMA["version"],4)
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
        config.update(steps=20,evaluate_every_steps=10,tyg_confirmation_series=10,tyg_periodic_action="stop")
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

    def test_restored_opponent_snapshot_must_match_the_original_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot=Path(folder).resolve()
            original=str(training.ROOT/"concon_v1/controller.py")
            copied=str(snapshot/"concon_v1/controller.py")
            parent=dict(schema_hash=SCHEMA_HASH,verification=False,config=dict(evaluation_series=10),
                        frozen_inputs={original:"original"})
            with patch.dict(os.environ,GC_OPPONENT_SNAPSHOT=str(snapshot)):
                training.validate_evaluation_change(parent,dict(evaluation_series=2),{copied:"original"})
                with self.assertRaises(ValueError):
                    training.validate_evaluation_change(parent,dict(evaluation_series=2),{copied:"new model"})

    def test_periodic_and_final_respect_counts_and_formal_sample(self):
        for series,confirm in ((s,c) for s in (2,10) for c in (False,True)):
            with self.subTest(series=series,confirm=confirm),tempfile.TemporaryDirectory() as folder:
                config=json.loads((Path(__file__).resolve().parents[1]/"rl/training_config.json").read_text())
                config.update(steps=11,evaluate_every_steps=10,initial_evaluation_series=series,
                              final_evaluation_series=series,tyg_confirmation_series=series,tyg_periodic_action="stop")
                root=Path(folder)
                data_dir=root/"data/periodic"
                data_dir.mkdir(parents=True)
                np.savez_compressed(data_dir/"demonstrations.npz",obs=np.zeros((1,OBS_DIM),np.float32),
                    mask=np.ones((1,ACTION_DIM),np.bool_),teacher=np.array([0]),validation=np.array([False]))
                (data_dir/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,frozen_inputs={})))
                policy=ResidualPolicy()
                optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
                good=dict.fromkeys(("TYG","OMG","FRC","FNC","GG","SPS"),.9)
                policy.save(data_dir/"bc.pt")
                policy.save(data_dir/"latest.pt",step=10,episode=1,rates=good,last_evaluated_step=0,
                    optimizer_state_dict=optimizer.state_dict(),numpy_rng=np.random.default_rng(1).bit_generator.state,
                    python_rng=random.getstate(),torch_rng=torch.get_rng_state())
                screening=dict(complete=True,milestone_sample_complete=False,opponents_below_100_attack_rounds=list(good))
                results=[(good,{"complete":True}),(dict(good,TYG=.7) if confirm else good,screening)]
                if confirm:
                    results.append(({"TYG":.85},{"complete":True}))
                results.append((good,{"complete":True,"milestone_sample_complete":series>=10}))
                guardrail_metadata=[]
                original_save=ResidualPolicy.save
                def record_save(model,path,**metadata):
                    if Path(path).name=="last_passed_guardrail.pt":
                        guardrail_metadata.append(metadata)
                    return original_save(model,path,**metadata)
                with patch.object(training,"AI_ROOT",root),patch.object(training,"fingerprint",return_value={}), \
                     patch.object(training,"evaluate",side_effect=results) as evaluation, \
                     patch.object(training,"play_round",return_value=([{}],{"winner":"attacker"})) as rollout, \
                     patch.object(training,"optimize_ppo",return_value=.1),patch.object(ResidualPolicy,"save",new=record_save):
                    training.train("periodic",config,root/"snapshot",resume=True)
                rollout.assert_called_once()
                calls=evaluation.call_args_list
                self.assertEqual(calls[0].kwargs,dict(series_count=series,minimum_attack_rounds=100 if series>=10 else 1))
                self.assertEqual(calls[1].kwargs,dict(series_count=2,minimum_attack_rounds=1))
                if confirm:
                    self.assertEqual(calls[2].kwargs,dict(series_count=series,opponents=["TYG"],minimum_attack_rounds=100 if series>=10 else 1))
                self.assertEqual(calls[-1].kwargs,dict(series_count=series,minimum_attack_rounds=100 if series>=10 else 1))
                self.assertEqual([metadata["verified"] for metadata in guardrail_metadata],[False,series>=10])
                self.assertTrue(json.loads((root/"logs/periodic/status.json").read_text())["complete"])

    def test_periodic_warning_continues_without_confirmation_and_final_target_is_separate(self):
        for final_rate,final_complete in ((.5,True),(.9,True),(.9,False)):
            with self.subTest(final_rate=final_rate,final_complete=final_complete),tempfile.TemporaryDirectory() as folder:
                config=json.loads((training.AI_ROOT/"rl/training_config.json").read_text())
                config.update(steps=11,evaluate_every_steps=10,tyg_periodic_action="warn")
                root=Path(folder)
                data_dir=root/"data/warnings"
                data_dir.mkdir(parents=True)
                np.savez_compressed(data_dir/"demonstrations.npz",obs=np.zeros((1,OBS_DIM),np.float32),
                    mask=np.ones((1,ACTION_DIM),np.bool_),teacher=np.array([0]),validation=np.array([False]))
                (data_dir/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,frozen_inputs={})))
                policy=ResidualPolicy(hidden=16)
                optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
                good=dict.fromkeys(("TYG","OMG","FRC","FNC","GG","SPS"),.9)
                policy.save(data_dir/"bc.pt")
                policy.save(data_dir/"latest.pt",step=10,episode=1,rates=good,last_evaluated_step=0,
                    optimizer_state_dict=optimizer.state_dict(),numpy_rng=np.random.default_rng(1).bit_generator.state,
                    python_rng=random.getstate(),torch_rng=torch.get_rng_state())
                results=[(good,{"complete":True}),(dict(good,TYG=.5),{"complete":True}),
                         (dict(good,TYG=final_rate),{"complete":final_complete,"milestone_sample_complete":False})]
                with patch.object(training,"AI_ROOT",root),patch.object(training,"fingerprint",return_value={}), \
                     patch.object(training,"evaluate",side_effect=results) as evaluation, \
                     patch.object(training,"play_round",return_value=([{}],{"winner":"attacker"})) as rollout, \
                     patch.object(training,"optimize_ppo",return_value=.1),contextlib.redirect_stdout(io.StringIO()):
                    training.train("warnings",config,root/"snapshot",resume=True)
                rollout.assert_called_once()
                self.assertEqual(evaluation.call_count,3)  # Initial, pending periodic, final; no TYG confirmation.
                self.assertTrue(all("opponents" not in call.kwargs for call in evaluation.call_args_list))
                status=json.loads((root/"logs/warnings/status.json").read_text())
                self.assertEqual(status["complete"],final_complete)
                warnings=[json.loads(line) for line in (root/"logs/warnings/warnings.jsonl").read_text().splitlines()]
                self.assertEqual(warnings,[dict(step=10,warning="TYG_below_target",attack_rate=.5,target=.8)])
                _,latest=ResidualPolicy.load(data_dir/"latest.pt")
                self.assertFalse(latest["verified"])
                if final_complete:
                    self.assertTrue(status["training_complete"])
                    self.assertEqual(status["final_target_met"],final_rate>=.8)
                    self.assertEqual(status["last_evaluated_step"],11)
                    self.assertEqual((data_dir/"last_passed_guardrail.pt").exists(),final_rate>=.8)
                else:
                    self.assertEqual(status["stop_reason"],"insufficient_evaluation_sample")
                    self.assertEqual(latest["last_evaluated_step"],10)

    def test_bc_only_continuation_accepts_partial_initial_but_checks_reused_files(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            data_dir=root/"data/new"
            data_dir.mkdir(parents=True)
            bc=data_dir/"bc.pt"
            bc.write_bytes(b"saved BC")
            evaluation=root/"logs/new/evaluation_bc"
            evaluation.mkdir(parents=True)
            series=evaluation/"series_TYG_000.json"
            series.write_text("{}")
            parent=dict(schema_hash=SCHEMA_HASH,verification=False,config=dict(evaluation_series=10),frozen_inputs={})
            continuation=dict(parent_manifest=parent,artifact_hashes={"bc.pt":hashlib.sha256(bc.read_bytes()).hexdigest()},
                initial_evaluation=None,reused_initial_evaluation=dict(folder=str(evaluation),
                    hashes={series.name:hashlib.sha256(series.read_bytes()).hexdigest()}))
            (data_dir/"continuation.json").write_text(json.dumps(continuation))
            self.assertEqual(training.load_continuation(data_dir,dict(evaluation_series=2),{}),continuation)
            series.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"Reused initial evaluation differs"):
                training.load_continuation(data_dir,dict(evaluation_series=2),{})

    def test_partial_evaluation_reuses_only_requested_series_and_rejects_policy_changes(self):
        import hashlib
        import ghost_champions_v2.tools.fork_evaluation_run as fork
        from tools.run_eval import RUNTIME_FILES
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,target=root/"old",root/"new"
            source.mkdir()
            snapshot=root/"runtime"
            snapshot.mkdir()
            for name in RUNTIME_FILES:
                (snapshot/name).write_text("frozen")
            checkpoint=root/"bc.pt"
            checkpoint.write_bytes(b"weights")
            policy=root/"policy.py"
            policy.write_text("same policy")
            manifest=dict(controller="ghost_champions_v2",stage="entry",base_seed=20261007,python=sys.version,
                          opponents=["TYG"],series_count=10,source_hashes={"policy.py":fork.digest(policy)},
                          runtime_snapshot=str(snapshot),runtime_data_hashes={n:fork.digest(snapshot/n) for n in RUNTIME_FILES},
                          residual_checkpoint=dict(path="original",sha256=fork.digest(checkpoint)))
            (source/"manifest.json").write_text(json.dumps(manifest))
            for i in range(5):
                (source/f"series_TYG_{i:03d}.json").write_text(json.dumps(dict(index=i)))
            with patch.object(fork,"ROOT",root):
                reuse=fork.reuse_initial_series(source,target,checkpoint,2)
                self.assertEqual(len(reuse["hashes"]),2)
                self.assertEqual(len(list(source.glob("series_*.json"))),5)
                self.assertEqual(len(list(target.glob("series_*.json"))),2)
                new=json.loads((target/"manifest.json").read_text())
                self.assertEqual(new["series_count"],2)
                self.assertEqual(new["minimum_attack_rounds"],1)
                policy.write_text("different policy")
                with self.assertRaisesRegex(ValueError,"policy source changed"):
                    fork.reuse_initial_series(source,root/"bad",checkpoint,2)

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

    def test_active_plant_is_locked_but_opponent_does_not_restrict_actions(self):
        self.char.plant_timer=1
        _,mask=candidates(self.char,self.state,self.tactics,([10,10],"PLANT"))
        self.assertEqual(np.flatnonzero(mask).tolist(),[3])
        self.char.plant_timer=0
        self.tactics.opponent="TYG"
        _,mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertGreater(mask.sum(),1)
        for opponent in ("OMG","FRC","FNC","GG","SPS","unknown"):
            self.tactics.opponent=opponent
            _,other=candidates(self.char,self.state,self.tactics,self.proposal)
            np.testing.assert_array_equal(mask,other)

    def test_plant_start_and_deadline_movement_cannot_be_replaced_by_model(self):
        self.char.has_spike=True
        self.state["grid"][10,10]=2
        _,mask=candidates(self.char,self.state,self.tactics,([10,10],"PLANT"))
        self.assertEqual(np.flatnonzero(mask).tolist(),[3])
        self.tactics.plant_commit=dict(holder="self",target=(10,10),reason="plant_deadline")
        _,mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertEqual(np.flatnonzero(mask).tolist(),[0])
        self.tactics.plant_commit["holder"]="another"
        _,mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertGreater(mask.sum(),1)
        self.tactics.plant_commit["holder"]="self"
        self.char.has_spike=False
        self.state.update(is_planted=True,planted_pos=(10,10))
        _,mask=candidates(self.char,self.state,self.tactics,self.proposal)
        self.assertGreater(mask.sum(),1)

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
