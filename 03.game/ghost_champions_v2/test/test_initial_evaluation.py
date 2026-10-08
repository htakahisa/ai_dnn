"""Reuse completed historical baselines without relabeling them as new BC evals."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
import random
from pathlib import Path
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.rl.initial_evaluation import (
    REFERENCE_NAME, bind_initial_evaluation, load_initial_evaluation,
)
from ghost_champions_v2.rl.observation import OBS_DIM, SCHEMA_HASH
from ghost_champions_v2.rl.actions import ACTION_DIM
from ghost_champions_v2.rl.policy import ResidualPolicy
from ghost_champions_v2.rl.rollout import OPPONENTS
import ghost_champions_v2.rl.training as training


class InitialEvaluationTests(unittest.TestCase):
    def source(self, root, *, complete=True):
        folder = root / "logs/old/evaluation_bc"
        folder.mkdir(parents=True)
        checkpoint = root / "old_bc.pt"
        torch.save(dict(schema_hash="previous_observation_schema"), checkpoint)
        manifest = dict(controller="ghost_champions_v2", opponents=list(OPPONENTS), series_count=2,
                        runtime_snapshot="old_game_definitions", runtime_data_hashes={"game_core.py": "old"},
                        residual_checkpoint=dict(path=str(checkpoint),
                            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        summary = {name: dict(n=10, wins=5, attack_rate=.5) for name, _ in OPPONENTS.values()}
        for name, data in (("manifest.json", manifest), ("status.json", dict(complete=complete, missing_series=[])),
                           ("summary.json", summary)):
            (folder / name).write_text(json.dumps(data), encoding="utf-8")
        for code in OPPONENTS:
            for index in range(2):
                (folder / f"series_{code}_{index:03d}.json").write_text("{}")
        return folder

    def test_reference_retains_old_model_and_schema_without_copying_bc_or_series(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            data_dir = root / "data/new"
            reference = bind_initial_evaluation(data_dir, source)
            self.assertEqual(load_initial_evaluation(data_dir), reference)
            self.assertEqual(reference["source_schema_hash"], "previous_observation_schema")
            self.assertFalse(reference["evaluates_new_bc"])
            self.assertFalse(reference["initial_guardrail_passed"])
            self.assertEqual(reference["scheduled_series"], 12)
            self.assertEqual(list(data_dir.iterdir()), [data_dir / REFERENCE_NAME])
            self.assertEqual(bind_initial_evaluation(data_dir, source), reference)

    def test_incomplete_or_missing_series_and_invalid_results_are_rejected(self):
        for invalid in ("incomplete", "series", "results", "opponents", "checkpoint"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.source(root, complete=invalid != "incomplete")
                if invalid == "series":
                    (source / "series_TYG_001.json").unlink()
                elif invalid == "results":
                    summary = json.loads((source / "summary.json").read_text())
                    summary["Touyama Gaming"]["attack_rate"] = float("nan")
                    (source / "summary.json").write_text(json.dumps(summary))
                elif invalid == "opponents":
                    manifest = json.loads((source / "manifest.json").read_text())
                    manifest["opponents"] = ["TYG"]
                    (source / "manifest.json").write_text(json.dumps(manifest))
                elif invalid == "checkpoint":
                    (root / "old_bc.pt").write_bytes(b"different")
                with self.assertRaises(ValueError):
                    bind_initial_evaluation(root / "data/new", source)

    def test_reference_rejects_mutated_logs_and_started_run_cannot_change_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            data_dir = root / "data/new"
            bind_initial_evaluation(data_dir, source)
            series = source / "series_OMG_001.json"
            series.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                load_initial_evaluation(data_dir)
            (data_dir / "manifest.json").write_text("{}")
            (data_dir / REFERENCE_NAME).unlink()
            with self.assertRaisesRegex(ValueError, "existing training run"):
                bind_initial_evaluation(data_dir, source)

    def test_new_bc_collects_fresh_data_and_skips_initial_eval_even_when_old_tyg_is_low(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.source(root)
            data_dir = root / "data/new"
            bind_initial_evaluation(data_dir, source)
            config = json.loads((training.AI_ROOT / "rl/training_config.json").read_text())
            config.update(steps=1, evaluate_every_steps=10)
            policy = ResidualPolicy(hidden=16)
            data = dict(obs=np.zeros((1, OBS_DIM), np.float32), mask=np.ones((1, ACTION_DIM), np.bool_),
                        teacher=np.array([0]), validation=np.array([False]))
            rates = dict.fromkeys(OPPONENTS, .9)
            with patch.object(training, "AI_ROOT", root), patch.object(training, "fingerprint", return_value={}), \
                 patch.object(training, "collect", return_value=data) as collect, \
                 patch.object(training, "train_bc", return_value=(policy, {})) as bc, \
                 patch.object(training, "evaluate", return_value=(rates, {"complete": True, "milestone_sample_complete": False})) as evaluate, \
                 patch.object(training, "play_round", return_value=([{}], {"winner": "attacker"})) as rollout, \
                 patch.object(training, "optimize_ppo", return_value=.1), redirect_stdout(StringIO()):
                training.train("new", config, root / "snapshot")
            collect.assert_called_once()
            bc.assert_called_once()
            rollout.assert_called_once()
            evaluate.assert_called_once()  # Final evaluation is still required.
            self.assertEqual(evaluate.call_args.args[1].name, "evaluation_0000001")
            status = json.loads((root / "logs/new/status.json").read_text())
            self.assertTrue(status["complete"])
            self.assertFalse(status["initial_evaluation_reexecuted"])
            self.assertFalse(status["initial_evaluates_new_bc"])
            self.assertEqual(status["initial_evaluation_rates"]["TYG"], .5)
            manifest = json.loads((data_dir / "manifest.json").read_text())
            self.assertEqual(manifest["schema_hash"], SCHEMA_HASH)
            self.assertIn("initial_evaluation_reference_sha256", manifest)
            checkpoint = torch.load(data_dir / "bc.pt", weights_only=True)
            self.assertEqual(checkpoint["schema_hash"], SCHEMA_HASH)

    def test_periodic_tyg_guardrail_is_not_bypassed_by_historical_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bind_initial_evaluation(root / "data/new", self.source(root))
            config = json.loads((training.AI_ROOT / "rl/training_config.json").read_text())
            config.update(steps=2, evaluate_every_steps=1,tyg_periodic_action="stop")
            data = dict(obs=np.zeros((1, OBS_DIM), np.float32), mask=np.ones((1, ACTION_DIM), np.bool_),
                        teacher=np.array([0]), validation=np.array([False]))
            bad = dict.fromkeys(OPPONENTS, .5)
            with patch.object(training, "AI_ROOT", root), patch.object(training, "fingerprint", return_value={}), \
                 patch.object(training, "collect", return_value=data), \
                 patch.object(training, "train_bc", return_value=(ResidualPolicy(hidden=16), {})), \
                 patch.object(training, "evaluate", return_value=(bad, {"complete": True})) as evaluation, \
                 patch.object(training, "play_round", return_value=([{}], {"winner": "attacker"})) as rollout, \
                 patch.object(training, "optimize_ppo", return_value=.1), redirect_stdout(StringIO()):
                training.train("new", config, root / "snapshot")
            rollout.assert_called_once()
            self.assertEqual(evaluation.call_count, 2)  # Periodic six opponents + TYG confirmation.
            status = json.loads((root / "logs/new/status.json").read_text())
            self.assertEqual(status["stop_reason"], "TYG_regression")
            self.assertFalse(status["complete"])

    def test_recovery_forks_saved_bc_without_recollecting_or_repeating_initial_evaluation(self):
        import ghost_champions_v2.tools.fork_evaluation_run as fork
        for periodic_failure in (False, True):
            with self.subTest(periodic_failure=periodic_failure), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)
                initial=self.source(root)
                checkpoint=root/"old_bc.pt"
                ResidualPolicy(hidden=16).save(checkpoint,training_stage="bc",step=0)
                evaluation_manifest=json.loads((initial/"manifest.json").read_text())
                evaluation_manifest["residual_checkpoint"]["sha256"]=fork.digest(checkpoint)
                (initial/"manifest.json").write_text(json.dumps(evaluation_manifest))
                config=json.loads((training.AI_ROOT/"rl/training_config.json").read_text())
                config.update(steps=2 if periodic_failure else 1,evaluate_every_steps=1 if periodic_failure else 10,
                              tyg_periodic_action="stop")
                (root/"rl").mkdir()
                (root/"rl/training_config.json").write_text(json.dumps(config))
                snapshot=root/"runtime_saved"
                snapshot.mkdir()
                (snapshot/"game_core.py").write_text("frozen runtime")
                frozen={str(snapshot/"game_core.py"):fork.digest(snapshot/"game_core.py")}
                old=root/"data/old"
                old.mkdir(parents=True)
                shutil.copy2(checkpoint,old/"bc.pt")
                parent=dict(schema_hash=SCHEMA_HASH,verification=False,config=config,frozen_inputs=frozen)
                (old/"manifest.json").write_text(json.dumps(parent))
                data=dict(obs=np.zeros((1,OBS_DIM),np.float32),mask=np.ones((1,ACTION_DIM),np.bool_),
                          teacher=np.array([0]),validation=np.array([False]))
                np.savez_compressed(old/"demonstrations.npz",**data)
                (old/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,frozen_inputs=frozen)))
                original={p:fork.digest(p) for folder in (old,initial) for p in folder.iterdir()}
                with patch.object(fork,"AI_ROOT",root), patch.object(training,"fingerprint",return_value=frozen), \
                     patch.object(sys,"argv",["fork","--source","old","--run","new","--reuse-completed-initial"]), \
                     patch.object(sys,"path",list(sys.path)), patch.dict(os.environ,{},clear=False), redirect_stdout(StringIO()):
                    fork.main()
                prepared=json.loads((root/"logs/new/status.json").read_text())
                self.assertEqual(prepared["phase"],"ready_for_rl")
                self.assertFalse(prepared["initial_guardrail_passed"])
                continuation=json.loads((root/"data/new/continuation.json").read_text())
                self.assertEqual(continuation["runtime_snapshot"],str(snapshot))
                self.assertEqual(len(continuation["initial_evaluation"]["hashes"]),15)
                rates=dict.fromkeys(OPPONENTS,.5 if periodic_failure else .9)
                with patch.object(training,"AI_ROOT",root), patch.object(training,"fingerprint",return_value=frozen), \
                     patch.object(training,"collect") as collect, patch.object(training,"train_bc") as bc, \
                     patch.object(training,"evaluate",return_value=(rates,{"complete":True})) as evaluate, \
                     patch.object(training,"play_round",return_value=([{}],{"winner":"attacker"})) as rollout, \
                     patch.object(training,"optimize_ppo",return_value=.1), redirect_stdout(StringIO()):
                    training.train("new",config,snapshot,phase="rl",resume=True)
                collect.assert_not_called()
                bc.assert_not_called()
                rollout.assert_called_once()
                self.assertEqual(evaluate.call_count,2 if periodic_failure else 1)
                self.assertTrue(all(call.args[1].name!="evaluation_bc" for call in evaluate.call_args_list))
                status=json.loads((root/"logs/new/status.json").read_text())
                self.assertFalse(status["initial_evaluation_reexecuted"])
                self.assertEqual(status["complete"],not periodic_failure)
                if periodic_failure:
                    self.assertEqual(status["stop_reason"],"TYG_regression")
                self.assertEqual(original,{p:fork.digest(p) for p in original})
                self.assertEqual(fork.digest(old/"bc.pt"),fork.digest(root/"data/new/bc.pt"))

    def test_fork_of_continuation_keeps_original_bc_and_reuses_completed_ppo_evaluation(self):
        import ghost_champions_v2.tools.fork_evaluation_run as fork
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            initial=self.source(root)
            policy=ResidualPolicy(hidden=16)
            checkpoint=root/"old_bc.pt"
            policy.save(checkpoint,training_stage="bc",step=0)
            initial_manifest=json.loads((initial/"manifest.json").read_text())
            initial_manifest["residual_checkpoint"]["sha256"]=fork.digest(checkpoint)
            (initial/"manifest.json").write_text(json.dumps(initial_manifest))
            config=json.loads((training.AI_ROOT/"rl/training_config.json").read_text())
            config.update(steps=2,evaluate_every_steps=1,tyg_periodic_action="stop")
            (root/"rl").mkdir()
            (root/"rl/training_config.json").write_text(json.dumps(dict(config,tyg_periodic_action="warn")))
            snapshot=root/"runtime_saved"
            snapshot.mkdir()
            (snapshot/"game_core.py").write_text("frozen runtime")
            orchestration=str(training.AI_ROOT/"rl/training.py")
            frozen={str(snapshot/"game_core.py"):fork.digest(snapshot/"game_core.py"),
                    orchestration:fork.digest(Path(orchestration))}
            bc_frozen=dict(frozen,**{orchestration:"ancestor training orchestration"})
            old=root/"data/old"
            old.mkdir(parents=True)
            shutil.copy2(checkpoint,old/"bc.pt")
            data=dict(obs=np.zeros((1,OBS_DIM),np.float32),mask=np.ones((1,ACTION_DIM),np.bool_),
                      teacher=np.array([0]),validation=np.array([False]))
            np.savez_compressed(old/"demonstrations.npz",**data)
            (old/"demonstrations.json").write_text(json.dumps(dict(schema_hash=SCHEMA_HASH,frozen_inputs=bc_frozen)))
            ancestor=dict(schema_hash=SCHEMA_HASH,verification=False,config=config,frozen_inputs=bc_frozen)
            ancestor_reference=bind_initial_evaluation(root/"ancestor_reference",initial)
            parent_continuation=dict(parent_manifest=ancestor,
                artifact_hashes={name:fork.digest(old/name) for name in ("bc.pt","demonstrations.json","demonstrations.npz")},
                initial_evaluation=dict(folder=str(initial),hashes=ancestor_reference["hashes"]),
                initial_evaluation_as_baseline=True)
            (old/"continuation.json").write_text(json.dumps(parent_continuation))
            parent=dict(schema_hash=SCHEMA_HASH,verification=False,config=config,frozen_inputs=frozen,
                        continuation_sha256=fork.digest(old/"continuation.json"))
            (old/"manifest.json").write_text(json.dumps(parent))
            optimizer=torch.optim.Adam(policy.parameters(),lr=config["learning_rate"])
            policy.save(old/"latest.pt",step=1,episode=8,rates=dict.fromkeys(OPPONENTS,.6),last_evaluated_step=0,
                optimizer_state_dict=optimizer.state_dict(),numpy_rng=np.random.default_rng(17).bit_generator.state,
                python_rng=random.getstate(),torch_rng=torch.get_rng_state())
            evaluated=old/"evaluated_0000001.pt"
            policy.save(evaluated,step=1,verified=False)
            completed=root/"logs/old/evaluation_0000001"
            shutil.copytree(initial,completed)
            evaluated_manifest=json.loads((completed/"manifest.json").read_text())
            evaluated_manifest["residual_checkpoint"]=dict(path=str(evaluated),sha256=fork.digest(evaluated))
            (completed/"manifest.json").write_text(json.dumps(evaluated_manifest))
            original={p:fork.digest(p) for folder in (old,initial,completed) for p in folder.iterdir()}
            with patch.object(fork,"AI_ROOT",root),patch.object(training,"fingerprint",return_value=frozen), \
                 patch.object(sys,"argv",["fork","--source","old","--run","new","--reuse-completed-evaluation"]), \
                 patch.object(sys,"path",list(sys.path)),patch.dict(os.environ,{},clear=False),redirect_stdout(StringIO()):
                fork.main()
            new=root/"data/new"
            manifest=json.loads((new/"manifest.json").read_text())
            continuation=training.load_continuation(new,manifest["config"],frozen)
            self.assertEqual(continuation["bc_frozen_inputs"],bc_frozen)
            self.assertEqual(continuation["pending_evaluation"]["step"],1)
            self.assertTrue(continuation["initial_evaluation_as_baseline"])
            bad=dict.fromkeys(OPPONENTS,.5)
            with patch.object(training,"AI_ROOT",root),patch.object(training,"fingerprint",return_value=frozen), \
                 patch.object(training,"collect") as collect,patch.object(training,"train_bc") as bc, \
                 patch.object(training,"evaluate",return_value=(bad,{"complete":True})) as evaluate, \
                 patch.object(training,"play_round",return_value=([{}],{"winner":"attacker"})) as rollout, \
                 patch.object(training,"optimize_ppo",return_value=.1),redirect_stdout(StringIO()):
                training.train("new",manifest["config"],snapshot,phase="rl",resume=True)
            collect.assert_not_called()
            bc.assert_not_called()
            evaluate.assert_called_once()  # Only the final evaluation; initial and pending PPO eval are reused.
            self.assertEqual(evaluate.call_args.args[1].name,"evaluation_0000002")
            rollout.assert_called_once()
            self.assertEqual(rollout.call_args.kwargs["step"],1)
            self.assertEqual(rollout.call_args.args[1],config["seed"]+1000000+8)
            status=json.loads((root/"logs/new/status.json").read_text())
            self.assertTrue(status["complete"])
            self.assertFalse(status["final_target_met"])
            self.assertEqual(status["last_evaluated_step"],2)
            self.assertFalse((new/"last_passed_guardrail.pt").exists())
            _,latest=ResidualPolicy.load(new/"latest.pt")
            self.assertEqual(latest["episode"],9)
            self.assertEqual(original,{p:fork.digest(p) for p in original})
            (completed/"series_TYG_000.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"PPO evaluation artifacts changed"):
                training.load_continuation(new,manifest["config"],frozen)


if __name__ == "__main__":
    unittest.main()
