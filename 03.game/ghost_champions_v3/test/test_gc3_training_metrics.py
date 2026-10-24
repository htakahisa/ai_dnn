"""Check actual transition logging, numerical meaning, and read-only behavior."""
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from ghost_champions_v3.gc3_policy import FreshPolicy
from ghost_champions_v3.gc3_rollout import Recorder
from ghost_champions_v3.tools.gc3_training_metrics import record_training_metrics, without_logging_blocks, reconcile_resume_logs
from test_gc3_scratch import fixture

class EntropyFixture:
    def evaluate_records(self, rows):
        return torch.zeros(len(rows)),torch.tensor([row["known_entropy"] for row in rows]),torch.zeros(len(rows))

class TrainingMetricsTests(unittest.TestCase):
    def test_actual_discounted_rewards_and_gae_targets_are_logged(self):
        recorder=Recorder()
        recorder.record("living",dict(value=.2,phase="defender",known_entropy=.2))
        recorder.record("dead",dict(value=.4,phase="defender",known_entropy=.4))
        recorder.reward(0.,.9,False)
        recorder.record("living",dict(value=.3,phase="standard",known_entropy=.3))
        recorder.reward(-1.,.9,True)
        rows=recorder.advantages(.5)
        np.testing.assert_allclose([row["reward"] for row in rows],[0.,-.9,-1.])
        np.testing.assert_allclose([row["return"] for row in rows],[-.315,-.9,-1.])
        with TemporaryDirectory() as temp:
            metrics=record_training_metrics(temp,100,2,EntropyFixture(),rows,2)
            self.assertAlmostEqual(metrics["reward_mean"],-1.9/3)
            self.assertAlmostEqual(metrics["return_mean"],-2.215/3)
            self.assertAlmostEqual(metrics["entropy_mean"],.3,places=6)
            self.assertEqual(metrics["step"],103)
            stored=[json.loads(line) for line in (Path(temp)/"rollout_transitions.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in stored],[101,102,103])
            np.testing.assert_allclose([row["reward"] for row in stored],[0.,-.9,-1.])
            self.assertEqual(len((Path(temp)/"training_metrics.jsonl").read_text().splitlines()),1)

    def test_recording_does_not_change_policy_gradients_or_rng(self):
        torch.set_num_threads(1)
        policy=FreshPolicy.fresh("attacker",71)
        char,state=fixture()
        recorder=Recorder()
        for index in range(6):
            recorder.record("own",policy.select(char,state)[1])
            recorder.reward(-1. if index==5 else 0.,.99,index==5)
        rows=recorder.advantages(.95)
        parameters={key:val.clone() for key,val in policy.state_dict().items()}
        torch_state=torch.get_rng_state().clone()
        python_state=random.getstate()
        numpy_state=np.random.get_state()
        with torch.no_grad():
            expected_entropy=float(policy.evaluate_records(rows)[1].mean())
        with TemporaryDirectory() as temp:
            metrics=record_training_metrics(temp,0,1,policy,rows,2)
        self.assertAlmostEqual(metrics["entropy_mean"],expected_entropy,places=5)
        for key,val in policy.state_dict().items():
            self.assertTrue(torch.equal(parameters[key],val))
        self.assertTrue(all(parameter.grad is None for parameter in policy.parameters()))
        self.assertTrue(torch.equal(torch_state,torch.get_rng_state()))
        self.assertEqual(python_state,random.getstate())
        current=np.random.get_state()
        self.assertEqual(numpy_state[0],current[0])
        np.testing.assert_array_equal(numpy_state[1],current[1])
        self.assertEqual(numpy_state[2:],current[2:])

    def test_fingerprint_ignores_only_marked_logging_hooks(self):
        original=b"policy = 1\n"
        observed=original+b"# GC3_LOGGING_ONLY_BEGIN\nrecord_metrics()\n# GC3_LOGGING_ONLY_END\n"
        self.assertEqual(without_logging_blocks(observed),original)
        self.assertNotEqual(without_logging_blocks(observed.replace(b"policy = 1",b"policy = 2")),original)

    def test_resume_preserves_interrupted_tail_and_keeps_only_committed_metrics(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            contents = {
                "episodes.jsonl": b'{"seed":1}\n{"seed":2}\n{"seed":3}\n{"seed":',
                "rollout_transitions.jsonl": b'{"step":1,"batch_end_step":2}\n{"step":2,"batch_end_step":2}\n{"step":3,"batch_end_step":4}\n',
                "training_metrics.jsonl": b'{"step":2}\n{"step":4}\n',
            }
            for name, data in contents.items():
                (root/name).write_bytes(data)
            torch_state = torch.get_rng_state().clone()
            python_state = random.getstate()
            archive = reconcile_resume_logs(root, 2, 2)
            for name, original in contents.items():
                self.assertEqual((root/name).read_bytes()+(archive/name).read_bytes(), original)
            self.assertEqual(len((root/"episodes.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((root/"rollout_transitions.jsonl").read_text().splitlines()), 2)
            self.assertEqual((root/"training_metrics.jsonl").read_bytes(), b'{"step":2}\n')
            self.assertIsNone(reconcile_resume_logs(root, 2, 2))
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertEqual(python_state, random.getstate())

    def test_resume_does_not_trim_anything_when_a_committed_log_is_corrupt(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            episodes = b'{"seed":1}\n{"seed":2}\n'
            (root/"episodes.jsonl").write_bytes(episodes)
            (root/"training_metrics.jsonl").write_bytes(b'{"step":4}\n{"step":1}\n')
            with self.assertRaisesRegex(ValueError, "Non-monotonic"):
                reconcile_resume_logs(root, 2, 1)
            self.assertEqual((root/"episodes.jsonl").read_bytes(), episodes)
            self.assertFalse((root/"recovery").exists())

if __name__=="__main__":
    unittest.main()
