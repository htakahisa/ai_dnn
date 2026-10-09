"""Worker bounds/isolation/failures using lightweight fixtures, never live games."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from toruAI_v4.tv4_training_parallel import run_opponent_workers


class ParallelTests(unittest.TestCase):
    def test_real_children_overlap_with_bounded_workers_and_isolated_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            script = base / "fixture.py"
            script.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--opponents',nargs='+')
p.add_argument('--max-workers',type=int)
p.add_argument('--log-dir',type=Path)
p.add_argument('--sets',type=int)
a=p.parse_args()
start=time.time()
time.sleep(.2)
a.log_dir.mkdir(parents=True,exist_ok=True)
(a.log_dir/'result.json').write_text(json.dumps(dict(start=start,end=time.time(),pid=os.getpid(),cwd=os.getcwd(),sets=a.sets,workers=a.max_workers,opponents=a.opponents)))
''', encoding="utf-8")
            names = ["a", "b", "c", "d", "e", "f"]
            with patch("builtins.print"):
                self.assertEqual(run_opponent_workers(script, ["--sets", "7", "--opponents", *names],
                                                     names, 3, log_dir=base / "logs"), 0)
            rows = [json.loads((base / "logs" / name / "result.json").read_text()) for name in names]
            self.assertEqual(len(set(r["pid"] for r in rows)), 6)
            events = sorted([(r["start"], 1) for r in rows] + [(r["end"], -1) for r in rows])
            active = peak = 0
            for _, delta in events:
                active += delta
                peak = max(peak, active)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 3)
            for name, row in zip(names, rows):
                self.assertEqual(row["opponents"], [name])
                self.assertEqual(row["workers"], 1)
                self.assertEqual(row["sets"], 7)
                self.assertEqual(Path(row["cwd"]), base)

    def test_single_worker_or_opponent_uses_existing_execution(self):
        with patch("toruAI_v4.tv4_training_parallel.subprocess.Popen") as spawn:
            self.assertIsNone(run_opponent_workers(__file__, [], ["a", "b"], 1))
            self.assertIsNone(run_opponent_workers(__file__, [], ["a"], 3))
            spawn.assert_not_called()
        with self.assertRaises(ValueError):
            run_opponent_workers(__file__, [], ["a", "b"], 0)

    def test_failure_stops_running_children_and_does_not_start_pending_jobs(self):
        failed, running = Mock(), Mock()
        failed.poll.return_value = 7
        running.poll.return_value = None
        with patch("toruAI_v4.tv4_training_parallel.subprocess.Popen", side_effect=[failed, running]) as spawn, \
             patch("builtins.print"):
            self.assertEqual(run_opponent_workers(__file__, [], ["a", "b", "c"], 2), 7)
        self.assertEqual(spawn.call_count, 2)
        running.terminate.assert_called_once()
        running.wait.assert_called_once()

    def test_interrupt_cleans_up_children(self):
        running = Mock()
        running.poll.side_effect = [KeyboardInterrupt(), None]
        with patch("toruAI_v4.tv4_training_parallel.subprocess.Popen", return_value=running), patch("builtins.print"):
            self.assertEqual(run_opponent_workers(__file__, [], ["a", "b"], 2), 130)
        running.terminate.assert_called_once()

    def test_all_six_training_entries_use_source_worker_setting_and_cli_override(self):
        from toruAI_v4 import tv4_train_attacker_analysis, tv4_train_attacker_plant, tv4_train_attacker_guard
        from toruAI_v4 import tv4_train_defender_analysis, tv4_train_defender_search, tv4_train_retake
        for module in (tv4_train_attacker_analysis, tv4_train_attacker_plant, tv4_train_attacker_guard, tv4_train_defender_analysis):
            self.assertEqual(module.parser().parse_args([]).max_workers, module.MAX_PARALLEL_WORKERS)
            self.assertEqual(module.parser().parse_args(["--max-workers", "1"]).max_workers, 1)
        for phase in ("search", "retake"):
            module = tv4_train_defender_search if phase == "search" else tv4_train_retake
            self.assertEqual(tv4_train_defender_search.parse_arguments(["--phase", phase]).max_workers, module.MAX_PARALLEL_WORKERS)
            self.assertEqual(tv4_train_defender_search.parse_arguments(["--phase", phase, "--max-workers", "1"]).max_workers, 1)


if __name__ == "__main__":
    unittest.main()
