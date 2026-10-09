import hashlib
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ghost_champions_v2.rl import training
from ghost_champions_v2.tools import fork_evaluation_run as fork
from tools import run_eval


class CompetitionSnapshotTests(unittest.TestCase):
    def test_restore_preserves_editable_file_and_requires_exact_original_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / "runtime"
            snapshot.mkdir()
            original = root / "run_competition_manager.py"
            original.write_bytes(b"# edited UI\n")
            source = "# frozen manager\r\n"
            digest = hashlib.sha256(source.encode()).hexdigest()
            parent = dict(frozen_inputs={str(original): digest})
            with patch.object(fork, "ROOT", root), \
                 patch("ghost_champions_v2.tools.registry_compatibility.git_source", return_value=source):
                record = fork.freeze_competition_manager(parent, snapshot)
                self.assertEqual(record["sha256"], digest)
                self.assertEqual(original.read_bytes(), b"# edited UI\n")
                self.assertEqual((snapshot / original.name).read_bytes(), source.encode())
                self.assertEqual(fork.freeze_competition_manager(parent, snapshot), record)
                (snapshot / original.name).write_bytes(b"# changed snapshot\n")
                with self.assertRaisesRegex(ValueError, "snapshot differs"):
                    fork.freeze_competition_manager(parent, snapshot)

    def test_same_bytes_at_snapshot_path_are_compatible_but_changed_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            original = root / "run_competition_manager.py"
            copy = runtime / original.name
            parent = dict(schema_hash=training.SCHEMA_HASH, verification=False,
                          config={}, frozen_inputs={str(original): "frozen"})
            env = {"GC_RUNTIME_SNAPSHOT": str(runtime), "GC_OPPONENT_SNAPSHOT": ""}
            with patch.object(training, "ROOT", root), patch.dict(os.environ, env):
                training.validate_evaluation_change(parent, {}, {str(copy): "frozen"})
                with self.assertRaisesRegex(ValueError, "Policy/runtime inputs changed"):
                    training.validate_evaluation_change(parent, {}, {str(copy): "changed"})
                with self.assertRaisesRegex(ValueError, "Conflicting frozen input copies"):
                    training.validate_evaluation_change(parent, {}, {str(copy): "frozen", str(original): "changed"})
                parent["frozen_inputs"] = {str(copy): "frozen"}
                training.validate_evaluation_change(parent, {}, {str(copy): "frozen"})

    def test_freeze_monitors_the_copy_while_workspace_ui_can_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            copy = root / "runtime" / "run_competition_manager.py"
            copy.parent.mkdir()
            copy.write_bytes(b"frozen")
            frozen = {str(copy): hashlib.sha256(copy.read_bytes()).hexdigest()}
            (root / copy.name).write_bytes(b"new UI")
            training.assert_frozen(frozen)
            copy.write_bytes(b"modified")
            with self.assertRaises(RuntimeError):
                training.assert_frozen(frozen)

    def test_evaluation_rejects_a_source_snapshot_edited_by_a_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = root / "runtime"
            runtime.mkdir()
            for name in run_eval.RUNTIME_FILES:
                (runtime / name).write_bytes(b"definitions")
            manager = runtime / "run_competition_manager.py"
            manager.write_bytes(b"frozen")
            for name in ("run_game.py", "battle_logic.py", "ghost_champions_v1.py", "ghost_champions_v1_macro.py"):
                (root / name).write_bytes(b"source")
            def edited_result():
                manager.write_bytes(b"changed during evaluation")
                return "series.json", 0
            future = SimpleNamespace(result=edited_result)
            args = ["run_eval.py", "--controller", "v1", "--opponents", "FRC", "--series-count", "1",
                    "--runtime-snapshot", str(runtime), "--output", str(root / "evaluation")]
            with patch.object(run_eval, "ROOT", root), patch.object(sys, "argv", args), \
                 patch.dict(os.environ, {"GC_OPPONENT_SNAPSHOT": "", "GC_V2_CONFIG": ""}), \
                 patch.object(run_eval, "ProcessPoolExecutor") as pool, \
                 patch.object(run_eval, "as_completed", return_value=[future]), \
                 contextlib.redirect_stderr(io.StringIO()) as errors, contextlib.redirect_stdout(io.StringIO()):
                pool.return_value.__enter__.return_value.submit.return_value = future
                with self.assertRaises(SystemExit):
                    run_eval.main()
                self.assertIn("runtime source snapshot changed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
