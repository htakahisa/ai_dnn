import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ghost_champions_v3 import gc3_train_attacker_guard, gc3_train_defender_search


class TrainingOutputTests(unittest.TestCase):
    def test_transient_windows_lock_retries_without_losing_status(self):
        for training in (gc3_train_attacker_guard, gc3_train_defender_search):
            with self.subTest(training=training.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"status.json"
                replace = Path.replace
                calls = []
                def temporarily_locked(source, destination):
                    calls.append(destination)
                    if len(calls) < 3:
                        raise PermissionError("reader still holds status.json")
                    return replace(source, destination)
                with patch.object(Path, "replace", temporarily_locked), patch.object(training.time, "sleep") as sleep:
                    training.write_json(path, {"step": 15940, "complete": False})
                self.assertEqual(sleep.call_count, 2)
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["step"], 15940)
                self.assertFalse(path.with_suffix(".tmp").exists())

    def test_permanent_write_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "replace", side_effect=PermissionError), patch.object(gc3_train_attacker_guard.time, "sleep"):
            with self.assertRaises(PermissionError):
                gc3_train_attacker_guard.write_json(Path(directory)/"status.json", {"complete": False})


if __name__ == "__main__":
    unittest.main()
