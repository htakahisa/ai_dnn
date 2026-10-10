"""Checkpoint compatibility tests; never start training or modify saved models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import OrderedDict
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from touyama_v3.tv3_checkpoint import atomic_save, load_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_legacy_numpy_scalars_and_scoped_allowlist(self):
        before = torch.serialization.get_safe_globals().copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"replay": [[np.int64(7), np.float64(.5), np.bool_(True)]],
                        "evaluation": {"score": np.float32(.25)}}, path)
            loaded = load_checkpoint(path)
        self.assertEqual(loaded["replay"], [[7, .5, True]])
        self.assertIs(type(loaded["replay"][0][0]), int)
        self.assertIs(type(loaded["evaluation"]["score"]), float)
        self.assertEqual(torch.serialization.get_safe_globals(), before)

    def test_new_save_loads_without_allowlist(self):
        model = OrderedDict(weight=torch.tensor([1., 2.]))
        model._metadata = OrderedDict({"": {"version": 1}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_save(path, {"model": model, "replay": [(np.ones(2, np.float16),
                        np.int32(3), np.float64(.5))], "config": {"origin": (np.int64(2), 1)}})
            loaded = torch.load(path, weights_only=True)
        self.assertIs(type(loaded["replay"][0][1]), int)
        self.assertEqual(loaded["config"]["origin"], (2, 1))
        self.assertEqual(loaded["model"]._metadata, model._metadata)
        torch.testing.assert_close(loaded["replay"][0][0], torch.ones(2, dtype=torch.float16))

    def test_replace_retries_temporary_lock(self):
        original = Path.replace
        calls = []

        def locked_once(source, target):
            calls.append(source)
            if len(calls) == 1:
                raise PermissionError("temporary lock")
            return original(source, target)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            with patch.object(Path, "replace", locked_once), patch("touyama_v3.tv3_checkpoint.time.sleep"):
                atomic_save(path, {"completed_sets": 2})
            self.assertEqual(torch.load(path, weights_only=True)["completed_sets"], 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_persistent_lock_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            atomic_save(path, {"completed_sets": 1})
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), \
                    patch("touyama_v3.tv3_checkpoint.time.sleep"):
                with self.assertRaises(PermissionError):
                    atomic_save(path, {"completed_sets": 2})
            self.assertEqual(torch.load(path, weights_only=True)["completed_sets"], 1)
            recovery = list(Path(directory).glob("*.tmp"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual(torch.load(recovery[0], weights_only=True)["completed_sets"], 2)

    def test_lock_longer_than_old_retry_window_is_retried(self):
        original = Path.replace
        calls = []
        def locked_temporarily(source, target):
            calls.append(source)
            if len(calls) <= 12:
                raise PermissionError("reader still loading checkpoint")
            return original(source,target)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"latest.pt"
            with patch.object(Path,"replace",locked_temporarily), \
                    patch("touyama_v3.tv3_checkpoint.time.sleep"):
                atomic_save(path,{"completed_sets":3})
            self.assertEqual(torch.load(path,weights_only=True)["completed_sets"],3)
            self.assertEqual(len(calls),13)
            self.assertEqual(list(Path(directory).glob("*.tmp")),[])

    def test_failed_serialization_does_not_leave_a_partial_recovery_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"latest.pt"
            atomic_save(path,{"completed_sets":1})
            with patch("touyama_v3.tv3_checkpoint.torch.save",side_effect=OSError("disk write failed")):
                with self.assertRaises(OSError):
                    atomic_save(path,{"completed_sets":2})
            self.assertEqual(torch.load(path,weights_only=True)["completed_sets"],1)
            self.assertEqual(list(Path(directory).glob("*.tmp")),[])


if __name__ == "__main__":
    unittest.main()
