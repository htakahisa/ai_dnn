"""Tests for automatic GC attacker model promotion source resolution."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from gc_v1 import promote_attacker_gc as promote


class AttackerPromotionTests(unittest.TestCase):
    def make_bundle(self, root, selected_branch=None, same_as_targets=False):
        source_dir = root / "training"
        source_dir.mkdir()
        targets = {}
        names = {}
        for phase in promote.PHASES:
            target = root / f"active_{phase}.pt"
            target.write_bytes(f"old-{phase}".encode())
            targets[phase] = target
            if phase in promote.AUTO_PHASES:
                name = f"dqn_attacker_{phase}_gc_best_by_eval.pt"
                (source_dir / name).write_bytes(
                    (f"old-{phase}" if same_as_targets else f"new-{phase}").encode()
                )
                names[phase] = name
        bundle_path = source_dir / "best_by_eval_bundle.json"
        bundle_path.write_text(
            json.dumps({
                "episode": 2750,
                "selected_branch": selected_branch,
                "models": names,
                "evaluation": {"worst_timeout_rate": 0.19, "round_win_rate": 0.287},
            }),
            encoding="utf-8",
        )
        return bundle_path, targets

    def test_no_arguments_dry_run_uses_training_best_by_eval(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle_path, targets = self.make_bundle(root)
            output = io.StringIO()
            with patch.object(promote, "CANDIDATE_BUNDLE", bundle_path), \
                 patch.dict(promote.TARGETS, targets, clear=True), \
                 patch.object(sys, "argv", ["promote_attacker_gc", "--dry-run"]), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(promote.main(), 0)
            self.assertIn("episode=2750", output.getvalue())
            self.assertIn("bypasses safety selection", output.getvalue())
            self.assertIn("[DRY-RUN]", output.getvalue())
            for phase, target in targets.items():
                self.assertEqual(target.read_bytes(), f"old-{phase}".encode())

    def test_no_arguments_rejects_self_copy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            bundle_path, targets = self.make_bundle(
                Path(directory), same_as_targets=True,
            )
            with patch.object(promote, "CANDIDATE_BUNDLE", bundle_path), \
                 patch.dict(promote.TARGETS, targets, clear=True), \
                 patch.object(sys, "argv", ["promote_attacker_gc", "--dry-run"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "already active"):
                    promote.main()

    def test_no_arguments_installs_best_and_backs_up_old_models(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            bundle_path, targets = self.make_bundle(root)
            backup_root = root / "backups"
            with patch.object(promote, "CANDIDATE_BUNDLE", bundle_path), \
                 patch.dict(promote.TARGETS, targets, clear=True), \
                 patch.object(sys, "argv", [
                     "promote_attacker_gc", "--backup-root", str(backup_root),
                 ]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(promote.main(), 0)
            backup_dirs = list(backup_root.iterdir())
            self.assertEqual(len(backup_dirs), 1)
            manifest = json.loads(
                (backup_dirs[0] / "promotion_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(manifest["models"]), set(promote.AUTO_PHASES))
            for phase in promote.AUTO_PHASES:
                self.assertEqual(targets[phase].read_bytes(), f"new-{phase}".encode())
                self.assertEqual(
                    (backup_dirs[0] / targets[phase].name).read_bytes(),
                    f"old-{phase}".encode(),
                )
            self.assertEqual(targets["retrieve"].read_bytes(), b"old-retrieve")

    def test_selected_mode_rejects_baseline_bundle(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            bundle_path, targets = self.make_bundle(
                Path(directory), selected_branch="baseline",
            )
            with patch.object(promote, "SELECTED_BUNDLE", bundle_path), \
                 patch.dict(promote.TARGETS, targets, clear=True), \
                 patch.object(sys, "argv", [
                     "promote_attacker_gc", "--selected", "--dry-run",
                 ]):
                with self.assertRaisesRegex(ValueError, "no trained attacker bundle"):
                    promote.main()


if __name__ == "__main__":
    unittest.main()
