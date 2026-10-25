"""The shared engine must not require dependencies of an unselected legacy AI."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class OptionalAiImportTests(unittest.TestCase):
    def test_gc3_bc_evaluation_works_without_optional_legacy_dependencies(self):
        # A fresh process prevents installed/cached packages from hiding the
        # import error seen when BC evaluates its first completed epoch.
        script = textwrap.dedent('''
            import importlib.abc
            import sys
            blocked = ("gymnasium", "matplotlib", "tensorboard", "torch.utils.tensorboard")
            class MissingOptionalPackages(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                        raise ModuleNotFoundError("No module named " + repr(fullname), name=fullname)
            sys.meta_path.insert(0, MissingOptionalPackages())
            import torch
            torch.set_num_threads(1)
            from ghost_champions_v3.gc3_rollout import module_digest
            from ghost_champions_v3.preplant.gc3_checkpoint_attacker_micro import fresh_micro_policy
            from ghost_champions_v3.preplant.gc3_train_attacker_pipeline import evaluate_preplant,check_preplant_runtime
            check_preplant_runtime()
            policy = fresh_micro_policy()
            before = module_digest([policy])
            report = evaluate_preplant(policy, dict(SEED=20261025,
                OPPONENT_CODES=("FNC", "FRC", "TYG", "OMG", "SPS", "GRG", "GC"), EVALUATION_ROUNDS_PER_OPPONENT=1,
                FIXED_OPPONENT_AI="default"))
            assert len(report["rounds"]) == 7
            assert all(r["opponent_weights_unchanged"] for r in report["rounds"])
            assert before == module_digest([policy])
            assert all(name not in sys.modules for name in blocked)
            assert "learning_defender" not in sys.modules
            assert "learning_attacker" not in sys.modules
            assert "train_defender_combined" not in sys.modules
            assert "policy_attacker_controller" not in sys.modules
            assert "train_bc" not in sys.modules
            from ghost_champions_v3.tools.gc3_watch_attacker_preplant import prepare_round,DEFAULT_MODEL
            from unittest.mock import patch
            with patch('ghost_champions_v3.preplant.gc3_learn_attacker_preplant.SAVE_MATCH_HISTORY',False):
                game,info=prepare_round(DEFAULT_MODEL,'FNC',20261027)
                game._simulate_tick()
            assert info['preplant_stage']=='fresh'
            print("GC3_BC_EVALUATION_WITHOUT_OPTIONAL_PACKAGES_OK")
        ''')
        result = subprocess.run([sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
        output = result.stdout.decode("utf-8", errors="replace")
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("GC3_BC_EVALUATION_WITHOUT_OPTIONAL_PACKAGES_OK", output)

    def test_bc_checks_runtime_before_loading_data_or_updating_weights(self):
        from unittest.mock import patch
        from ghost_champions_v3.preplant import gc3_train_attacker_pipeline as pipeline
        with patch.object(pipeline,"check_preplant_runtime",side_effect=ModuleNotFoundError("missing evaluation dependency")),patch.object(
                pipeline,"dataset_manifest",side_effect=AssertionError("Training data loaded before import check")):
            with self.assertRaisesRegex(ModuleNotFoundError,"missing evaluation dependency"):
                pipeline.train_bc({})


if __name__ == "__main__":
    unittest.main()
