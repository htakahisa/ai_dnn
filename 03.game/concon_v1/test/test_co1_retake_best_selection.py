"""Compare readable previous weights; replace unreadable best with this run's best."""

import contextlib
import io
import pickle
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from concon_v1 import co1_train_defender_retake as training
from concon_v1.co1_retake_scenarios import get_scenario


def metrics(rate, retakes=10):
    return {"L": dict(retakes=retakes, mean_defuse_rate=rate,
                      min_defuse_rate=rate, moving_fire_rate=0.)}


class PreviousBestTests(unittest.TestCase):
    def run_training(self, directory, rates, compare=True, existing=True, load_error=None):
        directory = Path(directory)
        path = directory / get_scenario("L").model_path("best").name
        if existing:
            path.write_bytes(b"previous best weights")
        search = SimpleNamespace(model=torch.nn.Linear(1, 1), model_path=Path("search.pt"))
        model, previous = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)

        def schedule(*args, **kwargs):
            for episode in range(1, len(rates) + 1):
                kwargs["on_step"]([("L", (0,))], 1)
                yield dict(opponent="omoko_v1", site="L", planted=True, defused=True,
                           fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0,
                           end_reason="defused", round=episode,
                           training_episodes=dict(L=episode, R=0)), episode

        baseline = [metrics(.8)] if compare and existing and load_error is None else []
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(training, "COMPARE_PREVIOUS_BEST", compare))
            stack.enter_context(patch.object(training, "ConconDefenderSearchController", return_value=search))
            stack.enter_context(patch.object(training, "load_training_model", return_value=(model, {})))
            stack.enter_context(patch.object(training, "DefenderRetakeEnv"))
            stack.enter_context(patch.object(training, "iter_training_windows", side_effect=schedule))
            stack.enter_context(patch.object(training, "print_evaluation_summary"))
            stack.enter_context(patch.object(training, "make_checkpoint",
                side_effect=lambda model, site, episode, *args: dict(episode=episode, site=site)))
            load = stack.enter_context(patch.object(training, "load_previous_best", return_value=previous,
                                                   side_effect=load_error))
            evaluate = stack.enter_context(patch.object(training, "evaluate", side_effect=baseline + [metrics(rate) for rate in rates]))
            training.train(episodes=len(rates), checkpoint_interval=1, opponents=["omoko_v1"],
                           save_dir=directory, sites=("L",), epsilon_start=.05, epsilon_end=.05)
        self.assertEqual(load.call_count, int(compare and existing))
        if baseline:
            self.assertIs(evaluate.call_args_list[0].args[0]["L"], previous)
            self.assertEqual(evaluate.call_args_list[0].args[1:], evaluate.call_args_list[1].args[1:])
        return path

    def test_worse_or_equal_new_policy_keeps_previous_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.run_training(directory, [.7, .8])
            self.assertEqual(path.read_bytes(), b"previous best weights")
            latest = torch.load(Path(directory) / get_scenario("L").model_path("latest").name, weights_only=False)
            self.assertEqual(latest["episode"], 2)
            self.assertTrue(latest["compare_previous_best"])

    def test_improvement_replaces_best_and_later_regression_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.run_training(directory, [.7, .9, .85])
            self.assertEqual(torch.load(path, weights_only=False)["episode"], 2)

    def test_false_constant_uses_only_this_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.run_training(directory, [.7], compare=False)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint["episode"], 1)
            self.assertFalse(checkpoint["compare_previous_best"])

    def test_no_previous_file_selects_new_best(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.run_training(directory, [.7], existing=False)
            self.assertEqual(torch.load(path, weights_only=False)["episode"], 1)

    def test_unreadable_previous_best_is_replaced_and_best_within_run_is_kept(self):
        for error in (ValueError("map mismatch"), RuntimeError("weights mismatch"),
                      EOFError("truncated file"), pickle.UnpicklingError("invalid checkpoint")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = self.run_training(directory, [.7, .9, .8], load_error=error)
                self.assertEqual(torch.load(path, weights_only=False)["episode"], 2)
                latest = torch.load(Path(directory) / get_scenario("L").model_path("latest").name,
                                    weights_only=False)
                self.assertEqual(latest["episode"], 3)

    def test_zero_previous_evaluation_retakes_protects_existing_best(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(training, "load_previous_best", return_value=torch.nn.Linear(1, 1)), \
             patch.object(training, "evaluate_for_comparison", return_value=metrics(None, retakes=0)):
            path = Path(directory) / get_scenario("L").model_path("best").name
            path.write_bytes(b"old")
            scores, protected = training.previous_best_scores({"L": torch.nn.Linear(1, 1)},
                {"L": Path(directory)}, None, 10, 0, ["omoko_v1"], {"L": 6}, "cpu")
            self.assertIsNone(scores["L"])
            self.assertEqual(protected, {"L"})
            self.assertEqual(path.read_bytes(), b"old")

    def test_comparison_resets_seeds_and_restores_training_rng_even_on_error(self):
        def draws(*args):
            return random.random(), np.random.random(), torch.rand(1).item()
        for raises in (False, True):
            random.seed(11)
            np.random.seed(12)
            torch.manual_seed(13)
            python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            with patch.object(training, "evaluate", side_effect=RuntimeError("evaluation failed") if raises else draws):
                if raises:
                    with self.assertRaises(RuntimeError):
                        training.evaluate_for_comparison({}, None, 10, 1, [], 6)
                else:
                    first = training.evaluate_for_comparison({}, None, 10, 1, [], 6)
                    self.assertEqual(first, training.evaluate_for_comparison({}, None, 10, 1, [], 6))
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_no_argument_cli_uses_constant_and_temporary_override(self):
        with patch.object(training, "train") as train, patch.object(training, "COMPARE_PREVIOUS_BEST", False):
            training.main([])
            self.assertFalse(train.call_args.kwargs["compare_previous_best"])
            training.main(["--compare-previous-best"])
            self.assertTrue(train.call_args.kwargs["compare_previous_best"])
        with patch.object(training, "train") as train, patch.object(training, "COMPARE_PREVIOUS_BEST", True):
            training.main([])
            self.assertTrue(train.call_args.kwargs["compare_previous_best"])
            training.main(["--no-compare-previous-best"])
            self.assertFalse(train.call_args.kwargs["compare_previous_best"])

    def test_previous_loader_preserves_saved_weights(self):
        original = torch.nn.Linear(1, 1)
        checkpoint = dict(foundation_version=0,
                          model_state_dict=original.state_dict())
        state = torch.get_rng_state()
        with patch.object(training.torch, "load", return_value=checkpoint), \
             patch.object(training, "validate_checkpoint"), \
             patch.object(training, "RetakeDQN", side_effect=lambda *args, **kwargs: torch.nn.Linear(1, 1)):
            loaded = training.load_previous_best("L", Path("best.pt"), 6, "cpu")
            self.assertTrue(torch.equal(torch.get_rng_state(), state))
            torch.testing.assert_close(loaded.weight, original.weight)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
