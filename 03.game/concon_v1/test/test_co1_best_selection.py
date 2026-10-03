"""Check greedy evaluation, warmup gating, and balanced best-model selection."""

import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from concon_v1 import co1_train_attacker as training

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1 import evaluate_co1_attacker as evaluation_module
    from concon_v1 import co1_battle_training as battle_module


def metrics(mean, minimum):
    return {"success_rate": mean, "min_team_plant_rate": minimum}


def finished_training_env():
    """Finish rounds without policy steps so scheduling can be tested cheaply."""
    return SimpleNamespace(
        opponents=("gc_v1", "omoko_v1"), opponent="gc_v1", done=True,
        success=True, had_spike_drop=False, spike_recovered=False,
        elapsed_ticks=0, reset=lambda: ([], []),
        game=SimpleNamespace(round_over=True, round_timer=1, attacker_wins=1),
    )


class BestSelectionTests(unittest.TestCase):
    def test_elimination_wins_are_logged_directly_below_plants(self):
        env = finished_training_env()
        env.success = False
        output = io.StringIO()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(battle_module, "BattleRouteEnv", return_value=env),
              patch.object(training, "evaluate_checkpoint", return_value=metrics(0.0, 0.0)),
              patch.object(training.torch, "save") as save,
              contextlib.redirect_stdout(output)):
            training.train(episodes=20, save_dir=directory)
        lines = output.getvalue().splitlines()
        plant_index = next(i for i, line in enumerate(lines) if "  team100 " in line)
        self.assertIn("gc_v1=0/20(0.000)", lines[plant_index])
        self.assertIn("team_win100 gc_v1=20/20(1.000)", lines[plant_index + 1])
        self.assertIn("plant_or_elimination", lines[plant_index + 1])
        self.assertEqual(save.call_args_list[0].args[0]["team_win100"]["gc_v1"]["plants"], 20)

    def test_epsilon_reaches_the_exact_configured_floor(self):
        self.assertGreater(training.epsilon_by_episode(1399, 2000), training.EPSILON_END)
        self.assertEqual(training.epsilon_by_episode(1400, 2000), training.EPSILON_END)
        self.assertEqual(training.epsilon_by_episode(2000, 2000), training.EPSILON_END)

    def test_both_mean_and_worst_team_must_not_regress(self):
        best = metrics(0.7, 0.6)
        cases = [
            (metrics(0.725, 0.5), False),  # Better mean hides a weaker team.
            (metrics(0.65, 0.65), False),  # Better floor sacrifices the mean.
            (metrics(0.7, 0.6), True),
            (metrics(0.75, 0.6), True),
            (metrics(0.7, 0.65), True),
        ]
        for candidate, expected in cases:
            with self.subTest(candidate=candidate):
                self.assertEqual(training.qualifies_as_best(candidate, best), expected)
        self.assertTrue(training.qualifies_as_best(metrics(0.0, 0.0), None))

    def assert_rng_states_equal(self, states):
        self.assertEqual(random.getstate(), states[0])
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], states[1][0])
        np.testing.assert_array_equal(current_numpy[1], states[1][1])
        self.assertEqual(current_numpy[2:], states[1][2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), states[2]))

    def test_evaluation_uses_equal_rounds_seed_and_frozen_weights_for_every_team(self):
        checkpoint = {
            "training_mode": "battle", "opponents": ("gc_v1", "omoko_v1", "gc_v1"),
            "model_state_dict": {"weight": torch.tensor([1.0])},
        }
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        blobs = []

        def fake_evaluate(opponent, rounds, seed, *, frozen_checkpoint, map_name):
            self.assertEqual(map_name, "A1")
            self.assertEqual((rounds, seed), (20, 31))
            blobs.append(frozen_checkpoint)
            frozen = torch.load(io.BytesIO(frozen_checkpoint), weights_only=False)
            self.assertEqual(frozen["model_state_dict"]["weight"].item(), 1.0)
            # Loading/using an evaluation copy cannot change the training weights.
            frozen["model_state_dict"]["weight"].fill_(99)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.random(), np.random.random(), torch.rand(1)
            plants = {"gc_v1": 16, "omoko_v1": 10}[opponent]
            return {"plants": plants, "rounds": rounds, "plant_success_rate": plants / rounds}

        with (patch.object(evaluation_module, "evaluate", side_effect=fake_evaluate) as evaluate,
              contextlib.redirect_stdout(io.StringIO())):
            result = training.evaluate_checkpoint(checkpoint, rounds=20, seed=31)
        self.assertEqual(evaluate.call_count, 2)
        self.assertIs(blobs[0], blobs[1])
        self.assertEqual(result["epsilon"], 0.0)
        self.assertAlmostEqual(result["success_rate"], 0.65)
        self.assertEqual(result["min_team_plant_rate"], 0.5)
        self.assertEqual(result["team_plants"]["gc_v1"]["episodes"], 20)
        self.assertEqual(checkpoint["model_state_dict"]["weight"].item(), 1.0)
        self.assert_rng_states_equal(states)

    def test_failed_evaluation_also_restores_training_rng_states(self):
        states = random.getstate(), np.random.get_state(), torch.get_rng_state()

        def fail(*args, **kwargs):
            random.random(), np.random.random(), torch.rand(1)
            raise RuntimeError("evaluation failed")

        with patch.object(evaluation_module, "evaluate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                training.evaluate_checkpoint({"training_mode": "battle", "opponents": ("gc_v1",)})
        self.assert_rng_states_equal(states)

    def test_training_skips_warmup_and_saves_only_balanced_candidates(self):
        saved = []
        real_save = torch.save

        def save(checkpoint, path):
            saved.append((Path(path).name, checkpoint["episode"]))
            real_save(checkpoint, path)

        candidates = [metrics(0.6, 0.4), metrics(0.7, 0.3),
                      metrics(0.5, 0.5), metrics(0.7, 0.4)]
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(battle_module, "BattleRouteEnv", return_value=finished_training_env()),
              patch.object(training, "CHECKPOINT_INTERVAL", 1),
              patch.object(training, "evaluate_checkpoint", side_effect=candidates) as evaluate,
              patch.object(training.torch, "save", side_effect=save),
              contextlib.redirect_stdout(io.StringIO())):
            training.train(episodes=10, save_dir=directory, eval_rounds=20)
            self.assertEqual([call.args[0]["episode"] for call in evaluate.call_args_list],
                             [7, 8, 9, 10])
            self.assertTrue(all(call.args[0]["epsilon"] == training.EPSILON_END
                                for call in evaluate.call_args_list))
            self.assertEqual([episode for name, episode in saved if name.endswith("_best.pt")],
                             [7, 10])
            self.assertEqual([episode for name, episode in saved if name.endswith("_latest.pt")],
                             list(range(1, 11)))
            best = torch.load(Path(directory) / "co1_attacker_A1_best.pt", weights_only=False)
            self.assertEqual(best["evaluation"], candidates[-1])
            self.assertEqual(best["success_rate"], 1.0)  # Training score is separate.

    def test_force_save_keeps_numbered_models_at_every_checkpoint_interval(self):
        for enabled in (False, True):
            with (self.subTest(force_save=enabled),
                  tempfile.TemporaryDirectory() as directory,
                  patch.object(battle_module, "BattleRouteEnv", return_value=finished_training_env()),
                  patch.object(training, "CHECKPOINT_INTERVAL", 3),
                  patch.object(training, "evaluate_checkpoint", return_value=metrics(0.7, 0.6)) as evaluate,
                  contextlib.redirect_stdout(io.StringIO())):
                training.train(episodes=10, save_dir=directory, force_save=enabled)
                debug_paths = sorted(Path(directory).glob("co1_attacker_A1_episode_*.pt"))
                self.assertEqual(len(debug_paths), 4 if enabled else 0)
                for episode in (3, 6, 9, 10) if enabled else ():
                    debug = torch.load(
                        Path(directory) / f"co1_attacker_A1_episode_{episode}.pt",
                        weights_only=False)
                    self.assertEqual(debug["episode"], episode)
                    if episode < 7:
                        self.assertGreater(debug["epsilon"], training.EPSILON_END)
                    self.assertIsNone(debug["evaluation"])
                    self.assertIn("model_state_dict", debug)
                # Debug saving does not add greedy evaluations or replace latest/best.
                self.assertEqual(evaluate.call_count, 2)
                for kind in ("latest", "best"):
                    checkpoint = torch.load(
                        Path(directory) / f"co1_attacker_A1_{kind}.pt", weights_only=False)
                    self.assertEqual(checkpoint["episode"], 10)

    def test_existing_best_is_re_evaluated_once_and_kept_if_candidates_are_weaker(self):
        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / "co1_attacker_A1_best.pt"
            # Legacy best has no exploration-free metrics and a different opponent list.
            torch.save({"episode": 99, "success_rate": 0.0, "opponents": ("fnatic_v3",)},
                       best_path)
            original_bytes = best_path.read_bytes()

            def evaluate(checkpoint, rounds, seed):
                self.assertEqual(checkpoint["opponents"], ("gc_v1", "omoko_v1"))
                self.assertEqual((rounds, seed), (3, 4))
                return metrics(0.9, 0.8) if checkpoint["episode"] == 99 else metrics(0.95, 0.7)

            with (patch.object(battle_module, "BattleRouteEnv", return_value=finished_training_env()),
                  patch.object(training, "CHECKPOINT_INTERVAL", 1),
                  patch.object(training, "evaluate_checkpoint", side_effect=evaluate) as evaluator,
                  contextlib.redirect_stdout(io.StringIO())):
                training.train(episodes=10, save_dir=directory, seed=4, eval_rounds=3)
            self.assertEqual([call.args[0]["episode"] for call in evaluator.call_args_list],
                             [99, 7, 8, 9, 10])
            self.assertEqual(best_path.read_bytes(), original_bytes)
            latest = torch.load(Path(directory) / "co1_attacker_A1_latest.pt", weights_only=False)
            self.assertEqual(latest["episode"], 10)
            self.assertEqual(latest["evaluation"], metrics(0.95, 0.7))

    def test_incompatible_best_is_backed_up_and_training_saves_current_map(self):
        scenario = training.get_scenario("A2")
        mismatches = (
            {"waypoint_order": "abcd", "obs_dim": 28},
            {"obs_dim": 28},
            {"scenario_signature": "old-map"},
            {"map_name": "A1"},
        )
        for mismatch in mismatches:
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                best_path = Path(directory) / scenario.checkpoint_filename("best")
                previous = {
                    "episode": 99, "map_name": "A2",
                    "waypoint_order": scenario.waypoint_order,
                    "obs_dim": scenario.obs_dim, "scenario_signature": scenario.signature,
                    **mismatch,
                }
                torch.save(previous, best_path)
                original_bytes = best_path.read_bytes()
                output = io.StringIO()
                with (patch.object(battle_module, "BattleRouteEnv", return_value=finished_training_env()),
                      patch.object(training, "CHECKPOINT_INTERVAL", 1),
                      patch.object(training, "evaluate_checkpoint", return_value=metrics(0.0, 0.0)) as evaluate,
                      contextlib.redirect_stdout(output)):
                    training.train(episodes=10, save_dir=directory, map_name="A2", force_save=False)
                self.assertEqual([call.args[0]["episode"] for call in evaluate.call_args_list],
                                 [7, 8, 9, 10])
                backups = list(Path(directory).glob("co1_attacker_A2_best_incompatible_*.pt"))
                self.assertEqual(len(backups), 1)
                self.assertEqual(backups[0].read_bytes(), original_bytes)
                self.assertIn("Skipping incompatible existing best", output.getvalue())
                for kind in ("best", "latest"):
                    saved = torch.load(Path(directory) / scenario.checkpoint_filename(kind),
                                       weights_only=False)
                    training.validate_checkpoint_scenario(saved, scenario)
                    self.assertEqual(saved["episode"], 10)

    def test_incompatible_checkpoints_still_fail_direct_evaluation(self):
        with self.assertRaisesRegex(ValueError, "waypoint order"):
            training.evaluate_checkpoint({"map_name": "A2", "waypoint_order": "abcd"})


if __name__ == "__main__":
    unittest.main()
