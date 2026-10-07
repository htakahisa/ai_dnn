"""Opponent selection, shared fallback, and isolated training outputs."""

from pathlib import Path
import contextlib
import io
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1 import co1_retake_models as locations
from concon_v1.co1_train_defender_retake import main, train
from concon_v1.co1_defender_controller import ConconDefenderController
from concon_v1.co1_retake_scenarios import get_scenario


class OpponentModelTests(unittest.TestCase):
    def test_cli_defaults_and_single_opponent_alias(self):
        with patch("concon_v1.co1_train_defender_retake.train") as training:
            main([])
            self.assertIsNone(training.call_args.kwargs["model_opponent"])
            for option in ("--opponent", "--opponents"):
                main([option, "gc_v1"])
                self.assertEqual(training.call_args.kwargs["model_opponent"], "gc_v1")
                self.assertEqual(training.call_args.kwargs["opponents"], ["gc_v1"])
            main(["--opponents", "gc_v1", "frc_v1"])
            self.assertIsNone(training.call_args.kwargs["model_opponent"])

    def test_runtime_falls_back_per_site_and_changes_with_opponent(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(locations, "AI_DIRECTORY", Path(temporary)):
            dedicated = locations.opponent_model_path("L", "gc_v1")
            dedicated.parent.mkdir(parents=True)
            dedicated.touch()
            self.assertEqual(locations.runtime_model_path("L", "gc_v1"), dedicated)
            self.assertEqual(locations.runtime_model_path("R", "gc_v1"), get_scenario("R").model_path())
            shared = get_scenario("L").model_path().resolve()
            with patch("concon_v1.co1_defender_controller.ConconDefenderRetakeController") as loader:
                loader.side_effect = [Mock(), Mock(), Mock()]
                controller = ConconDefenderController(search_controller=Mock())
                game = SimpleNamespace(current_attacker_team_ai=SimpleNamespace(name="Ghost Champions v1"))
                controller.set_game(game)
                state = dict(grid=get_scenario("L").grid, is_planted=True, planted_pos=(7, 3))
                controller.decide_move(None, state)
                controller.decide_move(None, state)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(loader.call_args.kwargs["model_path"], dedicated.resolve())
                game.current_attacker_team_ai.name = "FRC v1"
                controller.decide_move(None, state)
                self.assertEqual(loader.call_args.kwargs["model_path"], shared)
                game.current_attacker_team_ai.name = "Ghost Champions v1"
                controller.decide_move(None, state)
                self.assertEqual(loader.call_count, 3)
                self.assertEqual(loader.call_args.kwargs["model_path"], dedicated.resolve())

    def test_dedicated_training_cannot_overwrite_shared_directory(self):
        with self.assertRaisesRegex(ValueError, "dedicated save directory"):
            train(opponents=["gc_v1"], model_opponent="gc_v1", save_dir=get_scenario("L").save_dir)

    def test_dedicated_training_reads_shared_weights_and_isolates_outputs(self):
        search = SimpleNamespace(model=torch.nn.Linear(1, 1), model_path=Path("search.pt"))
        metric = dict(retakes=1, mean_defuse_rate=.5, min_defuse_rate=.5, moving_fire_rate=0.)

        def schedule(*args, **kwargs):
            kwargs["on_step"]([("L", (0,)), ("R", (0,))], 1)
            yield dict(opponent="gc_v1", site="L", planted=True, defused=False,
                       fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0,
                       round=2, training_episodes=dict(L=1, R=1)), 2

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(locations, "AI_DIRECTORY", Path(temporary)), \
             contextlib.redirect_stdout(io.StringIO()), \
             patch("concon_v1.co1_train_defender_retake.ConconDefenderSearchController", return_value=search), \
             patch("concon_v1.co1_train_defender_retake.load_training_model",
                   side_effect=lambda *a, **k: (torch.nn.Linear(1, 1), {})) as load, \
             patch("concon_v1.co1_train_defender_retake.DefenderRetakeEnv"), \
             patch("concon_v1.co1_train_defender_retake.iter_training_windows", side_effect=schedule), \
             patch("concon_v1.co1_train_defender_retake.make_checkpoint", return_value={}), \
             patch("concon_v1.co1_train_defender_retake.evaluate", return_value={s: dict(metric) for s in ("L", "R")}), \
             patch("concon_v1.co1_train_defender_retake.print_evaluation_summary"), \
             patch("concon_v1.co1_train_defender_retake.torch.save") as save:
            train(episodes=2, checkpoint_interval=2, opponents=["gc_v1"], model_opponent="gc_v1",
                  epsilon_start=.05, epsilon_end=.05)
            for site, call in zip(("L", "R"), load.call_args_list):
                self.assertEqual(call.args[1], get_scenario(site).save_dir)
                self.assertIsNone(call.args[3])
                self.assertEqual(call.kwargs["initial_path"], get_scenario(site).model_path("best"))
            self.assertEqual(save.call_count, 4)
            for call in save.call_args_list:
                self.assertEqual(call.args[0]["model_opponent"], "gc_v1")
                self.assertEqual(call.args[1].parent, locations.opponent_directory("gc_v1"))
            self.assertTrue((locations.opponent_directory("gc_v1", "logs") / "retake_training_log.jsonl").is_file())
            self.assertFalse((locations.opponent_directory("gc_v1") / "retake_training_log.jsonl").exists())

    def test_resume_rejects_another_opponents_weights(self):
        search = SimpleNamespace(model=torch.nn.Linear(1, 1), model_path=Path("search.pt"))
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(locations, "AI_DIRECTORY", Path(temporary)), \
             patch("concon_v1.co1_train_defender_retake.ConconDefenderSearchController", return_value=search), \
             patch("concon_v1.co1_train_defender_retake.load_training_model",
                   return_value=(torch.nn.Linear(1, 1), {"model_opponent": "frc_v1"})):
            with self.assertRaisesRegex(ValueError, "different model target"):
                train(episodes=2, checkpoint_interval=2, opponents=["gc_v1"], model_opponent="gc_v1", resume=True)

    def test_all_runtime_ai_names_map_to_distinct_targets(self):
        for name, key in locations.OPPONENT_NAMES.items():
            game = SimpleNamespace(current_attacker_team_ai=SimpleNamespace(name=name))
            self.assertEqual(locations.game_opponent(game), key)
        self.assertIsNone(locations.game_opponent(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
