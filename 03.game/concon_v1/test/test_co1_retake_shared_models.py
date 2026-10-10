"""Every opponent uses the same site-specific shared retake policy."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_attacker_selection import OPPONENT_NAMES
from concon_v1.co1_defender_controller import ConconDefenderController
from concon_v1.co1_retake_scenarios import get_scenario
from concon_v1 import co1_train_defender_retake as training


class SharedRetakeTests(unittest.TestCase):
    def test_all_six_opponents_use_shared_best_without_reloading_on_opponent_change(self):
        with patch("concon_v1.co1_defender_controller.ConconDefenderRetakeController") as load:
            controller = ConconDefenderController(search_controller=Mock())
            game = SimpleNamespace(current_attacker_team_ai=SimpleNamespace(name=""))
            controller.set_game(game)
            for opponent in OPPONENT_NAMES:
                game.current_attacker_team_ai.name = opponent
                for site, position in (("L", (7, 3)), ("R", (7, 39))):
                    controller.decide_move(None, dict(grid=get_scenario(site).grid,
                        is_planted=True, planted_pos=position))
                    self.assertEqual(controller.retake_model_paths[site], get_scenario(site).model_path().resolve())
                controller.reset_round()
            self.assertEqual(load.call_count, 2)
            for call, site in zip(load.call_args_list, ("L", "R")):
                self.assertEqual(call.args, (site,))
                self.assertEqual(call.kwargs["model_path"], get_scenario(site).model_path().resolve())

    def test_single_training_opponent_does_not_create_a_dedicated_model(self):
        with patch.object(training, "train") as train:
            training.main(["--opponents", "gc_v1"])
            self.assertEqual(train.call_args.kwargs["opponents"], ["gc_v1"])
            self.assertIsNone(train.call_args.kwargs["save_dir"])
            self.assertNotIn("model_opponent", train.call_args.kwargs)
            training.main([])
            self.assertEqual(train.call_args.kwargs["opponents"], list(training.TRAIN_OPPONENTS))


if __name__ == "__main__":
    unittest.main()
