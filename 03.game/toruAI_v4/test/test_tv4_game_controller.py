"""Dropdown integration, opponent selection and phase-specific fallback."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.test.test_tv4_site import world
from toruAI_v4 import tv4_game_controller as module


class GameControllerTests(unittest.TestCase):
    def game(self, name="Fnatic v3"):
        game = world(Scenario())
        game.current_attacker_team_ai = SimpleNamespace(name=name)
        return game

    def test_unknown_opponent_and_missing_search_use_generic(self):
        for name in ("ロジック", "Fnatic v3"):
            with patch.object(module, "IQAwareController") as generic, \
                 patch.object(module, "load_policy", side_effect=FileNotFoundError("missing")), \
                 contextlib.redirect_stdout(io.StringIO()):
                controller = module.ToruV4GameDefenderController()
                game = self.game(name)
                controller.set_game(game)
                controller.decide_move(game.chars[0], {})
                generic.return_value.decide_move.assert_called_once()
                self.assertIsNone(controller.learned)

    def test_learned_search_and_partial_retake_choose_phase_and_actual_site(self):
        game = self.game()
        policy = Mock()
        search_hash = module.hashlib.sha256(b"search").hexdigest()
        def load(path, phase, scenario):
            if phase == "search":
                return policy, {"opponent": "fnatic_v3", "analysis_hashes": {"fnatic_v3": "analysis"}}
            if path.name == "retake_R_best.pt":
                raise FileNotFoundError("not trained")
            return policy, {"opponent": "fnatic_v3", "site": "L", "frozen_search_hash": search_hash,
                            "analysis_hashes": {"fnatic_v3": "analysis"}}
        def learned_factory(opponent, **kwargs):
            return SimpleNamespace(retake=kwargs["retake"], set_game=Mock(), prepare_team_tick=Mock(),
                                   decide_move=Mock(), reset_round=Mock(), record_opponent_round_end=Mock())
        with patch.object(module, "IQAwareController"), patch.object(module, "load_policy", side_effect=load), \
             patch.object(module, "load_analyses", return_value=({"fnatic_v3": Mock()}, {"fnatic_v3": "analysis"})), \
             patch.object(module, "ToruV4DefenderController", side_effect=learned_factory), \
             patch.object(Path, "read_bytes", return_value=b"search"), contextlib.redirect_stdout(io.StringIO()):
            controller = module.ToruV4GameDefenderController()
            controller.set_game(game)
            controller.prepare_team_tick()
            controller.decide_move(game.chars[0], {})
            self.assertEqual(controller.learned.decide_move.call_count, 1)
            game.is_planted, game.planted_pos = True, controller.scenario.sites["R"][0]
            controller.decide_move(game.chars[0], {"is_planted": True})
            controller.generic.decide_move.assert_called_once()
            game.planted_pos = controller.scenario.sites["L"][0]
            controller.decide_move(game.chars[0], {"is_planted": True})
            self.assertEqual(controller.learned.decide_move.call_count, 2)

    def test_rebinding_selects_the_new_opponent_without_reloading_each_tick(self):
        with patch.object(module, "IQAwareController"), \
             patch.object(module, "load_policy", side_effect=FileNotFoundError("missing")) as load, \
             contextlib.redirect_stdout(io.StringIO()):
            controller = module.ToruV4GameDefenderController()
            game = self.game()
            controller.set_game(game)
            controller.set_game(game)
            self.assertEqual(load.call_count, 1)
            self.assertIn("fnatic_v3", str(load.call_args.args[0]))
            game.current_attacker_team_ai.name = "Touyama Gaming v2"
            controller.set_game(game)
            self.assertEqual(controller.opponent, "touyama_v2")
            self.assertIn("touyama_v2", str(load.call_args.args[0]))

    def test_dropdown_and_team_factory_provide_toru_v4_both_roles(self):
        from roster_select import TEAM_AI_OPTIONS
        from toruAI_v4.tv4_train_analysis import legacy_root
        with legacy_root():
            from run_game import _build_team_ai
        from controllers import DefaultAttackerController
        self.assertEqual(TEAM_AI_OPTIONS["Toru AI v4"], "toru_ai_v4")
        team = _build_team_ai("toru_ai_v4")
        self.assertEqual(team.name, "Toru AI v4")
        self.assertIsInstance(team.attacker_factory(), DefaultAttackerController)
        self.assertIsInstance(team.defender_factory(), module.ToruV4GameDefenderController)


if __name__ == "__main__":
    unittest.main()
