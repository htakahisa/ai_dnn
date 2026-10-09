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
        from toruAI_v4.tv4_train_defender_analysis import legacy_root
        with legacy_root():
            from run_game import _build_team_ai
        self.assertEqual(TEAM_AI_OPTIONS["Toru AI v4"], "toru_ai_v4")
        team = _build_team_ai("toru_ai_v4")
        self.assertEqual(team.name, "Toru AI v4")
        self.assertIsInstance(team.attacker_factory(), module.ToruV4GameAttackerController)
        self.assertIsInstance(team.defender_factory(), module.ToruV4GameDefenderController)

    def test_attacker_loads_opponent_best_and_preserves_plant_guard_dispatch(self):
        game = self.game()
        game.current_defender_team_ai = SimpleNamespace(name='Fnatic v3')
        source = dict(analysis=Mock(), plant=Mock(), hashes={'analysis': 'analysis', 'plant': 'plant'},
                      paths={'analysis': 'attacker_analysis_best.pt', 'plant': 'attacker_plant_best.pt'})
        guard = Mock()
        def combined(*args):
            return SimpleNamespace(guard_policy=args[3], plant=SimpleNamespace(cache=None), set_game=Mock(),
                prepare_team_tick=Mock(), decide_move=Mock(), reset_round=Mock(), record_opponent_round_end=Mock())
        with patch.object(module, 'IQAwareController'), patch.object(module, 'load_sources', return_value={'fnatic_v3': source}) as sources, \
             patch.object(module, 'load_guard', return_value=(guard, {})) as load, \
             patch.object(module, 'ToruV4AttackerPlantGuardController', side_effect=combined), contextlib.redirect_stdout(io.StringIO()):
            controller = module.ToruV4GameAttackerController()
            controller.set_game(game)
            controller.set_game(game)
            self.assertEqual(sources.call_count, 1)
            self.assertEqual(load.call_args.args[3], source['hashes'])
            controller.decide_move(game.chars[0], {})
            game.is_planted, game.planted_pos = True, controller.scenario.sites['L'][0]
            controller.prepare_team_tick()
            controller.decide_move(game.chars[0], {'is_planted': True})
            self.assertEqual(controller.learned.decide_move.call_count, 2)
            controller.reset_round()
            controller.record_opponent_round_end()
            controller.learned.reset_round.assert_called_once()
            controller.learned.record_opponent_round_end.assert_called_once()
            controller.generic.decide_move.assert_not_called()

    def test_missing_guard_uses_learned_plant_then_generic_after_boundary(self):
        game = self.game()
        game.current_defender_team_ai = SimpleNamespace(name='Fnatic v3')
        source = dict(analysis=Mock(), plant=Mock(), hashes={}, paths={'analysis': 'analysis', 'plant': 'plant'})
        learned = SimpleNamespace(guard_policy=None, plant=SimpleNamespace(cache=(1, 'live', 3, False)),
            set_game=Mock(), decide_move=Mock(), prepare_team_tick=Mock())
        with patch.object(module, 'IQAwareController'), patch.object(module, 'load_sources', return_value={'fnatic_v3': source}), \
             patch.object(module, 'load_guard', side_effect=ValueError('guard hash mismatch')), \
             patch.object(module, 'ToruV4AttackerPlantGuardController', return_value=learned), contextlib.redirect_stdout(io.StringIO()):
            controller = module.ToruV4GameAttackerController()
            controller.set_game(game)
            controller.decide_move(game.chars[0], {})
            learned.decide_move.assert_called_once()
            game.is_planted, game.planted_pos, game.battle_tick = True, controller.scenario.sites['L'][0], 3
            controller.decide_move(game.chars[0], {'is_planted': True})
            controller.generic.decide_move.assert_not_called()
            game.battle_tick = 4
            controller.decide_move(game.chars[0], {'is_planted': True})
            controller.generic.decide_move.assert_called_once()
            self.assertIn('hash mismatch', controller.status['guard'])

    def test_attacker_observes_executed_moves_once_before_next_tick(self):
        game = self.game()
        game.battle_tick = 4
        controller = module.ToruV4GameAttackerController()
        plant = SimpleNamespace(cache=(game.current_round, 'live', 3, False),
                                decisions={'ally': 'previous action'}, observe_executed_moves=Mock())
        controller.game = game
        controller.learned = SimpleNamespace(plant=plant, guard_policy=Mock(), prepare_team_tick=Mock(),
                                            decide_move=Mock(), reset_round=Mock())
        controller.prepare_team_tick()
        controller.decide_move(game.chars[0], {})
        plant.observe_executed_moves.assert_called_once_with([c for c in game.chars if c.team == 'A'])
        self.assertEqual(plant.decisions, {})
        plant.cache = (game.current_round, 'live', 4, False)
        game.battle_tick = 5
        controller.prepare_team_tick()
        self.assertEqual(plant.observe_executed_moves.call_count, 2)
        controller.reset_round()
        self.assertIsNone(controller.observed_tick)

    def test_attacker_unknown_opponent_uses_generic_and_rebinds_for_side_swap(self):
        game = self.game()
        game.current_defender_team_ai = SimpleNamespace(name='Unknown AI')
        with patch.object(module, 'IQAwareController'), patch.object(module, 'load_sources', side_effect=FileNotFoundError('missing')) as load, \
             contextlib.redirect_stdout(io.StringIO()):
            controller = module.ToruV4GameAttackerController()
            controller.set_game(game)
            controller.decide_move(game.chars[0], {})
            controller.generic.decide_move.assert_called_once()
            load.assert_not_called()
            game.current_defender_team_ai.name = 'FRC v1'
            controller.set_game(game)
            self.assertEqual(controller.opponent, 'frc_v1')
            self.assertEqual(load.call_args.args[1], ('frc_v1',))


if __name__ == "__main__":
    unittest.main()
