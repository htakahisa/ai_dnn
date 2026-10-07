import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_attacker_controller import (
    ConconAttackerController, ConconRoundAttackerController,
)
from concon_v1.co1_battle_training import _run_from_project_root


class RoundModelTests(unittest.TestCase):
    def build(self, names=("A1", "A2", "A3"), **kwargs):
        def scenario(name):
            return SimpleNamespace(
                model_path=SimpleNamespace(is_file=lambda: True),
            )

        with patch("concon_v1.co1_attacker_controller.get_scenario", side_effect=scenario), \
                patch("concon_v1.co1_attacker_controller.ConconAttackerController",
                      side_effect=lambda **kw: Mock(map_name=kw["map_name"])):
            return ConconRoundAttackerController(map_names=names, seed=7, **kwargs)

    def test_each_round_randomly_selects_and_shares_one_map(self):
        from concon_v1.co1_attacker_scenarios import CONCON_ATTACKER_MAP
        names = ("A1", "A2", "A3", "A4")
        self.assertEqual(CONCON_ATTACKER_MAP, names)
        controller = self.build(names)
        selected = ("A4", "A4", "A1", "A3", "A2", "A4", "A2", "A1", "A3", "A3")
        with patch.object(controller.rng, "choice", side_effect=selected) as choice:
            for expected in selected:
                controller.reset_round()
                self.assertEqual(controller.current_map_name, expected)
                for _ in range(5):
                    controller.decide_move(Mock(), {})
                choice.assert_called_with(names)
            self.assertEqual(choice.call_count, len(selected))
        self.assertEqual(controller.controllers["A4"].reset_round.call_count, 3)
        self.assertEqual(controller.controllers["A4"].decide_move.call_count, 15)

    def test_score_and_side_swap_do_not_change_selection_candidates(self):
        names = ("A1", "A2", "A3", "A4")
        controller = self.build(names)
        game = SimpleNamespace(attacker_wins=0, defender_wins=0, _side_swap_count=0)
        controller.set_game(game)
        with patch.object(controller.rng, "choice", return_value="A4") as choice:
            for i in range(12):
                game.attacker_wins = i
                game._side_swap_count = i // 6
                controller.reset_round()
                self.assertEqual(controller.current_map_name, "A4")
                choice.assert_called_with(names)
        self.assertEqual(choice.call_count, 12)

    @_run_from_project_root
    def test_game_starts_each_round_with_random_selection(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR

        controller = self.build(("A1", "A2", "A3", "A4"))
        team = _build_team_ai("default")
        team.attacker_factory = lambda: controller
        with patch.object(controller.rng, "choice", side_effect=("A4", "A4", "A2")) as choice:
            game = VisualFPSBattle(NEW_MAZE_STR, team, _build_team_ai("default"), headless=True)
            game.special_round_banner = None
            self.assertEqual(controller.current_map_name, "A4")
            for expected in ("A4", "A2"):
                game.round_over = True
                game.attacker_wins += 1
                game.check_match_winner()
                self.assertEqual(controller.current_map_name, expected)
            self.assertEqual(choice.call_count, 3)

    def test_single_string_and_game_rebinding(self):
        controller = self.build("A1")
        game = Mock()
        controller.set_game(game)
        controller.reset_round()
        self.assertEqual(controller.current_map_name, "A1")
        controller.current_controller.set_game.assert_called_with(game)
        new_game = Mock()
        controller.set_game(new_game)
        controller.current_controller.set_game.assert_called_with(new_game)

    def test_postplant_uses_actual_site_once_and_reuses_loaded_controller(self):
        left, right = Mock(), Mock()
        left_factory, right_factory = Mock(return_value=left), Mock(return_value=right)
        controller = self.build("A1", postplant_factories={
            "left": (left_factory,), "right": (right_factory,),
        })
        game = Mock()
        controller.set_game(game)
        state = {"is_planted": True, "planted_pos": (0, 3), "grid": [[0] * 4]}
        for _ in range(2):
            controller.reset_round()
            self.assertIsNone(controller.current_controller.postplant_controller)
            for _ in range(5):
                controller.decide_move(Mock(), state)
            self.assertIs(controller.current_controller.postplant_controller, right)
        left_factory.assert_not_called()
        right_factory.assert_called_once()
        self.assertEqual(right.reset_round.call_count, 2)
        right.set_game.assert_called_with(game)

    def test_unconfigured_postplant_keeps_default(self):
        controller = self.build("A2")
        controller.reset_round()
        controller.decide_move(Mock(), {
            "is_planted": True, "planted_pos": (0, 0), "grid": [[0] * 4],
        })
        self.assertIsNone(controller.current_controller.postplant_controller)

    def test_runtime_adapter_dispatches_postplant_model(self):
        route = Mock(scenario=None)
        from concon_v1.co1_attacker_scenarios import get_scenario
        route.scenario = get_scenario("A1")
        adapter = ConconAttackerController(route_controller=route)
        adapter.enemy_sightings = Mock()
        adapter.enemy_sightings.guard_move.side_effect = lambda result, *_: result
        adapter.postplant_controller = Mock()
        adapter.postplant_controller.decide_move.return_value = ([1, 2], None)
        char = SimpleNamespace(pos=[1, 2])
        state = {"is_planted": True, "grid": [[0] * 4]}
        self.assertEqual(adapter.decide_move(char, state), ([1, 2], None))
        adapter.postplant_controller.decide_move.assert_called_once_with(char, state)
        route.decide_move.assert_not_called()

    def test_invalid_candidates(self):
        for names in ((), ("A1", "A1")):
            with self.assertRaises(ValueError):
                self.build(names)


if __name__ == "__main__":
    unittest.main()
