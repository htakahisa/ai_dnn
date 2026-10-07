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

    def test_first_six_rounds_cycle_and_share_selection(self):
        controller = self.build()
        for expected in ("A1", "A2", "A3", "A1", "A2", "A3"):
            controller.reset_round()
            for _ in range(5):
                controller.decide_move(Mock(), {})
            self.assertEqual(controller.current_map_name, expected)
        self.assertEqual(controller.controllers["A2"].reset_round.call_count, 2)
        self.assertEqual(controller.controllers["A2"].decide_move.call_count, 10)

    def play_opening(self, controller):
        game = SimpleNamespace(attacker_wins=0, defender_wins=0, _side_swap_count=0)
        controller.set_game(game)
        # A1: 1/2, A2: 2/2, A3: 0/2.
        for won in (True, True, False, False, True, False):
            controller.reset_round()
            if won:
                game.attacker_wins += 1
            else:
                game.defender_wins += 1
            controller.record_opponent_round_end()
            controller.record_opponent_round_end()
        self.assertEqual(controller.map_rounds, dict(A1=2, A2=2, A3=2))
        self.assertEqual(controller.map_wins, dict(A1=1, A2=2, A3=0))
        return game

    def test_best_win_rate_is_updated_after_each_round(self):
        controller = self.build(exploration_rate=0)
        game = self.play_opening(controller)
        for _ in range(3):
            controller.reset_round()
            self.assertEqual(controller.current_map_name, "A2")
            game.defender_wins += 1
            # reset_round also records results if the hook was omitted.
        controller.reset_round()
        self.assertEqual(controller.current_map_name, "A1")
        self.assertEqual(controller.map_rounds["A2"], 5)

    def test_exploration_uses_other_scenarios_and_ties_are_random(self):
        controller = self.build()
        self.play_opening(controller)
        with patch.object(controller.rng, "random", return_value=0.05), \
                patch.object(controller.rng, "choice", return_value="A3") as choice:
            controller.reset_round()
            choice.assert_called_once_with(("A1", "A3"))
        controller.map_wins = dict(A1=1, A2=1, A3=1)
        with patch.object(controller.rng, "choice", return_value="A2") as choice:
            controller.reset_round()
            choice.assert_called_once_with(("A1", "A2", "A3"))

    def test_side_swap_does_not_count_defensive_results(self):
        controller = self.build()
        game = self.play_opening(controller)
        controller.reset_round()
        game._side_swap_count += 1
        game.attacker_wins, game.defender_wins = 4, 3
        controller.record_opponent_round_end()
        self.assertEqual(sum(controller.map_rounds.values()), 6)

    def test_new_match_restarts_opening_and_stats(self):
        controller = self.build()
        self.play_opening(controller)
        controller.set_game(SimpleNamespace(attacker_wins=0, defender_wins=0))
        controller.reset_round()
        self.assertEqual(controller.current_map_name, "A1")
        self.assertEqual(sum(controller.map_rounds.values()), 0)

    @_run_from_project_root
    def test_game_round_end_hook_records_through_iq_wrapper(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR

        controller = self.build(exploration_rate=0)
        team = _build_team_ai("default")
        team.attacker_factory = lambda: controller
        game = VisualFPSBattle(NEW_MAZE_STR, team, _build_team_ai("default"), headless=True)
        game.special_round_banner = None
        for expected, won in zip(("A1", "A2", "A3", "A1", "A2", "A3"),
                                 (True, True, False, False, True, False)):
            self.assertEqual(controller.current_map_name, expected)
            game.round_over = True
            if won:
                game.attacker_wins += 1
            else:
                game.defender_wins += 1
            game.check_match_winner()
        self.assertEqual(controller.current_map_name, "A2")
        self.assertEqual(controller.map_wins, dict(A1=1, A2=2, A3=0))
        self.assertEqual(controller.map_rounds, dict(A1=2, A2=2, A3=2))

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
        for rate in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self.build(exploration_rate=rate)


if __name__ == "__main__":
    unittest.main()
