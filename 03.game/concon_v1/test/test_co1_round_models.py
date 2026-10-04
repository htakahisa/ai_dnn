import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from concon_v1.co1_attacker_controller import (
    ConconAttackerController, ConconRoundAttackerController,
)


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

    def test_selection_is_shared_and_allows_consecutive_same_model(self):
        controller = self.build()
        with patch.object(controller.rng, "choice", side_effect=["A2", "A2", "A3"]) as choice:
            for expected in ("A2", "A2", "A3"):
                controller.reset_round()
                for _ in range(5):
                    controller.decide_move(Mock(), {})
                self.assertEqual(controller.current_map_name, expected)
            self.assertEqual(choice.call_count, 3)
            self.assertTrue(all(call.args[0] == ("A1", "A2", "A3")
                                for call in choice.call_args_list))
        self.assertEqual(controller.controllers["A2"].reset_round.call_count, 2)
        self.assertEqual(controller.controllers["A2"].decide_move.call_count, 10)

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
