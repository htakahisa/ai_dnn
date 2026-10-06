"""Run the saved foundation through run_game's normal team factory and ticks."""

import contextlib
import io
from pathlib import Path
import sys
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_defender_scenario import get_scenario


class DefenderRuntimeTests(unittest.TestCase):
    def test_controller_switches_between_distinct_left_and_right_best_paths(self):
        from concon_v1.co1_defender_controller import ConconDefenderController
        from concon_v1.co1_retake_scenarios import get_scenario as retake_scenario
        with tempfile.TemporaryDirectory() as temporary:
            paths = {site: Path(temporary) / f"{site}_best.pt" for site in ("L", "R")}
            for path in paths.values():
                path.touch()
            search = Mock()
            left, right = Mock(), Mock()
            left.decide_move.return_value, right.decide_move.return_value = "left", "right"
            with patch("concon_v1.co1_defender_controller.ConconDefenderRetakeController", side_effect=[left, right]) as loader:
                controller = ConconDefenderController(search_controller=search, retake_model_paths=paths)
                controller.set_game(SimpleNamespace())
                state = dict(grid=retake_scenario("L").grid, is_planted=False)
                controller.decide_move(None, state)
                search.decide_move.assert_called_once()
                state.update(is_planted=True, planted_pos=(7, 3))
                self.assertEqual(controller.decide_move(None, state), "left")
                state["planted_pos"] = (7, 40)
                self.assertEqual(controller.decide_move(None, state), "right")
                state["planted_pos"] = (7, 3)
                self.assertEqual(controller.decide_move(None, state), "left")
                self.assertEqual(loader.call_count, 2)
                self.assertEqual(loader.call_args_list[0].args, ("L",))
                self.assertEqual(loader.call_args_list[0].kwargs["model_path"], paths["L"].resolve())
                self.assertEqual(loader.call_args_list[1].args, ("R",))
                self.assertEqual(loader.call_args_list[1].kwargs["model_path"], paths["R"].resolve())
                controller.reset_round()
                left.reset_round.assert_called_once()
                right.reset_round.assert_called_once()

    def test_missing_best_is_reported_once(self):
        from concon_v1.co1_defender_controller import ConconDefenderController
        from concon_v1.co1_retake_scenarios import get_scenario as retake_scenario
        with tempfile.TemporaryDirectory() as temporary:
            paths = {site: Path(temporary) / f"missing_{site}_best.pt" for site in ("L", "R")}
            controller = ConconDefenderController(search_controller=Mock(), retake_model_paths=paths)
            controller.default_controller = Mock()
            state = dict(grid=retake_scenario("L").grid, is_planted=True, planted_pos=(7, 3))
            with contextlib.redirect_stdout(io.StringIO()) as log:
                controller.decide_move(None, state)
                controller.decide_move(None, state)
            self.assertEqual(log.getvalue().count("checkpoint missing:"), 1)
            self.assertEqual(controller.default_controller.decide_move.call_count, 2)

    def test_run_game_factory_uses_saved_lr_best_with_production_iq(self):
        import torch
        from concon_v1.co1_retake_scenarios import get_scenario as retake_scenario, validate_checkpoint
        from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
        if not all(retake_scenario(site).model_path("best").is_file() for site in ("L", "R")):
            self.skipTest("train/evaluate both retake best checkpoints manually first")
        for site in ("L", "R"):
            scenario = retake_scenario(site)
            checkpoint = torch.load(scenario.model_path("best"), map_location="cpu", weights_only=False)
            try:
                validate_checkpoint(checkpoint, scenario, checkpoint.get("ability_distances", 6),
                                    allow_legacy_coordination=True)
            except ValueError as error:
                self.skipTest(f"retrain {site} best for the current A map/observations first: {error}")
        from concon_v1.co1_battle_training import _run_from_project_root
        from iq_controller_adapter import IQAwareController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        torch.set_num_threads(1)

        @_run_from_project_root
        def run():
            with contextlib.redirect_stdout(io.StringIO()) as log:
                from run_game import VisualFPSBattle, _build_team_ai
                defenders = get_preset("Gorigons")
                game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("concon_v1"),
                    headless=True, defender_roster=list(defenders.players),
                    defender_spike_holder_name=defenders.spike_holder, defender_igl_name=defenders.igl,
                    disable_side_swap=True)
                game.analytics_tracker, game.record_replay = None, False
                adapter = game.defender_controller.inner
                for site, position in (("L", (7, 3)), ("R", (7, 40))):
                    game.init_round()
                    game.defender_setup_phase.finish()
                    game.is_planted, game.planted_pos, game.spike_pos = True, position, None
                    game.detonate_timer = 30
                    with patch.object(adapter.search_controller, "decide_move", side_effect=AssertionError("retake used search")):
                        self.assertTrue(game.step_tick())
                    controller = adapter.retake_controllers[site]
                    self.assertIsInstance(controller, ConconDefenderRetakeController)
                    self.assertEqual(controller.scenario.map_name, site)
                    self.assertEqual(controller.model_path, retake_scenario(site).model_path("best").resolve())
                    self.assertFalse(controller.model.training)
                    self.assertTrue(all(not p.requires_grad for p in controller.model.parameters()))
                    self.assertIsInstance(game.defender_controller, IQAwareController)
                    self.assertEqual(game.detonate_timer, 29)
                self.assertIn("[ConCon defender retake] site=L", log.getvalue())
                self.assertIn("[ConCon defender retake] site=R", log.getvalue())
        run()

    @unittest.skipUnless(get_scenario().model_path.is_file(), "train the defender foundation manually first")
    def test_saved_model_runs_setup_and_live_through_game_factory(self):
        import torch
        from concon_v1.co1_battle_training import _run_from_project_root
        from concon_v1.co1_defender_controller import ConconDefenderController
        from iq_controller_adapter import IQAwareController
        from map_data import NEW_MAZE_STR
        from party_presets import get_preset
        from roster_select import TEAM_AI_OPTIONS

        torch.set_num_threads(1)
        defenders = get_preset("Gorigons")
        attackers = get_preset("Omoko Gaming")
        self.assertEqual(defenders.default_ai, TEAM_AI_OPTIONS["ConCon v1"])

        @_run_from_project_root
        def run():
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                from run_game import VisualFPSBattle, _build_team_ai
                game = VisualFPSBattle(
                    NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai(defenders.default_ai),
                    headless=True, attacker_roster=list(attackers.players), defender_roster=list(defenders.players),
                    spike_holder_name=attackers.spike_holder, defender_spike_holder_name=defenders.spike_holder,
                    attacker_igl_name=attackers.igl, defender_igl_name=defenders.igl, disable_side_swap=True,
                )
            game.analytics_tracker = None
            game.stop_after_round = True
            wrapper = game.defender_controller
            self.assertIsInstance(wrapper, IQAwareController)
            adapter = wrapper.inner_controller
            self.assertIsInstance(adapter, ConconDefenderController)
            self.assertEqual(adapter.search_controller.model_path, get_scenario().runtime_model_path.resolve())
            self.assertIn("[ConCon defender search] model=", log.getvalue())
            chars = [char for char in game.chars if char.team == "D"]
            starts = [tuple(char.pos) for char in chars]
            while game.defender_setup_phase.active:
                self.assertTrue(game.step_tick())
            self.assertTrue(any(tuple(char.pos) != start for char, start in zip(chars, starts)))
            assignments = dict(adapter.search_controller.assignments)
            self.assertEqual(len(set(assignments.values())), 5)
            for _ in range(10):
                if not game.step_tick():
                    break
            self.assertGreater(game.battle_tick, 0)
            self.assertEqual(adapter.search_controller.assignments, assignments)
            self.assertIs(game.defender_controller.inner_controller, adapter)

        run()


if __name__ == "__main__":
    unittest.main()
