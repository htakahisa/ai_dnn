"""Run the saved foundation through run_game's normal team factory and ticks."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from concon_v1.co1_defender_scenario import get_scenario


class DefenderRuntimeTests(unittest.TestCase):
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
