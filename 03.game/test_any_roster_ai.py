"""Selected policies run on replacement players in setup and both live phases."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from roster_utils import model_slots


class AnyRosterAITest(unittest.TestCase):
    def test_model_slots_keep_originals_and_reserve_dead_players(self):
        names = ("A", "B", "C", "D", "E")
        players = [SimpleNamespace(name=n, base_name=n, is_alive=True) for n in ("X", "B", "Y", "E", "Z")]
        slots = model_slots(players, names)
        self.assertEqual(slots, {"B": 1, "E": 4, "X": 0, "Y": 2, "Z": 3})
        players[0].is_alive = False
        self.assertEqual(model_slots(players, names), slots)
        # Opposing sides may use the same player with a runtime name suffix.
        players[1].name = "B_2"
        self.assertEqual(model_slots(players, names)["B_2"], 1)

    def test_every_selectable_ai_runs_on_different_members_in_all_phases(self):
        from roster_select import TEAM_AI_OPTIONS
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        from simulation_runtime import cpu_inference
        rosters = (["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"],
                   ["Furina", "Nanasaki", "Lohen", "WoohyuN", "Alfajer"])
        rival = ["Aspas", "valyn", "trent", "leaf", "tex"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, cpu_inference(enabled=True):
            for key, own in ((key, own) for key in TEAM_AI_OPTIONS.values() for own in rosters):
                with self.subTest(ai=key, roster=own), contextlib.redirect_stdout(io.StringIO()):
                    defenders = rival if own == rosters[0] else list(reversed(own))
                    attacker, defender = _build_team_ai(key), _build_team_ai(key)
                    game = VisualFPSBattle(NEW_MAZE_STR, attacker, defender, headless=True,
                        attacker_roster=own, defender_roster=defenders,
                        spike_holder_name=own[0], defender_spike_holder_name=defenders[0])
                    game.record_replay = False
                    # Keep controller diagnostics inside the test directory.
                    for wrapper in (game.attacker_controller, game.defender_controller):
                        raw = getattr(wrapper, "inner_controller", wrapper)
                        for ctrl in (raw, *[getattr(raw, attr) for attr in
                                     ("search_controller", "retake_controller", "guard_controller",
                                      "carry", "escort", "retrieve", "guard", "macro_controller")
                                     if hasattr(raw, attr)]):
                            if hasattr(ctrl, "verbose"):
                                ctrl.verbose = False
                            if hasattr(ctrl, "_debug_log_path"):
                                ctrl._debug_log_path = str(Path(directory) / f"{key}.log")
                    for _ in range(2):
                        game._simulate_tick()
                    game.defender_setup_phase.finish()
                    for _ in range(3):
                        game._simulate_tick()
                    r, c = next((r, c) for r, row in enumerate(game.grid)
                                for c, value in enumerate(row) if value == 2)
                    game.is_planted, game.planted_pos = True, (r, c)
                    game.spike_pos = None
                    for player in game.chars:
                        player.has_spike = False
                        player.ultimate_points = player.ultimate_cost
                    for _ in range(3):
                        game._simulate_tick()
                    self.assertIs(game.initial_attacker_team_ai, attacker)
                    self.assertIs(game.initial_defender_team_ai, defender)
                    if key in ("frc_v1", "frc_v1_baseline"):
                        from frc_v1.model import FrcPolicy
                        from frc_v1.baseline import FrcBaseline
                        self.assertIsInstance(game.attacker_controller.actor,
                                              FrcPolicy if key == "frc_v1" else FrcBaseline)
                        self.assertEqual({a.name for a in game.attacker_controller.snapshot.allies}, set(own))


if __name__ == "__main__":
    unittest.main()
