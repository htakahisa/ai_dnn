from pathlib import Path
import contextlib
import io
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ghost_champions_v2.defender_macro_v1.retake import CoveredRetake


def unit(name, pos, team="D"):
    return NS(name=name, pos=list(pos), team=team, is_alive=True,
              position_known=True, defuse_timer=0, facing="S")


class CoveredRetakeTests(unittest.TestCase):
    def setUp(self):
        self.actor = unit("near", (3, 6))
        self.ally = unit("far", (1, 1))
        self.enemy = unit("enemy", (10, 10), "A")
        self.state = dict(grid=np.zeros((12, 12), dtype=int),
                          chars=[self.actor, self.ally, self.enemy], is_planted=True,
                          planted_pos=(6, 6), smoke_cells={(5, 6), (6, 6)}, detonate_timer=30)
        self.retake = CoveredRetake()

    def test_one_runner_enters_cover_and_starts_defuse_instead_of_staying_outside(self):
        other_action = list(self.ally.pos)
        self.assertIs(self.retake.coordinate(self.ally, self.state, other_action), other_action)
        self.assertEqual(self.retake.runner, self.actor.name)
        for expected in ([4, 6], [5, 6]):
            action = self.retake.coordinate(self.actor, self.state, (self.actor.pos, {"facing": "S"}))
            self.assertEqual(action, (expected, {"facing": "S"}))
            self.actor.pos = action[0]
        self.assertEqual(self.retake.coordinate(self.actor, self.state, self.actor.pos), ([5, 6], "DEFUSE"))

    def test_existing_abilities_commands_and_defuse_progress_take_priority(self):
        for payload in ({"ability": "SMOKE", "target": (6, 6)}, {"ultimate": "RAID"}, "DEFUSE", "COLLECT_ORB"):
            action = (self.actor.pos, payload)
            self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)
        for active in (self.actor, self.ally):
            active.defuse_timer = 3
            action = list(self.actor.pos)
            self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)
            active.defuse_timer = 0

    def test_no_cover_adjacent_guard_or_insufficient_time_keeps_lower_action(self):
        action = list(self.actor.pos)
        self.state["smoke_cells"] = ()
        self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)
        self.state["smoke_cells"] = {(5, 6), (6, 6)}
        self.enemy.pos = [5, 7]
        self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)
        self.enemy.pos = [10, 10]
        self.state["detonate_timer"] = 7
        self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)

    def test_unreachable_or_occupied_cover_is_not_selected(self):
        self.state["grid"][4, :] = 1
        action = list(self.actor.pos)
        self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)
        self.assertIsNone(self.retake.runner)
        self.state["grid"][4, :] = 0
        self.state["smoke_cells"] = {(5, 6)}
        self.enemy.pos = [5, 6]
        self.assertIs(self.retake.coordinate(self.actor, self.state, action), action)

    def test_hidden_enemy_position_does_not_change_runner_or_route(self):
        self.enemy.position_known = False
        actions = []
        for pos in ((5, 6), (10, 10), (-10000, -10000)):
            self.enemy.pos = list(pos)
            self.retake.reset_round()
            actions.append(self.retake.coordinate(self.actor, self.state, list(self.actor.pos)))
        self.assertEqual(actions, [[4, 6]] * 3)
        self.retake.reset_round()
        self.assertIsNone(self.retake.runner)
        self.assertEqual(self.retake.commits, 0)

    def test_real_engine_with_iq_reaches_defuse_range_despite_model_staying(self):
        from ghost_champions_v1_macro import GhostChampionsV1DefenderController
        from ghost_champions_v2.defender_macro_v1.tools.watch_defender_gc import build_match
        from simulation_runtime import cpu_inference
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
            game, defender = build_match("GG", seed=127, headless=True)
            game.defender_setup_phase.finish()
            actor = next(c for c in game.chars if c.team == "D")
            enemy = next(c for c in game.chars if c.team == "A")
            for c in game.chars:
                c.is_alive = c in (actor, enemy)
            actor.pos, enemy.pos = [7, 40], [2, 40]
            actor.iq = 200
            enemy.reveal_remaining = 20
            game.is_planted, game.planted_pos, game.detonate_timer = True, (10, 40), 45
            cells = {(r, c) for r in range(8, 11) for c in range(39, 42) if game.grid[r, c] != 1}
            game.smokes = [dict(cells=cells, remaining_ticks=25, owner=actor.name, team="D", center=(9, 40))]
            def stay(char, state):
                return (list(char.pos), "DEFUSE") if char.defuse_timer > 0 else list(char.pos)
            with patch.object(GhostChampionsV1DefenderController, "decide_move", side_effect=stay):
                for tick in range(1, 13):
                    game.battle_tick = tick
                    game.defender_controller.perception_engine.clear_cache()
                    game.move_character(actor)
                    if actor.defuse_timer >= 6:
                        break
            self.assertGreaterEqual(actor.defuse_timer, 6)
            self.assertLessEqual(max(abs(actor.pos[0]-10), abs(actor.pos[1]-40)), 1)
            self.assertGreater(defender.macro.retake.commits, 0)


if __name__ == "__main__":
    unittest.main()
