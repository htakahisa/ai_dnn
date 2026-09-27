import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
from gc_v1.learning_defender_retake_gc import LearningDefenderRetakeGCController
import gc_v1.learning_defender_retake_gc as retake


def defender(name, pos, ability="HUNT", charges=0):
    return NS(name=name, pos=list(pos), team="D", is_alive=True,
              blind_remaining=0, defuse_timer=0, ability_name=ability,
              smoke_charges=charges, flash_charges=0, recon_charges=0,
              hp=100, max_hp=100, role="シーカー")


class DefenderRetakeCoordinationTests(unittest.TestCase):
    def controller(self):
        c = LearningDefenderRetakeGCController.__new__(
            LearningDefenderRetakeGCController)
        c._dist_map = None
        c._dist_map_source = None
        c._last_charge_state = {}
        c._last_detonate_timer = None
        c._utility_recent_min_timer = None
        c._lane_targets = {}
        c.verbose = False
        return c

    def test_isolated_visible_defender_regroups(self):
        c = self.controller()
        me = defender("me", (2, 2))
        ally = defender("ally", (8, 8))
        enemy = NS(name="enemy", pos=[2, 3], team="A", is_alive=True,
                   blind_remaining=0, reveal_remaining=0, hp=100, max_hp=100)
        grid = np.zeros((10, 10), dtype=int)
        state = {"grid": grid, "chars": [me, ally, enemy],
                 "is_planted": True, "planted_pos": [1, 1],
                 "detonate_timer": 55, "smoke_cells": ()}
        result = c.decide_move(me, state)
        self.assertEqual(result, [1, 2])

    def test_nearby_defender_can_smoke_before_defuse(self):
        c = self.controller()
        me = defender("smoker", (5, 5), ability="SMOKE", charges=1)
        ally = defender("defuser", (5, 6))
        enemy = NS(name="enemy", pos=[8, 8], team="A", is_alive=True,
                   blind_remaining=0, reveal_remaining=0, hp=100, max_hp=100)
        grid = np.zeros((12, 12), dtype=int)
        state = {"grid": grid, "chars": [me, ally, enemy],
                 "is_planted": True, "planted_pos": [5, 5],
                 "detonate_timer": 55, "smoke_cells": ()}
        result = c.decide_move(me, state)
        self.assertEqual(result[1]["ability"], "SMOKE")
        self.assertEqual(result[1]["target"], (5, 5))

    def test_started_defuse_continues_despite_visible_enemy(self):
        c = self.controller()
        me = defender("defuser", (5, 5))
        me.defuse_timer = 2
        ally = defender("ally", (5, 6))
        enemy = NS(name="enemy", pos=[5, 8], team="A", is_alive=True)
        state = {"grid": np.zeros((12, 12), dtype=int),
                 "chars": [me, ally, enemy], "is_planted": True,
                 "planted_pos": [5, 5], "detonate_timer": 40}
        self.assertEqual(c.decide_move(me, state), ([5, 5], "DEFUSE"))

    def test_smoke_protected_defuse_starts_before_deadline(self):
        c = self.controller()
        me = defender("defuser", (5, 5))
        ally = defender("ally", (5, 6))
        enemy = NS(name="enemy", pos=[5, 8], team="A", is_alive=True)
        state = {"grid": np.zeros((12, 12), dtype=int),
                 "chars": [me, ally, enemy], "is_planted": True,
                 "planted_pos": [5, 5], "detonate_timer": 40,
                 "smoke_cells": {(5, 5), (5, 6)}}
        self.assertEqual(c.decide_move(me, state), ([5, 5], "DEFUSE"))

    def test_facing_checkpoint_cannot_stay_far_from_retake(self):
        c = self.controller()
        c.facing_head_enabled = True
        me = defender("me", (9, 9))
        ally_a = defender("ally-a", (8, 8))
        ally_b = defender("ally-b", (7, 9))
        enemy = NS(name="enemy", pos=[1, 2], team="A", is_alive=True,
                   blind_remaining=0, reveal_remaining=0, hp=100, max_hp=100)
        state = {"grid": np.zeros((12, 12), dtype=int),
                 "chars": [me, ally_a, ally_b, enemy], "is_planted": True,
                 "planted_pos": [1, 1], "detonate_timer": 45,
                 "smoke_cells": ()}

        result = c.decide_move(me, state)

        self.assertNotEqual(result, list(me.pos))
        self.assertLess(
            abs(result[0] - 1) + abs(result[1] - 1),
            abs(me.pos[0] - 1) + abs(me.pos[1] - 1),
        )


if __name__ == "__main__":
    unittest.main()
