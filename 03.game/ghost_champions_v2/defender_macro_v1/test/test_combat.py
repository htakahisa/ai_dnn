from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ghost_champions_v2.defender_macro_v1.combat import CombatFacing, visible_targets


def unit(name, pos, team="D", **kwargs):
    return NS(name=name, pos=list(pos), team=team, is_alive=True,
              position_known=True, hp=100, facing="W", **kwargs)


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.actor = unit("defender", (2, 2))
        self.enemy = unit("attacker", (2, 7), "A")
        self.state = dict(grid=np.zeros((12, 12), dtype=int), chars=[self.actor, self.enemy],
                          is_planted=False, smoke_cells=())
        self.guard = CombatFacing()

    def test_repeated_learned_backwards_facing_cannot_cancel_observed_engagement(self):
        for tick in range(5):
            self.actor.facing = "W"
            result = self.guard.coordinate(self.actor, self.state, ([2, 2], {"facing": "W"}))
            self.assertEqual(result, ([2, 2], {"facing": "E"}))
            self.assertEqual(self.actor.facing, "E")

    def test_retake_facing_remains_owned_by_existing_retake_controller(self):
        self.state["is_planted"] = True
        result = ([2, 2], {"facing": "W"})
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)

    def test_cover_and_smoke_keep_original_action_when_there_is_no_shootable_target(self):
        result = ([2, 2], {"facing": "W"})
        self.enemy.position_known = False
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)
        self.enemy.position_known = True
        self.state["grid"][2, 4] = 1
        self.assertEqual(visible_targets(self.actor, self.state), [])
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)
        self.state["grid"][2, 4] = 0
        self.state["smoke_cells"] = {(2, 4)}
        self.assertEqual(visible_targets(self.actor, self.state), [])
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)
        self.state["smoke_cells"] = ()
        self.state["chars"].append(unit("blocker", (2, 4)))
        self.assertEqual(visible_targets(self.actor, self.state), [])
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)

    def test_clear_engagement_has_priority_over_a_nearer_sighting_behind_cover(self):
        hidden_by_cover = unit("near", (4, 2), "A")
        self.state["chars"].append(hidden_by_cover)
        self.state["grid"][3, 2] = 1
        self.assertEqual(self.guard.coordinate(self.actor, self.state, [2, 2])[1]["facing"], "E")

    def test_distant_sighting_behind_cover_does_not_cancel_patrol_facing(self):
        self.state["grid"] = np.zeros((30, 30), dtype=int)
        self.enemy.pos = [2, 25]
        self.state["grid"][2, 4] = 1
        result = ([2, 3], {"facing": "N"})
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)

    def test_hidden_enemy_coordinates_do_not_change_action(self):
        self.enemy.position_known = False
        action = ([2, 2], {"facing": "W"})
        for pos in ((2, 7), (2, 1), (-10000, -10000)):
            self.enemy.pos = list(pos)
            self.assertIs(self.guard.coordinate(self.actor, self.state, action), action)

    def test_engine_facing_lock_wins_over_enemy_and_learned_mutation(self):
        self.actor.facing = "W"
        result = self.guard.coordinate(self.actor, self.state, ([2, 3], {"facing": "W"}), locked_facing="N")
        self.assertEqual(result, ([2, 3], {"facing": "N"}))
        self.assertEqual(self.actor.facing, "N")

    def test_ability_target_and_movement_are_preserved_while_facing_is_corrected(self):
        payload = dict(ability="FLASH", target=(5, 7), facing="W")
        result = self.guard.coordinate(self.actor, self.state, ([3, 2], payload))
        self.assertEqual(result, ([3, 2], dict(payload, facing="E")))
        self.assertEqual(payload["facing"], "W")
        for command in ("DEFUSE", "COLLECT_ORB"):
            original = ([2, 2], command)
            self.assertIs(self.guard.coordinate(self.actor, self.state, original), original)

    def test_setup_never_uses_enemy_positions(self):
        self.state["defender_setup_active"] = True
        result = [2, 3]
        self.assertIs(self.guard.coordinate(self.actor, self.state, result), result)


if __name__ == "__main__":
    unittest.main()
