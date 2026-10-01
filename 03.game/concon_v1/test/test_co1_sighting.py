"""Shared enemy sightings and movement facing for the ConCon attackers."""

import unittest
from types import SimpleNamespace

from concon_v1.co1_attacker_controller import ConconAttackerController
from concon_v1.co1_attacker_sighting import (
    SIGHTING_MEMORY_TICKS, TeamEnemySightings,
)
from concon_v1.co1_train_attacker_A1 import GRID


class SightingTests(unittest.TestCase):
    def setUp(self):
        self.scout = SimpleNamespace(name="scout", team="A", pos=[5, 2],
                                     is_alive=True, has_spike=True)
        self.teammate = SimpleNamespace(name="teammate", team="A", pos=[5, 3],
                                        is_alive=True, has_spike=False)
        self.enemy = SimpleNamespace(name="enemy", team="D", pos=[5, 7],
                                     is_alive=True)
        self.chars = [self.scout, self.teammate, self.enemy]
        self.game = SimpleNamespace(
            check_line_of_sight=lambda observer, enemy: observer is self.scout,
        )

    def test_teammate_uses_scout_sighting_to_face_enemy_while_moving(self):
        sightings = TeamEnemySightings()
        sightings.observe(self.scout, self.chars, self.game, GRID, 4)
        sightings.observe(self.teammate, self.chars, self.game, GRID, 4)
        self.assertEqual(sightings.last_seen["enemy"], ((5, 7), 4))
        self.assertEqual(sightings.guard_move([5, 4], self.teammate.pos, 4),
                         ([5, 4], {"facing": "E"}))
        self.assertEqual(sightings.guard_move([5, 3], self.teammate.pos, 4),
                         ([5, 3], {"facing": "E"}))
        self.assertEqual(sightings.guard_move(([5, 3], {"facing": "N"}),
                                              self.teammate.pos, 4),
                         ([5, 3], {"facing": "N"}))
        self.assertEqual(sightings.guard_move(([5, 3], "PLANT"),
                                              self.teammate.pos, 4),
                         ([5, 3], "PLANT"))

    def test_only_visible_sightings_are_shared_and_memory_expires(self):
        sightings = TeamEnemySightings()
        sightings.observe(self.teammate, self.chars, self.game, GRID, 2)
        self.assertEqual(sightings.last_seen, {})
        sightings.observe(self.scout, self.chars, self.game, GRID, 3)
        self.assertEqual(sightings.facing(self.teammate.pos, 3), "E")
        self.assertIsNone(sightings.facing(self.teammate.pos,
                                           3 + SIGHTING_MEMORY_TICKS))
        sightings.observe(self.scout, self.chars, self.game, GRID, 20)
        sightings.reset_round()
        self.assertEqual(sightings.last_seen, {})

    def test_runtime_controller_shares_sighting_across_teammates(self):
        controller = ConconAttackerController.__new__(ConconAttackerController)
        controller.enemy_sightings = TeamEnemySightings()
        controller.route_controller = SimpleNamespace(
            set_game=lambda _: None, reset_round=lambda: None,
            decide_move=lambda char, _: [char.pos[0], char.pos[1] + 1],
        )
        controller.retrieve_controller = SimpleNamespace(
            set_game=lambda _: None, reset_round=lambda: None,
        )
        controller.default_controller = SimpleNamespace(reset_round=lambda: None)
        controller.set_game(self.game)
        state = {"chars": self.chars, "grid": GRID, "battle_tick": 5}
        controller.decide_move(self.scout, state)
        self.assertEqual(controller.decide_move(self.teammate, state),
                         ([5, 4], {"facing": "E"}))
        controller.reset_round()
        self.assertEqual(controller.enemy_sightings.last_seen, {})


if __name__ == "__main__":
    unittest.main()
