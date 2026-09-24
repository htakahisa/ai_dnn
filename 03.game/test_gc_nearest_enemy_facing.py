"""GC training-only nearest-enemy pre-aim supervision."""

import unittest
from types import SimpleNamespace

import numpy as np

from gc_v1.gc_facing import FACING_DIRS, nearest_alive_enemy_facing
from gc_v1 import train_defender_retake_gc as retake
from gc_v1 import train_defender_search_gc as search


def unit(name, team, pos, alive=True, facing="S"):
    return SimpleNamespace(
        name=name,
        team=team,
        pos=pos,
        is_alive=alive,
        facing=facing,
    )


class NearestEnemyFacingTests(unittest.TestCase):
    def test_shared_teacher_uses_chebyshev_distance_and_ignores_dead_enemy(self):
        actor = unit("defender", "D", (4, 4))
        diagonal_near = unit("near", "A", (2, 2))
        cardinal_far = unit("far", "A", (4, 7))
        dead_adjacent = unit("dead", "A", (4, 3), alive=False)

        self.assertEqual(
            nearest_alive_enemy_facing(
                actor, [cardinal_far, dead_adjacent, diagonal_near]
            ),
            "NW",
        )

    def test_search_teacher_prefers_unseen_nearest_enemy_over_observation_fallback(self):
        actor = unit("defender", "D", (4, 4))
        enemy = unit("attacker", "A", (4, 6))
        obs = np.zeros(search.OBS_DIM, dtype=np.float32)
        # A stale observation points west; privileged training state points east.
        obs[17], obs[18], obs[19] = 1.0, 0.0, -1.0

        target, confidence = search.observable_facing_target(
            obs, 0, actor, [enemy]
        )

        self.assertEqual(FACING_DIRS[target], "E")
        self.assertEqual(confidence, 1.0)

    def test_retake_teacher_does_not_require_line_of_sight(self):
        actor = unit("defender", "D", (4, 4))
        enemy = unit("attacker", "A", (2, 4))
        env = SimpleNamespace(
            attackers=lambda: [enemy],
            check_line_of_sight=lambda *_args: False,
            planted_pos=(8, 8),
        )

        target, confidence = retake.observable_facing_target(env, actor, 0)

        self.assertEqual(FACING_DIRS[target], "N")
        self.assertEqual(confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
