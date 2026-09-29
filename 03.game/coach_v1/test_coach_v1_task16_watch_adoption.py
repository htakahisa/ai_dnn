"""Versioned watch-point runtime and checkpoint compatibility checks."""

import unittest

from coach_v1.common.watch_point_versions import (
    LEGACY_WATCH_POINTS_HASH, TASK16_WATCH_POINTS_HASH,
)
from coach_v1.opponent_pool import OpponentKind, load_opponent_pool
from coach_v1.common.types import Side
from coach_v1.task15_self_play import build_opponent_team
from coach_v1.team_ai import build_coach_v1_team
from coach_v1.training.scenario_generator import ScenarioGenerator


class AdoptedWatchPointsTest(unittest.TestCase):
    def test_both_adopted_points_enter_training_sampler(self):
        visible = tuple((False,) * 44 for _ in range(26))
        occupied = tuple((23, 18 + slot) for slot in range(5))
        for side, situation, point_id in (
            (Side.ATTACKER, "carry", "watch_r11_c31"),
            (Side.DEFENDER, "search", "watch_r07_c34"),
        ):
            with self.subTest(point=point_id):
                generator = ScenarioGenerator(1600)
                selected = {
                    generator.generate(
                        actor_side=side, situation=situation, elapsed_ticks=100,
                        enemy_count=1, currently_visible=visible,
                        occupied_positions=occupied,
                    ).enemies[0].point_id
                    for _ in range(500)
                }
                self.assertIn(point_id, selected)

    def test_current_team_uses_all_new_watch_points_on_both_sides(self):
        team = build_coach_v1_team()
        for controller in (team.get_attacker_controller(),
                           team.get_defender_controller()):
            self.assertEqual(TASK16_WATCH_POINTS_HASH,
                             controller.encoder.watch_points_hash)
            self.assertEqual(36, len(controller.memory._watch_points))
            self.assertTrue(all(actor.encoder.watch_points_hash
                                == TASK16_WATCH_POINTS_HASH
                                for actor in controller.characters.values()))

    def test_historical_opponent_keeps_old_config_and_characters(self):
        spec = next(item for item in load_opponent_pool().opponents
                    if item.kind is OpponentKind.HISTORICAL_COACH)
        team = build_opponent_team(spec)
        for controller in (team.get_attacker_controller(),
                           team.get_defender_controller()):
            self.assertEqual(LEGACY_WATCH_POINTS_HASH,
                             controller.encoder.watch_points_hash)
            self.assertEqual(35, len(controller.memory._watch_points))
            self.assertTrue(all(actor.encoder.watch_points_hash
                                == LEGACY_WATCH_POINTS_HASH
                                for actor in controller.characters.values()))


if __name__ == "__main__":
    unittest.main()
