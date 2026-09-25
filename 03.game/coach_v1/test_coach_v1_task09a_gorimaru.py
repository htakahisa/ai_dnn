import dataclasses
import unittest

import numpy as np

from map_data import NEW_MAZE_STR
from coach_v1.common.constants import FIXED_ROSTER
from coach_v1.common.types import Facing, Side
from coach_v1.perception.team_perception import EnemySighting, SightingSource, TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_trainer import CharacterTrainer
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.gorimaru_rollout import (
    acceptable_facings_from_snapshot, collect_gorimaru_rollout, label_facing_from_snapshot,
)


class GorimaruRolloutTest(unittest.TestCase):
    def test_hidden_enemy_truth_does_not_change_training_facing_label(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        grid = np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        builder = TeamPerceptionBuilder()
        labels = []
        for hidden_position in ((2, 2), (3, 2)):
            allies = [FakeCharacter(slot.character_name, "A", (23, 18 + i),
                                    facing="E", alive=i == 0, blind=1)
                      for i, slot in enumerate(FIXED_ROSTER)]
            snapshot = builder.build(
                game=FakeGame(grid, allies + [FakeCharacter("hidden", "D", hidden_position)]),
                side=Side.ATTACKER,
            )
            self.assertEqual((), snapshot.sightings)
            labels.append((label_facing_from_snapshot(snapshot),
                           acceptable_facings_from_snapshot(snapshot)))
        self.assertEqual([(Facing.E, (Facing.E,))] * 2, labels)

    def test_shared_reports_choose_one_legal_nearby_direction(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        grid = np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        allies = [FakeCharacter(slot.character_name, "A", (23, 18 + i), alive=i == 0)
                  for i, slot in enumerate(FIXED_ROSTER)]
        snapshot = TeamPerceptionBuilder().build(game=FakeGame(grid, allies), side="attacker")
        with_reports = dataclasses.replace(snapshot, sightings=(
            EnemySighting("far", (2, 2), 0, SightingSource.NORMAL),
            EnemySighting("near", (22, 18), 1, SightingSource.NORMAL),
        ))
        self.assertEqual(Facing.N, label_facing_from_snapshot(with_reports))

    def test_two_opposed_shared_sightings_accept_both_directions(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        grid = np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        allies = [FakeCharacter(slot.character_name, "A", (22, 18 + i), alive=i == 0)
                  for i, slot in enumerate(FIXED_ROSTER)]
        snapshot = TeamPerceptionBuilder().build(game=FakeGame(grid, allies), side="attacker")
        with_reports = dataclasses.replace(snapshot, sightings=(
            EnemySighting("north", (20, 18), 1, SightingSource.NORMAL),
            EnemySighting("south", (23, 18), 2, SightingSource.NORMAL),
        ))
        accepted = acceptable_facings_from_snapshot(with_reports)
        self.assertIn(Facing.N, accepted)
        self.assertIn(Facing.S, accepted)
        self.assertNotIn(Facing.E, accepted)

    def test_real_game_collector_returns_only_actor_safe_examples(self):
        records = collect_gorimaru_rollout(side=Side.DEFENDER, seed=100, ticks=4)
        self.assertGreater(len(records), 0)
        for item in records:
            observation = item.example.observation
            self.assertEqual((29, 26, 44), observation.grid.shape)
            self.assertFalse(hasattr(observation, "game"))
            self.assertFalse(hasattr(observation, "enemy_position"))
            self.assertTrue(observation.mask.facing[tuple(Facing).index(item.example.facing)])

    def test_multi_facing_loss_accepts_any_reported_direction(self):
        record = collect_gorimaru_rollout(side=Side.DEFENDER, seed=100, ticks=1)[0]
        trainer = CharacterTrainer(0, seed=123)
        single = dataclasses.replace(record.example, facing=Facing.N,
                                     acceptable_facings=None)
        multiple = dataclasses.replace(single, acceptable_facings=tuple(Facing))
        self.assertLess(trainer.evaluate([multiple]).loss, trainer.evaluate([single]).loss)
        with self.assertRaisesRegex(ValueError, "acceptable facing"):
            trainer.evaluate([dataclasses.replace(single,
                                                 acceptable_facings=(Facing.E, Facing.E))])

    def test_varied_encounter_collector_uses_safe_labels(self):
        for encounter in ("west", "east", "crossfire"):
            records = collect_gorimaru_rollout(
                side=Side.ATTACKER, seed=410, ticks=2, encounter=encounter,
            )
            self.assertTrue(records)
            self.assertTrue(all(item.encounter == encounter for item in records))
            self.assertTrue(all(item.example.facing in item.example.acceptable_facings
                                for item in records))

    def test_rollout_without_sighting_has_no_arbitrary_facing_target(self):
        records = collect_gorimaru_rollout(
            side=Side.DEFENDER, seed=410, ticks=3, encounter="natural",
            mask_unsighted_facing=True,
        )
        unseen = [item for item in records if not item.has_sighting]
        self.assertTrue(unseen)
        self.assertTrue(all(item.example.acceptable_facings == tuple(Facing)
                            for item in unseen))
        legacy = collect_gorimaru_rollout(
            side=Side.DEFENDER, seed=410, ticks=3, encounter="natural",
        )
        self.assertTrue(all(item.example.acceptable_facings == (item.example.facing,)
                            for item in legacy if not item.has_sighting))

    def test_watchpoint_curriculum_can_opt_in_to_multiple_facing_labels(self):
        standard = build_character_curriculum(0, seed=501, count=12)
        multiple = build_character_curriculum(0, seed=501, count=12,
                                              multi_facing=True)
        self.assertEqual([item.category for item in standard],
                         [item.category for item in multiple])
        self.assertTrue(all(item.example.acceptable_facings is None for item in standard))
        self.assertTrue(all(item.example.facing in item.example.acceptable_facings
                            for item in multiple))


if __name__ == "__main__":
    unittest.main()
