import dataclasses
import unittest

import numpy as np

from map_data import NEW_MAZE_STR
from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.learning_character_gongon import GongonPolicy
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.common.constants import CHARACTER_CHECKPOINT_PATHS
from coach_v1.observation.character_encoder import CharacterInputError
from coach_v1.observation.character_encoder import CharacterObservationEncoder, CoachInstruction
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.facing_labels import acceptable_facings_from_snapshot
from coach_v1.training.gongon_rollout import collect_gongon_rollout


class GongonTaskTest(unittest.TestCase):
    def test_collector_uses_slot_one_and_never_requests_hunt(self):
        for encounter in ("near", "near_fixed", "west", "east", "crossfire", "natural"):
            records = collect_gongon_rollout(side=Side.ATTACKER, seed=410,
                                             encounter=encounter, ticks=2)
            self.assertTrue(records)
            for record in records:
                example = record.example
                self.assertEqual(encounter, record.encounter)
                self.assertEqual(1, example.observation.vector[85])
                self.assertEqual(1, example.observation.vector[84:89].sum())
                self.assertFalse(example.use_ability)
                self.assertFalse(example.observation.mask.ability_use[1])
                self.assertIn(example.facing, example.acceptable_facings)
                if not record.has_sighting:
                    self.assertEqual(tuple(Facing), example.acceptable_facings)

    def test_hidden_enemy_move_changes_neither_actor_observation_nor_label(self):
        grid = np.asarray([[int(cell) for cell in row]
                           for row in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        builder = TeamPerceptionBuilder()
        encoder = CharacterObservationEncoder()
        instruction = CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                                       TacticalIntent.HOLD)
        outputs = []
        for position in ((2, 2), (3, 2)):
            allies = [FakeCharacter(roster.character_name, "A", (23, 18 + slot),
                                    alive=slot == 1, blind=1)
                      for slot, roster in enumerate(FIXED_ROSTER)]
            snapshot = builder.build(game=FakeGame(
                grid, allies + [FakeCharacter("hidden", "D", position)]),
                side=Side.ATTACKER)
            self.assertEqual((), snapshot.sightings)
            belief = BeliefMemory(points.for_side(snapshot.side)).update(snapshot)
            observation = encoder.encode(snapshot, belief, situation="carry", slot=1,
                                         instruction=instruction)
            outputs.append((observation, tuple(Facing) if not snapshot.sightings
                            else acceptable_facings_from_snapshot(snapshot, slot=1)))
        np.testing.assert_array_equal(outputs[0][0].grid, outputs[1][0].grid)
        np.testing.assert_array_equal(outputs[0][0].vector, outputs[1][0].vector)
        np.testing.assert_array_equal(outputs[0][0].mask.ability_target,
                                      outputs[1][0].mask.ability_target)
        self.assertEqual(outputs[0][1], outputs[1][1])

    def test_checkpoint_and_watchpoint_random_coverage(self):
        records = build_character_curriculum(1, seed=2001, count=100,
                                             multi_facing=True)
        self.assertIn("point", {row.category for row in records})
        self.assertIn("random", {row.category for row in records})
        self.assertEqual({Side.ATTACKER, Side.DEFENDER}, {row.side for row in records})
        policy = GongonPolicy()
        for row in records[:5]:
            self.assertFalse(policy.act(row.example.observation).use_ability)

    def test_default_policy_routes_only_by_legal_side_feature(self):
        records = build_character_curriculum(1, seed=2001, count=4)
        policy = GongonPolicy()
        attacker = CharacterPolicy(1, CHARACTER_CHECKPOINT_PATHS["gongon"] / "best.pt")
        defender = CharacterPolicy(1, CHARACTER_CHECKPOINT_PATHS["gongon"] / "defender_best.pt")
        for row in records:
            observation = row.example.observation
            expected = attacker if row.side is Side.ATTACKER else defender
            self.assertEqual(expected.act(observation), policy.act(observation))
        bad_vector = records[0].example.observation.vector.copy()
        bad_vector[0:2] = 0
        with self.assertRaises(CharacterInputError):
            policy.act(dataclasses.replace(records[0].example.observation,
                                           vector=bad_vector))


if __name__ == "__main__":
    unittest.main()
