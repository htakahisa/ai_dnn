import dataclasses
import unittest

import numpy as np

from map_data import NEW_MAZE_STR
from coach_v1.common.constants import CHECKPOINTS_DIR, FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.evaluate_task10_rollout import evaluate
from coach_v1.learning_character_base import CharacterPolicy
from coach_v1.observation.character_encoder import CharacterObservationEncoder, CoachInstruction
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.perception.team_perception import EnemySighting, SightingSource
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.kurimaru_rollout import collect_kurimaru_rollout, label_kurimaru


class KurimaruTaskTest(unittest.TestCase):
    def test_five_player_collector_labels_slot_four_and_legal_flash(self):
        for side in (Side.ATTACKER, Side.DEFENDER):
            for encounter in ("near", "near_fixed", "west", "east", "crossfire", "natural"):
                records = collect_kurimaru_rollout(side=side, seed=410,
                                                    encounter=encounter, ticks=2)
                self.assertTrue(records)
                for row in records:
                    example = row.example
                    self.assertEqual(side, row.side)
                    self.assertEqual(encounter, row.encounter)
                    self.assertEqual(1, example.observation.vector[88])
                    self.assertEqual(1, example.observation.vector[84:89].sum())
                    self.assertIn(example.facing, example.acceptable_facings)
                    if example.use_ability:
                        self.assertTrue(row.has_sighting)
                        self.assertTrue(example.observation.mask.ability_target[example.target])
                    else:
                        self.assertIsNone(example.target)
                    if not row.has_sighting:
                        self.assertEqual(tuple(Facing), example.acceptable_facings)

    def test_hidden_enemy_truth_changes_neither_observation_nor_teacher(self):
        grid = np.asarray([[int(cell) for cell in row]
                           for row in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        builder = TeamPerceptionBuilder()
        instruction = CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                                       TacticalIntent.HOLD)
        outputs = []
        for position in ((2, 2), (3, 2)):
            allies = [FakeCharacter(roster.character_name, "A", (23, 18 + slot),
                                    alive=slot == 4, blind=1)
                      for slot, roster in enumerate(FIXED_ROSTER)]
            snapshot = builder.build(game=FakeGame(
                grid, allies + [FakeCharacter("hidden", "D", position)]),
                side=Side.ATTACKER)
            self.assertEqual((), snapshot.sightings)
            belief = BeliefMemory(points.for_side(snapshot.side)).update(snapshot)
            observation = CharacterObservationEncoder().encode(
                snapshot, belief, situation="carry", slot=4, instruction=instruction)
            outputs.append(label_kurimaru(snapshot, observation))
        for field in ("grid", "vector"):
            np.testing.assert_array_equal(getattr(outputs[0].observation, field),
                                          getattr(outputs[1].observation, field))
        np.testing.assert_array_equal(outputs[0].observation.mask.ability_target,
                                      outputs[1].observation.mask.ability_target)
        self.assertEqual((outputs[0].facing, outputs[0].use_ability, outputs[0].target,
                          outputs[0].acceptable_facings),
                         (outputs[1].facing, outputs[1].use_ability, outputs[1].target,
                          outputs[1].acceptable_facings))

    def test_target_uses_shared_report_and_game_mask(self):
        grid = np.asarray([[int(cell) for cell in row]
                           for row in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        allies = [FakeCharacter(roster.character_name, "A", (23, 18 + slot),
                                alive=slot == 4)
                  for slot, roster in enumerate(FIXED_ROSTER)]
        snapshot = TeamPerceptionBuilder().build(
            game=FakeGame(grid, allies), side=Side.ATTACKER)
        points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        belief = BeliefMemory(points.for_side(snapshot.side)).update(snapshot)
        instruction = CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE,
                                       TacticalIntent.HOLD)
        observation = CharacterObservationEncoder().encode(
            snapshot, belief, situation="carry", slot=4, instruction=instruction)
        legal = np.argwhere(observation.mask.ability_target)
        targets = []
        for cell in (legal[0], legal[-1]):
            report = EnemySighting("reported", (int(cell[0]), int(cell[1])),
                                   0, SightingSource.NORMAL)
            example = label_kurimaru(dataclasses.replace(snapshot, sightings=(report,)),
                                  observation)
            self.assertTrue(example.use_ability)
            self.assertTrue(observation.mask.ability_target[example.target])
            targets.append(example.target)
        self.assertNotEqual(*targets)

    def test_watchpoint_distribution_and_checkpoint_slot(self):
        records = build_character_curriculum(4, seed=2004, count=100,
                                             multi_facing=True)
        self.assertIn("point", {row.category for row in records})
        self.assertIn("random", {row.category for row in records})
        self.assertEqual({Side.ATTACKER, Side.DEFENDER}, {row.side for row in records})
        actor = CharacterPolicy(4)
        with self.assertRaises(ValueError):
            CharacterPolicy(3, CHECKPOINTS_DIR / "characters" / "kurimaru" / "best.pt")
        for row in records[:4]:
            action = actor.act(row.example.observation)
            self.assertIsInstance(action.facing, Facing)
            if action.use_ability:
                self.assertTrue(row.example.observation.mask.ability_target[action.target])

    def test_real_game_flash_effects_are_attributed_to_a_slot(self):
        result = evaluate(Side.ATTACKER, ticks=20, seed=0, near=True)
        self.assertGreater(result["flash_enemy_hits"], 0)
        self.assertEqual(result["flash_enemy_hits"],
                         sum(result["flash_enemy_hits_by_slot"].values()))
        self.assertIn("4", result["flash_enemy_hits_by_slot"])


if __name__ == "__main__":
    unittest.main()
