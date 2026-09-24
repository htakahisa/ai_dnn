import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from map_data import NEW_MAZE_STR
from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.learning_character_gongon import GongonPolicy
from coach_v1.learning_character_gonta import GontaPolicy
from coach_v1.learning_character_kunta import KuntaPolicy
from coach_v1.learning_character_kurimaru import KurimaruPolicy
from coach_v1.models import CharacterModelConfig
from coach_v1.observation.character_encoder import CharacterObservationEncoder, CoachInstruction
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.training.character_curriculum import build_character_curriculum
from coach_v1.training.character_trainer import CharacterTrainer


_POLICIES = (GongonPolicy, GontaPolicy, KuntaPolicy, KurimaruPolicy)


class RemainingCharacterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_curriculum_covers_both_sides_point_and_random_and_respects_masks(self):
        for slot in range(5):
            records = build_character_curriculum(slot, seed=2000 + slot, count=80)
            self.assertEqual({"attacker", "defender"}, {item.side.value for item in records})
            self.assertIn("point", {item.category for item in records})
            self.assertIn("random", {item.category for item in records})
            for item in records:
                example = item.example
                self.assertEqual(int(example.use_ability), bool(example.target is not None))
                self.assertTrue(example.observation.mask.ability_use[int(example.use_ability)])
                if example.target is not None:
                    self.assertTrue(example.observation.mask.ability_target[example.target])
                if slot == 1:
                    self.assertFalse(example.use_ability)
                    self.assertFalse(example.observation.mask.ability_use[1])

    def test_independent_checkpoint_loads_only_matching_character(self):
        config = CharacterModelConfig(hidden_channels=4, hidden_features=8)
        for slot, policy_type in enumerate(_POLICIES, start=1):
            records = build_character_curriculum(slot, seed=60 + slot, count=8)
            examples = [item.example for item in records]
            with tempfile.TemporaryDirectory() as directory:
                trainer = CharacterTrainer(slot, seed=40 + slot, config=config,
                                           directory=Path(directory))
                trainer.fit(examples, examples, epochs=1, batch_size=4)
                policy = policy_type(Path(directory) / "best.pt")
                action = policy.act(examples[0].observation)
                self.assertFalse(hasattr(action, "movement"))
                self.assertTrue(examples[0].observation.mask.facing.any())
                with self.assertRaises(ValueError):
                    _POLICIES[(slot % 4)](Path(directory) / "best.pt")
                with self.assertRaises(ValueError):
                    policy.act(dataclasses.replace(examples[0].observation, map_hash="0" * 64))
                other_slot = (slot + 1) % 5
                foreign = build_character_curriculum(other_slot, seed=60, count=1)[0].example.observation
                with self.assertRaises(ValueError):
                    policy.act(foreign)

    def test_unseen_enemy_position_does_not_change_any_character_observation(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        grid = np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        points = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        builder = TeamPerceptionBuilder()
        encoder = CharacterObservationEncoder()
        instruction = CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE, TacticalIntent.HOLD)
        for slot in range(5):
            observations = []
            for enemy_position in ((2, 2), (3, 2)):
                allies = [FakeCharacter(roster.character_name, "A", (23, 18 + i),
                                        alive=i == slot, blind=1)
                          for i, roster in enumerate(FIXED_ROSTER)]
                game = FakeGame(grid, allies + [FakeCharacter("hidden", "D", enemy_position)])
                snapshot = builder.build(game=game, side="attacker")
                self.assertEqual((), snapshot.sightings)
                belief = BeliefMemory(points.for_side(snapshot.side)).update(snapshot)
                observations.append(encoder.encode(snapshot, belief, situation="carry",
                                                   slot=slot, instruction=instruction))
            np.testing.assert_array_equal(observations[0].grid, observations[1].grid)
            np.testing.assert_array_equal(observations[0].vector, observations[1].vector)
            np.testing.assert_array_equal(observations[0].mask.ability_target,
                                          observations[1].mask.ability_target)


if __name__ == "__main__":
    unittest.main()
