import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from map_data import NEW_MAZE_STR
from coach_v1.common.constants import FIXED_ROSTER, WATCH_POINTS_CONFIG_PATH
from coach_v1.common.types import Facing, MovementAction, ObjectiveAction, TacticalIntent
from coach_v1.common.watch_points import load_watch_points
from coach_v1.learning_character_gorimaru import GorimaruPolicy
from coach_v1.models import CharacterModel, CharacterModelConfig, select_character_action
from coach_v1.observation.character_encoder import CharacterObservationEncoder, CoachInstruction
from coach_v1.perception import BeliefMemory, TeamPerceptionBuilder
from coach_v1.test_coach_v1_task03_team_perception import FakeCharacter, FakeGame
from coach_v1.test_coach_v1_task05_coach_encoder import _memory, _snapshot
from coach_v1.train_character_gorimaru import train_gorimaru
from coach_v1.training.character_environment import CharacterEnvironment
from coach_v1.training.character_trainer import CharacterTrainer, CharacterTrainingExample
from coach_v1.training.scenario_generator import ScenarioGenerator


class CharacterTrainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def _examples(self):
        snapshot = _snapshot()
        belief = _memory(snapshot.side).update(snapshot)
        encoder = CharacterObservationEncoder()
        examples = []
        for movement, facing in ((MovementAction.MOVE_E, Facing.E), (MovementAction.MOVE_W, Facing.W)):
            for intent, use in ((TacticalIntent.CLEAR_AREA, False), (TacticalIntent.UTILITY_REQUEST, True)):
                obs = encoder.encode(snapshot, belief, situation="carry", slot=0,
                                     instruction=CoachInstruction(movement, ObjectiveAction.NONE, intent))
                examples.append(CharacterTrainingExample(obs, facing, use, (23, 18) if use else None,
                                                         True if use else None))
        return examples

    def test_training_validation_checkpoint_resume_and_actor_agree(self):
        examples = self._examples()
        train = examples * 12
        validation = examples * 3
        config = CharacterModelConfig(hidden_channels=8, hidden_features=64)
        with tempfile.TemporaryDirectory() as directory:
            baseline = CharacterTrainer(0, seed=17, config=config, learning_rate=0.003,
                                        directory=Path(directory))
            before = baseline.evaluate(validation)
            trainer = train_gorimaru(train, validation, seed=17, epochs=20, batch_size=8,
                                     directory=Path(directory), config=config, learning_rate=0.003)
            after = trainer.evaluate(validation)
            self.assertLess(after.loss, before.loss)
            self.assertGreater(after.facing_accuracy, 0.75)
            self.assertGreater(after.ability_accuracy, 0.75)
            self.assertGreater(after.unused_ability_accuracy, 0.75)
            self.assertEqual(1.0, after.ability_effective_rate)
            self.assertEqual(20, trainer.epoch)
            self.assertEqual(20, len(trainer.history))
            self.assertTrue((Path(directory) / "latest.pt").exists())
            self.assertTrue((Path(directory) / "best.pt").exists())

            policy = GorimaruPolicy(Path(directory) / "best.pt")
            action = policy.act(examples[0].observation)
            self.assertIsInstance(action.facing, Facing)
            self.assertFalse(hasattr(action, "movement"))
            with self.assertRaises(ValueError):
                policy.act(dataclasses.replace(examples[0].observation, map_hash="0" * 64))
            step = CharacterEnvironment().prepare(_snapshot(), _memory(_snapshot().side).update(_snapshot()),
                                                  situation="carry", slot=0,
                                                  instruction=CoachInstruction(MovementAction.MOVE_E,
                                                                               ObjectiveAction.NONE,
                                                                               TacticalIntent.CLEAR_AREA))
            CharacterEnvironment().resolve(step, action)
            resumed = CharacterTrainer(0, seed=17, config=config, learning_rate=0.003,
                                       directory=Path(directory))
            resumed.resume()
            self.assertEqual(trainer.training_step, resumed.training_step)
            resumed.fit(train, validation, epochs=1, batch_size=8)
            self.assertEqual(21, resumed.epoch)
            self.assertEqual(21, len(resumed.history))

    def test_wrong_character_and_map_checkpoint_rejected(self):
        examples = self._examples()
        config = CharacterModelConfig(hidden_channels=4, hidden_features=8)
        with tempfile.TemporaryDirectory() as directory:
            trainer = CharacterTrainer(0, seed=1, config=config, directory=Path(directory))
            trainer.fit(examples, examples, epochs=1, batch_size=4)
            with self.assertRaises(ValueError):
                CharacterTrainer(1, seed=1, config=config, directory=Path(directory)).resume()
            payload = torch.load(Path(directory) / "latest.pt", weights_only=False)
            payload["metadata"]["map_hash"] = "0" * 64
            bad = Path(directory) / "bad.pt"
            torch.save(payload, bad)
            with self.assertRaises(ValueError):
                GorimaruPolicy(bad)

    def test_masks_labels_and_hidden_position_invariance(self):
        examples = self._examples()
        rows = NEW_MAZE_STR.strip().splitlines()
        grid = np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)
        allies = [FakeCharacter(roster.character_name, "A", (23, 18 + slot),
                                alive=slot == 0) for slot, roster in enumerate(FIXED_ROSTER)]
        builder = TeamPerceptionBuilder()
        baseline = builder.build(game=FakeGame(grid, allies), side=_snapshot().side)
        generator = ScenarioGenerator(13)
        positions = [generator.generate(
            actor_side="attacker", situation="carry", elapsed_ticks=100, enemy_count=1,
            currently_visible=baseline.currently_visible,
            occupied_positions=(ally.position for ally in baseline.allies),
        ).enemies[0].position for _ in range(2)]
        self.assertNotEqual(*positions)
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        observations = []
        for position in positions:
            snapshot = builder.build(game=FakeGame(grid, allies + [FakeCharacter("hidden", "D", position)]),
                                     side=baseline.side)
            self.assertEqual((), snapshot.sightings)
            belief = BeliefMemory(config.for_side(snapshot.side)).update(snapshot)
            observations.append(CharacterObservationEncoder().encode(
                snapshot, belief, situation="carry", slot=0,
                instruction=CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE, TacticalIntent.HOLD)))
        first, second = observations
        np.testing.assert_array_equal(first.grid, second.grid)
        np.testing.assert_array_equal(first.vector, second.vector)
        np.testing.assert_array_equal(first.mask.ability_target, second.mask.ability_target)
        model = CharacterModel(CharacterModelConfig(hidden_channels=4, hidden_features=8))
        self.assertEqual(select_character_action(model, first), select_character_action(model, second))
        invalid = dataclasses.replace(examples[0], use_ability=True, target=(0, 0))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            CharacterTrainer(0, seed=1, directory=Path(directory)).fit([invalid], examples, epochs=1)
        other_slot = CharacterObservationEncoder().encode(
            snapshot, belief, situation="carry", slot=1,
            instruction=CoachInstruction(MovementAction.STAY, ObjectiveAction.NONE, TacticalIntent.HOLD))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            CharacterTrainer(0, seed=1, directory=Path(directory)).fit(
                [dataclasses.replace(examples[0], observation=other_slot)], examples, epochs=1)


if __name__ == "__main__":
    unittest.main()
