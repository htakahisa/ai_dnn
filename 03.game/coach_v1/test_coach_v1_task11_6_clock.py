"""Public timer and checkpoint compatibility tests for Task 11.6."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from game_core import ROUND_DURATION_TICKS, SPIKE_DETONATION_TICKS
from map_data import NEW_MAZE_STR
import numpy as np

from coach_v1.common.checkpoint import build_checkpoint_metadata, build_checkpoint_payload
from coach_v1.common.types import ModelTarget, Side
from coach_v1.common.versions import COACH_OBSERVATION_VERSION, LEGACY_COACH_OBSERVATION_VERSION
from coach_v1.coordinator import TeamExecutionCoordinator
from coach_v1.evaluate_task11 import evaluate
from coach_v1.learning_coach import load_coach_policy
from coach_v1.models.coach_model import CoachActorModel, CoachModelConfig
from coach_v1.observation.character_encoder import CharacterObservationEncoder
from coach_v1.observation.coach_encoder import COACH_VECTOR_FIELDS, CoachObservationEncoder
from coach_v1.training.coach_environment import CoachTrainingEnvironment, _STAY
from coach_v1.training.coach_trainer import CoachTrainer


class PublicClockTest(unittest.TestCase):
    def test_postplant_unseen_enemy_relocation_changes_only_critic(self):
        rows = NEW_MAZE_STR.strip().splitlines()
        for side in Side:
            with self.subTest(side=side):
                environment = CoachTrainingEnvironment(side, seed=8, max_ticks=1)
                environment.reset()
                game = environment.game
                game.is_planted = True
                game.planted_pos = tuple(game.target_plant_pos)
                game.detonate_timer = 12
                for character in game.chars:
                    character.has_spike = False
                enemy_code = "D" if side is Side.ATTACKER else "A"
                enemy = next(c for c in game.chars if c.team == enemy_code and c.is_alive)
                allied = {tuple(c.pos) for c in game.chars
                          if c.team != enemy_code and c.is_alive}
                states = []
                for row, cells in enumerate(rows):
                    for column, cell in enumerate(cells):
                        if cell == "1" or (row, column) in allied:
                            continue
                        enemy.pos = [row, column]
                        environment.memory.reset()
                        environment.controller.reset_round()
                        if environment.sensor.build(game=game, side=side).sightings:
                            continue
                        states.append(environment._observe())
                        if len(states) == 2:
                            break
                    if len(states) == 2:
                        break
                self.assertEqual(2, len(states))
                np.testing.assert_array_equal(states[0].observation.grid,
                                              states[1].observation.grid)
                np.testing.assert_array_equal(states[0].observation.vector,
                                              states[1].observation.vector)
                self.assertFalse(np.array_equal(states[0].critic_enemy_truth,
                                                states[1].critic_enemy_truth))

    def test_public_active_countdown_changes_coach_v2_but_not_character_v1(self):
        for side in Side:
            with self.subTest(side=side):
                environment = CoachTrainingEnvironment(side, seed=8, max_ticks=1)
                environment.reset()
                game = environment.game
                round_index = COACH_VECTOR_FIELDS.index("round_time_remaining")
                detonation_index = COACH_VECTOR_FIELDS.index("detonation_time_remaining")

                game.round_timer = 20
                environment.memory.reset()
                preplant_snapshot = environment.sensor.build(game=game, side=side)
                preplant_belief = environment.memory.update(preplant_snapshot)
                preplant = environment.encoder.encode(
                    preplant_snapshot, preplant_belief,
                    situation="carry" if side is Side.ATTACKER else "search",
                )
                self.assertEqual(COACH_OBSERVATION_VERSION, preplant.version)
                self.assertEqual((86,), preplant.vector.shape)
                self.assertAlmostEqual(20 / ROUND_DURATION_TICKS,
                                       float(preplant.vector[round_index]))
                self.assertEqual(0, preplant.vector[detonation_index])

                game.is_planted = True
                for character in game.chars:
                    character.has_spike = False
                game.planted_pos = tuple(next(c.pos for c in game.chars if c.is_alive))
                game.detonate_timer = 12
                environment.memory.reset()
                postplant_snapshot = environment.sensor.build(game=game, side=side)
                postplant_belief = environment.memory.update(postplant_snapshot)
                situation = "guard" if side is Side.ATTACKER else "retake"
                postplant = environment.encoder.encode(
                    postplant_snapshot, postplant_belief, situation=situation,
                )
                self.assertEqual(0, postplant.vector[round_index])
                self.assertAlmostEqual(12 / SPIKE_DETONATION_TICKS,
                                       float(postplant.vector[detonation_index]))
                character = CharacterObservationEncoder()
                original = character.encode(
                    postplant_snapshot, postplant_belief, situation=situation,
                    slot=0, instruction=_STAY[0],
                )
                self.assertEqual((106,), original.vector.shape)

    def test_legacy_coach_checkpoint_uses_v1_encoder_and_new_training_uses_v2(self):
        config = CoachModelConfig(vector_features=84)
        encoder = CoachObservationEncoder(version=LEGACY_COACH_OBSERVATION_VERSION)
        metadata = build_checkpoint_metadata(
            target=ModelTarget.coach(Side.ATTACKER),
            map_hash=encoder.map_hash,
            watch_points_hash=encoder.watch_points_hash,
            model_config=config.to_dict(), training_seed=11, training_step=1,
        )
        metadata = replace(metadata, observation_version=LEGACY_COACH_OBSERVATION_VERSION)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "latest.pt"
            torch.save(build_checkpoint_payload(
                metadata=metadata,
                model_state_dict=CoachActorModel(config).state_dict(),
                optimizer_state_dict=None,
            ), path)
            policy = load_coach_policy(Side.ATTACKER, path)
            self.assertEqual(LEGACY_COACH_OBSERVATION_VERSION, policy.encoder.version)
            environment = CoachTrainingEnvironment(Side.ATTACKER, seed=8, max_ticks=1)
            environment.reset()
            coordinator = TeamExecutionCoordinator(Side.ATTACKER, policy,
                                                   environment.characters)
            self.assertEqual(LEGACY_COACH_OBSERVATION_VERSION,
                             coordinator.encoder.version)
            snapshot = environment.sensor.build(game=environment.game, side=Side.ATTACKER)
            belief = environment.memory.update(snapshot)
            observation = policy.encoder.encode(snapshot, belief, situation="carry")
            self.assertEqual((84,), observation.vector.shape)
            self.assertEqual(5, len(policy.act(observation)))
            self.assertEqual(COACH_OBSERVATION_VERSION, environment.encoder.version)
            self.assertEqual((86,), environment.state.observation.vector.shape)
            report = evaluate(Side.ATTACKER, stage="2v1", seeds=range(1),
                              max_ticks=1, checkpoint=path)
            self.assertEqual(1.0, report["results"]["trained"]["ticks"])
            original_bytes = path.read_bytes()
            trainer = CoachTrainer(Side.ATTACKER, seed=11, directory=Path(root))
            with self.assertRaisesRegex(ValueError, "different observation version"):
                trainer.save()
            self.assertEqual(original_bytes, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
