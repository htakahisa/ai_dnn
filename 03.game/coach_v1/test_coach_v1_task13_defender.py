"""Task 13 defender curriculum, information-boundary and checkpoint tests."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.evaluate_task13 import rule_actions
from coach_v1.learning_coach_defender import load_defender_coach
from coach_v1.training.coach_environment import DEFENDER_STAGES, CoachTrainingEnvironment
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.defender_imitation import fit_imitation
from coach_v1.training.defender_teacher import teacher_actions
from coach_v1.training.scenario_generator import ScenarioGenerator


class Task13DefenderTest(unittest.TestCase):
    def test_all_stages_run_real_game_and_publish_only_public_objectives(self):
        for stage in DEFENDER_STAGES:
            with self.subTest(stage=stage):
                environment = CoachTrainingEnvironment(
                    Side.DEFENDER, seed=13, stage=stage, max_ticks=1,
                )
                state = environment.reset()
                self.assertEqual(stage == "initial_setup",
                                 environment.game.defender_setup_phase.active)
                self.assertEqual(stage in {"group_up", "ability_retake", "defuse_escort"},
                                 environment.game.is_planted)
                self.assertEqual((27, 26, 44), state.observation.grid.shape)
                transition = environment.step(teacher_actions(state.observation, state.mask))
                self.assertTrue(transition.done)
                if stage == "ability_retake":
                    self.assertEqual(1.0, transition.metrics["defender_ability_preseeded"])
        with self.assertRaises(ValueError):
            CoachTrainingEnvironment(Side.ATTACKER, seed=13, stage="defuse_escort")

    def test_initial_setup_uses_real_setup_ticks_and_one_team_decision(self):
        environment = CoachTrainingEnvironment(
            Side.DEFENDER, seed=13, stage="initial_setup", max_ticks=20,
        )
        state = environment.reset()
        starts = [tuple(character.pos) for character in environment.game.chars
                  if character.team == "D"]
        while state is not None:
            transition = environment.step(teacher_actions(state.observation, state.mask))
            state = transition.next_state
        ends = [tuple(character.pos) for character in environment.game.chars
                if character.team == "D"]
        self.assertEqual(20, environment.ticks)
        self.assertEqual(100, len(environment.controller.action_log))
        self.assertNotEqual(starts, ends)
        self.assertEqual(1.0, transition.metrics["setup_complete"])

    def test_teacher_completes_real_defuse_and_assigns_one_defuser(self):
        environment = CoachTrainingEnvironment(
            Side.DEFENDER, seed=15, stage="defuse_escort", max_ticks=35,
        )
        state = environment.reset()
        successes = 0.0
        while state is not None:
            actions = teacher_actions(state.observation, state.mask)
            self.assertLessEqual(sum(action.objective is ObjectiveAction.DEFUSE
                                     for action in actions), 1)
            transition = environment.step(actions)
            successes += transition.metrics["retake_success"]
            state = transition.next_state
        self.assertEqual(1.0, successes)
        self.assertEqual(1, environment.game.defender_wins)

    def test_hidden_enemy_truth_changes_critic_only_and_not_labels(self):
        for stage in ("full_round", "group_up", "defuse_escort"):
            with self.subTest(stage=stage):
                environment = CoachTrainingEnvironment(
                    Side.DEFENDER, seed=13, stage=stage, max_ticks=1,
                )
                environment.reset()
                enemy = next(character for character in environment.game.chars
                             if character.team == "A")
                visible = np.asarray(environment.sensor.build(
                    game=environment.game, side=Side.DEFENDER).currently_visible)
                occupied = {tuple(character.pos) for character in environment.game.chars
                            if character is not enemy}
                hidden = [(row, column) for row, column in np.argwhere(environment.game.grid != 1)
                          if not visible[row, column] and (row, column) not in occupied]
                self.assertGreaterEqual(len(hidden), 2)
                states = []
                for position in hidden[:2]:
                    enemy.pos = list(position)
                    environment.memory.reset()
                    states.append(environment._observe())
                np.testing.assert_array_equal(states[0].observation.grid,
                                              states[1].observation.grid)
                np.testing.assert_array_equal(states[0].observation.vector,
                                              states[1].observation.vector)
                self.assertFalse(np.array_equal(states[0].critic_enemy_truth,
                                                states[1].critic_enemy_truth))
                self.assertEqual(teacher_actions(states[0].observation, states[0].mask),
                                 teacher_actions(states[1].observation, states[1].mask))
                self.assertEqual(rule_actions(states[0].observation, states[0].mask),
                                 rule_actions(states[1].observation, states[1].mask))

    def test_defender_imitation_and_actor_checkpoint_are_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            trainer = CoachTrainer(Side.DEFENDER, seed=5, directory=path)
            before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
            fit_imitation(trainer, episodes=1, stage="defuse_escort", max_ticks=2)
            self.assertEqual("imitation_defuse_escort", trainer.history[-1]["stage"])
            self.assertTrue(any(not torch.equal(before[name], value)
                                for name, value in trainer.actor.state_dict().items()))
            payload = torch.load(path / "latest.pt", map_location="cpu", weights_only=False)
            self.assertNotIn("critic_state_dict", payload)
            self.assertIsNone(payload["optimizer_state_dict"])
            policy = load_defender_coach(path / "latest.pt")
            self.assertEqual("coach-observation-v2", policy.encoder.version)
            resumed = CoachTrainer(Side.DEFENDER, seed=5, directory=path)
            resumed.resume()
            self.assertEqual(trainer.episode, resumed.episode)

    def test_full_round_enemy_distribution_uses_70_20_10_mix(self):
        environment = CoachTrainingEnvironment(
            Side.DEFENDER, seed=13, stage="full_round", max_ticks=1,
        )
        environment.reset()
        snapshot = environment.sensor.build(game=environment.game, side=Side.DEFENDER)
        generator = ScenarioGenerator(1313)
        counts = Counter(
            enemy.category
            for _ in range(100)
            for enemy in generator.generate(
                actor_side=Side.DEFENDER, situation="search", elapsed_ticks=20,
                enemy_count=5, currently_visible=snapshot.currently_visible,
                occupied_positions=(ally.position for ally in snapshot.allies),
            ).enemies
        )
        self.assertTrue(0.62 <= counts["point"] / 500 <= 0.78)
        self.assertTrue(0.12 <= counts["jitter"] / 500 <= 0.28)
        self.assertTrue(0.04 <= counts["random"] / 500 <= 0.16)


if __name__ == "__main__":
    unittest.main()
