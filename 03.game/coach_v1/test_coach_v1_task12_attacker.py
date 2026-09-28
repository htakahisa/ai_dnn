"""Attacker curriculum, actor information boundary, and paired entry points."""

from pathlib import Path
from collections import Counter
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from coach_v1.common.types import ObjectiveAction, Side
from coach_v1.evaluate_task12 import rule_actions
from coach_v1.learning_coach_attacker import load_attacker_coach
from coach_v1.training.coach_environment import (
    ATTACKER_STAGES, CoachTrainingEnvironment, _STAY, _has_multiple_enemy_angles,
)
from coach_v1.training.attacker_teacher import teacher_actions
from coach_v1.training.attacker_imitation import fit_imitation
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.training.scenario_generator import ScenarioGenerator


class Task12AttackerTest(unittest.TestCase):
    def test_all_stages_execute_and_objective_state_is_public(self):
        for stage in ATTACKER_STAGES:
            with self.subTest(stage=stage):
                env = CoachTrainingEnvironment(Side.ATTACKER, seed=11,
                                               stage=stage, max_ticks=1)
                state = env.reset()
                self.assertEqual(5, sum(c.is_alive for c in env.game.chars if c.team == "A"))
                self.assertEqual(5, sum(c.is_alive for c in env.game.chars if c.team == "D"))
                self.assertEqual(stage == "retrieve", env.game.spike_pos is not None)
                self.assertEqual(stage == "post_plant", env.game.is_planted)
                self.assertEqual(stage == "utility_entry",
                                 bool(np.any(state.observation.grid[15] > 0)))
                self.assertEqual(100 if stage == "full_round" else 100 - env.game.battle_tick,
                                 env.game.round_timer)
                if stage == "full_round":
                    self.assertEqual(20, env.scenario.elapsed_ticks)
                    self.assertIn("point", [enemy.category for enemy in env.scenario.enemies])
                self.assertEqual((27, 26, 44), state.observation.grid.shape)
                self.assertTrue(env.step(_STAY).done)
        with self.assertRaises(ValueError):
            CoachTrainingEnvironment(Side.DEFENDER, seed=11, stage="plant")

    def test_plant_mask_and_simple_rule_use_only_actor_observation(self):
        env = CoachTrainingEnvironment(Side.ATTACKER, seed=11, stage="plant", max_ticks=1)
        state = env.reset()
        carrier = next(i for i, c in enumerate(env.game.chars[:5]) if c.has_spike)
        actions = rule_actions(state.observation, state.mask)
        self.assertEqual(ObjectiveAction.PLANT, actions[carrier].objective)
        for slot, action in enumerate(actions):
            self.assertTrue(state.mask.movement[slot, tuple(type(action.movement)).index(action.movement)])
            self.assertTrue(state.mask.objective[slot, tuple(ObjectiveAction).index(action.objective)])

    def test_hidden_truth_changes_critic_only_in_full_round_and_post_plant(self):
        for stage in ("full_round", "utility_entry", "post_plant"):
            with self.subTest(stage=stage):
                env = CoachTrainingEnvironment(Side.ATTACKER, seed=11,
                                               stage=stage, max_ticks=1)
                env.reset()
                enemy = next(c for c in env.game.chars if c.team == "D")
                visible = np.asarray(env.sensor.build(game=env.game,
                                                       side=Side.ATTACKER).currently_visible)
                occupied = {tuple(c.pos) for c in env.game.chars if c is not enemy}
                hidden = [(r, c) for r, c in np.argwhere(env.game.grid != 1)
                          if not visible[r, c] and (r, c) not in occupied]
                self.assertGreaterEqual(len(hidden), 2)
                states = []
                for position in hidden[:2]:
                    enemy.pos = list(position)
                    env.memory.reset()
                    states.append(env._observe())
                np.testing.assert_array_equal(states[0].observation.grid,
                                              states[1].observation.grid)
                np.testing.assert_array_equal(states[0].observation.vector,
                                              states[1].observation.vector)
                self.assertFalse(np.array_equal(states[0].critic_enemy_truth,
                                                states[1].critic_enemy_truth))
                self.assertEqual(teacher_actions(states[0].observation, states[0].mask),
                                 teacher_actions(states[1].observation, states[1].mask))

    def test_attacker_training_and_actor_checkpoint_are_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = CoachTrainer(Side.ATTACKER, seed=7, directory=Path(temp))
            history = trainer.fit(episodes=1, stage="plant", max_ticks=2,
                                  save_rollouts=False)
            self.assertEqual("plant", history[-1]["stage"])
            payload = torch.load(Path(temp) / "latest.pt", map_location="cpu",
                                 weights_only=False)
            self.assertNotIn("critic_state_dict", payload)
            self.assertIsNone(payload["optimizer_state_dict"])
            policy = load_attacker_coach(Path(temp) / "latest.pt")
            self.assertEqual("coach-observation-v2", policy.encoder.version)
            trainer2 = CoachTrainer(Side.ATTACKER, seed=7, directory=Path(temp))
            trainer2.resume()
            self.assertEqual(trainer.episode, trainer2.episode)

    def test_ally_lost_before_its_turn_does_not_break_rollout(self):
        env = CoachTrainingEnvironment(Side.ATTACKER, seed=11,
                                       stage="full_round", max_ticks=1)
        env.reset()
        acting = [c for c in env._pending_moves if c.team == "A"]
        first, last = acting[0], acting[-1]
        original_move = env.game.move_character

        def lose_later_ally(character):
            original_move(character)
            if character is first:
                last.is_alive = False
                last.hp = 0

        env.game.move_character = lose_later_ally
        transition = env.step(_STAY)
        self.assertTrue(transition.done)
        self.assertEqual(4, len(env.controller.action_log))

    def test_imitation_updates_only_attacker_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = CoachTrainer(Side.ATTACKER, seed=5, directory=Path(temp))
            before = {name: value.clone() for name, value in trainer.actor.state_dict().items()}
            fit_imitation(trainer, episodes=1, max_ticks=2)
            self.assertEqual("imitation_full_round", trainer.history[-1]["stage"])
            fit_imitation(trainer, episodes=1, max_ticks=2, on_policy=True)
            self.assertEqual("aggregation_full_round", trainer.history[-1]["stage"])
            fit_imitation(trainer, episodes=1, max_ticks=2, stage="retrieve")
            self.assertEqual("imitation_retrieve", trainer.history[-1]["stage"])
            self.assertTrue(any(not torch.equal(before[name], value)
                                for name, value in trainer.actor.state_dict().items()))
            payload = torch.load(Path(temp) / "latest.pt", map_location="cpu",
                                 weights_only=False)
            self.assertNotIn("critic_state_dict", payload)
            self.assertIsNone(payload["optimizer_state_dict"])

    def test_multiple_angles_require_two_legal_distinct_sight_lines(self):
        enemy = SimpleNamespace(team="D", is_alive=True, pos=[0, 0])
        first = SimpleNamespace(team="A", is_alive=True, pos=[1, 0],
                                facing="N", blind_remaining=0)
        second = SimpleNamespace(team="A", is_alive=True, pos=[0, 1],
                                 facing="W", blind_remaining=0)
        game = SimpleNamespace(chars=[first, second, enemy],
                               check_cell_line_of_sight=lambda *_args, **_kwargs: True)
        self.assertTrue(_has_multiple_enemy_angles(game, [first, second]))
        second.pos = [2, 0]
        self.assertFalse(_has_multiple_enemy_angles(game, [first, second]))
        second.pos = [0, 1]
        second.blind_remaining = 1
        self.assertFalse(_has_multiple_enemy_angles(game, [first, second]))

    def test_full_round_initial_enemy_mix_uses_setup_travel_time(self):
        env = CoachTrainingEnvironment(Side.ATTACKER, seed=11,
                                       stage="full_round", max_ticks=1)
        env.reset()
        snapshot = env.sensor.build(game=env.game, side=Side.ATTACKER)
        generator = ScenarioGenerator(727)
        counts = Counter(
            enemy.category
            for _ in range(100)
            for enemy in generator.generate(
                actor_side=Side.ATTACKER, situation="carry", elapsed_ticks=20,
                enemy_count=5, currently_visible=snapshot.currently_visible,
                occupied_positions=(ally.position for ally in snapshot.allies),
            ).enemies
        )
        self.assertTrue(0.62 <= counts["point"] / 500 <= 0.78)
        self.assertTrue(0.12 <= counts["jitter"] / 500 <= 0.28)
        self.assertTrue(0.04 <= counts["random"] / 500 <= 0.16)


if __name__ == "__main__":
    unittest.main()
