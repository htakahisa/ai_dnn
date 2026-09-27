"""Task 11 model, training boundary, real-game rollout and checkpoint tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from coach_v1.common.types import MovementAction, ObjectiveAction, Side, TacticalIntent
from coach_v1.learning_coach import load_coach_policy
from coach_v1.models.coach_model import (
    CoachActorModel, CoachCritic, CoachModelConfig, action_feedback,
    legal_action_mask,
)
from coach_v1.observation.character_encoder import CoachInstruction
from coach_v1.observation.coach_encoder import CoachObservationEncoder
from coach_v1.perception.belief_memory import BeliefMemory
from coach_v1.perception.team_perception import TeamPerceptionBuilder
from coach_v1.common.constants import WATCH_POINTS_CONFIG_PATH
from coach_v1.common.watch_points import load_watch_points
from coach_v1.training.coach_environment import CoachTrainingEnvironment, _STAY
from coach_v1.training.coach_trainer import CoachTrainer
from coach_v1.evaluate_task11 import evaluate
from coach_v1.test_coach_v1_task10_coordinator import make_game
from map_data import NEW_MAZE_STR
from map_data_defender_setup import is_setup_position_allowed
from game_core import DEFUSE_REQUIRED_TICKS


class CoachTask11Test(unittest.TestCase):
    def test_five_heads_and_critic_requires_explicit_truth(self):
        actor = CoachActorModel()
        critic = CoachCritic()
        grid = torch.zeros(2, 27, 26, 44)
        vector = torch.zeros(2, 84)
        move, intent, objective, hidden = actor(grid, vector)
        self.assertEqual((2, 5, 5), tuple(move.shape))
        self.assertEqual((2, 5, 9), tuple(intent.shape))
        self.assertEqual((2, 5, 3), tuple(objective.shape))
        self.assertEqual((2, 96), tuple(hidden.shape))
        self.assertEqual((2,), tuple(critic(grid, vector, torch.zeros(2, 1, 26, 44)).shape))
        with self.assertRaises(ValueError):
            critic(grid, vector, torch.zeros(2, 5, 26, 44))

    def test_unseen_enemy_truth_changes_only_critic_input(self):
        encoder = CoachObservationEncoder()
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        observations = []
        truths = []
        for enemy_pos in ((4, 6), (4, 7)):
            game, _, _ = make_game(enemy_pos=enemy_pos)
            snapshot = TeamPerceptionBuilder().build(game=game, side=Side.ATTACKER)
            self.assertFalse(snapshot.sightings)
            belief = BeliefMemory(config.for_side(Side.ATTACKER)).update(snapshot)
            observations.append(encoder.encode(snapshot, belief, situation="carry"))
            truths.append(enemy_pos)
        np.testing.assert_array_equal(observations[0].grid, observations[1].grid)
        np.testing.assert_array_equal(observations[0].vector, observations[1].vector)
        self.assertNotEqual(truths[0], truths[1])
        model = CoachActorModel().eval()
        with torch.no_grad():
            outputs = [model(torch.from_numpy(o.grid.copy())[None],
                             torch.from_numpy(o.vector.copy())[None]) for o in observations]
        for first, second in zip(*outputs):
            torch.testing.assert_close(first, second)

    def test_mask_uses_only_legal_walkability_and_dead_slots(self):
        game, _, _ = make_game()
        game.chars[4].is_alive = False
        snapshot = TeamPerceptionBuilder().build(game=game, side=Side.ATTACKER)
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        belief = BeliefMemory(config.for_side(Side.ATTACKER)).update(snapshot)
        observation = CoachObservationEncoder().encode(snapshot, belief, situation="carry")
        mask = legal_action_mask(observation)
        self.assertEqual(1, int(mask.movement[4].sum()))
        self.assertTrue(mask.movement[4, tuple(MovementAction).index(MovementAction.STAY)])
        self.assertEqual(1, int(mask.objective[4].sum()))

    def test_mask_excludes_visible_ally_occupied_tile(self):
        environment = CoachTrainingEnvironment(Side.ATTACKER, seed=1,
                                               stage="2v1", max_ticks=1)
        state = environment.reset()
        self.assertFalse(state.mask.movement[0,
                         tuple(MovementAction).index(MovementAction.MOVE_E)])
        self.assertFalse(state.mask.movement[1,
                         tuple(MovementAction).index(MovementAction.MOVE_W)])

    def test_defender_setup_mask_blocks_forbidden_cells(self):
        game, _, _ = make_game(side=Side.DEFENDER)
        game.defender_setup_phase.active = True
        game.defender_setup_phase.ticks_remaining = 3
        snapshot = TeamPerceptionBuilder().build(game=game, side=Side.DEFENDER)
        config = load_watch_points(WATCH_POINTS_CONFIG_PATH, NEW_MAZE_STR)
        belief = BeliefMemory(config.for_side(Side.DEFENDER)).update(snapshot)
        observation = CoachObservationEncoder().encode(snapshot, belief, situation="search")
        mask = legal_action_mask(observation)
        from coach_v1.common.constants import MOVEMENT_DELTAS
        for slot, ally in enumerate(snapshot.allies):
            for index, move in enumerate(MovementAction):
                if not mask.movement[slot, index] or move is MovementAction.STAY:
                    continue
                dr, dc = MOVEMENT_DELTAS[move.value]
                self.assertTrue(is_setup_position_allowed(ally.position[0] + dr,
                                                          ally.position[1] + dc))

    def test_real_game_curriculum_and_frozen_characters(self):
        for side in Side:
            with self.subTest(side=side):
                environment = CoachTrainingEnvironment(side, seed=9, stage="2v1", max_ticks=1)
                state = environment.reset()
                self.assertEqual((27, 26, 44), state.observation.grid.shape)
                self.assertEqual((1, 26, 44), state.critic_enemy_truth.shape)
                self.assertAlmostEqual(0.2, float(state.critic_enemy_truth.sum()))
                self.assertTrue(all(not parameter.requires_grad for policy in environment.characters.values()
                                    for parameter in policy.model.parameters()))
                transition = environment.step(_STAY)
                self.assertTrue(transition.done)
                self.assertEqual(2, len(environment.controller.action_log))

    def test_remaining_curriculum_stages_have_expected_alive_counts(self):
        for stage, count in (("2v2", 2), ("3v3", 3), ("5v5", 5)):
            for side in Side:
                with self.subTest(stage=stage, side=side):
                    environment = CoachTrainingEnvironment(side, seed=3,
                                                           stage=stage, max_ticks=1)
                    state = environment.reset()
                    self.assertAlmostEqual(count / 5, float(state.critic_enemy_truth.sum()))
                    environment.step(_STAY)

    def test_real_training_state_keeps_unseen_truth_out_of_actor(self):
        environment = CoachTrainingEnvironment(Side.ATTACKER, seed=5, max_ticks=1)
        environment.reset()
        enemy = next(c for c in environment.game.chars if c.team == "D" and c.is_alive)
        states = []
        for position in ((4, 6), (4, 7)):
            enemy.pos = list(position)
            environment.memory.reset()
            states.append(environment._observe())
        np.testing.assert_array_equal(states[0].observation.grid, states[1].observation.grid)
        np.testing.assert_array_equal(states[0].observation.vector, states[1].observation.vector)
        self.assertFalse(np.array_equal(states[0].critic_enemy_truth,
                                        states[1].critic_enemy_truth))

    def test_headless_round_transition_ends_episode(self):
        environment = CoachTrainingEnvironment(Side.ATTACKER, seed=4, max_ticks=2)
        environment.reset()
        round_ally = next(c for c in environment.game.chars if c.team == "A" and c.is_alive)
        original_process = environment.game.process_battle

        def next_round():
            round_ally.is_alive = False
            environment.game.current_round += 1
            environment.game.defender_wins += 1
            environment.game.init_round()

        environment.game.process_battle = next_round
        try:
            transition = environment.step(_STAY)
        finally:
            environment.game.process_battle = original_process
        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state)
        self.assertEqual(-0.05, transition.metrics["reward_death"])

    def test_time_limit_counts_final_ticks_new_clear_cells(self):
        environment = CoachTrainingEnvironment(Side.DEFENDER, seed=1,
                                               stage="2v1", max_ticks=1)
        environment.reset()
        transition = environment.step(_STAY)
        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_state)
        self.assertGreater(transition.metrics["new_clear_cells"], 0)
        self.assertGreater(transition.metrics["reward_clear"], 0)

    def test_round_end_preserves_finished_rounds_clear_reward(self):
        environment = CoachTrainingEnvironment(Side.DEFENDER, seed=1,
                                               stage="2v1", max_ticks=2)
        environment.reset()
        original_process = environment.game.process_battle

        def next_round():
            environment.game.current_round += 1
            environment.game.defender_wins += 1
            environment.game.init_round()

        environment.game.process_battle = next_round
        try:
            transition = environment.step(_STAY)
        finally:
            environment.game.process_battle = original_process
        self.assertTrue(transition.done)
        self.assertGreater(transition.metrics["new_clear_cells"], 0)

    def test_defuse_bonus_survives_headless_round_reset(self):
        environment = CoachTrainingEnvironment(Side.DEFENDER, seed=2,
                                               stage="2v1", max_ticks=2)
        environment.reset()
        defuser = next(c for c in environment.game.chars if c.team == "D" and c.is_alive)
        environment.game.is_planted = True
        environment.game.planted_pos = tuple(defuser.pos)
        environment.memory.reset()
        environment.controller.reset_round()
        environment.state = environment._observe()
        original_process = environment.game.process_battle

        def next_round():
            defuser.defuse_timer = DEFUSE_REQUIRED_TICKS
            environment.game.current_round += 1
            environment.game.defender_wins += 1
            environment.game.init_round()

        environment.game.process_battle = next_round
        try:
            transition = environment.step(_STAY)
        finally:
            environment.game.process_battle = original_process
        self.assertEqual(0.1, transition.metrics["reward_objective"])

    def test_training_environment_rejects_occupied_ally_move(self):
        environment = CoachTrainingEnvironment(Side.ATTACKER, seed=1,
                                               stage="2v1", max_ticks=1)
        environment.reset()
        actions = list(_STAY)
        actions[0] = CoachInstruction(MovementAction.MOVE_E,
                                      ObjectiveAction.NONE, TacticalIntent.HOLD)
        with self.assertRaises(ValueError):
            environment.step(tuple(actions))

    def test_action_feedback_reports_own_failed_move_only(self):
        environment = CoachTrainingEnvironment(Side.ATTACKER, seed=1,
                                               stage="2v1", max_ticks=2)
        previous = environment.reset().observation
        failed_actions = list(_STAY)
        failed_actions[0] = CoachInstruction(MovementAction.MOVE_E,
                                             ObjectiveAction.NONE, TacticalIntent.HOLD)
        failed = action_feedback(previous, previous, tuple(failed_actions))
        self.assertEqual(1, failed[0, tuple(MovementAction).index(MovementAction.MOVE_E)])
        self.assertEqual(1, failed[0, 6])
        actions = list(_STAY)
        actions[0] = CoachInstruction(MovementAction.MOVE_W,
                                      ObjectiveAction.NONE, TacticalIntent.HOLD)
        current = environment.step(tuple(actions)).next_state.observation
        feedback = action_feedback(current, previous, tuple(actions))
        self.assertEqual((5, 7), feedback.shape)
        self.assertEqual(1, feedback[0, tuple(MovementAction).index(MovementAction.MOVE_W)])
        self.assertEqual(1, feedback[:, 5].sum())
        self.assertEqual(0, feedback[:, 6].sum())

    def test_feedback_input_and_legacy_checkpoint_are_separate(self):
        actor = CoachActorModel(CoachModelConfig(action_feedback=True))
        grid = torch.zeros(1, 27, 26, 44)
        vector = torch.zeros(1, 84)
        with self.assertRaises(ValueError):
            actor(grid, vector, feedback=torch.zeros(1, 5, 6))
        with tempfile.TemporaryDirectory() as root:
            legacy = CoachTrainer(Side.DEFENDER, seed=23, directory=Path(root),
                                  config=CoachModelConfig(action_feedback=False))
            legacy.save()
            policy = load_coach_policy(Side.DEFENDER, Path(root) / "latest.pt")
            self.assertFalse(policy.model.config.action_feedback)
            self.assertNotIn("action_feedback", legacy._metadata().model_config)

    def test_feedback_training_rollout_uses_safe_input(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = CoachTrainer(Side.ATTACKER, seed=29, directory=Path(root),
                                   config=CoachModelConfig(action_feedback=True))
            environment = CoachTrainingEnvironment(Side.ATTACKER, seed=29,
                                                   stage="2v1", max_ticks=2)
            trajectory, _ = trainer.collect(environment, episode=0)
            self.assertEqual(2, len(trajectory))
            self.assertEqual((5, 7), trajectory[0].feedback.shape)
            np.testing.assert_array_equal(trajectory[0].feedback,
                                          np.zeros((5, 7), dtype=np.float32))
            trainer.update(trajectory, epochs=1)
            trainer.save()
            policy = load_coach_policy(Side.ATTACKER, Path(root) / "latest.pt")
            self.assertTrue(policy.model.config.action_feedback)

    def test_rollout_update_save_resume_and_actor_only_loader(self):
        with tempfile.TemporaryDirectory() as root:
            trainer = CoachTrainer(Side.ATTACKER, seed=17, directory=Path(root))
            environment = CoachTrainingEnvironment(Side.ATTACKER, seed=17, max_ticks=2)
            rollout_path = Path(root) / "rollout.pt"
            trajectory, metrics = trainer.collect(environment, episode=0,
                                                  rollout_path=rollout_path)
            self.assertEqual(2, len(trajectory))
            self.assertTrue(rollout_path.exists())
            self.assertEqual(2, metrics["ticks"])
            trainer.update(trajectory, epochs=1)
            trainer.episode = 1
            trainer.save()
            payload = torch.load(Path(root) / "latest.pt", weights_only=False)
            self.assertNotIn("critic_state_dict", payload)
            self.assertIsNone(payload["optimizer_state_dict"])
            policy = load_coach_policy(Side.ATTACKER, Path(root) / "latest.pt")
            instructions = policy.act(trajectory_observation(environment, trajectory[0]))
            self.assertEqual(5, len(instructions))
            resumed = CoachTrainer(Side.ATTACKER, seed=17, directory=Path(root))
            resumed.resume()
            self.assertEqual(trainer.training_step, resumed.training_step)
            self.assertEqual(1, resumed.episode)
            for name, weights in trainer.actor.state_dict().items():
                torch.testing.assert_close(weights, resumed.actor.state_dict()[name])
            for name, weights in trainer.critic.state_dict().items():
                torch.testing.assert_close(weights, resumed.critic.state_dict()[name])
            report = evaluate(Side.ATTACKER, stage="2v1", seeds=range(1),
                              max_ticks=1, checkpoint=Path(root) / "latest.pt")
            self.assertEqual(trainer.training_step, report["training_step"])
            self.assertEqual(64, len(report["checkpoint_sha256"]))
            with self.assertRaises(ValueError):
                load_coach_policy(Side.DEFENDER, Path(root) / "latest.pt")

    def test_resume_advances_to_next_curriculum_stage(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            first = CoachTrainer(Side.ATTACKER, seed=31, directory=directory)
            first.fit(episodes=1, stage="2v1", max_ticks=2, save_rollouts=False)
            second = CoachTrainer(Side.ATTACKER, seed=31, directory=directory)
            second.resume()
            history = second.fit(episodes=1, stage="2v2", max_ticks=2,
                                 save_rollouts=False)
            self.assertEqual(["2v1", "2v2"], [item["stage"] for item in history])
            self.assertEqual(2, second.episode)
            self.assertEqual(2, len(history))

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as root:
            continuous = CoachTrainer(Side.DEFENDER, seed=37,
                                      directory=Path(root) / "continuous")
            continuous.fit(episodes=2, stage="2v1", max_ticks=2,
                           save_rollouts=False)
            interrupted = CoachTrainer(Side.DEFENDER, seed=37,
                                       directory=Path(root) / "resumed")
            interrupted.fit(episodes=1, stage="2v1", max_ticks=2,
                            save_rollouts=False)
            resumed = CoachTrainer(Side.DEFENDER, seed=37,
                                   directory=Path(root) / "resumed")
            resumed.resume()
            resumed.fit(episodes=1, stage="2v1", max_ticks=2,
                        save_rollouts=False)
            self.assertEqual(continuous.history, resumed.history)
            for name, weights in continuous.actor.state_dict().items():
                torch.testing.assert_close(weights, resumed.actor.state_dict()[name],
                                           rtol=0, atol=0)
            for name, weights in continuous.critic.state_dict().items():
                torch.testing.assert_close(weights, resumed.critic.state_dict()[name],
                                           rtol=0, atol=0)


def trajectory_observation(environment, step):
    from coach_v1.observation.coach_encoder import CoachObservation
    return CoachObservation(step.grid, step.vector,
                            "coach-observation-v1", environment.encoder.map_hash,
                            environment.encoder.watch_points_hash)


if __name__ == "__main__":
    unittest.main()
