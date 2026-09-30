"""Real-game rollout, actor/critic separation and checkpoint/PPO round trips."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from frc_v1.actions import validate_action
from frc_v1.baseline import FrcBaseline
from frc_v1.environment import FrcRoundEnvironment
from frc_v1.model import FrcPolicy
from frc_v1.train import action_record, gae


class FrcLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_environment(self, side="A", stage="threats"):
        env = FrcRoundEnvironment(side, stage=stage, opponent="default", seed=73)
        with contextlib.redirect_stdout(io.StringIO()):
            observation = env.reset()
        return env, observation

    def test_actor_actions_do_not_depend_on_privileged_critic(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A")
        first = policy.model.distribution([observation], deterministic=True, critic=[np.zeros(100, np.float32)])
        second = policy.model.distribution([observation], deterministic=True, critic=[np.ones(100, np.float32)])
        for name in first[0]:
            torch.testing.assert_close(first[0][name], second[0][name])
        torch.testing.assert_close(first[1], second[1])

    def test_policy_can_sample_real_legal_actions_and_recompute_log_probability(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A", deterministic=False)
        decision = policy.sample(observation, critic=env.critic_state())
        record, old_log, _ = policy.last_sample
        for slot, action in enumerate(decision.actions):
            validate_action(env.controller.snapshot, observation.masks, slot, action)
        _, log_prob, entropy, value = policy.model.distribution([observation], records=[record], critic=[env.critic_state()])
        self.assertAlmostEqual(old_log, float(log_prob.detach()[0]), places=5)
        self.assertTrue(torch.isfinite(entropy).all())
        loss = -log_prob.mean() + value.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.model.parameters() if p.grad is not None))
        env.step(decision)

    def test_checkpoint_load_restores_same_actions_and_rejects_other_side(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A")
        expected = policy.sample(observation)
        with tempfile.TemporaryDirectory(prefix="frc_checkpoint_", dir=Path(__file__).parent) as folder:
            path = Path(folder) / "policy.pt"
            policy.save(path, training={"smoke_only": True})
            restored = FrcPolicy.load(path, side="A")
            self.assertEqual(expected, restored.sample(observation))
            with self.assertRaises(ValueError):
                FrcPolicy.load(path, side="D")

    def test_missing_learned_model_never_silently_uses_baseline(self):
        with self.assertRaises(FileNotFoundError):
            FrcPolicy.load(Path(__file__).parent / "frc_v1" / "missing_policy.pt", side="A")

    def test_teacher_decisions_valid_across_curriculum_and_both_sides(self):
        teacher = FrcBaseline()
        for side, stage in (("A", "support"), ("D", "support"), ("A", "balemoon"), ("D", "balemoon")):
            with self.subTest(side=side, stage=stage):
                env, observation = self.make_environment(side, stage)
                positions = [tuple(c.pos) for c in env.game.chars if c.is_alive]
                self.assertEqual(len(positions), len(set(positions)))
                policy = FrcPolicy(env.controller.snapshot.grid, side)
                for _ in range(3):
                    decision = teacher.act(observation, env.controller.snapshot, env.controller.belief)
                    record = action_record(decision, env.game.width)
                    _, log_prob, _, _ = policy.model.distribution([observation], records=[record])
                    self.assertGreater(float(log_prob.detach()[0]), -10000)
                    result = env.step(decision)
                    observation = result.observation
                    if result.terminated:
                        break

    def test_gae_stops_bootstrapping_across_terminal_round(self):
        advantages, returns = gae([1, 2], [0.5, 0.7], [True, False], 3, gamma=0.9, lam=1)
        self.assertAlmostEqual(float(advantages[0]), 0.5)
        self.assertAlmostEqual(float(returns[1]), 4.7, places=5)

    def test_real_terminal_round_preserves_characters_and_winning_reward(self):
        env, observation = self.make_environment("D", "match")
        env.game.defender_setup_phase.finish()
        env.game.round_timer = 1
        env.game._prepare_team_controllers_tick()
        characters = tuple(env.game.chars)
        result = env.step(env.controller.decision)
        self.assertTrue(result.terminated)
        self.assertEqual("D", result.metrics["winner"])
        self.assertGreater(result.reward, 0.9)
        self.assertEqual(1, env.game.current_round)
        self.assertEqual(characters, tuple(env.game.chars))

    def test_gui_and_competition_registry_preserve_default_and_expose_both_modes(self):
        from roster_select import TEAM_AI_OPTIONS
        from run_competition_manager import CONTROLLER_OPTIONS
        from run_game import _build_team_ai
        self.assertEqual("Toru AI v3.1", next(iter(TEAM_AI_OPTIONS)))
        for values in (TEAM_AI_OPTIONS.values(), CONTROLLER_OPTIONS.values()):
            self.assertIn("frc_v1", values)
            self.assertIn("frc_v1_baseline", values)
        team = _build_team_ai("frc_v1_baseline")
        self.assertTrue(team.get_attacker_controller().handles_team_perception)
        self.assertEqual("D", team.get_defender_controller().side)


if __name__ == "__main__":
    unittest.main()
