"""Idle actions must be sampled, penalized only when avoidable, and learned."""

import random
import unittest

import numpy as np
import torch

from concon_v1.co1_attacker_common import ACTION_WAIT, SharedRouteDQN, _choose_action
from concon_v1.co1_train_attacker import (
    RouteEnv, _optimize, epsilon_by_episode, training_epsilon_by_episode,
)
from concon_v1.co1_attacker_rewards import avoidable_wait_penalty


class IdleLearningTests(unittest.TestCase):
    def test_guided_exploration_also_samples_avoidable_wait(self):
        env = RouteEnv(0, map_name='A3')
        observations, masks = env._collect()
        index = next(i for i, mask in enumerate(masks) if mask[:4].any())
        rng = random.Random(0)
        actions = [_choose_action(None, observations[index], masks[index], 1.0, rng,
                                  route=env.routes[index], position=env.positions[index])
                   for _ in range(200)]
        self.assertIn(ACTION_WAIT, actions)
        self.assertTrue(any(action < 4 for action in actions))
        self.assertTrue(all(masks[index][action] for action in actions))

    def test_penalty_preserves_contact_blocked_and_goal_waits(self):
        obs = np.zeros(30, dtype=np.float32)
        mask = np.array([True, False, False, False, True, False])
        self.assertEqual(avoidable_wait_penalty(ACTION_WAIT, obs, mask, 2), 0.1)
        self.assertEqual(avoidable_wait_penalty(0, obs, mask, 2), 0.0)
        self.assertEqual(avoidable_wait_penalty(ACTION_WAIT, obs, mask, 0), 0.0)
        blocked = mask.copy(); blocked[0] = False
        self.assertEqual(avoidable_wait_penalty(ACTION_WAIT, obs, blocked, 2), 0.0)
        for flag in (-2, -1):
            contact = obs.copy(); contact[flag] = 1.0
            self.assertEqual(avoidable_wait_penalty(ACTION_WAIT, contact, mask, 2), 0.0)

    def test_route_environment_charges_avoidable_wait(self):
        env = RouteEnv(0, map_name='A3')
        _, masks = env._collect()
        index = next(i for i, mask in enumerate(masks) if mask[:4].any())
        transition = env.step([ACTION_WAIT] * 5)
        self.assertAlmostEqual(transition[2][index], -0.105)

    def test_greedy_training_rounds_do_not_change_exploration_schedule(self):
        self.assertGreater(training_epsilon_by_episode(100), 0.5)
        self.assertEqual(training_epsilon_by_episode(500), 0.0)
        self.assertEqual(training_epsilon_by_episode(501), epsilon_by_episode(501))
        self.assertEqual(epsilon_by_episode(700), 0.05)
        self.assertEqual(training_epsilon_by_episode(700), 0.0)
        self.assertEqual(training_epsilon_by_episode(701), 0.05)

    def test_optimizer_lowers_wait_relative_to_movement_from_failure(self):
        model = SharedRouteDQN(obs_dim=3)
        target = SharedRouteDQN(obs_dim=3)
        with torch.no_grad():
            for network in (model, target):
                for parameter in network.parameters():
                    parameter.zero_()
        observation = np.zeros(3, dtype=np.float32)
        mask = np.array([True, False, False, False, True, False])
        replay = [(observation, ACTION_WAIT, -0.105, observation, mask, 0.0, 1)]
        _optimize(model, target, torch.optim.Adam(model.parameters(), lr=0.001),
                  replay, 1, 0.99)
        values = model(torch.as_tensor(observation).unsqueeze(0))[0]
        self.assertLess(values[ACTION_WAIT].item(), values[0].item())


if __name__ == '__main__':
    unittest.main()
