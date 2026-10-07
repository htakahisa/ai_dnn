"""Checks for stable defender-retake fine-tuning after a warm start."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from gc_v1 import train_defender_retake_gc as retake


class DefenderRetakeTrainingTests(unittest.TestCase):
    def test_expanded_model_updates_only_new_action_inputs(self):
        torch.manual_seed(419)
        net = retake.DuelingQNet(59, retake.N_ACTIONS)
        before = {name: tensor.clone() for name, tensor in net.state_dict().items()}
        retake.train_only_new_retake_context(net, 53)
        optimizer = torch.optim.Adam(
            (parameter for parameter in net.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        observations = torch.randn(16, 59)
        loss = net(observations).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.equal(net.feature[0].weight[:, :53], before["feature.0.weight"][:, :53]))
        self.assertFalse(torch.equal(net.feature[0].weight[:, 53:], before["feature.0.weight"][:, 53:]))
        for name, tensor in net.state_dict().items():
            if name.startswith(("feature.", "value_head.", "adv_head.")) and name != "feature.0.weight":
                self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_warm_start_uses_small_relative_exploration_schedule(self):
        self.assertAlmostEqual(retake.exploration_epsilon(501, 500, 20000, True), 0.08)
        self.assertAlmostEqual(retake.exploration_epsilon(1501, 500, 20000, True), 0.02)
        self.assertAlmostEqual(retake.exploration_epsilon(1, 0, 20000, False), 0.99993875)

    def test_training_batch_keeps_successful_baseline_transitions(self):
        online = retake.ReplayBuffer()
        baseline = retake.ReplayBuffer()
        state = np.zeros(1, dtype=np.float32)
        mask = np.ones(retake.N_ACTIONS, dtype=bool)
        for _ in range(256):
            online.push(state, 4, 0.0, state, 0.0, mask, mask)
        for _ in range(64):
            baseline.push(state, retake.ACTION_DEFUSE, 1.0, state, 1.0, mask, mask)

        net = torch.nn.Linear(1, 1)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        observed = {}

        def fake_td_loss(model, _target, batch, _gamma):
            observed["actions"] = batch.action
            return model.weight.sum() * 0

        with patch.object(retake, "compute_td_loss", side_effect=fake_td_loss):
            retake.train_step(net, net, optimizer, online, 256, 0.99, baseline)

        self.assertEqual(observed["actions"].count(4), 192)
        self.assertEqual(observed["actions"].count(retake.ACTION_DEFUSE), 64)


if __name__ == "__main__":
    unittest.main()
