"""Regression tests for defender-retake checkpoint expansion."""

import unittest
from unittest.mock import patch

import torch

from gc_v1 import train_defender_retake_gc as retake


class DefenderRetakeWarmStartTests(unittest.TestCase):
    def test_53_to_59_preserves_q_and_facing_outputs(self):
        torch.manual_seed(417)
        old = retake.DuelingQNet(53, 9).eval()
        expanded = retake.DuelingQNet(59, 9).eval()
        old_obs, old_actions = retake.expand_retake_policy_state(
            {"model_state_dict": old.state_dict()}, expanded
        )
        self.assertEqual((old_obs, old_actions), (53, 9))
        for key in ("feature.0.weight", "facing_feature.0.weight"):
            self.assertTrue(torch.equal(
                expanded.state_dict()[key][:, :53], old.state_dict()[key]
            ))
            self.assertTrue(torch.count_nonzero(
                expanded.state_dict()[key][:, 53:]
            ).item() == 0)

        observation = torch.randn(8, 53)
        added_context = torch.randn(8, 6)
        widened_observation = torch.cat((observation, added_context), dim=1)
        actions = torch.arange(8) % 9
        with torch.no_grad():
            torch.testing.assert_close(
                expanded(widened_observation), old(observation),
                atol=1e-6, rtol=1e-6,
            )
            torch.testing.assert_close(
                expanded.facing_values(widened_observation, actions),
                old.facing_values(observation, actions),
                atol=1e-6, rtol=1e-6,
            )

    def test_legacy_action_and_facing_rows_are_preserved(self):
        torch.manual_seed(418)
        old = retake.DuelingQNet(37, 7).eval()
        expanded = retake.DuelingQNet(59, 9).eval()
        retake.expand_retake_policy_state(old.state_dict(), expanded)
        for key in ("adv_head.2.weight", "adv_head.2.bias"):
            self.assertTrue(torch.equal(
                expanded.state_dict()[key][:7], old.state_dict()[key]
            ))
        self.assertTrue(torch.equal(
            expanded.state_dict()["facing_output.weight"][:, :71],
            old.state_dict()["facing_output.weight"],
        ))
        self.assertEqual(
            torch.count_nonzero(
                expanded.state_dict()["facing_output.weight"][:, 71:]
            ).item(), 0,
        )

    def test_preflight_rejects_a_collapsed_expansion(self):
        collapsed = {
            "episodes": 90,
            "seeds": [3026091700, 5026091700, 6026091700],
            "defuse_rate": 0.0,
            "entry_rate": 0.0,
            "worst_defuse_rate": 0.0,
        }
        baseline = {**collapsed, "defuse_rate": 0.6, "entry_rate": 0.8}
        with patch.object(retake, "evaluate_retake_fixed", return_value=collapsed):
            with self.assertRaisesRegex(RuntimeError, "regressed before training"):
                retake.verify_expanded_retake_baseline(object(), baseline, 30)


if __name__ == "__main__":
    unittest.main()
