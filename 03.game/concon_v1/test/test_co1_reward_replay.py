"""Route decisions retain rewards while contact/abilities own subsequent ticks."""

import contextlib
import io
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1.co1_train_attacker import RouteReplayCollector, _optimize
    from concon_v1.co1_attacker_common import SharedRouteDQN


class RewardReplayTests(unittest.TestCase):
    def setUp(self):
        self.collector = RouteReplayCollector(2, gamma=0.9)
        self.replay = []

    def tick(self, rewards, applied=(False, False), *, done=False,
             interrupted=False, alive=(True, True), active=(True, True),
             route_active=True, value=0):
        old = [np.full(3, value, dtype=np.float32) for _ in range(2)]
        nxt = [obs + 1 for obs in old]
        masks = [np.ones(6, dtype=bool) for _ in range(2)]
        return self.collector.record(
            (old, masks, rewards, nxt, masks, done), [2, 3], applied,
            active, alive, self.replay, interrupted=interrupted,
            route_active=route_active,
        )

    def test_plant_rewards_reach_escorts_without_terminal_policy_calls(self):
        self.tick([0.035, 0.035], (True, True))
        self.tick([-0.005, -0.005])  # Contact or an ability; no route action.
        self.tick([9.995, 9.995], done=True)
        self.assertEqual(len(self.replay), 2)
        for sample in self.replay:
            self.assertAlmostEqual(sample[2], 0.035 + 0.9 * -0.005 + 0.9**2 * 9.995)
            self.assertEqual(sample[5:], (1.0, 3))
        self.assertTrue(all(p is None for p in self.collector.pending))

    def test_next_actual_decision_closes_previous_action_before_current_reward(self):
        self.tick([1.0, 0.0], (True, False))
        self.tick([2.0, 0.0], value=1)
        self.tick([3.0, 0.0], (True, False), value=7)
        previous = self.replay[0]
        self.assertAlmostEqual(previous[2], 1.0 + 0.9 * 2.0)
        np.testing.assert_array_equal(previous[3], np.full(3, 7))
        self.assertEqual(previous[5:], (0.0, 2))
        self.tick([10.0, 0.0], done=True)
        self.assertAlmostEqual(self.replay[1][2], 3.0 + 0.9 * 10.0)

    def test_failed_round_penalty_is_kept_without_policy_call(self):
        self.tick([0.035, 0.0], (True, False))
        self.tick([-3.005, -3.005], done=True)
        self.assertEqual(len(self.replay), 1)
        self.assertAlmostEqual(self.replay[0][2], 0.035 - 0.9 * 3.005)
        self.assertEqual(self.replay[0][5], 1.0)

    def test_spike_drop_closes_phase_and_retrieval_rewards_are_excluded(self):
        self.tick([0.035, 0.035], (True, True))
        self.tick([-3.005, -3.005], interrupted=True)
        self.tick([10.0, 10.0], done=True, route_active=False)
        self.assertEqual(len(self.replay), 2)
        self.assertTrue(all(sample[5] == 1.0 for sample in self.replay))
        self.assertTrue(all(p is None for p in self.collector.pending))

    def test_death_closes_only_dead_actor_and_cannot_receive_later_rewards(self):
        self.tick([0.035, 0.035], (True, True))
        self.tick([-0.005, -0.005], alive=(False, True))
        self.tick([10.0, 10.0], done=True, active=(False, True), alive=(False, True))
        self.assertEqual(len(self.replay), 2)
        self.assertAlmostEqual(self.replay[0][2], 0.035 - 0.9 * 0.005)
        self.assertEqual(self.replay[0][5:], (1.0, 2))
        self.assertAlmostEqual(self.replay[1][2], 0.035 - 0.9 * 0.005 + 0.9**2 * 10)

    def test_consecutive_route_decisions_keep_one_tick_returns(self):
        self.tick([0.035, 0.035], (True, True))
        self.tick([0.25, 0.25], (True, True), done=True)
        self.assertEqual(len(self.replay), 4)
        self.assertEqual([s[2] for s in self.replay], [0.035, 0.25, 0.035, 0.25])
        self.assertTrue(all(s[6] == 1 for s in self.replay))

    def test_optimizer_discounts_bootstrap_by_elapsed_ticks(self):
        model = SharedRouteDQN(obs_dim=3)
        target = SharedRouteDQN(obs_dim=3)
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()
            target.value[-1].bias.fill_(2.0)
        replay = [(np.zeros(3), 2, 1.0, np.ones(3), np.ones(6, dtype=bool), 0.0, 3)]
        optimizer = torch.optim.Adam(model.parameters())
        loss_function = torch.nn.functional.smooth_l1_loss
        with patch('concon_v1.co1_train_attacker.nn.functional.smooth_l1_loss',
                   wraps=loss_function) as loss:
            _optimize(model, target, optimizer, replay, 1, 0.9)
        self.assertAlmostEqual(loss.call_args.args[1].item(), 1.0 + 0.9**3 * 2.0, places=6)

    def test_real_a3_plant_flushes_all_five_pending_route_actions(self):
        from concon_v1.co1_battle_training import BattleRouteEnv
        from concon_v1.co1_attacker_scenarios import get_scenario

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        scenario = get_scenario('A3')
        model = SharedRouteDQN(obs_dim=scenario.obs_dim)
        collector = RouteReplayCollector(5)
        replay = []
        with contextlib.redirect_stdout(io.StringIO()):
            env = BattleRouteEnv(0, ['omoko_v1'], model=model,
                                 map_name=scenario, learn_setup=True)
            for _ in range(150):
                active = list(env.alive)
                transition = env.step(epsilon=1.0)
                before = len(replay)
                collector.record(
                    transition, env.actions, env.policy_action_applied, active,
                    env.alive, replay, route_active=env.route_active_before_step,
                    interrupted=env.retrieve_active or env.success,
                )
                if env.done:
                    break
        self.assertTrue(env.success, 'BFS exploration must reach the plant in this fixture')
        self.assertTrue(any(not applied for applied in env.policy_action_applied))
        terminal = [sample for sample in replay[before:] if sample[5]]
        self.assertEqual(len(terminal), 5)
        self.assertTrue(all(sample[2] > 0 for sample in terminal))
        self.assertTrue(all(p is None for p in collector.pending))


if __name__ == '__main__':
    unittest.main()
