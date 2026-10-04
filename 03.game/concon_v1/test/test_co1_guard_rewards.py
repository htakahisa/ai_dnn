"""Regressions for holding, aim, and team credit after an actor dies."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from concon_v1.co1_guard_rewards import decision_reward, GAMMA, DEATH_PENALTY, ROUND_REWARD
from concon_v1.co1_guard_battle_training import GuardBattleEnv
from concon_v1.co1_guard_common import WAIT_ACTION, ACTION_DIM, observation_dim
from concon_v1.co1_guard_scenarios import get_scenario
from concon_v1.co1_train_guard import qualifies_as_best
from concon_v1.evaluate_co1_guard import behavior_summary


def context(**changes):
    result = dict(position=(2, 2), goal=(2, 2), aim=(2, 4), spike=(2, 3),
                  distance_goal=0, distance_spike=1, fireable=False, tap=False,
                  target=None, stopped=1, can_move=True, tick=0)
    result.update(changes)
    return result


class GuardRewardTests(unittest.TestCase):
    def test_leaving_and_returning_is_worse_than_holding(self):
        for facing in ("E", "NE", "W"):
            with self.subTest(facing=facing):
                hold = decision_reward(WAIT_ACTION, context(), (2, 2), facing, 0)
                leave = decision_reward(24, context(), (2, 3), facing, 1)
                back = decision_reward(16, context(position=(2, 3), distance_goal=1),
                                       (2, 2), facing, 0)
                self.assertLess(leave + GAMMA * back, 0)
                self.assertLess(leave + GAMMA * back, hold * (1 + GAMMA))

    def test_aim_is_signed_and_distinguishes_front_from_diagonal_everywhere(self):
        for ctx, pos in ((context(), (2, 2)),
                         (context(position=(2, 1), distance_goal=1), (2, 1)),
                         (context(fireable=True, target=(2, 4)), (2, 2))):
            with self.subTest(ctx=ctx):
                rewards = [decision_reward(WAIT_ACTION, ctx, pos, facing, ctx['distance_goal'])
                           for facing in ("E", "NE", "W")]
                self.assertGreater(rewards[0], rewards[1])
                self.assertGreater(rewards[1], rewards[2])

    def test_contact_stopping_and_second_stationary_tick_are_preferred(self):
        first = decision_reward(WAIT_ACTION, context(fireable=True, target=(2, 4), stopped=0),
                                (2, 2), "E", 0)
        second = decision_reward(WAIT_ACTION, context(fireable=True, target=(2, 4)),
                                 (2, 2), "E", 0)
        moving = decision_reward(24, context(fireable=True, target=(2, 4)), (2, 3), "E", 1)
        self.assertGreater(second, first)
        self.assertGreater(first, moving)

    def test_blocked_tap_approach_has_no_leave_goal_penalty(self):
        ctx = context(tap=True, position=(2, 1), goal=(2, 1), distance_spike=2)
        approach = decision_reward(24, ctx, (2, 2), "E", 1)
        wait = decision_reward(WAIT_ACTION, ctx, (2, 1), "E", 2)
        blocked = decision_reward(WAIT_ACTION, {**ctx, 'can_move': False}, (2, 1), "E", 2)
        utility = decision_reward((5 + 2 * 3) * 8, ctx, (2, 1), "E", 2)
        self.assertGreater(approach, 0)
        self.assertGreater(approach, wait)
        self.assertGreater(blocked, wait)
        self.assertGreater(utility, wait)

    def test_quiet_return_is_better_than_avoidable_wait(self):
        ctx = context(position=(2, 1), distance_goal=1)
        returning = decision_reward(24, ctx, (2, 2), "E", 0)
        waiting = decision_reward(WAIT_ACTION, ctx, (2, 1), "E", 1)
        self.assertGreater(returning, waiting)


class GuardTeamCreditTests(unittest.TestCase):
    def test_dead_and_disabled_actors_receive_round_result_once_with_elapsed_discount(self):
        for won in (False, True):
            with self.subTest(won=won):
                env = object.__new__(GuardBattleEnv)
                env.scenario = get_scenario('L')
                env.attackers = [SimpleNamespace(is_alive=True) for _ in range(5)]
                env.defenders = []
                env.done = False
                env.elapsed_ticks = 0
                env.metrics = {'tap_ticks': 0}
                env._quiet_history = {}
                obs = np.zeros(observation_dim(env.scenario), dtype=np.float16)
                env.pending = [dict(obs=obs.copy(), action=WAIT_ACTION, reward=0.0, duration=0)
                               for _ in range(2)] + [None] * 3
                # No decisions simulate engine-disabled actors. Actor zero dies
                # on tick one; the other remains alive until the round finishes.
                def tick():
                    env.attackers[0].is_alive = False
                    if env.elapsed_ticks == 2:
                        env.game.round_over = True
                        env.game.attacker_wins = int(won)
                env.game = SimpleNamespace(step_tick=tick, round_over=False,
                                           match_over=False, attacker_wins=0)
                transitions, rewards, done = env.step()
                self.assertFalse(done)
                self.assertFalse(transitions)
                self.assertEqual(rewards[0], -DEATH_PENALTY)
                self.assertIsNotNone(env.pending[0])
                _, rewards, _ = env.step()
                self.assertEqual(rewards[0], 0)
                transitions, _, done = env.step()
                self.assertTrue(done)
                self.assertEqual(len(transitions), 2)
                terminal = ROUND_REWARD if won else -ROUND_REWARD
                self.assertAlmostEqual(transitions[0][2], -DEATH_PENALTY + GAMMA**2 * terminal)
                self.assertAlmostEqual(transitions[1][2], GAMMA**2 * terminal)
                self.assertTrue(all(record[5] == 1 and record[6] == 3 for record in transitions))
                self.assertFalse(any(env.pending))


class GuardBehaviorEvaluationTests(unittest.TestCase):
    def test_rates_use_decision_counts_and_handle_no_contact(self):
        summary = behavior_summary([
            {'quiet_decisions': 10, 'quiet_at_goal_decisions': 2,
             'quiet_leave_goal_decisions': 1, 'quiet_reversals': 2},
            {'quiet_decisions': 30, 'quiet_at_goal_decisions': 8,
             'quiet_leave_goal_decisions': 1, 'quiet_reversals': 2},
        ])
        self.assertEqual(summary['quiet_leave_goal_rate'], .2)
        self.assertEqual(summary['quiet_reversal_rate'], .1)
        self.assertEqual(summary['moving_fire_rate'], 0)
        self.assertEqual(behavior_summary([])['behavior_error'], 0)

    def test_equal_win_rates_prefer_stable_behavior(self):
        stable = {'mean_win_rate': .6, 'min_team_win_rate': .4,
                  'behavior': {'behavior_error': .05}}
        jitter = {**stable, 'behavior': {'behavior_error': .4}}
        self.assertFalse(qualifies_as_best(jitter, stable))
        self.assertTrue(qualifies_as_best(stable, jitter))
        self.assertFalse(qualifies_as_best({**stable, 'mean_win_rate': .5}, jitter))
        self.assertTrue(qualifies_as_best({**jitter, 'mean_win_rate': .7}, stable))


if __name__ == '__main__':
    unittest.main()
