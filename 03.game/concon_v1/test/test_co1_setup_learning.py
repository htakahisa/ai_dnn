"""Setup movement must produce trainable decisions under production restrictions."""

import contextlib
import io
import unittest
from unittest.mock import patch

import torch

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1.co1_battle_training import BattleRouteEnv
    from concon_v1.co1_attacker_common import ACTION_PLANT, ACTION_WAIT, SharedRouteDQN
    from concon_v1.co1_attacker_scenarios import get_scenario
    from map_data_defender_setup import is_setup_position_allowed


class SetupLearningTests(unittest.TestCase):
    def make_env(self, map_name):
        model = SharedRouteDQN(obs_dim=get_scenario(map_name).obs_dim)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.advantage[-1].bias[:4] = 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            return BattleRouteEnv(seed=0, opponents=['omoko_v1'], model=model,
                                  map_name=map_name, learn_setup=True)

    def test_setup_decisions_are_returned_for_replay_and_continue_into_live(self):
        for map_name in ('A2', 'A3'):
            with self.subTest(map_name=map_name):
                env = self.make_env(map_name)
                self.assertTrue(env.game.defender_setup_phase.active)
                initial = list(env.positions)
                ticks = env.game.defender_setup_phase.ticks_remaining
                for _ in range(ticks):
                    transition = env.step(epsilon=0.0)
                    self.assertEqual(env.game.battle_tick, 0)
                    self.assertTrue(all(env.policy_action_applied))
                    self.assertFalse(transition[-1])
                    for index, mask in enumerate(transition[1]):
                        self.assertTrue(mask[env.actions[index]])
                        self.assertFalse(mask[ACTION_PLANT])
                        self.assertTrue(transition[0][index].any())
                self.assertFalse(env.game.defender_setup_phase.active)
                self.assertTrue(all(before != after for before, after in zip(initial, env.positions)))
                routes = env.route_controller._routes
                env.step(epsilon=0.0)
                self.assertEqual(env.game.battle_tick, 1)
                self.assertIs(env.route_controller._routes, routes)
                env.reset()
                self.assertTrue(env.game.defender_setup_phase.active)
                self.assertEqual(env.elapsed_ticks, 0)

    def test_setup_masks_reject_forbidden_cells_and_plants(self):
        env = self.make_env('A3')
        env.step(epsilon=0.0)
        actor = env.attackers[2]
        actor.pos = [22, 27]
        route = env.route_controller._routes[actor.name]
        route.set_stage(1, actor.pos, env.scenario.grid, goal=(8, 40), goal_index=0)
        state = {'chars': env.game.chars, 'grid': env.game.grid,
                 'defender_setup_active': True}
        _, mask = env.route_controller.policy_inputs(actor, state)
        from concon_v1.co1_attacker_common import CARDINAL_MOVES
        for action, (dr, dc) in enumerate(CARDINAL_MOVES):
            if not is_setup_position_allowed(actor.pos[0] + dr, actor.pos[1] + dc):
                self.assertFalse(mask[action])
        self.assertNotEqual(env.game.grid[22, 28], 1)
        self.assertFalse(is_setup_position_allowed(22, 28))
        self.assertFalse(mask[3])
        self.assertFalse(mask[ACTION_PLANT])
        self.assertTrue(mask[ACTION_WAIT])
        actor.pos = [8, 40]
        route.set_stage(2, actor.pos, env.scenario.grid, goal=(8, 40), goal_index=0)
        _, mask = env.route_controller.policy_inputs(actor, state)
        self.assertFalse(mask[ACTION_PLANT])
        self.assertTrue(mask[ACTION_WAIT])

    def test_setup_skips_disabled_abilities_and_contact_stops(self):
        env = self.make_env('A3')
        with patch.object(env.controller.fixed_smokes, 'choose', side_effect=AssertionError), \
                patch.object(env.controller.fixed_flashes, 'choose', side_effect=AssertionError), \
                patch.object(env.controller.fixed_recons, 'choose', side_effect=AssertionError), \
                patch('concon_v1.co1_learn_attacker.preplant_contact_action', side_effect=AssertionError):
            env.step(epsilon=0.0)
        self.assertTrue(all(env.policy_action_applied))


if __name__ == '__main__':
    unittest.main()
