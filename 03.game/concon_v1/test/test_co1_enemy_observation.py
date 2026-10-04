"""The route policy can distinguish perceived contact from an empty firing line."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1.co1_attacker_common import GORIGONS, SharedRouteDQN
    from concon_v1.co1_attacker_scenarios import get_scenario, validate_checkpoint_scenario
    from concon_v1.co1_learn_attacker import ConconAttackerRouteController
    from iq_perception import PerceivedCharacter, PerceivedGameView


class EnemyObservationTests(unittest.TestCase):
    def make_inputs(self, map_name='A3', *, known=True, visible=True, fireable=True):
        scenario = get_scenario(map_name)
        real_chars = [SimpleNamespace(name=name, team='A', is_alive=True,
                                     pos=list(pos), has_spike=i == 2, plant_timer=0)
                      for i, (name, pos) in enumerate(zip(GORIGONS.players, scenario.attacker_spawns))]
        real_chars.append(SimpleNamespace(name='enemy', team='D', is_alive=True,
                                         pos=[22, 25], hp=100, has_spike=False))
        chars = [PerceivedCharacter(c, pos=list(c.pos), position_known=True) for c in real_chars[:-1]]
        chars.append(PerceivedCharacter(real_chars[-1], pos=[22, 25] if known else [-999, -999],
                                        position_known=known))
        real_game = SimpleNamespace(check_line_of_sight=Mock(return_value=visible),
                                    check_shot_line_of_sight=Mock(return_value=fireable))
        view = PerceivedGameView(real_game, {'grid': scenario.grid, 'chars': chars},
                                 {id(real): proxy for real, proxy in zip(real_chars, chars)})
        controller = ConconAttackerRouteController(model=SharedRouteDQN(scenario.obs_dim),
                                                  map_name=scenario, seed=0)
        controller.set_game(view)
        state = {'grid': view.grid, 'chars': view.chars, 'battle_tick': 5}
        controller._prepare_route(chars[0], state)
        return scenario, controller, chars[0], state, real_game

    def test_contact_changes_only_the_new_observation_flags(self):
        for map_name in ('A1', 'A2', 'A3'):
            with self.subTest(map=map_name):
                scenario, controller, actor, state, game = self.make_inputs(map_name)
                contact, contact_mask = controller.policy_inputs(actor, state)
                game.check_line_of_sight.return_value = False
                game.check_shot_line_of_sight.return_value = False
                empty, empty_mask = controller.policy_inputs(actor, state)
                self.assertEqual(contact.shape, (scenario.obs_dim,))
                np.testing.assert_array_equal(contact[-2:], [1, 1])
                np.testing.assert_array_equal(empty[-2:], [0, 0])
                np.testing.assert_array_equal(contact[:-2], empty[:-2])
                np.testing.assert_array_equal(contact_mask, empty_mask)

    def test_visible_enemy_with_blocked_shot_has_a_distinct_input(self):
        _, controller, actor, state, _ = self.make_inputs(fireable=False)
        observation, _ = controller.policy_inputs(actor, state)
        np.testing.assert_array_equal(observation[-2:], [1, 0])

    def test_hidden_enemy_is_not_used_even_if_line_of_sight_stub_returns_true(self):
        _, controller, actor, state, game = self.make_inputs(known=False)
        observation, _ = controller.policy_inputs(actor, state)
        np.testing.assert_array_equal(observation[-2:], [0, 0])
        game.check_line_of_sight.assert_not_called()
        game.check_shot_line_of_sight.assert_not_called()

    def test_setup_has_no_enemy_contact_information(self):
        _, controller, actor, state, game = self.make_inputs()
        state['defender_setup_active'] = True
        observation, _ = controller.policy_inputs(actor, state)
        np.testing.assert_array_equal(observation[-2:], [0, 0])
        game.check_line_of_sight.assert_not_called()
        game.check_shot_line_of_sight.assert_not_called()

    def test_old_models_without_contact_inputs_are_rejected(self):
        for map_name in ('A1', 'A2', 'A3'):
            scenario = get_scenario(map_name)
            checkpoint = {'map_name': map_name, 'waypoint_order': scenario.waypoint_order,
                          'obs_dim': scenario.obs_dim - 2}
            with self.subTest(map=map_name), self.assertRaisesRegex(ValueError, 'retrain'):
                validate_checkpoint_scenario(checkpoint, scenario)

    def test_loading_frozen_weights_preserves_other_controllers_random_state(self):
        scenario = get_scenario('A3')
        model = SharedRouteDQN(scenario.obs_dim)
        checkpoint = {'map_name': 'A3', 'obs_dim': scenario.obs_dim, 'n_actions': 6,
                      'training_roster': GORIGONS.players, 'spike_carrier': GORIGONS.spike_holder,
                      'model_state_dict': model.state_dict()}
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        before = torch.get_rng_state().clone()
        ConconAttackerRouteController(checkpoint_bytes=buffer.getvalue(), map_name=scenario)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
