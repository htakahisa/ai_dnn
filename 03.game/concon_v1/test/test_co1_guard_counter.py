"""Utility pressure, safe movement, defuse priority and old weight migration."""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from concon_v1.co1_guard_common import (
    GuardDQN, LEGACY_FEATURE_DIM, MAP_CHANNELS, PRE_COUNTER_ACTION_DIM,
    SELF_SMOKE_ACTION, WAIT_ACTION, decode_action, threat_exposure,
)
from concon_v1.co1_guard_scenarios import get_scenario
from concon_v1.co1_guard_rewards import decision_reward
from concon_v1.co1_guard_battle_training import GuardBattleEnv, OPPONENTS, START_MODES
from concon_v1.co1_learn_guard import ConconGuardController
from concon_v1.co1_train_guard import make_checkpoint


class GuardCounterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def context(self, **changes):
        result = dict(position=(2, 2), goal=(2, 2), aim=(2, 5), spike=(2, 5),
                      distance_goal=0, distance_spike=3, fireable=True, target=(2, 5),
                      tap=False, stopped=2, can_move=True, impaired=True, blind=True,
                      tap_remaining=6, exposures={(2, 2): 1, (1, 2): 0, (2, 3): 1})
        result.update(changes)
        return result

    def test_blind_cover_move_beats_exposed_hold_and_counter_is_available(self):
        ctx = self.context()
        hold = decision_reward(WAIT_ACTION, ctx, (2, 2), 'E', 0)
        cover = decision_reward(0, ctx, (1, 2), 'E', 1)
        cast = decision_reward(64, ctx, (2, 2), 'E', 0)
        self.assertGreater(cover, hold)
        self.assertGreater(cast, hold)

    def test_cover_is_not_rewarded_over_urgent_blind_defuse_contest(self):
        ctx = self.context(tap=True, tap_remaining=2)
        retreat = decision_reward(0, ctx, (1, 2), 'E', 4)
        approach = decision_reward(24, ctx, (2, 3), 'E', 2)
        self.assertGreater(approach, retreat)

    def test_recon_and_smoke_vision_ignore_smoke_but_walls_stop_lanes(self):
        grid = np.zeros((5, 7), dtype=int)
        threats = [((2, 5), False)]
        smoke = {(2, 3)}
        self.assertEqual(threat_exposure((2, 1), threats, grid, smoke), 0)
        self.assertEqual(threat_exposure((2, 1), threats, grid, smoke, True), 1)
        self.assertEqual(threat_exposure((2, 1), [((2, 5), True)], grid, smoke), 1)
        grid[2, 4] = 1
        self.assertEqual(threat_exposure((2, 1), threats, grid, smoke, True), 0)

    def test_old_observation_and_outputs_load_with_zero_new_input_weights(self):
        scenario = get_scenario('L')
        model = GuardDQN(scenario)
        checkpoint = make_checkpoint(model, scenario, 1, OPPONENTS, START_MODES)
        checkpoint['obs_dim'] = MAP_CHANNELS * scenario.grid.size + LEGACY_FEATURE_DIM
        checkpoint['n_actions'] = PRE_COUNTER_ACTION_DIM
        old = checkpoint['model_state_dict']
        old['head.0.weight'] = old['head.0.weight'][:, :-12]
        old['head.2.weight'] = old['head.2.weight'][:PRE_COUNTER_ACTION_DIM]
        old['head.2.bias'] = old['head.2.bias'][:PRE_COUNTER_ACTION_DIM]
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        controller = ConconGuardController(checkpoint_bytes=buffer.getvalue())
        self.assertTrue(torch.equal(controller.model.head[0].weight[:, :-12], old['head.0.weight']))
        self.assertEqual(controller.model.head[0].weight[:, -12:].count_nonzero(), 0)
        self.assertTrue(torch.equal(controller.model.head[2].weight[:PRE_COUNTER_ACTION_DIM], old['head.2.weight']))

    def test_smoke_alone_does_not_trigger_evasion_and_self_smoke_executes(self):
        env = GuardBattleEnv(opponents=['omoko_v1'])
        env.reset('hold', attacker_count=5, defender_count=5)
        char = next(char for char in env.attackers if char.ability_name == 'SMOKE')
        char.smoke_charges = 1
        state = dict(grid=env.game.grid, chars=env.attackers, is_planted=True,
                     planted_pos=env.game.planted_pos, smoke_cells={tuple(char.pos)}, battle_tick=0)
        _, mask, ctx = env.controller.policy_inputs(char, state)
        self.assertFalse(ctx['impaired'])
        self.assertTrue(mask[SELF_SMOKE_ACTION:].any())
        action = int(np.flatnonzero(mask[SELF_SMOKE_ACTION:])[0]) + SELF_SMOKE_ACTION
        pos, payload = decode_action(action, tuple(char.pos), ctx['targets'])
        self.assertEqual(pos, char.pos)
        self.assertTrue(env.game.execute_ai_ability(char, payload))
        self.assertEqual(char.smoke_charges, 0)

    def test_blind_inputs_keep_disclosed_memory_and_hide_unknown_live_positions(self):
        # Use actual roster actors and production-shaped inputs.
        env = GuardBattleEnv(opponents=['omoko_v1'])
        env.reset('hold', attacker_count=5, defender_count=5)
        char = env.attackers[0]
        char.blind_remaining = 3
        enemy = SimpleNamespace(name='hidden', team='D', is_alive=True,
                                position_known=False, pos=[-1, -1])
        remembered = tuple(env.defenders[0].pos)
        env.controller.sightings[enemy.name] = (remembered, 0)
        state = dict(grid=env.game.grid, chars=[char, enemy], is_planted=True,
                     planted_pos=env.game.planted_pos, battle_tick=1)
        obs, _, ctx = env.controller.policy_inputs(char, state)
        self.assertTrue(ctx['impaired'])
        self.assertEqual(ctx['targets'][2], remembered)
        enemy.pos = [0, 0]
        obs2, _, _ = env.controller.policy_inputs(char, state)
        np.testing.assert_array_equal(obs, obs2)

    def test_pressure_snapshots_use_all_six_opponents_and_record_effects(self):
        self.assertEqual(set(OPPONENTS), {'omoko_v1', 'touyama_v2', 'gc_v1',
                                         'frc_v1', 'fnatic_v3', 'toru_ai_v3.1'})
        for site in ('L', 'R'):
            for mode in ('pressure', 'pressure_tap'):
                env = GuardBattleEnv(map_name=site, opponents=['omoko_v1'])
                env.reset(mode, attacker_count=5, defender_count=5)
                self.assertTrue(all(any(getattr(char, field) > 0 for field in
                                       ('blind_remaining', 'reveal_remaining', 'electric_remaining'))
                                    for char in env.attackers))
                env.step()
                self.assertGreater(env.metrics['impaired_decisions'], 0)

    def test_blindness_bypasses_quiet_holding_values_and_allows_learned_escape(self):
        model = GuardDQN(get_scenario('L'), navigation=True)
        env = GuardBattleEnv(model=model, opponents=['omoko_v1'])
        env.reset('hold', attacker_count=5, defender_count=5)
        char = env.attackers[0]
        state = dict(grid=env.game.grid, chars=env.attackers, is_planted=True,
                     planted_pos=env.game.planted_pos, battle_tick=0)
        _, mask, _ = env.controller.policy_inputs(char, state)
        move = int(np.flatnonzero(mask[:32])[0])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.head[2].bias[move] = 100
            model.navigation_values.weight[:, 32:40] = 200
        destination, _ = env.controller.decide_move(char, state)
        self.assertEqual(destination, char.pos)
        char.blind_remaining = 3
        destination, _ = env.controller.decide_move(char, state)
        self.assertNotEqual(destination, char.pos)


if __name__ == '__main__':
    unittest.main()
