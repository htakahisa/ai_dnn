"""Advice contracts and fixed-target integration without training or matches."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omoko_v1.ability_advisor_v1 import ov1_learn_attacker_ability_advisor as advisor
from omoko_v1.ability_advisor_v1.ov1_collect_attacker_ability_advisor import route_metadata
from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import scenario
import ov1_learning_attacker_escort as escort
import ov1_learning_attacker_carry as carry
from ov1_attacker_controller import Ov1AttackerController


class AbilityAdviceTests(unittest.TestCase):
    def fixture(self):
        context = scenario('carry_L')
        actor = NS(name='a', team='A', is_alive=True, pos=(8, 2), facing='N',
                   smoke_charges=1, has_spike=True)
        state = dict(grid=context.grid, chars=[actor], battle_tick=30)
        game = NS(_projectile_path=lambda start, aim: [start, aim])
        return context, actor, state, game

    def checkpoint(self, context, opponent, point):
        model = advisor.AbilityJudge()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.layers[-1].bias.data.fill_(-10.)
        return dict(version=advisor.VERSION, label_version=advisor.LABEL_VERSION,
                    wall_los_version=advisor.WALL_LOS_VERSION,
                    feature_names=advisor.feature_names(), skip_threshold=.1,
                    scenarios={context.map_name: dict(signature=context.signature, carry_sha256='frozen')},
                    groups={advisor.group_key(context, opponent, 'SMOKE', point): dict(validated=True)},
                    model_state_dict=model.state_dict())

    def test_validated_model_saves_and_changed_policy_uses(self):
        context, actor, state, game = self.fixture()
        point, opponent = (8, 3), advisor.OPPONENTS[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(self.checkpoint(context, opponent, point), path)
            judge = advisor.FixedAbilityAdvisor(path)
            judge.bind_route(context, 'frozen')
            judge.observe(context, actor, state, opponent)
            self.assertFalse(judge.advise(context, actor, state, opponent, 'SMOKE', point, point, game).use)
            self.assertNotIn(actor.name, judge.pending)
            state['chars'].append(NS(name='seen', team='D', is_alive=True,
                                     pos=point, facing='W', position_known=True))
            self.assertTrue(judge.advise(context, actor, state, opponent, 'SMOKE', point, point, game).use)
            state['chars'].pop()
            judge.bind_route(context, 'different')
            self.assertTrue(judge.advise(context, actor, state, opponent, 'SMOKE', point, point, game).use)

    def test_missing_and_concon_models_preserve_fixed_use(self):
        context, actor, state, game = self.fixture()
        judge = advisor.FixedAbilityAdvisor(None)
        judge.observe(context, actor, state, advisor.OPPONENTS[0])
        self.assertTrue(judge.advise(context, actor, state, advisor.OPPONENTS[0],
                                     'SMOKE', (8, 3), (8, 3), game).use)
        checkpoint = self.checkpoint(context, advisor.OPPONENTS[0], (8, 3))
        checkpoint['version'] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'concon.pt'
            torch.save(checkpoint, path)
            with self.assertWarns(UserWarning):
                self.assertIsNone(advisor.FixedAbilityAdvisor(path).model)

    def test_hidden_enemy_cannot_change_features_or_memory(self):
        context, actor, state, game = self.fixture()
        memory = advisor.MatchMemory()
        memory.observe(context, actor, state, 'concon_v1')
        before, _ = advisor.build_features(context, actor, state, 'concon_v1', 'SMOKE', (8, 3), (8, 3), memory)
        state['chars'].append(NS(name='hidden', team='D', is_alive=True,
                                 pos=(8, 3), facing='W', position_known=False))
        memory.observe(context, actor, state, 'concon_v1')
        after, _ = advisor.build_features(context, actor, state, 'concon_v1', 'SMOKE', (8, 3), (8, 3), memory)
        np.testing.assert_array_equal(before, after)
        self.assertFalse(memory.current['sightings'])

    def test_use_history_requires_actual_charge_decrease(self):
        context, actor, state, _ = self.fixture()
        judge = advisor.FixedAbilityAdvisor(None)
        judge.observe(context, actor, state, 'concon_v1')
        judge.pending[actor.name] = ('SMOKE', (8, 3), 1, 30)
        judge.observe(context, actor, state, 'concon_v1')
        self.assertFalse(judge.memory.current['uses'])
        actor.smoke_charges = 0
        judge.observe(context, actor, state, 'concon_v1')
        self.assertIn(('SMOKE', (8, 3)), judge.memory.current['uses'])

    def test_escort_skipped_point_is_reconsidered_and_other_point_available(self):
        ctrl = escort.Ov1LearningAttackerEscortController.__new__(escort.Ov1LearningAttackerEscortController)
        ctrl._lineup_used_cells = {'SMOKE': set()}
        grid = np.zeros((12, 12), dtype=int)
        with patch.object(escort, 'ABILITY_LINEUP_POINTS', {'SMOKE': [(5, 5), (5, 6)]}):
            ctrl.lineup_advice = lambda ability, point: point == (5, 6)
            self.assertEqual(ctrl._available_lineup_cell('SMOKE', grid, (5, 4), (5, 4)), (5, 6))
            self.assertFalse(ctrl._lineup_used_cells['SMOKE'])
            ctrl.lineup_advice = lambda ability, point: True
            self.assertEqual(ctrl._available_lineup_cell('SMOKE', grid, (5, 4), (5, 4)), (5, 5))

    def test_carry_saving_all_fixed_points_masks_ability_and_keeps_moving(self):
        ctrl = carry.Ov1LearningAttackerCarryController.__new__(carry.Ov1LearningAttackerCarryController)
        grid = np.zeros((12, 12), dtype=int)
        ctrl.game = NS(grid=grid, battle_tick=10)
        ctrl._carrier_name = None
        ctrl._active_target = (5, 8)
        ctrl._reached_waypoint = True
        ctrl._sighting = None
        ctrl._smoke_cells = lambda: set()
        ctrl._plantable_cell = lambda pos: False
        ctrl._get_target_dist_map = lambda target: np.zeros_like(grid)
        ctrl._update_sighting = lambda *args: None
        ctrl._build_observation = lambda *args, **kwargs: []
        availability = []
        ctrl._build_mask = lambda char, on_target, available, **kwargs: availability.append(available)
        ctrl._select_action = lambda obs, mask: 0
        ctrl._decode_action = lambda action: ('MOVE', 'E')
        ctrl.lineup_advice = lambda ability, point: False
        ctrl.ability_lineup_points = {'RECON': [(5, 6)]}
        actor = NS(name='carrier', team='A', pos=(5, 4), has_spike=True,
                   ability_name='RECON', is_alive=True, facing='N')
        result = ctrl.decide_move(actor, dict(chars=[actor]))
        self.assertEqual(availability, [False])
        self.assertNotEqual(tuple(result[0]), actor.pos)
        self.assertNotIn('ability', result[1])

    def test_controller_and_collector_bind_same_frozen_policies(self):
        ctrl = Ov1AttackerController(device='cpu', ability_advisor=advisor.FixedAbilityAdvisor(None))
        metadata = route_metadata()
        self.assertEqual(set(metadata), set(advisor.MAPS))
        for name, contract in metadata.items():
            self.assertEqual(contract['carry_sha256'], ctrl.ability_advisor.route_hashes[name])
            self.assertEqual(contract['signature'], ctrl._advisor_scenarios[name].signature)


if __name__ == '__main__':
    unittest.main()
