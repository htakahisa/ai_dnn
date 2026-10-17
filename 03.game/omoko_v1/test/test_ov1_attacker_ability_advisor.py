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
from omoko_v1 import ov1_map_data_escort as escort_map
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

    def test_validated_model_saves_and_changed_policy_stops(self):
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
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'policy_mismatch'):
                judge.bind_route(context, 'different')

    def test_missing_and_incompatible_models_stop(self):
        context, actor, state, game = self.fixture()
        for path in (None, Path('missing_adviser.pt')):
            with self.subTest(path=path), self.assertRaisesRegex(advisor.AbilityAdvisorError, 'not found'):
                advisor.FixedAbilityAdvisor(path)
        checkpoint = self.checkpoint(context, advisor.OPPONENTS[0], (8, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            checkpoint['version'] = 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'load failed'):
                advisor.FixedAbilityAdvisor(path)
            path.write_bytes(b'broken checkpoint')
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'load failed'):
                advisor.FixedAbilityAdvisor(path)

    def test_disabled_advisor_retains_evaluation_baseline(self):
        context, actor, state, game = self.fixture()
        judge = advisor.FixedAbilityAdvisor(None, enabled=False)
        self.assertTrue(judge.advise(context, actor, state, advisor.OPPONENTS[0],
                                     'SMOKE', (8, 3), (8, 3), game).use)
        self.assertEqual(judge.last_advice.reason, 'advisor_disabled')

    def test_compatible_model_can_choose_precontact_use(self):
        context, actor, state, game = self.fixture()
        point, opponent = (8, 3), advisor.OPPONENTS[0]
        checkpoint = self.checkpoint(context, opponent, point)
        checkpoint['model_state_dict']['layers.4.bias'].fill_(10.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(checkpoint, path)
            judge = advisor.FixedAbilityAdvisor(path)
            judge.bind_route(context, 'frozen')
            self.assertTrue(judge.advise(context, actor, state, opponent,
                                         'SMOKE', point, point, game).use)
            self.assertEqual(judge.last_advice.reason, 'model')

    def test_per_group_threshold_is_used_and_invalid_threshold_stops(self):
        context, actor, state, game = self.fixture()
        opponent, point = advisor.OPPONENTS[0], (8, 3)
        checkpoint = self.checkpoint(context, opponent, point)
        checkpoint['model_state_dict']['layers.4.bias'].zero_()  # score=0.5
        key = advisor.group_key(context, opponent, 'SMOKE', point)
        checkpoint['groups'][key]['skip_threshold'] = .8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(checkpoint, path)
            judge = advisor.FixedAbilityAdvisor(path)
            judge.bind_route(context, 'frozen')
            self.assertFalse(judge.advise(context, actor, state, opponent,
                                         'SMOKE', point, point, game).use)
            checkpoint['groups'][key]['skip_threshold'] = float('nan')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'load failed'):
                advisor.FixedAbilityAdvisor(path)

    def test_scenario_mismatch_stops_at_binding_before_enemy_contact(self):
        context, actor, state, game = self.fixture()
        point, opponent = (8, 3), advisor.OPPONENTS[0]
        checkpoint = self.checkpoint(context, opponent, point)
        checkpoint['scenarios'][context.map_name]['signature'] = 'old'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(checkpoint, path)
            judge = advisor.FixedAbilityAdvisor(path)
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'scenario_mismatch'):
                judge.bind_route(context, 'frozen')
            state['chars'].append(NS(name='seen', team='D', is_alive=True,
                                     pos=point, facing='W', position_known=True))
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'scenario_mismatch'):
                judge.advise(context, actor, state, opponent, 'SMOKE', point, point, game)

    def test_unvalidated_group_uses_ability_and_invalid_score_stops(self):
        context, actor, state, game = self.fixture()
        point, opponent = (8, 3), advisor.OPPONENTS[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(self.checkpoint(context, opponent, point), path)
            judge = advisor.FixedAbilityAdvisor(path)
            judge.bind_route(context, 'frozen')
            advice = judge.advise(context, actor, state, opponent, 'SMOKE', (8, 4), (8, 4), game)
            self.assertTrue(advice.use)
            self.assertEqual(advice.reason, 'unvalidated_use')
            self.assertIn(actor.name, judge.pending)
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'unknown opponent'):
                judge.advise(context, actor, state, None, 'SMOKE', point, point, game)
            judge.model.layers[-1].bias.data.fill_(float('nan'))
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'invalid_model_score'):
                judge.advise(context, actor, state, opponent, 'SMOKE', point, point, game)
            self.assertEqual(judge.decisions, {'use': 1, 'save': 0})

    def test_model_without_validated_groups_uses_unvalidated_abilities(self):
        context, actor, state, game = self.fixture()
        checkpoint = self.checkpoint(context, advisor.OPPONENTS[0], (8, 3))
        checkpoint['groups'] = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'advisor.pt'
            torch.save(checkpoint, path)
            judge = advisor.FixedAbilityAdvisor(path)
            judge.bind_route(context, 'frozen')
            advice = judge.advise(context, actor, state, advisor.OPPONENTS[0],
                                  'SMOKE', (8, 3), (8, 3), game)
            self.assertTrue(advice.use)
            self.assertEqual(advice.reason, 'unvalidated_use')

    def test_defender_start_checks_adviser_before_loading_defender_models(self):
        import ov1_defender_controller as defender
        with patch.object(defender, 'preflight_default_advisor',
                          side_effect=advisor.AbilityAdvisorError('adviser startup failure')) as check, \
                patch.object(defender, 'Ov1LearningDefenderSearchController') as search:
            with self.assertRaisesRegex(advisor.AbilityAdvisorError, 'adviser startup failure'):
                defender.Ov1DefenderController(device='cpu')
            check.assert_called_once_with()
            search.assert_not_called()

    def test_defender_preflight_checks_all_four_attacker_routes(self):
        from omoko_v1.ability_advisor_v1 import ov1_advisor_attacker_context as context
        judge = advisor.FixedAbilityAdvisor(None, enabled=False)
        with patch.object(advisor, 'DeployedAbilityAdvisor', return_value=judge), \
                patch.object(context, 'policy_digest', return_value='frozen'):
            context.preflight_default_advisor()
        self.assertEqual(judge.route_hashes, dict.fromkeys(advisor.MAPS, 'frozen'))

    def test_signature_ignores_python_source_text_but_tracks_layout(self):
        before = scenario('carry_L')
        # Includes guard map edits and carry/escort inference edits. No raw
        # Python source file may be consulted for the geometry contract.
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('source hash')):
            self.assertEqual(scenario('carry_L').signature, before.signature)
        with patch.object(escort_map, 'NEW_MAZE_STR', escort_map.NEW_MAZE_STR.replace('S', 'R', 1)):
            self.assertNotEqual(scenario('carry_L').signature, before.signature)
        with patch.object(escort_map, 'NEW_MAZE_STR', '\n' + escort_map.NEW_MAZE_STR + '\n'):
            self.assertEqual(scenario('carry_L').signature, before.signature)

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
        judge = advisor.FixedAbilityAdvisor(None, enabled=False)
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
        for points in ([(5, 6)], []):
            with self.subTest(points=points):
                ctrl.ability_lineup_points = {'RECON': points}
                availability.clear()
                result = ctrl.decide_move(actor, dict(chars=[actor]))
                self.assertEqual(availability, [False])
                self.assertNotEqual(tuple(result[0]), actor.pos)
                self.assertNotIn('ability', result[1])

    def test_controller_and_collector_bind_same_frozen_policies(self):
        ctrl = Ov1AttackerController(device='cpu', ability_advisor=advisor.FixedAbilityAdvisor(None, enabled=False))
        metadata = route_metadata()
        self.assertEqual(set(metadata), set(advisor.MAPS))
        for name, contract in metadata.items():
            self.assertEqual(contract['carry_sha256'], ctrl.ability_advisor.route_hashes[name])
            self.assertEqual(contract['signature'], ctrl._advisor_scenarios[name].signature)


if __name__ == '__main__':
    unittest.main()
