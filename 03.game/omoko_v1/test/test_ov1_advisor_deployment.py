"""Production ability routing, shared memory and mandatory dual-model validation."""

from pathlib import Path
import contextlib
import io
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from omoko_v1.ability_advisor_v1 import ov1_learn_attacker_ability_advisor as module


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name) / 'reference.pt'
        self.adopted = Path(self.temp.name) / 'flash_smoke.pt'
        self.context = NS(map_name='carry_L', signature='layout', grid=np.zeros((10, 10)))
        self.actor = NS(name='caster', pos=(2, 2), team='A', is_alive=True, facing='N',
                        smoke_charges=1, flash_charges=1, recon_charges=1)
        self.state = dict(grid=self.context.grid, chars=[self.actor], battle_tick=4)
        self.game = NS(_projectile_path=lambda start, aim: [start, aim])
        self.point = (2, 3)
        self.opponent = 'frc_v1'
        self.write(self.reference, -10.)
        self.write(self.adopted, 10.)

    def write(self, path, bias, policy='frozen'):
        model = module.AbilityJudge()
        for parameter in model.parameters():
            parameter.data.zero_()
        model.layers[-1].bias.data.fill_(bias)
        checkpoint = dict(version=module.VERSION, label_version=module.LABEL_VERSION,
                          wall_los_version=module.WALL_LOS_VERSION, feature_names=module.feature_names(),
                          skip_threshold=.1, model_state_dict=model.state_dict(),
                          scenarios={name:dict(signature='layout', carry_sha256=policy) for name in module.MAPS},
                          groups={module.group_key(self.context, self.opponent, ability, self.point):
                                  dict(validated=True) for ability in module.ABILITIES})
        torch.save(checkpoint, path)

    def judge(self):
        judge = module.DeployedAbilityAdvisor(self.reference, self.adopted)
        judge.bind_route(self.context, 'frozen')
        return judge

    def advice(self, judge, ability):
        return judge.advise(self.context, self.actor, self.state, self.opponent,
                            ability, self.point, self.point, self.game)

    def test_flash_and_smoke_use_adopted_weights_recon_uses_reference(self):
        judge = self.judge()
        self.assertTrue(self.advice(judge, 'FLASH').use)
        self.assertTrue(self.advice(judge, 'SMOKE').use)
        self.assertFalse(self.advice(judge, 'RECON').use)
        self.assertEqual(judge.model_path, self.reference)

    def test_routing_uses_the_models_in_both_directions(self):
        self.write(self.reference, 10.)
        self.write(self.adopted, -10.)
        judge = self.judge()
        self.assertFalse(self.advice(judge, 'FLASH').use)
        self.assertFalse(self.advice(judge, 'SMOKE').use)
        self.assertTrue(self.advice(judge, 'RECON').use)

    def test_both_models_must_be_present(self):
        absent = Path(self.temp.name) / 'missing.pt'
        for reference, adopted in ((absent, self.adopted), (self.reference, absent)):
            with self.subTest(reference=reference), self.assertRaises(module.AbilityAdvisorError):
                module.DeployedAbilityAdvisor(reference, adopted)

    def test_binding_validates_both_models_and_all_routes(self):
        judge = self.judge()
        for name in module.MAPS:
            judge.bind_route(NS(map_name=name, signature='layout'), 'frozen')
        self.assertEqual(set(judge.route_hashes), set(module.MAPS))
        self.assertEqual(set(judge.flash_smoke_advisor.route_hashes), set(module.MAPS))
        self.write(self.adopted, 10., policy='different')
        with self.assertRaisesRegex(module.AbilityAdvisorError, 'policy_mismatch'):
            self.judge()

    def test_sources_share_observations_usage_history_and_decision_counts(self):
        judge = self.judge()
        judge.start_round('carry_L', 1)
        judge.observe(self.context, self.actor, self.state, self.opponent)
        memory = judge.memory
        self.assertTrue(self.advice(judge, 'SMOKE').use)
        self.actor.smoke_charges = 0
        self.state['battle_tick'] += 1
        judge.observe(self.context, self.actor, self.state, self.opponent)
        self.assertIn(('SMOKE', self.point), judge.memory.current['uses'])
        self.assertFalse(self.advice(judge, 'RECON').use)
        self.assertIs(judge.memory, memory)
        self.assertEqual(judge.decisions, dict(use=1, save=1))
        judge.start_round('carry_L', 2)
        self.assertIn(('SMOKE', self.point), judge.memory.rounds[-1]['uses'])

    def test_source_is_restored_after_inference_failure(self):
        judge = self.judge()
        judge.flash_smoke_bundle[0].layers[-1].bias.data.fill_(float('nan'))
        with self.assertRaisesRegex(module.AbilityAdvisorError, 'invalid_model_score'):
            self.advice(judge, 'FLASH')
        self.assertIs(judge.model, judge.recon_bundle[0])
        self.assertEqual(judge.model_path, self.reference)

    def test_controller_default_constructs_the_production_combination(self):
        from omoko_v1 import ov1_attacker_controller as controller
        disabled = module.FixedAbilityAdvisor(None, enabled=False)
        with patch.object(controller, 'DeployedAbilityAdvisor', return_value=disabled) as factory, \
             patch.object(controller, 'policy_digest', return_value='frozen'), \
             patch.object(controller, 'Ov1LearningAttackerCarryController', return_value=NS()), \
             patch.object(controller, 'Ov1LearningAttackerEscortController', return_value=NS()), \
             patch.object(controller, 'Ov1LearningAttackerRetrieveController', return_value=NS()), \
             patch.object(controller, 'Ov1LearningAttackerGuardController', return_value=NS()):
            result = controller.Ov1AttackerController()
        factory.assert_called_once_with()
        self.assertIs(result.ability_advisor, disabled)

    def test_evaluator_keeps_reference_candidate_and_deployed_arms_distinct(self):
        from omoko_v1.ability_advisor_v1 import ov1_evaluate_attacker_ability_advisor as evaluation
        seen = []

        def fake_match(opponent, seed, rounds, judge, maps, **kwargs):
            seen.append((type(judge), judge.enabled, judge.model_path))
            return dict(rounds=1, attacker_wins=1, charges_at_plant=[], decisions=judge.decisions)

        task = ('frc_v1', 0, self.reference, module.sha256_file(self.reference), 1, 1,
                module.MAPS, 0, self.adopted, module.sha256_file(self.adopted),
                self.adopted, module.sha256_file(self.adopted))
        with patch.object(evaluation, 'play_match', side_effect=fake_match), \
             contextlib.redirect_stdout(io.StringIO()):
            _, report = evaluation.evaluate_opponent(task)
        self.assertEqual(len(seen), 4)
        self.assertFalse(seen[0][1])
        self.assertEqual(seen[1][2], self.reference)
        self.assertEqual(seen[2][2], self.adopted)
        self.assertIs(seen[3][0], module.DeployedAbilityAdvisor)
        self.assertIn('deployed_vs_advised', report)
        self.assertIn('candidate_vs_deployed', report)


if __name__ == '__main__':
    unittest.main()
