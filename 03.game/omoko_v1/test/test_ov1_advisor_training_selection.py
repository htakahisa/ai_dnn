"""Decision retention, validation selection and measured consumption, without training."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from omoko_v1.ability_advisor_v1 import ov1_train_attacker_ability_advisor as training
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_usage import AbilityUsageMonitor
from omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_advisor import summarize, compare_summaries


def row(round_number=1, **changes):
    value = dict(match_id='match', round_number=round_number, group='point',
                 opponent='frc_v1', tick=1, caster='one', label=0,
                 features=[0.] * len(training.feature_names()))
    value.update(changes)
    return value


class TrainingSelectionTests(unittest.TestCase):
    def test_keep_decisions_at_different_times_and_for_different_casters(self):
        original = row()
        inputs = [original, dict(original), row(tick=2), row(caster='two'), row(label=1)]
        self.assertEqual(len(training.deduplicate_samples(inputs)), 4)

    def test_preserved_sequences_have_equal_total_training_weight(self):
        weights = training.round_weights([row(tick=i) for i in range(4)] + [row(2)])
        self.assertAlmostEqual(float(weights[:4].sum()), float(weights[4]), places=5)

    def test_report_counts_early_permission_even_when_last_request_saves(self):
        report = training.classify_report([row(tick=1), row(tick=2)], [.8, .01], .1)['point']
        self.assertEqual(report['rounds'], 1)
        self.assertEqual(report['unnecessary_use_rounds'], 1)
        self.assertEqual(report['unnecessary_allowed'], 1)

    def test_many_requests_cannot_fabricate_round_support(self):
        rows = [row(tick=i) for i in range(100)]
        group = training.select_groups(rows, [.1] * len(rows))['point']
        self.assertFalse(group['validated'])

    def test_many_saves_in_one_round_cannot_fabricate_saved_round_support(self):
        rows = [row(r) for r in range(20)] + [row(0, tick=t) for t in range(2, 20)]
        report = training.classify_report(rows, [.99] * 20 + [.01] * 18, .1)['point']
        self.assertGreater(report['skipped'], 5)
        self.assertEqual(report['skipped_rounds'], 1)
        self.assertFalse(training.safe_group(report, 20, .05))

    def test_selection_saves_unnecessary_early_requests(self):
        rows = [row(r, tick=t) for r in range(20) for t in (1, 2)]
        group = training.select_groups(rows, [.8, .01] * 20)['point']
        self.assertTrue(group['validated'])
        report = training.operating_report(rows, [.8, .01] * 20, {'point':group})['totals']
        self.assertEqual(report['unnecessary_allowed'], 0)
        self.assertEqual(report['unnecessary_use_rounds'], 0)

    def test_selection_preserves_hidden_positive_candidates(self):
        rows = [row(r, tick=t, label=int(t == 2)) for r in range(20) for t in (1, 2)]
        group = training.select_groups(rows, [.4, .8] * 20)['point']
        report = training.operating_report(rows, [.4, .8] * 20, {'point':group})['totals']
        self.assertTrue(group['validated'])
        self.assertEqual(report['missed_positives'], 0)
        self.assertEqual(report['unnecessary_allowed'], 0)

    def test_direct_evidence_always_overrides_selected_threshold(self):
        value = row()
        value['features'][training.feature_names().index('direct_relevance')] = 1.
        report = training.operating_report([value], [.01],
                                          {'point':dict(validated=True, skip_threshold=.99)})
        self.assertEqual(report['totals']['unnecessary_allowed'], 1)

    def test_actual_operating_report_includes_use_all_fallback(self):
        report = training.operating_report([row()], [.01], {})
        self.assertEqual(report['totals']['unnecessary_allowed'], 1)


class UsageMonitorTests(unittest.TestCase):
    def test_comparison_aggregates_actual_consumption_instead_of_advice_counts(self):
        def outcome(casts):
            return dict(rounds=1, attacker_wins=1, charges_at_plant=[],
                        decisions=dict(use=100, save=0),
                        actual_usage={'carry|FLASH':dict(casts=casts,
                                                       teacher_unnecessary_casts=casts,
                                                       zero_enemy_effect_projectiles=casts)})
        before = summarize([outcome(3), outcome(2)])
        after = summarize([outcome(1), outcome(1)])
        comparison = compare_summaries(before, after)['actual_usage']['carry|FLASH']['casts']
        self.assertEqual(comparison['reduction'], 3)
        self.assertEqual(comparison['reduction_rate'], .6)

    def fixture(self):
        actor = NS(name='one', team='A', pos=(0, 0), has_spike=True, is_alive=True, flash_charges=1)
        game = NS(grid=np.zeros((3, 3)), chars=[actor], is_planted=False,
                  current_round=1, battle_tick=1, flash_projectiles=[], recon_projectiles=[],
                  _projectile_path=lambda start, target: [start, target], _smoke_cells=lambda: (),
                  _explode_flash=lambda projectile, impact=None: None,
                  _explode_recon=lambda projectile, impact=None: None)

        def execute(owner, action):
            if owner.flash_charges == 0:
                return False
            owner.flash_charges -= 1
            game.flash_projectiles.append(dict(path=[(0, 0), (1, 1)], progress=1, team='A'))
            return True

        game.execute_ai_ability = execute
        monitor = AbilityUsageMonitor(NS(pending={'one':('FLASH', (1, 1), 1, 1)}))
        monitor.observe_tick(game, 'before')
        return actor, game, monitor

    def test_rejected_requests_do_not_count_as_consumption(self):
        actor, game, monitor = self.fixture()
        game.execute_ai_ability(actor, dict(ability='FLASH', target=(1, 1)))
        game.execute_ai_ability(actor, dict(ability='FLASH', target=(1, 1)))
        self.assertEqual(monitor.report()['all']['casts'], 1)
        self.assertEqual(monitor.report()['advised_fixed']['casts'], 1)
        self.assertEqual(monitor.report()['all']['unresolved_projectiles'], 1)

    def test_launch_teacher_and_resolution_effect_are_separate(self):
        actor, game, monitor = self.fixture()
        target = 'omoko_v1.ability_advisor_v1.ov1_evaluate_attacker_ability_usage.relevant_enemy_positions'
        with patch(target, side_effect=[[], [(1, 2)]]):
            game.execute_ai_ability(actor, dict(ability='FLASH', target=(1, 1)))
            game._explode_flash(game.flash_projectiles[0])
        stats = monitor.report()['all']
        self.assertEqual(stats['teacher_unnecessary_casts'], 1)
        self.assertEqual(stats['resolved_projectiles'], 1)
        self.assertEqual(stats['zero_enemy_effect_projectiles'], 0)


if __name__ == '__main__':
    unittest.main()
