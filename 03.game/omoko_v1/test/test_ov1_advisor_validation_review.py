"""Check report reconstruction and the real unvalidated-use fallback."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omoko_v1.ability_advisor_v1.ov1_review_attacker_advisor_validation import (
    counts, operational_totals, passes_extra_gate,
)
from omoko_v1.ability_advisor_v1.ov1_review_attacker_advisor_scores import retry_report


class ValidationReviewTests(unittest.TestCase):
    def test_last_request_hides_an_earlier_unnecessary_permission(self):
        row = dict(group='point', match_id='match', round_number=1, label=0)
        report = retry_report([row, row], [True, False])['point']
        self.assertEqual(report['all_negative_rounds_last_allowed'], 0)
        self.assertEqual(report['all_negative_rounds_any_allowed'], 1)

    def test_changing_labels_are_not_called_an_all_negative_round(self):
        row = dict(group='point', match_id='match', round_number=1, label=0)
        report = retry_report([row, dict(row, label=1)], [True, False])['point']
        self.assertEqual(report['all_negative_rounds'], 0)
        self.assertEqual(report['rounds_with_changing_labels'], 1)

    def test_confusion_matrix_separates_missed_positives_from_saved_negatives(self):
        matrix = counts(dict(samples=100, positives=20, skipped=62, missed_positives=2))
        self.assertEqual(matrix['saved_negatives'], 60)
        self.assertEqual(matrix['unnecessary_allowed'], 20)
        self.assertEqual(matrix['unnecessary_use_rate'], .25)

    def test_both_folds_need_negative_evidence(self):
        group = dict(validated=True,
                     validation=dict(samples=10, positives=0, skipped=10),
                     test=dict(samples=10, positives=0, skipped=8))
        self.assertFalse(passes_extra_gate(group, .1))
        self.assertTrue(passes_extra_gate(group, .2))
        group['test'] = dict(samples=10, positives=10, skipped=0)
        self.assertFalse(passes_extra_gate(group, .2))

    def test_rejection_restores_usage_for_saved_negatives_and_positives(self):
        groups = {'point': dict(test=dict(samples=100, positives=20,
                                         skipped=62, missed_positives=2))}
        before = operational_totals(groups, {'point': True})
        after = operational_totals(groups, {'point': False})
        self.assertEqual((before['unnecessary_allowed'], before['necessary_saved']), (20, 2))
        self.assertEqual((after['unnecessary_allowed'], after['necessary_saved']), (80, 0))


if __name__ == '__main__':
    unittest.main()
