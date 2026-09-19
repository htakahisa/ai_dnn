import unittest
from types import SimpleNamespace
from game_core import _apply_combo_bonus
from effect_text_generator import format_stat_bonus


def unit(sample=-0.10):
    return SimpleNamespace(condition_modifier=sample, sampled_condition_modifier=sample,
                           condition_bonus=0.0, form_variance=5.0, max_condition_delta=0.20,
                           base_accuracy_before_condition=0.80,
                           base_hs_rate_before_condition=0.50,
                           accuracy=0.80 * (1 + sample), hs_rate=0.50 * (1 + sample))


class ConditionBonusTests(unittest.TestCase):
    def test_add_to_draw_without_reroll(self):
        c = unit()
        self.assertTrue(_apply_combo_bonus(c, 'condition_bonus', 0.05))
        self.assertAlmostEqual(c.condition_modifier, -0.05)
        self.assertAlmostEqual(c.accuracy, 0.80 * 0.95)
        self.assertAlmostEqual(c.hs_rate, 0.50 * 0.95)
        self.assertEqual(c.sampled_condition_modifier, -0.10)

    def test_negative_and_zero_variance(self):
        c = unit(0)
        c.form_variance = c.max_condition_delta = 0
        _apply_combo_bonus(c, 'condition_bonus', -0.03)
        self.assertAlmostEqual(c.condition_modifier, -0.03)

    def test_other_bonuses_and_variance_order(self):
        a, b = unit(), unit()
        _apply_combo_bonus(a, 'accuracy', 0.10)
        _apply_combo_bonus(a, 'condition_bonus', 0.05)
        _apply_combo_bonus(a, 'form_variance', -2)
        _apply_combo_bonus(b, 'form_variance', -2)
        _apply_combo_bonus(b, 'condition_bonus', 0.05)
        _apply_combo_bonus(b, 'accuracy', 0.10)
        self.assertAlmostEqual(a.condition_modifier, -0.01)
        self.assertAlmostEqual(a.accuracy, b.accuracy)
        self.assertAlmostEqual(a.accuracy, 0.80 * 0.99 + 0.10)

    def test_stack_and_bounds(self):
        c = unit(0)
        _apply_combo_bonus(c, 'condition_bonus', 0.30)
        _apply_combo_bonus(c, 'condition_bonus', 0.30)
        self.assertEqual(c.condition_modifier, 0.40)
        self.assertFalse(_apply_combo_bonus(c, 'condition_bonus', float('nan')))

    def test_effect_text(self):
        self.assertEqual(format_stat_bonus('condition_bonus', 0.05), '調子補正 +5ポイント')


if __name__ == '__main__':
    unittest.main()
