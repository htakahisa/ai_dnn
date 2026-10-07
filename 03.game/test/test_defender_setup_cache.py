"""Setup movement checks reuse map validation and still respect edited maps."""

import unittest
from unittest.mock import patch

import map_data_defender_setup as setup_map


class DefenderSetupCacheTests(unittest.TestCase):
    def setUp(self):
        setup_map._setup_rows.cache_clear()
        self.addCleanup(setup_map._setup_rows.cache_clear)

    def test_repeated_movement_checks_validate_map_only_once(self):
        with patch.object(setup_map, 'DEFENDER_SETUP_MASK_STR', '01\n10'), \
                patch.object(setup_map, '_rows', wraps=setup_map._rows) as validate:
            for _ in range(100):
                self.assertTrue(setup_map.is_setup_position_allowed(0, 0))
                self.assertFalse(setup_map.is_setup_position_allowed(0, 1))
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(setup_map.validate_against_map('00\n00'), [])
            self.assertEqual(setup_map.get_setup_forbidden_floor_cells('00\n00'), [(0, 1), (1, 0)])
            self.assertEqual(validate.call_count, 1)

    def test_map_edits_change_permissions_without_manual_cache_reset(self):
        for text, allowed in (('01', True), ('10', False), ('01', True)):
            with patch.object(setup_map, 'DEFENDER_SETUP_MASK_STR', text):
                self.assertEqual(setup_map.is_setup_position_allowed(0, 0), allowed)

    def test_invalid_edits_are_validated_even_after_a_valid_map(self):
        with patch.object(setup_map, 'DEFENDER_SETUP_MASK_STR', '01'):
            self.assertTrue(setup_map.is_setup_position_allowed(0, 0))
        for text in ('', '02', '01\n0'):
            with self.subTest(text=text), patch.object(setup_map, 'DEFENDER_SETUP_MASK_STR', text):
                with self.assertRaises(ValueError):
                    setup_map.is_setup_position_allowed(0, 0)

    def test_returned_masks_cannot_mutate_cached_permissions(self):
        with patch.object(setup_map, 'DEFENDER_SETUP_MASK_STR', '01\n10'):
            mask = setup_map.get_setup_mask()
            mask[0][0] = 1
            self.assertTrue(setup_map.is_setup_position_allowed(0, 0))
            self.assertEqual(setup_map.get_setup_mask(), [[0, 1], [1, 0]])
            for row, col in ((-1, 0), (0, -1), (2, 0), (0, 2)):
                self.assertFalse(setup_map.is_setup_position_allowed(row, col))


if __name__ == '__main__':
    unittest.main()
