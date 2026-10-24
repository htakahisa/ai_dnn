"""Public utility examples and FRC/right scope; no optimizer updates."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frc_v1.actions import FrcAction
from frc_v1.perception import FrcTeamSnapshot, Sighting
from touyama_v3.tv3_model_extensions import ExtensionContext, build_extensions
from touyama_v3 import tv3_custom_train_attacker_plant_frc_v1_R as custom


class RightUtilityTests(unittest.TestCase):
    def setUp(self):
        self.caster = NS(slot=0, position=(3, 1), ability_name='FLASH', blind=0,
                         hp=100, max_hp=100, has_spike=False)
        self.supporter = NS(slot=1, position=(4, 7), facing='N', alive=True, blind=0)
        self.snapshot = FrcTeamSnapshot('A', 1, 'battle', 10,
            tuple(tuple(0 for _ in range(12)) for _ in range(9)),
            (self.supporter,), (), (Sighting(0, (3, 7), 'visible'),), (), (), (), (), (),
            None, None, False, 80, 0, False)
        self.context = ExtensionContext('attacker_plant', 'frc_v1', 'R',
                                         self.snapshot, self.caster, (3, 8))
        actions = [FrcAction('STAY', 'E') for _ in range(58)]
        for i in range(40, 48):
            actions[i] = FrcAction('ABILITY', 'E', None)
        actions[40] = FrcAction('ABILITY', 'E', (3, 1))  # Zero flight.
        actions[41] = FrcAction('ABILITY', 'E', (3, 7))
        actions[56] = FrcAction('PLANT', 'E')
        self.inputs = NS(observation=np.zeros(969, np.float32), mask=np.ones(58, bool),
                         actions=tuple(actions), teacher=0,
                         combat=NS(reason='advance'), support=None)
        self.pipeline = build_extensions(('tv3_custom_train_attacker_plant_frc_v1_R',),
                                          role='attacker_plant', opponent='frc_v1')

    def test_flash_teaches_supported_legal_cast_without_changing_model_inputs(self):
        mask, obs, actions = self.inputs.mask.copy(), self.inputs.observation.copy(), self.inputs.actions
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 41)
        np.testing.assert_array_equal(mask, self.inputs.mask)
        np.testing.assert_array_equal(obs, self.inputs.observation)
        self.assertEqual(actions, self.inputs.actions)

    def test_masked_cast_is_never_taught(self):
        self.inputs.mask[41] = False
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 0)

    def test_supported_target_count_beats_total_coverage(self):
        actions = list(self.inputs.actions)
        actions[42] = FrcAction('ABILITY', 'E', (2, 7))
        self.inputs.actions = tuple(actions)
        self.inputs.mask[40] = False
        supported = Sighting(0, (3, 7), 'visible')
        unsupported = Sighting(1, (8, 1), 'visible')
        def coverage(geometry, snapshot, origin, target, sightings):
            hits = (unsupported, unsupported, unsupported) if target == (3, 7) else (supported,)
            return target, 1, hits
        with patch.object(custom, 'covered_threats', side_effect=coverage):
            custom.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 42)

    def test_scope_excludes_left_other_opponent_and_role(self):
        for changes in (dict(site='L'), dict(opponent='omoko_v1'), dict(role='attacker_guard')):
            self.pipeline.adjust_inputs(replace(self.context, **changes), self.inputs)
            self.assertEqual(self.inputs.teacher, 0)

    def test_retreat_low_hp_blind_holder_and_support_movement_keep_teacher(self):
        for field, value in (('blind', 1), ('hp', 35), ('has_spike', True)):
            before = getattr(self.caster, field)
            setattr(self.caster, field, value)
            self.pipeline.adjust_inputs(self.context, self.inputs)
            self.assertEqual(self.inputs.teacher, 0)
            setattr(self.caster, field, before)
        self.inputs.combat.reason = 'cover_retreat'
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 0)
        self.inputs.combat.reason = 'advance'
        self.inputs.support = dict(steps=1)
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 0)
        self.inputs.support = None
        self.inputs.teacher = 56
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 56)

    def test_flash_requires_facing_support_and_smoke_free_impact(self):
        self.supporter.facing = 'S'
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 0)
        self.supporter.facing = 'N'
        snapshot = replace(self.snapshot, smoke_cells=((3, 8),))
        self.pipeline.adjust_inputs(replace(self.context, snapshot=snapshot), self.inputs)
        self.assertEqual(self.inputs.teacher, 0)

    def test_recon_only_adds_examples_for_smoke_obstructed_unrevealed_targets(self):
        self.caster.ability_name = 'RECON'
        self.pipeline.adjust_inputs(self.context, self.inputs)
        self.assertEqual(self.inputs.teacher, 0)
        snapshot = replace(self.snapshot, smoke_cells=((4, 7),))
        self.pipeline.adjust_inputs(replace(self.context, snapshot=snapshot), self.inputs)
        self.assertEqual(self.inputs.teacher, 41)
        self.inputs.teacher = 0
        snapshot = replace(snapshot, sightings=(Sighting(0, (3, 7), 'reveal'),))
        self.pipeline.adjust_inputs(replace(self.context, snapshot=snapshot), self.inputs)
        self.assertEqual(self.inputs.teacher, 0)

    def test_pending_effect_and_deadline_do_not_create_extra_cast_example(self):
        for snapshot in (replace(self.snapshot, effects=(NS(kind='FLASH', phase='flight'),)),
                         replace(self.snapshot, round_timer=1), replace(self.snapshot, sightings=())):
            self.pipeline.adjust_inputs(replace(self.context, snapshot=snapshot), self.inputs)
            self.assertEqual(self.inputs.teacher, 0)

    def test_recon_does_not_treat_wall_blocked_line_as_smoke_only(self):
        self.caster.ability_name = 'RECON'
        self.supporter.position = (6, 7)
        grid = np.asarray(self.snapshot.grid).copy()
        grid[5, 7] = 1
        snapshot = replace(self.snapshot, grid=tuple(map(tuple, grid)), smoke_cells=((4, 7),))
        self.pipeline.adjust_inputs(replace(self.context, snapshot=snapshot), self.inputs)
        self.assertEqual(self.inputs.teacher, 0)

    def test_projectile_and_wall_geometry_match_scenario(self):
        from touyama_v3.tv3_scenario import Scenario
        scenario = Scenario()
        geometry = custom._PublicGeometry(NS(grid=scenario.grid))
        for start, aim in (((14, 39), (7, 38)), ((11, 40), (10, 40)), ((8, 3), (7, 4))):
            self.assertEqual(geometry.clear(start, aim), scenario.clear(start, aim))
            self.assertEqual(geometry._projectile_path(start, aim), scenario._projectile_path(start, aim))


if __name__ == '__main__':
    unittest.main()
