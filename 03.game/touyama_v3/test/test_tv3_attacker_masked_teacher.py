"""A blocked retreat must not become a forward teacher example."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from frc_v1 import FACING
from frc_v1.actions import FrcAction
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from grid_lines import wall_line_cells
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_learn_attacker_masked_teacher import masked_retreat_teacher
from touyama_v3.tv3_learn_attacker_plant import PlantDQN,PlantEncoder,policy_schema,PLANT_VERSION,demonstration_kind_loss
from touyama_v3.tv3_scenario import Scenario


class MaskedRetreatTeacherTests(unittest.TestCase):
    def setUp(self):
        snap=FrcPerceptionBuilder('A').build(attacker_world())
        self.grid=np.zeros((9,9),dtype=int)
        allies=tuple(replace(a,position=(4,4) if a.slot==0 else (8,a.slot),blind=0)
                     for a in snap.allies)
        self.snap=replace(snap,allies=allies,sightings=(Sighting(0,(0,4),'normal'),),
                          visible_cells=(),smoke_cells=())
        self.scenario=NS(grid=self.grid,clear=lambda a,b:not any(self.grid[p]==1 for p in wall_line_cells(a,b)))
        self.inputs=NS(actions=tuple(FrcAction(kind,f) for kind in ('STAY','N','E','S','W') for f in FACING),
                       mask=np.zeros(40,dtype=bool),combat=NS(position=(4,3),facing='N'))
        self.inputs.mask[:16]=True  # The intended W retreat is unavailable; N approaches the enemy.

    def test_blocked_retreat_does_not_use_site_distance_to_choose_forward(self):
        selected=masked_retreat_teacher(self.scenario,self.snap,self.snap.allies[0],self.inputs,{})
        self.assertEqual(self.inputs.actions[selected].kind,'STAY')

    def test_available_wall_cover_has_priority_over_staying_exposed(self):
        self.grid[3,5]=1
        self.inputs.mask[16:24]=True
        selected=masked_retreat_teacher(self.scenario,self.snap,self.snap.allies[0],self.inputs,{})
        self.assertEqual(self.inputs.actions[selected].kind,'E')

    def test_teacher_respects_the_final_action_mask_and_facing(self):
        self.inputs.mask[:]=False
        self.inputs.mask[11]=True
        self.assertEqual(masked_retreat_teacher(self.scenario,self.snap,self.snap.allies[0],self.inputs,{}),11)

    def test_new_teacher_version_keeps_saved_versions_distinct(self):
        scenario=Scenario()
        old=policy_schema(scenario,18); new=policy_schema(scenario,19)
        self.assertNotIn('masked_retreat_teacher',old)
        self.assertIn('masked_retreat_teacher',new)
        self.assertEqual(PlantDQN(18).net[0].in_features,PlantDQN(19).net[0].in_features)
        self.assertEqual(PlantEncoder(scenario,19).execution_version,19)
        self.assertEqual(PLANT_VERSION,19)

    def test_warm_start_only_permits_documented_teacher_changes(self):
        from touyama_v3.tv3_train_attacker_plant import warm_start_schema_compatible
        scenario=Scenario()
        new=policy_schema(scenario,19)
        for version in (16,18):
            self.assertTrue(warm_start_schema_compatible(policy_schema(scenario,version),new))
        self.assertFalse(warm_start_schema_compatible(policy_schema(scenario,19),policy_schema(scenario,18)))
        changed=dict(new,wall_los='different')
        self.assertFalse(warm_start_schema_compatible(policy_schema(scenario,16),changed))


class TeacherVariantGradientTests(unittest.TestCase):
    def test_kind_loss_must_not_reward_a_different_facing(self):
        values=torch.zeros((1,58),requires_grad=True)
        with torch.no_grad():
            values[0,26]=1.680621
            values[0,31]=1.854010
        teachers=torch.tensor([26])
        masks=torch.ones_like(values,dtype=torch.bool)
        original=torch.autograd.grad(demonstration_kind_loss(values,teachers,masks),values)[0]
        corrected=torch.autograd.grad(demonstration_kind_loss(values,teachers,masks,teacher_variant=True),values)[0]
        self.assertLess(original[0,31].item(),0.)  # Gradient descent increases the wrong facing.
        self.assertEqual(original[0,26].item(),0.)
        self.assertLess(corrected[0,26].item(),0.)
        self.assertEqual(corrected[0,31].item(),0.)

    def test_ability_variants_are_targets_and_the_teacher_target_gets_credit(self):
        values=torch.zeros((1,58),requires_grad=True)
        with torch.no_grad():
            values[0,43]=2.
        gradients=torch.autograd.grad(demonstration_kind_loss(values,torch.tensor([40]),
            torch.ones_like(values,dtype=torch.bool),teacher_variant=True),values)[0]
        self.assertLess(gradients[0,40].item(),0.)
        self.assertEqual(gradients[0,43].item(),0.)

    def test_masked_teacher_variant_is_rejected(self):
        masks=torch.ones((1,58),dtype=torch.bool)
        masks[0,26]=False
        with self.assertRaises(ValueError):
            demonstration_kind_loss(torch.zeros((1,58)),torch.tensor([26]),masks,teacher_variant=True)


if __name__=='__main__':
    unittest.main()
