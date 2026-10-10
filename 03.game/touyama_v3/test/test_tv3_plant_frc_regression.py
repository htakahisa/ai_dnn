"""FRC timeout regressions, without training or optimizer updates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import Counter, deque
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_learn_attacker_plant import (
    demonstration_kind_loss, sample_plant_batch, CARRIER_FEATURE_INDEX, OBS_DIM, EDGE_TRAVERSAL_WINDOW)
from touyama_v3.tv3_train_attacker_plant import carrier_stall_cost, learning_settings


class PlantFrcRegression(unittest.TestCase):
    def test_total_limit_constant_is_connected_to_argument_defaults(self):
        from unittest.mock import patch
        from touyama_v3 import tv3_train_attacker_plant as training
        with patch.object(training,'TOTAL_TRAINING_SET_LIMIT',100):
            self.assertEqual(training.parser().parse_args([]).total_sets,100)
            self.assertEqual(training.parser().parse_args(['--total-sets','150']).total_sets,150)

    def test_resume_total_limit_accounts_for_each_opponents_completed_sets(self):
        from touyama_v3.tv3_train_attacker_plant import remaining_training_sets
        self.assertEqual(remaining_training_sets(100,51,100),49)
        self.assertEqual(remaining_training_sets(100,61,100),39)
        self.assertEqual(remaining_training_sets(100,100,100),0)
        self.assertEqual(remaining_training_sets(100,51,None),100)
        self.assertEqual(remaining_training_sets(10,51,100),10)
        with self.assertRaises(ValueError):
            remaining_training_sets(100,51,-1)

    def test_edge_budget_expires_and_preserves_recent_crossings(self):
        controller = Controller.__new__(Controller)
        key = (0, ((2,2),(2,3)))
        controller.edges = Counter({key:2})
        controller.edge_history = deque([(1,key),(4,key)])
        controller._expire_edges(1+EDGE_TRAVERSAL_WINDOW)
        self.assertEqual(controller.edges[key],1)
        controller._expire_edges(4+EDGE_TRAVERSAL_WINDOW)
        self.assertNotIn(key,controller.edges)
        self.assertFalse(controller.edge_history)

    def test_recent_edge_limit_never_blocks_combat_retreat(self):
        controller = Controller.__new__(Controller)
        grid = np.zeros((7,7),dtype=int)
        controller.scenario = NS(grid=grid)
        controller.snapshot = NS(tick=2)
        controller.route = NS(cells=((3,3),(3,4)))
        controller.cursors = {0:0}
        controller.attack_plan = NS(scout_goals={})
        controller.retriever = None
        controller.edge_history = deque()
        controller.edges = Counter({(0,((3,3),(3,4))):2})
        controller.encoder = NS(combat_coach=NS(contacts=lambda *args: ()))
        ally = NS(slot=0,position=(3,3),blind=0)
        from touyama_v3.tv3_learn_attacker_plant import MOVEMENTS
        east = MOVEMENTS.index('E')*8
        inputs = NS(mask=np.ones(58,bool),combat=NS(reason='advance'))
        controller._restrict_movement(ally,inputs,set())
        self.assertFalse(inputs.mask[east])
        inputs.mask[:] = True
        inputs.combat.reason = 'cover_retreat'
        controller._restrict_movement(ally,inputs,set())
        self.assertTrue(inputs.mask[east])
        inputs.mask[:] = True
        controller.snapshot.tick = 20
        controller.edges.clear()
        controller._restrict_movement(ally,inputs,{(3,4)})
        self.assertFalse(inputs.mask[east])

    def test_kind_loss_ignores_facing_variants_but_not_wrong_movement(self):
        mask = torch.ones((1,58),dtype=torch.bool)
        values = torch.zeros((1,58))
        values[0,15] = 5
        correct = demonstration_kind_loss(values,torch.tensor([8]),mask)
        same = demonstration_kind_loss(values,torch.tensor([15]),mask)
        wrong = demonstration_kind_loss(values,torch.tensor([0]),mask)
        self.assertAlmostEqual(float(correct),float(same))
        self.assertLess(float(correct),float(wrong))
        mask[0,8:16] = False
        self.assertTrue(torch.isfinite(demonstration_kind_loss(values,torch.tensor([56]),mask)))

    def test_carrier_sampling_keeps_other_roles_and_handles_no_carrier(self):
        ordinary = [np.zeros(OBS_DIM)]
        carrier = [np.zeros(OBS_DIM)]
        carrier[0][CARRIER_FEATURE_INDEX] = 1
        rows = [ordinary]*99+[carrier]
        batch = sample_plant_batch(rows,np.random.default_rng(5),20,.65)
        self.assertGreaterEqual(sum(r is carrier for r in batch),13)
        self.assertTrue(any(r is ordinary for r in batch))
        self.assertEqual(len(sample_plant_batch([ordinary],np.random.default_rng(5),20,.65)),20)

    def test_stall_penalty_requires_legality_and_excludes_tactical_wait(self):
        before = NS(has_spike=True,blind=0,plant_progress=0)
        inputs = NS(combat=NS(contacts=0,reason='advance'),
                    observation=np.array([0.,1.]),actions=[NS(kind='E')],teacher=0)
        action = NS(kind='STAY')
        self.assertGreater(carrier_stall_cost(before,inputs,action),0)
        inputs.observation[:] = 0
        self.assertEqual(carrier_stall_cost(before,inputs,action),0)
        inputs.observation[:] = 1
        inputs.actions[0].kind = 'STAY'
        self.assertEqual(carrier_stall_cost(before,inputs,action),0)
        inputs.actions[0].kind = 'E'
        inputs.combat.contacts = 1
        self.assertEqual(carrier_stall_cost(before,inputs,action),0)
        inputs.combat.contacts = 0
        before.plant_progress = 1
        self.assertEqual(carrier_stall_cost(before,inputs,action),0)

    def test_frc_settings_are_opponent_specific(self):
        frc, other = learning_settings('frc_v1',200),learning_settings('gc_v1',200)
        self.assertEqual(frc['updates'],400)
        self.assertEqual(other['updates'],200)
        self.assertGreater(frc['demonstration_kind_weight'],other['demonstration_kind_weight'])

    def test_inside_site_point_is_not_a_separate_entry(self):
        controller = Controller.__new__(Controller)
        grid = np.zeros((5,5),int)
        grid[1,1:3] = 2
        controller.scenario = NS(grid=grid)
        plan = NS(changed=False,scout_goals={1:(1,1),2:(2,2)})
        controller.route_planner = NS(update=lambda *args: plan,
            flank_entries={1:((1,1),(1,1)),2:((2,2),(1,2))},events=[{}])
        controller.analysis_encoder = controller.analysis = controller.route = controller.route_mode = None
        self.assertTrue(controller.select_route(None,None))
        self.assertNotIn(1,controller.route_planner.flank_entries)
        self.assertNotIn(1,plan.scout_goals)
        self.assertIn(2,controller.route_planner.flank_entries)

    def test_frozen_toru_layout_loads_old_best_with_corrected_runtime_los(self):
        from touyama_v3.tv3_opponents import frozen_toru_scenario, TORU_BEST_DIRECTORY
        from toruAI_v4.tv4_attacker_guard_controller import ToruV4AttackerPlantGuardController
        from toruAI_v4.tv4_defender_controller import ToruV4DefenderController
        scenario = frozen_toru_scenario()
        ToruV4AttackerPlantGuardController.from_best('touyama_v2',best_dir=TORU_BEST_DIRECTORY,scenario=scenario)
        ToruV4DefenderController('touyama_v2',scenario=scenario,
            search_path=TORU_BEST_DIRECTORY/'touyama_v2'/'search_best.pt',retake_path=TORU_BEST_DIRECTORY)
        scenario.grid = np.array([[0,0],[0,1],[0,1]])
        self.assertFalse(scenario.clear((0,1),(2,0)))


if __name__ == '__main__':
    unittest.main()
