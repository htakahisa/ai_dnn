"""Repair a blocked support destination without forcing the chosen movement."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from frc_v1.perception import FrcPerceptionBuilder
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_attacker_entry_yield import yielding_entry_goal
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_attacker_site_entry import traffic_distances


class EntryYieldTests(unittest.TestCase):
    def setUp(self):
        self.scenario=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(self.scenario)
        positions=((12,40),(11,40),(10,41),(8,40),(10,40))
        allies=tuple(replace(a,position=positions[a.slot],alive=True,has_spike=a.slot==1) for a in snapshot.allies)
        self.snapshot=replace(snapshot,allies=allies)
        self.helper=allies[4]

    def test_long_occupied_shortcut_yields_to_nearby_free_site_cell(self):
        self.assertEqual(traffic_distances(self.snapshot,self.helper,(10,42))[self.helper.position],16)
        self.assertEqual(yielding_entry_goal(self.scenario,self.snapshot,self.helper,(10,42),(10,40),'R'),(9,40))

    def test_usable_goal_carrier_and_nonblocking_helpers_keep_their_goal(self):
        open_snapshot=replace(self.snapshot,allies=tuple(replace(a,alive=False) if a.slot==2 else a for a in self.snapshot.allies))
        self.assertIsNone(yielding_entry_goal(self.scenario,open_snapshot,self.helper,(10,42),(10,40),'R'))
        self.assertIsNone(yielding_entry_goal(self.scenario,self.snapshot,replace(self.helper,has_spike=True),(10,42),(10,40),'R'))
        self.assertIsNone(yielding_entry_goal(self.scenario,self.snapshot,replace(self.helper,position=(9,40)),(10,42),(10,40),'R'))

    def test_no_free_reachable_site_goal_does_not_invent_a_destination(self):
        limited=NS(grid=self.scenario.grid,sites={'R':((10,40),(10,41),(8,40))})
        self.assertIsNone(yielding_entry_goal(limited,self.snapshot,self.helper,(10,42),(10,40),'R'))

    def test_v17_goal_change_persists_and_v16_is_unchanged(self):
        c=Controller.__new__(Controller)
        c.scenario=self.scenario; c.route=NS(site='R',cells=((11,40),(10,40)))
        c.cursors={}; c.retriever=None; c.site_entry_state=None; c.attack_plan=NS(scout_goals={})
        c.endpoints={4:(10,42)}; c.events=[]
        c.inputs={self.snapshot.allies[1].name:NS(goal=(10,40))}
        c.encoder=NS(execution_version=16)
        self.assertEqual(c.goal_for(self.snapshot,self.helper),(10,42))
        c.encoder.execution_version=17
        self.assertEqual(c.goal_for(self.snapshot,self.helper),(9,40))
        self.assertEqual(c.endpoints[4],(9,40))
        self.assertEqual(c.goal_for(self.snapshot,replace(self.helper,position=(9,40))),(9,40))
        self.assertEqual(len(c.events),1)


if __name__=='__main__':
    unittest.main()
