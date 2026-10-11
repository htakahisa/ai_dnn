"""Owned flash must cover every known incoming line before releasing an edge."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
from collections import Counter,deque
import unittest
import numpy as np
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_learn_attacker_corner_utility import (
    FriendlyFlashTracker,confirmed_flash_covers_step,confirmed_flash_step_window,flash_window_allows_completion)
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller


class FlashProgressTests(unittest.TestCase):
    def test_completion_needs_arrival_time_and_exclusive_expiry_margin(self):
        self.assertFalse(flash_window_allows_completion(6,2,4))
        self.assertTrue(flash_window_allows_completion(7,2,4))
        self.assertFalse(flash_window_allows_completion(10,-1,4))
        self.assertFalse(flash_window_allows_completion(0,0,0))
    def setUp(self):
        self.scenario=attacker_world()
        base=FrcPerceptionBuilder('A').build(self.scenario)
        self.ally=replace(base.allies[0],position=(14,39),blind=0)
        self.snapshot=replace(base,allies=(self.ally,)+tuple(replace(a,alive=False) for a in base.allies[1:]),
            tick=88,sightings=(Sighting(4,(2,40),'normal'),),smoke_cells=(),
            effects=(NS(kind='FLASH',phase='active',position=(6,40)),))
        self.tracker=FriendlyFlashTracker(self.scenario)
        self.tracker.record(84,(7,33),(6,42))

    def test_public_confirmation_expiry_blindness_and_uncovered_line(self):
        step=(14,40)
        check=lambda snapshot:confirmed_flash_covers_step(self.scenario,snapshot,self.ally,step,{},self.tracker)
        self.assertFalse(check(replace(self.snapshot,effects=())))
        self.assertTrue(check(self.snapshot))
        self.assertGreater(confirmed_flash_step_window(self.scenario,self.snapshot,self.ally,step,{},self.tracker),0)
        self.assertFalse(check(replace(self.snapshot,tick=97,effects=())))
        self.assertFalse(confirmed_flash_covers_step(self.scenario,self.snapshot,replace(self.ally,blind=1),step,{},self.tracker))
        # A second line without owned public suppression cannot be ignored.
        tracker=NS(suppressed=lambda snap,p,tracks:p==(2,40))
        snapshot=replace(self.snapshot,sightings=self.snapshot.sightings+(Sighting(3,(4,40),'normal'),))
        # Remove the first body blocker so both potential lines are evaluated.
        snapshot=replace(snapshot,sightings=(Sighting(4,(2,40),'history'),Sighting(3,(4,40),'history')))
        self.assertFalse(confirmed_flash_covers_step(self.scenario,snapshot,self.ally,step,{},tracker))

    def test_edge_release_respects_engine_reservation_and_direction(self):
        controller=Controller.__new__(Controller)
        controller.scenario=self.scenario
        controller.snapshot=self.snapshot
        controller.route=NS(cells=((14,39),(14,40),(13,40)))
        controller.cursors={0:0}; controller.attack_plan=NS(scout_goals={})
        controller.retriever=None; controller.edge_history=deque()
        controller.edges=Counter({(0,((14,39),(14,40))):2})
        controller.analysis_encoder=NS(history=NS(tracks={}))
        controller.encoder=NS(execution_version=16,flash_tracker=self.tracker,
            empty_evidence=NS(tactical_snapshot=lambda snapshot,**kw:snapshot),
            combat_coach=NS(contacts=lambda *args:()))
        traffic=np.full(self.scenario.grid.shape,-1.)
        traffic[14,39],traffic[14,40]=4.,3.
        def inputs():
            mask=np.zeros(58,bool); mask[:8]=True; mask[16:24]=True
            return NS(mask=mask,traffic=traffic,combat=NS(reason='advance'))
        value=inputs(); controller._restrict_movement(self.ally,value,set())
        self.assertTrue(value.mask[16])
        value=inputs(); controller._restrict_movement(self.ally,value,{(14,40)})
        self.assertFalse(value.mask[16])
        value=inputs(); value.mask[16:24]=False
        controller._restrict_movement(self.ally,value,set()); self.assertFalse(value.mask[16])
        traffic[14,40]=5.
        value=inputs(); controller._restrict_movement(self.ally,value,set())
        self.assertFalse(value.mask[16])
        self.assertTrue(value.mask[0])  # The action is never forced to advance.


if __name__=='__main__':
    unittest.main()
