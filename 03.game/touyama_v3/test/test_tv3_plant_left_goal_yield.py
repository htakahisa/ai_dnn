"""Left occupancy-chain repair without action forcing or weight updates."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch, MagicMock
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_plant_left_goal_yield import LeftGoalYield, left_yield_proposal


class LeftGoalYieldTests(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario()
        positions = ((8,3),(10,3),(8,4),(9,3),(10,2))
        self.allies = tuple(NS(slot=i,position=p,alive=True,has_spike=i==1,plant_progress=0,
                               blind=0,movement_disabled=0,warping=False,ramp_blocked=False,reveal=0,facing='N')
                            for i,p in enumerate(positions))
        self.snapshot = NS(allies=self.allies, grid=tuple(map(tuple,self.scenario.grid)),
                           round_number=9, tick=0, phase='battle', is_planted=False,
                           smoke_cells=(), sightings=())
        self.goals = {0:(8,3),1:(9,3),2:(7,3),3:(8,4),4:(7,4)}

    def test_dependency_chain_releases_arrived_ally_to_unused_goal(self):
        plan = left_yield_proposal(self.scenario,self.snapshot,self.goals)
        self.assertEqual(plan['slot'],0)
        self.assertEqual(plan['goal'],(8,2))
        self.assertNotIn(plan['goal'],self.goals.values())
        self.assertIn(3,plan['blockers'])
        self.assertIn(2,plan['blockers'])

    def test_immediate_entrance_blocker_has_no_free_exit(self):
        from touyama_v3.tv3_attacker_entry_yield import yielding_entry_goal
        self.assertIsNone(yielding_entry_goal(self.scenario,self.snapshot,self.allies[3],
                                             self.goals[3],self.goals[1],'L'))
        self.assertIsNotNone(left_yield_proposal(self.scenario,self.snapshot,self.goals))

    def test_waits_for_persistent_jam_and_keeps_new_goal_after_arrival(self):
        repair = LeftGoalYield()
        for tick in range(3):
            self.snapshot.tick=tick
            self.assertEqual(repair.update(self.scenario,self.snapshot,self.goals,site='L'),self.goals)
        self.snapshot.tick=3
        goals=repair.update(self.scenario,self.snapshot,self.goals,site='L')
        self.assertEqual(goals[0],(8,2))
        self.allies[0].position=(8,2)
        self.snapshot.tick=4
        self.assertEqual(repair.update(self.scenario,self.snapshot,self.goals,site='L')[0],(8,2))
        self.assertEqual(len(repair.events),1)
        self.allies[1].position=(9,3)
        self.snapshot.tick=5
        self.assertEqual(repair.update(self.scenario,self.snapshot,self.goals,site='L'),self.goals)

    def test_no_change_on_right_setup_or_coordinated_wait(self):
        for side in ('R','L'):
            repair=LeftGoalYield()
            for tick in range(8):
                self.snapshot.tick=tick
                result=repair.update(self.scenario,self.snapshot,self.goals,site=side,
                                     coordinated_wait=side=='L')
                self.assertEqual(result,self.goals)
        self.snapshot.phase='setup'
        self.assertEqual(LeftGoalYield().update(self.scenario,self.snapshot,self.goals,site='L'),self.goals)

    def test_no_carrier_no_occupied_goal_and_plant_in_progress_are_unchanged(self):
        self.allies[1].has_spike=False
        self.assertIsNone(left_yield_proposal(self.scenario,self.snapshot,self.goals))
        self.allies[1].has_spike=True
        self.allies[1].plant_progress=1
        self.assertIsNone(left_yield_proposal(self.scenario,self.snapshot,self.goals))
        self.allies[1].plant_progress=0
        self.assertIsNone(left_yield_proposal(self.scenario,self.snapshot,{**self.goals,1:(7,3)}))

    def test_blind_arrived_ally_is_not_selected_for_yield(self):
        self.allies[0].blind=1
        plan=left_yield_proposal(self.scenario,self.snapshot,self.goals)
        self.assertTrue(plan is None or plan['slot']!=0)

    def test_public_exposure_increase_is_not_admitted(self):
        threat=NS(position=(0,0))
        with patch('touyama_v3.tv3_learn_attacker_plant_left_goal_yield.incoming_clear',
                   side_effect=lambda scenario,snapshot,ally,position,threat: position!=ally.position), \
                patch('touyama_v3.tv3_learn_attacker_plant_left_goal_yield.shot_clear',return_value=False):
            self.assertIsNone(left_yield_proposal(self.scenario,self.snapshot,self.goals,(threat,)))

    def test_existing_public_fire_support_is_preserved(self):
        with patch('touyama_v3.tv3_learn_attacker_plant_left_goal_yield.shot_clear',return_value=True):
            self.assertIsNone(left_yield_proposal(self.scenario,self.snapshot,self.goals,(NS(position=(0,0)),)))

    def test_tick_gap_and_round_change_reset_trigger(self):
        repair=LeftGoalYield()
        for tick in (0,1,8):
            self.snapshot.tick=tick
            self.assertEqual(repair.update(self.scenario,self.snapshot,self.goals,site='L'),self.goals)
        self.snapshot.round_number=10
        self.snapshot.tick=9
        self.assertEqual(repair.update(self.scenario,self.snapshot,self.goals,site='L'),self.goals)

    def test_full_review_right_does_not_recompute_goals_and_clears_left_plan(self):
        from touyama_v3.tv3_review_attacker_plant_left_goal_yield_evaluation import GoalRepairReview
        controller=MagicMock()
        controller.route.site='R'
        review=GoalRepairReview(lambda controller,snapshot,ally:(10,40))
        self.assertEqual(review.goal_for(controller,self.snapshot,self.allies[0]),(10,40))
        controller._assigned_goal_for.assert_not_called()
        state=review.states[controller]
        state['repair'].plan={'slot':0,'goal':(8,2),'entry':(9,3),'carrier_slot':1}
        review.goal_for(controller,self.snapshot,self.allies[0])
        self.assertIsNone(state['repair'].plan)

    def test_full_review_keeps_state_separate_for_each_controller(self):
        from touyama_v3.tv3_review_attacker_plant_left_goal_yield_evaluation import GoalRepairReview
        review=GoalRepairReview(lambda controller,snapshot,ally:(10,40))
        controllers=[MagicMock(),MagicMock()]
        for controller in controllers:
            controller.route.site='R'
            review.goal_for(controller,self.snapshot,self.allies[0])
        self.assertIsNot(review.states[controllers[0]]['repair'],review.states[controllers[1]]['repair'])


if __name__=='__main__':
    unittest.main()
