"""Right-entry support regression on public snapshots; no training updates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
import torch
from frc_v1.perception import FrcPerceptionBuilder, Sighting
from frc_v1.actions import build_masks
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_learn_attacker_support import support_targets, support_path, support_advice, SupportCoach
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, Route
from touyama_v3.tv3_learn_attacker_plant import PlantEncoder, OBS_DIM, policy_schema, observation_dim
from touyama_v3.tv3_attacker_combat import CombatAdvice
from touyama_v3.tv3_train_attacker_plant import support_progress_credit


class SupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = attacker_world()
        snapshot = FrcPerceptionBuilder('A').build(self.game)
        positions = {0:(16,40), 1:(11,40), 2:(13,40), 3:(14,40), 4:(10,42)}
        allies = tuple(replace(a, position=positions[a.slot], has_spike=a.slot == 1,
                               blind=0, forced_facing=False, facing='N', charges=0,
                               hp=100., max_hp=100.) for a in snapshot.allies)
        self.snap = replace(snapshot, allies=allies, tick=55, round_timer=45,
                            visible_cells=(), sightings=(Sighting(3,(7,38),'normal'),))
        self.helper = next(a for a in allies if a.slot == 4)
        self.route = Route('R', ((11,40),(10,40)), ('h',))
        self.targets = support_targets(self.game, self.snap, self.route, {})

    def test_actual_right_niche_has_support_path_that_leaves_plant_goal_open(self):
        self.assertFalse(self.game.clear((10,42),(7,38)))
        plan = support_path(self.game, self.snap, self.helper, self.targets, self.route)
        self.assertEqual(plan['goal'], (9,40))
        self.assertEqual(plan['step'], (10,41))
        self.assertEqual(plan['steps'], 3)
        self.assertTrue(self.game.clear(plan['goal'],plan['target']))

    def test_occupied_lane_is_not_promised_as_reachable_support(self):
        blocked = replace(self.snap, allies=tuple(replace(a,position=(10,41)) if a.slot==0 else a
                                                  for a in self.snap.allies))
        self.assertIsNone(support_path(self.game,blocked,self.helper,self.targets))

    def test_blind_or_wounded_helper_keeps_escape_advice(self):
        for helper in (replace(self.helper,blind=3),replace(self.helper,hp=10.)):
            self.assertIsNone(support_path(self.game,self.snap,helper,self.targets))
        advice=CombatAdvice((10,42),'N','cover_retreat',(7,38),2,2,0,0)
        plan=dict(goal=(10,40),step=(10,41),steps=2,target=(7,38))
        self.assertIs(support_advice(self.game,self.snap,self.helper,advice,self.targets,plan),advice)

    def test_v11_teaches_support_move_where_v10_teaches_goal_wait(self):
        encoder=AttackerEncoder(self.game)
        observation=encoder.observe(self.snap,[])
        model=AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(self.game.names))
        analysis=model.analyze(encoder,self.snap,observation,[self.route])
        results={version:PlantEncoder(self.game,version).encode(self.snap,self.helper,(10,42),self.route,
                 analysis,{},build_masks(self.snap),0,'supported') for version in (10,11)}
        self.assertEqual(results[10].actions[results[10].teacher].kind,'STAY')
        self.assertEqual(results[11].actions[results[11].teacher].kind,'W')
        self.assertEqual(len(results[10].observation),892)
        self.assertEqual(len(results[11].observation),observation_dim(11))
        self.assertTrue(results[11].mask[:8].any())  # Learned waiting remains available.
        self.assertNotEqual(policy_schema(self.game,10),policy_schema(self.game,11))

    def test_support_shaping_round_trip_has_no_discounted_reward_gain(self):
        from touyama_v3.tv3_learn_attacker_plant import GAMMA
        field=np.zeros((3,3),int)
        field[1,1],field[1,2]=2,1
        inputs=NS(support_distances=field)
        forward=support_progress_credit(NS(position=(1,1)),(1,2),inputs,False)
        back=support_progress_credit(NS(position=(1,2)),(1,1),inputs,False)
        wait=support_progress_credit(NS(position=(1,1)),(1,1),inputs,False)
        self.assertAlmostEqual(forward+GAMMA*back,wait+GAMMA*wait)

    def test_private_enemy_coordinates_do_not_enter_support_plan(self):
        before=support_path(self.game,self.snap,self.helper,self.targets)
        for c in self.game.chars:
            if c.team=='D':
                c.pos=(1,18)
        after=support_path(self.game,self.snap,self.helper,support_targets(self.game,self.snap,self.route,{}))
        self.assertEqual(before,after)

    def test_healthy_helper_covers_imminent_entry_but_wounded_carrier_does_not_commit(self):
        helper=replace(self.helper,position=(10,40))
        snap=replace(self.snap,allies=tuple(helper if a.slot==4 else a for a in self.snap.allies))
        plan=support_path(self.game,snap,helper,self.targets)
        self.assertEqual(plan['steps'],0)
        advice=CombatAdvice((10,41),'NW','cover_retreat',(7,38),1,1,0,0)
        covering=support_advice(self.game,snap,helper,advice,self.targets,plan)
        self.assertEqual(covering.reason,'stop_shoot')
        wounded=replace(snap,allies=tuple(replace(a,hp=10.) if a.has_spike else a for a in snap.allies))
        self.assertIs(support_advice(self.game,wounded,helper,advice,self.targets,plan),advice)

    def test_support_commitment_survives_first_step_and_clears_when_threat_invalid(self):
        coach=SupportCoach()
        advice=CombatAdvice(self.helper.position,'NW','advance',None,0,0,0,0)
        self.assertIsNone(coach.plan(self.game,self.snap,self.helper,(9,42),self.route,self.targets,advice))
        plan=coach.plan(self.game,self.snap,self.helper,self.helper.position,self.route,self.targets,advice)
        self.assertEqual(plan['step'],(10,41))
        moved=replace(self.helper,position=(10,41))
        snap=replace(self.snap,tick=56,allies=tuple(moved if a.slot==4 else a for a in self.snap.allies))
        following=coach.plan(self.game,snap,moved,self.helper.position,self.route,self.targets,advice)
        self.assertEqual(following['step'],(10,40))
        self.assertIsNone(coach.plan(self.game,snap,moved,self.helper.position,self.route,(),advice))
        self.assertEqual(coach.active,{})

    def test_existing_ready_support_does_not_recruit_another_helper(self):
        snap=replace(self.snap,allies=tuple(replace(a,position=(9,38),facing='N') if a.slot==0 else a
                                         for a in self.snap.allies))
        advice=CombatAdvice(self.helper.position,'NW','advance',None,0,0,0,0)
        self.assertIsNone(SupportCoach().plan(self.game,snap,self.helper,self.helper.position,self.route,self.targets,advice))


if __name__ == '__main__':
    unittest.main()
