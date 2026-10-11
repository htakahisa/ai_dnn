"""Public danger semantics and experimental teacher; no learning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
import numpy as np
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from frc_v1.actions import FrcAction
from frc_v1 import FACING
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_attacker_combat import CombatAdvice
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear
from touyama_v3.tv3_review_attacker_exposure_teacher import incoming_clear,exposure_advice,risk_threats,RouteProgressProbe
from touyama_v3.tv3_attacker_route_planner import AttackPlan
from touyama_v3.tv3_learn_attacker_analysis import Route
from grid_lines import wall_line_cells


class ExposureProbeTests(unittest.TestCase):
    def setUp(self):
        snapshot=FrcPerceptionBuilder('A').build(attacker_world())
        self.grid=np.zeros((9,9),int)
        self.scenario=NS(grid=self.grid,clear=lambda a,b: not any(self.grid[p]==1 for p in wall_line_cells(a,b)))
        positions={0:(4,4),1:(8,0),2:(8,1),3:(8,2),4:(8,3)}
        allies=tuple(replace(a,position=positions[a.slot],facing='S',blind=0,reveal=0,has_spike=a.slot==0,
                             hp=100.,max_hp=100.) for a in snapshot.allies)
        self.snap=replace(snapshot,allies=allies,tick=50,visible_cells=(),sightings=(),
                          smoke_cells=((4,4),),grid=tuple(map(tuple,self.grid)))
        self.ally=next(a for a in allies if a.slot==0)
        self.threat=Sighting(3,(1,4),'history')

    def test_own_recon_reveal_makes_smoke_asymmetric_not_safe_cover(self):
        marked=replace(self.ally,reveal=10)
        self.assertFalse(incoming_clear(self.scenario,self.snap,self.ally,self.ally.position,self.threat))
        self.assertTrue(incoming_clear(self.scenario,self.snap,marked,marked.position,self.threat))
        self.assertFalse(shot_clear(self.scenario,self.snap,marked.position,self.threat.position,marked.slot))

    def test_wall_and_public_body_still_block_marked_incoming_line(self):
        marked=replace(self.ally,reveal=10)
        snap=replace(self.snap,allies=tuple(replace(a,position=(3,4)) if a.slot==1 else a for a in self.snap.allies))
        self.assertFalse(incoming_clear(self.scenario,snap,marked,marked.position,self.threat))
        self.grid[3,4]=1
        self.assertFalse(incoming_clear(self.scenario,self.snap,marked,marked.position,self.threat))

    def test_elapsed_wait_does_not_justify_unsupported_known_peek(self):
        self.grid[3,5]=1
        ally=replace(self.ally,position=(4,5),facing='N')
        snap=replace(self.snap,allies=tuple(ally if a.slot==0 else a for a in self.snap.allies),smoke_cells=())
        actions=tuple(FrcAction(move,face) for move in ('STAY','N','E','S','W') for face in FACING)
        advice=CombatAdvice((4,4),'N','advance',(1,4),0,1,0,20)
        inputs=NS(actions=actions,teacher=32,mask=np.ones(40,bool),combat=advice)
        from unittest.mock import patch
        with patch('touyama_v3.tv3_review_attacker_exposure_teacher.WAIT_FOR_UNSUPPORTED_LINE',True):
            changed,debug=exposure_advice(self.scenario,snap,ally,inputs,{3:((1,4),49,(0,0))})
        self.assertEqual(debug['future'],1)
        self.assertEqual(changed.position,ally.position)
        self.assertEqual(changed.reason,'assemble')

    def test_wait_ablation_preserves_risk_diagnostics_without_forcing_stay(self):
        self.grid[3,5]=1
        ally=replace(self.ally,position=(4,5),facing='N')
        snap=replace(self.snap,allies=tuple(ally if a.slot==0 else a for a in self.snap.allies),smoke_cells=())
        actions=tuple(FrcAction(move,face) for move in ('STAY','N','E','S','W') for face in FACING)
        advice=CombatAdvice((4,4),'N','advance',(1,4),0,1,0,20)
        inputs=NS(actions=actions,teacher=32,mask=np.ones(40,bool),combat=advice)
        changed,debug=exposure_advice(self.scenario,snap,ally,inputs,{3:((1,4),49,(0,0))})
        self.assertEqual(debug['future'],1)
        self.assertEqual(changed,advice)

    def test_memory_expires_or_clears_on_death_and_empty_visibility(self):
        tracks={3:((1,4),25,(0,0))}
        self.assertTrue(risk_threats(self.snap,tracks))
        self.assertFalse(risk_threats(replace(self.snap,tick=66),tracks))
        self.assertFalse(risk_threats(replace(self.snap,visible_cells=((1,4),)),tracks))
        dead=replace(self.snap,enemies=tuple(replace(e,alive=False) if e.enemy_id==3 else e for e in self.snap.enemies))
        self.assertFalse(risk_threats(dead,tracks))

    def test_visible_recon_flight_predicts_possible_marking_through_smoke(self):
        self.scenario._projectile_path=lambda start,aim: ((1,4),(2,4))
        effect=NS(kind='RECON',phase='flight',position=(0,4),direction=(1.,0.),cells=())
        snap=replace(self.snap,effects=(effect,),smoke_cells=((4,3),))
        threat=replace(self.threat,position=(1,3))
        self.assertTrue(incoming_clear(self.scenario,snap,self.ally,(4,3),threat))
        # A different landing area cannot remove smoke protection here.
        self.scenario._projectile_path=lambda start,aim: ((8,8),)
        self.assertFalse(incoming_clear(self.scenario,snap,self.ally,(4,3),threat))

    def test_route_danger_triggers_detour_before_immediate_step_is_exposed(self):
        from unittest.mock import patch
        self.grid[3,5]=self.grid[3,6]=1
        ally=replace(self.ally,position=(4,6),facing='W')
        snap=replace(self.snap,allies=tuple(ally if a.slot==0 else a for a in self.snap.allies),smoke_cells=())
        actions=tuple(FrcAction(move,face) for move in ('STAY','N','E','S','W') for face in FACING)
        inputs=NS(actions=actions,teacher=32,mask=np.ones(40,bool),
                  combat=CombatAdvice((4,5),'W','advance',(1,4),0,0,0,0))
        route=Route('R',((4,6),(4,5),(4,4),(3,4)),('h',))
        plan=dict(path=((4,6),(5,6),(6,6)),steps=2,cost=2)
        with patch('touyama_v3.tv3_review_attacker_exposure_teacher.ENABLE_EXPOSURE_DETOURS',True), \
             patch('touyama_v3.tv3_review_attacker_exposure_teacher.safe_plant_detour',return_value=plan):
            advice,debug=exposure_advice(self.scenario,snap,ally,inputs,{3:((1,4),49,(0,0))},'R',route)
        self.assertEqual(debug['current'],0)
        self.assertEqual(debug['future'],0)
        self.assertTrue(debug['upcoming_route_lines'])
        self.assertEqual(advice.position,(5,6))
        self.assertTrue(inputs.diagnostic_allow_detour)

    def test_detour_returns_feasible_shorter_path_when_safest_path_misses_deadline(self):
        from unittest.mock import patch
        from touyama_v3.tv3_review_attacker_exposure_teacher import safe_plant_detour
        start, risky, goal=(0,0),(0,1),(0,2)
        safe=(start,(1,0),(2,0),(2,1),(2,2),(1,2),goal)
        graph={p:set() for p in safe+(risky,)}
        for a,b in list(zip(safe,safe[1:]))+[(start,risky),(risky,goal)]:
            graph[a].add(b); graph[b].add(a)
        scenario=NS(sites={'R':(goal,),'L':()},neighbors=lambda p:graph[p])
        ally=replace(self.ally,position=start)
        snapshot=replace(self.snap,allies=(ally,),round_timer=11)
        with patch('touyama_v3.tv3_review_attacker_exposure_teacher.incoming_clear',
                   side_effect=lambda sc,snap,a,p,t:p==risky):
            plan=safe_plant_detour(scenario,snapshot,ally,'R',(self.threat,))
            self.assertEqual(plan['path'],(start,risky,goal))
            self.assertEqual(plan['budget'],2)
            self.assertIsNone(safe_plant_detour(scenario,replace(snapshot,round_timer=10),ally,'R',(self.threat,)))

    def test_committed_detour_cannot_override_current_cover_retreat(self):
        from unittest.mock import patch
        from touyama_v3.tv3_review_attacker_exposure_teacher import ExposureEncoderProbe
        actions=tuple(FrcAction(move,face) for move in ('STAY','N','E','S','W') for face in FACING)
        original=CombatAdvice((4,4),'N','advance',(1,4),0,1,0,20)
        retreat=replace(original,position=(4,5),reason='cover_retreat')
        inputs=NS(actions=actions,teacher=0,mask=np.ones(40,bool),combat=original)
        base=NS(scenario=self.scenario,encode=lambda *args:inputs)
        probe=ExposureEncoderProbe(base)
        probe.detours[0]=('R',{3})
        debug=dict(current=1,future=1)
        route=Route('R',((4,4),(3,4)),('h',))
        ally=replace(self.ally,ability_name='FLASH')
        with patch('touyama_v3.tv3_review_attacker_exposure_teacher.exposure_advice',return_value=(retreat,debug)), \
             patch('touyama_v3.tv3_review_attacker_exposure_teacher.safe_plant_detour') as detour, \
             patch('touyama_v3.tv3_review_attacker_exposure_teacher.validate_action') as utility:
            result=probe.encode(self.snap,ally,(3,4),route,None,{3:((1,4),49,(0,0))},None,0,'supported')
        detour.assert_not_called()
        utility.assert_not_called()
        self.assertEqual(result.combat.reason,'cover_retreat')

    def test_repeated_peek_retreat_cannot_reset_route_progress(self):
        route=Route('R',((12,40),(11,40),(10,40)),('h',))
        base=NS(update=lambda *a,**kw:AttackPlan(route,'supported','entry',False,'observe',(10,40),{}))
        probe=RouteProgressProbe(base)
        for tick,position in ((0,(12,40)),(1,(11,40)),(2,(12,40)),(9,(11,40))):
            probe.progress_tick=tick
            snap=NS(tick=tick,allies=(NS(alive=True,has_spike=True,position=position),))
            probe.update(snap,None,None,None,route,'supported')
        self.assertEqual(probe.progress_tick,1)

    def test_off_route_site_arrival_keeps_actions_available_until_actual_plant_deadline(self):
        from touyama_v3.tv3_review_attacker_exposure_teacher import PlantDeadlineProbe
        base=NS(update=lambda *args,**kwargs:None,progress_tick=0)
        probe=PlantDeadlineProbe(base)
        route=Route('R',((12,40),(11,40),(10,40)),('h',))
        encoder=NS(scenario=NS(sites={'R':((9,40),(10,40))}))
        holder=NS(alive=True,has_spike=True,position=(9,40),plant_progress=0)
        snap=NS(allies=(holder,),round_timer=4)
        plan=probe.update(snap,None,encoder,None,route,'supported')
        self.assertEqual(plan.phase,'plant')
        self.assertFalse(plan.changed)
        self.assertEqual(plan.route,route)
        snap.round_timer=3
        self.assertIsNone(probe.update(snap,None,encoder,None,route,'supported'))
        holder.plant_progress=1
        self.assertIsNotNone(probe.update(snap,None,encoder,None,route,'supported'))
        holder.position=(11,40)
        self.assertIsNone(probe.update(snap,None,encoder,None,route,'supported'))

    def test_real_encoder_keeps_legal_plant_without_forcing_learned_selection(self):
        import torch
        from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
        from touyama_v3.tv3_learn_attacker_plant import ACTION_DIM
        from touyama_v3.tv3_attacker_route_planner import AttackPlan
        from touyama_v3.tv3_review_attacker_exposure_teacher import PlantDeadlineProbe
        from touyama_v3.test.test_tv3_attacker_route_planner import PublicContactModel
        class StayPolicy(torch.nn.Module):
            execution_version=11
            def forward(self,x):
                scores=torch.zeros((len(x),ACTION_DIM)); scores[:,0]=100
                return scores
        for enabled,teacher in ((False,False),(True,True),(True,False)):
            game=attacker_world()
            holder=next(c for c in game.chars if c.has_spike)
            holder.pos=[9,40]
            for c in game.chars:
                if c is not holder:
                    c.hp=0
            game.round_timer=4
            controller=TouyamaV3AttackerPlantController(game,PublicContactModel(),StayPolicy(),training=teacher)
            controller.teacher_probability=1. if teacher else 0.
            controller.set_game(game)
            controller.route=Route('R',((12,40),(11,40),(10,40)),('h',))
            controller.route_mode='supported'
            controller.route_planner.plan=AttackPlan(controller.route,'supported','entry',False,'observe',(10,40),{})
            if enabled:
                controller.route_planner=PlantDeadlineProbe(controller.route_planner)
            controller.prepare_team_tick()
            if not enabled:
                self.assertFalse(controller.plans)
            else:
                _,inputs,_=next(iter(controller.plans.values()))
                self.assertTrue(inputs.mask[-2])
                selected,inputs,_=next(iter(controller.plans.values()))
                self.assertEqual(inputs.actions[selected].kind,'PLANT' if teacher else 'STAY')


if __name__=='__main__':
    unittest.main()
