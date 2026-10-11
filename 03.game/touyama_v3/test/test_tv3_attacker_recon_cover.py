"""Incoming recon fire, reachable wall cover, and preserved legacy plant contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from frc_v1.actions import build_masks,MOVE_STEPS,KINDS
from grid_lines import line_cells,wall_line_cells
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_learn_attacker_fire_support import shot_clear
from touyama_v3.tv3_learn_attacker_recon_cover import (
    incoming_clear,recon_cover_plan,recon_cover_features,RECON_COVER_FEATURES,
)
from touyama_v3.tv3_learn_attacker_plant import PlantEncoder,PlantDQN,policy_schema,observation_dim


class ReconCoverTests(unittest.TestCase):
    def setUp(self):
        self.world=attacker_world()
        base=FrcPerceptionBuilder('A').build(self.world)
        grid=np.zeros((9,9),int); grid[3,4]=1
        ally=replace(base.allies[0],position=(4,3),reveal=15,blind=0,hp=100.,max_hp=100.)
        allies=tuple(ally if a.slot==0 else replace(a,alive=False) for a in base.allies)
        self.snap=replace(base,grid=tuple(map(tuple,grid)),allies=allies,tick=30,
            sightings=(Sighting(0,(4,7),'normal'),),smoke_cells=((4,5),),visible_cells=())
        self.ally=ally
        self.scenario=NS(grid=grid,_line_cells=line_cells,
            clear=lambda a,b:not any(grid[p]==1 for p in wall_line_cells(a,b)))

    def test_reveal_opens_only_incoming_smoke_line_and_cover_teacher_goes_behind_wall(self):
        enemy=self.snap.sightings[0]
        self.assertFalse(shot_clear(self.scenario,self.snap,self.ally.position,enemy.position,0))
        self.assertTrue(incoming_clear(self.scenario,self.snap,self.ally,self.ally.position,enemy))
        self.assertFalse(incoming_clear(self.scenario,self.snap,replace(self.ally,reveal=0),self.ally.position,enemy))
        plan=recon_cover_plan(self.scenario,self.snap,self.ally,{},build_masks(self.snap),(4,4))
        self.assertEqual(plan['step'],(3,3))
        self.assertFalse(incoming_clear(self.scenario,self.snap,self.ally,plan['goal'],enemy))
        features=recon_cover_features(self.scenario,self.snap,self.ally,{},plan)
        self.assertEqual(len(features),RECON_COVER_FEATURES)
        self.assertGreater(features[0],0.)
        self.assertGreater(features[1],0.)
        self.assertEqual(features[3],0.) # N is behind a wall.

    def test_dead_stale_and_confirmed_empty_tracks_do_not_create_retreat(self):
        snap=replace(self.snap,sightings=())
        for s,tracks in (
            (snap,{0:((4,7),0,(0,0))}),
            (replace(snap,visible_cells=((4,7),)),{0:((4,7),29,(0,0))}),
            (replace(snap,enemies=tuple(replace(e,alive=False) for e in snap.enemies)),{0:((4,7),29,(0,0))}),
        ):
            self.assertIsNone(recon_cover_plan(self.scenario,s,self.ally,tracks,build_masks(s),(4,4)))

    def test_cover_search_does_not_take_an_illegal_first_move(self):
        masks=NS(kind=build_masks(self.snap).kind.copy())
        masks.kind[self.ally.slot,KINDS.index('N')]=False
        plan=recon_cover_plan(self.scenario,self.snap,self.ally,{},masks,(4,4))
        self.assertIsNotNone(plan)
        self.assertNotEqual(plan['step'],(3,3))
        dr,dc=plan['step'][0]-self.ally.position[0],plan['step'][1]-self.ally.position[1]
        kind=next(k for k,v in MOVE_STEPS.items() if v==(dr,dc))
        self.assertTrue(masks.kind[self.ally.slot,KINDS.index(kind)])

    def test_safe_position_can_move_safely_but_does_not_repeek_while_marked(self):
        ally=replace(self.ally,position=(3,3))
        snap=replace(self.snap,allies=(ally,)+self.snap.allies[1:])
        self.assertIsNone(recon_cover_plan(self.scenario,snap,ally,{},build_masks(snap),(2,3)))
        self.assertEqual(recon_cover_plan(self.scenario,snap,ally,{},build_masks(snap),(4,3))['step'],(3,3))
        self.assertIsNone(recon_cover_plan(self.scenario,snap,replace(ally,reveal=0),{},build_masks(snap),(4,3)))

    def test_real_right_approach_reaches_wall_cover_without_entering_site(self):
        base=FrcPerceptionBuilder('A').build(self.world)
        ally=replace(base.allies[0],position=(11,40),reveal=15,blind=0,hp=100.,max_hp=100.)
        snap=replace(base,allies=tuple(ally if a.slot==0 else replace(a,alive=False) for a in base.allies),
            sightings=(),smoke_cells=((8,40),),visible_cells=())
        tracks={0:((2,40),snap.tick,(0,0))}
        path=[ally.position]
        for _ in range(12):
            plan=recon_cover_plan(self.world,snap,ally,tracks,build_masks(snap),(10,40))
            if plan is None or plan['steps']==0:
                break
            self.assertNotEqual(plan['step'],ally.position)
            ally=replace(ally,position=plan['step'])
            snap=replace(snap,allies=(ally,)+snap.allies[1:])
            path.append(ally.position)
            if not incoming_clear(self.world,snap,ally,ally.position,Sighting(0,(2,40),'history')):
                break
        self.assertGreater(len(path)-1,0)
        self.assertFalse(incoming_clear(self.world,snap,ally,ally.position,Sighting(0,(2,40),'history')))
        self.assertTrue(all(self.world.grid[p]!=2 for p in path))

    def test_multi_step_corridor_reaches_wall_cover_beyond_three_cells(self):
        grid=np.ones((19,9),int); grid[:,4]=0; grid[12,5]=0
        ally=replace(self.ally,position=(4,4))
        snap=replace(self.snap,grid=tuple(map(tuple,grid)),allies=(ally,)+self.snap.allies[1:],
            sightings=(Sighting(0,(1,4),'normal'),),smoke_cells=((2,4),))
        scenario=NS(grid=grid,clear=lambda a,b:not any(grid[p]==1 for p in wall_line_cells(a,b)))
        plan=recon_cover_plan(scenario,snap,ally,{},build_masks(snap),(3,4))
        self.assertEqual(plan['step'],(5,4))
        self.assertEqual(plan['goal'],(12,5))
        self.assertEqual(plan['steps'],9)
        self.assertFalse(incoming_clear(scenario,snap,ally,plan['goal'],snap.sightings[0]))

    def test_new_encoder_teaches_legal_escape_and_legacy_keeps_advance(self):
        from touyama_v3.tv3_learn_attacker_analysis import Route
        base=FrcPerceptionBuilder('A').build(self.world)
        ally=replace(base.allies[0],position=(11,40),reveal=15,blind=0,hp=100.,max_hp=100.,charges=0)
        snap=replace(base,allies=tuple(ally if a.slot==0 else replace(a,alive=False) for a in base.allies),
            sightings=(),smoke_cells=((8,40),),visible_cells=())
        route=Route('R',((11,40),(10,40)),('h',))
        analysis=dict(placement={i:{**{n:1/14 for n in self.world.names},'dead':0.} for i in range(5)},
                      placement_entropy=1.,trained_rounds=1)
        tracks={0:((2,40),snap.tick,(0,0))}
        old=PlantEncoder(self.world,12).encode(snap,ally,(10,40),route,analysis,tracks,build_masks(snap),0,'supported')
        new=PlantEncoder(self.world,13).encode(snap,ally,(10,40),route,analysis,tracks,build_masks(snap),0,'supported')
        self.assertEqual(old.combat.reason,'advance')
        self.assertEqual(new.combat.reason,'cover_retreat')
        dr,dc=MOVE_STEPS[new.actions[new.teacher].kind]
        self.assertEqual((ally.position[0]+dr,ally.position[1]+dc),new.combat.position)
        self.assertTrue(new.mask[new.teacher])
        self.assertEqual(len(old.observation),951)
        self.assertEqual(len(new.observation),969)
        self.assertTrue(np.isfinite(new.observation).all())
        self.assertNotIn('recon_cover',policy_schema(self.world,12))
        self.assertIn('recon_cover',policy_schema(self.world,13))


if __name__=='__main__':
    unittest.main()
