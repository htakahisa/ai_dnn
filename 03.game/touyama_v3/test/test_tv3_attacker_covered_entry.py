"""Covered approach geometry, real flash effect, and route/assembly context; no learning."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from frc_v1.perception import FrcPerceptionBuilder,Sighting
from frc_v1.actions import build_masks,validate_action,FrcAction
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_review_attacker_safe_entry import covered_paths,corner_flash_options
from touyama_v3.tv3_attacker_entry_utility import projectile_impact
from touyama_v3.tv3_learn_attacker_analysis import Route,shortest_path,branch_sequence,candidate_routes
from touyama_v3.tv3_attacker_site_entry import PlantSiteAssembly


class CoveredEntryTests(unittest.TestCase):
    def test_changed_entry_only_preserves_initial_preparation_and_releases_stale_target(self):
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        snapshot=replace(snapshot,allies=tuple(replace(a,position=(22,35)) if a.has_spike else a
                                              for a in snapshot.allies))
        old=next(r for r in candidate_routes(game,(22,35),100) if r.site=='R')
        legacy=PlantSiteAssembly(game)
        scoped=PlantSiteAssembly(game,follow_route_entry=True,bind_changed_entry_only=True)
        self.assertEqual(legacy.observe(snapshot,old),scoped.observe(snapshot,old))
        stage=next(r for r in covered_paths(game,(22,35),'R',((2,40),)) if r['staging']==(7,37))
        cells=stage['path']+shortest_path(game,stage['staging'],game.sites['R'],game.sites['L'])[1:]
        changed=Route('R',cells,branch_sequence(game,cells))
        self.assertIsNone(scoped.observe(snapshot,changed))
        self.assertFalse(scoped.assigned)
        self.assertTrue(scoped.entry_changed)
        self.assertTrue(scoped.bind_changed_entry_only)
        self.assertIsNone(scoped.observe(snapshot,changed))

    def test_branch_b_d_h_can_reach_c_but_route_knowledge_exposes_it(self):
        from touyama_v3.tv3_review_attacker_entry_candidates import entrance_audit
        game=attacker_world()
        outside=game.retake_entries['R']['c'][0]
        for branch,steps in (('b',11),('d',6),('h',3)):
            cells=game.branches[branch]
            row=entrance_audit(game,cells[len(cells)//2],outside,'R',67,((2,40),))
            self.assertTrue(row['reachable_outside_sites'])
            self.assertEqual(row['approach_steps'],steps)
            self.assertTrue(row['known_wall_exposure'])

    def test_v4_flank_drops_goal_not_reachable_without_entering_other_site(self):
        from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
        from touyama_v3.test.test_tv3_attacker_route_planner import PublicContactModel
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        holder=next(a.slot for a in snapshot.allies if a.has_spike)
        helper=next(a.slot for a in snapshot.allies if not a.has_spike)
        snapshot=replace(snapshot,allies=tuple(
            replace(a,position=(11,40)) if a.slot==holder else
            replace(a,position=(7,3)) if a.slot==helper else replace(a,alive=False)
            for a in snapshot.allies),sightings=(Sighting(0,(7,3),'normal'),))
        encoder=AttackerEncoder(game,4)
        planner=AdaptiveAttackPlanner()
        planner.flank_site='R'
        planner.flank_entries={helper:((7,39),(7,40))}
        plan=planner.update(snapshot,encoder.observe(snapshot,[]),encoder,PublicContactModel())
        self.assertEqual(plan.route.site,'R')
        self.assertNotIn(helper,planner.flank_entries)
        self.assertNotIn(helper,plan.scout_goals)
        self.assertTrue(any(r['slot']==helper and r['reason']=='outside_path_unreachable'
                            for r in planner.events[-1]['rejected_entry_assignments']))

    def test_site_shortcut_must_not_trigger_preparation_at_wrong_entry(self):
        from grid_paths import distance_map
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        snapshot=replace(snapshot,allies=tuple(replace(a,position=(11,40)) if a.has_spike else a
                                              for a in snapshot.allies))
        blocked=set(game.sites['L'])|set(game.sites['R'])
        prefix=shortest_path(game,(11,40),((7,37),),blocked)
        suffix=shortest_path(game,(7,37),game.sites['R'],game.sites['L'])
        route=Route('R',prefix+suffix[1:],branch_sequence(game,prefix+suffix[1:]))
        board=game.grid.copy(); board[board==2]=1
        self.assertEqual(distance_map(game.grid,(7,37))[11,40],7)
        self.assertEqual(distance_map(board,(7,37))[11,40],25)
        legacy=PlantSiteAssembly(game,follow_route_entry=False)
        checked=PlantSiteAssembly(game,follow_route_entry=True)
        self.assertIsNotNone(legacy.observe(snapshot,route))
        self.assertIsNone(checked.observe(snapshot,route))
        self.assertFalse(checked.assigned)

    def test_v4_flank_drops_reachable_goal_when_plant_deadline_is_insufficient(self):
        from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder
        from touyama_v3.test.test_tv3_attacker_route_planner import PublicContactModel
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        holder=next(a.slot for a in snapshot.allies if a.has_spike)
        helper=next(a.slot for a in snapshot.allies if not a.has_spike)
        snapshot=replace(snapshot,round_timer=11,allies=tuple(
            replace(a,position=(11,40)) if a.slot==holder else
            replace(a,position=(7,37)) if a.slot==helper else replace(a,alive=False)
            for a in snapshot.allies),sightings=(Sighting(0,(7,3),'normal'),))
        encoder=AttackerEncoder(game,4)
        planner=AdaptiveAttackPlanner()
        planner.flank_site='R'; planner.flank_entries={helper:((7,39),(7,40))}
        plan=planner.update(snapshot,encoder.observe(snapshot,[]),encoder,PublicContactModel())
        self.assertNotIn(helper,plan.scout_goals)
        self.assertTrue(any(r['slot']==helper and r['reason']=='entry_deadline'
                            for r in planner.events[-1]['rejected_entry_assignments']))

    def test_covered_path_to_flash_staging_is_legal_and_shielded_by_walls(self):
        game=attacker_world()
        rows=covered_paths(game,(22,35),'R',((2,40),))
        stage=next(r for r in rows if r['staging']==(7,37))
        self.assertEqual(stage['steps'],43)
        self.assertTrue(all(not game.clear(p,(2,40)) for p in stage['path']))
        self.assertTrue(all(b in game.neighbors(a) for a,b in zip(stage['path'],stage['path'][1:])))
        self.assertFalse(set(stage['path']).intersection(game.sites['L']+game.sites['R']))

    def test_corner_flash_is_legal_and_real_engine_blinds_known_enemy(self):
        game=attacker_world()
        caster,enemy=game.chars[0],game.chars[-1]
        for char in game.chars:
            if char not in (caster,enemy):
                char.hp=0
        caster.pos=[7,37]; caster.ability_name='FLASH'; caster.flash_charges=1
        enemy.pos=[2,40]
        game.flash_bursts=[]
        game.analytics_tracker=None
        snapshot=FrcPerceptionBuilder('A').build(game)
        options=corner_flash_options(game,(7,37),'R',((2,40),))
        option=next(r for r in options if r['aim']==(4,40))
        self.assertFalse(game.check_cell_line_of_sight(tuple(enemy.pos),tuple(caster.pos)))
        validate_action(snapshot,build_masks(snapshot),0,FrcAction('ABILITY','NE',option['aim']))
        impact,delay=projectile_impact(game,tuple(caster.pos),option['aim'],'FLASH')
        self.assertEqual(delay,1)
        game._explode_flash(dict(team='A',owner=str(caster.name)),impact)
        self.assertGreater(enemy.blind_remaining,0)
        self.assertEqual(caster.blind_remaining,0)

    def test_changed_same_site_entry_discards_old_rally_and_preserves_approach(self):
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        snapshot=replace(snapshot,allies=tuple(replace(a,position=(22,35)) if a.has_spike else a for a in snapshot.allies))
        holder=next(a for a in snapshot.allies if a.has_spike)
        old=next(r for r in candidate_routes(game,holder.position,100) if r.site=='R')
        stage=next(r for r in covered_paths(game,holder.position,'R',((2,40),)) if r['staging']==(7,37))
        cells=stage['path']+shortest_path(game,stage['staging'],game.sites['R'],game.sites['L'])[1:]
        changed=Route('R',cells,branch_sequence(game,cells))
        for follows in (False,True):
            assembly=PlantSiteAssembly(game,follow_route_entry=follows)
            state=assembly.observe(snapshot,old)
            self.assertIsNotNone(state)
            previous=dict(assembly.assigned)
            state=assembly.observe(snapshot,changed)
            if follows:
                self.assertIsNone(state)
                self.assertFalse(assembly.assigned)
                near=replace(snapshot,allies=tuple(replace(a,position=(7,37)) if a.has_spike else a for a in snapshot.allies))
                self.assertIsNotNone(assembly.observe(near,changed))
                self.assertEqual(assembly.primary_entry,((7,39),(7,40)))
            else:
                self.assertEqual(assembly.assigned,previous)

    def test_corner_flash_candidate_covers_two_enemies_where_legacy_offers_no_legal_cast(self):
        import numpy as np
        from touyama_v3.tv3_learn_attacker_plant import PlantEncoder,observation_dim
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        allies=tuple(replace(a,position=(8,31),ability_name='FLASH',charges=1,forced_facing=False,
                             blind=0,has_spike=False) if a.slot==0 else replace(a,alive=False,has_spike=False)
                     for a in snapshot.allies)
        positions={3:(7,38),4:(2,40)}
        snapshot=replace(snapshot,allies=allies,
            enemies=tuple(replace(e,alive=e.enemy_id in positions) for e in snapshot.enemies),
            sightings=tuple(Sighting(i,p,'reveal') for i,p in positions.items()),
            visible_cells=tuple(positions.values()),smoke_cells=())
        placement={i:{**{n:0. for n in game.names},'dead':float(i not in positions)} for i in range(5)}
        for i,p in positions.items():
            placement[i][game.names[int(game.region[p])]]=1.
        analysis=dict(placement=placement,placement_entropy=0.,trained_rounds=120)
        cells=shortest_path(game,(8,31),game.sites['R'],game.sites['L'])
        route=Route('R',cells,branch_sequence(game,cells))
        tracks={i:(p,snapshot.tick,(0,0)) for i,p in positions.items()}
        inputs={v:PlantEncoder(game,v).encode(snapshot,allies[0],route.cells[-1],route,analysis,tracks,
                                             build_masks(snapshot),0,'supported') for v in (11,12)}
        self.assertFalse(inputs[11].mask[40:48].any())
        proposed=inputs[12]
        self.assertEqual(len(proposed.observation),observation_dim(12))
        self.assertTrue(np.isfinite(proposed.observation).all())
        cast=proposed.actions[proposed.teacher]
        self.assertEqual(cast.kind,'ABILITY')
        self.assertEqual(cast.target,(6,40))
        impact,delay=projectile_impact(game,allies[0].position,cast.target,'FLASH')
        self.assertTrue(all(game.clear(impact,p) for p in positions.values()))
        self.assertEqual(delay,3)

    def test_owned_flash_window_requires_public_burst_and_fresh_target(self):
        from touyama_v3.tv3_learn_attacker_corner_utility import FriendlyFlashTracker
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        snapshot=replace(snapshot,tick=88,sightings=(Sighting(4,(2,40),'normal'),),
                         effects=(NS(kind='FLASH',phase='active',position=(6,40)),))
        tracks={4:((2,40),84,(0,0))}
        tracker=FriendlyFlashTracker(game)
        self.assertFalse(tracker.suppressed(snapshot,(2,40),tracks))
        tracker.record(84,(7,33),(6,42))
        self.assertTrue(tracker.suppressed(snapshot,(2,40),tracks))
        # Display can expire before blindness; retain the public confirmed forecast.
        self.assertTrue(tracker.suppressed(replace(snapshot,tick=92,effects=()),(2,40),tracks))
        self.assertFalse(tracker.suppressed(replace(snapshot,tick=97,effects=()),(2,40),tracks))
        self.assertFalse(tracker.suppressed(replace(snapshot,tick=94,sightings=(),effects=()),(2,40),tracks))

    def test_expected_flash_cannot_suppress_before_burst_or_through_smoke(self):
        from touyama_v3.tv3_learn_attacker_corner_utility import FriendlyFlashTracker
        game=attacker_world()
        snapshot=replace(FrcPerceptionBuilder('A').build(game),tick=88,
                         sightings=(Sighting(4,(2,40),'normal'),),effects=())
        tracker=FriendlyFlashTracker(game)
        tracker.record(84,(7,33),(6,42))
        self.assertFalse(tracker.suppressed(snapshot,(2,40),{}))
        self.assertFalse(tracker.suppressed(replace(snapshot,tick=89,effects=(),smoke_cells=()),(2,40),{}))
        effect=NS(kind='FLASH',phase='active',position=(6,40))
        snapshot=replace(snapshot,effects=(effect,),smoke_cells=((3,40),))
        self.assertFalse(tracker.suppressed(snapshot,(2,40),{}))
        self.assertFalse(tracker.suppressed(replace(snapshot,tick=89,effects=(),smoke_cells=()),(2,40),{}))

    def test_live_controller_confirms_own_cast_from_public_charge_without_training_hook(self):
        import torch
        from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
        from touyama_v3.tv3_learn_attacker_plant import ACTION_DIM
        from touyama_v3.test.test_tv3_attacker_route_planner import PublicContactModel
        class CastPolicy(torch.nn.Module):
            execution_version=12
            def forward(self,x):
                q=torch.zeros((len(x),ACTION_DIM)); q[:,40:48]=100
                return q
        class RightModel(PublicContactModel):
            def analyze(self,*args,**kwargs):
                result=super().analyze(*args,**kwargs)
                result['candidates']=[c for c in result['candidates'] if c['route'].site=='R']
                return result
        game=attacker_world()
        snapshot=FrcPerceptionBuilder('A').build(game)
        snapshot=replace(snapshot,allies=tuple(replace(a,position=(8,31),ability_name='FLASH',charges=1,
            forced_facing=False,blind=0,has_spike=True) if a.slot==0 else replace(a,alive=False,has_spike=False)
            for a in snapshot.allies),sightings=(Sighting(4,(2,40),'reveal'),Sighting(3,(7,38),'reveal')),
            enemies=tuple(replace(e,alive=e.enemy_id in (3,4)) for e in snapshot.enemies),smoke_cells=())
        controller=TouyamaV3AttackerPlantController(game,RightModel(),CastPolicy())
        controller.set_game(game)
        controller.sensor.build=lambda _:snapshot
        _,command=controller.decide_move(game.chars[0],{})
        self.assertEqual(command.get('ability'),'FLASH',repr(command)+' '+repr(
            [(name,i,[(j,x.actions[j].target,bool(x.mask[j])) for j in range(40,48)])
             for name,(i,x,a) in controller.plans.items()]))
        self.assertFalse(controller.encoder.flash_tracker.records)
        game.battle_tick=1
        snapshot=replace(snapshot,tick=1,allies=tuple(replace(a,charges=0) if a.slot==0 else a for a in snapshot.allies))
        controller.prepare_team_tick()
        self.assertEqual(len(controller.encoder.flash_tracker.records),1)


if __name__=='__main__':
    unittest.main()
