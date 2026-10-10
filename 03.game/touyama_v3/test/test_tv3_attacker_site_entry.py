"""Rally, public traffic and supported-fight demonstrations; no training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile
import unittest
import numpy as np
import torch

from grid_paths import distance_map
from touyama_v3.tv3_attacker_site_entry import (
    PlantSiteAssembly, traffic_distances, site_entry_features, ENTRY_FEATURES,
    teacher_distances, mission_progress, MAX_ENTRY_WAIT,
)
from touyama_v3.tv3_learn_attacker_plant import PlantInputs, PlantEncoder, load_plant, policy_schema
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController as Controller
from touyama_v3.tv3_attacker_combat import CombatAdvice


class SiteEntryTests(unittest.TestCase):
    def setUp(self):
        grid = np.zeros((9, 13), np.int32)
        grid[2:5, 9] = 2
        self.scenario = NS(grid=grid, sites={'L': tuple((r,9) for r in range(2,5)), 'R': ()})
        self.scenario.neighbors = lambda p: [(p[0]+dr,p[1]+dc) for dr,dc in ((0,1),(0,-1),(1,0),(-1,0))
            if 0 <= p[0]+dr < 9 and 0 <= p[1]+dc < 13 and grid[p[0]+dr,p[1]+dc] != 1]
        self.scenario.site_dist = {'L': np.minimum.reduce([distance_map(grid,p) for p in self.scenario.sites['L']])}
        self.allies = tuple(NS(slot=i, position=(i+1,2), alive=True, has_spike=i==2, plant_progress=0) for i in range(5))
        self.snapshot = NS(grid=grid, allies=self.allies, tick=0, round_timer=100)
        self.route = NS(site='L', cells=tuple((3,c) for c in range(2,10)))
        self.assembly = PlantSiteAssembly(self.scenario)

    def test_distinct_rally_cells_and_two_entries_without_private_enemy_state(self):
        state = self.assembly.observe(self.snapshot, self.route)
        self.assertEqual(len(set(state['goals'].values())), 5)
        self.assertEqual(len(state['flank']), 2)
        self.assertTrue(all(self.scenario.grid[p] != 2 for p in state['goals'].values()))
        self.assertGreater(len(set(self.assembly.entries.values())), 1)
        self.assertFalse(state['launched'])
        self.snapshot.hidden_enemy_positions = ((2,9),(3,9),(4,9))
        repeat = PlantSiteAssembly(self.scenario).observe(self.snapshot,self.route)
        self.assertEqual(state, repeat)

    def test_rally_is_persistent_and_launches_after_all_arrive(self):
        state = self.assembly.observe(self.snapshot, self.route)
        original = dict(state['goals'])
        for a in self.allies:
            a.position = original[a.slot]
        self.snapshot.tick=3
        launched = self.assembly.observe(self.snapshot, self.route)
        self.assertTrue(launched['launched'])
        self.assertEqual(launched['ready'],set(range(5)))
        self.assertTrue(all(self.scenario.grid[p] == 2 for p in launched['goals'].values()))
        self.assertEqual(self.assembly.assigned, original)

    def test_wait_is_bounded_and_dead_players_do_not_block_launch(self):
        first=self.assembly.observe(self.snapshot,self.route)
        for a in self.allies[:2]:
            a.position=first['goals'][a.slot]
        self.assembly.observe(self.snapshot,self.route)
        self.snapshot.tick=7
        self.assertTrue(self.assembly.observe(self.snapshot,self.route)['launched'])
        fresh=PlantSiteAssembly(self.scenario)
        state=fresh.observe(self.snapshot,self.route)
        for a in self.allies:
            if a.slot==2:
                a.position=state['goals'][a.slot]
            else:
                a.alive=False
        self.assertTrue(fresh.observe(self.snapshot,self.route)['launched'])

    def test_only_one_ready_player_cannot_wait_beyond_the_limit(self):
        first=self.assembly.observe(self.snapshot,self.route)
        self.allies[2].position=first['goals'][2]
        waiting=self.assembly.observe(self.snapshot,self.route)
        self.assertEqual(waiting['ready'],{2})
        self.snapshot.tick=MAX_ENTRY_WAIT
        state=self.assembly.observe(self.snapshot,self.route)
        self.assertTrue(state['launched'])
        self.assertFalse(state['wait'])
        self.assertEqual(self.scenario.grid[state['goals'][2]],2)

    def test_spike_retriever_does_not_stay_at_the_previous_support_goal(self):
        first=self.assembly.observe(self.snapshot,self.route)
        self.allies[2].alive=False
        self.allies[2].has_spike=False
        self.allies[0].has_spike=True
        self.allies[0].position=first['goals'][0]
        recovered=self.assembly.observe(self.snapshot,self.route)
        self.assertTrue(recovered['launched'])
        self.assertNotEqual(recovered['goals'][0],first['goals'][0])
        self.assertEqual(self.scenario.grid[recovered['goals'][0]],2)

    def test_traffic_teacher_rejects_long_detours_around_moving_allies(self):
        static=distance_map(self.scenario.grid,(3,9))
        position=(3,2)
        self.assertIs(teacher_distances(static,static+10,position),static)
        short=static+2
        self.assertIs(teacher_distances(static,short,position),short)
        self.assertIs(teacher_distances(static,np.full_like(static,-1),position),static)

    def test_progress_toward_rally_but_away_from_plant_is_not_rewarded(self):
        self.assertEqual(mission_progress(self.scenario,(3,6),(3,5)),-1.)
        self.assertEqual(mission_progress(self.scenario,(3,5),(3,6)),1.)
        self.assertEqual(mission_progress(self.scenario,(3,6),(3,6)),0.)

    def test_round_trip_never_earns_progress_despite_moving_blockers(self):
        loop=((3,6),(3,5),(4,5),(4,6),(3,6))
        reward=sum(mission_progress(self.scenario,a,b) for a,b in zip(loop,loop[1:]))
        self.assertEqual(reward,0.)
        # Existing tick penalty makes the trip strictly worse than no trip.
        from touyama_v3.tv3_train_attacker_plant import TICK_PENALTY,PROGRESS_REWARD
        self.assertLess(PROGRESS_REWARD*reward-(len(loop)-1)*TICK_PENALTY,0.)

    def test_teleport_away_then_walk_back_cannot_exploit_clipped_progress(self):
        loop=((3,8),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8))
        self.assertEqual(sum(mission_progress(self.scenario,a,b) for a,b in zip(loop,loop[1:])),0.)

    def test_low_clock_releases_rally_and_opposite_side_resets_it(self):
        self.assembly.observe(self.snapshot,self.route)
        self.snapshot.round_timer=10
        self.assertTrue(self.assembly.observe(self.snapshot,self.route)['launched'])
        self.scenario.sites['R']=self.scenario.sites['L']
        self.scenario.site_dist['R']=self.scenario.site_dist['L']
        self.snapshot.round_timer=100
        self.route.site='R'
        self.assertFalse(self.assembly.observe(self.snapshot,self.route)['launched'])

    def test_traffic_detour_can_advance_when_static_distance_increases(self):
        grid=np.zeros((7,7),np.int32)
        ally=NS(slot=0,position=(3,1),alive=True,plant_progress=0)
        blockers=[NS(slot=i+1,position=p,alive=True) for i,p in enumerate(((2,2),(3,2),(4,2)))]
        snap=NS(grid=grid,allies=(ally,*blockers),round_timer=50)
        traffic=traffic_distances(snap,ally,(3,5))
        static=distance_map(grid,(3,5))
        self.assertGreater(static[2,1],static[ally.position])
        self.assertLess(traffic[2,1],traffic[ally.position])
        features=site_entry_features(snap,ally,(3,5),traffic,None)
        self.assertEqual(len(features),ENTRY_FEATURES)
        self.assertEqual(features[3],1.)  # North decreases the traffic distance.
        mask=np.ones(58,bool)
        inputs=NS(mask=mask, distances=static, traffic=traffic)
        self.assertTrue(Controller._has_progress_move(ally,inputs))

    def test_old_plant_rejected_but_analysis_schema_is_unchanged(self):
        from touyama_v3.tv3_scenario import Scenario
        scenario=Scenario()
        schema=policy_schema(scenario)
        schema.pop('site_entry')
        schema['version']=7
        schema['obs_dim']-=ENTRY_FEATURES
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'old.pt'
            torch.save(dict(schema=schema,model={}),p)
            with self.assertRaisesRegex(ValueError,'schema mismatch'):
                load_plant(p,scenario)

    def test_rally_does_not_override_learned_stay_action(self):
        from touyama_v3.test.tv3_fixtures import attacker_world
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, Route
        from touyama_v3.tv3_attacker_route_planner import AttackPlan
        from touyama_v3.tv3_learn_attacker_plant import ACTION_DIM
        game=attacker_world()
        encoder=AttackerEncoder(game)
        model=AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(game.names))
        class Stay(torch.nn.Module):
            def forward(self,x):
                values=torch.zeros((len(x),ACTION_DIM))
                values[:,0]=100
                return values
        controller=Controller(game,model,Stay(),training=False)
        controller.set_game(game)
        for a,p in zip(game.chars[:5],((15,3),(16,3),(17,3),(18,3),(19,3))):
            a.pos=list(p)
        route=Route('L',tuple((r,3) for r in range(17,8,-1)),('approach','site'))
        plan=AttackPlan(route,'supported','entry',True,'test',route.cells[-1],{})
        with patch.object(controller.route_planner,'update',return_value=plan):
            controller.prepare_team_tick()
        self.assertTrue(controller.site_entry_state['active'])
        for name,(index,inputs,ally) in controller.plans.items():
            self.assertEqual(inputs.actions[index].kind,'STAY')
            self.assertTrue(inputs.mask[index])

    def test_no_live_enemy_skips_unnecessary_rally(self):
        self.snapshot.enemies=(NS(alive=False),)*5
        self.assertIsNone(self.assembly.observe(self.snapshot,self.route))

    def test_fast_traffic_map_matches_original_bfs_and_returns_independent_arrays(self):
        rng=np.random.default_rng(42)
        for _ in range(30):
            grid=(rng.random((9,13)) < .2).astype(np.int32)
            free=[tuple(map(int,p)) for p in np.argwhere(grid==0)]
            selected=rng.choice(len(free),7,replace=False)
            positions=[free[i] for i in selected]
            allies=tuple(NS(slot=i,position=p,alive=True) for i,p in enumerate(positions[:5]))
            snap=NS(grid=grid,allies=allies)
            ally=allies[0]
            for goal in (positions[5], allies[1].position):
                reserved=(positions[6],ally.position,goal)
                reference=grid.copy()
                for p in reserved:
                    if p != ally.position:
                        reference[p]=1
                for a in allies[1:]:
                    if a.position != goal:
                        reference[a.position]=1
                reference[goal]=grid[goal]
                expected=distance_map(reference,goal)
                actual=traffic_distances(snap,ally,goal,reserved)
                np.testing.assert_array_equal(actual,expected)
                actual[:]=-999
                np.testing.assert_array_equal(traffic_distances(snap,ally,goal,reserved),expected)

    def test_legacy_v9_two_contact_teacher_and_carrier_order_are_preserved(self):
        from frc_v1.perception import FrcPerceptionBuilder
        from frc_v1.actions import build_masks
        from touyama_v3.test.tv3_fixtures import attacker_world
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
        from touyama_v3.tv3_collect_attacker_analysis import AnalysisCollectorController
        game=attacker_world()
        encoder=AttackerEncoder(game)
        model=AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(game.names))
        controller=AnalysisCollectorController(game,model,np.random.default_rng(0))
        controller.set_game(game)
        controller.prepare_team_tick()
        snap=controller.snapshot
        ally=replace(snap.allies[0],charges=0)
        belief=model.analyze(encoder,snap,controller.frames[0]['observation'],[controller.route])
        advice=CombatAdvice((ally.position[0]+1,ally.position[1]),ally.facing,'cover_retreat',None,2,2,3,0)
        plant=PlantEncoder(game, execution_version=9)
        with patch.object(plant.combat_coach,'advise',return_value=advice), \
             patch('touyama_v3.tv3_learn_attacker_plant.predicted_entry_utility',return_value=None):
            inputs=plant.encode(snap,ally,controller.route.cells[-1],controller.route,belief,{},build_masks(snap),0,'supported')
        self.assertEqual(inputs.combat.reason,'stop_shoot')
        self.assertEqual(inputs.actions[inputs.teacher].kind,'STAY')
        from touyama_v3.tv3_learn_attacker_plant import CARRIER_FEATURE_INDEX
        expected=[float(a.has_spike) for a in sorted(snap.allies,key=lambda a:(a.position,a.ability_name))]
        np.testing.assert_array_equal(inputs.observation[CARRIER_FEATURE_INDEX+10:CARRIER_FEATURE_INDEX+15],expected)


if __name__=='__main__':
    unittest.main()
