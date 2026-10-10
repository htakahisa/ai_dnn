"""Public combat geometry and genuine legacy execution; no learning updates."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from frc_v1.perception import FrcPerceptionBuilder, Sighting
from frc_v1.actions import build_masks
from grid_lines import wall_line_cells, line_cells
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_attacker_combat import CombatAdvice, combat_schema, effective_utility
from touyama_v3.tv3_learn_attacker_fire_support import (
    public_threats, shot_clear, target_support, fire_support_features,
    defensive_advice, smoke_screen,
)
from touyama_v3.tv3_learn_attacker_plant import PlantDQN, PlantEncoder, policy_schema, load_plant


class FireSupportTests(unittest.TestCase):
    def setUp(self):
        self.game = attacker_world()
        snap = FrcPerceptionBuilder('A').build(self.game)
        grid = np.zeros((9, 9), dtype=int)
        positions = ((4,3), (0,0), (0,2), (0,4), (0,6))
        allies = tuple(replace(a, position=positions[a.slot], blind=0 if a.slot==0 else 5,
                               facing='E', charges=0, forced_facing=False) for a in snap.allies)
        self.snap = replace(snap, tick=30, grid=tuple(map(tuple,grid)), allies=allies,
                            sightings=(Sighting(0,(4,7),'normal'),), visible_cells=())
        self.grid = grid
        self.scenario = NS(grid=grid, _line_cells=line_cells,
                           clear=lambda a,b: not any(grid[p]==1 for p in wall_line_cells(a,b)))
        self.ally = allies[0]

    def test_recent_history_is_invalidated_by_empty_visibility_death_and_age(self):
        tracks={0:((4,6),29,None),1:((5,7),29,None),2:((6,7),0,None)}
        threats=public_threats(self.snap,tracks)
        self.assertEqual([s.enemy_id for s in threats],[0,1])
        self.assertEqual(threats[0].position,(4,7))
        self.assertEqual(threats[1].source,'history')
        empty=replace(self.snap,visible_cells=((5,7),))
        self.assertEqual(len(public_threats(empty,tracks)),1)
        dead=replace(self.snap,enemies=tuple(replace(e,alive=False) if e.enemy_id==1 else e for e in self.snap.enemies))
        self.assertEqual(len(public_threats(dead,tracks)),1)

    def test_public_enemy_body_blocks_far_target_but_not_adjacent_shot(self):
        snap=replace(self.snap,sightings=self.snap.sightings+(Sighting(1,(4,5),'normal'),))
        self.assertFalse(shot_clear(self.scenario,snap,(4,3),(4,7),0))
        self.assertTrue(shot_clear(self.scenario,snap,(4,3),(4,5),0))
        smoke=replace(snap,smoke_cells=((4,3),(4,4)))
        self.assertTrue(shot_clear(self.scenario,smoke,(4,3),(4,4),0))

    def test_smoke_at_endpoint_blocks_normal_shot_but_public_recon_allows_shot(self):
        smoke=replace(self.snap,smoke_cells=((4,3),))
        self.assertFalse(shot_clear(self.scenario,smoke,(4,3),(4,7),0))
        revealed=replace(smoke,sightings=(Sighting(0,(4,7),'reveal'),))
        self.assertTrue(shot_clear(self.scenario,revealed,(4,3),(4,7),0))

    def test_support_requires_same_target_facing_and_unblocked_line(self):
        helper=replace(self.snap.allies[1],position=(6,3),blind=0)
        snap=replace(self.snap,allies=(self.ally,helper)+self.snap.allies[2:])
        self.assertEqual(target_support(self.scenario,snap,(4,7),0),1)
        wrong=replace(snap,allies=(self.ally,replace(helper,facing='W'))+snap.allies[2:])
        self.assertEqual(target_support(self.scenario,wrong,(4,7),0),0)
        self.grid[5,5]=1
        self.assertEqual(target_support(self.scenario,snap,(4,7),0),0)

    def test_single_unsupported_contact_teaches_legal_cover_not_forced_inference(self):
        self.grid[3,4]=1
        self.snap=replace(self.snap,grid=tuple(map(tuple,self.grid)))
        advice=CombatAdvice(self.ally.position,'E','stop_shoot',(4,7),1,1,4,0)
        result=defensive_advice(self.scenario,self.snap,self.ally,advice,{},build_masks(self.snap))
        self.assertEqual(result.reason,'cover_retreat')
        self.assertEqual(result.position,(3,3))
        self.assertEqual(result.supporters,0)
        features=fire_support_features(self.scenario,self.snap,self.ally,{})
        self.assertEqual(features.shape,(20,))
        self.assertGreater(features[2],0)
        self.assertEqual(features[4],0)

    def test_supported_single_contact_keeps_stationary_shooting(self):
        self.grid[3,4]=1
        helper=replace(self.snap.allies[1],position=(6,3),blind=0)
        snap=replace(self.snap,grid=tuple(map(tuple,self.grid)),allies=(self.ally,helper)+self.snap.allies[2:])
        advice=CombatAdvice(self.ally.position,'E','stop_shoot',(4,7),1,1,1,0)
        result=defensive_advice(self.scenario,snap,self.ally,advice,{},build_masks(snap))
        self.assertEqual(result.reason,'stop_shoot')
        self.assertEqual(result.position,self.ally.position)

    def test_memory_utility_and_screen_cover_actual_public_fire_line(self):
        snap=replace(self.snap,sightings=())
        threats=public_threats(snap,{0:((4,7),29,None)})
        mask=np.ones((9,9),bool)
        target=smoke_screen(self.scenario,snap,self.ally,threats,mask)
        self.assertIsNotNone(target)
        cast={'ability':'SMOKE','target':target}
        self.assertFalse(effective_utility(self.scenario,snap,self.ally,cast))
        self.assertTrue(effective_utility(self.scenario,replace(snap,sightings=threats),self.ally,cast))
        self.assertIsNone(smoke_screen(self.scenario,snap,self.ally,threats,np.zeros((9,9),bool)))

    def test_legacy_schema_load_preserves_weights_dimension_and_controller_execution(self):
        import torch
        from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
        model=PlantDQN(9)
        saved={'schema':policy_schema(self.game,9),'opponent':'frc_v1','analysis_hash':'fixed',
               'model':model.state_dict()}
        with patch('touyama_v3.tv3_learn_attacker_plant.load_checkpoint',return_value=saved):
            loaded,_=load_plant('unused',self.game,'frc_v1','fixed')
        x=torch.zeros((1,872))
        torch.testing.assert_close(model(x),loaded(x),rtol=0,atol=0)
        controller=TouyamaV3AttackerPlantController(self.game,None,loaded)
        self.assertEqual(controller.encoder.execution_version,9)
        self.assertEqual(policy_schema(self.game,10)['obs_dim'],892)
        self.assertEqual(policy_schema(self.game,9)['combat'],combat_schema())
        self.assertNotIn('fire_support',policy_schema(self.game,9))
        saved['schema']={**saved['schema'],'version':8}
        with patch('touyama_v3.tv3_learn_attacker_plant.load_checkpoint',return_value=saved):
            with self.assertRaises(ValueError):
                load_plant('unused',self.game)

    def test_new_encoder_builds_finite_features_without_restricting_escape_to_teacher(self):
        from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
        from touyama_v3.tv3_collect_attacker_analysis import AnalysisCollectorController
        encoder=AttackerEncoder(self.game)
        model=AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(self.game.names))
        collector=AnalysisCollectorController(self.game,model,np.random.default_rng(0))
        collector.set_game(self.game)
        collector.prepare_team_tick()
        snap=collector.snapshot
        ally=snap.allies[0]
        analysis=model.analyze(encoder,snap,collector.frames[0]['observation'],[collector.route])
        result=PlantEncoder(self.game).encode(snap,ally,collector.route.cells[-1],collector.route,
            analysis,{},build_masks(snap),0,'supported')
        self.assertEqual(len(result.observation),892)
        self.assertTrue(np.isfinite(result.observation).all())
        self.assertTrue(result.mask[result.teacher])
        self.assertTrue(any(result.mask[:8]))  # Waiting remains a learned alternative.


if __name__=='__main__':
    unittest.main()
