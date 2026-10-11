"""Selected-site labels and explicit legacy checkpoint compatibility; no optimization."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import torch
from touyama_v3.tv3_scenario import Scenario
from touyama_v3.tv3_learn_attacker_analysis import (
    AttackerEncoder,AttackerAnalysisModel,completion_target,load_analysis,candidate_routes)
from touyama_v3.tv3_review_attacker_route_risk import entry_routes


class AnalysisSiteCompletionTests(unittest.TestCase):
    def test_prediction_time_metric_uses_success_frames_after_site_change(self):
        from touyama_v3.tv3_train_attacker_analysis import prediction_metrics
        class Neutral(torch.nn.Module):
            def forward(self,x,z):
                return torch.zeros((len(x),5,2)),torch.zeros((len(x),5))
        targets=np.zeros((2,5),np.float32)
        targets[1,0]=1.; targets[1,4]=.2
        sample=dict(observations=np.zeros((2,1),np.float32),routes=np.zeros((2,1),np.float32),
                    outcomes=targets,placements=np.zeros((2,5),np.int64))
        metrics=prediction_metrics(Neutral(),[sample])
        self.assertAlmostEqual(metrics['plant_ticks_mae'],30.,places=5)

    def test_prediction_metrics_expose_overconfidence_worse_than_constant_half(self):
        from touyama_v3.tv3_train_attacker_analysis import prediction_metrics
        class Overconfident(torch.nn.Module):
            def forward(self,x,z):
                logits=torch.zeros((len(x),5))
                logits[:,0]=torch.logit(torch.tensor(.99))
                return torch.zeros((len(x),5,2)),logits
        sample=dict(observations=np.zeros((4,1),np.float32),routes=np.zeros((4,1),np.float32),
                    outcomes=np.zeros((4,5),np.float32),placements=np.zeros((4,5),np.int64))
        metrics=prediction_metrics(Overconfident(),[sample])
        self.assertAlmostEqual(metrics['brier'],.9801,places=5)
        self.assertEqual(metrics['constant_half_brier'],.25)
        self.assertLess(metrics['brier_skill_vs_constant_half'],0.)
        self.assertIsNone(prediction_metrics(Overconfident(),[])['brier_skill_vs_constant_half'])

    def test_entry_prediction_metrics_count_rounds_and_selected_site_outcomes(self):
        from touyama_v3.tv3_train_attacker_analysis import entry_prediction_metrics
        no_contact=dict(site='R',public_contact_count=0,predicted_plant_probability=.1,
                        entry_boundary=dict(outside=(11,40),inside=(10,40)))
        contact={**no_contact,'public_contact_count':1,'predicted_plant_probability':.99}
        rows=[dict(planted=True,site='R' if i==0 else 'L',
                   route_decisions=[no_contact]+[contact]*30) for i in range(5)]
        metrics=entry_prediction_metrics(rows)
        group=metrics['groups']['R:11,40:p>=0.9']
        self.assertEqual(metrics['matched_rounds'],5)
        self.assertEqual(group['rounds'],5)
        self.assertEqual(group['plants'],1)
        self.assertAlmostEqual(group['plant_rate'],.2)
        self.assertAlmostEqual(group['mean_prediction'],.99)
        self.assertEqual(entry_prediction_metrics([dict(planted=False,site=None,route_decisions=[])])['groups'],{})

    def test_diagnostic_entry_is_boundary_not_first_inside_site_cell(self):
        from touyama_v3.tv3_learn_attacker_analysis import Route,route_entry_positions
        scenario=Scenario()
        route=Route('R',((11,40),(10,40)),('h',))
        self.assertEqual(route_entry_positions(scenario,route),((11,40),(10,40)))
        inside=Route('R',((10,40),),('h',))
        self.assertEqual(route_entry_positions(scenario,inside),((-1,-1),(-1,-1)))

    def test_entry_credit_ignores_positions_already_inside_the_site(self):
        from touyama_v3.tv3_review_attacker_risk_learning import entry_credit
        rows=[dict(planted=True,site='R',round=2,route_decisions=[
            dict(tick=10,site='R',entry=[11,40]),
            dict(tick=20,site='R',entry=[7,39]),
            dict(tick=21,site='R',entry=[7,40])])]
        result=entry_credit(rows,Scenario())
        self.assertEqual(result['planted_rounds_with_same_site_entry_change'],1)
        self.assertEqual(result['earlier_c_decisions_before_other_right_entry'],1)
        self.assertEqual(result['ignored_non_boundary_entries'],1)
        self.assertEqual(result['examples'][0]['final_selected_entry'],[7,39])

    def test_cross_site_switch_is_not_selected_site_success(self):
        terminal=dict(planted=True,site='L')
        self.assertEqual(completion_target(terminal,'R',3),1.)
        self.assertEqual(completion_target(terminal,'R',4),0.)
        self.assertEqual(completion_target(terminal,'L',4),1.)
        self.assertEqual(completion_target(dict(planted=False,site=None),'R',4),0.)
        with self.assertRaises(ValueError):
            completion_target(dict(planted=True,site=None),'R',4)

    def test_versions_load_their_own_strict_contract_and_reject_forged_target(self):
        scenario=Scenario()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'analysis.pt'
            for version in (3,4):
                encoder=AttackerEncoder(scenario,version)
                model=AttackerAnalysisModel(len(encoder.fields),len(encoder.route_fields),len(scenario.names),version)
                state=dict(schema=encoder.schema(),model=model.state_dict(),opponent='frc_v1',trained_rounds=0)
                torch.save(state,path)
                loaded,restored,_=load_analysis(path,scenario,'frc_v1')
                self.assertEqual(loaded.execution_version,version)
                self.assertEqual(restored.schema(),encoder.schema())
                state['schema']['success_target']='unverified_target'
                torch.save(state,path)
                with self.assertRaisesRegex(ValueError,'schema'):
                    load_analysis(path,scenario,'frc_v1')
            with self.assertRaises(ValueError):
                AttackerEncoder(scenario,5)

    def test_entry_geometry_is_not_discarded_only_because_region_letters_match(self):
        grid=np.zeros((3,5),int)
        def neighbors(p):
            return tuple(q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))
                         if 0<=q[0]<3 and 0<=q[1]<5)
        scenario=NS(grid=grid,region=grid,names=('a',),branches={'a':((0,2),)},
                    sites={'L':((1,0),),'R':((1,4),(2,4))},neighbors=neighbors)
        current=[r for r in candidate_routes(scenario,(1,2),100) if r.site=='R']
        diagnostic=[r for r in entry_routes(scenario,(1,2),100) if r.site=='R']
        self.assertEqual(len(current),1)
        self.assertGreater(len({r.cells[-1] for r in diagnostic}),1)
        self.assertEqual({r.branches for r in diagnostic},{('a',)})
        for route in diagnostic:
            self.assertEqual(len(route.cells),len(set(route.cells)))
            self.assertNotIn((1,0),route.cells)
            self.assertTrue(all(b in neighbors(a) for a,b in zip(route.cells,route.cells[1:])))

    def test_approach_danger_is_distinct_from_unavoidable_entry_danger(self):
        from frc_v1.perception import FrcPerceptionBuilder,Sighting
        from touyama_v3.test.tv3_fixtures import attacker_world
        game=attacker_world()
        snapshot=replace(FrcPerceptionBuilder('A').build(game),tick=33,
                         sightings=(Sighting(4,(2,40),'normal'),),visible_cells=())
        encoder=AttackerEncoder(game,4)
        encoder.history.tracks={4:((2,40),33,(0,0))}
        routes=candidate_routes(game,(22,35),67)
        unsafe=next(r for r in routes if r.site=='R' and r.branches==('b','d','h'))
        covered=next(r for r in routes if r.site=='R' and r.branches==('b','a','f','k','l'))
        values=[dict(zip(encoder.route_fields,encoder.route_features(snapshot,r,'preserve'))) for r in (unsafe,covered)]
        self.assertGreater(values[0]['known_approach_wall_exposure'],0.)
        self.assertEqual(values[1]['known_approach_wall_exposure'],0.)
        self.assertGreater(values[1]['known_entry_wall_exposure'],0.)
        self.assertEqual(len(AttackerEncoder(game,3).route_fields),53)
        self.assertEqual(len(encoder.route_fields),55)


if __name__=='__main__':
    unittest.main()
