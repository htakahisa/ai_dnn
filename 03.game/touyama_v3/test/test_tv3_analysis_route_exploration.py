"""Route exploration, fixed executor parity and immutable collection provenance."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch
import hashlib
import tempfile
import unittest

import numpy as np
import torch

from frc_v1.perception import FrcPerceptionBuilder, Sighting
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_attacker_route_planner import AdaptiveAttackPlanner
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel, candidate_routes
from touyama_v3.tv3_learn_attacker_plant import PlantDQN
from touyama_v3.tv3_attacker_plant_controller import TouyamaV3AttackerPlantController
from touyama_v3.tv3_collect_attacker_analysis_plant import FixedPlantAnalysisCollector
from touyama_v3.tv3_train_attacker_analysis import collection_source


class SelectRoute:
    def __init__(self, index):
        self.index, self.calls = index, 0
    def random(self):
        return 0.
    def integers(self, size):
        self.calls += 1
        return self.index if self.calls % 2 else 0


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.game = attacker_world()
        self.encoder = AttackerEncoder(self.game)
        self.snap = replace(FrcPerceptionBuilder('A').build(self.game), sightings=())
        origin = next(a.position for a in self.snap.allies if a.has_spike)
        routes = candidate_routes(self.game, origin, 100)
        self.left = next(r for r in routes if r.site == 'L')
        self.right = next(r for r in routes if r.site == 'R')
        self.routes = [self.left, self.right]

        class Scores:
            def analyze(_, encoder, snap, observation, candidates):
                return {'candidates': [dict(route=r, mode='preserve', score=1. if r.site=='L' else -100.,
                    information_gain=0., observed_empty_fraction=0.) for r in candidates]}
        self.model = Scores()

    def update(self, planner, snap, old=None, **kwargs):
        with patch('touyama_v3.tv3_learn_attacker_analysis.candidate_routes', return_value=list(self.routes)):
            return planner.update(snap, self.encoder.observe(snap, []), self.encoder, self.model,
                old.route if old else None, old.mode if old else None, **kwargs)

    def test_all_routes_can_sample_low_score_and_bypass_retention(self):
        planner = AdaptiveAttackPlanner(site_switch_margin=200.)
        planner.enable_flanks = False
        first = self.update(planner, self.snap)
        index = sorted(r.key for r in self.routes).index(self.right.key)
        changed = replace(self.snap,tick=3,visible_cells=tuple(set(self.snap.visible_cells) ^ {self.right.cells[-1]}))
        second = self.update(planner, changed, first,
            rng=SelectRoute(index), exploration=1., exploration_scope='all_routes')
        self.assertEqual(second.route.site, 'R')
        self.assertTrue(planner.events[-1]['exploration_applied'])
        self.assertEqual(planner.events[-1]['exploration_eligible_routes'], 2)

    def test_legacy_score_window_still_excludes_low_score(self):
        planner = AdaptiveAttackPlanner()
        planner.enable_flanks = False
        selected = self.update(planner, self.snap, rng=np.random.default_rng(4), exploration=1.)
        self.assertEqual(selected.route.site, 'L')
        self.assertEqual(planner.events[-1]['exploration_eligible_routes'], 1)

    def test_round_start_exploration_is_not_repeated_on_every_review(self):
        planner=AdaptiveAttackPlanner()
        planner.enable_flanks=False
        index=sorted(r.key for r in self.routes).index(self.right.key)
        rng=SelectRoute(index)
        first=self.update(planner,self.snap,rng=rng,exploration=1.,
            exploration_scope='all_routes',exploration_round_start_only=True)
        self.assertEqual(first.route.site,'R')
        calls=rng.calls
        changed=replace(self.snap,tick=3,visible_cells=tuple(set(self.snap.visible_cells) ^ {self.right.cells[-1]}))
        self.update(planner,changed,first,rng=rng,exploration=1.,
            exploration_scope='all_routes',exploration_round_start_only=True)
        self.assertEqual(rng.calls,calls)
        self.assertFalse(planner.events[-1]['exploration_applied'])

    def test_auto_does_not_silently_replace_incompatible_existing_plant_with_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'best'/'frc_v1'
            source.mkdir(parents=True)
            (source/'attacker_plant_best.pt').write_bytes(b'incompatible')
            args=NS(resume=False,collection_executor='auto',frozen_plant_dir=root/'best')
            with patch('touyama_v3.tv3_guard_runtime.load_sources',side_effect=ValueError('schema mismatch')):
                with self.assertRaisesRegex(ValueError,'schema mismatch'):
                    collection_source('frc_v1',self.game,args,root/'data')
            self.assertFalse((root/'data').exists())

    def test_probe_allows_execution_but_is_bounded_and_interruptible(self):
        index = sorted(r.key for r in self.routes).index(self.right.key)
        for interruption in ('timeout', 'contact', 'damage'):
            planner = AdaptiveAttackPlanner()
            planner.enable_flanks = False
            first = self.update(planner,self.snap,rng=SelectRoute(index),exploration=1.,
                exploration_scope='all_routes', exploration_probe_steps=8, exploration_probe_ticks=12)
            quiet = self.update(planner,replace(self.snap,tick=3),first,
                exploration_scope='all_routes', exploration_probe_steps=8, exploration_probe_ticks=12)
            self.assertEqual(quiet.route.site,'R')
            self.assertTrue(planner.events[-1]['exploration_probe_active'])
            changed = replace(self.snap,tick=12 if interruption=='timeout' else 4)
            if interruption=='contact':
                changed=replace(changed,sightings=(Sighting(0,self.game.sites['R'][0],'normal'),))
            if interruption=='damage':
                changed=replace(changed,allies=tuple(replace(a,hp=a.hp-1) if a.slot==0 else a for a in changed.allies))
            self.update(planner,changed,quiet,exploration_scope='all_routes',
                exploration_probe_steps=8, exploration_probe_ticks=12)
            self.assertIsNone(planner.exploration_probe)

    def test_fixed_collector_matches_deployed_actions_without_exploration(self):
        model=AttackerAnalysisModel(len(self.encoder.fields),len(self.encoder.route_fields),len(self.game.names))
        model.eval()
        policy=PlantDQN()
        policy.eval()
        policy.requires_grad_(False)
        live=TouyamaV3AttackerPlantController(self.game,model,policy)
        collector=FixedPlantAnalysisCollector(self.game,model,policy,seed=42,
            exploration=0.,exploration_scope='all_routes',exploration_probe_steps=8,exploration_probe_ticks=12)
        for controller in (live,collector):
            controller.set_game(self.game)
            controller.sensor.build=lambda _game:self.snap
            controller.prepare_team_tick()
        self.assertEqual(live.actions,collector.actions)
        for name in live.plans:
            self.assertEqual(live.plans[name][0],collector.plans[name][0])
            np.testing.assert_array_equal(live.inputs[name].observation,collector.inputs[name].observation)
        self.assertEqual(len(collector.frames),1)
        collector.prepare_team_tick()
        self.assertEqual(len(collector.frames),1)
        self.assertFalse(collector.training)
        self.assertEqual(collector.epsilon,0.)
        self.assertTrue(all(not p.requires_grad for p in policy.parameters()))

    def test_resume_keeps_byte_identical_pinned_sources_after_production_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            production=root/'best'
            data=root/'data'
            directory=production/'frc_v1'
            directory.mkdir(parents=True)
            for phase in ('analysis','plant'):
                (directory/f'attacker_{phase}_best.pt').write_bytes(('original-'+phase).encode())
            sentinel=object()
            def load(_scenario,opponents,best,analysis):
                folder=Path(best)/'frc_v1'
                return {'frc_v1':{'plant':sentinel,
                    'paths':{k:str(folder/f'attacker_{k}_best.pt') for k in ('analysis','plant')},
                    'hashes':{k:hashlib.sha256((folder/f'attacker_{k}_best.pt').read_bytes()).hexdigest()
                              for k in ('analysis','plant')}}}
            args=NS(resume=False,collection_executor='auto',frozen_plant_dir=production)
            with patch('touyama_v3.tv3_guard_runtime.load_sources',side_effect=load):
                policy,metadata=collection_source('frc_v1',self.game,args,data)
                self.assertIs(policy,sentinel)
                for phase in ('analysis','plant'):
                    self.assertEqual((directory/f'attacker_{phase}_best.pt').read_bytes(),
                        (data/'collection_sources'/'frc_v1'/f'attacker_{phase}_best.pt').read_bytes())
                    (directory/f'attacker_{phase}_best.pt').write_bytes(b'new production model')
                (data/'frc_v1').mkdir()
                torch.save({'config':metadata},data/'frc_v1'/'latest.pt')
                args.resume=True
                _,resumed=collection_source('frc_v1',self.game,args,data)
                self.assertEqual(metadata,resumed)
                (data/'collection_sources'/'frc_v1'/'attacker_plant_best.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'hashes changed'):
                    collection_source('frc_v1',self.game,args,data)


if __name__=='__main__':
    unittest.main()
