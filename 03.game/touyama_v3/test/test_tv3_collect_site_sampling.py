"""Site selection isolation, quotas, and real-site filtering without learning."""
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from touyama_v3.tv3_collect_site_sampling import pending_site, site_sampling
from touyama_v3 import tv3_collect_attacker_guard as guard


class SiteSamplingTests(unittest.TestCase):
    def test_pending_site_prioritizes_zero_and_never_skips_it(self):
        counts = Counter({('fnatic_v3', 'L'): 50})
        self.assertEqual(pending_site(counts, 'fnatic_v3', ['L', 'R'], 50), 'R')
        counts['fnatic_v3', 'R'] = 50
        self.assertIsNone(pending_site(counts, 'fnatic_v3', ['L', 'R'], 50))

    def test_planner_candidates_filter_and_restore_after_exception(self):
        from touyama_v3 import tv3_learn_attacker_analysis as module
        from touyama_v3.tv3_scenario import Scenario
        scenario = Scenario()
        origin = tuple(np.argwhere(scenario.grid == 3)[0])
        original = module.candidate_routes
        before = original(scenario, origin, 100)
        for site in ('L', 'R'):
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                with site_sampling(site):
                    candidates = module.candidate_routes(scenario, origin, 100)
                    self.assertTrue(candidates)
                    self.assertEqual([r.key for r in candidates], [r.key for r in before if r.site == site])
                    raise RuntimeError('stop')
            self.assertIs(module.candidate_routes, original)

    def test_fnatic_can_collect_right_despite_public_touyama_left_rule(self):
        from fnatic_v3.controller import FnaticV3AttackerController as Controller
        original = Controller._plant_candidates
        controller = NS(positions=NS(plant_cells=((1, 1), (1, 8))), force_a_attack=True)
        grid = np.zeros((3, 10), dtype=int)
        with site_sampling('R', attacker='fnatic_v3'):
            candidates = Controller._plant_candidates(controller, grid)
            self.assertEqual(candidates, ((1, 8),))
            self.assertEqual(Controller._select_attack_target(controller, candidates, {}, grid), (1, 8))
            self.assertEqual(controller.attack_region, 'B')
        self.assertIs(Controller._plant_candidates, original)

    def test_gc_site_functions_restore_all_runtime_aliases(self):
        import importlib
        modules = [importlib.import_module(name) for name in
                   ('gc_v1.opponent_site_gc', 'ghost_champions_v1',
                    'gc_v1.learning_attacker_macro_gc_runtime')]
        originals = [(m.forced_attack_site, m.attack_plant_cells) for m in modules]
        grid = np.zeros((3, 10), dtype=int)
        grid[1, 1] = grid[1, 8] = 2
        for site, expected in (('L', (1, 1)), ('R', (1, 8))):
            with site_sampling(site, attacker='gc_v1'):
                for module in modules:
                    self.assertEqual(module.attack_plant_cells(None, grid), [expected])
            for module, original in zip(modules, originals):
                self.assertIs(module.forced_attack_site, original[0])
                self.assertIs(module.attack_plant_cells, original[1])

    def test_guard_requires_each_site_and_records_request(self):
        for limit, complete in ((3, False), (4, True)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                requested = []
                active = []
                @contextmanager
                def sampling(site):
                    requested.append(site)
                    active.append(site)
                    try:
                        yield
                    finally:
                        active.pop()
                def block(opponent, scenario, source, policy, seed, **kwargs):
                    game = NS(planted_pos=active[-1], current_round=1, chars=[])
                    kwargs['capture'](game, NS(guard=NS(initial_charges={})), opponent)
                    return [{'planted': True}], []
                source = {'fixed_roster_training': True, 'hashes': {'plant': 'p', 'analysis': 'a'},
                          'paths': {}, 'train_presets': ['Touyama Gaming'], 'plant_set': 10, 'analysis_set': 10}
                args = NS(torch_threads=1, opponents=['fnatic_v3'], plant_dir=root/'plant', analysis_dir=root/'analysis',
                          output_dir=root/'cases', log_dir=root/'logs', seed=42, resume=False,
                          cases=2, sites=['L', 'R'], site_sampling='targeted', max_blocks=limit, max_round_steps=400)
                with patch.object(guard, 'Scenario', return_value=NS(site_of=lambda p: p)), \
                     patch.object(guard, 'guard_schema', return_value={}), \
                     patch.object(guard, 'load_sources', return_value={'fnatic_v3': source}), \
                     patch.object(guard, 'site_sampling', side_effect=sampling), \
                     patch.object(guard, 'play_block', side_effect=block), \
                     patch.object(guard, 'valid_guard_case'), \
                     patch.object(guard, 'case_metadata', side_effect=lambda game, enemy: {'site': game.planted_pos, 'opponent': enemy, 'actors': []}), \
                     patch.object(guard, 'save_case', side_effect=lambda path, *args: path.write_bytes(b'case')):
                    self.assertEqual(guard.collect(args), complete)
                self.assertEqual(requested, ['L', 'R', 'L', 'R'][:limit])
                summary = json.loads((args.output_dir/'collection_summary.json').read_text())
                self.assertEqual(summary['complete'], complete)
                self.assertEqual(summary['by_site']['fnatic_v3'], {'L': 2, 'R': 2 if complete else 1})
                rows = guard.read_rows(args.output_dir/'cases.jsonl')
                self.assertTrue(all(r['requested_site'] == r['site'] for r in rows))

    def test_retake_site_evaluation_uses_independent_seeds_and_keeps_site_results_separate(self):
        from touyama_v3 import tv3_train_defender_retake as training
        plan = [('fnatic_v3', 'Touyama Gaming', 903000042+i*100) for i in range(6)]
        callback = object()
        with patch.object(training, 'site_sampling', return_value=nullcontext()) as sampling, \
             patch.object(training, 'rollout', return_value=([{'site': 'R'}, {'site': 'L'}, {'site': None}], [])) as rollout:
            result = training.evaluate_requested_site('fnatic_v3', 'R', plan, None, None, {}, {}, round_callback=callback)
        self.assertEqual(result, [{'site': 'R'}]*training.SITE_EVALUATION_SEED_COUNT)
        for call, (_, _, seed) in zip(rollout.call_args_list, plan):
            self.assertEqual(call.args[7], seed+training.SITE_EVALUATION_SEED_OFFSET+training.SITE_EVALUATION_RIGHT_SEED_OFFSET)
            self.assertNotIn('training', call.kwargs)
            self.assertIs(call.kwargs['round_callback'], callback)
        self.assertTrue(all(call.args == ('R',) for call in sampling.call_args_list))


if __name__ == '__main__':
    unittest.main()
