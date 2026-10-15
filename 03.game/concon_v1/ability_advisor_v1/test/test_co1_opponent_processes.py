"""Spawn isolation, deterministic seeds, and dataset publication on success only."""

from pathlib import Path
from collections import Counter
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from concon_v1.ability_advisor_v1.co1_opponent_processes import run_opponent_tasks
from concon_v1.ability_advisor_v1 import co1_collect_attacker_ability_advisor as collector
from concon_v1.ability_advisor_v1 import co1_evaluate_attacker_ability_advisor as evaluator


def process_probe(value):
    import torch
    if value < 0:
        raise ValueError('worker failure')
    return value*value,os.getpid(),torch.get_num_threads()


class OpponentProcessTests(unittest.TestCase):
    def test_spawn_preserves_order_and_limits_torch_threads(self):
        results = run_opponent_tasks(process_probe,[3,1,2],2)
        self.assertEqual([r[0] for r in results],[9,1,4])
        self.assertTrue(all(r[1] != os.getpid() and r[2] == 1 for r in results))

    def test_spawn_propagates_failure(self):
        with self.assertRaisesRegex(ValueError,'worker failure'):
            run_opponent_tasks(process_probe,[-1,2],2)

    def test_collection_merges_shards_with_original_seed_schedule(self):
        seen = []
        metadata = {'A1':{'signature':'test','carry_sha256':'hash'}}
        def fake_match(opponent,seed,rounds,advisor,map_names):
            seen.append((opponent,seed,advisor.match_id))
            advisor.route_hashes = {'A1':'hash'}
            advisor.groups = Counter({'group':1})
            advisor.stream.write(json.dumps({'event':'sample','match_id':advisor.match_id})+'\n')
            return {'attacker_wins':1,'rounds':1}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collector,'route_metadata',return_value=metadata), \
                 patch.object(collector,'play_match',side_effect=fake_match):
                path = collector.collect(matches=2,rounds=1,opponents=collector.OPPONENTS[:2],
                    map_names=('A1',),seed=10,output_dir=directory,workers=1)
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['event'],'metadata')
            self.assertEqual(rows[-1],{'event':'complete','samples':4})
            self.assertEqual(len(rows),6)
            self.assertEqual(sorted(seed for _,seed,_ in seen),[10,11,12,13])
            self.assertTrue(all(f'seed={seed}:' in match for _,seed,match in seen))
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_failed_collection_does_not_publish_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collector,'route_metadata',return_value={}), \
                 patch.object(collector,'play_match',side_effect=RuntimeError('simulation failed')):
                with self.assertRaisesRegex(RuntimeError,'simulation failed'):
                    collector.collect(matches=1,rounds=1,opponents=collector.OPPONENTS[:1],
                        map_names=('A1',),output_dir=directory,workers=1)
            self.assertEqual(list(Path(directory).glob('*.jsonl')),[])

    def test_evaluation_pairs_seeds_and_writes_combined_report(self):
        seen = []
        def fake_match(opponent,seed,rounds,advisor,maps):
            seen.append((opponent,seed,advisor.model is not None))
            return dict(rounds=1,attacker_wins=1,charges_at_plant=[],
                        decisions={'use':1,'save':0})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'report.json'
            with patch.object(evaluator,'sha256_file',return_value='hash'), \
                 patch.object(evaluator,'FixedAbilityAdvisor',
                              side_effect=lambda path:SimpleNamespace(model=path)), \
                 patch.object(evaluator,'play_match',side_effect=fake_match):
                report = evaluator.evaluate(model_path=Path(directory)/'model.pt',matches=2,
                    rounds=1,opponents=collector.OPPONENTS[:2],maps=('A1',),seed=20,
                    output=output,workers=1)
            self.assertEqual([seed for _,seed,_ in seen],[20,20,21,21,22,22,23,23])
            self.assertEqual([advised for _,_,advised in seen],[False,True]*4)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8')),report)
            self.assertEqual(len(report['opponents']),2)


if __name__ == '__main__':
    unittest.main()
