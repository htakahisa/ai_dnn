"""All collection outputs, including temporary shards, obey the byte limit."""

from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omoko_v1.ability_advisor_v1 import ov1_collect_attacker_ability_advisor as collection


class DatasetSizeTests(unittest.TestCase):
    def test_utf8_bytes_and_metadata_footer_fit_limit(self):
        records = [json.dumps({'event': 'sample', 'name': '定点' * 8, 'i': i},
                              ensure_ascii=False) + '\n' for i in range(30)]
        with tempfile.TemporaryDirectory() as directory:
            with collection.CappedJsonlWriter(directory, 'test', '.jsonl',
                                              {'event': 'metadata'}, max_bytes=400) as writer:
                for record in records:
                    writer.write(record)
            self.assertGreater(len(writer.paths), 1)
            actual = []
            for path in writer.paths:
                self.assertLessEqual(path.stat().st_size, 400)
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(rows[0]['event'], 'metadata')
                self.assertEqual(rows[-1]['event'], 'complete')
                self.assertEqual(rows[-1]['samples'], len(rows) - 2)
                actual.extend(rows[1:-1])
            self.assertEqual(actual, [json.loads(row) for row in records])

    def test_worker_files_rotate_and_oversized_record_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            with collection.CappedJsonlWriter(directory, 'worker', '.part', max_bytes=100) as writer:
                for _ in range(10):
                    writer.write('x' * 49 + '\n')
                with self.assertRaisesRegex(ValueError, 'one dataset record'):
                    writer.write('x' * 100 + '\n')
            self.assertEqual(len(writer.paths), 5)
            self.assertTrue(all(p.stat().st_size <= 100 for p in writer.paths))

    def test_collection_publishes_complete_bounded_files_and_cleans_worker_files(self):
        records = [json.dumps({'event': 'sample', 'i': i, 'features': [0.] * 40}) + '\n'
                   for i in range(20)]

        def fake_workers(function, tasks, workers):
            results = []
            for task in tasks:
                path = Path(task[-1])
                with collection.CappedJsonlWriter(path.parent, path.stem, '.part') as writer:
                    for record in records:
                        writer.write(record)
                results.append(([str(p) for p in writer.paths], Counter(group=len(records))))
            return results

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collection, 'MAX_DATASET_FILE_BYTES', 3000), \
                    patch.object(collection, 'feature_names', return_value=[]), \
                    patch.object(collection, 'route_metadata', return_value={}), \
                    patch.object(collection, 'run_opponent_tasks', side_effect=fake_workers), \
                    contextlib.redirect_stdout(io.StringIO()):
                paths = collection.collect(matches=1, rounds=1, opponents=('frc_v1',),
                                           output_dir=directory, workers=1)
            self.assertGreater(len(paths), 1)
            actual = []
            for path in paths:
                self.assertLessEqual(path.stat().st_size, 3000)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[0]['event'], 'metadata')
                self.assertEqual(rows[-1]['event'], 'complete')
                actual.extend(rows[1:-1])
            self.assertEqual(actual, [json.loads(row) for row in records])
            self.assertEqual(set(Path(directory).iterdir()), set(paths))


if __name__ == '__main__':
    unittest.main()
