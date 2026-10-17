"""Obsolete adviser datasets are deleted before collection and training."""

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
from omoko_v1.ability_advisor_v1 import ov1_train_attacker_ability_advisor as training


class DatasetCleanupTests(unittest.TestCase):
    def test_legacy_datasets_survive_cleanup_and_training_load(self):
        from omoko_v1.ability_advisor_v1.ov1_advisor_attacker_context import normalize_policy_digest
        legacy = '08f4bf3e7fc3995a0ce15e37615e6a2c7c9d17661f4170110ebbbb6683ddbfa1'
        current = {'carry_L': dict(signature='layout', carry_sha256=normalize_policy_digest(legacy))}
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_dataset(directory, scenarios={
                'carry_L': dict(signature='layout', carry_sha256=legacy)})
            before = path.read_bytes()
            with patch.object(collection, 'route_metadata', return_value=current), \
                    patch.object(training, 'route_metadata', return_value=current):
                rows, provenance = training.load_samples(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(provenance, current)
            self.assertEqual(path.read_bytes(), before)

    current = {'carry_L': {'signature': 'current', 'carry_sha256': 'weights'}}

    def write_dataset(self, directory, name='data.jsonl', **changes):
        header = dict(event='metadata', version=collection.VERSION,
                      label_version=collection.LABEL_VERSION,
                      wall_los_version=collection.WALL_LOS_VERSION,
                      feature_names=collection.feature_names(), scenarios=self.current)
        header.update(changes)
        sample = dict(event='sample', match_id='match', round_number=1,
                      group='group', label=1, features=[0.] * len(collection.feature_names()))
        path = Path(directory) / name
        path.write_text('\n'.join(json.dumps(row) for row in
                                  (header, sample, dict(event='complete', samples=1))),
                        encoding='utf-8')
        return path

    def test_changed_contract_and_policy_are_deleted_compatible_files_remain(self):
        cases = [dict(version='old'), dict(label_version='old'),
                 dict(wall_los_version='old'), dict(feature_names=['old']),
                 dict(scenarios={'carry_L': {'signature': 'old', 'carry_sha256': 'weights'}}),
                 dict(scenarios={'carry_L': {'signature': 'current', 'carry_sha256': 'old'}})]
        with tempfile.TemporaryDirectory() as directory:
            keep = self.write_dataset(directory, 'keep.jsonl')
            before = keep.read_bytes()
            obsolete = [self.write_dataset(directory, f'old{i}.jsonl', **change)
                        for i, change in enumerate(cases)]
            other = Path(directory) / 'notes.txt'
            other.write_text('keep', encoding='utf-8')
            with patch.object(collection, 'route_metadata', return_value=self.current), \
                    contextlib.redirect_stdout(io.StringIO()):
                deleted = collection.delete_incompatible_datasets(directory)
            self.assertEqual(set(deleted), set(obsolete))
            self.assertEqual(keep.read_bytes(), before)
            self.assertTrue(other.exists())
            self.assertTrue(all(not path.exists() for path in obsolete))

    def test_training_continues_with_current_data_after_deleting_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_dataset(directory, 'current.jsonl')
            old = self.write_dataset(directory, 'old.jsonl', version='old')
            with patch.object(collection, 'route_metadata', return_value=self.current), \
                    patch.object(training, 'route_metadata', return_value=self.current), \
                    contextlib.redirect_stdout(io.StringIO()):
                rows, provenance = training.load_samples(directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(provenance, self.current)
            self.assertFalse(old.exists())

    def test_only_old_data_is_deleted_and_requests_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            old = self.write_dataset(directory, version='old')
            with patch.object(collection, 'route_metadata', return_value=self.current), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, 'run the collector first'):
                training.load_samples(directory)
            self.assertFalse(old.exists())

    def test_collection_deletes_before_launching_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            old = self.write_dataset(directory, version='old')

            def stop_before_matches(*args):
                self.assertFalse(old.exists())
                raise RuntimeError('stop before actual collection')

            with patch.object(collection, 'route_metadata', return_value=self.current), \
                    patch.object(collection, 'run_opponent_tasks', side_effect=stop_before_matches), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(RuntimeError, 'stop before actual collection'):
                collection.collect(matches=1, rounds=1, opponents=('frc_v1',),
                                   output_dir=directory, workers=1)


if __name__ == '__main__':
    unittest.main()
