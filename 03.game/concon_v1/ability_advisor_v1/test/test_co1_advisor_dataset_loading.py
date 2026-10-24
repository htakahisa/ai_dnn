"""Interrupted collection must not prevent loading other completed datasets."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from concon_v1.ability_advisor_v1 import co1_train_attacker_ability_advisor as training


class DatasetLoadingTests(unittest.TestCase):
    def write_complete(self,directory):
        metadata = dict(event='metadata',version=training.VERSION,
                        label_version=training.LABEL_VERSION,wall_los_version=training.WALL_LOS_VERSION,
                        feature_names=training.feature_names(),scenarios={})
        sample = dict(event='sample',match_id='match',round_number=1,group='group',
                      label=1,features=[0.]*len(training.feature_names()))
        path = Path(directory)/'completed.jsonl'
        path.write_text('\n'.join(json.dumps(row) for row in
                        (metadata,sample,dict(event='complete',samples=1))),encoding='utf-8')

    def test_incomplete_and_truncated_files_are_skipped_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_complete(directory)
            for name,contents in [('empty',''),('interrupted','{"event":"sample"}\n'),
                                  ('truncated','{"event":"sample"}\n{"event":')]:
                Path(directory,f'{name}.jsonl').write_text(contents,encoding='utf-8')
            before = {p:p.read_bytes() for p in Path(directory).glob('*.jsonl')}
            log = io.StringIO()
            with patch.object(training,'route_metadata',return_value={}),contextlib.redirect_stdout(log):
                rows,_ = training.load_samples(directory)
            self.assertEqual(len(rows),1)
            self.assertEqual(log.getvalue().count('[SKIP]'),3)
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_only_incomplete_files_report_no_completed_data(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'interrupted.jsonl').write_text('',encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'no completed datasets'):
                training.load_samples(directory)
            with self.assertRaisesRegex(ValueError,'incomplete dataset'):
                training.load_samples(directory,skip_incomplete=False)

    def test_completed_but_incompatible_data_still_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_complete(directory)
            path = Path(directory,'completed.jsonl')
            lines = path.read_text(encoding='utf-8').splitlines()
            metadata = json.loads(lines[0])
            metadata['feature_names'] = ['outdated']
            lines[0] = json.dumps(metadata)
            path.write_text('\n'.join(lines),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'contract mismatch'):
                training.load_samples(directory)
