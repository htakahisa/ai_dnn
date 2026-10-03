"""Check BFS diagnostics without training or loading an attacker checkpoint."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from concon_v1.check_co1_bfs_battle import bfs_action, run_battle, write_replay
from concon_v1.co1_attacker_common import ACTION_PLANT, ACTION_WAIT
from concon_v1.co1_attacker_scenarios import get_scenario


class BfsBattleTests(unittest.TestCase):
    def test_bfs_respects_blocked_shortest_step_and_planting(self):
        route = SimpleNamespace(distance_map=np.array([[9, 8, 9], [2, 3, 4], [9, 5, 9]]))
        mask = np.array([False, True, False, True, True, False])
        self.assertEqual(bfs_action((1, 1), route, mask), 3)
        mask[3] = False
        self.assertEqual(bfs_action((1, 1), route, mask), 1)
        mask[1] = False
        self.assertEqual(bfs_action((1, 1), route, mask), ACTION_WAIT)
        mask[ACTION_PLANT] = True
        self.assertEqual(bfs_action((1, 1), route, mask), ACTION_PLANT)

    def test_real_battle_records_combat_and_deaths_without_model_inference(self):
        with patch('concon_v1.co1_battle_training._choose_action', side_effect=AssertionError('model used')), \
                contextlib.redirect_stdout(io.StringIO()):
            result = run_battle(get_scenario('A1'), 'gc_v1', 0, 100, 5)
        self.assertEqual(len(result['frames']), result['ticks'] + 1)
        self.assertTrue(any(f['shots'] for f in result['frames']))
        self.assertTrue(result['deaths'])
        for death in result['deaths']:
            before = result['frames'][death['tick'] - 1]
            self.assertTrue(next(a for a in before['actors'] if a['name'] == death['name'])['alive'])
            self.assertFalse(death['alive'])
        json.dumps(result)

    def test_replay_escapes_script_end_in_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.html'
            write_replay(path, {'map': '</script><script>alert(1)</script>'})
            html = path.read_text(encoding='utf-8')
        self.assertNotIn('__REPORT__', html)
        self.assertIn(r'\u003c/script>', html)
        self.assertNotIn('<script>alert(1)', html)


if __name__ == '__main__':
    unittest.main()
