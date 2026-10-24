"""Public match clock coverage and legacy contracts; no model optimization."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
from frc_v1.perception import FrcPerceptionBuilder
from touyama_v3.test.tv3_fixtures import attacker_world
from touyama_v3.tv3_learn_attacker_analysis import AttackerEncoder


class AttackerRoundClockTests(unittest.TestCase):
    def test_live_public_snapshot_preserves_second_half_absolute_round(self):
        game = attacker_world()
        game.current_round = 13
        snapshot = FrcPerceptionBuilder('A').build(game)
        self.assertEqual(snapshot.round_number, 13)
        old = AttackerEncoder(game, 3)
        self.assertEqual(old.observe(snapshot, [])[:12].sum(), 0.)
        new = AttackerEncoder(game, 4)
        for round_number in (1, 12, 13, 24, 25, 48, 49):
            observation = new.observe(replace(snapshot, round_number=round_number), [])
            self.assertEqual(observation[:12].sum(), 1.)
            self.assertEqual(observation[(round_number-1) % 12], 1.)
            self.assertAlmostEqual(observation[new.fields.index('absolute_match_round')], round_number/48.)
        self.assertNotIn('round_clock', old.schema())
        self.assertNotIn('absolute_match_round', old.fields)
        # Guard/defender shared history keeps its original input contract.
        self.assertEqual(new.history.encode(snapshot, [])[:12].sum(), 0.)

    def test_half_evaluation_covers_both_without_optimizer_or_model_mutation(self):
        from touyama_v3 import tv3_train_attacker_analysis as analysis
        from touyama_v3 import tv3_train_attacker_plant as plant
        game = attacker_world()
        calls = []
        def empty(*args, **kwargs):
            calls.append(kwargs['start_round'])
            return [], []
        config = dict(eval_presets=['Touyama Gaming'], evaluation_start_rounds=[1, 13])
        with patch.object(analysis, 'play_block', side_effect=empty):
            result = analysis.evaluate('frc_v1', game, None, config, [7])
        self.assertEqual(calls, [1, 1, 13, 13])
        self.assertEqual([p['start_round'] for p in result['roster_plan']], [1, 13])
        calls.clear()
        with patch.object(plant, 'rollout', side_effect=empty):
            result = plant.evaluate('frc_v1', game, None, None, config, [7])
        self.assertEqual(calls, [1, 13])


if __name__ == '__main__':
    unittest.main()
