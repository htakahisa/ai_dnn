import unittest
from types import SimpleNamespace
import numpy as np
from gc_v1.learning_defender_setup_gc_runtime import (
    LearningDefenderSetupGCRuntime, GC_ROSTER_ORDER, DEFENDER_SPAWNS,
    _build_obs, OBS_DIM, OPPONENT_DIM,
)


class SetupCompatibilityTests(unittest.TestCase):
    def test_old_model_moves_against_new_preset(self):
        runtime = LearningDefenderSetupGCRuntime(device='cpu', verbose=False)
        runtime.set_context(opponent_name='Carnal Lust Syndicate')
        chars = [SimpleNamespace(name=name, team='D', is_alive=True, pos=list(spawn))
                 for name, spawn in zip(GC_ROSTER_ORDER, DEFENDER_SPAWNS)]
        initial = [tuple(c.pos) for c in chars]
        for _ in range(10):
            for char in chars:
                char.pos = runtime.decide_setup_move(char, chars)
        self.assertTrue(runtime.round_initialized)
        self.assertNotEqual(initial, [tuple(c.pos) for c in chars])
        self.assertEqual(runtime.model.net[0].in_features, runtime.obs_dim)

    def test_unknown_opponent_has_checkpoint_sized_zero_vector(self):
        char = SimpleNamespace(pos=[1, 1])
        obs = _build_obs(char, 0, [], 'new team', 0,
                         {'Touyama Gaming': 0}, 1)
        self.assertEqual(len(obs), OBS_DIM - OPPONENT_DIM + 1)
        offset = OBS_DIM - OPPONENT_DIM - 3
        self.assertEqual(obs[offset], 0.0)
        self.assertTrue(np.isfinite(obs).all())


if __name__ == '__main__':
    unittest.main()
