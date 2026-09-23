"""Regression tests for learned Omoko defender-search coordination."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "omoko_v1"))

import ov1_learning_defender_search as runtime
import ov1_train_defender_search as search


class OmokoDefenderSearchCoordinationTests(unittest.TestCase):
    def test_training_and_runtime_observation_dimensions_match(self):
        self.assertEqual(search.OBS_DIM, runtime.OBS_DIM)
        self.assertEqual(search.ACTION_DIM, runtime.ACTION_DIM)
        self.assertEqual(search.OBS_DIM, 72)

    def test_joining_existing_line_of_sight_gets_larger_reward(self):
        before = {"ally": True, "mover": False}
        after = {"ally": True, "mover": True}

        joined = search.crossfire_coordination_reward("mover", before, after)
        maintained = search.crossfire_coordination_reward("ally", before, after)

        self.assertAlmostEqual(
            joined,
            search.CROSSFIRE_JOIN_REWARD + search.CROSSFIRE_MAINTAIN_REWARD,
        )
        self.assertAlmostEqual(maintained, search.CROSSFIRE_MAINTAIN_REWARD)
        self.assertGreater(joined, maintained)

    def test_known_nearby_spike_does_not_lock_all_movement_actions(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        defender = env.defenders[0]
        env.team_memory.spike_pos = tuple(defender.pos)
        env.team_memory.spike_held = True
        env._update_priority_dist_maps()

        _, masks = env._collect_observations()
        mask = masks[defender.name]
        moving_actions = np.concatenate([
            mask[base * len(search.FACING_DIRS):(base + 1) * len(search.FACING_DIRS)]
            for base in range(2, 10)
        ])

        self.assertTrue(moving_actions.any())


if __name__ == "__main__":
    unittest.main()
