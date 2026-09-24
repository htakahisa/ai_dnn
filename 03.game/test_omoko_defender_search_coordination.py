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

    def test_directly_visible_enemy_locks_movement_actions(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        defender = env.defenders[0]
        attacker = env.attackers[0]
        origin = tuple(defender.pos)
        candidate = next(
            (pos for pos in ((origin[0] - 1, origin[1]), (origin[0] + 1, origin[1]),
                             (origin[0], origin[1] - 1), (origin[0], origin[1] + 1))
             if 0 <= pos[0] < search.HEIGHT and 0 <= pos[1] < search.WIDTH
             and search.GRID[pos] != 1 and search.has_los(origin, pos)),
            None,
        )
        self.assertIsNotNone(candidate)
        attacker.pos = list(candidate)
        _, masks = env._collect_observations()
        mask = masks[defender.name]
        moving_actions = np.concatenate([
            mask[base * len(search.FACING_DIRS):(base + 1) * len(search.FACING_DIRS)]
            for base in range(2, 10)
        ])
        self.assertFalse(moving_actions.any())

    def test_s_markers_fire_at_the_twentieth_post_setup_tick(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        env.round_timer = search.MAX_TICKS - search.SCHEDULED_SMOKE_DELAY_TICKS + 1
        targets = env._scheduled_smoke_targets()

        smoke_names = {d.name for d in env.defenders if d.role == "SMOKE"}
        # A map may contain fewer S markers than smoke-capable defenders;
        # only one nearest defender is assigned per marker.
        self.assertEqual(len(targets), min(len(smoke_names), len(search.SMOKE_SITE_POSITIONS)))
        self.assertTrue(set(targets).issubset(smoke_names))
        self.assertEqual(set(targets.values()), set(search.SMOKE_SITE_POSITIONS))

        observations, masks = env._collect_observations()
        actions = {
            name: int(np.flatnonzero(masks[name])[0])
            for name in observations
        }
        env.scheduled_smoke_fired = False
        env.step(actions)
        self.assertEqual(
            sum(d.charges for d in env.defenders if d.role == "SMOKE"),
            len(smoke_names) - len(targets),
        )
        self.assertEqual(len(env.smokes), len(search.SMOKE_SITE_POSITIONS))


if __name__ == "__main__":
    unittest.main()
