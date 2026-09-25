"""Regression tests for learned Omoko defender-search coordination."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ov1_learning_defender_search as runtime
import ov1_train_defender_search as search


class OmokoDefenderSearchCoordinationTests(unittest.TestCase):
    def test_late_team_sighting_persists_after_visibility_is_lost(self):
        defender = search.UnitStub("defender", "D", (1, 1), "HUNT")
        attacker = search.UnitStub("attacker", "A", (1, 4), "HUNT")
        memories = (
            (search.TeamMemory(), lambda memory, tick: memory.update(
                [defender], [attacker], set(), round_tick=tick,
            )),
            (runtime._TeamMemory(), lambda memory, tick: memory.update(
                search.GRID, "D", [defender, attacker], round_tick=tick,
            )),
        )
        with patch.object(search, "has_los", return_value=True) as train_los, patch.object(
            runtime, "_has_los", return_value=True,
        ) as runtime_los:
            for memory, update in memories:
                train_los.return_value = True
                runtime_los.return_value = True
                memory.spike_pos = (4, 4)
                memory.spike_held = True
                update(memory, search.REINFORCE_FROM_TICK - 1)
                self.assertIsNone(memory.last_seen_enemy)
                update(memory, search.REINFORCE_FROM_TICK)
                self.assertEqual(memory.last_seen_enemy["pos"], tuple(attacker.pos))
                self.assertIsNone(memory.spike_pos)

                train_los.return_value = False
                runtime_los.return_value = False
                update(memory, search.REINFORCE_FROM_TICK + 1)
                self.assertEqual(memory.last_seen_enemy["tick_ago"], 1)
                for tick in range(2, search.SIGHTING_MEMORY_TICKS + 1):
                    update(memory, search.REINFORCE_FROM_TICK + tick)
                self.assertIsNotNone(memory.last_seen_enemy)
                update(memory, search.REINFORCE_FROM_TICK + search.SIGHTING_MEMORY_TICKS + 1)
                self.assertIsNone(memory.last_seen_enemy)

    def test_sighting_becomes_priority_destination_in_training_and_runtime(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        env.round_timer = 60
        for attacker in env.attackers:
            attacker.is_alive = False
        env.team_memory.spike_pos = None
        env.team_memory.last_seen_enemy = {
            "pos": search.SITE_POSITIONS[0], "name": "attacker", "tick_ago": 1,
        }
        env._update_priority_dist_maps()
        mode, dist_map, target = env._priority_mode_and_distmap(env.defenders[0])
        self.assertEqual((mode, target), ("sighting", "attacker"))
        self.assertIsNotNone(dist_map)

        defender = env.defenders[0]
        train_obs, _ = env._collect_observations()
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller.team_memory.last_seen_enemy = dict(env.team_memory.last_seen_enemy)
        controller.spike_dist_map = None
        controller.sighting_dist_map = runtime._bfs_distance_map(
            search.GRID, controller.team_memory.last_seen_enemy["pos"],
        )
        controller._assigned_dist_maps = {defender.name: defender.assigned_defense_dist_map}
        controller._assigned_setup_dist_maps = {defender.name: defender.assigned_setup_dist_map}
        runtime_obs, _ = controller._build_observation(
            defender,
            {"grid": search.GRID, "chars": env.defenders + env.attackers,
             "round_timer": env.round_timer},
            search.SITE_POSITIONS, False,
        )
        self.assertEqual(train_obs[defender.name][17], 1.0)
        self.assertEqual(train_obs[defender.name][31], 0.0)
        np.testing.assert_array_equal(train_obs[defender.name][17:21], runtime_obs[17:21])
        np.testing.assert_array_equal(train_obs[defender.name][29:35], runtime_obs[29:35])

    def test_training_and_runtime_observation_dimensions_match(self):
        self.assertEqual(search.OBS_DIM, runtime.OBS_DIM)
        self.assertEqual(search.ACTION_DIM, runtime.ACTION_DIM)
        self.assertEqual(search.OBS_DIM, 72)
        self.assertEqual(
            Path(search.MODEL_SAVE_PATH).resolve(),
            Path(runtime.DEFAULT_MODEL_PATH).resolve(),
        )

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

    def test_directly_visible_enemy_only_locks_movement_early(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        defender = env.defenders[0]
        attacker = env.attackers[0]
        for other in env.defenders[1:] + env.attackers[1:]:
            other.is_alive = False
        open_cell = next(
            (r, c) for r in range(1, search.HEIGHT - 1)
            for c in range(1, search.WIDTH - 1)
            if search.GRID[r, c] != 1
            and sum(search.GRID[r + dr, c + dc] != 1 for dr, dc in search.CARDINAL) >= 2
        )
        defender.pos = list(open_cell)
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

        env.round_timer = search.MAX_TICKS - search.REINFORCE_FROM_TICK + 1
        _, masks = env._collect_observations()
        mask = masks[defender.name]
        moving_actions = np.concatenate([
            mask[base * len(search.FACING_DIRS):(base + 1) * len(search.FACING_DIRS)]
            for base in range(2, 10)
        ])
        self.assertTrue(moving_actions.any())

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
