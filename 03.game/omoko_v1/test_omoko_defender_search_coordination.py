"""Regression tests for learned Omoko defender-search coordination."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ov1_learning_defender_search as runtime
import ov1_train_defender_search as search
from ov1_defender_search_support import (
    clear_los, find_support_ability_plan, line_cells, projectile_impact,
)


class OmokoDefenderSearchCoordinationTests(unittest.TestCase):
    def test_ability_support_uses_current_ally_sighting_and_valid_impact(self):
        grid = np.zeros((11, 13), dtype=np.int32)
        grid[2, 6] = 1
        shooter = search.UnitStub("shooter", "D", (5, 4), "HUNT")
        supporter = search.UnitStub("supporter", "D", (1, 6), "FLASH")
        enemy = search.UnitStub("enemy", "A", (5, 7), "HUNT")
        self.assertFalse(clear_los(grid, supporter.pos, enemy.pos))
        self.assertTrue(clear_los(grid, shooter.pos, enemy.pos))

        flash = find_support_ability_plan(
            grid, supporter, [supporter, shooter], [enemy], "FLASH",
        )
        self.assertIsNotNone(flash)
        self.assertEqual(flash.impact, projectile_impact(
            grid, supporter.pos, flash.aim, "FLASH",
        ))
        self.assertTrue(clear_los(grid, flash.impact, enemy.pos))
        enemy.blind_remaining = 3
        self.assertIsNone(find_support_ability_plan(
            grid, supporter, [supporter, shooter], [enemy], "FLASH",
        ))
        enemy.blind_remaining = 0
        shooter.is_alive = False
        self.assertIsNone(find_support_ability_plan(
            grid, supporter, [supporter, shooter], [enemy], "FLASH",
        ))
        shooter.is_alive = True

        smoke = find_support_ability_plan(
            grid, supporter, [supporter, shooter], [enemy], "SMOKE",
        )
        self.assertIsNotNone(smoke)
        smoke_cells = {(r, c)
                       for r in range(smoke.aim[0] - 1, smoke.aim[0] + 2)
                       for c in range(smoke.aim[1] - 1, smoke.aim[1] + 2)}
        self.assertFalse(smoke_cells.intersection(
            line_cells(shooter.pos, enemy.pos),
        ))

        enemy.pos = [6, 5]
        shooter.pos = [5, 3]
        recon = find_support_ability_plan(
            grid, supporter, [supporter, shooter], [enemy], "RECON",
        )
        self.assertIsNotNone(recon)
        self.assertLessEqual(max(abs(recon.impact[0] - enemy.pos[0]),
                                 abs(recon.impact[1] - enemy.pos[1])), 4)

    def test_runtime_casts_support_ability_without_own_enemy_los(self):
        grid = np.zeros((11, 13), dtype=np.int32)
        grid[2, 6] = 1
        shooter = search.UnitStub("shooter", "D", (5, 4), "HUNT")
        supporter = search.UnitStub("supporter", "D", (1, 6), "FLASH")
        supporter.flash_charges = 1
        enemy = search.UnitStub("enemy", "A", (5, 7), "HUNT")
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller._site_positions_cache = [(5, 7), (8, 3)]
        controller._assigned_positions = {}
        controller._reset_right_site_rotation()
        controller.verbose = False
        controller.model = lambda _obs: self.fail("Support should bypass movement policy")
        state = {"grid": grid, "chars": [supporter, shooter, enemy],
                 "battle_tick": 5, "smoke_cells": set()}
        with patch.object(controller, "_ensure_defense_assignment"), patch.object(
            controller, "_maybe_advance_tick"
        ), patch.object(controller, "_update_priority_dist_maps"), patch.object(
            controller, "_build_observation",
            return_value=(np.zeros(runtime.OBS_DIM, dtype=np.float32), []),
        ):
            next_pos, payload = controller.decide_move(supporter, state)
        self.assertEqual(next_pos, supporter.pos)
        self.assertEqual(payload["ability"], "FLASH")
        self.assertIsNotNone(projectile_impact(
            grid, supporter.pos, payload["target"], "FLASH",
        ))

    def test_training_recon_support_is_available_and_reveals_ally_target(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        for defender in env.defenders:
            defender.pos = list(search.DEFENSE_ASSIGNMENT[defender.name])
        for attacker in env.attackers:
            attacker.is_alive = False
        enemy = env.attackers[0]
        enemy.is_alive = True
        enemy.pos = [4, 38]
        enemy.has_spike = False
        recon = next(d for d in env.defenders if d.role == "RECON")
        self.assertFalse(search.has_los(recon.pos, enemy.pos))
        self.assertTrue(any(
            search.has_los(d.pos, enemy.pos) for d in env.defenders if d is not recon
        ))

        _, masks = env._collect_observations()
        allowed_support = np.flatnonzero(masks[recon.name][8:16])
        self.assertGreater(len(allowed_support), 0)
        action = int(8 + allowed_support[0])
        shooter = next(d for d in env.defenders
                       if d.role == "FLASH" and search.has_los(d.pos, enemy.pos))
        self.assertFalse(masks[shooter.name][8:16].any())
        actions = {
            defender.name: search.encode_action((0, 0), False, defender.facing)
            for defender in env.defenders
        }
        actions[recon.name] = action
        with patch.object(env, "_attacker_decide_move", return_value=(0, 0)):
            env.step(actions)
        self.assertEqual(recon.charges, 0)
        self.assertGreater(enemy.reveal_remaining, 0)

    def test_right_site_rotation_visits_t_before_watching_left_site(self):
        grid = search.GRID
        right_site = max(runtime._extract_site_positions(grid), key=lambda p: p[1])
        left_site = min(runtime._extract_site_positions(grid), key=lambda p: p[1])
        right_assignments = sorted(
            (name, pos) for name, pos in runtime._FIXED_DEFENSE_ASSIGNMENT.items()
            if pos[1] >= right_site[1] - 10
        )
        self.assertGreaterEqual(len(right_assignments), 3)
        defenders = [search.UnitStub(name, "D", pos, "HUNT")
                     for name, pos in right_assignments]
        enemy = search.UnitStub("enemy", "A", right_site, "HUNT")
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller._site_positions_cache = [right_site, left_site]
        controller._assigned_positions = dict(right_assignments)
        controller._reset_right_site_rotation()
        chars = defenders + [enemy]

        with patch.object(runtime, "_has_los", return_value=True):
            controller._update_right_site_rotation(defenders[0], grid, chars, 10)
        enemy.is_alive = False
        defenders[0].round_kills = 2
        controller._update_right_site_rotation(defenders[0], grid, chars, 14)
        self.assertFalse(controller._right_rotation_stage)
        controller._update_right_site_rotation(defenders[0], grid, chars, 15)
        self.assertEqual(len(controller._right_rotation_stage), 2)

        mover = next(d for d in defenders
                     if d.name in controller._right_rotation_stage)
        controller.verbose = False
        controller.model = lambda _obs: self.fail("Rotation should bypass movement policy")
        state = {"grid": grid, "chars": chars, "battle_tick": 15}
        with patch.object(controller, "_ensure_defense_assignment"), patch.object(
            controller, "_maybe_advance_tick"
        ), patch.object(controller, "_update_priority_dist_maps"), patch.object(
            controller, "_build_observation",
            return_value=(np.zeros(runtime.OBS_DIM, dtype=np.float32), []),
        ):
            next_pos, _ = controller.decide_move(mover, state)
        self.assertNotEqual(tuple(next_pos), tuple(mover.pos))

        for _ in range(100):
            if tuple(mover.pos) == runtime.ROTATION_TRANSIT_POSITION:
                break
            next_pos, _ = controller._right_site_rotation_move(mover, grid, chars)
            self.assertNotEqual(tuple(next_pos), tuple(mover.pos))
            mover.pos = next_pos
        else:
            self.fail("Rotating defender did not visit T")
        next_pos, _ = controller._right_site_rotation_move(mover, grid, chars)
        self.assertEqual(controller._right_rotation_stage[mover.name], "watch")
        self.assertNotEqual(tuple(next_pos), runtime.ROTATION_TRANSIT_POSITION)

    def test_seen_carrier_cancels_speculative_right_site_rotation(self):
        grid = search.GRID
        right_site = max(runtime._extract_site_positions(grid), key=lambda p: p[1])
        left_site = min(runtime._extract_site_positions(grid), key=lambda p: p[1])
        defender = search.UnitStub("defender", "D", (10, 38), "HUNT")
        enemy = search.UnitStub("carrier", "A", right_site, "HUNT", has_spike=True)
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller._site_positions_cache = [right_site, left_site]
        controller._assigned_positions = {defender.name: tuple(defender.pos)}
        controller._reset_right_site_rotation()
        with patch.object(runtime, "_has_los", return_value=True):
            controller._update_right_site_rotation(defender, grid, [defender, enemy], 10)
        enemy.is_alive = False
        defender.round_kills = 2
        controller._update_right_site_rotation(defender, grid, [defender, enemy], 20)
        self.assertFalse(controller._right_rotation_stage)

    def test_runtime_guards_dropped_spike_from_a_clear_angle(self):
        grid = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int32)
        spike_pos = (1, 5)
        defender = search.UnitStub(search.ROSTER_ORDER[0], "D", (1, 1), "HUNT")
        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller.team_memory.spike_pos = spike_pos
        controller.team_memory.spike_held = False
        controller._site_positions_cache = [spike_pos]
        controller.verbose = False
        state = {"grid": grid, "chars": [defender], "spike_pos": spike_pos,
                 "battle_tick": 40}

        class UnexpectedPolicy:
            def __call__(self, obs):
                raise AssertionError("Known dropped spike should bypass movement policy")

        controller.model = UnexpectedPolicy()
        with patch.object(controller, "_ensure_defense_assignment"), patch.object(
            controller, "_maybe_advance_tick"
        ), patch.object(controller, "_update_priority_dist_maps"), patch.object(
            controller, "_build_observation",
            return_value=(np.zeros(runtime.OBS_DIM, dtype=np.float32), []),
        ):
            for _ in range(10):
                next_pos, payload = controller.decide_move(defender, state)
                self.assertNotEqual(tuple(next_pos), spike_pos)
                if tuple(next_pos) == tuple(defender.pos):
                    self.assertTrue(runtime._has_los(grid, defender.pos, spike_pos))
                    self.assertEqual(payload["facing"],
                                     runtime._expected_facing(defender.pos, spike_pos))
                    break
                defender.pos = next_pos
            else:
                self.fail("Defender did not reach an angle on the dropped spike")

    def test_runtime_clears_ground_spike_memory_when_visible_cell_is_empty(self):
        grid = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int32)
        defender = search.UnitStub(search.ROSTER_ORDER[0], "D", (3, 3), "HUNT")
        memory = runtime._TeamMemory()
        memory.update(grid, "D", [defender], spike_ground_pos=(1, 5), round_tick=40)
        self.assertEqual(memory.spike_pos, (1, 5))
        self.assertFalse(memory.spike_held)
        memory.update(grid, "D", [defender], spike_ground_pos=None, round_tick=41)
        self.assertIsNone(memory.spike_pos)

    def test_ground_spike_watch_avoids_teammate_blocking_the_shot(self):
        grid = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ], dtype=np.int32)
        self.assertEqual(runtime._ground_spike_watch_move(
            grid, (2, 1), (2, 5), set()), (0, 0))
        self.assertNotEqual(runtime._ground_spike_watch_move(
            grid, (2, 1), (2, 5), {(2, 3)}), (0, 0))

    def test_runtime_stops_for_directly_visible_enemy_at_any_tick(self):
        class MovingPolicy:
            def __call__(self, obs):
                values = torch.zeros((obs.shape[0], runtime.ACTION_DIM),
                                     device=obs.device)
                values[:, 2 * len(runtime.FACING_DIRS):
                       10 * len(runtime.FACING_DIRS)] = 100.0
                values[:, len(runtime.FACING_DIRS):
                       2 * len(runtime.FACING_DIRS)] = 200.0
                return values

        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller.model = MovingPolicy()
        controller.verbose = False
        controller._site_positions_cache = search.SITE_POSITIONS
        controller.spike_dist_map = None
        controller._assigned_positions = {}
        controller._assigned_dist_maps = {}
        defender = search.UnitStub(search.ROSTER_ORDER[0], "D", (1, 18), "FLASH")
        defender.flash_charges = 1
        attacker = search.UnitStub("attacker", "A", (1, 19), "HUNT")
        obs = np.zeros(runtime.OBS_DIM, dtype=np.float32)
        tactical_offset = (runtime.AGENT_ID_OFFSET + runtime.AGENT_ID_DIM
                           + runtime.ULTIMATE_CONTEXT_DIM + runtime.ORB_CONTEXT_DIM)
        obs[tactical_offset] = 1.0
        state = {"grid": search.GRID, "chars": [defender, attacker],
                 "available_orbs": ()}

        with patch.object(controller, "_ensure_defense_assignment"), patch.object(
            controller, "_maybe_advance_tick"
        ), patch.object(controller, "_update_priority_dist_maps"), patch.object(
            controller, "_build_observation", return_value=(obs, [attacker])
        ) as build_observation:
            for tick in (5, 45):
                state["battle_tick"] = tick
                next_pos, payload = controller.decide_move(defender, state)
                self.assertEqual(next_pos, defender.pos)
                self.assertNotIn("ability", payload)
            build_observation.return_value = (obs, [])
            next_pos, _ = controller.decide_move(defender, state)
            self.assertNotEqual(next_pos, defender.pos)

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
        self.assertEqual(search.OBS_DIM, 72 + search.SITE_CONTEXT_DIM)
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

    def test_directly_visible_enemy_allows_retreat_even_early(self):
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
        self.assertTrue(moving_actions.any())

        env.round_timer = search.MAX_TICKS - search.REINFORCE_FROM_TICK + 1
        _, masks = env._collect_observations()
        mask = masks[defender.name]
        moving_actions = np.concatenate([
            mask[base * len(search.FACING_DIRS):(base + 1) * len(search.FACING_DIRS)]
            for base in range(2, 10)
        ])
        self.assertTrue(moving_actions.any())

    def test_site_context_matches_runtime_and_distinguishes_site_roles(self):
        env = search.SearchEnv()
        env.reset()
        env.in_setup_phase = False
        self.assertGreaterEqual(len(search.SITE_POSITIONS), 2)
        env.defenders[0].assigned_defense_pos = search.SITE_POSITIONS[1]
        env.defenders[1].assigned_defense_pos = search.SITE_POSITIONS[0]
        for attacker in env.attackers:
            attacker.is_alive = False
        target_site = search.SITE_POSITIONS[0]
        env.team_memory.spike_pos = tuple(map(int, target_site))
        env.team_memory.spike_held = True
        env._update_priority_dist_maps()
        train_obs, _ = env._collect_observations()

        controller = object.__new__(runtime.Ov1LearningDefenderSearchController)
        controller.team_memory = runtime._TeamMemory()
        controller.team_memory.spike_pos = env.team_memory.spike_pos
        controller.team_memory.spike_held = True
        controller.spike_dist_map = runtime._bfs_distance_map(
            search.GRID, controller.team_memory.spike_pos
        )
        controller.sighting_dist_map = None
        controller._assigned_positions = {
            defender.name: defender.assigned_defense_pos
            for defender in env.defenders
        }
        controller._assigned_dist_maps = {
            defender.name: defender.assigned_defense_dist_map
            for defender in env.defenders
        }
        controller._assigned_setup_dist_maps = {
            defender.name: defender.assigned_setup_dist_map
            for defender in env.defenders
        }
        for defender in env.defenders:
            runtime_obs, _ = controller._build_observation(
                defender,
                {"grid": search.GRID, "chars": env.defenders + env.attackers,
                 "round_timer": env.round_timer},
                search.SITE_POSITIONS, False,
            )
            np.testing.assert_allclose(
                train_obs[defender.name][-search.SITE_CONTEXT_DIM:],
                runtime_obs[-runtime.SITE_CONTEXT_DIM:],
            )
        same_site = [
            train_obs[d.name][-search.SITE_CONTEXT_DIM + 2]
            for d in env.defenders
        ]
        self.assertIn(1.0, same_site)
        self.assertIn(0.0, same_site)

    def test_site_representatives_are_reachable_plant_cells(self):
        for site, dist_map in zip(search.SITE_POSITIONS, search.SITE_DIST_MAPS):
            self.assertEqual(search.GRID[site], 2)
            self.assertEqual(dist_map[site], 0)
        self.assertEqual(search.SITE_POSITIONS,
                         runtime._extract_site_positions(search.GRID))

    def test_opposite_site_rotation_is_rewarded_without_chasing_spike_cell(self):
        self.assertGreaterEqual(len(search.SITE_POSITIONS), 2)
        target_site = search.SITE_POSITIONS[0]
        opposite_site = search.SITE_POSITIONS[1]
        target_map = search.SITE_DIST_MAPS[0]

        def reward_for(move):
            env = search.SearchEnv()
            env.reset()
            env.in_setup_phase = False
            for attacker in env.attackers:
                attacker.is_alive = False
            env.team_memory.spike_pos = target_site
            env.team_memory.spike_held = True
            env._update_priority_dist_maps()
            defender = env.defenders[0]
            defender.assigned_defense_pos = opposite_site
            defender.pos = list(opposite_site)
            defender._position_before_step = tuple(defender.pos)
            env.position_facing_ticks[defender.name] = search.FACING_SETUP_REWARD_TICKS
            defender.pos[0] += move[0]
            defender.pos[1] += move[1]
            defender.moved_this_tick = move != (0, 0)
            return env._compute_rewards({}, {}, {}, {})[defender.name]

        env = search.SearchEnv()
        env.reset()
        toward = search.bfs_best_direction(target_map, *opposite_site)
        self.assertNotEqual(toward, (0, 0))
        self.assertGreater(reward_for(toward), reward_for((0, 0)))

    def test_old_carrier_sighting_expires_in_training_and_runtime(self):
        defender = search.UnitStub("defender", "D", (1, 1), "HUNT")
        attacker = search.UnitStub("attacker", "A", (1, 4), "HUNT")
        for memory, update in (
            (search.TeamMemory(), lambda m: m.update(
                [defender], [attacker], set(), round_tick=1)),
            (runtime._TeamMemory(), lambda m: m.update(
                search.GRID, "D", [defender, attacker], round_tick=1)),
        ):
            memory.spike_pos = (1, 4)
            memory.spike_held = True
            with patch.object(search, "has_los", return_value=False), patch.object(
                runtime, "_has_los", return_value=False,
            ):
                for _ in range(search.SPIKE_CARRIER_MEMORY_TICKS + 1):
                    update(memory)
            self.assertIsNone(memory.spike_pos)

    def test_isolated_defender_gets_reward_for_regrouping(self):
        def reward_with_pressure(under_pressure):
            env = search.SearchEnv()
            env.reset()
            env.in_setup_phase = False
            defender = env.defenders[0]
            for attacker in env.attackers:
                attacker.is_alive = False
            env.team_memory.reset()
            env._update_priority_dist_maps()
            before = tuple(defender.assigned_defense_pos)
            toward_ally = next(
                (dr, dc) for dr, dc in search.CARDINAL
                if search.GRID[before[0] + dr, before[1] + dc] != 1
            )
            ally_pos = (before[0] + toward_ally[0],
                        before[1] + toward_ally[1])
            defender.pos = list(ally_pos)
            defender._position_before_step = before
            defender.moved_this_tick = True
            env._pressure_before_step = {
                defender.name: (under_pressure, ally_pos)
            }
            env.position_facing_ticks[defender.name] = search.FACING_SETUP_REWARD_TICKS
            return env._compute_rewards({}, {}, {}, {})[defender.name]

        self.assertAlmostEqual(
            reward_with_pressure(True) - reward_with_pressure(False),
            search.REGROUP_REWARD,
        )

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
