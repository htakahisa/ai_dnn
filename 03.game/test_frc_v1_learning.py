"""Real-game rollout, actor/critic separation and checkpoint/PPO round trips."""

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from frc_v1.actions import FrcAction, TeamDecision, MOVE_STEPS, build_masks, validate_action
from frc_v1.baseline import FrcBaseline, facing_to, plant_sites, route_step
from frc_v1.environment import FrcRoundEnvironment, plant_distances
from frc_v1.model import FrcPolicy
from frc_v1.navigation import (EAST_LONG_WAYPOINT, _danger_cells, _line_clear,
                               guard_attack_navigation, guard_tactical_utility,
                               site_approaches, site_watch_points)
from frc_v1.navigation import defense_anchor_positions, defense_site_assignments, guard_defense_navigation
from frc_v1.postplant import guard_attack_postplant, postplant_approaches, postplant_positions
from frc_v1.perception import Sighting
from frc_v1.memory import FrcMemory
from frc_v1.train import action_record, gae
from public_effects import DisplayEffect


class FrcLearningTests(unittest.TestCase):
    def test_attacker_moves_during_setup_without_replanning_each_tick(self):
        env, _ = self.make_environment(side="A", stage="match")
        snapshot = env.controller.snapshot
        self.assertEqual("setup", snapshot.phase)
        self.assertTrue(snapshot.setup_cells)
        policy = FrcPolicy(snapshot.grid, "A")
        starts = tuple(ally.position for ally in snapshot.allies)
        plan = None
        for _ in range(snapshot.tick):
            snapshot = env.controller.snapshot
            waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
            with patch.object(policy, "sample", return_value=waiting):
                decision = policy.act(env.controller.observation, snapshot, env.controller.belief)
            if plan is None:
                plan = policy._attack_setup_anchors
            self.assertEqual(plan, policy._attack_setup_anchors)
            for slot, action in enumerate(decision.actions):
                validate_action(snapshot, env.controller.observation.masks, slot, action)
            env.step(decision)
        self.assertNotEqual(starts[2], env.controller.snapshot.allies[2].position)
        self.assertEqual(plan[2], env.controller.snapshot.allies[2].position)
        site = plant_sites(snapshot.grid)[decision.site]
        current = env.controller.snapshot.allies
        self.assertLess(route_step(snapshot.grid, current[2].position, site)[1],
                        route_step(snapshot.grid, current[0].position, site)[1])
        self.assertLess(route_step(snapshot.grid, current[2].position, site)[1],
                        route_step(snapshot.grid, current[1].position, site)[1])
        setup_distance = route_step(snapshot.grid, current[2].position, site)[1]
        for _ in range(12):
            snapshot = env.controller.snapshot
            waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
            with patch.object(policy, "sample", return_value=waiting):
                decision = policy.act(env.controller.observation, snapshot, env.controller.belief)
            env.step(decision)
        current = env.controller.snapshot.allies
        self.assertLessEqual(route_step(snapshot.grid, current[2].position, site)[1],
                             setup_distance - 5)

    def test_plant_distance_uses_walkable_route(self):
        grid = ((0, 1, 2), (0, 1, 0), (0, 0, 0))
        distances = plant_distances(grid)
        self.assertEqual(6, int(distances[0, 0]))
        self.assertEqual(-1, int(distances[0, 1]))

    def test_future_ally_occupancy_does_not_close_single_corridor(self):
        grid = ((0, 0, 0, 0, 2),)
        self.assertEqual(("E", 4), route_step(grid, (0, 0), ((0, 4),),
                                              first_step_blocked=((0, 2),)))
        self.assertEqual(("STAY", 10000), route_step(grid, (0, 0), ((0, 4),),
                                                     first_step_blocked=((0, 1),)))

    def test_entry_opens_spawn_before_carrier_exits(self):
        env, observation = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        observation = env.controller.observation
        snapshot = env.controller.snapshot
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        first = guard_attack_navigation(waiting, snapshot, observation.masks)
        self.assertEqual("N", first.actions[2].kind)
        self.assertEqual("STAY", first.actions[0].kind)
        env.step(first)
        snapshot = env.controller.snapshot
        second = guard_attack_navigation(waiting, snapshot, env.controller.observation.masks)
        self.assertEqual("STAY", second.actions[0].kind)
        env.step(second)
        for _ in range(10):
            snapshot = env.controller.snapshot
            following = guard_attack_navigation(waiting, snapshot, env.controller.observation.masks)
            env.step(following)
            if following.actions[0].kind == "N":
                break
        else:
            self.fail("carrier never followed Lohen out of spawn")
        self.assertNotEqual((23, 18), env.controller.snapshot.allies[0].position)

    def test_carrier_reaches_both_sites_with_other_allies_holding_spawn(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        initial = env.controller.snapshot
        for site_index in (0, 1):
            with self.subTest(site=site_index):
                snapshot = initial
                for _ in range(100):
                    masks = build_masks(snapshot)
                    waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
                    decision = guard_attack_navigation(waiting, snapshot, masks, site_index=site_index)
                    for slot, action in enumerate(decision.actions):
                        validate_action(snapshot, masks, slot, action)
                    if decision.actions[0].kind == "PLANT":
                        break
                    moved = []
                    for ally, action in zip(snapshot.allies, decision.actions):
                        dr, dc = MOVE_STEPS.get(action.kind, (0, 0))
                        moved.append(replace(ally, position=(ally.position[0] + dr, ally.position[1] + dc)))
                    self.assertEqual(5, len({a.position for a in moved}))
                    snapshot = replace(snapshot, allies=tuple(moved), tick=snapshot.tick + 1)
                else:
                    self.fail(f"carrier remained unable to reach site {site_index}: "
                              f"positions={[a.position for a in snapshot.allies]}, "
                              f"actions={[a.kind for a in decision.actions]}")
                self.assertLess(_, 60, "the attack spent too long circling before a plant")
                self.assertNotEqual(3, snapshot.grid[snapshot.allies[1].position[0]][snapshot.allies[1].position[1]],
                                    "Lisa remained stuck in attacker spawn")

    def test_learned_attacker_rotates_site_between_match_rounds(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        policy = FrcPolicy(snapshot.grid, "A")
        with patch.object(policy, "sample", return_value=waiting):
            first = policy.act(env.controller.observation, snapshot, env.controller.belief)
            second = policy.act(env.controller.observation,
                                replace(snapshot, round_number=snapshot.round_number + 1),
                                env.controller.belief)
            third = policy.act(env.controller.observation,
                               replace(snapshot, round_number=snapshot.round_number + 2),
                               env.controller.belief)
            fourth = policy.act(env.controller.observation,
                                replace(snapshot, round_number=snapshot.round_number + 3),
                                env.controller.belief)
            long_waypoint = policy._navigation_waypoint
            fifth = policy.act(env.controller.observation,
                               replace(snapshot, round_number=snapshot.round_number + 4),
                               env.controller.belief)
        self.assertEqual((1, 0, 1, 0, 1),
                         (first.site, second.site, third.site, fourth.site, fifth.site))
        self.assertEqual(EAST_LONG_WAYPOINT, long_waypoint)

    def test_east_long_route_reaches_outer_corridor_before_site(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        visited = set()
        for _ in range(90):
            masks = build_masks(snapshot)
            waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
            decision = guard_attack_navigation(waiting, snapshot, masks, site_index=0,
                                               route_waypoint=EAST_LONG_WAYPOINT)
            for slot, action in enumerate(decision.actions):
                validate_action(snapshot, masks, slot, action)
            visited.add(snapshot.allies[2].position)
            if decision.actions[0].kind == "PLANT":
                break
            moved = []
            for ally, action in zip(snapshot.allies, decision.actions):
                dr, dc = MOVE_STEPS.get(action.kind, (0, 0))
                moved.append(replace(ally, position=(ally.position[0] + dr, ally.position[1] + dc)))
            snapshot = replace(snapshot, allies=tuple(moved), tick=snapshot.tick + 1)
        else:
            self.fail("east long route did not reach the site")
        self.assertIn(EAST_LONG_WAYPOINT, visited)
        self.assertNotIn((14, 23), visited)

    def test_long_route_support_does_not_retreat_behind_teammates(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        positions = ((22, 32), (22, 15), (18, 32), (16, 25), (16, 27))
        allies = tuple(replace(ally, position=positions[ally.slot]) for ally in snapshot.allies)
        snapshot = replace(snapshot, allies=allies)
        actions = [FrcAction(facing=ally.facing) for ally in allies]
        actions[1] = FrcAction("W", allies[1].facing)
        waiting = TeamDecision(tuple(actions))
        guarded = guard_attack_navigation(waiting, snapshot, build_masks(snapshot),
                                           site_index=0, route_waypoint=EAST_LONG_WAYPOINT)
        self.assertEqual("E", guarded.actions[1].kind)

    def test_support_clears_site_entrance_for_spike_carrier(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        positions = ((14, 40), (10, 40), (9, 40), (11, 40), (10, 41))
        allies = tuple(replace(ally, position=positions[ally.slot]) for ally in snapshot.allies)
        snapshot = replace(snapshot, allies=allies)
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in allies))
        guarded = guard_attack_navigation(waiting, snapshot, build_masks(snapshot), site_index=0)
        self.assertEqual("N", guarded.actions[2].kind)
        self.assertEqual("E", guarded.actions[4].kind)
        for slot, action in enumerate(guarded.actions):
            validate_action(snapshot, build_masks(snapshot), slot, action)

    def test_own_ash_area_does_not_block_attack_navigation(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        target = (7, 40)
        effect = DisplayEffect(100, "DESTRUCTION", "active", target,
                               cells=(target, (7, 39), (8, 40)))
        memory = FrcMemory()
        memory.update(snapshot)
        memory.record_own_cast("ASH", (8, 38), target, snapshot.tick)
        observed = replace(snapshot, tick=snapshot.tick + 3, effects=(effect,))
        belief = memory.update(observed)
        self.assertEqual("own", belief.effects[0].affiliation)
        self.assertFalse(_danger_cells(observed, belief))
        self.assertIn(target, _danger_cells(observed))

    def test_attacker_retrieves_spike_after_carrier_dies(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        allies = list(snapshot.allies)
        dropped = allies[0].position
        allies[0] = replace(allies[0], alive=False, has_spike=False)
        snapshot = replace(snapshot, allies=tuple(allies), spike_dropped=dropped)
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        masks = build_masks(snapshot)
        decision = guard_attack_navigation(waiting, snapshot, masks)
        self.assertEqual("REGROUP", decision.phase)
        self.assertEqual(1, decision.objective)
        self.assertEqual("W", decision.actions[1].kind)
        validate_action(snapshot, masks, 1, decision.actions[1])

    def test_carrier_waits_in_corridor_instead_of_detouring_backwards(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        allies = list(snapshot.allies)
        allies[0] = replace(allies[0], position=(12, 40))
        allies[1] = replace(allies[1], position=(11, 40))
        allies[2] = replace(allies[2], position=(10, 40))
        snapshot = replace(snapshot, allies=tuple(allies))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        decision = guard_attack_navigation(waiting, snapshot, build_masks(snapshot), site_index=0)
        self.assertEqual("STAY", decision.actions[0].kind)

    def test_postplant_holds_distinct_positions_aimed_at_retake_angles(self):
        for spike in ((8, 3), (7, 40)):
            with self.subTest(spike=spike):
                snapshot = replace(self.postplant_snapshot(), spike_planted=spike)
                anchors, approaches = postplant_positions(snapshot)
                self.assertGreaterEqual(len(approaches), 2)
                self.assertEqual(approaches, postplant_approaches(snapshot.grid, spike))
                self.assertEqual(len(anchors), len(set(anchors)))
                waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
                masks = build_masks(snapshot)
                guarded = guard_attack_postplant(waiting, snapshot, masks, plan=(anchors, approaches))
                self.assertEqual("POSTPLANT", guarded.phase)
                self.assertTrue(any(action.kind in MOVE_STEPS for action in guarded.actions))
                for slot, action in enumerate(guarded.actions):
                    validate_action(snapshot, masks, slot, action)
                    self.assertEqual(facing_to(snapshot.allies[slot].position,
                                               approaches[slot % len(approaches)]), action.facing)

    def test_postplant_recons_smoked_spike_and_closes_on_defuse(self):
        snapshot = replace(self.postplant_snapshot(), smoke_cells=((8, 3),), defuse_notified=True)
        actions = [FrcAction(facing=a.facing) for a in snapshot.allies]
        actions[1] = FrcAction("ABILITY", actions[1].facing, target=(8, 3))
        raw = TeamDecision(tuple(actions))
        masks = build_masks(snapshot)
        guarded = guard_attack_postplant(raw, snapshot, masks)
        self.assertEqual("ABILITY", guarded.actions[3].kind)
        self.assertLessEqual(max(abs(guarded.actions[3].target[0] - 8),
                                 abs(guarded.actions[3].target[1] - 3)), 2)
        self.assertNotEqual("ABILITY", guarded.actions[1].kind)
        self.assertTrue(any(action.kind in MOVE_STEPS for slot, action in enumerate(guarded.actions)
                            if slot != 3))
        for slot, action in enumerate(guarded.actions):
            validate_action(snapshot, masks, slot, action)
        cooling = guard_attack_postplant(raw, snapshot, masks, last_recon_tick=snapshot.tick - 1)
        self.assertNotEqual("ABILITY", cooling.actions[3].kind)
        nearby_smoke = replace(snapshot, smoke_cells=((8, 4),), defuse_notified=False)
        self.assertEqual("ABILITY", guard_attack_postplant(
            raw, nearby_smoke, build_masks(nearby_smoke)).actions[3].kind)

    def test_postplant_defuse_keeps_aim_on_spike_despite_distant_sighting(self):
        snapshot = replace(self.postplant_snapshot(), defuse_notified=True,
                           sightings=(Sighting(0, (20, 20), "vision"),))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        guarded = guard_attack_postplant(waiting, snapshot, build_masks(snapshot))
        for ally, action in zip(snapshot.allies, guarded.actions):
            self.assertEqual(facing_to(ally.position, snapshot.spike_planted), action.facing)

    def test_learned_policy_uses_postplant_plan_and_recon_cooldown(self):
        snapshot = replace(self.postplant_snapshot(), smoke_cells=((8, 3),), defuse_notified=True)
        observation = SimpleNamespace(masks=build_masks(snapshot))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        policy = FrcPolicy(snapshot.grid, "A")
        with patch.object(policy, "sample", return_value=waiting):
            first = policy.act(observation, snapshot, None)
            second = policy.act(observation, snapshot, None)
        self.assertEqual("ABILITY", first.actions[3].kind)
        self.assertNotEqual("ABILITY", second.actions[3].kind)
        self.assertIsNotNone(policy._postplant_plan)

    def test_postplant_recon_cast_is_executed_by_game(self):
        env, _ = self.make_environment(side="A", stage="match")
        game = env.game
        game.defender_setup_phase.finish()
        positions = ((11, 3), (11, 4), (11, 2), (7, 5), (12, 2))
        for ally in (character for character in game.chars if character.team == "A"):
            slot = ("Furina", "Lisa", "Lohen", "Jean", "Arlecchino").index(str(ally.base_name))
            ally.pos = list(positions[slot])
            ally.has_spike = False
        game.is_planted = True
        game.planted_pos = (8, 3)
        game.smokes = [{"center": (8, 3), "cells": {(8, 3), (8, 4), (7, 3)},
                        "remaining_ticks": 10, "owner": "opponent", "team": "D"}]
        game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        self.assertIn((8, 3), snapshot.smoke_cells)
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        policy = FrcPolicy(snapshot.grid, "A")
        with patch.object(policy, "sample", return_value=waiting):
            decision = policy.act(env.controller.observation, snapshot, env.controller.belief)
        self.assertEqual("ABILITY", decision.actions[3].kind)
        charges = snapshot.allies[3].charges
        env.step(decision)
        self.assertEqual(charges - 1, env.controller.snapshot.allies[3].charges)

    def postplant_snapshot(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        positions = ((11, 3), (11, 4), (11, 2), (7, 5), (12, 2))
        allies = tuple(replace(ally, position=positions[ally.slot]) for ally in snapshot.allies)
        return replace(snapshot, allies=allies, sightings=(), is_planted=True,
                       spike_planted=(8, 3))

    def test_defense_splits_sites_and_retakes_after_plant(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        assignments = defense_site_assignments(snapshot)
        anchors = defense_anchor_positions(snapshot, assignments)
        self.assertIn(0, assignments)
        self.assertIn(1, assignments)
        self.assertEqual(len(anchors), len(set(anchors)))
        for slot, anchor in enumerate(anchors):
            self.assertLessEqual(route_step(snapshot.grid, anchor, plant_sites(snapshot.grid)[assignments[slot]])[1], 2)
            self.assertTrue(any(_line_clear(snapshot.grid, anchor, entrance)
                                for entrance in site_watch_points(snapshot.grid,
                                    plant_sites(snapshot.grid)[assignments[slot]])))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        first = guard_defense_navigation(waiting, snapshot, build_masks(snapshot),
                                         site_assignments=assignments, anchor_positions=anchors)
        self.assertTrue(all(a.kind in MOVE_STEPS for a in first.actions))
        allies = list(snapshot.allies)
        allies[3] = replace(allies[3], position=anchors[3])
        anchored = replace(snapshot, allies=tuple(allies))
        self.assertEqual("STAY", guard_defense_navigation(
            waiting, anchored, build_masks(anchored), site_assignments=assignments,
            anchor_positions=anchors).actions[3].kind)
        planted = replace(anchored, is_planted=True, spike_planted=(7, 3))
        retake = guard_defense_navigation(waiting, planted, build_masks(planted),
                                          site_assignments=assignments, anchor_positions=anchors)
        self.assertEqual("RETAKE", retake.phase)
        self.assertIn(retake.actions[3].kind, MOVE_STEPS)
        for slot, action in enumerate(retake.actions):
            validate_action(planted, build_masks(planted), slot, action)

    def test_defense_moves_during_setup_and_varies_anchors_by_round(self):
        env, _ = self.make_environment(side="D", stage="match")
        snapshot = env.controller.snapshot
        self.assertEqual("setup", snapshot.phase)
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        assignments = defense_site_assignments(snapshot)
        anchors = defense_anchor_positions(snapshot, assignments)
        allowed = set(snapshot.setup_cells)
        setup_grid = tuple(tuple(0 if (r, c) in allowed else 1
                                 for c in range(len(row))) for r, row in enumerate(snapshot.grid))
        for ally, anchor in zip(snapshot.allies, anchors):
            self.assertIn(anchor, allowed)
            self.assertLessEqual(route_step(setup_grid, ally.position, (anchor,))[1], snapshot.tick)
        decision = guard_defense_navigation(waiting, snapshot, build_masks(snapshot),
                                            site_assignments=assignments, anchor_positions=anchors)
        self.assertEqual("PREPARE", decision.phase)
        self.assertTrue(any(action.kind in MOVE_STEPS for action in decision.actions))
        for slot, action in enumerate(decision.actions):
            validate_action(snapshot, build_masks(snapshot), slot, action)
        starts = tuple(a.position for a in snapshot.allies)
        for _ in range(4):
            current = env.controller.snapshot
            waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in current.allies))
            decision = guard_defense_navigation(waiting, current, build_masks(current),
                                                site_assignments=assignments, anchor_positions=anchors)
            env.step(decision)
        self.assertNotEqual(starts, tuple(a.position for a in env.controller.snapshot.allies))
        following = replace(snapshot, round_number=snapshot.round_number + 1)
        next_assignments = defense_site_assignments(following)
        next_anchors = defense_anchor_positions(following, next_assignments)
        self.assertNotEqual((assignments, anchors), (next_assignments, next_anchors))

    def test_defense_training_episodes_rotate_setup_positions(self):
        env, _ = self.make_environment(side="D", stage="match")
        snapshot = env.controller.snapshot
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        policy = FrcPolicy(snapshot.grid, "D")
        with patch.object(policy, "sample", return_value=waiting):
            policy.act(env.controller.observation, snapshot, env.controller.belief)
            first = policy._defense_assignments, policy._defense_anchors
            policy.act(env.controller.observation, replace(snapshot, phase="live", tick=1),
                       env.controller.belief)
            policy.act(env.controller.observation, snapshot, env.controller.belief)
            second = policy._defense_assignments, policy._defense_anchors
        self.assertNotEqual(first, second)

    def test_defense_refuses_site_smoke_and_defuses_when_close(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        smoke = TeamDecision(tuple(FrcAction("ABILITY", a.facing, target=(7, 3))
                                   if a.slot == 1 else FrcAction(facing=a.facing)
                                   for a in snapshot.allies))
        initial = guard_tactical_utility(guard_defense_navigation(
            smoke, snapshot, build_masks(snapshot)), snapshot, build_masks(snapshot)).actions[1]
        self.assertNotEqual((7, 3), initial.target)
        sighted_site = replace(snapshot, sightings=(Sighting(0, (7, 3), "reveal"),))
        redirected = guard_tactical_utility(guard_defense_navigation(
            smoke, sighted_site, build_masks(sighted_site)), sighted_site,
            build_masks(sighted_site)).actions[1]
        self.assertNotEqual((7, 3), redirected.target)
        allies = list(snapshot.allies)
        allies[0] = replace(allies[0], position=(7, 3))
        planted = replace(snapshot, allies=tuple(allies), is_planted=True, spike_planted=(7, 4))
        guarded = guard_tactical_utility(guard_defense_navigation(
            smoke, planted, build_masks(planted)), planted, build_masks(planted))
        self.assertEqual("DEFUSE", guarded.actions[0].kind)
        self.assertNotEqual("ABILITY", guarded.actions[1].kind)

    def test_defense_uses_recon_on_visible_enemy(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = replace(env.controller.snapshot, sightings=(Sighting(0, (2, 22), "reveal"),))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        masks = build_masks(snapshot)
        guarded = guard_defense_navigation(waiting, snapshot, masks)
        self.assertEqual("ABILITY", guarded.actions[3].kind)
        self.assertEqual((2, 22), guarded.actions[3].target)

    def test_defense_uses_smoke_and_ash_at_attacked_entrance(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        allies = list(snapshot.allies)
        allies[4] = replace(allies[4], position=(8, 38))
        snapshot = replace(snapshot, allies=tuple(allies),
                           sightings=(Sighting(0, (11, 40), "reveal"),))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        masks = build_masks(snapshot)
        guarded = guard_tactical_utility(waiting, snapshot, masks)
        self.assertEqual("ABILITY", guarded.actions[1].kind)
        self.assertEqual("ABILITY", guarded.actions[4].kind)
        for slot in (1, 4):
            validate_action(snapshot, masks, slot, guarded.actions[slot])
        repeated = guard_tactical_utility(waiting, snapshot, masks,
                                          cast_history={4: [guarded.actions[4].target]})
        if repeated.actions[4].kind == "ABILITY":
            self.assertGreater(max(abs(a - b) for a, b in zip(
                repeated.actions[4].target, guarded.actions[4].target)), 2)
        cooling = guard_tactical_utility(waiting, snapshot, masks,
                                          last_cast={1: snapshot.tick, 4: snapshot.tick})
        self.assertNotEqual("ABILITY", cooling.actions[1].kind)
        self.assertNotEqual("ABILITY", cooling.actions[4].kind)

    def test_defense_looks_down_mid_lane_and_reacts_to_site_casualty(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        east_site = plant_sites(snapshot.grid)[0]
        self.assertIn((7, 31), site_watch_points(snapshot.grid, east_site))
        allies = list(snapshot.allies)
        allies[2] = replace(allies[2], position=(7, 38))
        allies[3] = replace(allies[3], position=(4, 38))
        allies[4] = replace(allies[4], position=(2, 40))
        positioned = replace(snapshot, allies=tuple(allies))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in positioned.allies))
        aimed = guard_defense_navigation(waiting, positioned, build_masks(positioned),
                                           site_assignments=(1, 1, 0, 0, 0),
                                           anchor_positions=tuple(a.position for a in positioned.allies))
        self.assertEqual("W", aimed.actions[2].facing)
        mid_contact = replace(positioned, sightings=(Sighting(0, (6, 23), "reveal"),))
        early = guard_tactical_utility(waiting, mid_contact, build_masks(mid_contact))
        self.assertEqual(("ABILITY", (6, 23)),
                         (early.actions[1].kind, early.actions[1].target))
        self.assertNotEqual("ABILITY", early.actions[4].kind)
        allies[2] = replace(allies[2], alive=False)
        attacked = replace(positioned, allies=tuple(allies))
        utility = guard_tactical_utility(waiting, attacked, build_masks(attacked))
        self.assertEqual("ABILITY", utility.actions[1].kind)
        self.assertEqual("ABILITY", utility.actions[4].kind)

    def test_attacker_uses_smoke_and_ash_near_defender_entrance(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        allies = list(snapshot.allies)
        allies[4] = replace(allies[4], position=(8, 38))
        snapshot = replace(snapshot, allies=tuple(allies),
                           sightings=(Sighting(0, (5, 39), "reveal"),))
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies), site=0)
        masks = build_masks(snapshot)
        guarded = guard_tactical_utility(waiting, snapshot, masks)
        self.assertEqual("ABILITY", guarded.actions[1].kind)
        self.assertEqual("ABILITY", guarded.actions[4].kind)
        for slot in (1, 4):
            validate_action(snapshot, masks, slot, guarded.actions[slot])

    def test_defense_waits_for_teammate_before_retake(self):
        env, _ = self.make_environment(side="D", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot = env.controller.snapshot
        allies = list(snapshot.allies)
        allies[0] = replace(allies[0], position=(7, 6))
        planted = replace(snapshot, allies=tuple(allies), is_planted=True,
                          spike_planted=(7, 3), detonate_timer=30)
        waiting = TeamDecision(tuple(FrcAction(facing=a.facing) for a in planted.allies))
        first = guard_defense_navigation(waiting, planted, build_masks(planted))
        self.assertEqual("STAY", first.actions[0].kind)
        allies[1] = replace(allies[1], position=(8, 6))
        grouped = replace(planted, allies=tuple(allies))
        together = guard_defense_navigation(waiting, grouped, build_masks(grouped))
        self.assertIn(together.actions[0].kind, MOVE_STEPS)

    def test_training_actor_executes_navigation_but_keeps_sampled_record(self):
        env, _ = self.make_environment(side="A", stage="match")
        env.game.defender_setup_phase.finish()
        env.game._prepare_team_controllers_tick()
        snapshot, observation = env.controller.snapshot, env.controller.observation
        raw = TeamDecision(tuple(FrcAction(facing=a.facing) for a in snapshot.allies))
        record = action_record(raw, env.game.width)
        policy = FrcPolicy(snapshot.grid, "A", deterministic=False)
        policy.last_sample = (record, 0.0, 0.0)
        with patch.object(policy, "sample", return_value=raw) as sample:
            executed = policy.act(observation, snapshot, env.controller.belief,
                                  critic=env.critic_state())
        sample.assert_called_once()
        self.assertEqual(0, int(policy.last_sample[0]["kind"][2]))
        self.assertEqual("N", executed.actions[2].kind)
        for slot, action in enumerate(executed.actions):
            validate_action(snapshot, observation.masks, slot, action)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_environment(self, side="A", stage="threats"):
        env = FrcRoundEnvironment(side, stage=stage, opponent="default", seed=73)
        with contextlib.redirect_stdout(io.StringIO()):
            observation = env.reset()
        return env, observation

    def test_actor_actions_do_not_depend_on_privileged_critic(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A")
        first = policy.model.distribution([observation], deterministic=True, critic=[np.zeros(100, np.float32)])
        second = policy.model.distribution([observation], deterministic=True, critic=[np.ones(100, np.float32)])
        for name in first[0]:
            torch.testing.assert_close(first[0][name], second[0][name])
        torch.testing.assert_close(first[1], second[1])

    def test_policy_can_sample_real_legal_actions_and_recompute_log_probability(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A", deterministic=False)
        decision = policy.sample(observation, critic=env.critic_state())
        record, old_log, _ = policy.last_sample
        for slot, action in enumerate(decision.actions):
            validate_action(env.controller.snapshot, observation.masks, slot, action)
        _, log_prob, entropy, value = policy.model.distribution([observation], records=[record], critic=[env.critic_state()])
        self.assertAlmostEqual(old_log, float(log_prob.detach()[0]), places=5)
        self.assertTrue(torch.isfinite(entropy).all())
        loss = -log_prob.mean() + value.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.model.parameters() if p.grad is not None))
        env.step(decision)

    def test_checkpoint_load_restores_same_actions_and_rejects_other_side(self):
        env, observation = self.make_environment()
        policy = FrcPolicy(env.controller.snapshot.grid, "A")
        expected = policy.sample(observation)
        with tempfile.TemporaryDirectory(prefix="frc_checkpoint_", dir=Path(__file__).parent) as folder:
            path = Path(folder) / "policy.pt"
            policy.save(path, training={"smoke_only": True})
            restored = FrcPolicy.load(path, side="A")
            self.assertEqual(expected, restored.sample(observation))
            with self.assertRaises(ValueError):
                FrcPolicy.load(path, side="D")

    def test_missing_learned_model_never_silently_uses_baseline(self):
        with self.assertRaises(FileNotFoundError):
            FrcPolicy.load(Path(__file__).parent / "frc_v1" / "missing_policy.pt", side="A")

    def test_teacher_decisions_valid_across_curriculum_and_both_sides(self):
        teacher = FrcBaseline()
        for side, stage in (("A", "support"), ("D", "support"), ("A", "balemoon"), ("D", "balemoon")):
            with self.subTest(side=side, stage=stage):
                env, observation = self.make_environment(side, stage)
                positions = [tuple(c.pos) for c in env.game.chars if c.is_alive]
                self.assertEqual(len(positions), len(set(positions)))
                policy = FrcPolicy(env.controller.snapshot.grid, side)
                for _ in range(3):
                    decision = teacher.act(observation, env.controller.snapshot, env.controller.belief)
                    record = action_record(decision, env.game.width)
                    _, log_prob, _, _ = policy.model.distribution([observation], records=[record])
                    self.assertGreater(float(log_prob.detach()[0]), -10000)
                    result = env.step(decision)
                    observation = result.observation
                    if result.terminated:
                        break

    def test_gae_stops_bootstrapping_across_terminal_round(self):
        advantages, returns = gae([1, 2], [0.5, 0.7], [True, False], 3, gamma=0.9, lam=1)
        self.assertAlmostEqual(float(advantages[0]), 0.5)
        self.assertAlmostEqual(float(returns[1]), 4.7, places=5)

    def test_real_terminal_round_preserves_characters_and_winning_reward(self):
        env, observation = self.make_environment("D", "match")
        env.game.defender_setup_phase.finish()
        env.game.round_timer = 1
        env.game._prepare_team_controllers_tick()
        characters = tuple(env.game.chars)
        result = env.step(env.controller.decision)
        self.assertTrue(result.terminated)
        self.assertEqual("D", result.metrics["winner"])
        self.assertGreater(result.reward, 0.9)
        self.assertEqual(1, env.game.current_round)
        self.assertEqual(characters, tuple(env.game.chars))

    def test_gui_and_competition_registry_preserve_default_and_expose_both_modes(self):
        from roster_select import TEAM_AI_OPTIONS
        from run_competition_manager import CONTROLLER_OPTIONS
        from run_game import _build_team_ai
        self.assertEqual("Toru AI v3.1", next(iter(TEAM_AI_OPTIONS)))
        for values in (TEAM_AI_OPTIONS.values(), CONTROLLER_OPTIONS.values()):
            self.assertIn("frc_v1", values)
            self.assertIn("frc_v1_baseline", values)
        team = _build_team_ai("frc_v1_baseline")
        self.assertTrue(team.get_attacker_controller().handles_team_perception)
        self.assertEqual("D", team.get_defender_controller().side)

    def test_competition_learned_frc_loads_selected_checkpoint_pair(self):
        from run_competition_manager import CONTROLLER_OPTIONS
        from run_game import FRC_V1_ATTACKER_CHECKPOINT, FRC_V1_DEFENDER_CHECKPOINT, _build_team_ai

        with patch.object(FrcPolicy, "load", wraps=FrcPolicy.load) as load:
            team = _build_team_ai(CONTROLLER_OPTIONS["FRC v1（学習モデル）"])
            attacker = team.get_attacker_controller()
            defender = team.get_defender_controller()

        self.assertEqual("A", attacker.actor.side)
        self.assertEqual("D", defender.actor.side)
        self.assertEqual(
            [
                ((FRC_V1_ATTACKER_CHECKPOINT,), {"side": "A"}),
                ((FRC_V1_DEFENDER_CHECKPOINT,), {"side": "D"}),
            ],
            [(call.args, call.kwargs) for call in load.call_args_list],
        )


if __name__ == "__main__":
    unittest.main()
