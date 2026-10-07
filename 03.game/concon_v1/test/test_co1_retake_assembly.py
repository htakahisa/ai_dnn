"""Assembly release, stable entrances, facing and learned rally navigation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_retake_coordination import RetakeAssembly
from concon_v1.co1_retake_scenarios import get_scenario, make_checkpoint, validate_checkpoint
from concon_v1.co1_retake_common import build_inputs, decision_reward, MAP_CHANNELS, ULTIMATE_ACTION, DEFUSE_ACTION, MOVES
from concon_v1.co1_retake_foundation import navigation_cells, training_targets, teacher_values
from concon_v1.co1_retake_navigation import assembly_navigation, assembly_step_allowed
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.test.test_co1_defender_retake import actor
from concon_v1.test.test_co1_retake_foundation import fixture_model
from game_core import FACING_DIRECTIONS


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def planner(self):
        groups = (((2, 8), (3, 8), (4, 8)), ((2, 12), (3, 12), (4, 12)))
        scenario = SimpleNamespace(grid=np.zeros((7, 25), dtype=int), rally_groups=groups,
                                   rally_points=tuple(p for group in groups for p in group))
        return RetakeAssembly(scenario)

    def test_real_maps_group_adjacent_A_cells_into_two_entrances(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            self.assertEqual(len(scenario.rally_groups), 2)
            self.assertTrue(all(scenario.grid[p] != 1 for p in scenario.rally_points))

    def test_assignments_remain_stable_when_positions_change(self):
        planner = self.planner()
        allies = [actor(str(i), (0, i)) for i in range(5)]
        first = planner.update(allies, (3, 20), 55, 1)
        assignments = dict(first["assigned"])
        self.assertEqual(len(set(assignments.values())), 5)
        counts = [sum(p in group for p in assignments.values()) for group in planner.scenario.rally_groups]
        self.assertEqual(sorted(counts), [2, 3])
        for ally in allies:
            ally.pos[1] += 1
        self.assertEqual(planner.update(allies, (3, 20), 54, 2)["assigned"], assignments)

    def test_forward_advance_keeps_followers_goals_and_does_not_delay_release(self):
        planner = self.planner()
        planner = RetakeAssembly(planner.scenario, version=1)
        allies = [actor(str(i), pos) for i, pos in enumerate(((3, 8), (0, 8), (0, 7), (0, 12), (0, 13)))]
        planner.assigned = dict(zip((a.name for a in allies), ((3, 8), (2, 8), (4, 8), (2, 12), (3, 12))))
        original = dict(planner.assigned)
        plan = planner.update(allies, (3, 20), 55, 1)
        self.assertEqual(plan["assigned"]["0"], (4, 12))
        self.assertTrue(plan["waiting"])
        self.assertEqual({n: p for n, p in plan["assigned"].items() if n != "0"},
                         {n: p for n, p in original.items() if n != "0"})
        for ally in allies[1:]:
            ally.pos = list(original[ally.name])
        plan = planner.update(allies, (3, 20), 54, 2)
        self.assertFalse(plan["waiting"])
        self.assertNotEqual(tuple(allies[0].pos), plan["assigned"]["0"])
        # No return to assembly when an IQ estimate jitters away from A.
        allies[1].pos = [0, 0]
        self.assertFalse(planner.update(allies, (3, 20), 53, 3)["waiting"])

    def test_deadline_releases_without_distant_teammate(self):
        planner = self.planner()
        allies = [actor("near", (3, 8)), actor("far", (0, 0))]
        planner.assigned = {"near": (3, 8), "far": (3, 12)}
        self.assertTrue(planner.update(allies, (3, 20), 55, 1)["waiting"])
        plan = planner.update(allies, (3, 20), 23, 2)
        self.assertFalse(plan["waiting"])
        self.assertTrue(plan["urgent"])

    def test_release_at_other_A_cells_in_the_assigned_entrance(self):
        planner = self.planner()
        allies = [actor("one", (4, 8)), actor("two", (4, 12))]
        planner.assigned = {"one": (2, 8), "two": (2, 12)}
        plan = planner.update(allies, (3, 20), 55, 1)
        self.assertFalse(plan["waiting"])
        self.assertEqual(plan["near"], 2)
        self.assertEqual(planner.reached, set(planner.assigned.items()))

    def test_diagonal_IQ_error_does_not_delay_release(self):
        planner = self.planner()
        allies = [actor("one", (1, 7)), actor("two", (5, 13))]
        planner.assigned = {"one": (2, 8), "two": (4, 12)}
        self.assertFalse(planner.update(allies, (3, 20), 55, 1)["waiting"])

    def test_still_waits_for_teammate_two_straight_steps_from_entrance(self):
        planner = self.planner()
        allies = [actor("one", (3, 8)), actor("two", (3, 14))]
        planner.assigned = {"one": (3, 8), "two": (3, 12)}
        plan = planner.update(allies, (3, 20), 55, 1)
        self.assertTrue(plan["waiting"])
        self.assertNotIn("two", planner.ready)

    def test_wrong_entrance_does_not_count_as_arrival(self):
        planner = self.planner()
        allies = [actor("one", (3, 8)), actor("two", (4, 8))]
        planner.assigned = {"one": (3, 8), "two": (3, 12)}
        self.assertTrue(planner.update(allies, (3, 20), 55, 1)["waiting"])
        self.assertNotIn("two", planner.ready)

    def test_nearby_cell_across_wall_does_not_count_as_arrival(self):
        grid = np.zeros((7, 25), dtype=int)
        grid[2, 9] = grid[3, 8] = 1
        groups = (((2, 8),), ((3, 12),))
        scenario = SimpleNamespace(grid=grid, rally_groups=groups,
                                   rally_points=tuple(p for group in groups for p in group))
        planner = RetakeAssembly(scenario)
        allies = [actor("one", (3, 9)), actor("two", (3, 12))]
        planner.assigned = {"one": (2, 8), "two": (3, 12)}
        self.assertTrue(planner.update(allies, (3, 20), 55, 1)["waiting"])
        self.assertNotIn("one", planner.ready)

    def test_real_maps_release_at_any_cell_of_the_assigned_entrance(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            for version in (1, 2):
                for group in scenario.rally_groups:
                    for goal in group:
                        for position in group:
                            with self.subTest(site=site, version=version, goal=goal, position=position):
                                planner = RetakeAssembly(scenario, version=version)
                                allies = [actor("one", position), actor("two", position)]
                                planner.assigned = {"one": goal, "two": goal}
                                plan = planner.update(allies, (7, 3) if site == "L" else (7, 38), 55, 1)
                                self.assertFalse(plan["waiting"])
                                self.assertEqual(plan["near"], 2)

    def test_release_snapshot_is_shared_within_the_tick(self):
        planner = self.planner()
        allies = [actor("one", (0, 0)), actor("two", (0, 1))]
        first = planner.update(allies, (3, 20), 55, 1)
        self.assertTrue(first["waiting"])
        allies[0].pos = [3, 20]
        self.assertIs(planner.update(allies, (3, 20), 55, 1), first)
        self.assertFalse(planner.update(allies, (3, 20), 54, 2)["waiting"])

    def test_dead_actor_is_removed_from_assembly(self):
        planner = self.planner()
        allies = [actor("one", (0, 0)), actor("two", (0, 1))]
        planner.update(allies, (3, 20), 55, 1)
        plan = planner.update(allies[:1], (3, 20), 54, 2)
        self.assertNotIn("two", plan["assigned"])
        self.assertFalse(plan["waiting"])

    def controller_state(self):
        scenario = get_scenario("L")
        controller = ConconDefenderRetakeController("L", model=fixture_model(scenario), ability_distance=100)
        from concon_v1.co1_retake_common import GORIGONS
        chars = [actor(name, (22, 10 + i), ability="SMOKE") for i, name in enumerate(GORIGONS.players)]
        state = dict(grid=scenario.grid, chars=chars, planted_pos=(7, 3), smoke_cells=[],
                     battle_tick=1, detonate_timer=55, defender_defuse_info={})
        return controller, chars, state

    def test_fixed_casts_disabled_until_launch_and_A_channels_present(self):
        controller, chars, state = self.controller_state()
        obs, mask, context = build_inputs(controller, chars[0], state)
        self.assertTrue(context["waiting"])
        self.assertFalse(mask[40:ULTIMATE_ACTION].any())
        maps = obs[:controller.model.map_size].reshape(MAP_CHANNELS, *controller.scenario.grid.shape)
        self.assertEqual(maps[10].sum(), len(controller.scenario.rally_points))
        self.assertEqual(maps[11].sum(), len(chars))
        chars[0].pos = [12, 3]
        state.update(battle_tick=2, detonate_timer=9)
        self.assertTrue(build_inputs(controller, chars[0], state)[1][40:ULTIMATE_ACTION].any())

    def test_release_switches_goal_to_corresponding_entry(self):
        controller, chars, state = self.controller_state()
        for group, members in zip(controller.scenario.rally_groups, (chars[:3], chars[3:])):
            for index, char in enumerate(members):
                controller.assembly.assigned[char.name] = group[index]
                char.pos = list(group[-1 - index])
        self.assertTrue(any(controller.assembly.routes[p][tuple(char.pos)] > 1
                            for char in chars for p in [controller.assembly.assigned[char.name]]))
        for char in chars:
            _, _, context = build_inputs(controller, char, state)
            self.assertFalse(context["waiting"])
            self.assertIn(context["goal"], controller.scenario.entries_for(controller.assembly.assigned[char.name]))

    def test_learned_foundation_supports_A_goals_and_wait_without_defuse(self):
        controller, chars, state = self.controller_state()
        observation, mask, context = build_inputs(controller, chars[0], state)
        active, values = controller.model.foundation_inputs(torch.as_tensor(observation)[None])
        self.assertTrue(active.item())
        operations = values[0].detach().numpy()
        self.assertGreater(operations[:4].max(), operations[4])
        self.assertLess(operations[5], operations[:4].max())
        point = context["goal"]
        teacher = teacher_values(controller.scenario.grid, point, point, assembly=True)
        self.assertEqual(teacher.argmax(), 4)
        self.assertGreater(teacher[4], teacher[5])

    def test_all_reachable_A_approaches_are_covered_including_other_entrances(self):
        scenario = get_scenario("L")
        indices, _ = training_targets(scenario)
        covered = set(indices)
        cells = navigation_cells(scenario)
        for point in scenario.rally_points:
            target_index = cells.index(point)
            for origin in scenario.rally_points:
                self.assertIn(target_index * scenario.grid.size + origin[0] * scenario.grid.shape[1] + origin[1], covered)

    def test_shared_enemy_is_faced_from_the_peek_destination(self):
        controller, chars, state = self.controller_state()
        char = chars[0]
        char.pos = [7, 3]
        enemy = actor("shared", (8, 4), team="A")
        state["chars"] = [char, enemy]
        _, mask, context = build_inputs(controller, char, state)
        # Eastward movement exposes a shared enemy immediately to the south.
        facings = [FACING_DIRECTIONS[i] for i in np.flatnonzero(mask[24:32])]
        self.assertIn("S", facings)
        self.assertNotIn("N", facings)
        self.assertEqual(context["facing_targets"][(7, 4)], (8, 4))
        char.facing_forced_this_tick = True
        _, mask, _ = build_inputs(controller, char, state)
        self.assertEqual(np.flatnonzero(mask[24:32]).tolist(), [FACING_DIRECTIONS.index(char.facing)])

    def test_travel_facing_has_reward_but_other_facings_remain_learnable(self):
        controller, chars, state = self.controller_state()
        char = chars[0]
        char.pos = [7, 3]
        state["chars"] = [char]
        _, mask, context = build_inputs(controller, char, state)
        self.assertTrue(mask[24:32].all())
        self.assertGreater(decision_reward(24, context, (7, 4), "E"),
                           decision_reward(24, context, (7, 4), "W"))

    def test_old_behavior_checkpoint_is_rejected_explicitly(self):
        controller, _, _ = self.controller_state()
        checkpoint = make_checkpoint(controller.model, "L", 0, 6, "unused", [], 0)
        checkpoint["reward_version"] = 1
        with self.assertRaisesRegex(ValueError, "reward_version"):
            validate_checkpoint(checkpoint, controller.scenario, 6)

    def test_left_A_transfer_uses_southern_route_without_entering_site_lanes(self):
        scenario = get_scenario("L")
        front, routes = assembly_navigation(scenario, version=2)
        for goal in scenario.rally_groups[1]:
            position = (11, 16)
            route = [position]
            while position != goal:
                values = teacher_values(scenario.grid, goal, position, assembly=True,
                                        distances=routes[goal], front=front)
                operation = int(values.argmax())
                self.assertLess(operation, 4, (goal, position, values))
                dr, dc = MOVES[operation]
                position = position[0] + dr, position[1] + dc
                self.assertFalse(front[position], (goal, route, position))
                route.append(position)
                self.assertLess(len(route), 30)
            self.assertEqual(route[1], (12, 16))
            self.assertIn((18, 12), route)
            self.assertIn((18, 9), route)

    def test_site_entry_is_masked_during_assembly_and_enabled_on_launch(self):
        controller, chars, state = self.controller_state()
        controller.assembly = RetakeAssembly(controller.scenario, version=2)
        char = chars[0]
        char.pos = [10, 16]
        _, mask, context = build_inputs(controller, char, state)
        self.assertTrue(context["waiting"])
        self.assertTrue(controller.assembly.front[9, 16])
        self.assertFalse(mask[:8].any())
        self.assertTrue(mask[8:16].any())
        controller.assembly.launched = True
        state["battle_tick"] += 1
        self.assertTrue(build_inputs(controller, char, state)[1][:8].any())

    def test_actor_in_site_side_pocket_can_retreat_to_A(self):
        scenario = get_scenario("L")
        front, routes = assembly_navigation(scenario, version=2)
        position, goal = (12, 12), (11, 16)
        self.assertTrue(front[position])
        self.assertGreater(routes[goal][position], 0)
        crossed_to_rear = False
        for _ in range(40):
            if position == goal:
                break
            values = teacher_values(scenario.grid, goal, position, assembly=True,
                                    distances=routes[goal], front=front)
            operation = int(values.argmax())
            self.assertLess(operation, 4)
            dr, dc = MOVES[operation]
            destination = position[0] + dr, position[1] + dc
            self.assertTrue(assembly_step_allowed(front, position, destination))
            if crossed_to_rear:
                self.assertFalse(front[destination])
            crossed_to_rear |= not front[destination]
            position = destination
        self.assertEqual(position, goal)

    def test_restricted_assembly_checkpoint_is_rejected_for_version_one_training(self):
        controller, _, _ = self.controller_state()
        checkpoint = make_checkpoint(controller.model, "L", 0, 6, "unused", [], 0)
        checkpoint["coordination_version"] = 2
        with self.assertRaisesRegex(ValueError, "coordination_version"):
            validate_checkpoint(checkpoint, controller.scenario, 6)

    def test_default_foundation_and_battle_use_unrestricted_routes(self):
        from concon_v1.co1_attacker_common import bfs_distance_map
        controller, chars, state = self.controller_state()
        scenario = controller.scenario
        front, routes = assembly_navigation(scenario)
        self.assertEqual(controller.assembly.version, 3)
        self.assertFalse(front.any())
        indices, targets = training_targets(scenario)
        width, height = scenario.grid.shape[1], scenario.grid.shape[0]
        goals = navigation_cells(scenario)
        for goal, distances in routes.items():
            np.testing.assert_array_equal(distances, bfs_distance_map(scenario.grid, goal))
            position = (11, 16)
            index = goals.index(goal) * height * width + position[0] * width + position[1]
            row = np.flatnonzero(indices == index)
            self.assertEqual(len(row), 1)
            np.testing.assert_array_equal(targets[row[0]], teacher_values(
                scenario.grid, goal, position, assembly=True))
        chars[0].pos = [10, 16]
        _, mask, context = build_inputs(controller, chars[0], state)
        self.assertTrue(context["waiting"])
        self.assertTrue(mask[:8].any())
        checkpoint = make_checkpoint(controller.model, "L", 0, 6, "unused", [], 0)
        self.assertEqual(checkpoint["coordination_version"], 3)
        validate_checkpoint(checkpoint, scenario, 6)

    def test_old_coordination_checkpoints_are_rejected_for_inference(self):
        controller, _, _ = self.controller_state()
        for version in (1, 2):
            checkpoint = make_checkpoint(controller.model, "L", 0, 6, "unused", [], 0)
            checkpoint["coordination_version"] = version
            with patch("concon_v1.co1_learn_defender_retake.torch.load", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "coordination_version"):
                    ConconDefenderRetakeController("L")

    def test_legacy_inference_still_rejects_wrong_observations_and_unknown_versions(self):
        controller, _, _ = self.controller_state()
        checkpoint = make_checkpoint(controller.model, "L", 0, 6, "unused", [], 0)
        checkpoint["coordination_version"] = 1
        checkpoint["obs_dim"] -= 1
        with self.assertRaisesRegex(ValueError, "obs_dim"):
            validate_checkpoint(checkpoint, controller.scenario, 6, allow_legacy_coordination=True)
        checkpoint["obs_dim"] += 1
        checkpoint["coordination_version"] = 4
        with self.assertRaisesRegex(ValueError, "coordination_version"):
            validate_checkpoint(checkpoint, controller.scenario, 6, allow_legacy_coordination=True)

    def test_five_defenders_assemble_and_defuse_in_production_iq_engine(self):
        import contextlib
        import io
        from concon_v1.co1_defender_retake_training import DefenderRetakeEnv
        from concon_v1.co1_defender_scenario import get_scenario as search_scenario
        from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN
        from concon_v1.co1_retake_start_positions import START_CELLS

        for site in ("L", "R"):
            models = {s: fixture_model(get_scenario(s)) for s in ("L", "R")}
            env = DefenderRetakeEnv(models, DefenderSearchBattleDQN(search_scenario()), opponents=["omoko_v1"])
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset(opponent="omoko_v1")
            env.game.defender_setup_phase.finish()
            env.game.is_planted = True
            env.game.planted_pos = (7, 3) if site == "L" else (7, 38)
            env.game.spike_pos = None
            env.game.detonate_timer = 55
            for char in env.attackers:
                char.is_alive, char.hp = False, 0
            for char, slot in zip(env.defenders, "abcde"):
                char.pos = list(START_CELLS[slot][0])
            for _ in range(55):
                env.step()
                if env.done:
                    break
            self.assertTrue(env.metrics["waiting_decisions"] > 0)
            self.assertTrue(env.retakes[site].assembly.launched)
            self.assertTrue(env.game.is_defused, (site, env.end_reason, [c.pos for c in env.defenders]))


if __name__ == "__main__":
    unittest.main()
