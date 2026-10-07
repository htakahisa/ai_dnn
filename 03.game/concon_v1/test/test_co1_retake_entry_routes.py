"""Paired assembly/entry goals, route persistence and foundation supervision."""

import unittest
import numpy as np
import torch

from concon_v1.co1_retake_scenarios import get_scenario, build_scenario
from concon_v1.co1_retake_common import build_inputs, GORIGONS, MOVES
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_retake_foundation import navigation_cells, training_targets
from concon_v1.test.test_co1_defender_retake import actor
from concon_v1.test.test_co1_retake_foundation import fixture_model


class EntryRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_route(self, site):
        scenario = get_scenario(site)
        controller = ConconDefenderRetakeController(site, model=fixture_model(scenario))
        chars = [actor(name, group[0]) for name, group in zip(GORIGONS.players, scenario.rally_groups)]
        controller.assembly.assigned = {char.name: tuple(char.pos) for char in chars}
        state = dict(grid=scenario.grid, chars=chars, planted_pos=(7, 3) if site == "L" else (7, 38),
                     smoke_cells=[], battle_tick=1, detonate_timer=100, defender_defuse_info={})
        return controller, chars, state

    def test_learned_movement_visits_each_assigned_entry_before_spike_goal(self):
        for site in ("L", "R"):
            controller, chars, state = self.setup_route(site)
            initial = dict(controller.assembly.assigned)
            for _ in range(50):
                for char in chars:
                    observation, mask, context = build_inputs(controller, char, state)
                    entries = controller.scenario.entries_for(initial[char.name])
                    if char.name in controller.assembly.entered:
                        self.assertEqual(context["goal"], state["planted_pos"])
                        continue
                    self.assertIn(context["goal"], entries)
                    active, values = controller.model.foundation_inputs(torch.as_tensor(observation)[None])
                    self.assertTrue(active.item())
                    action = controller.choose_action(char, observation, mask, context)
                    self.assertLess(action, 32)
                    dr, dc = MOVES[action // 8]
                    char.pos = [char.pos[0] + dr, char.pos[1] + dc]
                self.assertEqual(controller.assembly.assigned, initial)
                state["battle_tick"] += 1
                if len(controller.assembly.entered) == len(chars):
                    break
            self.assertEqual(controller.assembly.entered, {char.name for char in chars})
            # Returning away from the entry after combat does not restart it.
            for char in chars:
                char.pos = list(initial[char.name])
                self.assertEqual(build_inputs(controller, char, state)[2]["goal"], state["planted_pos"])

    def test_deadline_and_single_survivor_can_go_directly_to_spike(self):
        for site in ("L", "R"):
            controller, chars, state = self.setup_route(site)
            state["detonate_timer"] = 9
            for char in chars:
                self.assertEqual(build_inputs(controller, char, state)[2]["goal"], state["planted_pos"])
            controller.reset_round()
            state.update(chars=chars[:1], detonate_timer=100, battle_tick=2)
            self.assertEqual(build_inputs(controller, chars[0], state)[2]["goal"], state["planted_pos"])

    def test_entry_teacher_never_teaches_defuse_even_at_blurred_plant_coordinates(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            cells = navigation_cells(scenario)
            indices, targets = training_targets(scenario)
            entries = [point for _, points in scenario.entry_points for point in points]
            for index in range(len(cells) - len(entries), len(cells)):
                entry_rows = targets[indices // scenario.grid.size == index]
                self.assertTrue(len(entry_rows))
                self.assertTrue(np.all(entry_rows[:, 5] == -2))

    def test_missing_paired_entry_is_rejected(self):
        from concon_v1.co1_map_retake_L import MAZE_STR
        with self.assertRaisesRegex(ValueError, "paired a/A and b/B"):
            build_scenario("L", MAZE_STR.replace("B", "0"))


if __name__ == "__main__":
    unittest.main()
