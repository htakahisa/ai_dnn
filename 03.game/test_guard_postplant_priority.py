import unittest
from types import SimpleNamespace as NS
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
from gc_v1.learning_attacker_guard_gc import LearningAttackerGuardGCController
import gc_v1.learning_attacker_guard_gc as guard
from abilities_los import AbilityLosMixin


class GuardPostplantPriorityTests(unittest.TestCase):
    def smoke_defuse_fixture(self):
        from unittest.mock import Mock
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController)
        controller.verbose = False
        controller.positioning_version = 5
        for method in ("_ensure_spike_dist_map", "_ensure_guard_assignment",
                       "_maybe_advance_tick", "_update_sighting_dist_map"):
            setattr(controller, method, Mock())
        controller._build_observation = Mock(side_effect=AssertionError(
            "A model hold/utility action must not override a smoke defuse"))
        char = NS(name="p", team="A", pos=[3, 0], is_alive=True)
        state = {"grid": np.zeros((7, 7), dtype=int), "chars": [char],
                 "is_planted": True, "planted_pos": [3, 4],
                 "smoke_cells": {(3, 3), (3, 4), (3, 5)},
                 "detonate_timer": 30, "defender_defuse_info": {"enemy": (1, 6)}}
        return controller, char, state

    def test_hidden_defuse_pushes_into_smoke_without_enemy_in_perceived_chars(self):
        controller, char, state = self.smoke_defuse_fixture()
        for version in (0, 1, 2, 5):
            controller.positioning_version = version
            char.pos = [3, 0]
            for expected_col in (1, 2, 3, 4):
                position, request = controller.decide_move(char, state)
                self.assertEqual(position, [3, expected_col])
                self.assertEqual(request["facing"], "E")
                char.pos = position

    def test_visible_defuser_inside_smoke_is_faced_instead_of_chasing(self):
        controller, char, state = self.smoke_defuse_fixture()
        char.pos = [3, 3]
        state["chars"].append(NS(name="enemy", team="D", pos=[3, 4], is_alive=True))
        self.assertEqual(controller.decide_move(char, state), ([3, 3], {"facing": "E"}))

    def test_center_without_visible_defuser_returns_valid_engine_action(self):
        controller, char, state = self.smoke_defuse_fixture()
        char.pos = [3, 4]
        self.assertEqual(controller.decide_move(char, state), [3, 4])

    def test_engine_executes_smoke_push_and_can_shoot_defuser(self):
        from test_ultimate_system import UltimateTestGame, make_character
        controller, _, _ = self.smoke_defuse_fixture()
        game = UltimateTestGame(height=7, width=7)
        char = make_character("Xdll", "A", (3, 0))
        enemy = make_character("Demon1", "D", (3, 4))
        enemy.defuse_timer = 1
        game.chars = [char, enemy]
        game.is_planted = True
        game.planted_pos = (3, 4)
        game.smokes = [{"cells": {(3, 3), (3, 4), (3, 5)}}]
        game.attacker_controller = controller
        for expected_col in (1, 2, 3):
            game.move_character(char)
            self.assertEqual(char.pos, [3, expected_col])
        game.move_character(char)
        self.assertEqual(char.pos, [3, 3])
        self.assertEqual(char.facing, "E")
        self.assertTrue(game.check_shot_line_of_sight(char, enemy))
        self.assertEqual(game._facing_angle_diff(char, enemy), 0)

    def test_recon_and_smoke_vision_allow_interrupting_without_rushing(self):
        for smoke_vision, recon in ((True, 0), (False, 2)):
            controller, char, state = self.smoke_defuse_fixture()
            char.sees_through_smoke = smoke_vision
            state["chars"].append(NS(name="enemy", team="D", pos=[3, 4],
                                     is_alive=True, reveal_remaining=recon))
            self.assertEqual(controller.decide_move(char, state),
                             ([3, 0], {"facing": "E"}))

    def test_smoke_interrupt_routes_around_blocking_teammates_and_walls(self):
        controller, char, state = self.smoke_defuse_fixture()
        state["chars"].append(NS(name="ally", team="A", pos=[3, 1], is_alive=True))
        state["grid"][2, 0] = 1
        self.assertEqual(controller.decide_move(char, state)[0], [4, 0])

    def test_occupied_spike_uses_free_defuse_zone_cell(self):
        controller, char, state = self.smoke_defuse_fixture()
        state["chars"].append(NS(name="ally", team="A", pos=[3, 4], is_alive=True))
        self.assertEqual(controller.decide_move(char, state)[0], [3, 1])

    def test_blocked_shot_pushes_instead_of_holding(self):
        controller, char, state = self.smoke_defuse_fixture()
        char.sees_through_smoke = True
        state["chars"].extend([
            NS(name="enemy", team="D", pos=[3, 4], is_alive=True),
            NS(name="ally", team="A", pos=[3, 2], is_alive=True)])
        self.assertEqual(controller.decide_move(char, state)[0], [2, 0])

    def test_no_active_tap_or_irrelevant_smoke_does_not_force_push(self):
        controller, char, state = self.smoke_defuse_fixture()
        self.assertIsNone(controller._smoke_defuse_interrupt(char, state, None))
        state["defender_defuse_info"] = {"enemy": (0, 6)}
        self.assertIsNone(controller._active_defuse_info(state))
        state["smoke_cells"] = {(0, 0)}
        self.assertIsNone(controller._smoke_defuse_interrupt(
            char, state, {"name": "enemy"}))

    def test_iq_perception_preserves_public_defuse_notification(self):
        from iq_perception import IQPerceptionEngine
        controller, char, state = self.smoke_defuse_fixture()
        view = NS(grid=state["grid"], chars=[char], spike_pos=None,
                  planted_pos=state["planted_pos"], target_plant_pos=None,
                  detonate_timer=30, is_planted=True)
        perceived = IQPerceptionEngine().build_perceived_state(
            viewer=char, game_state=state, game_view=view)
        self.assertEqual(perceived["defender_defuse_info"], {"enemy": (1, 6)})
        self.assertEqual(controller.decide_move(char, perceived)[0], [3, 1])

    def test_legacy_wrapper_does_not_replace_interrupt_with_another_fight(self):
        from unittest.mock import Mock, patch
        import ghost_champions_v1_macro as wrapper
        ctrl = wrapper.GhostChampionsV1AttackerController.__new__(
            wrapper.GhostChampionsV1AttackerController)
        ctrl.macro_controller = object()
        ctrl.guard = NS(positioning_version=0)
        ctrl._micro_cover_result = Mock(side_effect=AssertionError("cover override"))
        result = ([3, 1], {"facing": "E"})
        with patch.object(wrapper._BaseGCAttacker, "decide_move", return_value=result):
            self.assertIs(ctrl.decide_move(NS(), {
                "is_planted": True, "defender_defuse_info": {"enemy": (1, 6)}}), result)

    def test_postplant_los_matches_engine_smoke_rules(self):
        from gc_v1.learning_defender_retake_gc import _has_los as retake_los
        grid = np.zeros((7, 7), dtype=int)
        engine = AbilityLosMixin()
        engine.grid = grid
        engine.smokes = [{"cells": {(3, 3)}}]
        for start, end in [((3, 1), (3, 5)), ((3, 3), (3, 4)),
                           ((3, 3), (3, 5)), ((1, 1), (1, 5))]:
            expected = engine.check_cell_line_of_sight(start, end)
            for los in (guard._has_los, retake_los):
                self.assertEqual(los(grid, start, end, {(3, 3)}), expected)

    def test_active_defuse_releases_spike_watch_hold(self):
        import torch
        from unittest.mock import Mock
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController)
        controller.verbose = False
        controller._ensure_spike_dist_map = Mock()
        controller._ensure_guard_assignment = Mock()
        controller._maybe_advance_tick = Mock()
        controller._update_sighting_dist_map = Mock()
        controller._build_observation = Mock(return_value=(np.zeros(34, dtype=np.float32), []))
        controller._action_mask = Mock(return_value=np.ones(10, dtype=bool))
        controller.model = lambda x: torch.tensor([[0., 0., 10., 0., 0., 0., 0., 0., 0., 0.]], device=x.device)
        char = NS(name="p", pos=[3, 3], is_alive=True)
        state = {"grid": np.zeros((7, 7), dtype=int), "chars": [char],
                 "is_planted": True, "planted_pos": [3, 4],
                 "detonate_timer": 30, "defender_defuse_info": {"enemy": (2, 6)}}
        self.assertEqual(controller.decide_move(char, state), [2, 3])

    def test_guard_route_precedes_spike_watch_until_arrival(self):
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController
        )
        controller._assigned_guard_positions = {"p": (2, 4)}
        distances = np.array([
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [8, 7, 6, 5, 4],
            [9, 8, 7, 6, 5],
        ], dtype=np.int32)
        controller._assigned_dist_maps = {"p": distances}
        char = NS(name="p", pos=[2, 1], is_alive=True)
        step = controller._guard_position_step(char, np.zeros((5, 5), dtype=int), [char])
        self.assertEqual(step, [2, 2])

    def test_spike_watch_can_resume_after_guard_arrival(self):
        controller = LearningAttackerGuardGCController.__new__(
            LearningAttackerGuardGCController
        )
        controller._assigned_guard_positions = {"p": (2, 2)}
        controller._assigned_dist_maps = {"p": np.zeros((5, 5), dtype=np.int32)}
        char = NS(name="p", pos=[2, 2], is_alive=True)
        self.assertIsNone(controller._guard_position_step(
            char, np.zeros((5, 5), dtype=int), [char]
        ))


if __name__ == "__main__":
    unittest.main()
