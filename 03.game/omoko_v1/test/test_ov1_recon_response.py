"""Smoke/recon disclosure and enemy recon response at the real game boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "omoko_v1"))
sys.path.insert(0, str(ROOT / "test"))
from ov1_inference_recon_response import recon_response
from ov1_inference_team_tactics import shot_clear
from ov1_learning_attacker_escort import Ov1LearningAttackerEscortController
from ov1_attacker_controller import Ov1AttackerController
from ov1_defender_controller import Ov1DefenderController
from ov1_inference_damage_avoidance import escape_destruction_area
from public_effects import PublicEffectReader
from game_core import DEFUSE_REQUIRED_TICKS, SHOOT_INTERVAL_TICKS
from test_ultimate_system import UltimateTestGame, make_character


class ReconResponseTests(unittest.TestCase):
    def fixture(self):
        actor = NS(name="actor", team="A", pos=[4, 1], facing="E", is_alive=True,
                   reveal_remaining=0, sees_through_smoke=False)
        enemy = NS(name="enemy", team="D", pos=[4, 8], facing="W", is_alive=True,
                   position_known=True, reveal_remaining=0)
        state = dict(grid=np.zeros((9, 12), dtype=int), chars=[actor, enemy],
                     smoke_cells={(4, 3)}, public_effects=(), is_planted=False)
        return actor, enemy, state

    def nearest(self, actor, state):
        ctrl = Ov1LearningAttackerEscortController.__new__(Ov1LearningAttackerEscortController)
        return ctrl._nearest_visible_enemy(state["grid"], state["chars"], actor.team,
                                          tuple(actor.pos), game_state=state, viewer=actor)[0]

    def test_shared_position_behind_smoke_is_not_direct_sighting(self):
        actor, enemy, state = self.fixture()
        self.assertIsNone(self.nearest(actor, state))
        self.assertIsNone(recon_response(actor, state, {}))
        enemy.reveal_remaining = 3
        self.assertIs(self.nearest(actor, state), enemy)
        self.assertEqual(recon_response(actor, state, {}), ([4, 1], {"facing": "E"}))

    def test_unknown_enemy_and_wall_remain_hidden_after_recon(self):
        actor, enemy, state = self.fixture()
        enemy.reveal_remaining = 3
        enemy.position_known = False
        self.assertIsNone(self.nearest(actor, state))
        enemy.position_known = True
        state["grid"][4, 3] = 1
        self.assertIsNone(self.nearest(actor, state))
        self.assertIsNone(recon_response(actor, state, {}))

    def test_smoke_vision_and_adjacent_range_follow_engine_rules(self):
        actor, enemy, state = self.fixture()
        actor.sees_through_smoke = True
        self.assertIs(self.nearest(actor, state), enemy)
        actor.sees_through_smoke = False
        enemy.pos = [4, 2]
        state["smoke_cells"] = {(4, 1)}
        self.assertIs(self.nearest(actor, state), enemy)

    def test_shared_warning_does_not_become_a_direct_sighting_or_release_the_hold(self):
        actor, enemy, state = self.fixture()
        actor.ability_name = "FLASH"
        ctrl = Ov1LearningAttackerEscortController.__new__(Ov1LearningAttackerEscortController)
        ctrl._char_state = {}
        ctrl.greedy, ctrl.device = True, torch.device("cpu")
        ctrl._maybe_advance_tick = Mock()
        ctrl._refresh_watch_assignments = Mock()
        ctrl._omoko_escape_step = Mock(return_value=None)
        ctrl._resolve_carry_and_goal = Mock(return_value=((5, 1), (1, 5)))
        ctrl._opportunistic_orb_target = Mock(return_value=None)
        import ov1_learning_attacker_escort as module
        ctrl._build_obs = Mock(return_value=np.zeros(module.OBS_DIM, dtype=np.float32))
        ctrl._action_mask = Mock(return_value=np.ones(module.N_ACTIONS, dtype=bool))
        q = torch.zeros((1, module.N_ACTIONS))
        q[0, module.ACTION_UP * len(module.FACING_DIRS)] = 10
        ctrl.policy_net = Mock(return_value=q)
        self.assertIsNone(self.nearest(actor, state))
        self.assertEqual(ctrl.decide_move(actor, state)[0], actor.pos)
        enemy.position_known = False
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 1])

    def test_own_recon_seeks_wall_cover_even_inside_smoke(self):
        actor, enemy, state = self.fixture()
        actor.reveal_remaining = 3
        state["grid"][3, 2] = 1
        self.assertTrue(shot_clear(enemy, actor, state))
        result = recon_response(actor, state, {})
        self.assertEqual(result[0], [3, 1])
        actor.pos = result[0]
        self.assertFalse(shot_clear(enemy, actor, state))

    def test_defender_recon_preserves_policy_actions_in_search_and_retake(self):
        # Test both public and hidden threats, with and without a clear shot.
        # A prior retreat memory must never keep defense parked behind cover.
        for planted in (False, True):
            for visible in (False, True):
                for wall in (False, True):
                    for action in ("FLASH", "RECON", "SMOKE", "move", "DEFUSE"):
                        with self.subTest(planted=planted, visible=visible,
                                          wall=wall, action=action):
                            actor, enemy, state = self.fixture()
                            actor.team, enemy.team = "D", "A"
                            actor.reveal_remaining = 10
                            enemy.position_known = visible
                            state.update(is_planted=planted, planted_pos=(4, 2),
                                         detonate_timer=40)
                            if wall:
                                state["grid"][4, 3] = 1
                            memory = {(1, actor.name): True, "unknown_previous": (3, 1)}
                            self.assertIsNone(recon_response(actor, state, memory))
                            self.assertEqual(memory, {})
                            controller = Ov1DefenderController.__new__(Ov1DefenderController)
                            controller.search_controller = Mock()
                            controller.retake_controller = Mock()
                            phase = (controller.retake_controller if planted
                                     else controller.search_controller)
                            other = (controller.search_controller if planted
                                     else controller.retake_controller)
                            if action == "move":
                                decision = ([3, 1], {"facing": "N"})
                            elif action == "DEFUSE":
                                decision = ([4, 1], "DEFUSE")
                            else:
                                decision = ([4, 1], {"ability": action, "target": (4, 8)})
                            phase.decide_move.return_value = decision
                            result = controller.decide_move(actor, state)
                            phase.decide_move.assert_called_once_with(actor, state)
                            other.decide_move.assert_not_called()
                            self.assertEqual(result[0], decision[0])
                            if action in ("FLASH", "RECON", "SMOKE"):
                                self.assertEqual(result[1]["ability"], action)
                                self.assertEqual(result[1]["target"], (4, 8))
                            elif action == "DEFUSE":
                                self.assertEqual(result[1], "DEFUSE")

    def test_defender_recon_does_not_interrupt_active_defuse(self):
        actor, enemy, state = self.fixture()
        actor.team, enemy.team = "D", "A"
        actor.reveal_remaining, actor.defuse_timer = 10, 2
        state.update(is_planted=True, planted_pos=(4, 2), detonate_timer=40,
                     defender_defuse_info={actor.name: (2, DEFUSE_REQUIRED_TICKS)})
        controller = Ov1DefenderController.__new__(Ov1DefenderController)
        controller.retake_controller = Mock()
        controller.retake_controller.decide_move.return_value = ([4, 1], "DEFUSE")
        self.assertEqual(controller.decide_move(actor, state), ([4, 1], "DEFUSE"))
        controller.retake_controller.decide_move.assert_called_once_with(actor, state)

    def test_revealed_defender_executes_utility_in_real_engine(self):
        for name, ability, effect in (
            ("いぬさん", "FLASH", "flash_projectiles"),
            ("とりさん", "RECON", "recon_projectiles"),
            ("ねこさん", "SMOKE", "smokes"),
        ):
            with self.subTest(ability=ability):
                game = UltimateTestGame()
                actor = make_character(name, "D", (4, 1))
                enemy = make_character("Jean", "A", (4, 8))
                actor.reveal_remaining = 10
                game.chars = [actor, enemy]
                game.is_planted, game.planted_pos = True, (4, 8)
                controller = Ov1DefenderController.__new__(Ov1DefenderController)
                controller.retake_controller = Mock()
                controller.retake_controller.decide_move.return_value = (
                    [4, 1], {"ability": ability, "target": (4, 8)})
                game.defender_controller = controller
                charge_attr = ability.lower() + "_charges"
                before = getattr(actor, charge_attr)
                game.move_character(actor)
                self.assertEqual(getattr(actor, charge_attr), before - 1)
                self.assertTrue(getattr(game, effect))

    def test_hidden_shooter_is_not_read_to_choose_evasion(self):
        actor, enemy, state = self.fixture()
        actor.reveal_remaining = 3
        enemy.position_known = False
        first = recon_response(actor, state, {})
        enemy.pos = [3, 1]  # Hidden occupancy must not alter the chosen step.
        self.assertEqual(first, recon_response(actor, state, {}))
        self.assertNotEqual(first[0], actor.pos)

    def test_attacker_phases_prioritize_own_recon_escape(self):
        for phase in ("carry", "escort", "retrieve", "guard"):
            with self.subTest(phase=phase):
                actor, enemy, state = self.fixture()
                enemy.reveal_remaining = actor.reveal_remaining = 3
                actor.has_spike = phase == "carry"
                state["grid"][3, 2] = 1
                state["is_planted"] = phase == "guard"
                ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
                for role in ("carry", "escort", "retrieve", "guard", "search", "retake"):
                    setattr(ctrl, role + "_controller", Mock())
                result = ctrl.decide_move(actor, state)
                self.assertEqual(result[0], [3, 1])
                getattr(ctrl, phase + "_controller").decide_move.assert_not_called()

    def test_enemy_recon_does_not_cancel_carry_plant_or_postplant_phase(self):
        for phase in ("carry", "guard", "retake"):
            with self.subTest(phase=phase):
                actor, enemy, state = self.fixture()
                actor.has_spike = phase == "carry"
                enemy.reveal_remaining = 3
                state["is_planted"] = phase != "carry"
                self.assertIsNone(recon_response(actor, state, {}))

    def test_real_engine_fires_through_smoke_at_recon_target(self):
        game = UltimateTestGame()
        actor = make_character("Jean", "A", (4, 1))
        enemy = make_character("Lisa", "D", (4, 8))
        actor.facing, enemy.facing = "E", "E"
        enemy.reveal_remaining = 3
        game.chars = [actor, enemy]
        game.smokes = [{"cells": {(4, 3)}, "remaining_ticks": 5}]
        game.attacker_controller = NS(decide_move=lambda c, state: recon_response(c, state, {}))
        game.move_character(actor)
        game.battle_tick = SHOOT_INTERVAL_TICKS
        from unittest.mock import patch
        with patch("battle_logic.random.random", return_value=1.0):
            game._resolve_all_shots()
        self.assertTrue(any(s["shooter"] is actor and s["target"] is enemy for s in game.last_shots))

    def test_actual_ash_landing_is_a_danger_area_not_smoke(self):
        game = UltimateTestGame()
        actor = make_character("Jean", "A", (4, 4))
        enemy = make_character("Arlecchino", "D", (4, 8))
        game.chars = [actor, enemy]
        self.assertTrue(game.execute_ai_ability(enemy, {"ability": "ASH", "target": (4, 4)}))
        while game.ash_projectiles:
            game._advance_ash_projectiles()
        state = dict(grid=game.grid, chars=game.chars, public_effects=PublicEffectReader().read(game))
        self.assertFalse(game._smoke_cells())
        self.assertTrue(any(e.kind == "DESTRUCTION" for e in state["public_effects"]))
        self.assertNotEqual(escape_destruction_area(actor, state)[0], actor.pos)


if __name__ == "__main__":
    unittest.main()
