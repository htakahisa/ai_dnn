"""Awakened carrier seeks cover without bypassing real movement or shooting."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "omoko_v1"))
sys.path.insert(0, str(ROOT / "test"))

from ov1_inference_carry_escape import awakened_carry_escape, ESCAPE_AWAKENING_NAME
from ov1_inference_team_tactics import shot_clear
from ov1_learning_attacker_carry import Ov1LearningAttackerCarryController
from iq_controller_adapter import IQAwareController
from public_effects import DisplayEffect
from game_core import SHOOT_INTERVAL_TICKS
from test_ultimate_system import UltimateTestGame, make_character


class CarryAwakenedEscapeTests(unittest.TestCase):
    def fixture(self):
        game = UltimateTestGame()
        actor = make_character("おもこ", "A", (4, 1))
        actor.has_spike = True
        actor.active_awakenings[ESCAPE_AWAKENING_NAME] = {}
        actor.move_steps_per_tick = 4
        enemy = make_character("Lisa", "D", (4, 8))
        enemy.los_revealed = True
        ally = make_character("Jean", "A", (6, 1))
        game.chars = [actor, enemy, ally]
        game.grid[3, 2] = 1
        state = {"grid": game.grid, "chars": game.chars, "is_planted": False,
                 "smoke_cells": (), "public_effects": (), "current_round": 1}
        return game, actor, enemy, state

    def test_carry_prefers_cover_over_forward_plant_route(self):
        _, actor, enemy, state = self.fixture()
        ctrl = Ov1LearningAttackerCarryController.__new__(Ov1LearningAttackerCarryController)
        # Retreat must be selected before the normal target/BFS/model branch.
        position, payload = ctrl.decide_move(actor, state)
        self.assertEqual(position, [3, 1])
        self.assertNotIn("ability", payload)
        actor.pos = position
        self.assertFalse(shot_clear(enemy, actor, state))

    def test_real_engine_and_iq_wrapper_stop_at_cover_without_overshoot(self):
        game, actor, enemy, _ = self.fixture()
        ctrl = Ov1LearningAttackerCarryController.__new__(Ov1LearningAttackerCarryController)
        controller = IQAwareController(ctrl, viewer_team="A")
        # No model/geometry setup is needed when testing the escape branch.
        with patch.object(ctrl, "set_game"):
            controller.set_game(game)
            game.attacker_controller = controller
            game.move_character(actor)
        self.assertEqual(actor.pos, [3, 1])
        self.assertEqual(actor._navigation_step_limit, 1)
        self.assertTrue(actor.moved_this_tick)
        self.assertFalse(game.check_shot_line_of_sight(enemy, actor))
        self.assertEqual(actor.plant_timer, 0)

    def test_cover_is_held_until_awakening_expires_even_after_sighting_is_lost(self):
        _, actor, enemy, state = self.fixture()
        memory = {}
        actor.pos = awakened_carry_escape(actor, state, memory)[0]
        self.assertEqual(awakened_carry_escape(actor, state, memory)[0], actor.pos)
        enemy.position_known = False
        self.assertEqual(awakened_carry_escape(actor, state, memory)[0], actor.pos)
        actor.active_awakenings.clear()
        self.assertIsNone(awakened_carry_escape(actor, state, memory))
        self.assertFalse(memory)

    def test_only_awakened_omoko_carrier_uses_escape(self):
        for change in ("inactive", "escort", "other_player", "postplant", "setup"):
            with self.subTest(change=change):
                _, actor, _, state = self.fixture()
                if change == "inactive":
                    actor.active_awakenings.clear()
                elif change == "escort":
                    actor.has_spike = False
                elif change == "other_player":
                    actor.base_name = "Jean"
                elif change == "postplant":
                    state["is_planted"] = True
                else:
                    state["defender_setup_active"] = True
                self.assertIsNone(awakened_carry_escape(actor, state))

    def test_unknown_enemy_does_not_change_escape(self):
        _, actor, enemy, state = self.fixture()
        enemy.position_known = False
        before = awakened_carry_escape(actor, state)
        enemy.pos = [0, 0]
        self.assertIsNone(before)
        self.assertIsNone(awakened_carry_escape(actor, state))

    def test_occupied_cover_and_public_damage_zone_are_not_entered(self):
        for blocked in ("ally", "zone"):
            with self.subTest(blocked=blocked):
                _, actor, _, state = self.fixture()
                if blocked == "ally":
                    state["chars"].append(make_character("Furina", "A", (3, 1)))
                else:
                    state["public_effects"] = (DisplayEffect(1, "DESTRUCTION", "active", (3, 1), cells=((3, 1),)),)
                self.assertNotEqual(awakened_carry_escape(actor, state)[0], [3, 1])

    def test_equally_safe_cover_prefers_nearby_teammate(self):
        _, actor, _, state = self.fixture()
        state["grid"][5, 2] = 1
        # North and south both cut the ray in one step; ally is to the south.
        self.assertEqual(awakened_carry_escape(actor, state)[0], [5, 1])

    def test_open_ground_fallback_moves_away_instead_of_toward_enemy(self):
        _, actor, enemy, state = self.fixture()
        state["grid"][:] = 0
        step, payload = awakened_carry_escape(actor, state)
        old_gap = sum(abs(a - b) for a, b in zip(actor.pos, enemy.pos))
        self.assertGreater(sum(abs(a - b) for a, b in zip(step, enemy.pos)), old_gap)
        self.assertIn("facing", payload)

    def test_trapped_carrier_stays_and_can_still_shoot(self):
        game, actor, enemy, state = self.fixture()
        state["grid"][:] = 1
        state["grid"][4, 1:9] = 0
        state["chars"].append(make_character("blocker", "A", (4, 0)))
        # The only walkable escape direction is toward the shooter.
        result = awakened_carry_escape(actor, state)
        self.assertEqual(result[0], actor.pos)
        self.assertEqual(result[1]["facing"], "E")
        game.attacker_controller = NS(decide_move=lambda char, state: result)
        game.move_character(actor)
        self.assertEqual(actor.plant_timer, 0)
        self.assertTrue(game.check_shot_line_of_sight(actor, enemy))
        game.battle_tick = SHOOT_INTERVAL_TICKS
        with patch("battle_logic.random.random", return_value=1.0):
            game._resolve_all_shots()
        self.assertTrue(any(shot["shooter"] is actor for shot in game.last_shots))


if __name__ == "__main__":
    unittest.main()
