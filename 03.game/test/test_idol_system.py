"""Idol healing, death-only ultimates, and manual selection regressions."""

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import game_core
from controllers import DefaultAttackerController, DefaultDefenderController, UserInputController
from gc_v1.ultimate_tactics_gc import build_ultimate_action
from rendering_ui import RenderingUIMixin
from tactical_simulator import TacticalSimulator, create_sample_retake_scenario, get_character_resource_profile
from test_ultimate_system import UltimateTestGame, FixedController, make_character


class IdolTestGame(UltimateTestGame, RenderingUIMixin):
    pass


class IdolSystemTests(unittest.TestCase):
    def setUp(self):
        table = game_core._character_stats.CHARACTER_TABLE
        role_patch = patch.dict(table, {"Chronicle": replace(table["Chronicle"], role="アイドル")})
        role_patch.start()
        self.addCleanup(role_patch.stop)
        self.game = IdolTestGame()
        self.owner = make_character("Chronicle", "A", (2, 2), 99)
        self.ally = make_character("Leo", "A", (7, 2))
        self.enemy = make_character("Demon1", "D", (7, 8))
        self.game.chars = [self.owner, self.ally, self.enemy]

    def dance(self, **target):
        return self.game.execute_ai_ability(self.owner, {"ability": "DANCE", **target})

    def test_idol_resources_and_editor_profile(self):
        self.assertEqual((self.owner.ability_name, self.owner.dance_charges), ("DANCE", 3))
        self.assertEqual((self.owner.ultimate_name, self.owner.ultimate_cost, self.owner.ultimate_points),
                         ("SERENADE", 4, 4))
        self.assertEqual(get_character_resource_profile(self.owner.name),
                         {"ability": "DANCE", "max_charges": 3, "ultimate": "SERENADE", "ultimate_cost": 4})

    def test_dance_heals_fifty_and_can_use_all_three_charges(self):
        for remaining in (2, 1, 0):
            self.ally.hp = 20
            self.assertTrue(self.dance(target_name=self.ally.name))
            self.assertEqual((self.ally.hp, self.owner.dance_charges), (70, remaining))
        self.ally.hp = 20
        self.assertFalse(self.dance(target_name=self.ally.name))
        self.assertEqual(self.ally.hp, 20)

    def test_dance_cap_is_always_one_hundred(self):
        self.ally.hp, self.ally.max_hp = 90, 150
        self.assertTrue(self.dance(target_name=self.ally.name))
        self.assertEqual(self.ally.hp, 100)
        self.ally.hp, self.ally.max_hp = 80, 80
        self.assertTrue(self.dance(target_name=self.ally.name))
        self.assertEqual(self.ally.hp, 100)
        self.ally.hp = 150
        self.assertFalse(self.dance(target_name=self.ally.name))
        self.assertEqual((self.ally.hp, self.owner.dance_charges), (150, 1))

    def test_dance_rejects_dead_enemy_self_and_unknown_targets_without_spending(self):
        self.ally.is_alive = False
        for name in (self.ally.name, self.enemy.name, self.owner.name, "missing"):
            self.assertFalse(self.dance(target_name=name))
            self.assertEqual(self.owner.dance_charges, 3)
        self.ally.is_alive, self.ally.hp = True, 30
        self.owner.is_alive = False
        self.assertFalse(self.dance(target_name=self.ally.name))

    def test_dance_selects_by_cell_and_has_no_range_or_los_requirement(self):
        self.ally.hp = 1
        self.game.grid[4, :] = 1
        self.assertTrue(self.dance(target=tuple(self.ally.pos)))
        self.assertEqual(self.ally.hp, 51)

    def test_serenade_is_death_only_and_requires_four_points(self):
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "SERENADE"}))
        self.assertEqual(self.owner.ultimate_points, 4)
        self.owner.is_alive, self.owner.ultimate_points = False, 3
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "SERENADE"}))
        self.owner.ultimate_points = 4
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, {"ultimate": "SERENADE"}))
        self.assertEqual(self.owner.ultimate_points, 0)
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "SERENADE"}))

    def test_serenade_cannot_activate_after_either_team_is_wiped(self):
        self.owner.is_alive = False
        for ally_alive, enemy_alive in ((False, True), (True, False)):
            with self.subTest(ally_alive=ally_alive, enemy_alive=enemy_alive):
                self.ally.is_alive = ally_alive
                self.enemy.is_alive = enemy_alive
                self.owner.ultimate_points = 4
                self.enemy.reveal_remaining = 0
                self.assertFalse(self.game._selected_ultimate_ready(self.owner))
                self.assertFalse(self.game.execute_ai_ultimate(
                    self.owner, {"ultimate": "SERENADE"}))
                self.game._activate_ai_death_ultimates()
                self.assertEqual((self.owner.ultimate_points, self.enemy.reveal_remaining), (4, 0))

    def test_serenade_reveals_every_live_enemy_across_walls_and_preserves_longer_reveals(self):
        far = make_character("Derke", "D", (1, 10))
        dead = make_character("something", "D", (1, 8))
        dead.is_alive = False
        far.reveal_remaining = 9
        self.game.chars.extend([far, dead])
        self.game.grid[:, 5] = 1
        self.owner.is_alive = False
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, {"ultimate": "SERENADE"}))
        self.assertEqual((self.enemy.reveal_remaining, far.reveal_remaining, dead.reveal_remaining,
                          self.ally.reveal_remaining), (3, 9, 0, 0))
        self.assertTrue(self.game.is_visible_to_team(self.enemy, "A"))

    def test_dead_other_roles_cannot_cast(self):
        self.enemy.is_alive = False
        self.enemy.ultimate_points = self.enemy.ultimate_cost
        self.assertFalse(self.game.execute_ai_ultimate(self.enemy, {"ultimate": self.enemy.ultimate_name,
                                                                   "target": (1, 1)}))

    def test_ai_casts_on_death_while_manual_players_keep_their_points(self):
        self.game._kill_character(self.enemy, self.owner)
        self.assertEqual((self.owner.ultimate_points, self.enemy.reveal_remaining), (0, 3))
        self.owner.is_alive, self.owner.ultimate_points = True, 4
        self.enemy.reveal_remaining = 0
        self.game.attacker_controller = UserInputController()
        self.game._kill_character(self.enemy, self.owner)
        self.assertEqual((self.owner.ultimate_points, self.enemy.reveal_remaining), (4, 0))

    def test_dead_idol_can_be_selected_from_team_panel_and_cast(self):
        self.game.attacker_controller = UserInputController()
        self.owner.is_alive = False
        self.game.map_offset_x, self.game.map_pixel_width = 260, 240
        self.game.map_pixel_height, self.game.cell_size = 180, 20
        self.game.ability_mode = self.game.ultimate_mode = None
        self.assertTrue(self.game._handle_team_panel_click(20, 60))
        self.assertIs(self.game._selected_user_character(), self.owner)
        self.assertIsNotNone(self.game._ultimate_button_bounds())
        self.assertTrue(self.game._selected_ultimate_ready(self.owner))
        x1, y1, x2, y2 = self.game._ultimate_button_bounds()
        self.game.on_canvas_click(SimpleNamespace(x=(x1+x2)/2, y=(y1+y2)/2+game_core.COMBO_BANNER_HEIGHT))
        self.assertEqual((self.owner.ultimate_points, self.enemy.reveal_remaining), (0, 3))

    def test_dance_target_can_be_selected_from_panel_and_map(self):
        self.game.attacker_controller = UserInputController()
        self.game.map_offset_x, self.game.map_pixel_width = 260, 240
        self.game.map_pixel_height, self.game.cell_size = 180, 20
        self.game.ability_mode = self.game.ultimate_mode = None
        self.game._set_active_user_character("A", self.owner.name)
        self.ally.hp = 20
        self.game.ability_mode = ("DANCE", "A", self.owner.name)
        self.game._handle_team_panel_click(20, 160)
        self.assertEqual(self.ally.hp, 70)
        self.assertIs(self.game._selected_user_character(), self.owner)
        self.game.ability_mode = ("DANCE", "A", self.owner.name)
        self.game.on_canvas_click(SimpleNamespace(x=310, y=150+game_core.COMBO_BANNER_HEIGHT))
        self.assertEqual((self.ally.hp, self.owner.dance_charges), (100, 1))

    def test_default_ai_heals_injured_allies_and_dead_ultimate_payload_is_executable(self):
        self.ally.hp = 20
        state = {"grid": self.game.grid, "chars": self.game.chars}
        for controller in (DefaultAttackerController(), DefaultDefenderController()):
            self.assertEqual(controller._decide_ability(self.owner, state),
                             {"ability": "DANCE", "target_name": self.ally.name})
        self.assertIsNone(build_ultimate_action(self.game.grid, self.owner, self.game.chars))
        self.owner.is_alive = False
        self.assertEqual(build_ultimate_action(self.game.grid, self.owner, self.game.chars),
                         {"ultimate": "SERENADE"})

    def test_serenade_lasts_three_battle_ticks_and_is_saved_in_replay(self):
        scenario = create_sample_retake_scenario()
        scenario.attackers = [{"name": "Chronicle", "pos": (7, 3), "ultimate_points": 4},
                              {"name": "Leo", "pos": (8, 3)}]
        scenario.defenders = [{"name": "Demon1", "pos": (7, 8)}]
        scenario.initial_smokes = []
        scenario.planted_pos = (7, 4)
        with redirect_stdout(StringIO()):
            simulator = TacticalSimulator(scenario, attacker_ai_name="user", defender_ai_name="default")
            simulator.defender_controller = FixedController("MOVE")
            simulator.grid[:, 5] = 1
            for char in simulator.chars:
                char.accuracy = 0
            idol, _, enemy = simulator.chars
            idol.is_alive = False
            self.assertTrue(simulator.execute_ai_ultimate(idol, {"ultimate": "SERENADE"}))
            for remaining in (3, 2, 1, 0):
                simulator.step({})
                self.assertEqual(enemy.reveal_remaining, remaining)
            simulator._record_replay_frame()
        frame = simulator.replay_frames[1]
        self.assertEqual(frame["chars"][0]["ability"], "DANCE")
        self.assertEqual(frame["chars"][0]["ability_charges"], 3)
        self.assertIn("A", frame["chars"][2]["visible_to"])


if __name__ == "__main__":
    unittest.main()
