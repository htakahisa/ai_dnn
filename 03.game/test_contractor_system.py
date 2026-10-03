"""Destruction areas, life contracts, and ability visual-effect regressions."""

import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import game_core
from ability_effects import (draw_heal_sparkle, draw_serenade_flash,
                            TUNNEL_WARNING_COLOR, TUNNEL_ACTIVE_COLOR, BALEMOON_WARNING_COLOR)
from analytics.combat_tracker import CombatTracker
from analytics.replay_viewer import ReplayViewer
from combo_awakening import _snapshot_character_awakening_state, _restore_character_awakening_state
from controllers import DefaultAttackerController, DefaultDefenderController
from gc_v1.ultimate_tactics_gc import build_ultimate_action
from run_game import VisualFPSBattle
from tactical_simulator import TacticalSimulator, create_sample_retake_scenario, get_character_resource_profile
from test_engineer_system import RecordingCanvas
from test_ultimate_system import UltimateTestGame, FixedController, make_character


class ContractorSystemTests(unittest.TestCase):
    def setUp(self):
        table = game_core._character_stats.CHARACTER_TABLE
        roles = patch.dict(table, {"Derke": replace(table["Derke"], role="コントラクター"),
                                   "Chronicle": replace(table["Chronicle"], role="アイドル")})
        roles.start()
        self.addCleanup(roles.stop)
        self.game = UltimateTestGame(height=12, width=20)
        self.owner = make_character("Derke", "A", (4, 2), 99)
        self.enemy = make_character("Demon1", "D", (4, 4))
        self.ally = make_character("Leo", "A", (4, 5))
        self.idol = make_character("Chronicle", "A", (10, 17))
        self.game.chars = [self.owner, self.enemy, self.ally, self.idol]
        self.game.destruction_areas = []
        self.game.ash_projectiles = []
        self.game.balemoon_warnings = []

    def ash(self, target=(4, 4), *, land=True):
        result = self.game.execute_ai_ability(self.owner, {"ability": "ASH", "target": target})
        if result and land:
            while self.game.ash_projectiles:
                self.game._advance_ash_projectiles()
        return result

    def finish_balemoon_warning(self):
        for _ in range(4):
            self.game._advance_balemoon_warnings()

    def advance(self):
        self.game.battle_tick += 1
        self.game._advance_visual_effects()
        self.game._advance_ash_projectiles()
        self.game._advance_balemoon_warnings()
        self.game._advance_destruction_areas()

    def test_contractor_resources_and_editor_profile(self):
        self.assertEqual((self.owner.ability_name, self.owner.ash_charges), ("ASH", 3))
        self.assertEqual((self.owner.ultimate_name, self.owner.ultimate_cost, self.owner.ultimate_points),
                         ("BALEMOON", 5, 5))
        self.assertEqual(get_character_resource_profile(self.owner.name),
                         {"ability": "ASH", "max_charges": 3, "ultimate": "BALEMOON", "ultimate_cost": 5})

    def test_ash_radius_uses_euclidean_distance_and_rejects_walls(self):
        self.assertTrue(self.ash((4, 10)))  # Exactly eight cells.
        self.assertFalse(self.ash((4, 11)))
        self.assertFalse(self.ash((10, 8)))  # Diagonal distance sqrt(72).
        self.game.grid[4, 9] = 1
        self.assertFalse(self.ash((4, 9)))
        self.assertEqual(self.owner.ash_charges, 2)

    def test_ash_has_three_charges_and_creates_three_by_three_level_five(self):
        for remaining in (2, 1, 0):
            self.assertTrue(self.ash())
            self.assertEqual(self.owner.ash_charges, remaining)
        self.assertFalse(self.ash())
        area = self.game.destruction_areas[0]
        self.assertEqual((len(area["cells"]), area["level"], area["remaining_ticks"]), (9, 5, 10))

    def test_ash_area_clips_at_map_edge_and_excludes_walls(self):
        self.owner.pos = [0, 0]
        self.game.grid[1, 1] = 1
        self.assertTrue(self.ash((0, 0)))
        self.assertEqual(self.game.destruction_areas[0]["cells"], {(0, 0), (0, 1), (1, 0)})

    def test_ash_flies_at_flash_speed_and_only_creates_area_on_landing(self):
        self.enemy.pos = [4, 10]
        self.assertTrue(self.ash((4, 10), land=False))
        self.assertEqual(self.game.destruction_areas, [])
        self.assertEqual(self.owner.ash_charges, 2)
        for progress in (3, 6):
            self.advance()
            self.assertEqual(self.game.ash_projectiles[0]["progress"], progress)
            self.assertEqual(self.game.destruction_areas, [])
            self.assertEqual(self.enemy.life_contract_remaining, 0)
        self.advance()
        self.assertEqual(self.game.ash_projectiles, [])
        self.assertEqual(self.game.destruction_areas[0]["pos"], (4, 10))
        self.assertEqual((self.enemy.hp, self.enemy.life_contract_remaining), (90, 4))

    def test_ash_lands_before_wall_and_never_continues_beyond_target(self):
        self.game.grid[4, 6] = 1
        self.assertTrue(self.ash((4, 10), land=False))
        self.assertEqual(self.game.ash_projectiles[0]["path"][-1], (4, 5))
        self.game._advance_ash_projectiles()
        self.assertEqual(self.game.destruction_areas[0]["pos"], (4, 5))

    def test_self_targeted_ash_still_waits_for_one_tick(self):
        self.assertTrue(self.ash(tuple(self.owner.pos), land=False))
        self.assertEqual(self.game.destruction_areas, [])
        self.game._advance_ash_projectiles()
        self.assertEqual(self.game.destruction_areas[0]["pos"], tuple(self.owner.pos))

    def test_contract_lasts_five_damage_ticks_after_leaving_area(self):
        self.assertTrue(self.ash())
        self.game._trigger_destruction_areas(self.enemy)
        self.assertEqual(self.enemy.life_contract_remaining, 5)
        self.enemy.pos = [11, 19]
        for tick in range(1, 6):
            self.advance()
            self.assertEqual((self.enemy.hp, self.enemy.max_hp), (100-10*tick, 100-10*tick))
            self.assertEqual(self.enemy.life_contract_remaining, 5-tick)
        self.advance()
        self.assertEqual((self.enemy.hp, self.enemy.max_hp), (50, 50))
        self.assertEqual((self.ally.hp, self.ally.max_hp), (100, 100))

    def test_overlapping_areas_do_not_stack_damage_or_refresh_active_contract(self):
        self.ash()
        self.ash()
        self.advance()
        self.assertEqual((self.enemy.hp, self.enemy.max_hp, self.enemy.life_contract_remaining), (90, 90, 4))
        self.game._trigger_destruction_areas(self.enemy)
        self.assertEqual(self.enemy.life_contract_remaining, 4)
        self.advance()
        self.assertEqual((self.enemy.hp, self.enemy.max_hp, self.enemy.life_contract_remaining), (80, 80, 3))

    def test_area_lasts_ten_ticks_and_expired_areas_cannot_apply_contract(self):
        self.enemy.pos = [11, 19]
        self.ash()
        for _ in range(10):
            self.advance()
        self.assertEqual(self.game.destruction_areas[0]["remaining_ticks"], 0)
        self.enemy.pos = [4, 4]
        self.game._trigger_destruction_areas(self.enemy)
        self.assertEqual(self.enemy.life_contract_remaining, 0)
        self.advance()
        self.assertEqual(self.game.destruction_areas, [])

    def test_contract_can_kill_and_credits_the_area_owner(self):
        self.enemy.hp = 5
        self.enemy.has_spike = True
        self.ash()
        self.advance()
        self.assertFalse(self.enemy.is_alive)
        self.assertEqual((self.enemy.hp, self.enemy.max_hp, self.enemy.deaths, self.owner.kills), (0, 90, 1, 1))
        self.assertEqual(self.game.spike_pos, (4, 4))

    def test_reduced_max_hp_survives_awakening_expiry(self):
        snapshot = _snapshot_character_awakening_state(self.enemy)
        self.enemy.max_hp = 150
        self.ash()
        self.advance()
        _restore_character_awakening_state(self.enemy, snapshot)
        self.assertEqual(self.enemy.max_hp, 90)

    def test_dance_respects_contracted_max_hp_and_starts_five_tick_sparkle(self):
        self.ally.hp, self.ally.max_hp, self.ally.contract_max_hp_lost = 20, 60, 40
        self.assertTrue(self.game.execute_ai_ability(self.idol, {"ability": "DANCE", "target_name": self.ally.name}))
        self.assertEqual(self.ally.hp, 60)
        self.assertEqual(self.ally.heal_sparkle_remaining, 5)
        for remaining in (5, 4, 3, 2, 1, 0):
            self.advance()
            self.assertEqual(self.ally.heal_sparkle_remaining, remaining)

    def test_sparkle_rendering_changes_with_tick_and_flash_covers_entire_map(self):
        canvas = RecordingCanvas()
        draw_heal_sparkle(canvas, 2, 3, 20, 5)
        first = canvas.calls[0][1]
        self.assertTrue(all(call[2]["fill"] == "#ffd6e7" for call in canvas.calls))
        self.assertEqual(canvas.calls[0][2]["outline"], "#ffd6e7")
        canvas.calls.clear()
        draw_heal_sparkle(canvas, 2, 3, 20, 4)
        self.assertNotEqual(first, canvas.calls[0][1])
        canvas.calls.clear()
        draw_serenade_flash(canvas, 20, 12, 20, 260)
        self.assertEqual(canvas.calls[0][1], (260, 0, 660, 240))
        self.assertEqual(canvas.calls[0][2]["fill"], "#ffd6e7")

    def test_serenade_flash_lasts_one_battle_tick(self):
        self.idol.is_alive, self.idol.ultimate_points = False, 4
        self.assertTrue(self.game.execute_ai_ultimate(self.idol, {"ultimate": "SERENADE"}))
        self.assertEqual(self.game.serenade_flash_remaining, 1)
        self.advance()
        self.assertEqual(self.game.serenade_flash_remaining, 1)
        self.advance()
        self.assertEqual(self.game.serenade_flash_remaining, 0)

    def test_balemoon_heals_to_current_max_and_kills_all_idols_globally(self):
        enemy_idol = make_character("Chronicle", "D", (0, 19))
        enemy_idol.name = "enemy_idol"
        enemy_idol.iron_will_charges = self.idol.iron_will_charges = 1
        self.game.chars.append(enemy_idol)
        tracker = CombatTracker()
        tracker.register_players(self.game.chars)
        self.game.analytics_tracker = tracker
        self.owner.hp, self.owner.max_hp = 20, 70
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"}))
        self.assertEqual(self.owner.hp, 20)
        self.assertTrue(self.idol.is_alive)
        self.assertTrue(enemy_idol.is_alive)
        self.assertEqual(self.owner.ultimate_points, 0)
        self.finish_balemoon_warning()
        self.assertEqual(self.owner.hp, 70)
        self.assertFalse(self.idol.is_alive)
        self.assertFalse(enemy_idol.is_alive)
        self.assertTrue(self.ally.is_alive)
        self.assertEqual((self.owner.kills, self.owner.ultimate_points), (1, 1))
        self.assertEqual(tracker.stats[self.idol.name]["deaths"], 1)
        self.assertEqual(tracker.stats[self.owner.name]["kills"], 1)
        area = self.game.destruction_areas[0]
        self.assertEqual((len(area["cells"]), area["level"], area["remaining_ticks"]), (25, 10, 10))

    def test_balemoon_level_ten_contract_keeps_dealing_damage_after_exit(self):
        self.enemy.hp = self.enemy.max_hp = 200
        self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"})
        self.finish_balemoon_warning()
        self.game._trigger_destruction_areas(self.enemy)
        self.assertEqual(self.enemy.life_contract_remaining, 10)
        self.enemy.pos = [11, 19]
        for _ in range(10):
            self.advance()
        self.assertEqual((self.enemy.hp, self.enemy.max_hp, self.enemy.life_contract_remaining), (100, 100, 0))

    def test_balemoon_cannot_be_cast_when_dead_or_without_five_points(self):
        self.owner.ultimate_points = 4
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"}))
        self.owner.ultimate_points, self.owner.is_alive = 5, False
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"}))

    def test_balemoon_warns_three_full_ticks_before_any_effect_occurs(self):
        self.owner.hp = 20
        center = tuple(self.owner.pos)
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"}))
        self.owner.pos = [11, 0]
        for remaining in (2, 1, 0):
            self.advance()
            self.assertEqual(self.game.balemoon_warnings[0]["remaining_ticks"], remaining)
            self.assertEqual(self.game.destruction_areas, [])
            self.assertEqual(self.owner.hp, 20)
            self.assertTrue(self.idol.is_alive)
            self.assertEqual(self.enemy.life_contract_remaining, 0)
        self.advance()
        self.assertEqual(self.game.balemoon_warnings, [])
        self.assertEqual(self.game.destruction_areas[0]["pos"], center)
        self.assertEqual(self.owner.hp, self.owner.max_hp)
        self.assertFalse(self.idol.is_alive)
        self.assertEqual((self.enemy.hp, self.enemy.life_contract_remaining), (90, 9))

    def test_replay_preserves_ash_flight_and_light_red_balemoon_warning(self):
        self.ash((4, 10), land=False)
        self.game.execute_ai_ultimate(self.owner, {"ultimate": "BALEMOON"})
        self.game.replay_frames = []
        VisualFPSBattle._record_replay_frame(self.game)
        frame = json.loads(json.dumps(self.game.replay_frames[0]))
        self.assertEqual(frame["ash_projectiles"][0]["progress"], 0)
        self.assertEqual(frame["ash_projectiles"][0]["path"][-1], [4, 10])
        self.assertEqual(frame["balemoon_warnings"][0]["remaining_ticks"], 3)
        self.assertEqual(len(frame["balemoon_warnings"][0]["cells"]), 25)
        viewer = ReplayViewer.__new__(ReplayViewer)
        viewer.canvas = RecordingCanvas()
        viewer.frames = [frame]
        viewer.index, viewer.cell, viewer.grid = 0, 20, self.game.grid.tolist()
        viewer.status = SimpleNamespace(set=lambda text: None)
        viewer.view_mode = SimpleNamespace(get=lambda: "ALL")
        viewer.draw_frame()
        fills = {call[2].get("fill") for call in viewer.canvas.calls}
        self.assertIn(BALEMOON_WARNING_COLOR, fills)
        self.assertIn("#ff9ca8", fills)

    def test_ai_ash_and_balemoon_payloads_are_executable(self):
        state = {"grid": self.game.grid, "chars": self.game.chars}
        for controller in (DefaultAttackerController(), DefaultDefenderController()):
            self.assertEqual(controller._decide_ability(self.owner, state), {"ability": "ASH", "target": (4, 4)})
        self.assertEqual(build_ultimate_action(self.game.grid, self.owner, self.game.chars), {"ultimate": "BALEMOON"})

    def test_replay_serializes_hp_cap_contract_area_sparkle_and_flash(self):
        self.ash()
        self.advance()
        self.ally.heal_sparkle_remaining = 5
        self.game.serenade_flash_remaining = 1
        self.game.replay_frames = []
        VisualFPSBattle._record_replay_frame(self.game)
        frame = json.loads(json.dumps(self.game.replay_frames[0]))
        self.assertEqual((frame["chars"][1]["hp"], frame["chars"][1]["max_hp"], frame["chars"][1]["life_contract"]), (90, 90, 4))
        self.assertEqual(frame["destruction_areas"][0]["level"], 5)
        self.assertEqual(frame["chars"][0]["ability_charges"], 2)
        self.assertEqual(frame["chars"][2]["heal_sparkle"], 5)
        self.assertEqual(frame["serenade_flash"], 1)

    def test_tunnel_replay_uses_yellow_for_warning_and_active_phases(self):
        viewer = ReplayViewer.__new__(ReplayViewer)
        viewer.canvas = RecordingCanvas()
        viewer.frames = [{"tunnel_bursts": [{"phase": "warning", "cells": [[1, 1]]},
                                           {"phase": "active", "cells": [[1, 2]]}]}]
        viewer.index, viewer.cell, viewer.grid = 0, 20, [[0]*4 for _ in range(4)]
        viewer.status = SimpleNamespace(set=lambda text: None)
        viewer.view_mode = SimpleNamespace(get=lambda: "ALL")
        viewer.draw_frame()
        fills = {call[2].get("fill") for call in viewer.canvas.calls}
        self.assertIn(TUNNEL_WARNING_COLOR, fills)
        self.assertIn(TUNNEL_ACTIVE_COLOR, fills)

    def test_ash_damage_and_expiry_in_real_battle(self):
        scenario = create_sample_retake_scenario()
        scenario.attackers = [{"name": "Derke", "pos": (7, 3)}]
        scenario.defenders = [{"name": "Demon1", "pos": (7, 8)}]
        scenario.planted_pos, scenario.initial_smokes = (7, 4), []
        with redirect_stdout(StringIO()):
            simulator = TacticalSimulator(scenario, attacker_ai_name="default", defender_ai_name="default")
            simulator.attacker_controller = simulator.defender_controller = FixedController("MOVE")
            simulator.grid[7, 3:9] = 0
            enemy = simulator.chars[1]
            enemy.hp = enemy.max_hp = 200
            self.assertTrue(simulator.execute_ai_ability(simulator.chars[0], {"ability": "ASH", "target": (7, 8)}))
            # Keep shooting blocked after the projectile's path has been set.
            simulator.grid[:, 5] = 1
            simulator.run({}, max_ticks=12)
        self.assertEqual((enemy.hp, enemy.max_hp), (100, 100))
        self.assertEqual(simulator.destruction_areas, [])


if __name__ == "__main__":
    unittest.main()
