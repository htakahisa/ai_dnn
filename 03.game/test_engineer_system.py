"""Engineer mechanics, tick boundaries, visibility, and replay integration."""

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from analytics.replay_viewer import ReplayViewer
from game_core import COMBO_BANNER_HEIGHT
from gc_v1.ultimate_tactics_gc import build_ultimate_action
from rendering_ui import RenderingUIMixin
from run_game import VisualFPSBattle
from tactical_simulator import TacticalSimulator, create_sample_retake_scenario
from test_ultimate_system import UltimateTestGame, make_character


class MoveEast:
    def decide_move(self, char, state):
        return [char.pos[0], char.pos[1] + 1], "MOVE"


class RecordingCanvas:
    def __init__(self):
        self.calls = []

    def __getattr__(self, method):
        def record(*args, **kwargs):
            self.calls.append((method, args, kwargs))
        return record


class EngineerSystemTests(unittest.TestCase):
    def setUp(self):
        self.game = UltimateTestGame(height=12, width=20)
        self.owner = make_character("Alfajer", "A", (1, 1), 9)
        self.game.chars = [self.owner]
        self.game.defender_controller = MoveEast()

    def add_enemy(self, name, pos):
        enemy = make_character(name, "D", pos)
        self.game.chars.append(enemy)
        return enemy

    def advance(self):
        self.game.battle_tick += 1
        for char in self.game.chars:
            char.reveal_remaining = max(0, char.reveal_remaining - 1)
        self.game._advance_engineer_effects()

    def plant(self):
        self.assertTrue(self.game.execute_ai_ability(self.owner, {"ability": "RAMP"}))

    def test_engineer_starts_with_two_ramps_and_nine_point_neon(self):
        self.assertEqual((self.owner.role, self.owner.ability_name, self.owner.ramp_charges),
                         ("エンジニア", "RAMP", 2))
        self.assertEqual((self.owner.ultimate_name, self.owner.ultimate_cost), ("NEON", 9))
        self.assertEqual(self.owner.flash_charges, 0)

    def test_ramp_is_always_on_own_cell_and_cannot_exceed_inventory(self):
        self.assertTrue(self.game.execute_ai_ability(self.owner, {"ability": "RAMP", "target": (8, 8)}))
        self.assertEqual(self.game.ramp_traps[0]["pos"], (1, 1))
        self.assertFalse(self.game.execute_ai_ability(self.owner, {"ability": "RAMP"}))
        self.assertEqual(self.owner.ramp_charges, 1)
        self.owner.pos = [1, 2]
        self.plant()
        self.owner.pos = [1, 3]
        self.assertFalse(self.game.execute_ai_ability(self.owner, {"ability": "RAMP"}))

    def test_ramp_chains_transitively_and_ignores_allies_and_dead_players(self):
        self.plant()
        self.owner.pos = [9, 19]
        first = self.add_enemy("Demon1", (1, 1))
        second = self.add_enemy("Leo", (1, 6))
        third = self.add_enemy("Chronicle", (1, 11))
        far = self.add_enemy("Derke", (1, 17))
        dead = self.add_enemy("Aspas", (2, 1))
        dead.is_alive = False
        ally = make_character("Boaster", "A", (2, 2))
        self.game.chars.append(ally)
        self.game._trigger_ramp_traps(first)
        self.assertEqual(self.game.ramp_traps, [])
        for enemy in (first, second, third):
            self.assertEqual((enemy.electric_remaining, enemy.reveal_remaining, enemy.hp), (5, 5, 100))
        for char in (far, dead, ally, self.owner):
            self.assertEqual(char.electric_remaining, 0)

    def test_chain_uses_wall_aware_shortest_path_and_not_diagonal_distance(self):
        self.plant()
        first = self.add_enemy("Demon1", (1, 1))
        behind_wall = self.add_enemy("Leo", (1, 3))
        diagonal = self.add_enemy("Chronicle", (4, 4))
        self.game.grid[:, 2] = 1
        self.game._trigger_ramp_traps(first)
        self.assertEqual(behind_wall.electric_remaining, 0)
        self.assertEqual(diagonal.electric_remaining, 0)
        self.game.grid[:, 2] = 0
        self.assertNotIn((4, 4), self.game._ramp_reachable_cells((1, 1)))
        self.assertIn((1, 6), self.game._ramp_reachable_cells((1, 1)))

    def test_ally_does_not_trigger_trap(self):
        self.plant()
        self.game._trigger_ramp_traps(self.owner)
        self.assertEqual(len(self.game.ramp_traps), 1)
        self.assertEqual(self.owner.electric_remaining, 0)

    def test_ramp_stops_multi_step_movement_and_expires_after_five_ticks(self):
        self.owner.pos = [1, 2]
        self.plant()
        self.owner.pos = [9, 19]
        enemy = self.add_enemy("Demon1", (1, 1))
        enemy.move_steps_per_tick = 3
        self.game.move_character(enemy)
        self.assertEqual(enemy.pos, [1, 2])
        self.advance()
        self.assertEqual(enemy.electric_remaining, 5)
        for remaining in (4, 3, 2, 1):
            self.game.move_character(enemy)
            self.assertEqual(enemy.pos, [1, 2])
            self.advance()
            self.assertEqual((enemy.electric_remaining, enemy.reveal_remaining), (remaining, remaining))
        enemy.move_steps_per_tick = 1
        self.game.move_character(enemy)
        self.assertEqual(enemy.pos, [1, 3])
        self.advance()
        self.assertEqual((enemy.electric_remaining, enemy.reveal_remaining), (0, 0))

    def test_raid_stops_on_trap_in_middle_of_path(self):
        self.owner.pos = [1, 3]
        self.plant()
        self.owner.pos = [9, 19]
        enemy = self.add_enemy("something", (1, 1))
        enemy.facing = "E"
        enemy.ultimate_points = enemy.ultimate_cost
        self.assertTrue(self.game.execute_ai_ultimate(enemy, {"ultimate": "RAID"}))
        self.assertEqual(enemy.pos, [1, 3])
        self.assertEqual(enemy.electric_remaining, 5)

    def test_owner_death_immediately_removes_owned_traps_only(self):
        self.plant()
        second_owner = make_character("Alfajer", "D", (3, 3))
        second_owner.name = "second_engineer"
        self.game.chars.append(second_owner)
        self.assertTrue(self.game.execute_ai_ability(second_owner, {"ability": "RAMP"}))
        self.game._kill_character(second_owner, self.owner)
        self.assertEqual([trap["owner"] for trap in self.game.ramp_traps], [second_owner.name])

    def test_escape_teleport_triggers_trap_without_extending_effect_duration(self):
        self.plant()
        self.owner.pos = [9, 19]
        enemy = self.add_enemy("Demon1", (9, 1))
        enemy.ultimate_points = enemy.ultimate_cost
        self.assertTrue(self.game.execute_ai_ultimate(enemy, {"ultimate": "ESCAPE", "target": (1, 1)}))
        self.game.escape_portals[0]["remaining_ticks"] = 0
        self.game._advance_escape_portals()
        self.assertEqual(enemy.pos, [1, 1])
        self.assertEqual(enemy.electric_remaining, 5)
        for expected in (4, 3, 2, 1, 0):
            self.advance()
            self.assertEqual(enemy.electric_remaining, expected)

    def test_neon_requires_valid_target_and_points(self):
        for payload in ({"ultimate": "NEON"}, {"ultimate": "NEON", "target": (-1, 2)}):
            self.assertFalse(self.game.execute_ai_ultimate(self.owner, payload))
        self.assertEqual(self.owner.ultimate_points, 9)
        self.owner.ultimate_points = 8
        self.assertFalse(self.game.execute_ai_ultimate(self.owner, {"ultimate": "NEON", "target": (4, 4)}))

    def test_neon_warns_ten_ticks_then_deals_ten_damage_for_ten_ticks(self):
        enemy = self.add_enemy("Demon1", (4, 4))
        enemy.hp = enemy.max_hp = 200
        outside = self.add_enemy("Leo", (4, 8))
        self.owner.pos = [4, 3]
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, {"ultimate": "NEON", "target": (4, 4)}))
        self.assertEqual(self.owner.ultimate_points, 0)
        self.assertEqual(len(self.game.neon_bursts[0]["cells"]), 49)
        for _ in range(10):
            self.advance()
            self.assertEqual(enemy.hp, 200)
            self.assertEqual(self.game.neon_bursts[0]["phase"], "warning")
        for tick in range(1, 11):
            self.advance()
            self.assertEqual(enemy.hp, 200 - tick*10)
            self.assertEqual(self.game.neon_bursts[0]["phase"], "active")
        self.assertEqual((outside.hp, self.owner.hp), (100, 100))
        self.advance()
        self.assertEqual(self.game.neon_bursts, [])
        self.assertEqual(enemy.hp, 100)

    def test_neon_hits_new_arrivals_and_credits_lethal_damage(self):
        enemy = self.add_enemy("Demon1", (10, 10))
        self.game.execute_ai_ultimate(self.owner, {"ultimate": "NEON", "target": (4, 4)})
        for _ in range(11):
            self.advance()
        self.assertEqual(enemy.hp, 100)
        enemy.pos, enemy.hp = [4, 4], 10
        self.advance()
        self.assertFalse(enemy.is_alive)
        self.assertEqual((self.owner.kills, self.owner.ultimate_points, enemy.deaths), (1, 1, 1))

    def test_neon_continues_after_caster_dies_and_clips_map_edges(self):
        enemy = self.add_enemy("Demon1", (0, 0))
        self.game.execute_ai_ultimate(self.owner, {"ultimate": "NEON", "target": (0, 0)})
        self.assertEqual(len(self.game.neon_bursts[0]["cells"]), 16)
        self.owner.is_alive = False
        for _ in range(11):
            self.advance()
        self.assertEqual(enemy.hp, 90)

    def test_trap_rendering_is_hidden_from_enemy_team(self):
        self.plant()
        self.game.canvas = RecordingCanvas()
        self.game.cell_size = 20
        self.game._map_x = lambda x: x
        self.game.get_viewer_team = lambda: "D"
        RenderingUIMixin._draw_engineer_effects(self.game)
        self.assertEqual(self.game.canvas.calls, [])
        self.game.get_viewer_team = lambda: "A"
        RenderingUIMixin._draw_engineer_effects(self.game)
        self.assertEqual(len(self.game.canvas.calls), 1)

    def prepare_click_ui(self):
        self.game._handle_team_panel_click = lambda x, y: False
        self.game._selected_user_character = lambda: self.owner
        self.game._plant_button_bounds = lambda: None
        self.game._orb_button_bounds = lambda: None
        self.game._ability_button_bounds = lambda ability: (0, 240, 100, 270)
        self.game._ultimate_button_bounds = lambda: (110, 240, 220, 270)
        self.game.draw = lambda: None
        self.game.map_offset_x = 0
        self.game.map_pixel_width = 400
        self.game.cell_size = 20
        self.game.ability_mode = self.game.ultimate_mode = None

    def click(self, x, y):
        RenderingUIMixin.on_canvas_click(self.game, SimpleNamespace(x=x, y=y+COMBO_BANNER_HEIGHT))

    def test_neon_click_arms_cancels_and_casts_at_selected_cell(self):
        self.prepare_click_ui()
        self.click(120, 250)
        self.assertEqual(self.game.ultimate_mode, ("NEON", "A", self.owner.name))
        self.assertEqual(self.owner.ultimate_points, 9)
        self.click(120, 250)
        self.assertIsNone(self.game.ultimate_mode)
        self.click(120, 250)
        self.click(90, 90)
        self.assertEqual(self.game.neon_bursts[0]["pos"], (4, 4))
        self.assertEqual(self.owner.ultimate_points, 0)
        self.assertIsNone(self.game.ultimate_mode)

    def test_ramp_click_places_on_current_cell(self):
        self.prepare_click_ui()
        self.click(50, 250)
        self.assertEqual(self.game.ramp_traps[0]["pos"], tuple(self.owner.pos))
        self.assertEqual(self.owner.ramp_charges, 1)

    def test_replay_hides_traps_in_enemy_view_and_draws_neon(self):
        viewer = ReplayViewer.__new__(ReplayViewer)
        viewer.canvas = RecordingCanvas()
        viewer.frames = [{"ramp_traps": [{"pos": [1, 1], "team": "A"}],
                          "neon_bursts": [{"phase": "active", "cells": [[4, 4]]}]}]
        viewer.index = 0
        viewer.grid = [[0]*6 for _ in range(6)]
        viewer.cell = 20
        viewer.status = SimpleNamespace(set=lambda text: None)
        viewer.view_mode = SimpleNamespace(get=lambda: "D")
        viewer.draw_frame()
        traps = [call for call in viewer.canvas.calls if call[2].get("fill") == "#1773d1"]
        self.assertEqual(traps, [])
        self.assertTrue(any(call[0] == "create_line" for call in viewer.canvas.calls))
        viewer.canvas.calls.clear()
        viewer.view_mode = SimpleNamespace(get=lambda: "A")
        viewer.draw_frame()
        self.assertEqual(sum(call[2].get("fill") == "#1773d1" for call in viewer.canvas.calls), 1)

    def test_replay_contains_traps_electricity_inventory_and_neon(self):
        self.plant()
        enemy = self.add_enemy("Demon1", (1, 1))
        self.game._trigger_ramp_traps(enemy)
        self.owner.pos = [2, 2]
        self.plant()
        self.game.execute_ai_ultimate(self.owner, {"ultimate": "NEON", "target": (4, 4)})
        self.game.replay_frames = []
        VisualFPSBattle._record_replay_frame(self.game)
        frame = json.loads(json.dumps(self.game.replay_frames[0]))
        self.assertEqual(frame["ramp_traps"][0]["team"], "A")
        self.assertEqual(frame["chars"][1]["electric"], 5)
        self.assertEqual(frame["chars"][0]["ability_charges"], 0)
        self.assertEqual(frame["neon_bursts"][0]["phase"], "warning")

    def test_neon_ai_payload_is_executable(self):
        self.add_enemy("Demon1", (4, 4))
        payload = build_ultimate_action(self.game.grid, self.owner, self.game.chars)
        self.assertEqual(payload, {"ultimate": "NEON", "target": (4, 4)})
        self.assertTrue(self.game.execute_ai_ultimate(self.owner, payload))

    def test_tactical_simulator_runs_neon_through_real_battle_and_replay(self):
        scenario = create_sample_retake_scenario()
        scenario.attackers = [{"name": "Alfajer", "pos": (7, 3), "ability_charges": 1, "ultimate_points": 9}]
        scenario.defenders = [{"name": "Demon1", "pos": (9, 3)}]
        scenario.initial_smokes = []
        scenario.planted_pos = (7, 4)
        with redirect_stdout(StringIO()):
            simulator = TacticalSimulator(scenario, attacker_ai_name="default", defender_ai_name="default")
            for char in simulator.chars:
                char.accuracy = 0
            simulator.attacker_controller = simulator.defender_controller = SimpleNamespace(
                decide_move=lambda char, state: (list(char.pos), "MOVE"))
            self.assertEqual(simulator.chars[0].ramp_charges, 1)
            self.assertTrue(simulator.execute_ai_ultimate(simulator.chars[0], {"ultimate": "NEON", "target": (8, 3)}))
            result = simulator.run({}, max_ticks=11)
        self.assertEqual(simulator.chars[1].hp, 90)
        self.assertEqual(result.replay_frames[-1]["neon_bursts"][0]["phase"], "active")


if __name__ == "__main__":
    unittest.main()
