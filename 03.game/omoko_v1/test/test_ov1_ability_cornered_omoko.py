"""「追い詰められたおもこ」の配置・寿命・既存射撃との統合。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "test"))

from test_ultimate_system import UltimateTestGame, make_character
from omoko_v1 import ov1_ability_cornered_omoko as ability


class CorneredOmokoTests(unittest.TestCase):
    def setUp(self):
        debug = patch.object(ability, "DEBUG_OMOKO_CORNERED_ENABLED", False)
        debug.start()
        self.addCleanup(debug.stop)

    def fixture(self, side="A"):
        game = UltimateTestGame(height=16, width=16)
        actor = make_character("おもこ", side, (1, 1))
        enemy = make_character("Demon1", "D" if side == "A" else "A", (5, 5))
        enemy.facing = "N"
        enemy.hp = enemy.max_hp = 10000
        enemy.accuracy = 0
        actor.hs_rate = 0
        game.chars = [actor, enemy]
        return game, actor, enemy

    def tick(self, game):
        game._build_occupancy_counts()
        try:
            for char in game._move_order():
                game.move_character(char)
        finally:
            game._clear_occupancy_counts()
        game.battle_tick += 1
        game._advance_omoko_cornered()
        with patch("battle_logic.random.random", return_value=1):
            game._resolve_all_shots()

    def test_activation_only_for_last_living_omoko_on_either_side(self):
        for side in ("A", "D"):
            game, actor, enemy = self.fixture(side)
            ally = make_character("Leo", side, (2, 2))
            game.chars.append(ally)
            self.assertFalse(game._omoko_cornered_active(actor))
            self.assertFalse(game._omoko_cornered_active(enemy))
            ally.is_alive = False
            self.assertTrue(game._omoko_cornered_active(actor))
            actor.is_alive = False
            self.assertFalse(game._omoko_cornered_active(actor))

    def test_no_activation_after_enemies_are_eliminated_or_when_disabled(self):
        game, actor, enemy = self.fixture()
        with patch.object(ability, "OMOKO_CORNERED_DURATION_TICKS", 0):
            self.assertFalse(game._omoko_cornered_active(actor))
        enemy.is_alive = False
        self.assertFalse(game._omoko_cornered_active(actor))

    def test_debug_includes_setup_and_activates_at_35_with_living_allies(self):
        from types import SimpleNamespace
        game, actor, _ = self.fixture()
        game.chars.append(make_character("Leo", actor.team, (2, 2)))
        game.defender_setup_phase = SimpleNamespace(active=True, ticks_elapsed=30)
        game.battle_tick = 0
        with patch.object(ability, "DEBUG_OMOKO_CORNERED_ENABLED", True), patch.object(
                ability, "DEBUG_OMOKO_CORNERED_START_TICK", 35):
            self.assertFalse(game._omoko_cornered_active(actor))
            game.defender_setup_phase.active = False
            game.battle_tick = 3  # 次の移動フェーズは通算34tick。
            self.assertFalse(game._omoko_cornered_active(actor))
            game.battle_tick = 34 - 30
            game._advance_omoko_cornered()  # 射撃前の通算34tick。
            self.assertFalse(actor.omoko_cornered_triggered)
            game.move_character(actor)  # 次tickの移動フェーズで起動。
            self.assertTrue(actor.omoko_cornered_triggered)
            self.assertEqual(actor.pos, [1, 1])
            game.battle_tick = 35 - 30
            game._advance_omoko_cornered()
            self.assertIsNotNone(actor.omoko_cornered_target)
            self.assertEqual(actor.omoko_cornered_remaining, ability.OMOKO_CORNERED_DURATION_TICKS - 1)

    def test_debug_cannot_activate_dead_omoko_or_without_enemies(self):
        game, actor, enemy = self.fixture()
        game.battle_tick = 35
        with patch.object(ability, "DEBUG_OMOKO_CORNERED_ENABLED", True):
            actor.is_alive = False
            self.assertFalse(game._omoko_cornered_active(actor))
            actor.is_alive = True
            enemy.is_alive = False
            self.assertFalse(game._omoko_cornered_active(actor))

    def test_all_eight_facings_and_all_fallback_levels(self):
        steps = ((-1, 0), (-1, 1), (0, 1), (1, 1),
                 (1, 0), (1, -1), (0, -1), (-1, -1))
        for index, facing in enumerate(("N", "NE", "E", "SE", "S", "SW", "W", "NW")):
            game, actor, enemy = self.fixture()
            enemy.facing = facing
            for offsets in ((3, 4, 5), (2, 6), (1, 7), (0,)):
                expected = {(5 + steps[(index + offset) % 8][0],
                             5 + steps[(index + offset) % 8][1]) for offset in offsets}
                with self.subTest(facing=facing, offsets=offsets):
                    self.assertEqual(set(game._omoko_cornered_destinations(actor, enemy)), expected)
                for cell in expected:
                    game.chars.append(make_character("Leo", enemy.team, cell))
            self.assertEqual(game._omoko_cornered_destinations(actor, enemy), [])

    def test_real_map_corner_does_not_repeat_wall_blocked_current_position(self):
        import numpy as np
        from map_data import NEW_MAZE_STR
        game, actor, enemy = self.fixture()
        game.grid = np.array([[int(value) for value in row]
                              for row in NEW_MAZE_STR.strip().splitlines()])
        game.height, game.width = game.grid.shape
        actor.pos, enemy.pos = [9, 3], [10, 2]
        enemy.facing = "S"
        self.assertFalse(game.check_shot_line_of_sight(actor, enemy))
        self.assertEqual(set(game._omoko_cornered_destinations(actor, enemy)),
                         {(10, 1), (10, 3)})
        for _ in range(3):
            self.tick(game)
            self.assertIs(actor.omoko_cornered_target, enemy)
            self.assertTrue(game.check_shot_line_of_sight(actor, enemy))
            self.assertTrue(any(shot["shooter"] is actor and shot["target"] is enemy
                                for shot in game.last_shots))

    def test_occupied_reserved_outside_and_wall_cells_are_excluded(self):
        game, actor, enemy = self.fixture()
        game.chars.append(make_character("Leo", enemy.team, (6, 4)))
        game.grid[6, 5] = 1
        game.escape_portals = [{"owner": "someone", "pos": (6, 6)}]
        self.assertEqual(set(game._omoko_cornered_destinations(actor, enemy)), {(5, 4), (5, 6)})
        game.escape_portals.clear()
        enemy.pos = [15, 15]
        self.assertTrue(all(0 <= r < 16 and 0 <= c < 16
                            for r, c in game._omoko_cornered_destinations(actor, enemy)))

    def test_unreachable_enemy_is_removed_and_other_enemy_is_used(self):
        game, actor, enemy = self.fixture()
        other = make_character("Leo", enemy.team, (11, 11))
        game.chars.append(other)
        game.grid[4:7, 4:7] = 1
        game._advance_omoko_cornered()
        self.assertIs(actor.omoko_cornered_target, other)

    def test_all_unreachable_skips_and_still_consumes_duration(self):
        game, actor, _ = self.fixture()
        game.grid[4:7, 4:7] = 1
        game._advance_omoko_cornered()
        self.assertEqual(actor.pos, [1, 1])
        self.assertIsNone(actor.omoko_cornered_target)
        self.assertEqual(actor.omoko_cornered_remaining, ability.OMOKO_CORNERED_DURATION_TICKS - 1)

    def test_previous_shot_target_excluded_and_single_enemy_turns_each_tick(self):
        game, actor, enemy = self.fixture()
        self.tick(game)
        self.assertEqual(actor.omoko_cornered_last_target, enemy.name)
        first_position = tuple(actor.pos)
        self.assertIsNotNone(enemy.forced_facing_next_tick)
        self.tick(game)
        self.assertNotEqual(tuple(actor.pos), first_position)
        dr, dc = actor.pos[0] - enemy.pos[0], actor.pos[1] - enemy.pos[1]
        from game_core import FACING_VECTORS
        fx, fy = FACING_VECTORS[enemy.facing]
        self.assertLess(dr * fy + dc * fx, 0)
        other = make_character("Leo", enemy.team, (11, 11))
        game.chars.append(other)
        self.tick(game)
        self.assertIs(actor.omoko_cornered_target, other)
        self.assertEqual(actor.omoko_cornered_last_target, other.name)

    def test_excluded_previous_enemy_does_not_return_when_others_are_blocked(self):
        game, actor, enemy = self.fixture()
        self.tick(game)
        other = make_character("Leo", enemy.team, (11, 11))
        game.chars.append(other)
        game.grid[10:13, 10:13] = 1
        self.tick(game)
        self.assertIsNone(actor.omoko_cornered_target)
        self.assertFalse(any(s["shooter"] is actor for s in game.last_shots))

    def test_exact_duration_no_retrigger_and_new_round_state(self):
        game, actor, _ = self.fixture()
        with patch.object(ability, "OMOKO_CORNERED_DURATION_TICKS", 2):
            for _ in range(2):
                self.tick(game)
                self.assertTrue(actor.omoko_cornered_this_tick)
            self.tick(game)
            self.assertFalse(actor.omoko_cornered_this_tick)
            self.assertEqual(actor.omoko_cornered_remaining, 0)
            self.assertTrue(actor.omoko_cornered_triggered)
            replacement = make_character("おもこ", actor.team, (1, 1))
            game.chars[0] = replacement
            self.assertTrue(game._omoko_cornered_active(replacement))

    def test_normal_actions_are_replaced_and_progress_is_cancelled(self):
        game, actor, _ = self.fixture()
        game.attacker_controller.decide_move = Mock(side_effect=AssertionError("AI action"))
        actor.plant_timer = actor.defuse_timer = actor.orb_collect_timer = 2
        actor.is_planting = True
        game.move_character(actor)
        self.assertEqual((actor.plant_timer, actor.defuse_timer, actor.orb_collect_timer), (0, 0, 0))
        self.assertEqual(actor.pos, [1, 1])
        game._advance_omoko_cornered()
        self.assertNotEqual(actor.pos, [1, 1])

    def test_movement_lock_blocks_full_tick_but_duration_runs(self):
        game, actor, _ = self.fixture()
        actor.movement_disabled_remaining = 1
        self.tick(game)
        self.assertEqual(actor.pos, [1, 1])
        self.assertIsNone(actor.omoko_cornered_target)
        self.tick(game)
        self.assertNotEqual(actor.pos, [1, 1])

    def test_existing_shooting_keeps_damage_miss_and_smoke_rules(self):
        game, actor, enemy = self.fixture()
        game._advance_omoko_cornered()
        with patch("battle_logic.random.random", return_value=0):
            game._resolve_all_shots()
        self.assertEqual(enemy.hp, 9960)
        self.assertEqual(len([s for s in game.last_shots if s["shooter"] is actor]), 1)
        self.assertTrue(actor.moved_this_tick)
        game.smokes = [{"cells": {tuple(actor.pos)}}]
        game._resolve_all_shots()
        # 既存ルールでは隣接相手への射線はスモーク内でも通る。
        self.assertTrue(any(s["shooter"] is actor for s in game.last_shots))

    def test_teleport_shot_uses_stationary_accuracy_and_hs_without_stop_penalty(self):
        game, actor, enemy = self.fixture()
        actor.accuracy, actor.hs_rate = 0.4, 0.8
        enemy.dodge_rate = 0
        with patch.object(ability.random, "choice", side_effect=lambda values: values[len(values) // 2]):
            game._advance_omoko_cornered()
        self.assertTrue(actor.moved_this_tick)
        actor.stopped_after_move_this_tick = True
        with patch("battle_logic.random.random", side_effect=[0.7, 0.5]):
            game._resolve_all_shots()
        shot = next(s for s in game.last_shots if s["shooter"] is actor)
        self.assertAlmostEqual(shot["hit_chance"], 0.8)
        self.assertTrue(shot["headshot"])
        self.assertEqual(shot["damage"], 160)

        # 通常移動での射撃には従来の移動・停止ペナルティが残る。
        actor.omoko_cornered_this_tick = False
        with patch("battle_logic.random.random", side_effect=[0, 0.2]):
            game._resolve_all_shots()
        shot = next(s for s in game.last_shots if s["shooter"] is actor)
        self.assertAlmostEqual(shot["hit_chance"], 0.45)
        self.assertFalse(shot["headshot"])
        self.assertEqual(shot["damage"], 40)

    def test_last_kill_teleports_defender_to_spike_and_allows_normal_defuse(self):
        from test_ultimate_system import FixedController
        game, actor, enemy = self.fixture("D")
        game.is_planted = True
        game.planted_pos = (11, 11)
        game.grid[10:13, 10:13] = 1
        game.grid[10, 11] = 0  # 唯一の空きマス。
        enemy.hp = 1
        enemy.dodge_rate = 0
        actor.accuracy = 1
        game._save_ultimate_points = Mock()
        # 最終持続tickのキルでも解除位置へ移動する。
        with patch.object(ability, "OMOKO_CORNERED_DURATION_TICKS", 1):
            game.move_character(actor)
            game._advance_omoko_cornered()
        with patch("battle_logic.random.random", return_value=0):
            game._resolve_all_shots()
        self.assertFalse(enemy.is_alive)
        self.assertEqual(actor.pos, [10, 11])
        self.assertEqual(actor.facing, "S")
        self.assertEqual(actor.omoko_cornered_remaining, 0)
        self.assertFalse(actor.omoko_cornered_this_tick)
        game.defender_controller = FixedController("DEFUSE")
        game.is_defused = False
        from game_core import DEFUSE_REQUIRED_TICKS
        for expected in range(1, DEFUSE_REQUIRED_TICKS + 1):
            game.move_character(actor)
            game.battle_tick += 1
            game._advance_omoko_cornered()
            game._resolve_defuse_completion()
            self.assertEqual(actor.defuse_timer, expected)
            self.assertEqual(actor.pos, [10, 11])
            self.assertEqual(game.is_defused, expected == DEFUSE_REQUIRED_TICKS)

    def test_spike_return_excludes_occupied_and_reserved_cells(self):
        game, actor, enemy = self.fixture("D")
        game._advance_omoko_cornered()
        enemy.is_alive = False
        game.is_planted, game.planted_pos = True, (11, 11)
        game.grid[10:13, 10:13] = 1
        for pos in ((10, 10), (10, 11), (10, 12)):
            game.grid[pos] = 0
        game.chars.append(make_character("Leo", "D", (10, 10)))
        game.escape_portals = [{"owner": "someone", "pos": (10, 11)}]
        game._finish_omoko_cornered_combat()
        self.assertEqual(actor.pos, [10, 12])

    def test_no_spike_return_for_attacker_unplanted_or_blocked_surroundings(self):
        for side, planted, blocked in (("A", True, False), ("D", False, False),
                                       ("D", True, True)):
            with self.subTest(side=side, planted=planted, blocked=blocked):
                game, actor, enemy = self.fixture(side)
                game._advance_omoko_cornered()
                before = list(actor.pos)
                enemy.is_alive = False
                game.is_planted, game.planted_pos = planted, (11, 11)
                if blocked:
                    game.grid[10:13, 10:13] = 1
                game._finish_omoko_cornered_combat()
                self.assertEqual(actor.pos, before)
                self.assertFalse(game._omoko_cornered_active(actor))
                self.assertEqual(actor.omoko_cornered_remaining, 0)

    def test_surviving_enemy_keeps_ability_active_without_spike_return(self):
        game, actor, _ = self.fixture("D")
        game.is_planted, game.planted_pos = True, (11, 11)
        game._advance_omoko_cornered()
        before = list(actor.pos)
        remaining = actor.omoko_cornered_remaining
        game._finish_omoko_cornered_combat()
        self.assertEqual(actor.pos, before)
        self.assertTrue(actor.omoko_cornered_this_tick)
        self.assertEqual(actor.omoko_cornered_remaining, remaining)

    def test_real_step_tick_teleports_before_shots_and_resets_next_round(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from map_data import NEW_MAZE_STR
        with redirect_stdout(StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
            game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("default"), _build_team_ai("default"),
                headless=True,
                attacker_roster=["おもこ", "Leo", "Boaster", "Chronicle", "Alfajer"],
                defender_roster=["Aspas", "valyn", "trent", "leaf", "tex"],
            )
        game.defender_setup_phase.finish()
        game.record_replay = False
        actor = next(c for c in game.chars if c.name == "おもこ")
        enemy = next(c for c in game.chars if c.team != actor.team)
        for char in game.chars:
            char.is_alive = char is actor or char is enemy
        actor.pos, enemy.pos = [10, 10], [20, 20]
        game.grid[18:23, 18:23] = 0
        actor.hp = enemy.hp = 10000
        actor.accuracy = enemy.accuracy = 0
        from test_ultimate_system import FixedController
        game.attacker_controller = FixedController("MOVE")
        game.defender_controller = FixedController("MOVE")
        with patch("battle_logic.random.random", return_value=1):
            self.assertTrue(game.step_tick())
            first = tuple(actor.pos)
            self.assertEqual(max(abs(actor.pos[i] - enemy.pos[i]) for i in (0, 1)), 1)
            self.assertTrue(any(s["shooter"] is actor and s["target"] is enemy
                                for s in game.last_shots))
            self.assertTrue(game.step_tick())
            self.assertNotEqual(tuple(actor.pos), first)
        with redirect_stdout(StringIO()):
            game.init_round()
        fresh_actor = next(c for c in game.chars if c.name == "おもこ")
        self.assertFalse(fresh_actor.omoko_cornered_triggered)
        self.assertEqual(fresh_actor.omoko_cornered_remaining, 0)


if __name__ == "__main__":
    unittest.main()
