"""Only the shot-response rule owns combat facing; controllers cannot override it."""

import contextlib
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from game_core import SHOOT_INTERVAL_TICKS
from test_ultimate_system import UltimateTestGame, make_character


class CombatFacingRulesTests(unittest.TestCase):
    def fixture(self, side="A", name="Xdll"):
        game = UltimateTestGame()
        actor = make_character(name, side, (4, 4), 99)
        enemy = make_character("Demon1", "D" if side == "A" else "A", (4, 7))
        actor.facing = "W"
        enemy.facing = "E"
        actor.hp = enemy.hp = 100000
        actor.accuracy = enemy.accuracy = 0
        game.chars = [actor, enemy]
        game.battle_tick = SHOOT_INTERVAL_TICKS
        return game, actor, enemy

    def set_controller(self, game, actor, action="MOVE", destination=None):
        def decide(char, state):
            char.facing = "S"  # Model side effects must be ignored too.
            if isinstance(action, dict):
                return list(char.pos) if destination is None else destination, dict(action)
            return list(char.pos) if destination is None else destination, action
        ctrl = NS(decide_move=decide)
        if actor.team == "A":
            game.attacker_controller = ctrl
        else:
            game.defender_controller = ctrl

    def test_clear_shot_to_unseen_rear_enemy_does_not_turn_or_fire(self):
        game, actor, enemy = self.fixture()
        self.assertTrue(game.check_shot_line_of_sight(actor, enemy))
        self.assertGreater(game._facing_angle_diff(actor, enemy), 90)
        game._resolve_all_shots()
        self.assertEqual(actor.facing, "W")
        self.assertIsNone(actor.forced_facing_next_tick)
        self.assertFalse(any(s["shooter"] is actor for s in game.last_shots))

    def test_recon_does_not_turn_to_unseen_rear_enemy(self):
        game, actor, enemy = self.fixture()
        enemy.reveal_remaining = 4
        game.smokes = [{"cells": {(4, 5), (4, 6)}}]
        self.assertTrue(game.check_shot_line_of_sight(actor, enemy))
        game._resolve_all_shots()
        self.assertEqual(actor.facing, "W")
        self.assertFalse(any(s["shooter"] is actor for s in game.last_shots))

    def test_existing_incoming_shot_rule_still_turns_on_next_tick(self):
        game, actor, enemy = self.fixture()
        enemy.facing = "W"  # Enemy actually sees/fires at the actor.
        game._resolve_all_shots()
        self.assertTrue(any(s["shooter"] is enemy and s["target"] is actor
                            for s in game.last_shots))
        self.assertEqual(actor.facing, "W")
        self.assertEqual(actor.forced_facing_next_tick, "E")
        self.set_controller(game, actor, {"facing": "W"})
        game.move_character(actor)
        self.assertEqual(actor.facing, "E")
        self.assertTrue(actor.facing_forced_this_tick)
        self.assertIsNone(actor.forced_facing_next_tick)

    def test_explicit_and_movement_facing_cannot_override_rule_on_either_side(self):
        for side in ("A", "D"):
            for payload in ("MOVE", {"facing": "W"}):
                with self.subTest(side=side, payload=payload):
                    game, actor, _ = self.fixture(side)
                    actor.forced_facing_next_tick = "E"
                    self.set_controller(game, actor, payload, [3, 4])
                    game.move_character(actor)
                    self.assertEqual(actor.facing, "E")
                    if payload == "MOVE" or side == "D":
                        self.assertEqual(actor.pos, [3, 4])

    def test_rule_is_visible_to_controller_and_direct_mutation_is_restored(self):
        game, actor, _ = self.fixture()
        actor.forced_facing_next_tick = "N"
        seen = []
        def decide(char, state):
            seen.append((char.facing, char.facing_forced_this_tick))
            char.facing = "E"
            char.facing_forced_this_tick = False
            return list(char.pos), {"facing": "E"}
        game.attacker_controller = NS(decide_move=decide)
        game.move_character(actor)
        self.assertEqual(seen, [("N", True)])
        self.assertEqual(actor.facing, "N")

    def test_directional_ultimate_effect_uses_rule_direction_not_model_payload(self):
        for name in ("something", "Chronicle"):
            with self.subTest(name=name):
                game, actor, _ = self.fixture(name=name)
                actor.forced_facing_next_tick = "N"
                self.set_controller(game, actor, {"ultimate": actor.ultimate_name, "facing": "E"})
                expected = game._tunnel_cells(tuple(actor.pos), "N")
                game.move_character(actor)
                self.assertEqual(actor.facing, "N")
                self.assertEqual(actor.ultimate_points, 0)
                if actor.ultimate_name == "RAID":
                    self.assertEqual(actor.pos[1], 4)
                    self.assertLess(actor.pos[0], 4)
                else:
                    self.assertEqual(game.tunnel_bursts[0]["cells"], expected)

    def test_plant_defuse_orb_and_ability_returns_preserve_rule_facing(self):
        for action in ("PLANT", "DEFUSE", "COLLECT_ORB", {"ability": "RECON", "target": (4, 7)}):
            with self.subTest(action=action):
                side = "D" if action == "DEFUSE" else "A"
                game, actor, _ = self.fixture(side)
                actor.forced_facing_next_tick = "N"
                game.grid[4, 4] = 2
                actor.has_spike = True
                game.available_orbs = {(4, 4)}
                if action == "DEFUSE":
                    game.is_planted = True
                    game.planted_pos = (4, 4)
                self.set_controller(game, actor, action)
                game.move_character(actor)
                self.assertEqual(actor.facing, "N")
                self.assertTrue(actor.facing_forced_this_tick)

    def test_normal_controller_direction_resumes_after_rule_lock_expires(self):
        game, actor, _ = self.fixture()
        actor.forced_facing_next_tick = "N"
        self.set_controller(game, actor, {"facing": "E"})
        game.move_character(actor)
        self.assertEqual(actor.facing, "N")
        game.move_character(actor)
        self.assertEqual(actor.facing, "E")
        self.assertFalse(actor.facing_forced_this_tick)

    def test_failed_movement_does_not_release_rule_direction(self):
        game, actor, _ = self.fixture()
        actor.forced_facing_next_tick = "N"
        game.grid[3, 4] = 1
        self.set_controller(game, actor, {"facing": "E"}, [3, 4])
        game.move_character(actor)
        self.assertEqual(actor.pos, [4, 4])
        self.assertEqual(actor.facing, "N")

    def test_exception_cannot_leave_controller_facing_in_place(self):
        game, actor, _ = self.fixture()
        actor.forced_facing_next_tick = "N"
        def fail(char, state):
            char.facing = "E"
            raise RuntimeError("controller failed")
        game.attacker_controller = NS(decide_move=fail)
        with self.assertRaises(RuntimeError):
            game.move_character(actor)
        self.assertEqual(actor.facing, "N")

    def test_real_battle_tick_does_not_auto_turn_a_gc_actor_to_rear_enemy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from run_game import VisualFPSBattle, _build_team_ai
            from map_data import NEW_MAZE_STR
            game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai("default"),
                                   _build_team_ai("default"), headless=True)
        import numpy as np
        game.grid = np.zeros((9, 12), dtype=np.int32)
        game.height, game.width = game.grid.shape
        _, actor, enemy = self.fixture()
        game.chars = [actor, enemy]
        game.analytics_tracker = None
        game.check_match_winner = lambda: None
        class GhostChampionsV1TestController:
            auto_face_visible_enemy = True  # Old opt-in must not change rules.
        game.attacker_controller = NS(inner=GhostChampionsV1TestController())
        game.process_battle()
        self.assertEqual(actor.facing, "W")
        self.assertIsNone(actor.forced_facing_next_tick)

    def test_search_training_does_not_replace_policy_facing_with_los_auto_aim(self):
        from gc_v1 import train_defender_search_gc as search
        self.check_search_facing(search, (0, "W"))

    def test_legacy_search_v2_does_not_auto_turn_before_shots(self):
        try:
            from gc_v1 import train_defender_search_gc_v2 as search_v2
        except ImportError as error:
            if "GC_DEFENSE_DEPTH_BIAS_BY_MARKER" not in str(error):
                raise
            self.skipTest("Legacy Search v2 requires a removed depth-bias configuration")
        self.check_search_facing(search_v2, 0)

    def check_search_facing(self, module, action):
        env = module.SearchEnv()
        env.reset()
        actor, enemy = env.defenders[0], env.attackers[0]
        for unit in env.defenders + env.attackers:
            unit.is_alive = unit is actor or unit is enemy
            unit.facing = "W"
        actor.pos = [7, 3]
        enemy.pos = [7, 4]
        env._attacker_decide_move = lambda unit: (0, 0)
        snapshots = []
        env._resolve_shots = lambda: snapshots.append(actor.facing)
        with patch.object(module, "has_los", return_value=True):
            env.step({actor.name: action})
        self.assertEqual(snapshots, ["W"])

    def test_retake_training_does_not_auto_turn_before_shot_resolution(self):
        from gc_v1 import train_defender_retake_gc as training
        env = training.RetakeEnv()
        env.reset()
        actor, enemy = env.defenders()[0], env.attackers()[0]
        for unit in env.chars:
            unit.is_alive = unit is actor or unit is enemy
            unit.facing = "W"
        actor.pos = [7, 3]
        enemy.pos = [7, 4]
        env.attacker_stub.decide_move = lambda char, chars, planted: list(char.pos)
        snapshots = []
        def shots(current_los):
            snapshots.append(actor.facing)
            return []
        env._resolve_shots = shots
        with patch.object(env, "check_line_of_sight", return_value=True):
            env.step_tick({actor.name: 4})
        self.assertEqual(snapshots, ["W"])


if __name__ == "__main__":
    unittest.main()
