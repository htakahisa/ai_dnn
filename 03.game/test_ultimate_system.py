import unittest

import numpy as np

from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin
from game_core import Character, ORB_COLLECT_REQUIRED_TICKS
from map_data import NEW_MAZE_STR
from gc_v1.ultimate_tactics_gc import build_ultimate_action


class FixedController:
    def __init__(self, action):
        self.action = action

    def decide_move(self, char, game_state):
        return list(char.pos), self.action


class UltimateTestGame(AbilityLosMixin, BattleLogicMixin):
    def __init__(self, height=9, width=12):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.chars = []
        self.match_stats = {}
        self.monitor_drones = []
        self.monitor_drone_serial = 0
        self.escape_portals = []
        self.tunnel_bursts = []
        self.smokes = []
        self.flash_projectiles = []
        self.recon_projectiles = []
        self.flash_bursts = []
        self.recon_bursts = []
        self.available_orbs = set()
        self._movement_occupancy_counts = None
        self.active_defuser_name = None
        self.spike_pos = None
        self.is_planted = False
        self.planted_pos = None
        self.target_plant_pos = None
        self.detonate_timer = 55
        self.round_timer = 100
        self.analytics_tracker = None
        self.current_round = 1
        self.battle_tick = 1
        self.headless = True
        self.attacker_controller = FixedController("MOVE")
        self.defender_controller = FixedController("MOVE")


def make_character(name, team, pos, points=0):
    return Character(
        name,
        team,
        pos,
        "white",
        "red" if team == "A" else "green",
        ultimate_points=points,
    )


class UltimateSystemTests(unittest.TestCase):
    def test_gc_learned_ultimate_payloads_are_executable(self):
        grid = np.zeros((9, 12), dtype=np.int32)
        tiger = make_character("something", "A", (4, 1), 3)
        tiger.facing = "E"
        smoker = make_character("Demon1", "A", (2, 2), 6)
        seeker = make_character("Leo", "A", (3, 2), 8)
        flash = make_character("Chronicle", "A", (5, 2), 5)
        chars = [tiger, smoker, seeker, flash]

        self.assertEqual(build_ultimate_action(grid, tiger, chars), {"ultimate": "RAID"})
        self.assertEqual(
            build_ultimate_action(grid, smoker, chars, destination=(7, 10)),
            {"ultimate": "ESCAPE", "target": (7, 10)},
        )
        self.assertEqual(build_ultimate_action(grid, seeker, chars), {"ultimate": "MONITOR"})
        self.assertEqual(build_ultimate_action(grid, flash, chars), {"ultimate": "TUNNEL"})

    def test_role_ultimate_costs(self):
        expected = {
            "something": ("RAID", 3),
            "Demon1": ("ESCAPE", 6),
            "Leo": ("MONITOR", 8),
            "Chronicle": ("TUNNEL", 5),
        }
        for name, (ultimate, cost) in expected.items():
            char = make_character(name, "A", (1, 1), 99)
            self.assertEqual((char.ultimate_name, char.ultimate_cost), (ultimate, cost))
            self.assertEqual(char.ultimate_points, cost)

    def test_map_has_configured_orb_spawn_cells(self):
        rows = [line for line in NEW_MAZE_STR.strip().splitlines() if line]
        cells = [
            (row, col)
            for row, values in enumerate(rows)
            for col, value in enumerate(values)
            if value == "5"
        ]
        self.assertEqual(cells, [(16, 23), (17, 8), (20, 42)])

    def test_raid_stops_before_wall_and_spends_points(self):
        game = UltimateTestGame()
        tiger = make_character("something", "A", (4, 1), 3)
        tiger.facing = "E"
        game.grid[4, 4] = 1
        game.chars = [tiger]

        self.assertTrue(game.execute_ai_ultimate(tiger, {"ultimate": "RAID"}))
        self.assertEqual(tiger.pos, [4, 3])
        self.assertEqual(tiger.ultimate_points, 0)

    def test_escape_opens_portal_then_teleports_after_ten_ticks(self):
        game = UltimateTestGame()
        smoker = make_character("Demon1", "A", (2, 2), 6)
        game.chars = [smoker]

        self.assertTrue(
            game.execute_ai_ultimate(
                smoker,
                {"ultimate": "ESCAPE", "target": (7, 10)},
            )
        )
        self.assertEqual(smoker.pos, [2, 2])
        self.assertEqual(smoker.ultimate_points, 0)
        self.assertEqual(game.escape_portals[0]["pos"], (7, 10))

        game._advance_escape_portals()  # Activation tick.
        for _ in range(9):
            game.move_character(smoker)
            game._advance_escape_portals()
            self.assertEqual(smoker.pos, [2, 2])

        game.move_character(smoker)
        game._advance_escape_portals()
        self.assertEqual(smoker.pos, [7, 10])
        self.assertEqual(game.escape_portals, [])

    def test_escape_portal_reserves_destination_while_caster_is_immobile(self):
        game = UltimateTestGame()
        smoker = make_character("Demon1", "A", (2, 2), 6)
        teammate = make_character("Chronicle", "A", (7, 9))
        game.chars = [smoker, teammate]

        self.assertTrue(
            game.execute_ai_ultimate(
                smoker,
                {"ultimate": "ESCAPE", "target": (7, 10)},
            )
        )
        game.move_character(smoker)

        self.assertEqual(smoker.pos, [2, 2])
        self.assertTrue(
            game._is_position_occupied(teammate, (7, 10), tuple(teammate.pos))
        )

    def test_monitor_spawns_two_distinct_trackers_and_reveals(self):
        game = UltimateTestGame()
        seeker = make_character("Leo", "A", (4, 2), 8)
        close_enemy = make_character("Demon1", "D", (4, 6))
        far_enemy = make_character("Chronicle", "D", (7, 9))
        game.chars = [seeker, close_enemy, far_enemy]

        self.assertTrue(game.execute_ai_ultimate(seeker, {"ultimate": "MONITOR"}))
        self.assertEqual(len(game.monitor_drones), 2)
        self.assertEqual(
            {drone.target_name for drone in game.monitor_drones},
            {close_enemy.name, far_enemy.name},
        )
        game._advance_monitor_drones()
        self.assertGreater(close_enemy.reveal_remaining, 0)
        self.assertGreater(far_enemy.reveal_remaining, 0)

    def test_tunnel_warns_five_ticks_then_blinds_for_three_active_ticks(self):
        game = UltimateTestGame()
        flash = make_character("Chronicle", "A", (4, 2), 5)
        flash.facing = "E"
        enemy_ahead = make_character("Demon1", "D", (4, 8))
        enemy_behind = make_character("Leo", "D", (4, 1))
        ally = make_character("something", "A", (5, 8))
        game.chars = [flash, enemy_ahead, enemy_behind, ally]

        self.assertTrue(game.execute_ai_ultimate(flash, {"ultimate": "TUNNEL"}))
        self.assertEqual(enemy_ahead.blind_remaining, 0)
        self.assertEqual(game.tunnel_bursts[0]["phase"], "warning")

        for _ in range(5):
            game._advance_tunnel_bursts()
            self.assertEqual(game.tunnel_bursts[0]["phase"], "warning")
            self.assertEqual(enemy_ahead.blind_remaining, 0)

        game._advance_tunnel_bursts()
        self.assertEqual(game.tunnel_bursts[0]["phase"], "active")
        self.assertEqual(enemy_ahead.blind_remaining, 15)
        self.assertEqual(enemy_behind.blind_remaining, 0)
        self.assertEqual(ally.blind_remaining, 0)
        self.assertEqual(flash.ultimate_points, 0)

        enemy_behind.pos = [5, 8]
        game._advance_tunnel_bursts()
        self.assertEqual(enemy_behind.blind_remaining, 15)
        game._advance_tunnel_bursts()
        self.assertEqual(game.tunnel_bursts[0]["remaining_ticks"], 0)
        game._advance_tunnel_bursts()
        self.assertEqual(game.tunnel_bursts, [])

    def test_monitor_drone_has_two_hundred_hp_and_can_be_shot(self):
        game = UltimateTestGame()
        seeker = make_character("Leo", "A", (4, 7), 8)
        shooter = make_character("Demon1", "D", (4, 2))
        shooter.facing = "E"
        shooter.accuracy = 1.0
        game.chars = [seeker, shooter]
        game.execute_ai_ultimate(seeker, {"ultimate": "MONITOR"})
        seeker.is_alive = False

        game._resolve_all_shots(engagements=[])

        self.assertEqual(sorted(drone.hp for drone in game.monitor_drones), [160, 200])

    def test_recon_revealed_enemy_can_be_shot_through_smoke_only(self):
        game = UltimateTestGame()
        shooter = make_character("something", "A", (4, 2))
        target = make_character("Demon1", "D", (4, 8))
        game.chars = [shooter, target]
        game.smokes = [{"cells": {(4, 5)}, "remaining_ticks": 10}]

        self.assertFalse(game.check_shot_line_of_sight(shooter, target))

        target.reveal_remaining = 3
        self.assertTrue(game.check_shot_line_of_sight(shooter, target))

        ally = make_character("Leo", "A", (4, 6))
        game.chars.append(ally)
        self.assertFalse(game.check_shot_line_of_sight(shooter, target))

        game.chars.remove(ally)
        game.grid[4, 4] = 1
        self.assertFalse(game.check_shot_line_of_sight(shooter, target))

    def test_orb_collection_takes_thirty_consecutive_ticks_and_can_cancel(self):
        game = UltimateTestGame()
        collector = make_character("Demon1", "A", (2, 2))
        game.chars = [collector]
        game.available_orbs = {(2, 2)}
        game.attacker_controller = FixedController("COLLECT_ORB")

        for _ in range(2):
            game.move_character(collector)
        self.assertEqual(collector.orb_collect_timer, 2)

        game.attacker_controller.action = "MOVE"
        game.move_character(collector)
        self.assertEqual(collector.orb_collect_timer, 0)
        self.assertIn((2, 2), game.available_orbs)

        game.attacker_controller.action = "COLLECT_ORB"
        for _ in range(ORB_COLLECT_REQUIRED_TICKS):
            game.move_character(collector)
        self.assertEqual(collector.ultimate_points, 2)
        self.assertNotIn((2, 2), game.available_orbs)

    def test_kill_awards_one_point_up_to_cost(self):
        game = UltimateTestGame()
        shooter = make_character("something", "A", (2, 2), 2)
        target = make_character("Demon1", "D", (2, 3))
        game.chars = [shooter, target]

        game._kill_character(shooter, target)
        self.assertEqual(shooter.ultimate_points, 3)
        self.assertEqual(game.match_stats[shooter.name]["ultimate_points"], 3)


if __name__ == "__main__":
    unittest.main()
