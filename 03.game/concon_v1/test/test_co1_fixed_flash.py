"""Fixed flash activation, projectile reachability, and real ability execution."""

import contextlib
import io
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from concon_v1.co1_attacker_abilities import FixedFlashPlan
from concon_v1.co1_attacker_scenarios import (
    SCENARIOS, _load_scenario, build_scenario, get_scenario, parse_strategy_points,
)


class FixedFlashTests(unittest.TestCase):
    def test_map_parsing_validation_and_signature(self):
        for name in ("A1", "A2", "A3"):
            original = get_scenario(name)
            rows = original.strategy_map.replace("U", "0").strip().splitlines()
            r, c = next((r, c) for r, row in enumerate(rows) for c, value in enumerate(row)
                        if value == "0" and original.grid[r, c] != 1)
            rows[r] = rows[r][:c] + "U" + rows[r][c + 1:]
            kwargs = dict(max_candidate_bfs_distance=original.max_candidate_bfs_distance,
                          waypoint_order=original.waypoint_order,
                          smoke_trigger_bfs_distance=original.smoke_trigger_bfs_distance)
            marked = build_scenario(name, "\n".join(rows), original.plant_side,
                                    flash_trigger_bfs_distance=4, **kwargs)
            self.assertEqual(marked.flash_points, ((r, c),))
            self.assertEqual(marked.waypoint_points, original.waypoint_points)
            self.assertEqual(marked.uppercase_markers, original.uppercase_markers)
            changed = build_scenario(name, "\n".join(rows), original.plant_side,
                                     flash_trigger_bfs_distance=5, **kwargs)
            self.assertNotEqual(marked.signature, changed.signature)
        with self.assertRaisesRegex(ValueError, "reserved"):
            parse_strategy_points("auU", "au")
        original = get_scenario("A1")
        with self.assertRaisesRegex(ValueError, "flash point overlays a wall"):
            build_scenario("TEST", original.strategy_map.replace("1", "U", 1), "left")
        for distance in (-1, 0.5, True):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                build_scenario("TEST", original.strategy_map, "left",
                               flash_trigger_bfs_distance=distance)

    def test_scenarios_load_independent_flash_and_smoke_distances(self):
        settings = {name: replace(value, flash_trigger_bfs_distance=index)
                    for index, (name, value) in enumerate(SCENARIOS.items())}
        _load_scenario.cache_clear()
        try:
            with patch.dict(SCENARIOS, settings):
                for name, value in settings.items():
                    loaded = get_scenario(name)
                    self.assertEqual(loaded.flash_trigger_bfs_distance, value.flash_trigger_bfs_distance)
                    self.assertEqual(loaded.smoke_trigger_bfs_distance, value.smoke_trigger_bfs_distance)
        finally:
            _load_scenario.cache_clear()

    def test_bfs_boundary_dead_allies_retries_and_projectile_reachability(self):
        point = (0, 0)
        scenario = replace(get_scenario("A1"), flash_points=(point,), flash_trigger_bfs_distance=3)
        plan = FixedFlashPlan(scenario)
        flasher = SimpleNamespace(name="flash", pos=[0, 2], team="A", is_alive=True,
                                  ability_name="FLASH", flash_charges=2)
        ally = SimpleNamespace(name="ally", pos=[2, 1], team="A", is_alive=False)
        enemy = SimpleNamespace(name="enemy", pos=[0, 0], team="D", is_alive=True)
        path = [(0, 2), (1, 2), point]
        game = SimpleNamespace(
            grid=np.array([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]]),
            chars=[flasher, ally, enemy], _projectile_path=lambda start, goal: path)
        self.assertIsNone(plan.choose(flasher, game))
        ally.is_alive = True
        ally.pos = [0, 4]
        self.assertIsNone(plan.choose(flasher, game))
        ally.pos = [2, 1]
        expected = {"ability": "FLASH", "target": point}
        self.assertEqual(plan.choose(flasher, game), expected)
        self.assertEqual(plan.choose(flasher, game), expected)  # A rejected request can retry.
        flasher.flash_charges -= 1
        self.assertIsNone(plan.choose(flasher, game))  # Successful cast uses the point once.
        plan.reset_round()
        path = [(0, 2), (1, 2)]  # Wall stops the projectile before U.
        self.assertIsNone(plan.choose(flasher, game))
        path = [(0, 2)] * 100 + [point]  # Maximum flight time also limits reach.
        self.assertIsNone(plan.choose(flasher, game))
        path = [(0, 2), point]
        self.assertEqual(plan.choose(flasher, game), expected)
        flasher.flash_charges = 0
        self.assertIsNone(plan.choose(flasher, game))

    def test_battle_controller_launches_fixed_flash(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from concon_v1.co1_battle_training import BattleRouteEnv
            env = BattleRouteEnv(seed=0, opponents=["gc_v1"])
            flasher = next(char for char in env.attackers if char.ability_name == "FLASH")
            point = next(pos for pos in env.scenario.attacker_spawns
                         if tuple(flasher.pos) != pos
                         and env.game._projectile_path(tuple(flasher.pos), pos)[-1] == pos)
            scenario = replace(env.scenario, flash_points=(point,), flash_trigger_bfs_distance=0)
            env.controller.fixed_flashes = FixedFlashPlan(scenario)
            env.route_controller.scenario = scenario
            env.controller.retrieve_controller.allow_flash = False
            charges = flasher.flash_charges
            env.step(actions=[4] * 5)
        self.assertEqual(flasher.flash_charges, charges - 1)
        self.assertTrue(any(tuple(burst["pos"]) == point and burst["owner"] == flasher.name
                            for burst in env.game.flash_bursts))
        self.assertTrue(point in env.controller.fixed_flashes.pending
                        or point in env.controller.fixed_flashes.used_points)


if __name__ == "__main__":
    unittest.main()
