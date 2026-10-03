"""Fixed smoke map parsing and BFS activation without running training."""

import unittest
import contextlib
import io
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from concon_v1.co1_attacker_abilities import FixedSmokePlan
from concon_v1.co1_attacker_scenarios import build_scenario, get_scenario
from concon_v1.co1_attacker_scenarios import parse_strategy_points


class FixedSmokeTests(unittest.TestCase):
    def test_battle_controller_casts_fixed_smoke_with_real_charge(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from concon_v1.co1_battle_training import BattleRouteEnv
            original = get_scenario("A1")
            point = original.attacker_spawns[0]
            scenario = replace(original, smoke_points=(point,), smoke_trigger_bfs_distance=0)
            env = BattleRouteEnv(seed=0, opponents=["gc_v1"], map_name=scenario)
            smoker = next(char for char in env.attackers if char.ability_name == "SMOKE")
            self.assertEqual(smoker.smoke_charges, 1)
            env.step(actions=[4] * 5)
        self.assertEqual(smoker.smoke_charges, 0)
        self.assertTrue(any(tuple(smoke["center"]) == point and smoke["team"] == "A"
                            for smoke in env.game.smokes))

    def test_maps_accept_smoke_separately_from_waypoints(self):
        for name in ("A1", "A2", "A3"):
            original = get_scenario(name)
            rows = original.strategy_map.strip().splitlines()
            point = next((r, c) for r, row in enumerate(rows)
                         for c, value in enumerate(row) if value == "0" and original.grid[r, c] != 1)
            r, c = point
            rows[r] = rows[r][:c] + "S" + rows[r][c + 1:]
            marked = build_scenario(name, "\n".join(rows), original.plant_side,
                                    max_candidate_bfs_distance=original.max_candidate_bfs_distance,
                                    waypoint_order=original.waypoint_order,
                                    smoke_trigger_bfs_distance=4)
            self.assertEqual(marked.smoke_points, tuple(sorted((*original.smoke_points, point))))
            self.assertEqual(marked.waypoint_points, original.waypoint_points)
            self.assertEqual(marked.uppercase_markers, original.uppercase_markers)
            self.assertEqual(marked.obs_dim, original.obs_dim)
            self.assertNotEqual(marked.signature, original.signature)

    def test_invalid_smoke_wall_and_distance(self):
        with self.assertRaisesRegex(ValueError, "reserved"):
            parse_strategy_points("asS", "as")
        original = get_scenario("A1")
        with self.assertRaisesRegex(ValueError, "smoke point overlays a wall"):
            build_scenario("TEST", original.strategy_map.replace("1", "S", 1), "left")
        for distance in (-1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                build_scenario("TEST", original.strategy_map, "left",
                               smoke_trigger_bfs_distance=distance)

    def test_bfs_boundary_dead_enemies_unreachable_and_real_charge(self):
        grid = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 0, 0, 1, 0]])
        scenario = replace(get_scenario("A1"), smoke_points=((0, 0),),
                           smoke_trigger_bfs_distance=3)
        plan = FixedSmokePlan(scenario)
        smoker = SimpleNamespace(pos=[0, 2], team="A", is_alive=True,
                                 ability_name="SMOKE", smoke_charges=1)
        ally = SimpleNamespace(pos=[1, 0], team="A", is_alive=False)
        enemy = SimpleNamespace(pos=[0, 0], team="D", is_alive=True)
        game = SimpleNamespace(grid=grid, chars=[smoker, ally, enemy], smokes=[])
        self.assertIsNone(plan.choose(smoker, game))  # Geometrically close; BFS distance is six.
        ally.is_alive = True
        ally.pos = [0, 4]  # Unreachable cells have distance -1.
        self.assertIsNone(plan.choose(smoker, game))
        ally.pos = [2, 1]  # Exactly three BFS steps from S.
        expected = {"ability": "SMOKE", "target": (0, 0)}
        self.assertEqual(plan.choose(smoker, game), expected)
        self.assertEqual(plan.choose(smoker, game), expected)  # Retry a rejected request.
        game.smokes.append({"center": (0, 0), "team": "A"})
        self.assertIsNone(plan.choose(smoker, game))
        game.smokes.clear()
        self.assertIsNone(plan.choose(smoker, game))  # No recast after expiration.
        plan.reset_round()
        self.assertEqual(plan.choose(smoker, game), expected)
        smoker.smoke_charges = 0
        self.assertIsNone(plan.choose(smoker, game))


if __name__ == "__main__":
    unittest.main()
