"""Fixed recon parsing, wall endpoints, charge tracking, and battle execution."""

import contextlib
import io
import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from abilities_los import AbilityLosMixin
from concon_v1.co1_attacker_abilities import FixedReconPlan
from concon_v1.co1_attacker_scenarios import build_scenario, get_scenario, parse_strategy_points


class FixedReconTests(unittest.TestCase):
    def test_marker_validation_and_signature(self):
        for name in ("A1", "A2", "A3"):
            original = get_scenario(name)
            rows = original.strategy_map.strip().splitlines()
            r, c = next((r, c) for r, row in enumerate(rows) for c, value in enumerate(row)
                        if value == "0" and original.grid[r, c] != 1)
            rows[r] = rows[r][:c] + "R" + rows[r][c + 1:]
            kwargs = dict(max_candidate_bfs_distance=original.max_candidate_bfs_distance,
                          waypoint_order=original.waypoint_order)
            marked = build_scenario(name, "\n".join(rows), original.plant_side, **kwargs)
            self.assertEqual(marked.recon_points, ((r, c),))
            self.assertEqual(marked.waypoint_points, original.waypoint_points)
            self.assertEqual(marked.uppercase_markers, original.uppercase_markers)
            changed = build_scenario(name, "\n".join(rows), original.plant_side,
                                     recon_trigger_bfs_distance=7, **kwargs)
            self.assertNotEqual(marked.signature, changed.signature)
        with self.assertRaisesRegex(ValueError, "reserved"):
            parse_strategy_points("arR", "ar")
        original = get_scenario("A1")
        with self.assertRaisesRegex(ValueError, "recon point overlays a wall"):
            build_scenario("TEST", original.strategy_map.replace("1", "R", 1), "left")
        for distance in (-1, 0.5, True):
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                build_scenario("TEST", original.strategy_map, "left",
                               recon_trigger_bfs_distance=distance)

    def test_real_wall_endpoint_and_bfs_boundary(self):
        # R is on the floor immediately before a wall. A nearer wall blocks it.
        grid = np.zeros((3, 7), dtype=np.int32)
        grid[:, 6] = 1
        game = SimpleNamespace(grid=grid, height=3, width=7)
        game._line_cells = lambda start, end: AbilityLosMixin._line_cells(game, start, end)
        game._projectile_path = lambda start, goal: AbilityLosMixin._projectile_path(game, start, goal)
        caster = SimpleNamespace(name="recon", pos=[1, 0], team="A", is_alive=True,
                                 ability_name="RECON", recon_charges=2)
        game.chars = [caster]
        point = (1, 5)
        scenario = replace(get_scenario(), recon_points=(point,), recon_trigger_bfs_distance=4)
        plan = FixedReconPlan(scenario)
        self.assertIsNone(plan.choose(caster, game))
        plan = FixedReconPlan(replace(scenario, recon_trigger_bfs_distance=5))
        payload = plan.choose(caster, game)
        self.assertIsNotNone(payload)
        self.assertEqual(game._projectile_path(tuple(caster.pos), payload["target"])[-1], point)
        self.assertEqual(plan.choose(caster, game), payload)  # Rejected cast can retry.
        caster.recon_charges -= 1
        self.assertIsNone(plan.choose(caster, game))
        plan.reset_round()
        grid[:, 3] = 1
        ally = SimpleNamespace(name="ally", pos=point, team="A", is_alive=True)
        game.chars.append(ally)  # BFS trigger is satisfied despite the blocked shot.
        self.assertIsNone(plan.choose(caster, game))
        grid[:, 3] = 0
        self.assertIsNotNone(plan.choose(caster, game))
        # A point passed in flight is not an endpoint.
        plan = FixedReconPlan(replace(scenario, recon_points=((1, 3),),
                                     recon_trigger_bfs_distance=5))
        self.assertIsNone(plan.choose(caster, game))

    def test_battle_controller_launches_fixed_recon(self):
        with contextlib.redirect_stdout(io.StringIO()):
            from concon_v1.co1_battle_training import BattleRouteEnv
            env = BattleRouteEnv(seed=0, opponents=["gc_v1"])
            caster = next(char for char in env.attackers if char.ability_name == "RECON")
            game = env.game
            point = next(game._projectile_path(tuple(caster.pos), (r, c))[-1]
                         for r in range(game.height) for c in range(game.width)
                         if game.grid[r, c] != 1
                         and len(game._projectile_path(tuple(caster.pos), (r, c))) > 1)
            scenario = replace(env.scenario, recon_points=(point,), recon_trigger_bfs_distance=1000)
            env.controller.fixed_recons = FixedReconPlan(scenario)
            env.route_controller.scenario = scenario
            env.controller.retrieve_controller.allow_recon = False
            charges = caster.recon_charges
            env.step(actions=[4] * 5)
        self.assertEqual(caster.recon_charges, charges - 1)
        projectiles = [p for p in game.recon_projectiles if p["owner"] == caster.name]
        if projectiles:
            self.assertEqual(projectiles[0]["path"][-1], point)
        else:
            self.assertTrue(any(b["owner"] == caster.name for b in game.recon_bursts))
        self.assertTrue(point in env.controller.fixed_recons.pending
                        or point in env.controller.fixed_recons.used_points)


if __name__ == "__main__":
    unittest.main()
