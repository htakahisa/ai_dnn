"""Validate retake cast masks by executing production projectiles."""

import unittest

import numpy as np
import torch

from abilities_los import AbilityLosMixin
from concon_v1.co1_attacker_common import bfs_distance_map, GORIGONS
from concon_v1.co1_guard_common import ABILITIES
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_retake_common import RetakeDQN, build_inputs, decode_action, wall_clear
from concon_v1.co1_retake_config import DEFAULT_ABILITY_DISTANCES
from concon_v1.co1_retake_projectiles import projectile_aim
from concon_v1.co1_retake_scenarios import get_scenario
from concon_v1.test.test_co1_defender_retake import actor


class ProjectileGame(AbilityLosMixin):
    def __init__(self, grid, char):
        self.grid = grid
        self.height, self.width = grid.shape
        self.chars = [char]
        self.flash_projectiles, self.recon_projectiles = [], []
        self.flash_bursts, self.recon_bursts = [], []


class RetakeProjectileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_masks_on_both_sites_only_enable_casts_that_land_on_markers(self):
        for site in ("L", "R"):
            scenario = get_scenario(site)
            controller = ConconDefenderRetakeController(
                site, model=RetakeDQN(scenario), ability_distance=DEFAULT_ABILITY_DISTANCES[site])
            for ability in ("FLASH", "RECON"):
                distances = [bfs_distance_map(scenario.grid, point) for point in scenario.points[ability]]
                candidates = np.zeros(scenario.grid.shape, dtype=bool)
                for distance in distances:
                    candidates |= (distance >= 0) & (distance <= DEFAULT_ABILITY_DISTANCES[site][ability])
                enabled = rejected = 0
                for r, c in zip(*np.where(candidates)):
                    char = actor(GORIGONS.players[0], (int(r), int(c)), ability=ability)
                    controller.reset_round()
                    state = dict(grid=scenario.grid, chars=[char], smoke_cells=[], battle_tick=1,
                                 detonate_timer=55, defender_defuse_info={},
                                 planted_pos=(7, 3) if site == "L" else (7, 38))
                    _, mask, context = build_inputs(controller, char, state)
                    start = (5 + ABILITIES.index(ability) * 3) * 8
                    actions = np.flatnonzero(mask[start:start + 24]) + start
                    if not len(actions):
                        rejected += 1
                    for action in actions:
                        enabled += 1
                        with self.subTest(site=site, ability=ability, source=char.pos, action=action):
                            effect = context["ability_effect_points"][(int(action) // 8 - 5) % 3]
                            self.assertIn(effect, scenario.points[ability])
                            self.assertTrue(wall_clear(scenario.grid, tuple(char.pos), effect))
                            game = ProjectileGame(scenario.grid, char)
                            setattr(char, ability.lower() + "_charges", 1)
                            payload = decode_action(action, char, context)[1]
                            self.assertTrue(game.execute_ai_ability(char, payload))
                            for _ in range(max(scenario.grid.shape)):
                                game._advance_flash_projectiles()
                                game._advance_recon_projectiles()
                                if not game.flash_projectiles and not game.recon_projectiles:
                                    break
                            if ability == "FLASH":
                                self.assertEqual(game.flash_bursts[0]["pos"], effect)
                            else:
                                cells = game.recon_bursts[0]["cells"]
                                from game_core import RECON_REVEAL_SIZE
                                radius = RECON_REVEAL_SIZE // 2
                                expected = {(rr, cc) for rr in range(effect[0] - radius, effect[0] + radius + 1)
                                            for cc in range(effect[1] - radius, effect[1] + radius + 1)
                                            if 0 <= rr < game.height and 0 <= cc < game.width}
                                self.assertEqual(cells, expected)
                self.assertGreater(enabled, 0)
                self.assertGreater(rejected, 0)

    def test_flash_flight_limit_blocks_distant_marker_despite_clear_los(self):
        grid = np.zeros((3, 25), dtype=np.int32)
        grid[:, 21] = 1
        self.assertTrue(wall_clear(grid, (1, 1), (1, 20)))
        self.assertIsNone(projectile_aim(grid, (1, 1), (1, 20), "FLASH"))
        self.assertIsNotNone(projectile_aim(grid, (1, 1), (1, 20), "RECON"))

    def test_near_wall_blocks_both_projectiles(self):
        grid = np.zeros((5, 9), dtype=np.int32)
        grid[:, 3] = 1
        for ability in ("FLASH", "RECON"):
            self.assertIsNone(projectile_aim(grid, (2, 1), (2, 7), ability))

    def test_actual_map_wall_between_caster_and_all_markers_disables_casts(self):
        for site, source in (("L", (22, 10)), ("R", (9, 31))):
            scenario = get_scenario(site)
            controller = ConconDefenderRetakeController(site, model=RetakeDQN(scenario), ability_distance=100)
            for ability in ("FLASH", "RECON"):
                with self.subTest(site=site, ability=ability):
                    char = actor(GORIGONS.players[0], source, ability=ability)
                    controller.reset_round()
                    state = dict(grid=scenario.grid, chars=[char], smoke_cells=[], battle_tick=1,
                                 detonate_timer=55, defender_defuse_info={},
                                 planted_pos=(7, 3) if site == "L" else (7, 38))
                    for point in scenario.points[ability]:
                        self.assertFalse(wall_clear(scenario.grid, source, point))
                        self.assertLessEqual(bfs_distance_map(scenario.grid, point)[source], 100)
                    _, mask, context = build_inputs(controller, char, state)
                    start = (5 + ABILITIES.index(ability) * 3) * 8
                    self.assertFalse(mask[start:start + 24].any())
                    self.assertEqual(context["ability_targets"], (None, None, None))


if __name__ == "__main__":
    unittest.main()
