"""Regression checks for launch legality, normalized state features and plant entry."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from abilities_los import AbilityLosMixin
from ghost_champions_v3.gc3_engine_actions import legal_actions, projectile_targets, MODES
from ghost_champions_v3.gc3_observation import OBS_DIM, CONTEXT_WIDTH, BOARD_CHANNELS, CELLS
from ghost_champions_v3.gc3_policy import FreshPolicy, LEGACY_SCHEMA_HASH, SCHEMA_HASH
from ghost_champions_v3.gc3_rollout import setup_postplant, NoPostplantEntry, phase_start_snapshot
from ghost_champions_v3.test.test_gc3_scratch import fixture


class TrainingFixesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_projectile_mask_matches_actual_engine_for_every_floor_target(self):
        rng = np.random.default_rng(84)
        for _ in range(4):
            grid = (rng.random((7, 9)) < .3).astype(int)
            engine = AbilityLosMixin()
            engine.grid, engine.height, engine.width = grid, *grid.shape
            engine.flash_projectiles, engine.recon_projectiles = [], []
            for origin in [tuple(p) for p in np.argwhere(grid != 1)][::5]:
                mask = projectile_targets(grid, origin)
                for ability in ("FLASH", "RECON"):
                    char = SimpleNamespace(name="caster", team="A", is_alive=True,
                        pos=list(origin), ability_name=ability, flash_charges=1, recon_charges=1)
                    state = dict(grid=grid, chars=[char])
                    np.testing.assert_array_equal(legal_actions(char, state)[0]["target"], mask)
                    for target in map(tuple, np.argwhere(grid != 1)):
                        char.flash_charges = char.recon_charges = 1
                        expected = len(engine._projectile_path(origin, target)) > 1
                        actual = engine.execute_ai_ability(char, dict(ability=ability, target=target))
                        self.assertEqual(bool(mask[np.ravel_multi_index(target, grid.shape)]), expected)
                        self.assertEqual(actual, expected)

    def test_blocked_flash_does_not_block_neon_or_force_another_target(self):
        grid = np.zeros((5, 7), dtype=int)
        grid[2, 2] = 1
        char = SimpleNamespace(name="caster", team="A", is_alive=True, pos=[2, 1],
            ability_name="FLASH", flash_charges=1, ultimate_name="NEON",
            ultimate_points=7, ultimate_cost=7)
        masks, _ = legal_actions(char, dict(grid=grid, chars=[char]))
        blocked = np.ravel_multi_index((2, 5), grid.shape)
        impact = np.ravel_multi_index((1, 3), grid.shape)
        self.assertFalse(masks["target"][blocked])
        self.assertTrue(masks["ultimate_target"][blocked])
        # First step is clear: do not forbid an eventual wall impact.
        self.assertTrue(masks["target"][impact])
        self.assertTrue(masks["mode"][MODES.index("ULTIMATE")])

    def test_state_branch_keeps_gradient_when_static_map_is_large(self):
        policy = FreshPolicy.fresh("attacker", 53)
        self.assertFalse(any(isinstance(layer, torch.nn.Tanh) for layer in policy.modules()))
        obs = torch.ones(2, OBS_DIM)
        obs[:, CONTEXT_WIDTH:CONTEXT_WIDTH+BOARD_CHANNELS*CELLS] *= 1e5
        obs[1, 0] = .1
        obs.requires_grad_()
        features = policy.features(obs)
        self.assertGreater(float((features[0]-features[1]).abs().max().detach()), 1e-3)
        policy.critic(features).sum().backward()
        self.assertTrue(bool(torch.isfinite(obs.grad).all()))
        state_gradient = torch.cat((obs.grad[:, :CONTEXT_WIDTH],
                                   obs.grad[:, CONTEXT_WIDTH+BOARD_CHANNELS*CELLS:]), dim=-1)
        self.assertGreater(float(state_gradient.abs().sum()), .01)

    def test_sampled_ultimate_uses_its_own_target_mask_and_ppo_likelihood(self):
        char, state = fixture()
        char.ability_name, char.flash_charges = "FLASH", 1
        char.ultimate_name, char.ultimate_points, char.ultimate_cost = "NEON", 7, 7
        masks, _ = legal_actions(char, state)
        index = int(np.flatnonzero(masks["ultimate_target"] & ~masks["target"])[0])
        policy = FreshPolicy.fresh("attacker", 35)
        # Force this branch only inside the test to exercise the conditional mask.
        with torch.no_grad():
            policy.mode.weight.zero_()
            policy.mode.bias.fill_(-1000)
            policy.mode.bias[MODES.index("ULTIMATE")] = 1000
            policy.spatial.target.weight.zero_()
            policy.spatial.target.bias.fill_(-1000)
            policy.spatial.target.bias[index] = 1000
        command, record = policy.select(char, state)
        self.assertEqual(record["selections"]["target"], index)
        self.assertEqual(command[1]["ultimate"], "NEON")
        self.assertTrue(record["masks"]["target"][index])
        logp, _, _ = policy.evaluate_records([record])
        self.assertAlmostEqual(float(logp[0].detach()), record["logp"], places=5)

    def test_legacy_weights_are_preserved_for_inference_and_rejected_for_resume(self):
        torch.manual_seed(11)
        rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            old = FreshPolicy("attacker", legacy=True)
        with TemporaryDirectory() as temp:
            path = Path(temp)/"old.pt"
            old.save(path, trained=True, step=80071)
            with self.assertRaises(ValueError):
                FreshPolicy.load(path)
            restored, metadata = FreshPolicy.load(path, allow_legacy=True)
            self.assertEqual(metadata["schema_hash"], LEGACY_SCHEMA_HASH)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(old.parameters(), restored.parameters())))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            new = FreshPolicy.fresh("attacker", 11)
            self.assertEqual(new.schema_hash, SCHEMA_HASH)
            self.assertNotEqual(new.schema_hash, restored.schema_hash)

    def test_actual_plant_entry_keeps_hp_positions_charges_and_effect_objects(self):
        char = SimpleNamespace(name="survivor", team="A", pos=[3, 8], hp=17,
                               is_alive=True, flash_charges=0, facing="NE")
        dead = SimpleNamespace(name="dead", team="D", pos=[4, 9], hp=0,
                               is_alive=False, recon_charges=1, facing="W")
        smoke = {"cells": {(4, 7)}, "remaining_ticks": 3}
        game = SimpleNamespace(chars=[char, dead], smokes=[smoke], battle_tick=34,
            detonate_timer=55, is_planted=False, round_over=False, match_over=False)
        def tick():
            game.battle_tick += 1
            game.is_planted = True
            game.planted_pos = (3, 8)
            game.detonate_timer = 54
        game._simulate_tick = tick
        before_char = vars(char).copy()
        self.assertEqual(setup_postplant(game, 31), (3, 8))
        self.assertEqual(vars(char), before_char)
        self.assertIs(game.smokes[0], smoke)
        self.assertEqual(game.detonate_timer, 54)
        self.assertFalse(dead.is_alive)
        snapshot = phase_start_snapshot(game)
        self.assertEqual(snapshot["chars"][0]["hp"], 17)
        self.assertEqual(snapshot["chars"][0]["flash_charges"], 0)

    def test_failed_preplant_never_synthesizes_a_start(self):
        game = SimpleNamespace(is_planted=False, round_over=False, match_over=False)
        def tick():
            game.round_over = True
        game._simulate_tick = tick
        with self.assertRaises(NoPostplantEntry):
            setup_postplant(game, 41)
        self.assertFalse(game.is_planted)


if __name__ == "__main__":
    unittest.main()
