import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import torch

from map_data import NEW_MAZE_STR
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.rl.controller import LearnedAttackerController
from ghost_champions_v2.geometry import los, watch_cells
from ghost_champions_v3 import BASELINE_CHECKPOINT
from ghost_champions_v3.gc3_map_attacker_guard import GUARD_MAP, validate_map
from ghost_champions_v3.gc3_learn_attacker_guard import GuardPlan, GuardPolicy, GhostChampionsV3AttackerController


def unit(name, position, ability="RECON", team="A", known=True):
    return NS(name=name, pos=list(position), ability_name=ability, team=team, is_alive=True,
              position_known=known, hp=100, max_hp=100, recon_charges=2, smoke_charges=1,
              flash_charges=1, dance_charges=1, facing="S", ultimate_points=0, ultimate_cost=7,
              has_spike=False, plant_timer=0)


class GuardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.grid = np.array([[int(x) for x in row] for row in NEW_MAZE_STR.strip().splitlines()])
        self.actor = unit("self", (7, 1))
        self.tactics = AttackerTactics(load_config())
        self.state = dict(grid=self.grid, chars=[self.actor], battle_tick=40, is_planted=True,
                          planted_pos=(8, 3), smoke_cells=set(), enemy_roster=[])
        self.tactics.observe(self.actor, self.state)
        self.plan = GuardPlan()

    def test_new_map_covers_every_actual_plant_tile_and_validates_preaim_rays(self):
        validate_map(self.grid)
        declared = {p for plan in GUARD_MAP.values() for p in plan["plants"]}
        self.assertEqual(declared, {tuple(map(int, p)) for p in np.argwhere(self.grid == 2)})

    def test_recon_is_available_after_plant_and_selected_cast_is_committed_once(self):
        choices, mask, _, teacher, _, _ = self.plan.choices(self.actor, self.state, self.tactics)
        self.assertTrue(mask[24])
        self.assertEqual(choices[teacher][1]["ability"], "RECON")
        self.assertEqual(self.plan.cast_ticks, {})
        self.plan.grid = self.grid
        self.plan.commit(self.actor, choices[24], self.tactics, (8, 3))
        self.assertEqual(self.plan.cast_ticks["self"], 40)
        _, mask, _, _, _, _ = self.plan.choices(self.actor, self.state, self.tactics)
        self.assertFalse(mask[24])

    def test_hidden_enemy_positions_do_not_change_guard_targets_or_masks(self):
        enemy = unit("hidden", (-1, -1), team="D", known=False)
        self.state["chars"].append(enemy)
        first = self.plan.choices(self.actor, self.state, self.tactics)
        enemy.pos = [7, 3]
        enemy.hp = 1
        second = self.plan.choices(self.actor, self.state, self.tactics)
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[-1], second[-1])

    def test_seal_blocks_all_ability_candidates_and_teacher_cast(self):
        self.actor.ability_seal_remaining = 3
        choices, mask, _, teacher, _, _ = self.plan.choices(self.actor, self.state, self.tactics)
        self.assertFalse(any(mask[i] for i in (2, 16, 17, 18, 19, 20, 21, 24, 25, 26)))
        command = choices[teacher][1]
        self.assertFalse(isinstance(command, dict) and command.get("ability"))

    def test_assignments_are_unique_and_recomputed_after_a_death(self):
        self.state["chars"] += [unit(f"ally{i}", (11, i+1)) for i in range(4)]
        for ally in self.state["chars"]:
            self.tactics.observe(ally, self.state)
            self.plan.choices(ally, self.state, self.tactics)
        self.assertEqual(len({p.position for p in self.plan.posts.values()}), 5)
        self.state["chars"][-1].is_alive = False
        for ally in self.state["chars"]:
            if ally.is_alive:
                self.tactics.observe(ally, self.state)
                self.plan.choices(ally, self.state, self.tactics)
        self.assertEqual(len(self.plan.posts), 4)

    def test_smoke_preserves_all_guard_to_defuse_sight_lines(self):
        self.actor.ability_name = "SMOKE"
        choices, _, _, _, _, _ = self.plan.choices(self.actor, self.state, self.tactics)
        self.assertIsNotNone(choices[21])
        center = choices[21][1]["target"]
        smoke = {(center[0]+dr, center[1]+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        for post in self.plan.posts.values():
            for target in watch_cells(self.grid, (8, 3)):
                if los(self.grid, post.position, target):
                    self.assertTrue(los(self.grid, post.position, target, smoke))

    def test_preplant_delegates_without_running_or_recording_guard(self):
        controller = object.__new__(GhostChampionsV3AttackerController)
        proposal = ([7, 1], "HOLD")
        self.state["is_planted"] = False
        with patch.object(LearnedAttackerController, "decide_move", return_value=proposal) as baseline:
            self.assertEqual(controller.decide_move(self.actor, self.state), proposal)
            baseline.assert_called_once_with(self.actor, self.state)

    def test_guard_checkpoint_load_preserves_rng_and_rejects_v2_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"guard.pt"
            model = GuardPolicy()
            model.save(path)
            rng = torch.get_rng_state().clone()
            loaded, _ = GuardPolicy.load(path)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(loaded.hidden, model.hidden)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            GuardPolicy.load(BASELINE_CHECKPOINT)


if __name__ == "__main__":
    unittest.main()
