from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from map_data import NEW_MAZE_STR
from ghost_champions_v2.config import load_config
from ghost_champions_v2.tactics import AttackerTactics
from ghost_champions_v2.rl.observation import ObservationEncoder
from ghost_champions_v2.defender_macro_v1.controller import GhostChampionsV2DefenderController
from ghost_champions_v3 import gc3_map_attacker_postplant as post_map
from ghost_champions_v3 import gc3_map_defender_initial as initial_map
from ghost_champions_v3.gc3_learn_attacker_guard import GhostChampionsV3AttackerController, GuardPlan
from ghost_champions_v3.gc3_learn_defender_initial import GhostChampionsV3DefenderController
from ghost_champions_v3.gc3_spatial_policy import (
    SpatialCandidatePolicy, parse_map, public_context, sampled_distribution, phase_settings,
)


def actor(name="self", point=(7, 1), team="A"):
    return NS(name=name, team=team, pos=list(point), is_alive=True, hp=100, max_hp=100,
              ability_name="RECON", recon_charges=2, facing="S", plant_timer=0,
              has_spike=False, ultimate_points=0, ultimate_cost=7)


class SpatialPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261010)
        self.grid = np.array([[int(ch) for ch in row] for row in NEW_MAZE_STR.strip().splitlines()])
        self.char = actor()
        self.state = dict(grid=self.grid, chars=[self.char], battle_tick=30, planted_pos=(8, 3),
                          is_planted=True, smoke_cells=set(), enemy_roster=[])

    def test_both_maps_use_one_grid_and_match_arena_geometry(self):
        for text in (post_map.NEW_MAZE_STR, initial_map.DEFENDER_SETUP_POSITION_MAP_GC_STR):
            grid = parse_map(text)
            self.assertEqual(grid.shape, self.grid.shape)
            np.testing.assert_array_equal(grid == 1, self.grid == 1)
            self.assertTrue(np.any(grid == 5))
            self.assertTrue(np.any(grid == 6))
        self.assertEqual(initial_map.validate_against_map(NEW_MAZE_STR), [])

    def test_decorative_digits_are_floor_and_do_not_create_candidates(self):
        np.testing.assert_array_equal(parse_map("1234560"), [[1, 0, 0, 0, 5, 6, 0]])
        with patch.object(post_map, "NEW_MAZE_STR", "1234560"):
            self.assertEqual(post_map.get_gc_postplant_position_candidates(), [(0, 4)])
            self.assertEqual(post_map.get_gc_postplant_watch_candidates(), [(0, 5)])
        with patch.object(initial_map, "DEFENDER_SETUP_POSITION_MAP_GC_STR", "1234560"):
            self.assertEqual(initial_map.get_gc_setup_position_candidates(), [(0, 4)])
            self.assertEqual(initial_map.get_gc_setup_watch_candidates(), [(0, 5)])

    def test_distribution_keeps_unmarked_floor_selectable_and_masks_walls_only(self):
        policy = SpatialCandidatePolicy()
        _, record = policy.sample(self.char, self.state, post_map.NEW_MAZE_STR, kind="position", phase="postplant")
        distribution = policy.distribution(record["context"], record["features"], record["mask"], kind="position", phase="postplant")
        marker_grid = parse_map(post_map.NEW_MAZE_STR).reshape(-1)
        self.assertTrue(bool((distribution.probs[torch.tensor(marker_grid == 0)] > 0).any()))
        self.assertTrue(bool((distribution.probs[torch.tensor(marker_grid == 1)] == 0).all()))
        self.assertGreater(distribution.probs[torch.tensor(marker_grid == 5)].mean(), distribution.probs[torch.tensor(marker_grid == 0)].mean())

    def test_defender_distribution_has_more_entropy_and_retains_exploration(self):
        logits = torch.tensor([8., 0., -1., 100.])
        mask = torch.tensor([True, True, True, False])
        distributions = []
        for phase in ("postplant", "defender"):
            temperature, mix, _, _ = phase_settings(phase)
            distributions.append(sampled_distribution(logits, mask, temperature=temperature, uniform_mix=mix))
        self.assertGreater(distributions[1].entropy(), distributions[0].entropy())
        self.assertGreaterEqual(float(distributions[1].probs[2]), .15/3-1e-7)
        self.assertEqual(float(distributions[1].probs[3]), 0.)

    def test_repeated_samples_do_not_collapse_to_largest_logit(self):
        distribution = sampled_distribution(torch.tensor([1., 0.]), [True, True])
        choices = distribution.sample((200,))
        self.assertEqual(set(choices.tolist()), {0, 1})

    def test_recorded_log_prob_preserves_gradient_to_learned_spatial_head(self):
        policy = SpatialCandidatePolicy()
        _, record = policy.sample(self.char, self.state, post_map.NEW_MAZE_STR, kind="position", phase="postplant")
        logp, _ = policy.log_prob(record)
        self.assertAlmostEqual(float(logp.detach()), record["logp"], places=5)
        (-logp).backward()  # Derivative check only: no optimizer or weight update.
        gradient = policy.position[-1].weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_guard_uses_sample_even_for_a_historical_greedy_caller(self):
        controller = object.__new__(GhostChampionsV3AttackerController)
        controller.guard_mode = "greedy"
        controller.guard_plan = GuardPlan()
        controller.tactics = AttackerTactics(load_config())
        controller.encoder = ObservationEncoder()
        controller.guard_recorder = None
        logits = torch.zeros((1, 27))
        logits[0, 0] = 10
        distribution = torch.distributions.Categorical(logits=logits)
        distribution.sample = Mock(return_value=torch.tensor([8]))
        controller.guard_policy = Mock(return_value=(distribution, torch.zeros(1)))
        result = controller.decide_move(self.char, self.state)
        distribution.sample.assert_called_once()
        self.assertEqual(result, (self.char.pos, {"facing": "N"}))

    def test_setup_context_does_not_read_hidden_enemy_positions_or_state(self):
        enemy = actor("enemy", (7, 4), "D")
        self.state.update(defender_setup_active=True, chars=[self.char, enemy])
        first = public_context(self.char, self.state)
        enemy.pos, enemy.hp, enemy.is_alive = [1, 1], 1, False
        np.testing.assert_array_equal(first, public_context(self.char, self.state))

    def test_initial_controller_accepts_region_without_forcing_exact_destination(self):
        controller = object.__new__(GhostChampionsV3DefenderController)
        controller.initial_policy = SpatialCandidatePolicy()
        controller.initial_checkpoint = None
        controller.initial_map_key = initial_map.DEFENDER_SETUP_POSITION_MAP_GC_STR
        controller.initial_goals = {"self": (2, 19)}
        controller.initial_aims = {"self": (2, 25)}
        controller.initial_records = {}
        controller.macro = NS()
        self.char.team, self.char.pos = "D", [2, 18]
        self.state.update(defender_setup_active=True, is_planted=False)
        result = controller.decide_move(self.char, self.state)
        self.assertEqual(result[0], [2, 18])
        self.assertEqual(result[1], {"facing": "E"})

    def test_initial_controller_delegates_live_phase_to_existing_defense(self):
        controller = object.__new__(GhostChampionsV3DefenderController)
        with patch.object(GhostChampionsV2DefenderController, "decide_move", return_value=([7, 1], "HOLD")) as baseline:
            controller.decide_move(self.char, self.state)
            baseline.assert_called_once_with(self.char, self.state)


if __name__ == "__main__":
    unittest.main()
