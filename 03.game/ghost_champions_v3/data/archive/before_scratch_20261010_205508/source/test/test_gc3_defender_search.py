import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from ghost_champions_v3.gc3_learn_defender_search import DefenderSearchPolicy, GhostChampionsV3SearchController
from ghost_champions_v3.gc3_learn_attacker_guard import OBS_DIM, GuardPolicy
from ghost_champions_v3.gc3_learn_defender_initial import GhostChampionsV3DefenderController
from ghost_champions_v3.test.test_gc3_spatial_policy import actor


class DefenderSearchTests(unittest.TestCase):
    def test_search_distribution_is_batched_and_keeps_legal_exploration(self):
        policy = DefenderSearchPolicy()
        with torch.no_grad():
            policy.actor.weight.zero_()
            policy.actor.bias.fill_(-100.)
            policy.actor.bias[0] = 100.
        mask = torch.zeros((2, 27), dtype=torch.bool)
        mask[:, :3] = True
        dist, _ = policy(torch.zeros((2, OBS_DIM)), mask)
        self.assertTrue(bool((dist.probs[:, 1:3] >= .05/3-1e-7).all()))
        self.assertTrue(bool((dist.probs[:, 3:] == 0).all()))

    def test_deployment_head_is_part_of_optimizer_and_checkpoint(self):
        policy = DefenderSearchPolicy()
        optimizer = torch.optim.Adam(policy.parameters())
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        self.assertTrue(all(id(p) in parameter_ids for p in policy.spatial.parameters()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"search.pt"
            policy.save(path, step=0)
            state = torch.get_rng_state()
            loaded, _ = DefenderSearchPolicy.load(path)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            for key, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            with self.assertRaises(ValueError):
                GuardPolicy.load(path)

    def test_setup_and_retake_stay_in_their_existing_phase_handlers(self):
        controller = object.__new__(GhostChampionsV3SearchController)
        char = actor(team="D")
        for state in (dict(defender_setup_active=True), dict(is_planted=True)):
            with patch.object(GhostChampionsV3DefenderController, "decide_move", return_value="phase-handler") as parent:
                self.assertEqual(controller.decide_move(char, state), "phase-handler")
                parent.assert_called_once_with(char, state)


if __name__ == "__main__":
    unittest.main()
