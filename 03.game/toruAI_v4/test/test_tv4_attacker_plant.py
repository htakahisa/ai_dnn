"""Public-only plant action inputs, route ownership, rewards and checkpoints."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import deque
import copy
import tempfile
import unittest

import numpy as np
import torch

from frc_v1.actions import build_masks, validate_action
from toruAI_v4.test.test_tv4_attacker_analysis import world
from toruAI_v4.tv4_learn_attacker_analysis import AttackerEncoder, AttackerAnalysisModel
from toruAI_v4.tv4_learn_attacker_plant import (
    PlantDQN, OBS_DIM, ACTION_DIM, PLANT_ACTION, policy_schema, learn_plant, load_plant,
)
from toruAI_v4.tv4_attacker_plant_controller import ToruV4AttackerPlantController
from toruAI_v4.tv4_train_attacker_plant import save_latest, first_contact_preaim, best_rank


class PlantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self, game):
        encoder = AttackerEncoder(game)
        analysis = AttackerAnalysisModel(len(encoder.fields), len(encoder.route_fields), len(game.names))
        analysis.trained_rounds = 60
        analysis.eval()
        analysis.requires_grad_(False)
        controller = ToruV4AttackerPlantController(game, analysis, PlantDQN(), training=True)
        controller.teacher_probability = 1.
        controller.set_game(game)
        return controller

    def test_hidden_positions_do_not_change_inputs_route_or_actions(self):
        game = world()
        first = self.controller(game)
        first.prepare_team_tick()
        features = {name: inputs.observation.copy() for name, inputs in first.inputs.items()}
        actions = copy.deepcopy(first.actions)
        route = first.route
        for i, enemy in enumerate(game.chars[5:]):
            enemy.pos = [2, 6 + i]
        game.target_plant_pos = (7, 40)
        second = ToruV4AttackerPlantController(game, first.analysis, first.policy, training=True)
        second.teacher_probability = 1.
        second.set_game(game)
        second.prepare_team_tick()
        self.assertEqual(route, second.route)
        self.assertEqual(actions, second.actions)
        for name, observation in features.items():
            np.testing.assert_array_equal(observation, second.inputs[name].observation)
            self.assertEqual(len(observation), OBS_DIM)

    def test_legal_actions_and_teacher_follow_analysis_site(self):
        game = world()
        controller = self.controller(game)
        controller.prepare_team_tick()
        snapshot = controller.snapshot
        masks = build_masks(snapshot)
        for name, (_, inputs, ally) in controller.plans.items():
            self.assertTrue(inputs.mask[inputs.teacher])
            self.assertEqual(len(inputs.actions), ACTION_DIM)
            for i in np.flatnonzero(inputs.mask):
                validate_action(snapshot, masks, ally.slot, inputs.actions[i])
            self.assertIn(inputs.goal, set(controller.route.cells) | set(game.sites[controller.route.site])
                          | set(controller.attack_plan.scout_goals.values()))
        self.assertFalse(any(inputs.mask[PLANT_ACTION] for inputs in controller.inputs.values()))

    def test_on_site_teacher_plants_and_postplant_has_no_decisions(self):
        game = world()
        holder = next(c for c in game.chars if c.team == "A" and c.has_spike)
        controller = self.controller(game)
        controller.prepare_team_tick()
        holder.pos = list(controller.route.cells[-1])
        game.battle_tick += 1
        controller.prepare_team_tick()
        name = str(getattr(holder, "base_name", holder.name))
        self.assertEqual(controller.inputs[name].teacher, PLANT_ACTION)
        self.assertTrue(any(controller.inputs[name].mask[i] and action.kind == "STAY"
                            for i, action in enumerate(controller.inputs[name].actions)))
        game.is_planted = True
        game.planted_pos = tuple(holder.pos)
        controller.decisions = {}
        controller.decide_move(holder, {"is_planted": True})
        self.assertEqual(controller.decisions, {})

    def test_unseen_enemies_do_not_force_forward_movement(self):
        from dataclasses import replace
        from frc_v1.perception import FrcPerceptionBuilder
        game = world()
        controller = self.controller(game)
        snapshot = replace(FrcPerceptionBuilder("A").build(game), sightings=(), visible_cells=())
        controller.sensor.build = lambda _game: snapshot
        controller.prepare_team_tick()
        for inputs in controller.inputs.values():
            self.assertTrue(any(inputs.mask[i] and action.kind == "STAY"
                                for i, action in enumerate(inputs.actions)))

    def test_recent_public_contact_can_stop_even_without_current_sightings(self):
        from dataclasses import replace
        from frc_v1.perception import FrcPerceptionBuilder
        game = world()
        game.chars[0].pos = [14, 32]
        game.chars[0].facing = "W"
        game.chars[0].flash_charges = 0
        controller = self.controller(game)
        snapshot = replace(FrcPerceptionBuilder("A").build(game), sightings=(), visible_cells=())
        controller.sensor.build = lambda _game: snapshot
        controller.analysis_encoder.history.tracks[0] = ((14, 36), 0, (0., 0.))
        controller.prepare_team_tick()
        inputs = controller.inputs[game.chars[0].name]
        self.assertEqual(inputs.combat.reason, "stop_shoot")
        self.assertTrue(inputs.mask[inputs.teacher])
        self.assertEqual(inputs.actions[inputs.teacher].kind, "STAY")

    def test_shared_dqn_learning_and_checkpoint_compatibility(self):
        game = world()
        controller = self.controller(game)
        controller.prepare_team_tick()
        inputs = next(iter(controller.inputs.values()))
        transition = [inputs.observation.astype(np.float16), inputs.teacher, 1.,
            np.zeros(OBS_DIM, np.float16), np.ones(ACTION_DIM, bool), 1., inputs.teacher, inputs.mask.copy()]
        replay = deque([copy.deepcopy(transition) for _ in range(4)])
        policy = controller.policy
        target = copy.deepcopy(policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
        loss = learn_plant(policy, target, optimizer, replay, np.random.default_rng(0), 2, 4)
        self.assertTrue(np.isfinite(loss))
        reference = {"set": 1, "evaluation": {"selected": {"plant_rate": .5, "plants": 6, "rounds": 12}}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            save_latest(path, policy, target, optimizer, replay, 1, "fnatic_v3", policy_schema(game), {},
                        "frozen-analysis-hash", np.random.default_rng(0), reference)
            restored, state = load_plant(path, game, "fnatic_v3", "frozen-analysis-hash")
            self.assertEqual(state["completed_sets"], 1)
            self.assertEqual(state["last_evaluation"], reference)
            np.testing.assert_allclose(restored(torch.tensor(inputs.observation).unsqueeze(0)).detach().numpy(),
                                       policy(torch.tensor(inputs.observation).unsqueeze(0)).detach().numpy())
            with self.assertRaises(ValueError):
                load_plant(path, game, "fnatic_v3", "changed-analysis")

    def test_preaim_only_rewards_correct_geometry_and_success_has_priority(self):
        game = world()
        origin = (22, 18)
        target = (22, 21)
        self.assertGreater(first_contact_preaim(game, origin, target, "E"), 0.)
        self.assertEqual(first_contact_preaim(game, origin, target, "W"), 0.)
        low = {"plant_rate": .5, "mean_alive": 5., "mean_damage": 0., "ability_reserve_rate": 1.,
               "mean_ability_uses": 0., "mean_plant_ticks": 20.}
        high = {**low, "plant_rate": .75, "ability_reserve_rate": 0., "mean_damage": 100.}
        self.assertGreater(best_rank(high), best_rank(low))


if __name__ == "__main__":
    unittest.main()
