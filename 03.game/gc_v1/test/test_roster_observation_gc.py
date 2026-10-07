"""Opponent features reach GC networks and remain learnable after migration."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gc_v1"))

from gc_v1.roster_observation_gc import (
    ENEMY_ROSTER_DIM, PLAYER_FEATURE_DIM, ROSTER_METADATA,
    append_enemy_roster, base_checkpoint_dim, enemy_roster_features,
    expand_roster_state,
)
from roster_utils import roster_information
from game_core import Character


class RosterObservationTests(unittest.TestCase):
    def test_full_public_roster_matches_training_and_ignores_hidden_state(self):
        players = [Character(n, "D", [1, i], "white", "green")
                   for i, n in enumerate(("Leo", "Boaster", "Derke", "Chronicle", "Alfajer"))]
        state = roster_information(players, "A")
        expected = enemy_roster_features(roster=players)
        np.testing.assert_array_equal(expected, enemy_roster_features(game_state=state))
        np.testing.assert_array_equal(expected, enemy_roster_features(roster=list(reversed(players))))
        np.testing.assert_array_equal(expected, enemy_roster_features(
            game=NS(defender_roster=[p.base_name for p in players], chars=[])))
        for p in players:
            p.is_alive = False
            p.pos = [-100, -100]
            p.hp = 1
            p.has_spike = True
            p.accuracy = 100
            p.name += "_2"
        np.testing.assert_array_equal(expected, enemy_roster_features(roster=players))
        self.assertEqual(expected.shape, (ENEMY_ROSTER_DIM,))
        self.assertEqual(expected.dtype, np.float32)
        self.assertTrue(np.isfinite(expected).all())

    def test_public_empty_roster_padding_and_player_changes(self):
        features = enemy_roster_features(roster=["Leo"])
        self.assertEqual(features.reshape(5, PLAYER_FEATURE_DIM)[:, 0].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(np.count_nonzero(features[PLAYER_FEATURE_DIM:]), 0)
        self.assertFalse(np.array_equal(features, enemy_roster_features(roster=["Derke"])))
        empty = enemy_roster_features(game_state={"enemy_roster": []},
                                      game=NS(defender_roster=["Leo"]))
        self.assertEqual(np.count_nonzero(empty), 0)
        with self.assertRaises(ValueError):
            base_checkpoint_dim({"enemy_roster_version": 99}, 100)

    def test_action_and_facing_predictions_survive_legacy_expansion(self):
        from learning_attacker_carry_gc import AttackerCarryDuelingDQN
        from learning_attacker_escort_gc import DuelingQNetwork
        from learning_attacker_guard_gc import AttackerGuardDuelingDQN
        from learning_attacker_retrieve_gc import DuelingQNet
        from learning_defender_search_gc import DefenderSearchDuelingDQN
        from learning_defender_retake_gc import DefenderRetakeDuelingDQN
        constructors = (AttackerCarryDuelingDQN, lambda obs_dim: DuelingQNetwork(obs_dim, 8),
                        AttackerGuardDuelingDQN, DuelingQNet,
                        DefenderSearchDuelingDQN, DefenderRetakeDuelingDQN)
        torch.manual_seed(123)
        for constructor in constructors:
            with self.subTest(network=constructor):
                old = constructor(obs_dim=69).eval()
                new = constructor(obs_dim=69 + ENEMY_ROSTER_DIM).eval()
                new.load_state_dict(expand_roster_state(new, old.state_dict()))
                prefix = torch.randn(3, 69)
                expanded = torch.cat((prefix, torch.randn(3, ENEMY_ROSTER_DIM)), dim=1)
                torch.testing.assert_close(new(expanded), old(prefix))
                if hasattr(old, "facing_values"):
                    actions = torch.zeros(3, dtype=torch.long)
                    torch.testing.assert_close(new.facing_values(expanded, actions),
                                               old.facing_values(prefix, actions))

    def test_roster_columns_can_receive_gradients(self):
        from learning_attacker_carry_gc import AttackerCarryDuelingDQN
        torch.manual_seed(3)
        old = AttackerCarryDuelingDQN(obs_dim=29)
        new = AttackerCarryDuelingDQN(obs_dim=29 + ENEMY_ROSTER_DIM)
        new.load_state_dict(expand_roster_state(new, old.state_dict()))
        observation = append_enemy_roster(np.zeros(29), roster=["Leo", "Derke"])
        new(torch.from_numpy(observation).unsqueeze(0))[:, 0].sum().backward()
        self.assertGreater(torch.count_nonzero(new.feature[0].weight.grad[:, -ENEMY_ROSTER_DIM:]), 0)

    def test_retrieve_observation_uses_public_roster_when_no_enemies_are_visible(self):
        from learning_attacker_retrieve_gc import LearningAttackerRetrieveGCController
        controller = LearningAttackerRetrieveGCController.__new__(LearningAttackerRetrieveGCController)
        controller._dist_map = np.zeros((9, 12), dtype=np.int32)
        actor = Character("Absol", "A", [2, 2], "white", "red")
        public = [{"base_name": n} for n in ("Leo", "Boaster", "Derke", "Chronicle", "Alfajer")]
        observation = controller._build_observation(
            actor, np.zeros((9, 12), dtype=np.int32), [actor], [],
            game_state={"enemy_roster": public})
        np.testing.assert_array_equal(observation[-ENEMY_ROSTER_DIM:],
                                      enemy_roster_features(roster=public))

    def test_training_migration_keeps_learned_roster_at_end_of_larger_layout(self):
        import train_attacker_gc_real_curriculum as training
        old = training.escort_runtime.DuelingQNetwork(41 + ENEMY_ROSTER_DIM, 6)
        new = training.escort_runtime.DuelingQNetwork(113 + ENEMY_ROSTER_DIM, 8)
        checkpoint = dict(model_state_dict=old.state_dict(), **ROSTER_METADATA)
        new.load_state_dict(training.expand_policy_state(checkpoint, 113 + ENEMY_ROSTER_DIM, 8))
        for key in ("feature.0.weight", "facing_feature.0.weight"):
            torch.testing.assert_close(new.state_dict()[key][:, :41], old.state_dict()[key][:, :41])
            torch.testing.assert_close(new.state_dict()[key][:, -ENEMY_ROSTER_DIM:],
                                       old.state_dict()[key][:, -ENEMY_ROSTER_DIM:])
            self.assertEqual(torch.count_nonzero(new.state_dict()[key][:, 41:113]), 0)
        torch.testing.assert_close(new.advantage_head[-1].weight[:6], old.advantage_head[-1].weight)

    def test_real_game_delivers_roster_suffix_to_setup_and_live_networks(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR
        seen = set()
        with contextlib.redirect_stdout(io.StringIO()):
            game = VisualFPSBattle(
                NEW_MAZE_STR, _build_team_ai("gc"), _build_team_ai("gc"), headless=True,
                attacker_roster=["Xdll", "SyouTa", "Absol", "eKo", "SugarZ3ro"],
                defender_roster=["Leo", "Boaster", "Derke", "Chronicle", "Alfajer"],
                disable_side_swap=True)
            game.record_replay = False
            attacker = game.attacker_controller.inner_controller
            defender = game.defender_controller.inner_controller
            models = [("A", "carry", attacker.carry, "policy_net"),
                      ("A", "escort", attacker.escort, "policy_net"),
                      ("A", "macro", attacker.macro_controller, "model"),
                      ("A", "guard", attacker.guard, "model"),
                      ("D", "setup", defender.setup_planner, "model"),
                      ("D", "search", defender.search, "model"),
                      ("D", "opening", defender.opening_macro_controller, "selection_net")]
            def hook(team, label):
                def check(_model, inputs):
                    expected = enemy_roster_features(game=game, viewer_team=team)
                    np.testing.assert_array_equal(inputs[0][0, -ENEMY_ROSTER_DIM:].detach().cpu(), expected)
                    seen.add(label)
                return check
            handles = []
            for team, label, controller, attr in models:
                controller.verbose = False
                handles.append(getattr(controller, attr).register_forward_pre_hook(hook(team, label)))
            try:
                game._simulate_tick()
                game.defender_setup_phase.finish()
                for _ in range(4):
                    game._simulate_tick()
                # Macro can hold the whole squad before a phase policy runs.
                # Exercise both underlying phase boundaries with this lineup.
                state = dict(grid=game.grid, chars=game.chars, is_planted=False,
                             round_timer=game.round_timer, target_plant_pos=game.target_plant_pos,
                             **roster_information(game.chars, "A"))
                carrier = next(c for c in game.chars if c.team == "A" and c.has_spike)
                escort = next(c for c in game.chars if c.team == "A" and c is not carrier)
                attacker.carry.set_game(game)
                attacker.escort.set_game(game)
                attacker.carry.decide_move(carrier, state)
                attacker.escort.decide_move(escort, state)
                game.is_planted = True
                game.planted_pos = tuple(np.argwhere(game.grid == 2)[0])
                for char in game.chars:
                    char.has_spike = False
                game._simulate_tick()
            finally:
                for handle in handles:
                    handle.remove()
        self.assertTrue({"setup", "carry", "escort", "macro", "search", "guard"} <= seen, seen)

    def test_legacy_and_versioned_checkpoints_load_in_every_active_gc_layer(self):
        from run_game import _build_team_ai
        with contextlib.redirect_stdout(io.StringIO()):
            team = _build_team_ai("gc")
            attacker = team.get_attacker_controller().inner_controller
            defender = team.get_defender_controller().inner_controller
        layers = [(getattr(attacker, name), net) for name, net in
                  (("carry", "policy_net"), ("escort", "policy_net"),
                   ("retrieve", "model"), ("guard", "model"), ("macro_controller", "model"))]
        layers += [(getattr(defender, name), net) for name, net in
                   (("search", "model"), ("retake", "model"), ("setup_planner", "model"))]
        opening = defender.opening_macro_controller
        self.assertIsNotNone(opening)
        for controller, network_name in layers:
            self.assertIsNotNone(controller)
            model = getattr(controller, network_name)
            width = next(m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear))
            self.assertGreater(width, ENEMY_ROSTER_DIM)
        # Round-trip representative current phase policies with the version tag.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            for controller, network_name in layers[:4] + layers[5:7]:
                with self.subTest(controller=type(controller).__name__):
                    model = getattr(controller, network_name)
                    obs_dim = model.feature[0].in_features
                    action_module = getattr(model, "advantage_head",
                                            getattr(model, "advantage", getattr(model, "adv_head", None)))
                    payload = dict(model_state_dict=model.state_dict(), obs_dim=obs_dim,
                                   n_actions=action_module[-1].out_features, **ROSTER_METADATA,
                                   positioning_version=getattr(controller, "positioning_version", 0),
                                   facing_head_version=getattr(model, "facing_head_version", 0))
                    path = Path(directory) / f"{type(controller).__name__}.pt"
                    torch.save(payload, path)
                    with contextlib.redirect_stdout(io.StringIO()):
                        loaded = type(controller)(model_path=str(path))
                    self.assertEqual(getattr(loaded, network_name).feature[0].in_features, obs_dim)
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(getattr(loaded, network_name).state_dict()[key], value)
            for controller, network_names, state_keys in (
                (attacker.macro_controller, ("model",), ("model_state_dict",)),
                (defender.setup_planner, ("model",), ("model_state_dict",)),
                (opening, ("selection_net", "execution_net"),
                 ("selection_state_dict", "execution_state_dict")),
            ):
                with self.subTest(controller=type(controller).__name__):
                    payload = torch.load(controller.model_path, map_location="cpu", weights_only=False)
                    for network_name, state_key in zip(network_names, state_keys):
                        model = getattr(controller, network_name)
                        payload[state_key] = model.state_dict()
                    width = next(m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear))
                    payload.update(obs_dim=width, **ROSTER_METADATA)
                    path = Path(directory) / f"{type(controller).__name__}.pt"
                    torch.save(payload, path)
                    with contextlib.redirect_stdout(io.StringIO()):
                        loaded = type(controller)(model_path=path, verbose=False)
                    for network_name in network_names:
                        for key, value in getattr(controller, network_name).state_dict().items():
                            torch.testing.assert_close(getattr(loaded, network_name).state_dict()[key], value)


if __name__ == "__main__":
    unittest.main()
