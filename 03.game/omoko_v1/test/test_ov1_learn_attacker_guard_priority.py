"""Postplant movement and aim regression checks; no training or checkpoints."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "omoko_v1"))
import ov1_learning_attacker_guard as guard


class GuardPriorityTests(unittest.TestCase):
    def fixture(self):
        ctrl = guard.Ov1LearningAttackerGuardController.__new__(guard.Ov1LearningAttackerGuardController)
        for method in ("_ensure_spike_dist_map", "_ensure_guard_assignment",
                       "_maybe_advance_tick", "_update_sighting_dist_map"):
            setattr(ctrl, method, Mock())
        ctrl._build_observation = Mock(side_effect=AssertionError("Policy must not override emergency/arrival"))
        ctrl._assigned_dist_maps = {}
        actor = NS(name="a", team="A", pos=[3, 0], is_alive=True, facing="W")
        state = {"grid": np.zeros((7, 7), dtype=int), "chars": [actor],
                 "is_planted": True, "planted_pos": [3, 4],
                 "smoke_cells": {(3, 3), (3, 4)},
                 "defender_defuse_info": {"defuser": (1, 6)}}
        return ctrl, actor, state

    def test_hidden_defuser_pushes_through_smoke(self):
        ctrl, actor, state = self.fixture()
        for col in (1, 2, 3, 4):
            pos, payload = ctrl.decide_move(actor, state)
            self.assertEqual(pos, [3, col])
            self.assertEqual(payload["facing"], "E")
            actor.pos = pos
        pos, payload = ctrl.decide_move(actor, state)
        self.assertEqual(pos, [3, 4])
        self.assertNotEqual(payload["facing"], "W")

    def test_adjacent_defuser_is_faced_and_shooter_stops(self):
        ctrl, actor, state = self.fixture()
        actor.pos = [3, 4]
        state["chars"].append(NS(name="defuser", team="D", pos=[4, 4], is_alive=True))
        self.assertEqual(ctrl.decide_move(actor, state), ([3, 4], {"facing": "S"}))

    def test_unrelated_visible_enemy_does_not_cancel_push(self):
        ctrl, actor, state = self.fixture()
        state["chars"].append(NS(name="decoy", team="D", pos=[0, 0], is_alive=True))
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 1])

    def test_detours_around_ally_even_when_distance_increases(self):
        ctrl, actor, state = self.fixture()
        state["chars"].append(NS(name="ally", team="A", pos=[3, 1], is_alive=True))
        state["grid"][2, 0] = 1
        self.assertEqual(ctrl.decide_move(actor, state)[0], [4, 0])

    def test_occupied_spike_approaches_free_adjacent_cell(self):
        ctrl, actor, state = self.fixture()
        state["chars"].append(NS(name="ally", team="A", pos=[3, 4], is_alive=True))
        for col in (1, 2, 3):
            actor.pos, _ = ctrl.decide_move(actor, state)
            self.assertEqual(actor.pos, [3, col])
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 3])

    def test_blocked_shot_continues_push(self):
        ctrl, actor, state = self.fixture()
        state["smoke_cells"] = set()
        state["chars"] += [NS(name="defuser", team="D", pos=[3, 4], is_alive=True),
                            NS(name="ally", team="A", pos=[3, 2], is_alive=True)]
        self.assertEqual(ctrl.decide_move(actor, state)[0], [2, 0])

    def test_every_adjacent_defuse_cell_is_faced_from_spike(self):
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            with self.subTest(delta=(dr, dc)):
                ctrl, actor, state = self.fixture()
                actor.pos = [3, 4]
                enemy = NS(name="defuser", team="D", pos=[3 + dr, 4 + dc], is_alive=True)
                state["chars"].append(enemy)
                pos, payload = ctrl.decide_move(actor, state)
                self.assertEqual(pos, actor.pos)
                self.assertEqual(payload["facing"], guard._facing_towards(actor.pos, enemy.pos))

    def test_guard_arrival_ignores_stale_sighting_and_detours(self):
        ctrl, actor, state = self.fixture()
        state["defender_defuse_info"] = {"defuser": (0, 6)}
        ctrl.team_memory = NS(last_seen_enemy={"pos": (0, 0)})
        ctrl._assigned_dist_maps[actor.name] = guard._bfs_distance_map(state["grid"], (3, 4))
        state["chars"].append(NS(name="ally", team="A", pos=[3, 1], is_alive=True))
        self.assertEqual(ctrl.decide_move(actor, state)[0], [2, 0])

    def test_guard_holds_position_despite_policy_preferring_move_or_ultimate(self):
        ctrl, actor, state = self.fixture()
        state["defender_defuse_info"] = {}
        ctrl.team_memory = NS(last_seen_enemy=None)
        ctrl._assigned_dist_maps[actor.name] = guard._bfs_distance_map(state["grid"], tuple(actor.pos))
        ctrl._build_observation = Mock(return_value=(np.zeros(guard.OBS_DIM, dtype=np.float32), []))
        ctrl._action_mask = Mock(return_value=np.ones(guard.ACTION_DIM, dtype=bool))
        ctrl.device, ctrl.verbose = torch.device("cpu"), False
        values = torch.zeros((1, guard.ACTION_DIM))
        values[0, 2] = 1  # stay, E
        values[0, 16:] = 100  # prefer leaving unless masked
        ctrl.model = lambda obs: values.clone()
        self.assertEqual(ctrl.decide_move(actor, state), ([3, 0], {"facing": "E"}))

    def test_engine_can_shoot_after_defuse_aim(self):
        sys.path.insert(0, str(ROOT / "test"))
        from test_ultimate_system import UltimateTestGame, make_character
        ctrl, _, _ = self.fixture()
        game = UltimateTestGame(height=7, width=7)
        actor = make_character("Xdll", "A", (3, 3))
        enemy = make_character("Demon1", "D", (3, 4))
        enemy.defuse_timer = 1
        game.chars = [actor, enemy]
        game.is_planted, game.planted_pos = True, (3, 4)
        game.smokes = [{"cells": {(3, 3), (3, 4)}}]
        game.attacker_controller = ctrl
        game.move_character(actor)
        self.assertEqual(actor.pos, [3, 3])
        self.assertEqual(actor.facing, "E")
        self.assertTrue(game.check_shot_line_of_sight(actor, enemy))
        self.assertEqual(game._facing_angle_diff(actor, enemy), 0)


if __name__ == "__main__":
    unittest.main()
