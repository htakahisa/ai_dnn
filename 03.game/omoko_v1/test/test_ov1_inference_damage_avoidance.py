"""Verify zone avoidance across phases without loading or training models."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "omoko_v1"))
from public_effects import DisplayEffect
import ov1_inference_damage_avoidance as safety
from ov1_attacker_controller import Ov1AttackerController
from ov1_defender_controller import Ov1DefenderController


class DamageAvoidanceTests(unittest.TestCase):
    def fixture(self):
        actor = NS(name="a", team="A", is_alive=True, pos=[3, 1], facing="E", has_spike=True)
        cells = tuple((r, c) for r in (2, 3, 4) for c in (2, 3, 4))
        state = {"grid": np.zeros((7, 8), dtype=int), "chars": [actor],
                 "is_planted": False,
                 "public_effects": (DisplayEffect(1, "DESTRUCTION", "active", (3, 3), cells=cells),)}
        return actor, state

    def test_detour_never_enters_zone_and_reaches_far_edge(self):
        actor, state = self.fixture()
        for _ in range(12):
            proposed = ([actor.pos[0], actor.pos[1] + 1], {"facing": "E"})
            actor.pos, _ = safety.avoid_destruction_entry(actor, state, proposed)
            self.assertNotIn(tuple(actor.pos), safety.danger_cells(actor, state))
            if actor.pos[1] >= 5:
                break
        self.assertGreaterEqual(actor.pos[1], 5)

    def test_escape_overrides_stationary_actions(self):
        actor, state = self.fixture()
        actor.pos = [3, 3]
        for _ in range(2):
            old = tuple(actor.pos)
            actor.pos, payload = safety.escape_destruction_area(actor, state)
            self.assertEqual(sum(abs(a-b) for a, b in zip(old, actor.pos)), 1)
            self.assertEqual(payload, {"facing": "E"})
        self.assertNotIn(tuple(actor.pos), safety.danger_cells(actor, state))

    def test_escape_avoids_occupied_cells(self):
        actor, state = self.fixture()
        actor.pos = [2, 3]
        state["chars"].append(NS(name="ally", is_alive=True, pos=[1, 3]))
        self.assertNotEqual(safety.escape_destruction_area(actor, state)[0], [1, 3])

    def test_impassable_zone_waits_in_safety(self):
        actor, state = self.fixture()
        state["grid"][:3, :] = 1
        state["grid"][4:, :] = 1
        self.assertEqual(safety.avoid_destruction_entry(actor, state, ([3, 2], "MOVE"))[0], actor.pos)

    def test_no_zone_preserves_action_and_does_not_use_hidden_raw_areas(self):
        actor, state = self.fixture()
        state["public_effects"] = ()
        result = ([3, 2], {"ability": "FLASH", "target": (3, 5)})
        game = NS(destruction_areas=[{"cells": {(3, 2)}, "remaining_ticks": 10, "team": "D"}])
        self.assertIs(safety.avoid_destruction_entry(actor, state, result, game), result)

    def test_warning_is_avoided_but_smoke_is_not(self):
        actor, state = self.fixture()
        state["public_effects"] = (DisplayEffect(1, "BALEMOON", "warning", (3, 2), cells=((3, 2),)),)
        self.assertIn((3, 2), safety.danger_cells(actor, state))
        state["public_effects"] = (DisplayEffect(1, "SMOKE", "active", (3, 2), cells=((3, 2),)),)
        self.assertEqual(safety.danger_cells(actor, state), set())

    def test_all_attacker_phases_filter_entry(self):
        for phase in ("carry", "escort", "retrieve", "guard"):
            with self.subTest(phase=phase):
                actor, state = self.fixture()
                ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
                for name in ("carry", "escort", "retrieve", "guard"):
                    setattr(ctrl, name + "_controller", Mock(decide_move=Mock(return_value=([3, 2], {}))))
                ctrl.site_ability_used_by_team = False
                actor.has_spike = phase == "carry"
                state["is_planted"] = phase == "guard"
                if phase == "escort":
                    state["chars"].append(NS(name="holder", team="A", is_alive=True, pos=[6, 7], has_spike=True))
                pos, _ = ctrl.decide_move(actor, state)
                self.assertNotIn(tuple(pos), safety.danger_cells(actor, state))
                getattr(ctrl, phase + "_controller").decide_move.assert_called_once()

    def test_both_defender_phases_filter_entry(self):
        for planted in (False, True):
            actor, state = self.fixture()
            actor.team = "D"
            state["is_planted"] = planted
            ctrl = Ov1DefenderController.__new__(Ov1DefenderController)
            ctrl.search_controller = Mock(decide_move=Mock(return_value=([3, 2], {})))
            ctrl.retake_controller = Mock(decide_move=Mock(return_value=([3, 2], {})))
            pos, _ = ctrl.decide_move(actor, state)
            self.assertNotIn(tuple(pos), safety.danger_cells(actor, state))
            (ctrl.retake_controller if planted else ctrl.search_controller).decide_move.assert_called_once()

    def test_escape_precedes_phase_policy(self):
        actor, state = self.fixture()
        actor.pos = [2, 3]
        for cls in (Ov1AttackerController, Ov1DefenderController):
            ctrl = cls.__new__(cls)
            self.assertEqual(ctrl.decide_move(actor, state)[0], [1, 3])

    def test_public_sensor_supplies_visible_zone_without_state_injection(self):
        sys.path.insert(0, str(ROOT / "test"))
        from test_ultimate_system import UltimateTestGame, make_character
        game = UltimateTestGame(height=7, width=8)
        actor = make_character("Xdll", "A", (3, 1))
        actor.facing = "E"
        game.chars = [actor]
        game.destruction_areas = [{"cells": {(3, 2)}, "pos": (3, 2), "level": 5,
                                   "remaining_ticks": 10, "owner": "hidden", "team": "D"}]
        state = {"grid": game.grid, "chars": game.chars}
        self.assertIn((3, 2), safety.danger_cells(actor, state, game))
        self.assertNotEqual(safety.avoid_destruction_entry(actor, state, ([3, 2], {}), game)[0], [3, 2])

    def test_two_different_cells_do_not_stack_but_finished_contract_reapplies(self):
        sys.path.insert(0, str(ROOT / "test"))
        from test_contractor_system import ContractorSystemTests
        fixture = ContractorSystemTests()
        fixture.setUp()
        fixture.ash()
        fixture.advance()
        self.assertEqual((fixture.enemy.hp, fixture.enemy.life_contract_remaining), (90, 4))
        fixture.enemy.pos = [4, 5]  # A different cell in the same 3x3 zone.
        fixture.game._trigger_destruction_areas(fixture.enemy)
        self.assertEqual(fixture.enemy.life_contract_remaining, 4)
        for _ in range(4):
            fixture.advance()
        self.assertEqual((fixture.enemy.hp, fixture.enemy.max_hp, fixture.enemy.life_contract_remaining), (50, 50, 0))
        fixture.advance()  # Still in an active zone: a fresh five-tick contract.
        self.assertEqual((fixture.enemy.hp, fixture.enemy.max_hp, fixture.enemy.life_contract_remaining), (40, 40, 4))


if __name__ == "__main__":
    unittest.main()
