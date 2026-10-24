"""Shared aiming and the actual IQ/engine postplant execution boundary."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "omoko_v1"))
sys.path.insert(0, str(ROOT / "test"))
import ov1_inference_team_tactics as tactics
from ov1_attacker_controller import Ov1AttackerController
from ov1_defender_controller import Ov1DefenderController
from ov1_learning_attacker_guard import Ov1LearningAttackerGuardController
from iq_controller_adapter import IQAwareController
from test_ultimate_system import UltimateTestGame, make_character


class TeamTacticsTests(unittest.TestCase):
    def fixture(self):
        actor = NS(name="a", team="A", is_alive=True, pos=[3, 1], facing="N")
        enemy = NS(name="e", team="D", is_alive=True, pos=[5, 5], position_known=True)
        state = {"grid": np.zeros((8, 8), dtype=int), "chars": [actor, enemy],
                 "smoke_cells": (), "defender_defuse_info": {}}
        return actor, enemy, state

    def test_move_faces_teammate_sighting_even_behind_wall(self):
        actor, enemy, state = self.fixture()
        enemy.pos = [3, 5]
        state["grid"][3, 3] = 1
        self.assertEqual(tactics.apply_shared_facing(actor, state, ([2, 1], {"facing": "N"})),
                         ([2, 1], {"facing": "E"}))

    def test_all_six_phases_face_shared_enemy_while_moving(self):
        for phase in ("carry", "escort", "retrieve", "guard", "search", "retake"):
            with self.subTest(phase=phase):
                actor, enemy, state = self.fixture()
                enemy.pos = [3, 5]
                actor.has_spike = phase == "carry"
                state["is_planted"] = phase in ("guard", "retake")
                state["public_effects"] = ()
                cls = Ov1DefenderController if phase in ("search", "retake") else Ov1AttackerController
                ctrl = cls.__new__(cls)
                for role in ("carry", "escort", "retrieve", "guard", "search", "retake"):
                    setattr(ctrl, role + "_controller", Mock(decide_move=Mock(return_value=([2, 1], {"facing": "N"}))))
                ctrl.site_ability_used_by_team = False
                if phase == "escort":
                    state["chars"].append(NS(name="holder", team="A", is_alive=True, pos=[6, 1], has_spike=True))
                self.assertEqual(ctrl.decide_move(actor, state), ([2, 1], {"facing": "E"}))
                getattr(ctrl, phase + "_controller").decide_move.assert_called_once()

    def test_unknown_position_is_not_an_enemy_sighting(self):
        actor, enemy, state = self.fixture()
        result = ([2, 1], {"facing": "N"})
        for position, known in (([-1, -1], False), ([5, 5], False)):
            enemy.pos, enemy.position_known = position, known
            self.assertIs(tactics.apply_shared_facing(actor, state, result), result)

    def test_direct_shootable_threat_precedes_distant_team_sighting(self):
        actor, enemy, state = self.fixture()
        enemy.pos = [3, 3]
        state["grid"][3, 2] = 1
        state["chars"].append(NS(name="local", team="D", pos=[1, 1], is_alive=True))
        self.assertEqual(tactics.apply_shared_facing(actor, state, ([3, 1], {}))[1]["facing"], "N")

    def test_forced_facing_and_directional_ultimate_are_preserved(self):
        actor, _, state = self.fixture()
        result = ([3, 1], {"facing": "W"})
        actor.facing_forced_this_tick = True
        self.assertIs(tactics.apply_shared_facing(actor, state, result), result)
        actor.facing_forced_this_tick = False
        result = ([3, 1], {"ultimate": "TUNNEL", "facing": "W"})
        self.assertIs(tactics.apply_shared_facing(actor, state, result), result)

    def test_plant_action_keeps_string_while_facing_changes(self):
        actor, _, state = self.fixture()
        result = ([3, 1], "PLANT")
        self.assertIs(tactics.apply_shared_facing(actor, state, result), result)
        self.assertEqual(actor.facing, "SE")

    def test_shot_geometry_matches_engine_corner_smoke_and_body_rules(self):
        actor, enemy, state = self.fixture()
        game = UltimateTestGame(height=8, width=8)
        actor = make_character("Xdll", "A", (3, 1))
        enemy = make_character("Demon1", "D", (3, 3))
        ally = make_character("Leo", "A", (0, 0))
        game.chars = [actor, enemy, ally]
        state.update(grid=game.grid, chars=game.chars)
        for pos, walls, smoke, blocker in (
            ((3, 3), (), (), (0, 0)),
            ((3, 2), (), ((3, 1),), (0, 0)),
            ((4, 2), ((3, 2),), (), (0, 0)),
            ((3, 4), (), (), (3, 2)),
            ((3, 4), (), ((3, 4),), (0, 0)),
        ):
            enemy.pos, ally.pos = list(pos), list(blocker)
            game.grid[:] = 0
            for cell in walls:
                game.grid[cell] = 1
            game.smokes = [{"cells": set(smoke)}]
            state["smoke_cells"] = set(smoke)
            self.assertEqual(tactics.shot_clear(actor, enemy, state), game.check_shot_line_of_sight(actor, enemy))

    def test_iq_observation_remains_blurred_but_disclosed_tactics_are_precise(self):
        game = UltimateTestGame(height=8, width=8)
        actor = make_character("Xdll", "A", (3, 1))
        enemy = make_character("Demon1", "D", (3, 4))
        hidden = make_character("Leo", "D", (6, 6))
        enemy.los_revealed = True
        game.chars = [actor, enemy, hidden]
        game.planted_pos = (3, 3)
        captured = {}
        inner = NS(requires_exact_team_tactics=True, requires_public_effects=False,
                   set_game=Mock(), decide_move=lambda char, state: (captured.update(state) or [3, 1]))
        wrapper = IQAwareController(inner, viewer_team="A")
        wrapper.set_game(game)
        def blur(game, viewer, pos, *args):
            return None if pos is None else [min(7, pos[0] + 2), min(7, pos[1] + 2)]
        with patch.object(wrapper.perception_engine, "_blur_pos", side_effect=blur), \
                patch.object(wrapper.perception_engine, "_omit_enemy", return_value=False):
            wrapper.decide_move(actor, {"grid": game.grid, "chars": game.chars, "is_planted": True,
                                        "planted_pos": game.planted_pos, "defender_defuse_info": {}})
        self.assertEqual(captured["planted_pos"], [5, 5])
        shared = captured["team_tactical_state"]
        self.assertEqual(shared["planted_pos"], (3, 3))
        self.assertEqual(shared["chars"][1].pos, [3, 4])
        self.assertEqual(shared["chars"][2].pos, [-1, -1])
        self.assertFalse(shared["chars"][2].position_known)

    def test_guard_uses_public_spike_not_jittered_goal(self):
        actor, enemy, state = self.fixture()
        state.update(is_planted=True, planted_pos=[6, 6],
                     defender_defuse_info={"e": (1, 6)}, smoke_cells={(3, 3), (3, 4)})
        enemy.position_known, enemy.pos = False, [-1, -1]
        shared = {**state, "planted_pos": [3, 4]}
        state["team_tactical_state"] = shared
        ctrl = Ov1LearningAttackerGuardController.__new__(Ov1LearningAttackerGuardController)
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 2])

    def test_hidden_defuser_search_rotates_instead_of_leaving_spike(self):
        actor, enemy, state = self.fixture()
        actor.pos = [3, 4]
        enemy.position_known, enemy.pos = False, [-1, -1]
        state.update(is_planted=True, planted_pos=[3, 4], defender_defuse_info={"e": (1, 6)})
        ctrl = Ov1LearningAttackerGuardController.__new__(Ov1LearningAttackerGuardController)
        facings = set()
        for _ in range(4):
            pos, payload = ctrl.decide_move(actor, state)
            self.assertEqual(pos, [3, 4])
            facings.add(payload["facing"])
        self.assertEqual(len(facings), 4)

    def test_defuser_shot_precedes_damage_zone_retreat(self):
        from public_effects import DisplayEffect
        actor, enemy, state = self.fixture()
        actor.pos, enemy.pos = [3, 3], [3, 4]
        state.update(is_planted=True, defender_defuse_info={"e": (5, 6)},
                     public_effects=(DisplayEffect(1, "DESTRUCTION", "active", (3, 3), cells=((3, 3),)),))
        ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
        self.assertEqual(ctrl.decide_move(actor, state), ([3, 3], {"facing": "E"}))

    def test_guard_repositions_when_ally_blocks_shot_at_spike(self):
        actor, enemy, state = self.fixture()
        actor.pos, enemy.pos = [3, 3], [3, 5]
        state["chars"].append(NS(name="ally", team="A", is_alive=True, pos=[3, 4]))
        state.update(is_planted=True, planted_pos=[3, 3], defender_defuse_info={"e": (2, 6)})
        ctrl = Ov1LearningAttackerGuardController.__new__(Ov1LearningAttackerGuardController)
        pos, payload = ctrl.decide_move(actor, state)
        self.assertNotEqual(pos, [3, 3])
        actor.pos = pos
        for _ in range(3):
            if tactics.shot_clear(actor, enemy, state):
                break
            actor.pos, payload = ctrl.decide_move(actor, state)
        pos = actor.pos
        self.assertTrue(tactics.shot_clear(actor, enemy, state, pos))
        self.assertEqual(payload["facing"], tactics.facing_towards(pos, enemy.pos))

    def test_engine_iq_wrapper_interrupts_before_six_tick_defuse_finishes(self):
        for initially_known in (False, True):
            with self.subTest(initially_known=initially_known):
                self._assert_engine_interrupt(initially_known)

    def _assert_engine_interrupt(self, initially_known):
        game = UltimateTestGame(height=8, width=8)
        actor = make_character("Xdll", "A", (3, 3))
        enemy = make_character("Demon1", "D", (3, 4))
        actor.facing = "W"
        enemy.los_revealed = initially_known
        enemy.facing = "E"
        enemy.defuse_timer = 1
        game.chars = [actor, enemy]
        game.is_planted, game.is_defused, game.planted_pos = True, False, (3, 3)
        ctrl = Ov1AttackerController.__new__(Ov1AttackerController)
        for phase in ("carry", "escort", "retrieve", "guard"):
            setattr(ctrl, phase + "_controller", Mock())
        ctrl.guard_controller = Ov1LearningAttackerGuardController.__new__(Ov1LearningAttackerGuardController)
        wrapper = IQAwareController(ctrl, viewer_team="A")
        wrapper.set_game(game)
        game.attacker_controller = wrapper
        with patch.object(wrapper.perception_engine, "_omit_enemy", return_value=False), \
                patch("battle_logic.random.random", return_value=0.0):
            for tick in range(1, 6):
                game.battle_tick = tick
                game.move_character(actor)
                self.assertEqual(actor.pos, [3, 3])
                self.assertLessEqual(game._facing_angle_diff(actor, enemy), 90)
                game._resolve_all_shots()
                game._resolve_defuse_completion()
                self.assertFalse(game.is_defused)
                if not enemy.is_alive:
                    break
                enemy.defuse_timer += 1
                game._update_los_reveal()
        self.assertFalse(enemy.is_alive)
        self.assertLess(tick, 6)
        self.assertFalse(game.is_defused)


if __name__ == "__main__":
    unittest.main()
