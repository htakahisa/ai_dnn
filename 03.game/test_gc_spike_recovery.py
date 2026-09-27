"""Recovery regressions using both GC routing and actual battle physics."""

import contextlib
import io
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from gc_v1.retrieve_tactics_gc import SpikeRecoveryCoordinator
from ghost_champions_v1 import GhostChampionsV1AttackerController as BaseGC
from ghost_champions_v1_macro import GhostChampionsV1AttackerController as MacroGC
from game_core import PLANT_REQUIRED_TICKS
from iq_controller_adapter import IQAwareController
from test_ultimate_system import UltimateTestGame, make_character


def controller(cls=MacroGC):
    result = cls.__new__(cls)
    result.game = None
    result.spike_recovery = SpikeRecoveryCoordinator()
    result.site_ability_used_by_team = False
    idle = NS(decide_move=Mock(side_effect=lambda c, s: (list(c.pos), "MOVE")))
    result.carry = result.escort = result.retrieve = result.guard = result.fallback = idle
    if cls is MacroGC:
        result.macro_controller = NS(
            _sync_tick_once=Mock(),
            lurk_coordination_result=Mock(side_effect=lambda c, s: (list(c.pos), "MOVE")),
            set_game=Mock(), reset_round=Mock(),
        )
    return result


class SpikeRecoveryTests(unittest.TestCase):
    def fixture(self, cls=MacroGC):
        ctrl = controller(cls)
        actor = make_character("Xdll", "A", (3, 0))
        grid = np.zeros((7, 10), dtype=np.int32)
        grid[3, 9] = 2
        state = {"grid": grid, "chars": [actor], "spike_pos": (3, 5),
                 "is_planted": False, "round_timer": 22,
                 "target_plant_pos": (0, 0)}
        return ctrl, actor, state

    def test_idle_or_oscillating_models_cannot_override_recovery(self):
        for cls in (BaseGC, MacroGC):
            for enemy_visible in (False, True):
                with self.subTest(cls=cls, enemy_visible=enemy_visible):
                    ctrl, actor, state = self.fixture(cls)
                    if enemy_visible:
                        state["chars"].append(make_character("Demon1", "D", (3, 8)))
                    ctrl.retrieve.decide_move.side_effect = AssertionError("Retrieve model ran")
                    for col in range(1, 6):
                        result = ctrl.decide_move(actor, state)
                        self.assertEqual(result[:2], ([3, col], "MOVE"))
                        actor.pos = result[0]

    def test_nearest_retriever_is_sticky_even_if_teammate_becomes_closer(self):
        ctrl, actor, state = self.fixture()
        teammate = make_character("eKo", "A", (0, 0))
        state["chars"].append(teammate)
        actor.pos = ctrl.decide_move(actor, state)[0]
        teammate.pos = [2, 5]
        ctrl.decide_move(teammate, state)
        self.assertEqual(ctrl.spike_recovery.plan["retriever"], actor.name)
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 2])

    def test_perception_jitter_does_not_reverse_approach(self):
        ctrl, actor, state = self.fixture()
        for tick, noisy in enumerate(((3, 5), (2, 4), (4, 6), (3, 4), (2, 6))):
            state["spike_pos"] = noisy
            state["round_timer"] -= 1
            result = ctrl.decide_move(actor, state)
            self.assertEqual(result[0], [3, tick + 1])
            actor.pos = result[0]
        self.assertEqual(ctrl.spike_recovery.plan["target"], (3, 5))

    def test_pickup_commits_nearest_legal_plant_before_macro_lurk_wait(self):
        ctrl, actor, state = self.fixture()
        ctrl.decide_move(actor, state)
        actor.pos = [3, 5]
        actor.has_spike = True
        state["spike_pos"] = None
        state["grid"][3, 6] = 5  # Orb cells are NOT plantable.
        state["grid"][2, 5] = 2
        self.assertEqual(ctrl.decide_move(actor, state)[:2], ([2, 5], "MOVE"))
        actor.pos = [2, 5]
        for _ in range(PLANT_REQUIRED_TICKS):
            self.assertEqual(ctrl.decide_move(actor, state), ([2, 5], "PLANT"))
        ctrl.macro_controller.lurk_coordination_result.assert_not_called()
        ctrl.carry.decide_move.assert_not_called()

    def test_recovery_starts_before_deadline_and_reserves_full_plant_budget(self):
        ctrl, actor, state = self.fixture()
        state["round_timer"] = 90
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 1])
        budget = ctrl.spike_recovery.last_status
        self.assertGreaterEqual(budget["required_ticks"], 5 + 4 + PLANT_REQUIRED_TICKS)
        self.assertFalse(budget["urgent"])
        state["round_timer"] = 12
        actor.pos = [3, 1]
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 2])
        self.assertTrue(ctrl.spike_recovery.last_status["urgent"])

    def test_teammate_clears_corridor_instead_of_blocking_retriever(self):
        ctrl, actor, state = self.fixture()
        ctrl.decide_move(actor, state)
        blocker = make_character("eKo", "A", (3, 1))
        state["chars"].append(blocker)
        result = ctrl.decide_move(blocker, state)
        self.assertNotEqual(result[0][0], 3)
        blocker.pos = result[0]
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 1])

    def test_enemy_on_spike_does_not_freeze_far_away(self):
        ctrl, actor, state = self.fixture()
        state["chars"].append(make_character("Demon1", "D", (3, 5)))
        for col in range(1, 5):
            actor.pos = ctrl.decide_move(actor, state)[0]
            self.assertEqual(actor.pos, [3, col])

    def test_dead_retriever_reassigns_and_new_drop_restarts_recovery(self):
        ctrl, actor, state = self.fixture()
        other = make_character("eKo", "A", (0, 0))
        state["chars"].append(other)
        ctrl.decide_move(actor, state)
        actor.is_alive = False
        self.assertNotEqual(ctrl.decide_move(other, state)[0], other.pos)
        self.assertEqual(ctrl.spike_recovery.plan["retriever"], other.name)
        other.has_spike = True
        state["spike_pos"] = None
        ctrl.decide_move(other, state)
        self.assertEqual(ctrl.spike_recovery.plan["phase"], "PLANT")
        other.has_spike = False
        state["spike_pos"] = (0, 3)
        ctrl.decide_move(other, state)
        self.assertEqual(ctrl.spike_recovery.plan["phase"], "RECOVER")
        self.assertEqual(ctrl.spike_recovery.plan["estimate"], (0, 3))

    def test_arrived_estimate_searches_locally_without_waiting_forever(self):
        ctrl, actor, state = self.fixture()
        actual = (2, 4)
        state["spike_pos"] = (3, 5)
        positions = []
        for _ in range(20):
            actor.pos = ctrl.decide_move(actor, state)[0]
            positions.append(tuple(actor.pos))
            if tuple(actor.pos) == actual:
                actor.has_spike = True
                break
        self.assertTrue(actor.has_spike)
        self.assertIn((3, 5), positions)

    def test_starting_on_drop_waits_for_automatic_pickup(self):
        ctrl, actor, state = self.fixture()
        actor.pos = [3, 5]
        self.assertEqual(ctrl.decide_move(actor, state)[:2], ([3, 5], "MOVE"))
        actor.has_spike = True
        state["spike_pos"] = None
        self.assertEqual(ctrl.decide_move(actor, state)[0], [3, 6])

    def test_teammate_calls_do_not_advance_search_before_engine_pickup(self):
        ctrl, actor, state = self.fixture()
        ctrl.decide_move(actor, state)
        teammate = make_character("eKo", "A", (6, 0))
        state["chars"].append(teammate)
        actor.pos = [3, 4]
        actor.pos = ctrl.decide_move(actor, state)[0]
        ctrl.decide_move(teammate, state)
        self.assertEqual(ctrl.spike_recovery.plan["target"], (3, 5))
        self.assertEqual(ctrl.spike_recovery.plan["visited"], set())

    def test_wall_corrected_estimate_expands_search_instead_of_stalling(self):
        ctrl, actor, state = self.fixture()
        actual = (1, 5)
        for _ in range(60):
            actor.pos = ctrl.decide_move(actor, state)[0]
            if tuple(actor.pos) == actual:
                actor.has_spike = True
                break
        self.assertTrue(actor.has_spike)
        self.assertGreater(ctrl.spike_recovery.plan["search_radius"], 1)

    def test_isolated_estimate_does_not_recurse_forever(self):
        ctrl, actor, state = self.fixture()
        state["grid"][:] = 1
        state["grid"][3, 5] = 0
        actor.pos = [3, 5]
        for _ in range(5):
            self.assertEqual(ctrl.decide_move(actor, state)[:2], ([3, 5], "MOVE"))

    def test_set_game_does_not_transfer_recovered_spike_back_to_absol(self):
        ctrl, actor, state = self.fixture(BaseGC)
        absol = make_character("Absol", "A", (6, 0))
        state["chars"].append(absol)
        ctrl.decide_move(actor, state)
        actor.has_spike = True
        ctrl.set_game(NS(chars=state["chars"]))
        self.assertTrue(actor.has_spike)
        self.assertFalse(absol.has_spike)

    def test_regular_carry_and_postplant_remain_model_owned(self):
        ctrl, actor, state = self.fixture(BaseGC)
        state["spike_pos"] = None
        actor.has_spike = True
        ctrl.decide_move(actor, state)
        ctrl.carry.decide_move.assert_called_once()
        ctrl.carry.decide_move.reset_mock()
        ctrl.spike_recovery.plan = {"phase": "PLANT"}
        state["is_planted"] = True
        ctrl.decide_move(actor, state)
        self.assertIsNone(ctrl.spike_recovery.plan)
        ctrl.guard.decide_move.assert_called_once()

    def test_fast_character_does_not_overshoot_spike(self):
        ctrl, actor, state = self.fixture(BaseGC)
        game = UltimateTestGame(height=7, width=10)
        game.grid = state["grid"]
        game.chars = [actor]
        game.spike_pos = (3, 1)
        actor.move_steps_per_tick = 3
        game.attacker_controller = ctrl
        ctrl.set_game(game)
        game.move_character(actor)
        self.assertEqual(actor.pos, [3, 1])

    def test_normal_fast_move_keeps_original_speed_without_precision_request(self):
        game = UltimateTestGame(height=7, width=10)
        actor = make_character("Xdll", "A", (3, 0))
        actor.move_steps_per_tick = 3
        game.chars = [actor]
        game.attacker_controller = NS(decide_move=lambda c, s: ([3, 1], "MOVE"))
        game.move_character(actor)
        self.assertEqual(actor.pos, [3, 3])


class RecoveryBattleIntegrationTests(unittest.TestCase):
    def test_real_engine_and_iq_complete_pickup_to_plant_before_timeout(self):
        from run_game import VisualFPSBattle, _build_team_ai
        from map_data import NEW_MAZE_STR

        for visible in (False, True):
            for iq in (0, 100, 200):
                with self.subTest(visible=visible, iq=iq), contextlib.redirect_stdout(io.StringIO()):
                    game = VisualFPSBattle(NEW_MAZE_STR, _build_team_ai("default"),
                                           _build_team_ai("default"), headless=True)
                    grid = np.zeros((7, 10), dtype=np.int32)
                    grid[3, 9] = 2
                    if not visible:
                        grid[5, :] = 1
                    game.grid, game.height, game.width = grid, 7, 10
                    actor = make_character("Xdll", "A", (3, 0))
                    enemy = make_character("Demon1", "D", (6, 9))
                    actor.iq = iq
                    actor.effective_iq = iq
                    actor.accuracy = enemy.accuracy = 0.0
                    actor.hp = enemy.hp = 100000
                    game.chars = [actor, enemy]
                    game.spike_pos = (3, 5)
                    game.target_plant_pos = (6, 0)  # Bad original tactical target.
                    game.round_timer = 35
                    game.defender_setup_phase = NS(active=False, ticks_remaining=0)
                    game.analytics_tracker = None
                    game.check_match_winner = lambda: None
                    ctrl = controller()
                    game.attacker_controller = IQAwareController(ctrl)
                    game.attacker_controller.set_game(game)
                    self.assertEqual(game.check_line_of_sight(actor, enemy), visible)
                    picked_at = None
                    for tick in range(35):
                        game._build_occupancy_counts()
                        try:
                            game.move_character(actor)
                            game.process_battle()
                        finally:
                            game._movement_occupancy_counts = None
                        if actor.has_spike and picked_at is None:
                            picked_at = tick + 1
                        if game.is_planted or game.round_over:
                            break
                    self.assertIsNotNone(picked_at)
                    self.assertTrue(game.is_planted, (picked_at, actor.pos, ctrl.spike_recovery.plan))
                    self.assertGreater(game.round_timer, 0)
                    self.assertEqual(game.planted_pos, (3, 9))
                    ctrl.macro_controller.lurk_coordination_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
