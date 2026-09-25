import unittest
from types import SimpleNamespace

import numpy as np

from abilities_los import AbilityLosMixin
from battle_logic import BattleLogicMixin

from coach_v1.common.types import Side
from coach_v1.evaluate_task09a_gorimaru_smoke import (
    CHECKPOINTS, evaluate_smoke, marginal_shot_lanes,
)


class _ShotGame(BattleLogicMixin, AbilityLosMixin):
    def __init__(self):
        self.grid = np.zeros((26, 44), dtype=np.int8)
        self.chars = [
            SimpleNamespace(name="ally", team="A", pos=[10, 5], facing="W",
                            is_alive=True, reveal_remaining=0),
            SimpleNamespace(name="enemy", team="D", pos=[10, 10], facing="W",
                            is_alive=True, reveal_remaining=0),
        ]
        self.smokes = []


class GorimaruSmokeEvaluationTest(unittest.TestCase):
    def test_marginal_lanes_use_core_shot_los_and_restore_smoke(self):
        game = _ShotGame()
        smoke = {"cells": {(10, 8)}, "remaining_ticks": 25, "owner": "ally"}
        game.smokes = [smoke]
        original_list = game.smokes
        result = marginal_shot_lanes(game, "A", smoke)
        self.assertEqual(1, result["enemy_lanes_without"])
        self.assertEqual(1, result["enemy_lanes_blocked"])
        self.assertEqual(0, result["ally_lanes_blocked"])
        self.assertIs(original_list, game.smokes)
        game.chars[0].reveal_remaining = 1
        self.assertEqual(0, marginal_shot_lanes(game, "A", smoke)["enemy_lanes_blocked"])

    def test_redundant_smoke_has_no_marginal_effect(self):
        game = _ShotGame()
        previous = {"cells": {(10, 8)}, "remaining_ticks": 25, "owner": "other"}
        smoke = {"cells": {(10, 8)}, "remaining_ticks": 25, "owner": "ally"}
        game.smokes = [previous, smoke]
        self.assertEqual(0, marginal_shot_lanes(game, "A", smoke)["enemy_lanes_blocked"])

    def test_actual_game_smoke_activates_without_actor_truth(self):
        result = evaluate_smoke(
            side=Side.ATTACKER, seed=60, encounter="near",
            checkpoint=CHECKPOINTS["selected"], ticks=20,
        )
        self.assertEqual(1, result["smoke_requests"])
        self.assertEqual(1, result["smoke_successes"])
        self.assertGreater(result["smoke_active_evaluated_ticks"], 0)
        self.assertTrue(1 <= result["placement"]["cell_count"] <= 9)
        self.assertNotIn("enemy_positions", result["placement"])

    def test_shadow_smoke_compares_same_state_without_changing_live_action(self):
        baseline = evaluate_smoke(
            side=Side.ATTACKER, seed=60, encounter="near",
            checkpoint=CHECKPOINTS["selected"], ticks=20,
        )
        compared = evaluate_smoke(
            side=Side.ATTACKER, seed=60, encounter="near",
            checkpoint=CHECKPOINTS["selected"], ticks=20,
            shadow_checkpoint=CHECKPOINTS["prior_official"],
        )
        self.assertEqual(baseline["placement"]["target"],
                         compared["placement"]["target"])
        self.assertEqual(baseline["marginal_shot_lane_pair_ticks"],
                         compared["marginal_shot_lane_pair_ticks"])
        self.assertTrue(compared["placement"]["shadow_requested_smoke"])
        self.assertEqual(
            compared["placement"]["marginal_shot_lanes"]["enemy_lanes_without"],
            compared["placement"]["shadow_marginal_shot_lanes"]["enemy_lanes_without"],
        )
        self.assertNotIn("enemy_positions", compared["placement"])


if __name__ == "__main__":
    unittest.main()
