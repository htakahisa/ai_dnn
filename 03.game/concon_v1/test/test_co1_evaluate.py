"""Checks that spike drops and plants are counted independently."""

import contextlib
import io
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with contextlib.redirect_stdout(io.StringIO()):
    from concon_v1.evaluate_co1_attacker import (
        LimitedRoundBattle,
        evaluate,
        summarize_spike_drops,
    )
    from run_game import VisualFPSBattle


class EvaluationDropTests(unittest.TestCase):
    def test_evaluation_game_stops_after_one_round(self):
        battle = object.__new__(LimitedRoundBattle)
        battle.round_limit = 1
        battle.round_results = []
        battle.current_round = 1
        battle.attacker_wins = 0
        battle.defender_wins = 0
        battle.overtime = False
        battle.previous_attacker_wins = 0
        battle.match_over = False
        battle.is_defused = False
        battle.is_planted = False
        battle.round_timer = 1
        battle.battle_tick = 1
        battle.planted_pos = None
        battle.attacker_controller = SimpleNamespace(
            route_controller=SimpleNamespace(_pattern_index=None)
        )
        battle.defender_controller = SimpleNamespace()
        battle.chars = [
            SimpleNamespace(team="A", is_alive=True, round_kills=0),
            SimpleNamespace(team="D", is_alive=True, round_kills=0),
        ]
        with (patch.object(LimitedRoundBattle, "init_round"),
              patch.object(LimitedRoundBattle, "_record_replay_frame")):
            battle.attacker_wins += 1
            battle.check_match_winner()
        self.assertEqual(len(battle.round_results), 1)
        self.assertTrue(battle.match_over)
        self.assertEqual(battle.current_round, 1)

    def test_each_trial_creates_a_fresh_one_round_game(self):
        games = []
        loaded_checkpoints = []

        def make_game(*args, **kwargs):
            args[1].attacker_factory()
            game = SimpleNamespace(
                attacker_controller=SimpleNamespace(
                    route_controller=SimpleNamespace(rng=random.Random())
                ),
                current_round=1,
                round_results=[],
            )

            def run():
                game.chosen = game.attacker_controller.route_controller.rng.random()
                game.round_results.append({
                    "winner": "A", "planted": True, "end_reason": "detonated",
                    "attacker_alive_at_plant": 3, "defender_alive_at_plant": 2,
                    "attacker_kills": 1, "defender_kills": 0,
                    "first_site_tick": 20, "max_plant_progress": 4,
                    "spike_drop_count": 0,
                })

            game.run = run
            games.append(game)
            return game

        with (patch("concon_v1.evaluate_co1_attacker.LimitedRoundBattle",
                    side_effect=make_game),
              patch("concon_v1.evaluate_co1_attacker._build_team_ai",
                    side_effect=lambda key: SimpleNamespace()),
              patch("concon_v1.evaluate_co1_attacker.ConconAttackerController",
                    side_effect=lambda checkpoint_bytes, map_name: loaded_checkpoints.append(checkpoint_bytes)),
              patch("concon_v1.evaluate_co1_attacker.torch.load",
                    return_value={"episode": 1450, "success_rate": 0.91})):
            result = evaluate("gc_v1", rounds=3, seed=4,
                              frozen_checkpoint=b"fixed-model")
        self.assertEqual(len(games), 3)
        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["plants"], 3)
        self.assertEqual(result["model_episode"], 1450)
        self.assertEqual(loaded_checkpoints, [b"fixed-model"] * 3)
        self.assertEqual([record["trial"] for record in result["details"]], [1, 2, 3])
        self.assertTrue(all(game.round_limit == 1 and game.current_round == 1
                            for game in games))
        self.assertEqual(len({game.chosen for game in games}), 3)

    def test_drop_summary_includes_successful_plants_and_repeat_drops(self):
        rounds = [
            {"planted": True, "spike_drop_count": 2},
            {"planted": True, "spike_drop_count": 0},
            {"planted": False, "spike_drop_count": 1},
            {"planted": False, "spike_drop_count": 0},
        ]
        self.assertEqual(sum(record["planted"] for record in rounds), 2)
        self.assertEqual(summarize_spike_drops(rounds), {
            "spike_drop_events": 3,
            "spike_drop_rounds": 2,
            "plants_after_drop": 1,
            "no_plant_after_drop": 1,
        })

    def test_plant_after_spike_drop_counts_even_if_defused(self):
        battle = object.__new__(LimitedRoundBattle)
        battle.round_limit = 1
        battle.round_results = []
        battle.current_round = 1
        battle.attacker_wins = 0
        battle.previous_attacker_wins = 0
        battle.defender_wins = 1
        battle.overtime = False
        battle.match_over = False
        battle.is_planted = True
        battle.is_defused = True
        battle.round_timer = 1
        battle.battle_tick = 30
        battle.planted_pos = (2, 3)
        battle.spike_drop_count = 1
        battle.plant_tick = 20
        battle.plant_alive_counts = {"A": 1, "D": 2}
        battle.attacker_controller = SimpleNamespace(
            route_controller=SimpleNamespace(_pattern_index=None)
        )
        battle.defender_controller = SimpleNamespace()
        battle.chars = [
            SimpleNamespace(team="A", is_alive=True, round_kills=0),
            SimpleNamespace(team="D", is_alive=True, round_kills=0),
        ]
        with patch.object(LimitedRoundBattle, "_record_replay_frame"):
            battle.check_match_winner()
        record = battle.round_results[0]
        self.assertEqual(record["winner"], "D")
        self.assertEqual(record["end_reason"], "defused")
        self.assertTrue(record["planted"])
        self.assertEqual(summarize_spike_drops([record])["plants_after_drop"], 1)

    def test_each_carrier_death_increments_event_count(self):
        battle = object.__new__(LimitedRoundBattle)
        battle.is_planted = False
        battle.battle_tick = 7
        battle.attacker_controller = SimpleNamespace(
            route_controller=SimpleNamespace(_routes={})
        )
        carrier = SimpleNamespace(team="A", has_spike=True, pos=[2, 3], name="carrier")
        with patch.object(VisualFPSBattle, "_kill_character"):
            battle._kill_character(None, carrier)
            battle.battle_tick = 12
            battle._kill_character(None, carrier)
        self.assertEqual(battle.spike_drop_count, 2)
        self.assertEqual(battle.spike_drop_tick, 7)

    def test_direct_status_death_is_counted_before_battle_drops_spike(self):
        battle = object.__new__(LimitedRoundBattle)
        battle.is_planted = False
        battle.battle_tick = 9
        battle.attacker_controller = SimpleNamespace(
            route_controller=SimpleNamespace(_routes={})
        )
        battle.chars = [SimpleNamespace(
            team="A", has_spike=True, is_alive=False, pos=[2, 3], name="carrier"
        )]
        with patch.object(VisualFPSBattle, "process_battle"):
            battle.process_battle()
        self.assertEqual(battle.spike_drop_count, 1)
        self.assertEqual(battle.spike_drop_tick, 9)


if __name__ == "__main__":
    unittest.main()
