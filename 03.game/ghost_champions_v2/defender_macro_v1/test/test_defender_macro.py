from pathlib import Path
import sys
from types import SimpleNamespace as NS
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ghost_champions_v2.defender_macro_v1.analysis import (
    AttackSiteAnalysis, GCFeatureHistory, Scenario, SiteModel, StableDecision, load_model, public_snapshot, schema,
)
from ghost_champions_v2.defender_macro_v1.controller import GCDefenderMacro, GhostChampionsV2DefenderController
from ghost_champions_v1_macro import GhostChampionsV1DefenderController


def unit(name, position, team="D", known=True):
    return NS(name=name, pos=list(position), team=team, position_known=known,
              is_alive=True, hp=100, facing="S", blind_remaining=0)


class DefenderMacroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scenario = Scenario()

    def setUp(self):
        self.analysis = AttackSiteAnalysis(scenario=self.scenario, model=SiteModel(len(GCFeatureHistory(self.scenario).fields)))
        self.allies = [unit(str(i), post.watch) for i, post in enumerate(self.scenario.posts)]
        self.state = dict(grid=self.scenario.grid, chars=self.allies, battle_tick=1, round_timer=100,
                          is_planted=False, spike_pos=None, planted_pos=None)

    def ready_history(self):
        self.analysis.previous_rounds = [dict(site=s, winner="A", contacts=dict(A=0, Mid=0, B=0)) for s in ("L", "R")*2]

    def test_hidden_enemy_trajectory_has_no_effect_on_inputs(self):
        enemy = unit("hidden", (8, 3), "A", False)
        self.state["chars"] += [enemy]
        first = self.analysis.history.encode(public_snapshot(self.allies[0], self.state, 1), [])
        enemy.pos = [7, 40]
        enemy.hp = 1
        enemy.has_spike = True
        second = GCFeatureHistory(self.scenario).encode(public_snapshot(self.allies[0], self.state, 1), [])
        np.testing.assert_array_equal(first, second)

    def test_setup_cannot_observe_raw_enemy_positions(self):
        self.state.update(defender_setup_active=True)
        self.state["chars"] += [unit("enemy", (8, 3), "A")]
        self.analysis.observe(self.allies[0], self.state, 1)
        self.assertEqual(self.analysis.frames, [])
        self.assertEqual(self.analysis.history.tracks, {})

    def test_public_history_survives_reset_but_live_tracks_do_not(self):
        self.state["chars"] += [unit("seen", (8, 3), "A")]
        self.analysis.observe(self.allies[0], self.state, 1)
        self.assertTrue(self.analysis.history.tracks)
        self.state.update(is_planted=True, planted_pos=(8, 3), battle_tick=2)
        self.analysis.observe(self.allies[0], self.state, 1)
        self.analysis.finish_round(winner="A")
        self.analysis.reset_round()
        self.assertEqual(self.analysis.previous_rounds[0]["site"], "L")
        self.assertEqual(self.analysis.previous_rounds[0]["contacts"]["A"], 1)
        self.assertEqual(self.analysis.history.tracks, {})
        self.state.update(is_planted=False, planted_pos=None, battle_tick=1)
        self.state["chars"] = self.allies
        self.analysis.observe(self.allies[0], self.state, 2)
        feature = self.analysis.frames[-1]["features"]
        fields = self.analysis.history.fields
        self.assertEqual(feature[fields.index("recent_1_A")], 1)
        self.assertEqual(feature[fields.index("recent_1_contacts_A")], .2)

    def test_no_public_plant_leaks_into_preplant_frames(self):
        self.analysis.observe(self.allies[0], self.state, 1)
        before = self.analysis.frames[0]["features"].copy()
        self.state.update(is_planted=True, planted_pos=(7, 40), battle_tick=2)
        self.analysis.observe(self.allies[0], self.state, 1)
        self.assertEqual(len(self.analysis.frames), 1)
        np.testing.assert_array_equal(before, self.analysis.frames[0]["features"])

    def test_prediction_runs_once_per_tick_not_once_per_defender(self):
        self.state["chars"] += [unit("seen1", (8, 3), "A"), unit("seen2", (9, 3), "A")]
        with patch.object(self.analysis.model, "probabilities", return_value=np.array([.9, .1])) as predict:
            for c in self.allies:
                self.analysis.observe(c, self.state, 1)
            self.assertEqual(predict.call_count, 1)
            self.assertIsNone(self.analysis.gate.side)
            for tick in (2, 3):
                self.state["battle_tick"] = tick
                self.analysis.observe(self.allies[0], self.state, 1)
            self.assertEqual(self.analysis.gate.side, "A")

    def test_uncertain_predictions_expire_and_reversals_need_confirmation(self):
        gate = StableDecision()
        for tick in range(3):
            gate.update([.9, .1], tick)
        self.assertEqual(gate.side, "A")
        gate.update([.1, .9], 3)
        gate.update([.9, .1], 4)
        self.assertEqual(gate.side, "A")
        for tick in range(5, 13):
            gate.update([.5, .5], tick)
        self.assertIsNone(gate.side)
        for tick in range(13, 16):
            gate.update([.1, .9], tick)
        self.assertEqual(gate.side, "B")

    def test_skipped_ticks_do_not_count_as_consecutive_confirmation(self):
        gate = StableDecision()
        for tick in (1, 10, 20):
            gate.update([.9, .1], tick)
        self.assertIsNone(gate.side)
        for tick in (21, 22):
            gate.update([.9, .1], tick)
        self.assertEqual(gate.side, "A")

    def test_round_winner_stays_correct_when_score_counters_swap_sides(self):
        with patch.object(GhostChampionsV1DefenderController, "__init__", return_value=None), \
             patch.object(GhostChampionsV1DefenderController, "reset_round"), \
             patch.object(GhostChampionsV1DefenderController, "decide_move", return_value=([1, 2], "MOVE")):
            c = GhostChampionsV2DefenderController(analysis=self.analysis)
            c.game = NS(current_round=24, attacker_wins=3, defender_wins=10,
                        attacker_team_name="enemy", defender_team_name="GC")
            c.decide_move(self.allies[0], self.state)
            c.game.attacker_team_name, c.game.defender_team_name = "GC", "enemy"
            c.game.attacker_wins, c.game.defender_wins = 11, 3
            c.reset_round()
            self.assertEqual(self.analysis.previous_rounds[-1]["winner"], "D")

    def test_model_confidence_does_not_create_unobserved_rotation(self):
        self.ready_history()
        macro = GCDefenderMacro(self.analysis)
        self.analysis.gate.side = "A"
        sentinel = (self.allies[0].pos, {"facing": "NE"})
        self.assertIs(macro.coordinate(self.allies[0], self.state, sentinel), sentinel)
        self.assertEqual(macro.goals, {})

    def test_long_range_engagement_is_not_overridden_by_rotation(self):
        self.ready_history()
        macro = GCDefenderMacro(self.analysis)
        self.analysis.gate.side = "B"
        self.state["grid"] = np.zeros_like(self.scenario.grid)
        actor = self.allies[4]
        self.state["chars"] += [unit("seen", (12, 38), "A")]
        base = (actor.pos, {"facing": "S"})
        result = macro.coordinate(actor, self.state, base)
        self.assertEqual(result[0], actor.pos)
        from gc_v1.gc_facing import facing_towards
        self.assertEqual(result[1]["facing"], facing_towards(actor.pos, (12, 38)))

    def test_smoke_or_unknown_enemy_keeps_lower_controller_response(self):
        self.ready_history()
        for hidden, smoke in ((True, ()), (False, {(8, 38)})):
            macro = GCDefenderMacro(self.analysis)
            self.analysis.gate.side = "B"
            self.state["grid"] = np.zeros_like(self.scenario.grid)
            actor = self.allies[4]
            actor.pos = [3, 38]
            self.state["chars"] = self.allies + [unit("seen", (12, 38), "A", not hidden)]
            self.state["smoke_cells"] = smoke
            self.assertIs(macro.coordinate(actor, self.state, actor.pos), actor.pos)

    def test_iq_wrapper_and_real_movement_apply_corrected_facing_each_tick(self):
        from ghost_champions_v2.defender_macro_v1.tools.watch_defender_gc import build_match
        from simulation_runtime import cpu_inference
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
            game, defender = build_match("GG", seed=127, headless=True)
            game.defender_setup_phase.finish()
            actor = next(c for c in game.chars if c.team == "D")
            enemy = next(c for c in game.chars if c.team == "A")
            for c in game.chars:
                c.is_alive = c in (actor, enemy)
            actor.pos, enemy.pos = [6, 22], [6, 27]
            actor.iq = 200
            enemy.reveal_remaining = 20
            def backwards(char, state):
                char.facing = "W"
                return list(char.pos), {"facing": "W"}
            with patch.object(GhostChampionsV1DefenderController, "decide_move", side_effect=backwards):
                for tick in range(1, 5):
                    game.battle_tick = tick
                    game.defender_controller.perception_engine.clear_cache()
                    actor.facing = "W"
                    game.move_character(actor)
                    self.assertEqual(actor.facing, "E")
                    self.assertFalse(actor.moved_this_tick)
                actor.forced_facing_next_tick = "N"
                game.battle_tick += 1
                game.defender_controller.perception_engine.clear_cache()
                game.move_character(actor)
                self.assertEqual(actor.facing, "N")

    def test_skill_retake_and_movement_lock_remain_owned_by_lower_controller(self):
        macro = GCDefenderMacro(self.analysis)
        self.analysis.gate.side = "A"
        for command in ({"ability": "RECON"}, {"ultimate": "RAID"}, "DEFUSE"):
            result = (self.allies[0].pos, command)
            self.assertIs(macro.coordinate(self.allies[0], self.state, result), result)
        sentinel = (self.allies[0].pos, "MOVE")
        self.state["is_planted"] = True
        self.assertIs(macro.coordinate(self.allies[0], self.state, sentinel), sentinel)
        self.state["is_planted"] = False
        self.allies[0].fate_loom_remaining = 2
        self.assertIs(macro.coordinate(self.allies[0], self.state, sentinel), sentinel)

    def test_unknown_prediction_keeps_balanced_allocation_and_dead_players_get_no_goal(self):
        macro = GCDefenderMacro(self.analysis)
        sentinel = (self.allies[0].pos, "MOVE")
        macro.coordinate(self.allies[0], self.state, sentinel)
        self.assertEqual(macro.allocation, dict(A=2, Mid=1, B=2))
        self.analysis.gate.side = "B"
        self.allies[0].is_alive = False
        macro.coordinate(self.allies[1], self.state, sentinel)
        self.assertNotIn("0", macro.goals)

    def test_lower_movement_and_facing_survive_history_bias_and_prediction(self):
        self.analysis.previous_rounds = [dict(site="R", rush_site="B")]*6
        macro = GCDefenderMacro(self.analysis)
        self.analysis.gate.side = "B"
        for c in self.allies:
            result = ([c.pos[0], c.pos[1]], "MOVE", dict(facing="NW", move_step_limit=1))
            self.assertIs(macro.coordinate(c, self.state, result), result)

    def test_schema_mismatch_rejects_model_and_load_preserves_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.pt"
            payload = dict(schema=schema(self.scenario), model=self.analysis.model.state_dict(), labeled_rounds=2)
            torch.save(payload, path)
            before = torch.random.get_rng_state()
            model, _ = load_model(path, self.scenario)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertIsInstance(model, SiteModel)
            payload["schema"]["sensor"] = "hidden_truth"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_model(path, self.scenario)

    def test_actual_controller_uses_base_until_history_ready_and_preserves_retake(self):
        with patch.object(GhostChampionsV1DefenderController, "__init__", return_value=None), \
             patch.object(GhostChampionsV1DefenderController, "decide_move", return_value=([1, 2], "DEFUSE")) as lower:
            c = GhostChampionsV2DefenderController(analysis=self.analysis)
            c.game = NS(current_round=1, attacker_wins=0, defender_wins=0)
            self.state["defender_setup_active"] = True
            result = c.decide_move(self.allies[0], self.state)
            self.assertEqual(result, ([1, 2], "DEFUSE"))
            self.assertEqual(c.macro.allocation, dict(A=2, Mid=1, B=2))
            self.assertEqual(self.analysis.frames, [])
            self.state.update(defender_setup_active=False, is_planted=True, planted_pos=(8, 3))
            self.assertEqual(c.decide_move(self.allies[0], self.state), ([1, 2], "DEFUSE"))
            self.assertEqual(lower.call_count, 2)

    def test_live_defender_launcher_uses_checkpoint_and_iq_in_real_engine(self):
        from ghost_champions_v2.defender_macro_v1.tools.watch_defender_gc import build_match
        from simulation_runtime import cpu_inference
        from iq_controller_adapter import IQAwareController
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.pt"
            torch.save(dict(schema=schema(self.scenario), model=self.analysis.model.state_dict(), labeled_rounds=2), path)
            with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
                game, defender = build_match("OMG", checkpoint=path, seed=127, headless=True)
                game.stop_after_round = True
                game._record_replay_frame = lambda: None
                self.assertIsInstance(game.defender_controller, IQAwareController)
                self.assertIs(game.defender_controller.inner, defender)
                for _ in range(35):
                    game.step_tick()
                    if game.round_over:
                        break
            self.assertTrue(defender.macro.analysis.frames)
            self.assertIsNotNone(defender.macro.analysis.model)
            self.assertEqual(defender.macro.analysis.previous_rounds, [])

    def test_match_seed_also_reproduces_private_opening_randomness(self):
        from ghost_champions_v2.defender_macro_v1.tools.watch_defender_gc import build_match
        from simulation_runtime import cpu_inference
        with contextlib.redirect_stdout(io.StringIO()), cpu_inference():
            for opponent in ("OMG", "GG"):
                traces = []
                for _ in range(2):
                    game, defender = build_match(opponent, seed=814, headless=True)
                    game.stop_after_round = True
                    game._record_replay_frame = lambda: None
                    trace = []
                    for _ in range(85):
                        game.step_tick()
                        trace.append(tuple((c.name, tuple(c.pos), c.hp, c.is_alive) for c in game.chars))
                        if game.round_over:
                            break
                    traces.append((trace, defender.opening_macro_controller.rng.getstate()))
                self.assertEqual(traces[0], traces[1], opponent)


if __name__ == "__main__":
    unittest.main()
