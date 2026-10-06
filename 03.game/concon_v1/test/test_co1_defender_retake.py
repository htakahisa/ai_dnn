"""Retake utility legality, learned choices, coordination and phase isolation."""

from pathlib import Path
from collections import Counter
import contextlib
import io
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_retake_scenarios import get_scenario, build_scenario, plant_site, validate_checkpoint, normalize_site_ability_distances
from concon_v1.co1_retake_common import (
    RetakeDQN, build_inputs, coordination, ACTION_DIM, DEFUSE_ACTION,
    ULTIMATE_ACTION, GORIGONS, observation_dim, decision_reward, decode_action,
)
from concon_v1.co1_learn_defender_retake import ConconDefenderRetakeController
from concon_v1.co1_defender_retake_training import DefenderRetakeEnv, RetakeTrainingAdapter, TrainingRetakeController
from concon_v1.co1_train_defender_retake import optimize, summarize, make_checkpoint, main, epsilon_by_episode, DEFAULT_ABILITY_DISTANCES
from concon_v1.co1_train_defender_retake import balanced_retake_schedule, iter_retake_rounds, evaluate, CHECKPOINT_INTERVAL
from concon_v1.co1_retake_logging import print_summary, print_retry_progress, print_evaluation_summary, GREEN, RESET
from concon_v1.co1_map_retake_L import MAZE_STR


def actor(name, pos, team="D", ability="NONE", known=True):
    return SimpleNamespace(name=name, pos=list(pos), team=team, facing="W", is_alive=True,
        position_known=known, hp=100, max_hp=100, blind_remaining=0, stun_remaining=0,
        reveal_remaining=0, moved_this_tick=False, facing_forced_this_tick=False,
        ability_name=ability, smoke_charges=int(ability == "SMOKE"), flash_charges=int(ability == "FLASH"),
        recon_charges=int(ability == "RECON"), ultimate_points=0, ultimate_cost=5, ultimate_name="NEON",
        defuse_timer=0, round_kills=0)


class RetryFixture:
    """Repeat no plant, plant without a defender action, then a lost retake."""
    def __init__(self):
        self.attempts = Counter()
        self.epsilons = []

    def reset(self, opponent):
        self.opponent, self.done = opponent, False
        self.attempts[opponent] += 1

    def step(self, epsilon):
        self.epsilons.append(epsilon)
        self.done = True
        return [], [0.] * 5, True

    def result(self):
        kind = (self.attempts[self.opponent] - 1) % 3
        return dict(opponent=self.opponent, planted=kind != 0, site="L" if kind != 0 else None,
                    excluded_from_retake=kind != 2, retake_decisions=5 if kind == 2 else 0,
                    defused=False, fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)


class SiteRetryFixture(RetryFixture):
    def result(self):
        kind = (self.attempts[self.opponent] - 1) % 4
        site = "R" if kind == 2 else "L" if kind else None
        return dict(opponent=self.opponent, planted=kind != 0, site=site,
                    excluded_from_retake=kind < 2, retake_decisions=5 if kind >= 2 else 0,
                    defused=kind == 3, fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)


class RetakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.scenario = get_scenario("L")
        self.model = RetakeDQN(self.scenario)
        self.controller = ConconDefenderRetakeController("L", model=self.model)
        self.char = actor(GORIGONS.players[0], (7, 3))
        self.state = dict(grid=self.scenario.grid, chars=[self.char], is_planted=True,
                          planted_pos=(7, 3), smoke_cells=[], battle_tick=1, detonate_timer=40,
                          defender_defuse_info={})

    def test_maps_and_signatures(self):
        self.assertNotEqual(self.scenario.signature, get_scenario("R").signature)
        self.assertEqual(len(self.scenario.points["FLASH"]), 2)
        self.assertEqual(plant_site((7, 3), self.scenario.grid), "L")
        with self.assertRaises(ValueError):
            build_scenario("L", MAZE_STR.replace("111111", "011111", 1))

    def test_observation_forward_and_defuse_decode(self):
        obs, mask, context = build_inputs(self.controller, self.char, self.state)
        self.assertEqual(len(obs), observation_dim(self.scenario))
        self.assertEqual(self.model(torch.tensor(obs)[None]).shape, (1, ACTION_DIM))
        self.assertTrue(mask[DEFUSE_ACTION])
        self.assertEqual(decode_action(DEFUSE_ACTION, self.char, context), ([7, 3], "DEFUSE"))
        self.state["defender_defuse_info"] = {"other": (1, 6)}
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][DEFUSE_ACTION])

    def test_smoke_spike_range_without_wall_los(self):
        self.char.ability_name, self.char.smoke_charges = "SMOKE", 1
        self.char.pos = [12, 6]
        self.state["planted_pos"] = (7, 3)
        self.controller.ability_distances["SMOKE"] = 20
        from concon_v1.co1_retake_common import wall_clear
        self.assertFalse(wall_clear(self.scenario.grid, tuple(self.char.pos), self.state["planted_pos"]))
        _, mask, context = build_inputs(self.controller, self.char, self.state)
        action = 40 + 6  # current W facing
        self.assertTrue(mask[action])
        self.assertTrue(mask[32:40].any())  # casting does not force resource use
        self.assertEqual(decode_action(action, self.char, context)[1]["target"], (7, 3))
        self.controller.ability_distances["SMOKE"] = 1
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][action])

    def test_flash_full_wall_los_and_two_targets(self):
        self.char.ability_name, self.char.flash_charges = "FLASH", 1
        self.char.pos = [7, 3]
        self.controller.ability_distances["FLASH"] = 20
        _, mask, context = build_inputs(self.controller, self.char, self.state)
        self.assertTrue(mask[78])
        self.assertTrue(mask[86])
        for target_index, point in enumerate(context["ability_targets"]):
            action = (5 + 3 + target_index) * 8 + 6
            if mask[action]:
                self.assertIsNotNone(point)
                from concon_v1.co1_retake_common import wall_clear
                self.assertTrue(wall_clear(self.scenario.grid, tuple(self.char.pos), point))
        # Both markers are blocked from this source; a partial projectile path
        # through the first cell must not be accepted as full target LOS.
        self.char.pos = [12, 6]
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][64:88].any())

    def test_hidden_enemy_never_enters_inputs_or_ultimate_mask(self):
        hidden = actor("enemy", (8, 3), "A", known=False)
        self.state["chars"].append(hidden)
        self.char.ultimate_points = 5
        first, mask, _ = build_inputs(self.controller, self.char, self.state)
        self.assertFalse(mask[ULTIMATE_ACTION:DEFUSE_ACTION].any())
        hidden.pos = [5, 38]
        second = build_inputs(self.controller, self.char, self.state)[0]
        np.testing.assert_array_equal(first, second)
        hidden.pos, hidden.position_known = [8, 3], True
        self.assertTrue(build_inputs(self.controller, self.char, self.state)[1][ULTIMATE_ACTION:DEFUSE_ACTION].any())

    def test_independent_ability_limits_and_unrestricted_ultimate_distance(self):
        self.controller.ability_distances = dict(FLASH=1, RECON=1, SMOKE=8)
        self.char.ability_name, self.char.flash_charges = "FLASH", 1
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][64:88].any())
        self.controller.ability_distances["FLASH"] = 2
        self.assertTrue(build_inputs(self.controller, self.char, self.state)[1][64:88].any())
        self.char.pos = [12, 3]
        self.char.ability_name, self.char.recon_charges = "RECON", 1
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][88:112].any())
        self.controller.ability_distances["RECON"] = 2
        self.assertTrue(build_inputs(self.controller, self.char, self.state)[1][88:112].any())
        self.char.ability_name, self.char.smoke_charges = "SMOKE", 1
        self.assertTrue(build_inputs(self.controller, self.char, self.state)[1][46])
        self.controller.ability_distances["SMOKE"] = 1
        self.assertFalse(build_inputs(self.controller, self.char, self.state)[1][46])
        self.char.ultimate_points = 5
        self.state["chars"].append(actor("enemy", (12, 5), "A"))
        self.controller.ability_distances = dict(FLASH=1, RECON=1, SMOKE=1)
        self.assertTrue(build_inputs(self.controller, self.char, self.state)[1][ULTIMATE_ACTION:DEFUSE_ACTION].any())

    def test_checkpoint_preserves_individual_limits_and_accepts_legacy_shared_limit(self):
        distances = dict(FLASH=4, RECON=6, SMOKE=8)
        checkpoint = make_checkpoint(self.model, "L", 1, distances, "search.pt", [], 1)
        validate_checkpoint(checkpoint, self.scenario, distances)
        with self.assertRaises(ValueError):
            validate_checkpoint(checkpoint, self.scenario, 6)
        with patch("concon_v1.co1_learn_defender_retake.torch.load", return_value=checkpoint):
            controller = ConconDefenderRetakeController("L")
        self.assertEqual(controller.ability_distances, distances)
        checkpoint.pop("ability_distances")
        checkpoint["ability_distance"] = 6
        validate_checkpoint(checkpoint, self.scenario, dict(FLASH=6, RECON=6, SMOKE=6))

    def test_cli_individual_distances_and_defaults(self):
        with patch("concon_v1.co1_train_defender_retake.train") as train:
            main([])
            self.assertEqual(train.call_args.kwargs["ability_distance"], DEFAULT_ABILITY_DISTANCES)
            main(["--flash-distance", "4", "--recon-distance", "5", "--smoke-distance", "8"])
            self.assertEqual(train.call_args.kwargs["ability_distance"], {site: dict(FLASH=4, RECON=5, SMOKE=8) for site in ("L", "R")})
            main(["--ability-distance", "7", "--flash-distance", "3"])
            self.assertEqual(train.call_args.kwargs["ability_distance"], {site: dict(FLASH=3, RECON=7, SMOKE=7) for site in ("L", "R")})

    def test_cli_left_and_right_ability_limits_and_precedence(self):
        with patch("concon_v1.co1_train_defender_retake.train") as train:
            main(["--flash-distance-l", "3", "--recon-distance-l", "4", "--smoke-distance-l", "5",
                  "--flash-distance-r", "8", "--recon-distance-r", "9", "--smoke-distance-r", "10"])
            self.assertEqual(train.call_args.kwargs["ability_distance"],
                             dict(L=dict(FLASH=3, RECON=4, SMOKE=5), R=dict(FLASH=8, RECON=9, SMOKE=10)))
            main(["--ability-distance", "6", "--flash-distance", "7", "--left-flash-distance", "2"])
            self.assertEqual(train.call_args.kwargs["ability_distance"],
                             dict(L=dict(FLASH=2, RECON=6, SMOKE=6), R=dict(FLASH=7, RECON=6, SMOKE=6)))

    def test_environment_controllers_use_their_own_site_limits(self):
        distances = dict(L=dict(FLASH=3, RECON=4, SMOKE=5), R=dict(FLASH=8, RECON=9, SMOKE=10))
        models = {site: RetakeDQN(get_scenario(site)) for site in ("L", "R")}
        env = DefenderRetakeEnv(models, None, ability_distance=distances)
        for site in ("L", "R"):
            controller = TrainingRetakeController(env, site)
            self.assertEqual(controller.ability_distances, distances[site])
        expanded = normalize_site_ability_distances(6)
        expanded["L"]["FLASH"] = 1
        self.assertEqual(expanded["R"]["FLASH"], 6)
        with self.assertRaises(ValueError):
            normalize_site_ability_distances({"L": distances["L"]})

    def test_site_specific_checkpoint_loading_and_resume_validation(self):
        distances = dict(L=dict(FLASH=3, RECON=4, SMOKE=5), R=dict(FLASH=8, RECON=9, SMOKE=10))
        for site in ("L", "R"):
            scenario = get_scenario(site)
            model = RetakeDQN(scenario)
            checkpoint = make_checkpoint(model, site, 1, distances, "search.pt", [], 1)
            self.assertEqual(checkpoint["ability_distances"], distances[site])
            validate_checkpoint(checkpoint, scenario, distances[site])
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, scenario, distances["R" if site == "L" else "L"])
            with patch("concon_v1.co1_learn_defender_retake.torch.load", return_value=checkpoint):
                controller = ConconDefenderRetakeController(site)
            self.assertEqual(controller.ability_distances, distances[site])

    def test_safe_wait_and_deadline(self):
        grid = np.zeros((3, 30), dtype=np.int32)
        near = actor(GORIGONS.players[0], (1, 7))
        far = actor(GORIGONS.players[1], (1, 20))
        from concon_v1.co1_retake_coordination import RetakeAssembly
        scenario = SimpleNamespace(grid=grid, rally_points=((1, 7), (1, 8)),
                                   rally_groups=(((1, 7), (1, 8)),))
        controller = SimpleNamespace(assembly=RetakeAssembly(scenario))
        state = dict(grid=grid, chars=[near, far], planted_pos=(1, 0), detonate_timer=40)
        context = coordination(near, state, controller)
        self.assertTrue(context["waiting"])
        self.assertTrue(context["safe"])
        state["detonate_timer"] = 19
        state["battle_tick"] = 1
        context = coordination(near, state, controller)
        self.assertFalse(context["waiting"])
        self.assertTrue(context["urgent"])
        self.assertEqual(context["goal"], (1, 0))

    def test_epsilon_schedule_and_invalid_settings(self):
        self.assertEqual(epsilon_by_episode(1, 100), .15)
        self.assertEqual(epsilon_by_episode(70, 100), .05)
        self.assertEqual(epsilon_by_episode(100, 100), .05)
        values = [epsilon_by_episode(round, 100, .4, .1, .5) for round in range(1, 101)]
        self.assertEqual(values[0], .4)
        self.assertEqual(values[49], .1)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertEqual(epsilon_by_episode(1, 100, 0., 0.), 0.)
        for start, end, ratio in ((1.1, .1, .7), (.1, .2, .7), (1., -.1, .7),
                                  (1., .1, 0.), (1., .1, 1.1), (float("nan"), .1, .7)):
            with self.assertRaises(ValueError):
                epsilon_by_episode(1, 100, start, end, ratio)

    def test_cli_epsilon_and_force_save_settings(self):
        with patch("concon_v1.co1_train_defender_retake.train") as train:
            main([])
            self.assertEqual(train.call_args.kwargs["epsilon_start"], .15)
            self.assertEqual(train.call_args.kwargs["epsilon_end"], .05)
            self.assertEqual(train.call_args.kwargs["epsilon_decay_ratio"], .7)
            self.assertEqual(train.call_args.kwargs["checkpoint_interval"], CHECKPOINT_INTERVAL)
            self.assertFalse(train.call_args.kwargs["force_save"])
            main(["--epsilon-start", ".3", "--epsilon-end", ".02", "--epsilon-decay-ratio", ".5", "--force-save"])
            self.assertEqual(train.call_args.kwargs["epsilon_start"], .3)
            self.assertEqual(train.call_args.kwargs["epsilon_end"], .02)
            self.assertEqual(train.call_args.kwargs["epsilon_decay_ratio"], .5)
            self.assertTrue(train.call_args.kwargs["force_save"])
            main(["--no-force-save"])
            self.assertFalse(train.call_args.kwargs["force_save"])

    def test_stationary_shooting_and_smoke_defuse_rewards(self):
        enemy = actor("enemy", (8, 3), "A")
        self.state["chars"].append(enemy)
        _, _, context = build_inputs(self.controller, self.char, self.state)
        still = decision_reward(36, context, (7, 3), "S")
        moving = decision_reward(4, context, (8, 3), "S")
        self.assertGreater(still, moving)
        self.state["smoke_cells"] = [(7, 3), (8, 3)]
        _, _, context = build_inputs(self.controller, self.char, self.state)
        self.assertTrue(context["smoke_defuse"])
        self.assertGreater(decision_reward(DEFUSE_ACTION, context, (7, 3), "S"), 0)

    def test_search_and_retake_dispatch_same_tick(self):
        search = SimpleNamespace(scenario=self.scenario, decide_move=lambda char, state: "search")
        left = SimpleNamespace(decide_move=lambda char, state: "left")
        right = SimpleNamespace(decide_move=lambda char, state: "right")
        adapter = RetakeTrainingAdapter(search, dict(L=left, R=right))
        self.state["is_planted"] = False
        self.assertEqual(adapter.decide_move(self.char, self.state), "search")
        self.state["is_planted"] = True
        self.assertEqual(adapter.decide_move(self.char, self.state), "left")
        self.state["planted_pos"] = (7, 38)
        self.assertEqual(adapter.decide_move(self.char, self.state), "right")

    def test_terminal_transition_keeps_site_and_duration(self):
        env = object.__new__(DefenderRetakeEnv)
        obs = np.zeros(observation_dim(self.scenario), dtype=np.float16)
        mask = np.zeros(ACTION_DIM, dtype=bool)
        mask[32] = True
        env.retake_pending = [dict(site="L", obs=obs, action=32, reward=3., duration=2)]
        env.transitions = []
        env.finish_retake_pending(0, obs, mask, True)
        self.assertEqual(env.transitions[0][0], "L")
        self.assertEqual(env.transitions[0][1][-2:], (1., 2))
        optimizer = torch.optim.Adam(self.model.parameters())
        self.assertIsNotNone(optimize(self.model, self.model, optimizer, [env.transitions[0][1]], batch_size=1))

    def test_preplant_round_has_no_retake_score(self):
        record = dict(opponent="omoko_v1", site=None, planted=False, defused=False,
                      fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)
        result = summarize([record], ["omoko_v1"])
        self.assertEqual(result["L"]["retakes"], 0)
        self.assertIsNone(result["L"]["mean_defuse_rate"])
        self.assertEqual(result["L"]["opponents"]["omoko_v1"]["excluded_preplant"], 1)

    def test_each_fifty_episode_window_contains_ten_retakes_per_team(self):
        opponents = [f"team{index}" for index in range(5)]
        schedule = list(balanced_retake_schedule(opponents, 113, 50, random.Random(0)))
        self.assertEqual(Counter(schedule[:50]), Counter(dict.fromkeys(opponents, 10)))
        self.assertEqual(Counter(schedule[50:100]), Counter(dict.fromkeys(opponents, 10)))
        tail = Counter(schedule[100:])
        self.assertEqual(sum(tail.values()), 13)
        self.assertLessEqual(max(tail.values()) - min(tail.values()), 1)

    def test_excluded_rounds_retry_same_opponent_without_consuming_episode(self):
        opponents = [f"team{index}" for index in range(5)]
        env, records = RetryFixture(), []
        for opponent in balanced_retake_schedule(opponents, 50, 50, random.Random(0)):
            attempted = list(iter_retake_rounds(env, opponent, 1, epsilon=.3))
            self.assertEqual([record["counted_episode"] for record in attempted], [False, False, True])
            self.assertTrue(all(record["opponent"] == opponent for record in attempted))
            records.extend(attempted)
        self.assertEqual(len(records), 150)
        self.assertEqual(sum(record["counted_episode"] for record in records), 50)
        self.assertTrue(all(epsilon == .3 for epsilon in env.epsilons))
        summary = summarize(records, opponents)
        for opponent in opponents:
            metrics = summary["L"]["opponents"][opponent]
            self.assertEqual(metrics["retakes"], 10)
            self.assertEqual(metrics["rounds"], 30)
            self.assertEqual(metrics["excluded_no_retake"], 20)
            self.assertEqual(metrics["defuses"], 0)  # losses still count

    def test_evaluation_retries_until_each_opponent_has_required_retakes(self):
        env = SiteRetryFixture()
        with patch("concon_v1.co1_train_defender_retake.DefenderRetakeEnv", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate({}, None, rounds=2, opponents=["one", "two"])
        self.assertEqual(env.attempts, Counter(one=4, two=4))
        for opponent in ("one", "two"):
            self.assertEqual(summary["L"]["opponents"][opponent]["retakes"], 1)
            self.assertEqual(summary["R"]["opponents"][opponent]["retakes"], 1)
            self.assertEqual(summary["L"]["opponents"][opponent]["excluded_other_site"], 1)
            self.assertEqual(summary["R"]["opponents"][opponent]["excluded_other_site"], 1)
        self.assertEqual(summary["L"]["mean_defuse_rate"], 1.)
        self.assertEqual(summary["R"]["mean_defuse_rate"], 0.)
        self.assertTrue(all(epsilon == 0. for epsilon in env.epsilons))

    def test_evaluation_defaults_to_checkpoint_interval_total_retakes(self):
        env = SiteRetryFixture()
        opponents = [f"team{index}" for index in range(5)]
        with patch("concon_v1.co1_train_defender_retake.DefenderRetakeEnv", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate({}, None, opponents=opponents)
        self.assertEqual(sum(summary[site]["retakes"] for site in ("L", "R")), CHECKPOINT_INTERVAL)
        for site in ("L", "R"):
            for opponent in opponents:
                self.assertEqual(summary[site]["opponents"][opponent]["retakes"], CHECKPOINT_INTERVAL // 10)
            self.assertEqual(summary[site]["required_retakes_per_opponent"], CHECKPOINT_INTERVAL // 5)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_evaluation_summary(summary)
        left_text, right_text = output.getvalue().split("R model evaluation:")
        self.assertIn("L model evaluation:", left_text)
        per_site = CHECKPOINT_INTERVAL // 10
        self.assertIn(f"defender wins {per_site}/{per_site} (100.0%)", left_text)
        self.assertIn(f"defender wins 0/{per_site} (0.0%)", right_text)
        self.assertNotIn("defender wins 10/20", output.getvalue())

    def test_evaluation_does_not_wait_for_a_site_the_team_never_attacks(self):
        env = RetryFixture()
        with patch("concon_v1.co1_train_defender_retake.DefenderRetakeEnv", return_value=env), contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate({}, None, rounds=10, opponents=["one"])
        self.assertEqual(summary["L"]["retakes"], 10)
        self.assertEqual(summary["R"]["retakes"], 0)
        self.assertIsNone(summary["R"]["mean_defuse_rate"])

    def test_readable_team_log_uses_rounds_once_and_highlights_rates(self):
        base = dict(opponent="one", planted=True, site="L", excluded_from_retake=False, defused=True,
                    fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)
        records = [base, dict(base, site="R", defused=False),
                   dict(base, planted=False, site=None, defused=False, excluded_from_retake=True),
                   dict(base, defused=False, excluded_from_retake=True)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summarize(records, ["one"]), "Training summary:")
        text = output.getvalue()
        self.assertIn("one: defender wins 1/2 (50.0%) | losses=1 | L=1/1 R=0/1", text)
        self.assertIn("retake_rate=2/4 (50.0%)", text)
        self.assertIn("defuse_rate=1/2 (50.0%) | excluded=2", text)
        self.assertNotIn("2/8", text)  # site summaries share the round denominator
        highlights = [line.strip() for line in text.splitlines() if GREEN in line]
        self.assertEqual(len(highlights), 4)
        self.assertTrue(all(line.startswith(GREEN) and line.endswith(RESET) for line in highlights))

    def test_console_log_handles_missing_retakes_and_retry_progress(self):
        record = dict(opponent="one", planted=False, site=None, defused=False)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summarize([record], ["one"]), "Evaluation summary:")
            print_retry_progress("one", 0, 10, 20)
        text = output.getvalue()
        self.assertIn("defuse_rate=0/0 (N/A)", text)
        self.assertIn("mean_defuse_rate=N/A", text)
        self.assertIn("completed=0/10 | attempted_rounds=20", text)
        self.assertIn("retake_rate=0/20 (0.0%) | excluded=20", text)

    def test_console_loss_reasons_count_retakes_once_and_exclude_search_outcomes(self):
        base = dict(opponent="one", planted=True, site="L", excluded_from_retake=False, defused=False,
                    fire_decisions=0, moving_fire_decisions=0, smoke_defuse_decisions=0)
        records = [dict(base, end_reason="detonated"),
                   dict(base, site="R", end_reason="timeout"),
                   dict(base, site="R", end_reason="defender_eliminated"),
                   dict(base, defused=True, end_reason="defused"),
                   dict(base, planted=False, site=None, excluded_from_retake=True, end_reason="defender_eliminated"),
                   dict(base, excluded_from_retake=True, end_reason="defender_eliminated")]
        summary = summarize(records, ["one"])
        self.assertEqual(summary["L"]["time_expired"], 1)
        self.assertEqual(summary["R"]["time_expired"], 1)
        self.assertEqual(summary["L"]["defender_eliminated"], 0)
        self.assertEqual(summary["R"]["defender_eliminated"], 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary, "Evaluation summary:")
        text = output.getvalue()
        team_line = next(line for line in text.splitlines() if "one: defender wins" in line)
        self.assertIn("losses=3", team_line)
        self.assertIn("time_expired=2 defender_eliminated=1", team_line)
        overall_line = next(line for line in text.splitlines() if "Overall:" in line)
        self.assertIn("time_expired=2 defender_eliminated=1", overall_line)

    def test_iq_wrapped_environment_reset(self):
        from concon_v1.co1_defender_scenario import get_scenario as search_scenario
        from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN
        from iq_controller_adapter import IQAwareController
        models = {site: RetakeDQN(get_scenario(site)) for site in ("L", "R")}
        env = DefenderRetakeEnv(models, DefenderSearchBattleDQN(search_scenario()), opponents=["omoko_v1"])
        env.reset(opponent="omoko_v1")
        self.assertIsInstance(env.game.defender_controller, IQAwareController)
        self.assertIs(env.game.defender_controller.inner, env.adapter)
        transitions, _, _ = env.step()
        self.assertFalse(transitions)
        self.assertTrue(all(pending is None for pending in env.retake_pending))

    def test_engine_defuse_action_and_postplant_terminal_credit(self):
        from concon_v1.co1_defender_scenario import get_scenario as search_scenario
        from concon_v1.co1_defender_search_common import DefenderSearchBattleDQN
        models = {site: RetakeDQN(get_scenario(site)) for site in ("L", "R")}
        with torch.no_grad():
            for model in models.values():
                for parameter in model.parameters():
                    parameter.zero_()
                model.head[2].bias[DEFUSE_ACTION] = 10
                model.head[2].bias[32] = 1
        env = DefenderRetakeEnv(models, DefenderSearchBattleDQN(search_scenario()), opponents=["omoko_v1"])
        env.reset(opponent="omoko_v1")
        # Controlled engine fixture, not a training start mode.
        env.game.defender_setup_phase.finish()
        env.game.is_planted = True
        env.game.planted_pos = (7, 3)
        env.game.spike_pos = None
        env.game.detonate_timer = 30
        env.defenders[0].pos = [7, 3]
        env.game.attacker_controller = SimpleNamespace(decide_move=lambda char, state: (list(char.pos), {"facing": char.facing}))
        transitions = []
        for _ in range(8):
            produced, _, _ = env.step()
            transitions.extend(produced)
            if env.done:
                break
        self.assertTrue(env.done)
        self.assertTrue(env.result()["defused"])
        self.assertEqual(env.metrics["defuse_decisions"], 6)
        self.assertTrue(transitions)
        self.assertTrue(all(site == "L" for site, _ in transitions))
        self.assertTrue(any(item[-2] == 1 and item[2] > 9 for _, item in transitions))


if __name__ == "__main__":
    unittest.main()
