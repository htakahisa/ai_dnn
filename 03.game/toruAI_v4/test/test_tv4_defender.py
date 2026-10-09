"""Generic defender privacy, roster invariance, phase routing and learning."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from unittest.mock import patch
import numpy as np
import torch

from frc_v1.perception import FrcPerceptionBuilder
from frc_v1.actions import build_masks, validate_action
from toruAI_v4.test.test_tv4_site import world
from toruAI_v4.tv4_scenario import Scenario
from toruAI_v4.tv4_model import SiteModel
from toruAI_v4.tv4_observer import FeatureHistory
from toruAI_v4.tv4_defender_policy import (
    PolicyEncoder, DefenderDQN, ACTION_DIM, OBS_DIM, DEFUSE_ACTION, staging_positions, assign_goals, learn_dqn,
)
from toruAI_v4.tv4_defender_controller import ToruV4DefenderController
from toruAI_v4.tv4_train_defender_search import eligible_presets, evaluation_plan, summarize_defender
from toruAI_v4.tv4_retake_combat import RETAKE_OBS_DIM


class GenericDefenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenario = Scenario()
        torch.set_num_threads(1)

    def inputs(self, snapshot=None):
        snapshot = snapshot or FrcPerceptionBuilder("D").build(world(self.scenario))
        goal = self.scenario.posts[0].watch
        return snapshot, PolicyEncoder(self.scenario).encode(snapshot, snapshot.allies[0], goal, [.5, .5], {})

    def test_training_defaults_to_new_and_resume_requires_explicit_option(self):
        from toruAI_v4.tv4_train_defender_search import parse_arguments, TRAINING_SETS
        from toruAI_v4.tv4_train_retake import TRAINING_SETS as RETAKE_SETS
        for phase in ("search", "retake"):
            args = parse_arguments(["--phase", phase])
            self.assertFalse(args.resume)
            self.assertFalse(args.eval_only)
            self.assertEqual(args.sets, RETAKE_SETS if phase == "retake" else TRAINING_SETS)
            self.assertTrue(parse_arguments(["--phase", phase, "--resume"]).resume)
            self.assertFalse(parse_arguments(["--phase", phase, "--fresh"]).resume)

    def test_resume_cannot_be_combined_with_fresh_or_evaluation(self):
        from toruAI_v4.tv4_train_defender_search import parse_arguments
        import contextlib
        import io
        for option in ("--fresh", "--eval-only"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(["--resume", option])

    def test_search_training_routes_all_or_selected_opponents_independently(self):
        from toruAI_v4 import tv4_train_defender_search as trainer
        from toruAI_v4.tv4_scenario import OPPONENTS
        for options, selected in (([], list(OPPONENTS)),
                                  (["--opponents", "fnatic_v3", "touyama_v2"], ["fnatic_v3", "touyama_v2"])):
            models = {opponent: Mock() for opponent in selected}
            hashes = {opponent: opponent + "_hash" for opponent in selected}
            with patch.object(trainer, "load_analyses", return_value=(models, hashes)), \
                 patch.object(trainer, "train_search_opponent") as train:
                trainer.main([*options, "--max-workers", "1"])
            self.assertEqual(train.call_count, len(selected))
            for call, opponent in zip(train.call_args_list, selected):
                args, scenario, analysis, signatures = call.args
                self.assertEqual(args.opponents, [opponent])
                self.assertEqual(analysis, {opponent: models[opponent]})
                self.assertEqual(signatures, {opponent: hashes[opponent]})
                self.assertFalse(args.resume)

    def test_search_evaluates_each_set_and_keeps_better_intermediate_model(self):
        import contextlib
        import io
        import tempfile
        from toruAI_v4 import tv4_train_defender_search as trainer
        scores = (.3, .7, .2)
        evaluated, saved = [], []
        plan = [("gc_v1", "test_preset", 999)]

        def evaluate(current_plan, *args):
            evaluated.append(current_plan)
            return dict(plants=1, by_opponent={}, mean_search_readiness=scores[len(evaluated) - 1],
                        mean_search_survivors=4., mean_search_resources=1., mean_plant_distance=5.)

        def save(path, state):
            if path.name == "search_best.pt":
                saved.append(state["completed_sets"])

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stderr(io.StringIO()):
            args = trainer.parse_arguments([])
            self.assertEqual(args.eval_every, 1)
            args.opponents, args.sets = ["gc_v1"], 3
            args.data_dir, args.best_dir, args.log_dir = (Path(temp) / name for name in ("data", "best", "logs"))
            with patch.object(trainer, "eligible_presets", return_value=["test_preset"]), \
                 patch.object(trainer, "evaluation_plan", return_value=plan), \
                 patch.object(trainer, "rollout", return_value=([], [])), \
                 patch.object(trainer, "learn_dqn", return_value=0.), \
                 patch.object(trainer, "evaluate", side_effect=evaluate), \
                 patch.object(trainer, "evaluation_summary", return_value="summary"), \
                 patch.object(trainer, "atomic_save", side_effect=save):
                trainer.train_search_opponent(args, self.scenario, {}, {"gc_v1": "analysis"})
        self.assertEqual(len(evaluated), 3)
        self.assertTrue(all(current is plan for current in evaluated))
        self.assertEqual(saved, [1, 2])

    def test_deployment_rejects_shared_or_wrong_opponent_search(self):
        from toruAI_v4 import tv4_defender_controller as module
        policy = DefenderDQN(OBS_DIM)
        for saved in ({}, {"opponent": "touyama_v2"}):
            with patch.object(module, "load_policy", return_value=(policy, saved)), self.assertRaises(ValueError):
                ToruV4DefenderController("fnatic_v3", scenario=self.scenario)
        with patch.object(module, "load_policy", return_value=(policy, {"opponent": "fnatic_v3"})) as load, \
             patch.object(Path, "read_bytes", return_value=b"weights"):
            controller = ToruV4DefenderController("fnatic_v3", scenario=self.scenario)
        self.assertIs(controller.search, policy)
        self.assertEqual(load.call_args.args[0], module.DEFENDER_BEST / "fnatic_v3" / "search_best.pt")

    def test_roster_names_do_not_change_policy_observation_or_mask(self):
        snapshot, first = self.inputs()
        renamed = replace(snapshot, allies=tuple(replace(a, name=f"new_player_{i}") for i, a in enumerate(snapshot.allies)),
                          enemies=tuple(replace(e, name=f"new_enemy_{i}") for i, e in enumerate(snapshot.enemies)))
        _, second = self.inputs(renamed)
        np.testing.assert_array_equal(first.observation, second.observation)
        np.testing.assert_array_equal(first.mask, second.mask)
        self.assertEqual(len(first.observation), OBS_DIM)

    def test_ally_order_does_not_change_shared_player_inputs(self):
        snapshot, first = self.inputs()
        rearranged = replace(snapshot, allies=tuple(reversed(snapshot.allies)))
        second = PolicyEncoder(self.scenario).encode(rearranged, snapshot.allies[0], self.scenario.posts[0].watch, [.5, .5], {})
        np.testing.assert_array_equal(first.observation, second.observation)
        np.testing.assert_array_equal(first.mask, second.mask)

    def test_own_stats_and_ability_are_observed(self):
        snapshot, first = self.inputs()
        changed = replace(snapshot.allies[0], hp=25., accuracy=.99, ability_name="SMOKE", charges=2)
        snapshot = replace(snapshot, allies=(changed,) + snapshot.allies[1:])
        _, second = self.inputs(snapshot)
        self.assertFalse(np.array_equal(first.observation, second.observation))

    def test_all_unmasked_actions_pass_the_engine_public_masks(self):
        snapshot, inputs = self.inputs()
        masks = build_masks(snapshot)
        for index in np.flatnonzero(inputs.mask):
            validate_action(snapshot, masks, snapshot.allies[0].slot, inputs.actions[index])
        self.assertEqual(len(inputs.mask), ACTION_DIM)
        self.assertFalse(inputs.mask[DEFUSE_ACTION])

    def test_plant_location_is_the_retake_goal_and_defuse_is_legal(self):
        snapshot = FrcPerceptionBuilder("D").build(world(self.scenario))
        plant = self.scenario.sites["L"][0]
        ally = replace(snapshot.allies[0], position=plant)
        snapshot = replace(snapshot, allies=(ally,) + snapshot.allies[1:], is_planted=True, spike_planted=plant)
        inputs = PolicyEncoder(self.scenario).encode(snapshot, ally, plant, [.1, .9], {})
        self.assertTrue(inputs.mask[DEFUSE_ACTION])
        self.assertEqual(inputs.teacher, DEFUSE_ACTION)
        goals = assign_goals(snapshot, self.scenario, [.1, .9], staging_positions(self.scenario))
        self.assertLessEqual(max(abs(goals[ally.slot][0] - plant[0]), abs(goals[ally.slot][1] - plant[1])), 1)

    def test_staging_positions_remain_outside_the_site(self):
        stages = staging_positions(self.scenario)
        for side, positions in stages.items():
            expected = {p for points in self.scenario.rally_points[side].values() for p in points}
            self.assertEqual(set(positions), expected)
            self.assertGreaterEqual(len(set(positions)), 5)
            for pos in positions:
                self.assertNotEqual(self.scenario.grid[pos], 2)
                self.assertEqual(self.scenario.rally_dist[side][pos], 0)

    def test_rally_distances_use_paths_to_only_lowercase_a_b(self):
        from grid_paths import distance_map
        for side, groups in self.scenario.rally_points.items():
            points = tuple(p for cells in groups.values() for p in cells)
            self.assertTrue({"a", "b"} <= set(groups) <= {"a", "b", "c"})
            origin = self.scenario.spawn
            expected = min(distance_map(self.scenario.grid, p)[origin] for p in points)
            self.assertEqual(self.scenario.rally_dist[side][origin], expected)

    def test_optional_c_rally_and_C_entry_are_distinct_from_ability_markers(self):
        from toruAI_v4.tv4_map_retake_R import MAZE_STR
        rows = MAZE_STR.strip().splitlines()
        for r, col, marker in ((14, 34, "c"), (11, 40, "C")):
            self.assertNotEqual(self.scenario.grid[r, col], 1)
            rows[r] = rows[r][:col] + marker + rows[r][col + 1:]
        with patch("toruAI_v4.tv4_map_retake_R.MAZE_STR", "\n".join(rows)):
            scenario = Scenario()
        self.assertIn((14, 34), scenario.rally_points["R"]["c"])
        self.assertIn((11, 40), scenario.retake_entries["R"]["c"])
        self.assertNotIn((11, 40), staging_positions(scenario)["R"])
        self.assertEqual(scenario.rally_dist["R"][14, 34], 0)
        self.assertEqual(scenario.signature, self.scenario.signature)

    def test_previous_checkpoint_staging_can_resume_with_new_rally_geometry(self):
        from toruAI_v4.tv4_defender_controller import normalize_policy_schema, policy_metadata
        from toruAI_v4.tv4_defender_policy import legacy_staging_positions
        expected = policy_metadata(self.scenario)
        saved = {"schema": {**expected, "staging": legacy_staging_positions(self.scenario)}, "completed_sets": 50}
        updated = normalize_policy_schema(saved, self.scenario)
        self.assertEqual(updated["schema"], expected)
        self.assertEqual(updated["completed_sets"], 50)

    def test_hidden_enemy_position_and_carrier_do_not_affect_actions(self):
        game = world(self.scenario)
        analysis = SiteModel(len(FeatureHistory(self.scenario).fields))
        policy = DefenderDQN(OBS_DIM)
        def controller():
            c = ToruV4DefenderController("fnatic_v3", scenario=self.scenario, search=policy, retake=policy,
                                        analyses={"fnatic_v3": analysis})
            c.set_game(game)
            c.prepare_team_tick()
            return c
        first = controller()
        for i, enemy in enumerate(game.chars[5:]):
            enemy.pos = [4, 6 + i]
            enemy.has_spike = i == 2
        game.target_plant_pos = (5, 40)
        second = controller()
        self.assertEqual(first.actions, second.actions)
        for name in first.inputs:
            np.testing.assert_array_equal(first.inputs[name].observation, second.inputs[name].observation)

    def test_mid_tick_plant_switches_to_retake_before_executing_defender(self):
        game = world(self.scenario)
        analysis = SiteModel(len(FeatureHistory(self.scenario).fields))
        policy = DefenderDQN(OBS_DIM)
        c = ToruV4DefenderController("fnatic_v3", scenario=self.scenario, search=policy, retake=DefenderDQN(RETAKE_OBS_DIM),
                                    analyses={"fnatic_v3": analysis})
        c.set_game(game)
        c.prepare_team_tick()
        game.is_planted, game.planted_pos = True, self.scenario.sites["L"][0]
        c.decide_move(game.chars[0], {"is_planted": True})
        self.assertEqual(c.decisions[game.chars[0].name][0], "retake")

    def test_search_evaluation_uses_explicit_bootstrap_retake_without_loading_models(self):
        game = world(self.scenario)
        c = ToruV4DefenderController("fnatic_v3", scenario=self.scenario, search=DefenderDQN(OBS_DIM),
                                    retake=None, training=True,
                                    analyses={"fnatic_v3": SiteModel(len(FeatureHistory(self.scenario).fields))})
        c.training = False
        c.set_game(game)
        self.assertIsNone(c.retake)
        game.is_planted, game.planted_pos = True, self.scenario.sites["L"][0]
        c.prepare_team_tick()
        self.assertTrue(c.actions)

    def test_search_only_stops_actions_at_plant_without_entering_retake(self):
        game = world(self.scenario)
        policy = DefenderDQN(OBS_DIM)
        c = ToruV4DefenderController("fnatic_v3", scenario=self.scenario, search=policy, retake=policy,
                                    analyses={"fnatic_v3": SiteModel(len(FeatureHistory(self.scenario).fields))})
        c.set_game(game)
        c.stop_at_plant = True
        c.prepare_team_tick()
        game.is_planted, game.planted_pos = True, self.scenario.sites["L"][0]
        own = game.chars[0]
        with patch.object(policy, "forward", wraps=policy.forward) as forward:
            action = c.decide_move(own, {"is_planted": True})
            self.assertEqual(action[0], own.pos)
            forward.assert_not_called()
        self.assertEqual(c.decisions, {})

    def test_search_plants_are_not_counted_as_losses_or_retakes(self):
        from toruAI_v4.tv4_train_defender_search import summarize_defender, evaluation_summary, PLANT_ADVANTAGE_REWARD
        record = dict(scope="search", planted=True, won=None, defused=False, plant_defenders=5,
                      plant_attackers=3, plant_advantage=2, plant_distance=8., retake_ticks=0,
                      plant_resource_retention=.8, skill_uses=1, search_survivors=5,
                      search_resource_retention=.8, search_readiness=1., reward=1.)
        metrics = summarize_defender([record])
        self.assertEqual(metrics["completed_rounds"], 0)
        self.assertIsNone(metrics["win_rate"])
        self.assertIsNone(metrics["retake_win_rate"])
        self.assertEqual(metrics["mean_plant_advantage"], 2)
        self.assertNotIn("リテイク勝率", evaluation_summary(metrics))
        self.assertGreater(PLANT_ADVANTAGE_REWARD * (5 - 3), PLANT_ADVANTAGE_REWARD * (5 - 5))

    def test_training_and_evaluation_teams_are_valid_against_each_opponent(self):
        from toruAI_v4.tv4_train_defender_search import TRAINING_PRESETS, EVALUATION_PRESETS
        from toruAI_v4.tv4_scenario import OPPONENTS
        self.assertFalse(set(TRAINING_PRESETS) & set(EVALUATION_PRESETS))
        plan = evaluation_plan(tuple(OPPONENTS), EVALUATION_PRESETS, 3, 42)
        self.assertEqual(len(plan), 18)
        for opponent in OPPONENTS:
            self.assertTrue(eligible_presets(TRAINING_PRESETS, opponent))
            self.assertEqual(len({preset for opp, preset, seed in plan if opp == opponent}), 3)

    def test_double_dqn_updates_actual_weights(self):
        model = DefenderDQN(OBS_DIM)
        import copy
        target = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        obs = np.zeros(OBS_DIM, np.float16)
        replay = [[obs, 0, 2., obs, np.ones(ACTION_DIM, bool), 1.] for _ in range(8)]
        before = model(torch.tensor(obs, dtype=torch.float32)).detach().clone()
        loss = learn_dqn(model, target, optimizer, replay, np.random.default_rng(1), updates=3, batch_size=8)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(model(torch.tensor(obs, dtype=torch.float32))[0]), float(before[0]))

    def test_search_demonstration_preserves_last_flash_but_mask_allows_it(self):
        from frc_v1.perception import Sighting
        snapshot = FrcPerceptionBuilder("D").build(world(self.scenario))
        ally = replace(snapshot.allies[0], ability_name="FLASH", charges=1)
        enemy = (14, 17)
        snapshot = replace(snapshot, allies=(ally,) + snapshot.allies[1:], sightings=(Sighting(0, enemy, "normal"),))
        encoder = PolicyEncoder(self.scenario)
        inputs = encoder.encode(snapshot, ally, self.scenario.posts[0].watch, [.9, .1], {0: (enemy, 0, (0, 0))})
        self.assertTrue(inputs.mask[40])
        self.assertNotEqual(inputs.teacher, 40)
        ally = replace(ally, charges=2)
        snapshot = replace(snapshot, allies=(ally,) + snapshot.allies[1:])
        self.assertEqual(encoder.encode(snapshot, ally, self.scenario.posts[0].watch, [.9, .1], {0: (enemy, 0, (0, 0))}).teacher, 40)

    def test_retake_selects_left_or_right_network_using_the_actual_plant(self):
        game = world(self.scenario)
        game.is_planted, game.planted_pos = True, self.scenario.sites["L"][0]
        left, right = DefenderDQN(RETAKE_OBS_DIM), DefenderDQN(RETAKE_OBS_DIM)
        c = ToruV4DefenderController("fnatic_v3", scenario=self.scenario, search=DefenderDQN(OBS_DIM), retake={"L": left, "R": right},
                                    analyses={"fnatic_v3": SiteModel(len(FeatureHistory(self.scenario).fields))})
        c.set_game(game)
        with patch.object(left, "forward", wraps=left.forward) as l, patch.object(right, "forward", wraps=right.forward) as r:
            c.prepare_team_tick()
            self.assertTrue(l.called)
            self.assertFalse(r.called)
            previous = l.call_count
            game.planted_pos, game.battle_tick = self.scenario.sites["R"][0], 1
            c.prepare_team_tick()
            self.assertEqual(l.call_count, previous)
            self.assertTrue(r.called)

    def test_retake_samples_are_separated_by_actual_site(self):
        from toruAI_v4.tv4_train_retake import split_retakes
        records = [{"site": "L"}, {"site": "R"}, {"site": None}]
        left, right = [None] * 6 + ["L"], [None] * 6 + ["R"]
        parts = split_retakes(records, [left, right])
        self.assertEqual(parts["L"], ([records[0]], [left]))
        self.assertEqual(parts["R"], ([records[1]], [right]))

    def test_search_selection_balances_readiness_instead_of_forbidding_plants(self):
        from toruAI_v4.tv4_train_defender_search import score_best
        camping = dict(mean_search_readiness=.5, mean_search_survivors=5., mean_search_resources=1., mean_plant_distance=35.)
        ready = dict(mean_search_readiness=.8, mean_search_survivors=4.5, mean_search_resources=.8, mean_plant_distance=8.)
        wiped = dict(mean_search_readiness=-1., mean_search_survivors=0., mean_search_resources=0., mean_plant_distance=None)
        self.assertGreater(score_best(ready, "search"), score_best(camping, "search"))
        self.assertLess(score_best(wiped, "search"), score_best(camping, "search"))

    def test_retake_margin_allows_relative_outlier_when_time_is_sufficient(self):
        from toruAI_v4.tv4_train_defender_search import retake_arrival_readiness
        result = retake_arrival_readiness([1, 1, 1, 1, 24], 55)
        self.assertTrue(result["plant_all_ready"])
        self.assertEqual(result["plant_min_margin"], 5.)
        self.assertEqual(result["plant_arrival_penalty"], 0.)

    def test_retake_margin_penalizes_one_late_player_despite_small_average(self):
        from toruAI_v4.tv4_train_defender_search import retake_arrival_readiness
        result = retake_arrival_readiness([1, 1, 1, 1, 35], 55)
        self.assertEqual(result["plant_late_count"], 1)
        self.assertEqual(result["plant_ready_count"], 4)
        self.assertFalse(result["plant_all_ready"])
        self.assertAlmostEqual(result["plant_arrival_penalty"], 3.1)
        self.assertEqual(result["plant_min_margin"], -6.)
        self.assertEqual(retake_arrival_readiness([24], 54)["plant_late_count"], 1)

    def test_retake_margin_unreachable_and_empty_teams(self):
        from toruAI_v4.tv4_train_defender_search import retake_arrival_readiness
        for distance in [-1, float("inf")]:
            result = retake_arrival_readiness([distance], 55)
            self.assertEqual(result["plant_late_count"], 1)
            self.assertEqual(result["plant_arrival_penalty"], 5.)
        result = retake_arrival_readiness([], 55)
        self.assertFalse(result["plant_all_ready"])
        self.assertIsNone(result["plant_min_margin"])

    def test_resource_retention_excludes_dead_players_and_caps_regeneration(self):
        from toruAI_v4.tv4_train_defender_search import retained_resources
        a = SimpleNamespace(is_alive=True, ability_name="SMOKE", smoke_charges=3)
        self.assertEqual(retained_resources([a], {"a": 2}), 1.)
        a.is_alive = False
        self.assertEqual(retained_resources([a], {"a": 2}), 0.)
        self.assertIsNone(retained_resources([a], {"a": 0}))

    def test_distant_players_have_more_incentive_to_rotate_and_less_to_wait(self):
        from toruAI_v4.tv4_train_defender_search import early_rotation_reward
        inputs = SimpleNamespace(distances=np.array([[4., 3., 32., 31., 0., -1.]]),
                                 rotation_confidence=.8, combat_contact=False)
        near_move = early_rotation_reward(inputs, (0, 0), (0, 1))
        far_move = early_rotation_reward(inputs, (0, 2), (0, 3))
        self.assertGreater(far_move, near_move)
        self.assertGreater(near_move, 0.)
        self.assertLess(early_rotation_reward(inputs, (0, 2), (0, 2)),
                        early_rotation_reward(inputs, (0, 0), (0, 0)))
        self.assertLess(early_rotation_reward(inputs, (0, 3), (0, 2)), 0.)
        self.assertEqual(early_rotation_reward(inputs, (0, 4), (0, 4)), 0.)
        self.assertEqual(early_rotation_reward(inputs, (0, 5), (0, 5)), 0.)

    def test_rotation_pressure_stops_for_uncertainty_and_combat(self):
        from toruAI_v4.tv4_train_defender_search import early_rotation_reward
        inputs = SimpleNamespace(distances=np.array([[32., 31.]]),
                                 rotation_confidence=.8, combat_contact=False)
        for kwargs in ({"damage": 1.}, {"kills": 1}):
            self.assertEqual(early_rotation_reward(inputs, (0, 0), (0, 0), **kwargs), 0.)
        inputs.combat_contact = True
        self.assertEqual(early_rotation_reward(inputs, (0, 0), (0, 0)), 0.)
        inputs.combat_contact, inputs.rotation_confidence = False, .64
        self.assertEqual(early_rotation_reward(inputs, (0, 0), (0, 1)), 0.)

    def test_rotation_context_uses_public_snapshot_and_excludes_setup_and_retake(self):
        snapshot, _ = self.inputs()
        ally = snapshot.allies[0]
        encoder = PolicyEncoder(self.scenario)
        for phase, planted, expected in (("live", False, .8), ("setup", False, 0.), ("live", True, 0.)):
            current = replace(snapshot, phase=phase, is_planted=planted,
                              spike_planted=self.scenario.sites["L"][0] if planted else None)
            inputs = encoder.encode(current, ally, self.scenario.posts[0].watch, [.8, .2], {})
            self.assertEqual(inputs.rotation_confidence, expected)


if __name__ == "__main__":
    unittest.main()
