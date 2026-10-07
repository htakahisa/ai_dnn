"""Search observations, tactical rewards, frozen weights and real IQ rollouts."""

import contextlib
from dataclasses import replace
import io
import json
from collections import Counter
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from concon_v1.co1_defender_common import DefenderSearchDQN, GORIGONS, observation_dim as basic_dim
from concon_v1.co1_defender_scenario import get_scenario, phase_scenario, validate_checkpoint
from concon_v1.co1_defender_search_common import (
    DefenderSearchBattleDQN, load_search_weights, build_search_inputs, ACTION_DIM,
    MEMORY_TICKS, POST_KILL_HOLD_TICKS, observation_dim,
)
from concon_v1.co1_defender_search_training import DefenderSearchEnv, OPPONENTS, START_MODES
from concon_v1.co1_defender_search_rewards import support_goal, prepare_reward_context, decision_reward, ROUND_REWARD, search_score, facing_target
from concon_v1.evaluate_co1_defender_search import behavior_summary, survival_summary, survival_log
from concon_v1.co1_learn_defender_search import ConconDefenderSearchController
from concon_v1.co1_train_defender_search import make_checkpoint
from concon_v1.co1_defender_search_battle_training import (
    optimize, make_battle_checkpoint, qualifies_as_best, epsilon_by_episode, EPSILON_END,
    comparison_checkpoint,
    log_best_comparison,
    train_battle,
    stationary_combat_loss,
    memory_navigation_loss,
    balanced_opponent_schedule,
)
from concon_v1.co1_defender_positioning import evaluate_positioning


def actor(name, pos, team="D", facing="W", known=True, ability="NONE"):
    return SimpleNamespace(name=name, team=team, pos=list(pos), facing=facing,
        is_alive=True, position_known=known, hp=100, max_hp=100, round_kills=0,
        blind_remaining=0, reveal_remaining=0, moved_this_tick=False,
        ability_name=ability, smoke_charges=int(ability == "SMOKE"),
        flash_charges=int(ability == "FLASH"), recon_charges=int(ability == "RECON"),
        facing_forced_this_tick=False)


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.scenario = get_scenario()
        cls.foundation = None
        if cls.scenario.model_path.exists():
            saved = torch.load(cls.scenario.model_path, map_location="cpu", weights_only=False)
            try:
                validate_checkpoint(saved, cls.scenario)
            except ValueError:
                pass  # Test fixtures must not depend on a user's old map checkpoint.
            else:
                cls.foundation = saved
        if cls.foundation is None:
            from concon_v1.co1_guard_positioning import training_targets
            model = DefenderSearchDQN(cls.scenario)
            with torch.no_grad():
                for setup in (False, True):
                    indices, values = training_targets(phase_scenario(cls.scenario, setup))
                    model.navigation_values.weight[indices + int(setup) * model.phase_size] = values
                for preferred in range(4):
                    for neighbors in range(16):
                        if neighbors & (1 << preferred):
                            values = model.navigation_yield_values.weight[preferred * 16 + neighbors].reshape(5, 8)
                            values[:4], values[preferred ^ 1], values[preferred], values[4] = 2., 5., -6., -2.
            cls.foundation = make_checkpoint(model, cls.scenario, {"passed": True}, 0)

    def model(self):
        model = DefenderSearchBattleDQN(self.scenario)
        load_search_weights(model, self.foundation, self.scenario)
        return model

    def inputs(self, chars, tick=0, controller=None, setup=False):
        controller = controller or ConconDefenderSearchController(model=self.model())
        controller.assignments = {char.name: "abcde"[index] for index, char in enumerate(
            [char for char in chars if char.team == "D"])}
        state = dict(chars=chars, grid=self.scenario.grid, battle_tick=tick, defender_setup_active=setup)
        return controller, build_search_inputs(controller, chars[0], state)

    def test_warm_start_preserves_all_foundation_weights_and_freezes_them(self):
        model = self.model()
        for key, value in self.foundation["model_state_dict"].items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertFalse(model.navigation_values.weight.requires_grad)
        self.assertFalse(model.navigation_yield_values.weight.requires_grad)
        self.assertTrue(model.head[2].weight.requires_grad)

    def test_extreme_tactical_logits_cannot_change_quiet_setup_or_live_navigation(self):
        model = self.model()
        with torch.no_grad():
            model.head[2].bias.fill_(1000.)
        result = evaluate_positioning(model, self.scenario, trials=2)
        self.assertTrue(result["passed"], result)

    def test_hidden_enemy_state_cannot_change_observation_or_action_mask(self):
        own = actor(GORIGONS.players[0], (9, 13))
        hidden = actor("enemy", (9, 8), team="A", known=False)
        controller, (before, mask, _) = self.inputs([own, hidden])
        hidden.pos, hidden.hp, hidden.facing, hidden.blind_remaining = [10, 33], 1, "N", 12
        _, (after, after_mask, _) = self.inputs([own, hidden], controller=controller)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(mask, after_mask)
        self.assertEqual(controller.sightings, {})

    def test_shared_memory_retains_only_disclosed_position_and_expires(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        controller, (_, _, first) = self.inputs([own, enemy])
        self.assertTrue(first["active"])
        enemy.position_known, enemy.pos = False, [10, 33]
        _, (observation, _, second) = self.inputs([own, enemy], tick=1, controller=controller)
        self.assertEqual(second["target"], (9, 8))
        extra = observation[basic_dim(self.scenario):basic_dim(self.scenario) + self.scenario.grid.size].reshape(self.scenario.grid.shape)
        self.assertGreater(extra[9, 8], 0.)
        self.assertEqual(extra[10, 33], 0.)
        _, (_, _, expired) = self.inputs([own, enemy], tick=MEMORY_TICKS, controller=controller)
        self.assertFalse(expired["active"])
        self.assertEqual(controller.sightings, {})

    def test_kill_keeps_recent_enemy_direction_without_a_fixed_facing_override(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        controller, _ = self.inputs([own, enemy])
        own.round_kills = 1
        enemy.is_alive, enemy.position_known, enemy.pos = False, False, [-1, -1]
        _, (_, mask, context) = self.inputs([own, enemy], tick=1, controller=controller)
        self.assertTrue(context["killed"])
        self.assertEqual(context["target"], (9, 8))
        self.assertTrue(mask[32:40].all())
        self.assertTrue(context["post_kill_hold"])
        context = prepare_reward_context(context, self.scenario.grid)
        self.assertIsNone(context["support_goal"])
        self.assertGreater(decision_reward(38, context, (9, 13), "W"),
                           decision_reward(14, context, (10, 13), "W"))
        self.assertGreater(decision_reward(38, context, (9, 13), "W"),
                           decision_reward(34, context, (9, 13), "E"))
        # A fresh shared report on the other side must not replace own kill direction.
        other = actor("other", (10, 33), team="A")
        _, (_, _, context) = self.inputs([own, enemy, other], tick=2, controller=controller)
        self.assertEqual(context["target"], (9, 8))
        _, (_, _, context) = self.inputs([own, enemy, other], tick=1 + POST_KILL_HOLD_TICKS, controller=controller)
        self.assertFalse(context["post_kill_hold"])

    def test_new_fireable_enemy_takes_priority_over_post_kill_watch(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        own.round_kills = 1
        enemy.is_alive = False
        next_enemy = actor("next", (9, 9), team="A")
        _, (_, _, context) = self.inputs([own, enemy, next_enemy], tick=1, controller=controller)
        self.assertFalse(context["post_kill_hold"])
        self.assertEqual(context["target"], (9, 9))

    def test_team_death_report_does_not_activate_search_until_memory_timeout(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        enemy.is_alive, enemy.position_known, enemy.pos = False, False, [-1, -1]
        _, (_, _, context) = self.inputs([own, enemy], tick=1, controller=controller)
        self.assertFalse(context["active"])
        self.assertIsNone(context["target"])
        self.assertFalse(controller.sightings["enemy"]["alive"])
        self.assertEqual(controller.sightings["enemy"]["pos"], (9, 8))

    def test_iq_omission_is_not_treated_as_a_death_report(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        enemy.is_alive = False  # IQ omission keeps position_known=True.
        _, (_, _, context) = self.inputs([own, enemy], tick=1, controller=controller)
        self.assertTrue(context["active"])
        self.assertTrue(controller.sightings["enemy"]["alive"])

    def test_kill_watch_signal_persists_and_returns_to_foundation_when_watch_ends(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        own.round_kills = 1
        enemy.is_alive, enemy.position_known, enemy.pos = False, False, [-1, -1]
        start = controller.model.basic_dim + 2 * self.scenario.grid.size
        for tick in (1, 2, POST_KILL_HOLD_TICKS):
            _, (obs, _, context) = self.inputs([own, enemy], tick=tick, controller=controller)
            self.assertTrue(context["post_kill_hold"])
            self.assertEqual(obs[start + 3], 1.)
        _, (obs, _, context) = self.inputs([own, enemy], tick=1 + POST_KILL_HOLD_TICKS, controller=controller)
        self.assertFalse(context["active"])
        self.assertEqual(obs[start + 3], 0.)
        self.assertIn("enemy", controller.sightings)  # Dead record cannot prolong search.

    def test_flash_candidates_require_an_engaged_ally_but_do_not_force_casting(self):
        own = actor(GORIGONS.players[0], (9, 13), ability="FLASH")
        ally = actor(GORIGONS.players[1], (9, 10), facing="W")
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        controller = ConconDefenderSearchController(model=self.model())
        controller.game = SimpleNamespace(_smoke_cells=lambda: set())
        with patch("concon_v1.co1_defender_search_common._impact_aim", return_value=(9, 9)) as aim:
            _, (_, mask, context) = self.inputs([own, ally, enemy], controller=controller)
            self.assertTrue(mask[45])
            self.assertTrue(mask[:40].any())
            self.assertEqual(context["ability_payloads"][45]["ability"], "FLASH")
            self.assertIn("require_flash_hit", aim.call_args.kwargs)
            ally.facing = "E"
            _, (_, mask, _) = self.inputs([own, ally, enemy], controller=controller)
            self.assertFalse(mask[45:50].any())

    def test_stationary_fire_remains_preferred_after_a_shot(self):
        own = actor(GORIGONS.players[0], (9, 13), facing="W")
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        controller, (_, _, context) = self.inputs([own, enemy])
        self.assertTrue(context["fireable"])
        context = prepare_reward_context(context, self.scenario.grid)
        stop = decision_reward(38, context, (9, 13), "W")
        move = decision_reward(14, context, (10, 13), "W")
        self.assertGreater(stop, move)
        context.update(after_shot=True, killed=False, retreat_available=True)
        context["threats"][1] = False
        stop = decision_reward(38, context, (9, 13), "W")
        retreat = decision_reward(14, context, (10, 13), "W")
        self.assertGreater(stop, retreat)

    def test_shared_sighting_support_does_not_require_ally_engagement(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        _, (_, _, context) = self.inputs([own, enemy])
        context.update(fireable=False, engaged=[], position=(2, 4))
        context["actor"].pos = [2, 4]
        grid = np.zeros((9, 20), dtype=np.int32)
        grid[:, 10], grid[4, 10] = 1, 0
        enemy.pos = [2, 16]
        ally = actor("ally", (2, 18), facing="E")
        context["chars"] = [context["actor"], ally, enemy]
        context["disclosed"] = [enemy]
        prepared = prepare_reward_context(context, grid)
        self.assertIsNotNone(prepared["support_goal"])
        self.assertEqual(prepared["target"], (2, 16))

    def test_facing_reward_starts_at_the_peek_and_not_behind_the_wall(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        grid[:3, 4] = 1
        own = actor("own", (2, 3))
        enemy = actor("enemy", (3, 6), team="A")
        context = dict(actor=own, position=(2, 3), goal=(2, 3), aim=(6, 3),
                       target=(3, 6), chars=[own, enemy], disclosed=[enemy], engaged=[],
                       fireable=False, smoke=set(), lines=[False, True, False, False, False],
                       neutralized=False, post_kill_hold=False)
        context = prepare_reward_context(context, grid)
        self.assertIsNone(facing_target(context, (2, 3)))
        self.assertIsNone(facing_target(context, (1, 3)))
        self.assertEqual(decision_reward(34, context, (2, 3), "E"),
                         decision_reward(38, context, (2, 3), "W"))
        self.assertEqual(decision_reward(2, context, (1, 3), "E"),
                         decision_reward(6, context, (1, 3), "W"))
        self.assertEqual(facing_target(context, (3, 3)), (3, 6))
        self.assertGreater(decision_reward(10, context, (3, 3), "E"),
                           decision_reward(14, context, (3, 3), "W"))
        own.pos = [3, 3]
        context.update(position=(3, 3), fireable=True)
        self.assertGreater(decision_reward(34, context, (3, 3), "E"),
                           decision_reward(10, context, (4, 3), "E"))

    def test_smoke_and_occupied_ray_also_disable_shared_target_facing_reward(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        own = actor("own", (3, 2))
        enemy = actor("enemy", (3, 6), team="A")
        context = dict(actor=own, position=(3, 2), goal=(3, 2), aim=(6, 2),
                       target=(3, 6), chars=[own, enemy], disclosed=[enemy], engaged=[],
                       fireable=False, smoke={(3, 4)}, lines=[False] * 5, post_kill_hold=False)
        for blocked_by in ("smoke", "ally"):
            with self.subTest(blocked_by=blocked_by):
                if blocked_by == "ally":
                    context["smoke"] = set()
                    context["chars"] = [own, enemy, actor("ally", (3, 4))]
                prepared = prepare_reward_context(context, grid)
                self.assertIsNone(facing_target(prepared, own.pos))
                self.assertEqual(decision_reward(34, prepared, own.pos, "E"),
                                 decision_reward(38, prepared, own.pos, "W"))

    def test_remembered_and_post_kill_facing_do_not_earn_rewards_through_walls(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        grid[:, 4] = 1
        own = actor("own", (3, 2))
        context = dict(actor=own, position=(3, 2), goal=(3, 2), aim=(6, 2),
                       target=(3, 6), chars=[own], disclosed=[], engaged=[], fireable=False,
                       smoke=set(), lines=[False] * 5, post_kill_hold=True)
        prepared = prepare_reward_context(context, grid)
        self.assertFalse(prepared["post_kill_hold"])
        self.assertIsNone(facing_target(prepared, own.pos))
        self.assertEqual(decision_reward(34, prepared, own.pos, "E"),
                         decision_reward(38, prepared, own.pos, "W"))

    def test_peek_facing_uses_a_clear_enemy_instead_of_a_blocked_support_target(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        grid[:, 4] = 1
        own = actor("own", (3, 2))
        blocked = actor("blocked", (3, 6), team="A")
        clear = actor("clear", (5, 2), team="A")
        context = dict(actor=own, target=(3, 6), chars=[own, blocked, clear],
                       disclosed=[blocked, clear], grid=grid, smoke=set())
        self.assertEqual(facing_target(context, (3, 2)), (5, 2))

    def test_equal_distance_support_prefers_a_different_firing_angle(self):
        grid = np.zeros((7, 7), dtype=np.int32)
        own = actor("own", (3, 1))
        ally = actor("ally", (3, 2))
        enemy = actor("enemy", (3, 5), team="A")
        context = dict(actor=own, position=(3, 1), chars=[own, ally, enemy],
                       disclosed=[enemy], engaged=[], fireable=False, smoke=set())
        goal, distance = support_goal(context, grid)
        self.assertEqual(distance, 1)
        self.assertIn(goal, ((2, 1), (4, 1)))

    def test_shot_phase_reads_only_own_shot_result_and_not_hidden_target_position(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        controller = ConconDefenderSearchController(model=self.model())
        controller.game = SimpleNamespace(_smoke_cells=lambda: set(), last_shots=[])
        self.inputs([own, enemy], controller=controller)
        _, (_, _, before_shooting) = self.inputs([own, enemy], tick=1, controller=controller)
        self.assertFalse(before_shooting["after_shot"])
        controller.game.last_shots = [{"shooter": own, "target": SimpleNamespace(name="enemy")}]
        _, (_, _, after_shooting) = self.inputs([own, enemy], tick=2, controller=controller)
        self.assertTrue(after_shooting["after_shot"])

    def test_support_bfs_is_training_only_and_respects_the_distance_limit(self):
        grid = np.zeros((9, 20), dtype=np.int32)
        grid[:, 10] = 1
        grid[4, 10] = 0
        own = actor("own", (2, 4))
        ally = actor("ally", (2, 18))
        enemy = actor("enemy", (2, 16), team="A")
        context = dict(actor=own, position=tuple(own.pos), chars=[own, ally, enemy],
                       engaged=[enemy], fireable=False, smoke=set())
        self.assertIsNotNone(support_goal(context, grid, limit=6))
        self.assertIsNone(support_goal(context, grid, limit=1))
        context["position"] = (0, 0)
        own.pos = [0, 0]
        self.assertIsNone(support_goal(context, grid, limit=6))

    def test_smoke_has_a_reserve_cost_and_movement_remains_available(self):
        own = actor(GORIGONS.players[0], (9, 13), ability="SMOKE")
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        _, (_, mask, context) = self.inputs([own, enemy])
        self.assertTrue(mask[40])
        self.assertTrue(mask[:40].any())
        self.assertLess(decision_reward(40, context, tuple(own.pos), own.facing), -.1)

    def test_behavior_evaluation_rewards_post_kill_holding_and_smoke_conservation(self):
        record = dict(fire_decisions=1, moving_fire_decisions=0, aligned_fire_decisions=1,
            post_kill_decisions=1, stationary_aligned_post_kill=1,
            support_opportunities=1, support_progress=1,
            memory_return_decisions=0, aligned_memory_returns=0, flash_casts=0, smoke_casts=0,
            enemy_flash_ticks=0, ally_flash_ticks=0, distant_post_departures=0, search_ticks=10)
        good = behavior_summary([record])
        departed = behavior_summary([{**record, "stationary_aligned_post_kill": 0}])
        spent_smoke = behavior_summary([{**record, "smoke_casts": 1}])
        self.assertEqual(good["post_kill_hold_rate"], 1.)
        self.assertLess(good["behavior_error"], departed["behavior_error"])
        self.assertLess(good["behavior_error"], spent_smoke["behavior_error"])

    def test_optimizer_updates_tactical_head_without_changing_foundation(self):
        model, target = self.model(), self.model()
        target.load_state_dict(model.state_dict())
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A", facing="E")
        _, (observation, mask, _) = self.inputs([own, enemy])
        before = model.navigation_values.weight.detach().clone()
        head_before = model.head[2].weight.detach().clone()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)
        replay = [(observation, 38, 1., observation, mask, 1., 1)]
        self.assertIsNotNone(optimize(model, target, optimizer, replay, batch_size=1))
        self.assertTrue(torch.equal(before, model.navigation_values.weight))
        self.assertFalse(torch.equal(head_before, model.head[2].weight))

    def test_combat_auxiliary_loss_learns_stationary_aim_over_legal_movement(self):
        model = self.model()
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        _, (observation, mask, context) = self.inputs([own, enemy])
        self.assertTrue(context["fireable"])
        obs = torch.as_tensor(observation).unsqueeze(0)
        values = torch.zeros((1, ACTION_DIM), requires_grad=True)
        move = int(np.flatnonzero(mask[:32])[0])
        with torch.no_grad():
            values[0, move] = 2.
            values[0, 34] = 10.  # Wrong stationary facing cannot satisfy the loss.
            illegal = np.flatnonzero(~mask[:32])
            if len(illegal):
                values[0, int(illegal[0])] = 100.
        loss = stationary_combat_loss(model, obs, values)
        self.assertGreater(float(loss), 2.)
        loss.backward()
        self.assertGreater(float(values.grad[0, move]), 0.)
        self.assertLess(float(values.grad[0, 32:40].sum()), 0.)
        self.assertEqual(float(values.grad[0, 34]), 0.)
        if len(illegal):
            self.assertEqual(float(values.grad[0, int(illegal[0])]), 0.)

    def test_combat_auxiliary_loss_keeps_blind_advance_and_quiet_navigation_available(self):
        model = self.model()
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        enemy.blind_remaining = 10
        _, (observation, _, context) = self.inputs([own, enemy])
        self.assertTrue(context["neutralized"])
        values = torch.ones((1, ACTION_DIM), requires_grad=True)
        self.assertEqual(float(stationary_combat_loss(model, torch.as_tensor(observation).unsqueeze(0), values)), 0.)
        _, (quiet, _, context) = self.inputs([own])
        self.assertFalse(context["active"])
        self.assertEqual(float(stationary_combat_loss(model, torch.as_tensor(quiet).unsqueeze(0), values)), 0.)

    def test_post_kill_auxiliary_loss_trains_stop_even_after_the_kill_tick(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        own.round_kills = 1
        enemy.is_alive, enemy.position_known = False, False
        self.inputs([own, enemy], tick=1, controller=controller)
        _, (obs, mask, context) = self.inputs([own, enemy], tick=2, controller=controller)
        self.assertFalse(context["killed"])
        self.assertTrue(context["post_kill_hold"])
        values = torch.zeros((1, ACTION_DIM), requires_grad=True)
        move = int(np.flatnonzero(mask[:32])[0])
        with torch.no_grad():
            values[0, move] = 2.
        loss = stationary_combat_loss(controller.model, torch.as_tensor(obs).unsqueeze(0), values)
        self.assertGreater(float(loss), 2.)
        loss.backward()
        self.assertGreater(float(values.grad[0, move]), 0.)
        self.assertLess(float(values.grad[0, 32:40].sum()), 0.)

    def test_memory_only_loss_learns_return_instead_of_opposite_movement(self):
        own = actor(GORIGONS.players[0], (9, 13))
        enemy = actor("enemy", (9, 8), team="A")
        controller, _ = self.inputs([own, enemy])
        enemy.position_known, enemy.pos = False, [-1, -1]
        _, (obs, mask, _) = self.inputs([own, enemy], tick=1, controller=controller)
        tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            teacher = DefenderSearchDQN.forward(controller.model, tensor[:, :controller.model.basic_dim])[0]
            preferred = int(teacher.masked_fill(~torch.as_tensor(mask[:40]), -torch.inf).argmax()) // 8
        alternatives = [move for move in range(5) if move != preferred and mask[move * 8:(move + 1) * 8].any()]
        wrong = alternatives[0] * 8
        values = torch.zeros((1, ACTION_DIM), requires_grad=True)
        with torch.no_grad():
            values[0, wrong] = 2.
        loss = memory_navigation_loss(controller.model, tensor, values)
        self.assertGreater(float(loss), 2.)
        loss.backward()
        self.assertGreater(float(values.grad[0, wrong]), 0.)
        self.assertLess(float(values.grad[0, preferred * 8:(preferred + 1) * 8].sum()), 0.)
        _, (quiet, _, _) = self.inputs([own], tick=MEMORY_TICKS, controller=controller)
        self.assertEqual(float(memory_navigation_loss(controller.model, torch.as_tensor(quiet).unsqueeze(0), values)), 0.)

    def test_best_selection_rejects_regression_in_kill_watch_or_memory_return(self):
        baseline = dict(positioning={"passed": True}, mean_search_score=.4, min_team_search_score=.3,
            behavior=dict(behavior_error=1., post_kill_decisions=10, post_kill_hold_rate=.8,
                          memory_motion_decisions=10, memory_navigation_error_rate=.1))
        candidate = {**baseline, "mean_search_score": .6,
                     "behavior": {**baseline["behavior"], "post_kill_hold_rate": .7}}
        self.assertFalse(qualifies_as_best(candidate, baseline))
        candidate["behavior"].update(post_kill_hold_rate=.8, memory_navigation_error_rate=.2)
        self.assertFalse(qualifies_as_best(candidate, baseline))
        candidate["behavior"]["memory_navigation_error_rate"] = .1
        self.assertTrue(qualifies_as_best(candidate, baseline))

    def test_return_facing_reward_cannot_pay_for_a_two_cell_oscillation(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        own = actor("own", (3, 2), facing="E")
        context = dict(actor=own, position=(3, 2), goal=(3, 1), aim=(3, 6),
                       target=(3, 6), chars=[own], disclosed=[], engaged=[], fireable=False,
                       smoke=set(), lines=[False] * 5, post_kill_hold=False)
        outward = prepare_reward_context(context, grid)
        away_reward = decision_reward(26, outward, (3, 3), "E")
        own.pos = [3, 3]
        inward = prepare_reward_context({**context, "position": (3, 3)}, grid)
        return_reward = decision_reward(18, inward, (3, 2), "E")
        self.assertLess(away_reward, 0.)
        self.assertLess(away_reward + .99 * return_reward, 0.)

    def test_breaking_shot_is_penalized_even_against_a_blinded_enemy(self):
        grid = np.zeros((7, 9), dtype=np.int32)
        grid[:3, 4] = 1
        own = actor("own", (3, 3))
        enemy = actor("enemy", (3, 6), team="A")
        context = dict(actor=own, position=(3, 3), goal=(3, 3), aim=(3, 6),
                       target=(3, 6), chars=[own, enemy], disclosed=[enemy], engaged=[],
                       fireable=True, smoke=set(), lines=[False, True, True, True, True], post_kill_hold=False)
        for blinded in (False, True):
            with self.subTest(blinded=blinded):
                context["neutralized"] = blinded
                prepared = prepare_reward_context(context, grid)
                stationary = decision_reward(34, prepared, (3, 3), "E")
                retreat = decision_reward(2, prepared, (2, 3), "E")
                backward = decision_reward(18, prepared, (3, 2), "E")
                self.assertGreater(stationary, backward)
                self.assertGreater(backward, retreat)
                self.assertLess(retreat, -.4)
                if blinded:
                    forward = decision_reward(26, prepared, (3, 4), "E")
                    self.assertGreater(forward, backward)
                    self.assertGreater(forward, stationary)

    def test_best_selection_accepts_improved_score_despite_more_moving_fire(self):
        baseline = dict(positioning={"passed": True}, mean_search_score=.4, min_team_search_score=.3,
                        behavior=dict(behavior_error=1., moving_fire_rate=.2, stationary_aligned_fire_rate=.7))
        candidate = {**baseline, "mean_search_score": .6, "min_team_search_score": .4,
                     "behavior": dict(behavior_error=.9, moving_fire_rate=.3, stationary_aligned_fire_rate=.7)}
        self.assertTrue(qualifies_as_best(candidate, baseline))
        candidate["behavior"].update(moving_fire_rate=.1, stationary_aligned_fire_rate=.6)
        self.assertTrue(qualifies_as_best(candidate, baseline))
        candidate["behavior"]["stationary_aligned_fire_rate"] = .8
        self.assertTrue(qualifies_as_best(candidate, baseline))
        candidate["behavior"].update(stationary_aligned_fire_rate=.3, normal_stationary_aligned_fire_rate=.8)
        self.assertTrue(qualifies_as_best(candidate, baseline))
        candidate["mean_search_score"] = .3
        self.assertFalse(qualifies_as_best(candidate, baseline))

    def test_normal_combat_metrics_exclude_blind_forward_opportunities(self):
        record = dict(fire_decisions=10, moving_fire_decisions=1, aligned_fire_decisions=3,
            normal_fire_decisions=4, aligned_normal_fire_decisions=3,
            post_kill_decisions=0, stationary_aligned_post_kill=0,
            support_opportunities=0, support_progress=0,
            memory_return_decisions=0, aligned_memory_returns=0, flash_casts=0, smoke_casts=0,
            enemy_flash_ticks=0, ally_flash_ticks=0, distant_post_departures=0, search_ticks=10)
        summary = behavior_summary([record])
        self.assertEqual(summary["moving_fire_rate"], .25)
        self.assertEqual(summary["normal_stationary_aligned_fire_rate"], .75)
        moving_record = {**record, "moving_fire_decisions": 4,
                         "aligned_fire_decisions": 0, "aligned_normal_fire_decisions": 0}
        moving_summary = behavior_summary([moving_record])
        self.assertEqual(moving_summary["moving_fire_rate"], 1.)
        self.assertEqual(moving_summary["behavior_error"], summary["behavior_error"])
        baseline = dict(positioning={"passed": True}, mean_search_score=.6,
                        min_team_search_score=.4, behavior=summary)
        candidate = {**baseline, "behavior": moving_summary}
        self.assertTrue(qualifies_as_best(candidate, baseline))

    def test_actual_iq_round_preserves_setup_and_search_only_outcome(self):
        env = DefenderSearchEnv(self.model(), seed=19, opponents=["omoko_v1"])
        env.reset()
        from iq_controller_adapter import IQAwareController
        self.assertIsInstance(env.game.defender_controller, IQAwareController)
        self.assertIs(env.game.defender_controller.inner_controller.search_controller, env.controller)
        transitions = []
        with patch.object(env.adapter.default_controller, "decide_move", side_effect=AssertionError("retake during search")):
            while not env.done:
                batch, _, _ = env.step(epsilon=0.)
                transitions.extend(batch)
        self.assertGreater(env.metrics["quiet_decisions"], 0)
        self.assertEqual(env.result()["start_mode"], "round")
        self.assertTrue(all(transition[6] >= 1 for transition in transitions))
        self.assertTrue(all(len(transition[4]) == ACTION_DIM for transition in transitions))
        self.assertIn(env.result()["end_reason"], ("attacker_eliminated", "defender_eliminated", "planted", "timeout"))
        self.assertFalse(env.game.is_defused)

    def test_artificial_start_modes_are_rejected_before_creating_a_game(self):
        env = DefenderSearchEnv(self.model(), opponents=["omoko_v1"])
        self.assertEqual(START_MODES, ("round",))
        for mode in ("contact", "hold"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "normal 5v5 round from spawn"):
                env.reset(mode)
        self.assertIsNone(env.game)

    def test_all_registered_opponents_start_with_iq_and_restore_project_directory(self):
        directory = Path.cwd()
        model = self.model()
        for opponent in OPPONENTS:
            with self.subTest(opponent=opponent):
                env = DefenderSearchEnv(model, seed=2)
                initial = env.reset(opponent=opponent)
                self.assertEqual(initial["opponent"], opponent)
                self.assertEqual(initial["attacker_alive"], 5)
                self.assertEqual(initial["defender_alive"], 5)
                self.assertTrue(env.game.defender_setup_phase.active)
                self.assertEqual(env.game.battle_tick, 0)
                for team, spawn_value in ((env.attackers, 3), (env.defenders, 4)):
                    self.assertEqual(len({tuple(char.pos) for char in team}), 5)
                    self.assertTrue(all(char.is_alive and char.hp > 0 for char in team))
                    self.assertTrue(all(env.game.grid[tuple(char.pos)] == spawn_value for char in team))
                while env.game.defender_setup_phase.active and not env.done:
                    env.step(epsilon=0.)
                for _ in range(3):
                    if not env.done:
                        env.step(epsilon=0.)
                self.assertGreater(env.game.battle_tick, 0)
                self.assertTrue(env.game.current_defender_team_ai.use_iq_perception)
                self.assertEqual(Path.cwd(), directory)

    def test_planting_ends_credit_immediately_and_snapshots_survivors(self):
        env = DefenderSearchEnv(self.model(), opponents=["omoko_v1"])
        env.defenders = [actor(name, (9, 13)) for name in GORIGONS.players]
        env.attackers = [actor("enemy1", (9, 8), team="A"), actor("enemy2", (9, 9), team="A")]
        env.opponent, env.start_mode = "omoko_v1", "round"
        env.pending = [dict(obs=np.zeros(observation_dim(self.scenario), dtype=np.float16),
                            action=38, reward=0., duration=0)] + [None] * 4
        env.done, env.planted, env.elapsed_ticks, env.search_ticks, env.metrics = False, False, 0, 0, {}
        game = SimpleNamespace(defender_setup_phase=SimpleNamespace(active=False), is_planted=False,
            round_over=False, match_over=False, attacker_wins=False, is_defused=False, detonate_timer=50)
        calls = []
        def tick():
            calls.append(1)
            game.is_planted = True
            if len(calls) == 2:
                env.defenders[0].is_alive, env.defenders[0].hp = False, 0
                game.round_over, game.is_defused = True, True
        game.step_tick = tick
        env.game = game
        batch, _, done = env.step()
        self.assertTrue(done)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0][5:], (1., 1))
        self.assertAlmostEqual(batch[0][2], ROUND_REWARD * (2 * search_score("planted", 5, 2) - 1))
        self.assertEqual(env.result()["end_reason"], "planted")
        self.assertEqual(env.result()["plant_defender_alive"], 5)
        self.assertEqual(env.result()["plant_attacker_alive"], 2)
        self.assertIsNone(env.result()["winner"])
        with self.assertRaises(RuntimeError):
            env.step()
        self.assertEqual(len(calls), 1)
        self.assertFalse(game.is_defused)

    def test_plant_survival_means_exclude_episodes_without_a_plant(self):
        records = [dict(planted=True, plant_defender_alive=4, plant_attacker_alive=2, search_score=.7),
                   dict(planted=True, plant_defender_alive=2, plant_attacker_alive=4, search_score=.3),
                   dict(planted=False, plant_defender_alive=None, plant_attacker_alive=None, search_score=1.)]
        result = survival_summary(records)
        self.assertEqual(result["plant_rounds"], 2)
        self.assertEqual(result["mean_plant_defender_alive"], 3.)
        self.assertEqual(result["mean_plant_attacker_alive"], 3.)
        self.assertIn("defender=3.00 attacker=3.00", survival_log(result))
        no_plants = survival_summary(records[2:])
        self.assertIsNone(no_plants["mean_plant_defender_alive"])
        self.assertIn("defender=n/a attacker=n/a", survival_log(no_plants))

    def test_training_adapter_never_calls_retake_when_plant_completes_mid_tick(self):
        from concon_v1.co1_defender_search_training import SearchTrainingAdapter
        search = ConconDefenderSearchController(model=self.model())
        adapter = SearchTrainingAdapter(search_controller=search)
        own = actor(GORIGONS.players[0], (9, 13))
        with patch.object(adapter.default_controller, "decide_move", side_effect=AssertionError("retake used")):
            self.assertEqual(adapter.decide_move(own, {"is_planted": True}), ([9, 13], {"facing": "W"}))

    def test_survivors_affect_best_selection_at_the_same_win_rate(self):
        self.assertGreater(search_score("planted", 4, 2), search_score("planted", 2, 4))
        self.assertEqual(search_score("planted", 4, 4), search_score("planted", 1, 1))
        baseline = dict(positioning={"passed": True}, mean_win_rate=.2, min_team_win_rate=.1,
                        mean_search_score=.4, min_team_search_score=.3, behavior={"behavior_error": .5})
        improved = {**baseline, "mean_search_score": .6, "min_team_search_score": .4}
        self.assertTrue(qualifies_as_best(improved, baseline))
        self.assertFalse(qualifies_as_best(baseline, improved))

    def test_plant_score_orders_every_headcount_by_numerical_advantage(self):
        for defenders in range(1, 6):
            for attackers in range(1, 6):
                score = search_score("planted", defenders, attackers)
                self.assertAlmostEqual(score, .5 + (defenders - attackers) / 10)
                if defenders < attackers:
                    self.assertLess(score, .5)
                elif defenders > attackers:
                    self.assertGreater(score, .5)
                else:
                    self.assertEqual(score, .5)
        self.assertLess(search_score("planted", 4, 5), search_score("planted", 1, 1))
        self.assertEqual(search_score("attacker_eliminated", 3, 0), 1.)
        self.assertEqual(search_score("timeout", 3, 4), 1.)
        self.assertEqual(search_score("defender_eliminated", 0, 4), 0.)

    def test_plant_log_separates_advantage_from_preplant_wins(self):
        def record(defenders, attackers, planted=True):
            return dict(planted=planted, plant_defender_alive=defenders if planted else None,
                        plant_attacker_alive=attackers if planted else None,
                        search_score=search_score("planted" if planted else "attacker_eliminated", defenders, attackers))
        deficit = survival_summary([record(3, 4), record(3, 0, False)])
        equal = survival_summary([record(3, 3)])
        self.assertGreater(deficit["mean_search_score"], equal["mean_search_score"])
        self.assertLess(deficit["mean_plant_search_score"], equal["mean_plant_search_score"])
        self.assertAlmostEqual(deficit["mean_plant_search_score"], .4)
        self.assertEqual(equal["mean_plant_search_score"], .5)
        self.assertEqual(deficit["mean_plant_advantage"], -1.)
        self.assertEqual(deficit["plant_advantage_rate"], 0.)
        self.assertIn("plant_search_score=0.400", survival_log(deficit))
        advantageous = survival_summary([record(4, 2), record(3, 3)])
        self.assertEqual(advantageous["plant_advantage_rate"], .5)
        no_plants = survival_summary([record(3, 0, False)])
        self.assertIsNone(no_plants["mean_plant_search_score"])
        self.assertIsNone(no_plants["plant_advantage_rate"])

    def test_checkpoint_load_and_best_selection_keep_basic_and_search_separate(self):
        model = self.model()
        run = dict(start_episode=100, support_distance=6, requested_additional_episodes=1000)
        checkpoint = make_battle_checkpoint(model, self.scenario, 150, run, ["omoko_v1"], {"passed": True})
        self.assertEqual(checkpoint["battle_episodes_this_run"], 50)
        self.assertNotEqual(self.scenario.model_path.name, self.scenario.battle_model_path().name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "search.pt"
            torch.save(checkpoint, path)
            with contextlib.redirect_stdout(io.StringIO()):
                controller = ConconDefenderSearchController(model_path=path)
            self.assertTrue(controller.model.combat)
            self.assertTrue(torch.equal(model.head[2].weight, controller.model.head[2].weight))
        baseline = dict(positioning={"passed": True}, mean_search_score=.5, min_team_search_score=.3, behavior={"behavior_error": 1.})
        candidate = {**baseline, "positioning": {"passed": False}, "mean_win_rate": 1.}
        self.assertFalse(qualifies_as_best(candidate, baseline))
        candidate = {**baseline, "behavior": {"behavior_error": .5}}
        self.assertTrue(qualifies_as_best(candidate, baseline))
        self.assertEqual(epsilon_by_episode(1000, 1000), EPSILON_END)

    def test_first_best_requires_positioning_but_no_foundation_comparison(self):
        candidate = dict(positioning={"passed": True}, mean_search_score=-1.,
                         min_team_search_score=-2., behavior={"behavior_error": 1.})
        identity = dict(path="latest.pt", episode=100, policy_type="search")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.jsonl"
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertTrue(log_best_comparison(candidate, None, identity, None, {"seed": 0}, path))
                failed = {**candidate, "positioning": {"passed": False}}
                self.assertFalse(log_best_comparison(failed, None, identity, None, {"seed": 0}, path))
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["accepted"] for record in records], [True, False])
            self.assertIsNone(records[0]["current"])
            self.assertIn("CREATE first best", output.getvalue())
            self.assertIn("NO best saved", output.getvalue())
        improved = {**candidate, "mean_search_score": 0.}
        self.assertTrue(qualifies_as_best(improved, candidate))
        self.assertFalse(qualifies_as_best(candidate, improved))

    def test_retraining_replaces_previous_best_then_selects_best_within_run(self):
        first = dict(positioning={"passed": True}, mean_search_score=-1.,
                     min_team_search_score=-2., behavior={"behavior_error": 1.}, opponents={})
        second = {**first, "mean_search_score": -3.}
        module = "concon_v1.co1_defender_search_battle_training"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resume = directory / "foundation.pt"
            best_path = directory / self.scenario.battle_model_path().name
            torch.save(self.foundation, resume)
            torch.save({**self.foundation, "episode": 999}, best_path)
            with patch(module + ".DefenderSearchEnv") as env, \
                    patch(module + ".evaluate_positioning", return_value={"passed": True}), \
                    patch(module + ".epsilon_by_episode", return_value=EPSILON_END), \
                    patch(module + ".evaluate", side_effect=[first, second]) as evaluate_mock, \
                    patch(module + ".print_summary"), contextlib.redirect_stdout(io.StringIO()):
                env.return_value.done = True
                env.return_value.result.return_value = {"opponent": "fixture"}
                train_battle(episodes=2, save_dir=directory, resume=resume,
                             opponents=["omoko_v1"], eval_rounds=1, checkpoint_interval=1)
            self.assertEqual(evaluate_mock.call_count, 2)
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            self.assertEqual(best["episode"], 1)
            self.assertEqual(best["evaluation"]["mean_search_score"], -1.)
            records = [json.loads(line) for line in
                       (directory / "battle_best_comparison_log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIsNone(records[0]["current"])
            self.assertTrue(records[0]["accepted"])
            self.assertEqual(records[1]["current"]["episode"], 1)
            self.assertFalse(records[1]["accepted"])

    def test_opponent_schedule_balances_full_and_partial_windows(self):
        opponents = tuple(OPPONENTS)
        for count in (1, 3, 5, 12, 50):
            with self.subTest(count=count):
                schedule = balanced_opponent_schedule(opponents, count, random.Random(19))
                self.assertEqual(len(schedule), count)
                counts = Counter(schedule)
                totals = [counts[name] for name in opponents]
                self.assertLessEqual(max(totals) - min(totals), 1)
                self.assertEqual(schedule, balanced_opponent_schedule(opponents, count, random.Random(19)))
                if count == 50:
                    self.assertEqual(sum(totals), 50)

    def test_training_balances_each_summary_window_and_final_partial_window(self):
        module = "concon_v1.co1_defender_search_battle_training"
        selected = []
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resume = directory / "foundation.pt"
            torch.save(self.foundation, resume)
            with patch(module + ".DefenderSearchEnv") as env, \
                    patch(module + ".evaluate_positioning", return_value={"passed": True}), \
                    patch(module + ".epsilon_by_episode", return_value=1.), \
                    patch(module + ".evaluate") as evaluate_mock, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                def reset(*, opponent):
                    selected.append(opponent)

                env.return_value.reset.side_effect = reset
                env.return_value.done = True
                env.return_value.result.side_effect = lambda: dict(
                    opponent=selected[-1], winner="D", end_reason="attacker_eliminated",
                    planted=False, search_score=1.)
                train_battle(episodes=112, save_dir=directory, resume=resume,
                             checkpoint_interval=50)
            for begin in (0, 50):
                counts = Counter(selected[begin:begin + 50])
                totals = [counts[name] for name in OPPONENTS]
                self.assertEqual(sum(totals), 50)
                self.assertLessEqual(max(totals) - min(totals), 1)
            final_counts = Counter(selected[100:])
            final_totals = [final_counts[name] for name in OPPONENTS]
            self.assertEqual(sum(final_totals), 12)
            self.assertLessEqual(max(final_totals) - min(final_totals), 1)
            self.assertEqual(output.getvalue().count("Training summary:"), 3)
            evaluate_mock.assert_not_called()
            latest = torch.load(directory / self.scenario.battle_model_path("latest").name,
                                map_location="cpu", weights_only=False)
            self.assertEqual(latest["battle_training_run"]["opponent_sampling"],
                             "balanced_per_checkpoint_window")

    def test_old_map_best_is_preserved_and_new_base_becomes_comparison(self):
        buffer = io.BytesIO()
        torch.save(self.foundation, buffer)
        foundation_bytes = buffer.getvalue()
        old_best = {**self.foundation, "scenario_signature": "old_map"}
        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / self.scenario.battle_model_path().name
            torch.save(old_best, best_path)
            original = best_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                selected = comparison_checkpoint(best_path, foundation_bytes, self.scenario)
            self.assertEqual(selected, foundation_bytes)
            self.assertFalse(best_path.exists())
            backups = list(Path(directory).glob("*_incompatible_*.pt"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            runtime_scenario = replace(self.scenario)
            with patch.object(type(runtime_scenario), "save_dir", new=property(lambda _: Path(directory))):
                self.assertEqual(runtime_scenario.runtime_model_path, runtime_scenario.model_path)

    def test_valid_comparison_best_is_kept_and_absent_best_uses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            best_path = Path(directory) / "best.pt"
            self.assertEqual(comparison_checkpoint(best_path, b"resume", self.scenario), b"resume")
            torch.save(self.foundation, best_path)
            original = best_path.read_bytes()
            self.assertEqual(comparison_checkpoint(best_path, b"resume", self.scenario), original)
            self.assertEqual(best_path.read_bytes(), original)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 1)

    def test_explicit_old_map_resume_still_fails_checkpoint_validation(self):
        old = {**self.foundation, "scenario_signature": "old_map"}
        with self.assertRaisesRegex(ValueError, "checkpoint/map mismatch"):
            load_search_weights(self.model(), old, self.scenario)

    def test_runtime_selects_current_base_when_best_has_an_old_map(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(type(self.scenario), "save_dir", new=property(lambda _: Path(directory))):
                best = self.scenario.battle_model_path()
                torch.save({**self.foundation, "scenario_signature": "old_map"}, best)
                original = best.read_bytes()
                self.assertEqual(self.scenario.runtime_model_path, self.scenario.model_path)
                self.assertEqual(best.read_bytes(), original)
                torch.save(self.foundation, best)
                self.assertEqual(self.scenario.runtime_model_path, best)

    def test_cli_defaults_to_additional_search_learning_and_keeps_basic_mode_explicit(self):
        from concon_v1.co1_train_defender_search import main
        with patch.object(sys, "argv", ["co1_train_defender_search.py", "--episodes", "7"]), \
                patch("concon_v1.co1_defender_search_battle_training.train_battle") as train:
            main()
            self.assertEqual(train.call_args.kwargs["episodes"], 7)
            self.assertIsNone(train.call_args.kwargs["resume"])
        with patch.object(sys, "argv", ["co1_train_defender_search.py", "--mode", "positioning"]), \
                patch("concon_v1.co1_train_defender_search.train") as train:
            main()
            train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
