"""Curriculum limits, evaluation isolation and multi-agent replay separation."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
import train_attacker_gc_real_curriculum as curriculum
import navigation_intent_gc as intent


class RealCurriculumTests(unittest.TestCase):
    def test_navigation_loop_does_not_repeat_progress_reward(self):
        tracker = curriculum.ProgressTracker()
        self.assertGreater(tracker.step("Absol", (2, 2), 10, 9, False), 0)
        rewards = []
        for _ in range(10):
            rewards.append(tracker.step("Absol", (2, 2), 9, 10, False))
            rewards.append(tracker.step("Absol", (2, 2), 10, 9, False))
        self.assertLess(sum(rewards), 0)

    def test_final_plant_approach_does_not_teach_a_one_cell_hold(self):
        obs = np.zeros(57, dtype=np.float32)
        obs[25] = obs[9] = 1 / 80
        obs[-1] = 1
        obs[39:43] = (-1, -1, -1, 1)
        mask = np.ones(11, dtype=bool)
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask), 0)
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask, final_approach=True), 8)
        obs[25] = 0
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask, final_approach=True), 10)

    def test_escort_teacher_clears_carrier_route_even_at_own_waypoint(self):
        from types import SimpleNamespace as NS
        obs = np.zeros(67, dtype=np.float32)
        obs[40] = obs[-1] = 1
        obs[49:53] = (-1, 1, -1, -1)
        mask = np.ones(6, dtype=bool)
        mask[1] = False
        with patch.object(curriculum, "can_engage", return_value=False):
            action = curriculum.observable_teacher_action("escort", obs, mask, NS(game=object()), (object(), {"chars": []}))
        self.assertIn(action, (0, 2, 3))

        # v6 also clears any adjacent progress cell, including alternate route
        # cells that the legacy exact-next-cell feature (index 40) misses.
        clearance_obs = np.zeros(76, dtype=np.float32)
        clearance_obs[75] = 1
        clearance_obs[49:53] = (-1, 1, -1, -1)
        with patch.object(curriculum, "can_engage", return_value=False):
            action = curriculum.observable_teacher_action(
                "escort", clearance_obs, np.ones(6, dtype=bool),
                NS(game=object()), (object(), {"chars": []}))
        self.assertEqual(action, 1)

        directional_obs = np.zeros(80, dtype=np.float32)
        directional_obs[75] = 1
        directional_obs[76:80] = (0, 0, 1, 0)
        directional_obs[49:53] = (1, 1, -1, 1)
        with patch.object(curriculum, "can_engage", return_value=False):
            action = curriculum.observable_teacher_action(
                "escort", directional_obs, np.ones(6, dtype=bool),
                NS(game=object()), (object(), {"chars": []}))
        self.assertEqual(action, curriculum.escort_runtime.ACTION_LEFT)

    def test_observable_combat_teacher_stops_all_three_phases(self):
        from types import SimpleNamespace as NS
        with patch.object(curriculum, "can_engage", return_value=True):
            for phase, dim, action in (("carry", 57, 0), ("escort", 67, 4), ("guard", 34, 0)):
                self.assertEqual(curriculum.observable_teacher_action(phase, np.zeros(dim), np.ones(11 if phase != "escort" else 6, dtype=bool),
                                 NS(game=object()), (object(), {"chars": []})), action)

    def test_labels_are_collected_after_bootstrap_without_replacing_policy_action(self):
        session = curriculum.CurriculumSession.__new__(curriculum.CurriculumSession)
        net = Mock(return_value=torch.tensor([[0., 0., 5., 0., 0., 0.]]))
        session.policies = {"escort": net}
        session.controllers = {"escort": object()}
        session.actor = "escort"
        session.pending = {}
        session.demonstrations = {"escort": []}
        session.collect_demonstrations = True
        session.teacher_probability = session.epsilon = 0
        session.decision_context = None
        choose = session.selector("escort")
        with patch.object(curriculum, "observable_teacher_action", return_value=0):
            self.assertEqual(choose(np.zeros(67, dtype=np.float32), np.ones(6, dtype=bool)), 2)
        self.assertEqual(session.pending[("escort", "escort")].action, 2)
        self.assertEqual(session.demonstrations["escort"][0][1], 0)

    def test_quality_gate_rejects_bad_seed_hidden_by_average(self):
        metrics = dict(carry_no_entry_rate=.2, timeout_rate=.08,
                       worst_carry_no_entry_rate=.4, worst_timeout_rate=.2,
                       worst_round_win_rate=.3, round_win_rate=.4, carry_spawn_tick_rate=.1)
        self.assertFalse(curriculum.entry_quality_passed(metrics, .25, .1))
        self.assertEqual(curriculum.selection_score(metrics, .25, .1)[0], 0)

    def test_guard_teacher_uses_wrapper_perception_without_a_controller_game(self):
        from types import SimpleNamespace as NS
        session = curriculum.CurriculumSession.__new__(curriculum.CurriculumSession)
        view = object()
        session.game = NS(attacker_controller=NS(inner_controller=NS(game=view)))
        session.controllers = {"guard": NS()}
        session.policies = {"guard": Mock(return_value=torch.zeros((1, 11)))}
        session.actor = "guard"
        session.pending = {}
        session.collect_demonstrations = True
        session.demonstrations = {"guard": []}
        session.teacher_probability = session.epsilon = 0
        session.decision_context = (object(), {"chars": []})
        session.action_goals = {}
        with patch.object(curriculum, "observable_teacher_action", return_value=0) as label:
            session.selector("guard")(np.zeros(34, dtype=np.float32), np.ones(11, dtype=bool))
        self.assertIs(label.call_args.kwargs["view"], view)

    def test_actual_duel_stay_has_no_unlimited_positive_reward(self):
        self.assertEqual(curriculum.combat_reward("carry", 0, False), 0)
        self.assertLess(curriculum.combat_reward("carry", 0, True) - 0.05, 0)
        self.assertGreater(curriculum.combat_reward("carry", 1, True), curriculum.combat_reward("carry", 2, True))
        self.assertGreater(curriculum.combat_reward("escort", 4, True), curriculum.combat_reward("escort", 0, True))

    def test_navigation_bonus_cannot_reward_moving_during_an_actual_duel(self):
        tracker = curriculum.ProgressTracker()
        advance = tracker.step("Absol", (2, 2), 10, 9, True)
        moving = advance + curriculum.combat_reward("carry", 2, True) - 0.05
        standing = curriculum.combat_reward("carry", 0, True) - 0.05
        self.assertLess(moving, standing)
        self.assertEqual(advance, 0)

    def test_expanded_models_preserve_source_when_new_features_are_zero(self):
        for old_dim, new_dim, actions in ((31, 39, 11), (41, 49, 6), (39, 57, 11),
                                          (49, 67, 6), (57, 61, 11), (67, 71, 6),
                                          (61, 65, 11), (65, 66, 11),
                                          (71, 75, 6), (75, 76, 6), (76, 80, 6)):
            source = curriculum.escort_runtime.DuelingQNetwork(old_dim, actions)
            extended = curriculum.escort_runtime.DuelingQNetwork(new_dim, actions)
            extended.load_state_dict(curriculum.expand_policy_state({"model_state_dict": source.state_dict()}, new_dim))
            old = torch.randn(4, old_dim)
            new = torch.cat((old, torch.zeros(4, new_dim - old_dim)), dim=1)
            torch.testing.assert_close(source(old), extended(new))

    def test_screening_features_measure_teammates_ahead_of_carrier(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((5, 20), dtype=int)
        carrier = NS(name="Absol", team="A", pos=(2, 3), is_alive=True, has_spike=True)
        ahead = NS(name="Xdll", team="A", pos=(2, 6), is_alive=True, has_spike=False)
        behind = NS(name="eKo", team="A", pos=(2, 1), is_alive=True, has_spike=False)
        env = NS(current_strategy="A_SPLIT",
                 targets={c.name: (2, 15) for c in (carrier, ahead, behind)},
                 assignment={"Absol": ("A", "SITE", "MAIN"),
                             "Xdll": ("A", "SITE", "MAIN"),
                             "eKo": ("A", "SITE", "MAIN")})
        game = NS(grid=grid, target_plant_pos=(2, 15), attacker_controller=NS(macro_controller=NS(env=env)))
        carry_features = intent.carrier_screening_features(game, carrier, [carrier, ahead, behind], {}, False)
        escort_features = intent.carrier_screening_features(game, ahead, [carrier, ahead, behind], {}, True)
        self.assertEqual(carry_features[1], .25)
        self.assertEqual(carry_features[2], .5)
        self.assertEqual(carry_features[3], 0)
        self.assertEqual(escort_features[1], 1)
        self.assertEqual(escort_features[3], 1)
        formation = intent.carrier_formation_features(
            game, ahead, [carrier, ahead, behind], {}, True)
        self.assertEqual(formation[0], 1)  # selected MAIN escort
        self.assertEqual(formation[1], 1)  # exactly three route cells ahead
        status = intent.carrier_screening_status(game, carrier, [carrier, ahead, behind], {})
        self.assertFalse(intent.designated_route_blocking(status, carrier))
        ahead.pos = (2, 4)
        status = intent.carrier_screening_status(game, carrier, [carrier, ahead, behind], {})
        self.assertTrue(intent.designated_route_blocking(status, carrier))
        self.assertEqual(intent.carrier_route_blocking_feature(
            game, ahead, [carrier, ahead, behind], {}), 1)
        np.testing.assert_array_equal(
            intent.carrier_route_clearance_features(
                game, ahead, [carrier, ahead, behind], {}),
            np.array((1, 1, 0, 1), dtype=np.float32),
        )
        ahead.pos = (2, 1)
        behind.pos = (2, 5)
        # The assignment is stable while strategy/carrier/group stay unchanged.
        status = intent.carrier_screening_status(game, carrier, [carrier, ahead, behind], {})
        self.assertEqual(status["designated"].name, "Xdll")

    def test_screening_assignment_follows_mid_group_and_skips_fake_sell(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((5, 20), dtype=int)
        carrier = NS(name="carry", team="A", pos=(2, 3), is_alive=True, has_spike=True)
        mid = NS(name="mid", team="A", pos=(2, 2), is_alive=True, has_spike=False)
        b_group = NS(name="b", team="A", pos=(3, 2), is_alive=True, has_spike=False)
        env = NS(current_strategy="MID_TO_B",
                 targets={c.name: (2, 15) for c in (carrier, mid, b_group)},
                 assignment={"carry": ("MID", "DEEP", "MID"),
                             "mid": ("MID", "DEEP", "MID"),
                             "b": ("B", "STAGING", "B")})
        game = NS(grid=grid, target_plant_pos=(2, 15),
                  attacker_controller=NS(macro_controller=NS(env=env)))
        status = intent.carrier_screening_status(game, carrier, [carrier, mid, b_group], {})
        self.assertEqual(status["designated"].name, "mid")
        env.current_strategy = "FAKE_A_TO_B"
        env.assignment["carry"] = ("MID", "STAGING", "FAKE_WAIT")
        env.assignment["mid"] = ("A", "FORWARD", "FAKE_SELL")
        self.assertIsNone(intent.carrier_screening_status(
            game, carrier, [carrier, mid, b_group], {})["designated"])

    def test_screening_teacher_keeps_carrier_moving_and_sends_one_escort_to_formation(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((5, 20), dtype=int)
        carrier = NS(name="Absol", team="A", pos=(2, 3), is_alive=True, has_spike=True)
        escort = NS(name="Xdll", team="A", pos=(2, 2), is_alive=True, has_spike=False)
        env = NS(current_strategy="A_SPLIT", targets={"Absol": (2, 15), "Xdll": (2, 15)},
                 assignment={"Absol": ("A", "SITE", "MAIN"),
                             "Xdll": ("A", "SITE", "MAIN")})
        game = NS(grid=grid, target_plant_pos=(2, 15), battle_tick=10, round_timer=80,
                  attacker_controller=NS(macro_controller=NS(env=env)))
        state = {"chars": [carrier, escort], "round_timer": 80}
        carry_controller = NS(game=game, _intent_distance_cache={})
        escort_controller = NS(game=game, _intent_distance_cache={})
        with patch.object(curriculum, "can_engage", return_value=False):
            carry_obs = np.zeros(66)
            carry_obs[curriculum.runtime.TACTICAL_OBS_DIM + 3] = 1
            carry_mask = np.ones(11, dtype=bool)
            carry_mask[curriculum.runtime.PLANT_ACTION_INDEX] = False
            self.assertEqual(curriculum.observable_teacher_action(
                "carry", carry_obs, carry_mask, carry_controller,
                (carrier, state)), 8)
            escort_mask = np.ones(6, dtype=bool)
            escort_mask[curriculum.escort_runtime.ACTION_RIGHT] = False
            self.assertIn(curriculum.observable_teacher_action(
                "escort", np.zeros(75), escort_mask, escort_controller,
                (escort, state)), (curriculum.escort_runtime.ACTION_UP,
                                    curriculum.escort_runtime.ACTION_DOWN))

    def test_screening_teacher_remains_active_in_last_eight_route_cells(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((5, 20), dtype=int)
        carrier = NS(name="Absol", team="A", pos=(2, 7), is_alive=True, has_spike=True)
        escort = NS(name="Xdll", team="A", pos=(2, 6), is_alive=True, has_spike=False)
        env = NS(current_strategy="A_RUSH", targets={"Absol": (2, 15), "Xdll": (2, 15)},
                 assignment={"Absol": ("A", "SITE", "MAIN"),
                             "Xdll": ("A", "SITE", "MAIN")})
        game = NS(grid=grid, target_plant_pos=(2, 15), battle_tick=10, round_timer=80,
                  attacker_controller=NS(macro_controller=NS(env=env)))
        controller = NS(game=game, _intent_distance_cache={})
        state = {"chars": [carrier, escort], "round_timer": 80}
        with patch.object(curriculum, "can_engage", return_value=False):
            action = curriculum.observable_teacher_action(
                "escort", np.zeros(75), np.ones(6, dtype=bool), controller,
                (escort, state))
        self.assertEqual(action, curriculum.escort_runtime.ACTION_RIGHT)

        with patch.object(curriculum, "can_engage", return_value=False):
            carry_action = curriculum.observable_teacher_action(
                "carry", np.zeros(66), np.ones(11, dtype=bool), controller,
                (carrier, state))
        self.assertEqual(carry_action, 0)  # Let the nearby screener establish entry.

        screen = intent.carrier_screening_status(game, carrier, [carrier, escort], {})
        fresh = np.zeros(66)
        self.assertTrue(curriculum.carrier_entry_sync_needed(screen, state, fresh))
        self.assertFalse(curriculum.carrier_entry_sync_needed(
            screen, {"round_timer": 14}, fresh))
        waited = fresh.copy()
        waited[curriculum.runtime.TACTICAL_OBS_DIM + 14] = 0.10
        self.assertFalse(curriculum.carrier_entry_sync_needed(screen, state, waited))
        screen["designated_distance"] = 7
        self.assertFalse(curriculum.carrier_entry_sync_needed(screen, state, fresh))

    def test_feature_only_training_preserves_zero_feature_behavior(self):
        policy = curriculum.runtime.AttackerCarryDuelingDQN(obs_dim=66)
        zero_feature = torch.randn(4, 66)
        zero_feature[:, 65] = 0
        before = policy(zero_feature).detach().clone()
        old_column = policy.feature[0].weight[:, :65].detach().clone()
        new_column = policy.feature[0].weight[:, 65].detach().clone()
        curriculum.restrict_policy_to_input_columns(policy, (65,))
        optimizer = torch.optim.Adam(
            [parameter for parameter in policy.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        active = zero_feature.clone()
        active[:, 65] = 1
        loss = -policy(active)[:, 0].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.testing.assert_close(policy(zero_feature), before)
        torch.testing.assert_close(policy.feature[0].weight[:, :65], old_column)
        self.assertFalse(torch.equal(policy.feature[0].weight[:, 65], new_column))

    def test_navigation_history_reports_executed_loop_and_local_occupancy(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((5, 5), dtype=int)
        grid[1, 2] = 1
        distances = curriculum.runtime._bfs_distance_map(grid, (2, 4))
        game = NS(grid=grid, battle_tick=0, round_timer=100)
        char = NS(name="Absol", pos=(2, 2), is_alive=True)
        ally = NS(name="ally", pos=(2, 3), is_alive=True)
        history = {}
        first = intent.navigation_context_features(game, char, [char, ally], distances, history, (2, 4))
        self.assertEqual(first[4], 1)  # wall above
        self.assertEqual(first[11], 1)  # occupied forward cell
        self.assertEqual(first[3], 1)  # its distance gain remains observable
        game.battle_tick = 1
        char.pos = (2, 1)
        intent.navigation_context_features(game, char, [char, ally], distances, history, (2, 4))
        game.battle_tick = 2
        char.pos = (2, 2)
        returned = intent.navigation_context_features(game, char, [char, ally], distances, history, (2, 4))
        np.testing.assert_array_equal(returned[12:14], (0, 1))
        self.assertGreater(returned[14], 0)
        self.assertGreater(returned[15], 0)
        # Repeated builds during one tick must not rewrite movement/history.
        np.testing.assert_array_equal(returned, intent.navigation_context_features(game, char, [char, ally], distances, history, (2, 4)))
        history.clear()
        self.assertEqual(intent.navigation_context_features(game, char, [char, ally], distances, history, (2, 4))[15], 0)

    def test_teacher_respects_macro_route_mask_and_leaves_fights_to_policy(self):
        obs = np.zeros(57, dtype=np.float32)
        obs[25] = 0.5
        obs[39:43] = (1, -1, -1, 1)
        mask = np.ones(11, dtype=bool)
        mask[2] = False
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask), 8)
        obs[13] = 1
        self.assertIsNone(curriculum.navigation_teacher_action("carry", obs, mask))
        obs[13] = 0
        obs[25] = 0
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask), 10)
        obs[25] = 0.5  # An ordinary site tile does not imply forced planting.
        self.assertEqual(curriculum.navigation_teacher_action("carry", obs, mask), 8)
        escort = np.zeros(67, dtype=np.float32)
        escort[49:53] = (-1, -1, 1, -1)
        self.assertEqual(curriculum.navigation_teacher_action("escort", escort, np.ones(6, dtype=bool)), 2)

    def test_n_step_credit_preserves_actor_phase_boundaries_and_durations(self):
        obs = np.zeros(2, dtype=np.float32)
        mask = np.ones(2, dtype=bool)
        def row(action, reward, terminal=False, ticks=1):
            return (obs, action, reward, obs, mask, terminal, ticks)
        rows = [("a", row(0, 2, ticks=2)), ("b", row(1, 100, True)),
                ("a", row(1, 4, True)), ("a", row(0, 999, True))]
        backed_up = list(curriculum.n_step_transitions(rows, 0.5, 5))
        self.assertEqual(backed_up[0][2], 3)
        self.assertEqual(backed_up[0][6], 3)
        self.assertTrue(backed_up[0][5])
        self.assertEqual([r[2] for r in backed_up[1:]], [4, 999, 100])

    def test_failed_checkpoint_selection_prioritizes_entry_failures(self):
        failed = dict(carry_no_entry_rate=0.5, timeout_rate=0.4, worst_round_win_rate=0.3,
                      round_win_rate=0.3, carry_spawn_tick_rate=0.1)
        progress = dict(failed, carry_no_entry_rate=0.3, timeout_rate=0.2, worst_round_win_rate=0.2)
        self.assertGreater(curriculum.selection_score(progress, .25, .1), curriculum.selection_score(failed, .25, .1))
        passed = dict(progress, carry_no_entry_rate=.2, timeout_rate=.05, worst_round_win_rate=.1)
        self.assertGreater(curriculum.selection_score(passed, .25, .1), curriculum.selection_score(progress, .25, .1))

    def test_new_carry_ability_actions_have_one_stationary_meaning(self):
        from types import SimpleNamespace as NS
        rt = curriculum.runtime
        controller = rt.LearningAttackerCarryGCController.__new__(rt.LearningAttackerCarryGCController)
        controller.positioning_version = 5
        controller.game = NS(grid=np.zeros((5, 5), dtype=int))
        char = NS(name="Absol", pos=(2, 2), ability_name="FLASH", flash_charges=1, smoke_charges=0, recon_charges=0)
        controller._learned_ability_target = Mock(return_value=(2, 4))
        mask = controller._build_mask(char, [char], False)
        self.assertTrue(mask[1])
        self.assertFalse(any(mask[[3, 5, 7, 9]]))
        self.assertTrue(all(mask[[0, 2, 4, 6, 8]]))
        controller._learned_ability_target.return_value = None
        self.assertFalse(controller._build_mask(char, [char], False)[1])

    def test_new_escort_masks_currently_occupied_route_cells(self):
        from types import SimpleNamespace as NS
        rt = curriculum.escort_runtime
        controller = rt.LearningAttackerEscortGCController.__new__(rt.LearningAttackerEscortGCController)
        controller.positioning_version = 3
        char = NS(name="escort", pos=(2, 2), flash_charges=0, smoke_charges=0, recon_charges=0)
        ally = NS(name="Absol", pos=(2, 3), is_alive=True)
        mask = controller._action_mask(char, np.zeros((5, 5), dtype=int), [char, ally])
        self.assertFalse(mask[rt.ACTION_RIGHT])
        self.assertTrue(mask[rt.ACTION_UP])
        self.assertTrue(mask[rt.ACTION_STAY])

    def test_completed_plant_counts_as_entry_despite_macro_retarget(self):
        self.assertTrue(curriculum.entered_site(dict(planted=True, carry_entry_tick=None)))
        self.assertTrue(curriculum.entered_site(dict(planted=False, carry_entry_tick=40)))
        self.assertFalse(curriculum.entered_site(dict(planted=False, carry_entry_tick=None)))

    def test_evaluation_disables_teacher_even_when_explicitly_requested(self):
        from types import SimpleNamespace as NS
        session = curriculum.CurriculumSession.__new__(curriculum.CurriculumSession)
        session.arrived = set()
        session.action_goals = {}
        session.transitions = {p: [] for p in curriculum.PHASES}
        session.controllers = {"guard": NS(_assigned_guard_positions={})}
        session.pending = {}
        session.reset = Mock()
        session.game = NS(chars=[], attacker_wins=0, round_over=True, is_planted=False,
                          battle_tick=0, round_timer=100)
        session.play(123, (100, 60, 40), training=False, teacher_probability=1.0)
        self.assertEqual(session.teacher_probability, 0)
        self.assertFalse(session.collect_demonstrations)
        self.assertTrue(all(not samples for samples in session.demonstrations.values()))

    def test_fake_role_uses_macro_waypoint_instead_of_carrier_target(self):
        from types import SimpleNamespace as NS
        grid = np.zeros((7, 9), dtype=int)
        macro = NS(env=NS(current_strategy="FAKE_A_TO_B", targets={"eKo": (2, 2)},
                          assignment={"eKo": ("A", "SITE", "FAKE_SELL")}))
        game = NS(grid=grid, target_plant_pos=(2, 7), attacker_controller=NS(macro_controller=macro))
        escort = NS(name="eKo", has_spike=False)
        self.assertEqual(intent.navigation_intent(game, escort), ((2, 2), "FAKE_A_TO_B", "FAKE_SELL"))

    def test_combat_visibility_excludes_ally_block_and_back_of_head(self):
        from types import SimpleNamespace as NS
        a = NS(name="a", pos=(2, 1), team="A", is_alive=True)
        d = NS(name="d", pos=(2, 4), team="D", is_alive=True)
        ally = NS(name="ally", pos=(2, 2), team="A", is_alive=True)
        game = NS(check_line_of_sight=Mock(return_value=True),
                  _line_cells=Mock(return_value=[(2, 1), (2, 2), (2, 3), (2, 4)]),
                  _facing_angle_diff=Mock(return_value=0))
        self.assertFalse(intent.can_engage(game, a, [a, d, ally]))
        self.assertTrue(intent.can_engage(game, a, [a, d]))
        game._facing_angle_diff.return_value = 180
        self.assertFalse(intent.can_engage(game, a, [a, d]))

    def test_intent_escort_action_is_executed_without_macro_override(self):
        from types import SimpleNamespace as NS
        import ghost_champions_v1_macro as wrapper
        ctrl = wrapper.GhostChampionsV1AttackerController.__new__(wrapper.GhostChampionsV1AttackerController)
        ctrl.carry = NS(positioning_version=4)
        ctrl.escort = NS(positioning_version=2)
        ctrl.macro_controller = Mock()
        ctrl.macro_controller.coordinate.side_effect = AssertionError("movement override")
        holder = NS(name="Absol", team="A", is_alive=True, has_spike=True)
        escort = NS(name="eKo", team="A", is_alive=True, has_spike=False)
        result = [4, 3]
        with patch.object(wrapper._BaseGCAttacker, "decide_move", return_value=result):
            self.assertIs(ctrl.decide_move(escort, {"chars": [holder, escort]}), result)
        ctrl.macro_controller._sync_tick_once.assert_called_once()

    def test_opponent_starts_weak_and_stays_at_final_strength(self):
        start, final = (50, 10, 0), (100, 60, 40)
        self.assertEqual(curriculum.opponent_stats(1, 6000, start, final), start)
        self.assertEqual(curriculum.opponent_stats(6000, 6000, start, final), final)
        self.assertEqual(curriculum.opponent_stats(10000, 6000, start, final), final)
        self.assertEqual(curriculum.opponent_stats(1, 0, start, final), final)

    def test_strength_increases_monotonically(self):
        levels = [curriculum.opponent_stats(i, 10, (50, 10, 0), (100, 60, 40)) for i in range(1, 15)]
        for before, after in zip(levels, levels[1:]):
            self.assertTrue(all(a <= b for a, b in zip(before, after)))

    def test_evaluation_uses_final_opponent_and_normal_plans(self):
        session = Mock()
        def play(*args, **kwargs):
            random.random()
            np.random.random()
            return dict(attacker_win=True, planted=True, registered=True,
                        guard_position_ticks=2, guard_ticks=4, ticks=70)
        session.play.side_effect = play
        random.seed(37)
        np.random.seed(37)
        expected_py, expected_np = random.random(), np.random.random()
        random.seed(37)
        np.random.seed(37)
        metrics = curriculum.evaluate(session, 3, 100, (100, 60, 40))
        self.assertEqual(random.random(), expected_py)
        self.assertEqual(np.random.random(), expected_np)
        self.assertEqual(metrics["round_win_rate"], 1)
        self.assertEqual(metrics["guard_position_tick_rate"], 0.5)
        for i, call in enumerate(session.play.call_args_list):
            self.assertEqual(call.args, (100 + i, (100, 60, 40)))
            self.assertEqual(call.kwargs, {"training": False})

    def test_shared_escort_keeps_each_actors_transition_separate(self):
        session = curriculum.CurriculumSession.__new__(curriculum.CurriculumSession)
        session.policies = {"escort": curriculum.escort_runtime.DuelingQNetwork(41, 6)}
        session.pending = {}
        session.transitions = {"escort": []}
        session.epsilon = 0
        obs = np.zeros(41, dtype=np.float32)
        mask = np.ones(6, dtype=bool)
        choose = session.selector("escort")
        session.actor = "a"
        choose(obs, mask)
        session.pending[("escort", "a")].reward = 3
        session.pending[("escort", "a")].ticks = 1
        session.actor = "b"
        choose(obs, mask)
        self.assertFalse(session.transitions["escort"])
        session.actor = "a"
        choose(obs + 1, mask)
        name, transition = session.transitions["escort"][0]
        self.assertEqual(name, "a")
        self.assertEqual(transition[2], 3)
        np.testing.assert_array_equal(transition[3], obs + 1)
        self.assertIn(("escort", "b"), session.pending)

    def test_new_escort_policy_owns_movement_near_carrier(self):
        from types import SimpleNamespace as NS
        rt = curriculum.escort_runtime
        controller = rt.LearningAttackerEscortGCController.__new__(rt.LearningAttackerEscortGCController)
        controller.positioning_version = 1
        controller._char_state = {}
        controller._build_obs = Mock(return_value=np.zeros(41, dtype=np.float32))
        controller._action_mask = Mock(return_value=np.ones(6, dtype=bool))
        controller._select_action = Mock(return_value=rt.ACTION_UP)
        char = NS(name="escort", pos=[2, 2], team="A", is_alive=True)
        carrier = NS(name="carry", pos=[2, 3], team="A", is_alive=True, has_spike=True)
        state = {"grid": np.zeros((5, 5), dtype=int), "chars": [char, carrier]}
        with patch.object(rt, "choose_pre_entry_ability", return_value=None):
            self.assertEqual(controller.decide_move(char, state), [1, 2])


if __name__ == "__main__":
    unittest.main()
