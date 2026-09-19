"""CPU regression tests; no training and no checkpoint writes."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "gc_v1"))
import train_attacker_macro_gc_v28 as training
import learning_attacker_macro_gc_runtime as runtime
from ghost_champions_v1_macro import GhostChampionsV1AttackerController


def unit(name, pos, spike=False):
    return NS(name=name, pos=pos, team="A", is_alive=True, has_spike=spike)


class MacroCoordinationTests(unittest.TestCase):
    def cover_controller(self, strategy="A_SPLIT", phase=None):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController)
        holder = unit("Absol", (5, 5), True)
        units = [holder, unit("main1", (4, 5)), unit("main2", (5, 4)),
                 unit("other1", (12, 12)), unit("other2", (13, 12))]
        controller.env = NS(current_strategy=strategy, _fake_phase=phase,
                            _fake_group_names={"other1", "other2"},
                            assignment={c.name: ("A", "DEEP", "MAIN")
                                        for c in units})
        return controller, holder, units

    def test_split_cover_does_not_recall_support(self):
        controller, holder, units = self.cover_controller()
        for c in units[3:]:
            controller.env.assignment[c.name] = ("Mid", "STAGING", "SUPPORT")
        self.assertEqual([c.name for c in controller._macro_cover_escorts(
            holder, {"chars": units})], ["main1", "main2"])

    def test_fake_sellers_keep_rotate_route(self):
        controller, holder, units = self.cover_controller("FAKE_A_TO_B", "ROTATE")
        for c in units:
            controller.env.assignment[c.name] = ("B", "STAGING", "FAKE_ROTATE")
        self.assertEqual([c.name for c in controller._macro_cover_escorts(
            holder, {"chars": units})], ["main1", "main2"])

    def test_default_keeps_scout_and_mid_control(self):
        controller, holder, units = self.cover_controller("DEFAULT")
        controller.env.assignment["other1"] = ("B", "INFO", "OPPOSITE_SCOUT")
        controller.env.assignment["other2"] = ("Mid", "FORWARD", "MID_CONTROL")
        self.assertEqual(len(controller._macro_cover_escorts(holder, {"chars": units})), 2)

    def test_dead_main_cover_can_be_replaced(self):
        controller, holder, units = self.cover_controller()
        for c in units[1:3]:
            c.is_alive = False
        for c in units[3:]:
            controller.env.assignment[c.name] = ("Mid", "STAGING", "SUPPORT")
        self.assertEqual([c.name for c in controller._macro_cover_escorts(
            holder, {"chars": units})], ["other1"])

    def test_default_info_lead_is_not_forced_back(self):
        controller, holder, units = self.cover_controller("DEFAULT")
        controller.env.assignment["main1"] = ("A", "INFO", "MAIN_LEAD")
        controller.env.assignment["other1"] = ("B", "INFO", "OPPOSITE_SCOUT")
        controller.env.assignment["other2"] = ("Mid", "FORWARD", "MID_CONTROL")
        self.assertEqual([c.name for c in controller._macro_cover_escorts(
            holder, {"chars": units})], ["main2"])

    def test_wrapper_does_not_overwrite_macro_roles(self):
        controller = GhostChampionsV1AttackerController.__new__(
            GhostChampionsV1AttackerController)
        controller.macro_controller = object()
        holder = unit("Absol", (5, 5), True)
        seller = unit("seller", (12, 12))
        result = ([12, 13], "MOVE")
        self.assertIs(controller._cover_result(seller, {"chars": [holder, seller]},
                                               result), result)

    def test_plant_deadline_precedes_waits_and_abilities(self):
        controller, holder, units = self.cover_controller()
        controller._plant_commit_holder = None
        controller._holder_on_plant_cell = lambda *a: False
        forced = ([6, 5], "MOVE")
        controller._hard_plant_deadline_result = lambda *a: forced
        controller._sync_tick_once = lambda *a: self.fail("deadline was delayed")
        self.assertIs(controller.coordinate(holder, {"chars": units},
                                              (list(holder.pos), {"ability": "FLASH"})), forced)

    def test_confirmed_threat_does_not_recall_split_support(self):
        controller, holder, units = self.cover_controller()
        units[2].pos = (1, 1)  # Only one main escort currently covers holder.
        seller = units[3]
        controller.env.assignment[seller.name] = ("Mid", "STAGING", "SUPPORT")
        controller.env.assignment[units[4].name] = ("Mid", "STAGING", "SUPPORT")
        controller.env.targets = {seller.name: (12, 13)}
        controller._plant_commit_holder = None
        controller._holder_on_plant_cell = lambda *a: False
        for method in ("_hard_plant_deadline_result", "_direct_carrier_plant_result",
                       "_emergency_plant_result", "_carrier_fast_route"):
            setattr(controller, method, lambda *a: None)
        controller._sync_tick_once = lambda *a: None
        controller._attacker_has_confirmed_threat = lambda *a: True
        controller._tick_id = lambda: 3
        controller._attacker_wait_ticks = {}
        with patch.object(runtime, "_bfs_next_step", side_effect=lambda g, s, t, o: t):
            result = controller.coordinate(seller, {"chars": units,
                                                   "grid": np.zeros((20, 20))},
                                           ([12, 11], "MOVE"))
        self.assertEqual(result[0], [12, 13])

    def fake_env(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        env.tick = 0
        env._fake_direction = "A_TO_B"
        env._fake_group_names = {"seller1", "seller2"}
        env.attackers = [unit("seller1", (1, 1)), unit("seller2", (1, 2)),
                         unit("Absol", (5, 5), True), unit("cover", (5, 6))]
        env.pressure = {"A": 1.0}
        env.control = {"A_FORWARD": 1.0}
        env._fake_sell_dwell = 0
        return env

    def test_pressure_without_sellers_does_not_complete_sell(self):
        env = self.fake_env()
        with patch.object(training, "side_of_pos", return_value="Mid"):
            self.assertFalse(env._fake_sell_trigger_ready())

    def test_fake_dwell_is_once_per_tick(self):
        env = self.fake_env()
        with patch.object(training, "side_of_pos", return_value="A"):
            self.assertFalse(env._fake_sell_trigger_ready())
            self.assertFalse(env._fake_sell_trigger_ready())
            self.assertEqual(env._fake_sell_dwell, 1)
            env.tick += 1
            self.assertTrue(env._fake_sell_trigger_ready())

    def test_lost_seller_contact_resets_dwell(self):
        env = self.fake_env()
        env._fake_sell_dwell = 5
        with patch.object(training, "side_of_pos", return_value="Mid"):
            self.assertFalse(env._fake_sell_trigger_ready())
            self.assertEqual(env._fake_sell_dwell, 0)

    def test_fake_redeploy_needs_carrier_and_cover(self):
        env = self.fake_env()
        env.current_strategy = "FAKE_A_TO_B"
        env._fake_phase = "ROTATE"
        env._diag_fake_max_opposite_count = 0
        env._advance_fake_rotate_targets = lambda: None
        env._set_fake_phase_targets = lambda phase: setattr(env, "_fake_phase", phase)
        env.fake_value = 0.0
        with patch.object(training, "side_of_pos",
                          side_effect=lambda pos: "B" if pos[0] == 1 else "Mid"):
            env._update_fake_option_phase()
            self.assertEqual(env._fake_phase, "ROTATE")
        with patch.object(training, "side_of_pos", return_value="B"), \
                patch.object(training, "nearest_distance", return_value=1):
            env._update_fake_option_phase()
            self.assertEqual(env._fake_phase, "EXECUTE")

    def test_split_wait_is_bounded(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        main, support = unit("main", (5, 5)), unit("support", (10, 10))
        env.attackers = [main, support]
        env.current_strategy = "A_SPLIT"
        env.target_site = "A"
        env.assignment = {"main": ("A", "DEEP", "MAIN"),
                          "support": ("Mid", "STAGING", "SUPPORT")}
        env.targets = {"main": (5, 5), "support": (10, 10)}
        env.tick = 20
        with patch.object(training, "nearest_distance", return_value=0), \
                patch.object(training, "target_for_side", return_value=(6, 5)):
            env._advance_assignment_phase_if_needed(main)
            self.assertEqual(env.assignment["main"][1], "DEEP")
            env.tick += training.SPLIT_SYNC_WAIT_MAX_TICKS
            env._advance_assignment_phase_if_needed(main)
            self.assertEqual(env.assignment["main"][1], "SITE")

    def test_fake_bonus_requires_effect_and_plant(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        for flag in ("_default_to_decision", "_smart_rotate_completed",
                     "_split_completed", "_lurk_touched", "_cut_touched"):
            setattr(env, flag, False)
        env._fake_completed = True
        env.curriculum_mode = "FAKE"
        env.tick = training.ROUND_DURATION_TICKS
        env._fake_entry_effect = 0.0
        without_effect = env._plant_completion_bonus()
        env._fake_entry_effect = 1.0
        self.assertAlmostEqual(env._plant_completion_bonus() - without_effect,
                               training.PLANT_AFTER_FAKE_BONUS + 0.8)

    def test_split_ready_support_releases_main_immediately(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        main, support = unit("main", (5, 5)), unit("support", (10, 10))
        env.attackers = [main, support]
        env.current_strategy = "A_SPLIT"
        env.target_site = "A"
        env.assignment = {"main": ("A", "DEEP", "MAIN"),
                          "support": ("A", "SPLIT_ENTRY", "SUPPORT")}
        env.targets = {"main": (5, 5), "support": (10, 10)}
        env.tick = 20
        with patch.object(training, "nearest_distance", return_value=0), \
                patch.object(training, "target_for_side", return_value=(6, 5)):
            env._advance_assignment_phase_if_needed(main)
            self.assertEqual(env.assignment["main"][1], "SITE")

    def test_split_dead_support_does_not_cause_wait(self):
        env = training.MacroEnv.__new__(training.MacroEnv)
        main, support = unit("main", (5, 5)), unit("support", (10, 10))
        support.is_alive = False
        env.attackers = [main, support]
        env.current_strategy = "A_SPLIT"
        env.target_site = "A"
        env.assignment = {"main": ("A", "DEEP", "MAIN"),
                          "support": ("Mid", "STAGING", "SUPPORT")}
        env.targets = {"main": (5, 5), "support": (10, 10)}
        env.tick = 20
        with patch.object(training, "nearest_distance", return_value=0), \
                patch.object(training, "target_for_side", return_value=(6, 5)):
            env._advance_assignment_phase_if_needed(main)
            self.assertEqual(env.assignment["main"][1], "SITE")

    def test_existing_checkpoint_loads_on_cpu(self):
        controller = runtime.LearningAttackerMacroGCController(device="cpu", verbose=False)
        self.assertEqual(controller.env.build_observation().shape, (training.OBS_DIM,))

    def test_short_simulation_keeps_finite_observations(self):
        for strategy in ("DEFAULT", "A_SPLIT", "FAKE_A_TO_B"):
            env = training.MacroEnv()
            env.reset()
            env.current_strategy = strategy
            env._apply_strategy_assignments(strategy, initial=True)
            for _ in range(8):
                obs, reward, done, _ = env.step(training.STRATEGY_TO_INDEX[strategy])
                self.assertEqual(obs.shape, (training.OBS_DIM,))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
                if done:
                    break


if __name__ == "__main__":
    unittest.main()
