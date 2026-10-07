"""CPU regression tests; no training and no checkpoint writes."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gc_v1"))
import train_attacker_macro_gc_v28 as training
import learning_attacker_macro_gc_runtime as runtime
import learning_defender_opening_macro_gc_runtime as defender_opening
import ghost_champions_v1 as base_gc
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

    def test_fake_main_holds_until_lurk_group_takes_contact(self):
        controller, holder, units = self.cover_controller("FAKE_A_TO_B", "SELL")
        lurker = units[3]
        controller.env._fake_group_names = {lurker.name}
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller.game = NS(last_engagements=[])

        self.assertEqual(
            controller._fake_main_wait_result(holder),
            (list(holder.pos), "MOVE"),
        )

        defender = NS(name="defender", team="D", is_alive=True, pos=(12, 13))
        controller.game.last_engagements = [(lurker, defender)]
        self.assertIsNone(controller._fake_main_wait_result(holder))
        self.assertTrue(controller._fake_contact_confirmed)

        controller._fake_contact_confirmed = False
        controller.game.last_engagements = []
        self.assertEqual(
            controller._fake_main_wait_result(holder, {"round_timer": 1}),
            (list(holder.pos), "MOVE"),
        )

    def test_fake_lurker_is_never_blocked_by_main_wait_gate(self):
        controller, _holder, units = self.cover_controller("FAKE_B_TO_A", "SELL")
        lurker = units[3]
        controller.env._fake_group_names = {lurker.name}
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller.game = NS(last_engagements=[])

        self.assertIsNone(controller._fake_main_wait_result(lurker))

    def test_default_main_holds_until_opposite_lurk_takes_contact(self):
        controller, holder, units = self.cover_controller("DEFAULT")
        lurker = units[3]
        controller.env.assignment[lurker.name] = ("B", "LURK", "OPPOSITE_SCOUT_LURK")
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False

        self.assertEqual(controller._lurk_group_names(), {lurker.name})
        self.assertEqual(controller._fake_main_wait_result(holder),
                         (list(holder.pos), "MOVE"))
        defender = NS(name="defender", team="D", is_alive=True, pos=(12, 13))
        controller.game.last_engagements = [(lurker, defender)]
        self.assertIsNone(controller._fake_main_wait_result(holder))

    def test_default_gate_does_not_release_carrier_at_plant_deadline(self):
        controller, holder, units = self.cover_controller("DEFAULT")
        controller.env.assignment[units[3].name] = (
            "B", "INFO", "OPPOSITE_SCOUT"
        )
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        self.assertEqual(
            controller._fake_main_wait_result(holder, {"round_timer": 1}),
            (list(holder.pos), "MOVE"),
        )

    def clear_site_controller(self, strategy="DEFAULT", side="B"):
        from map_data import NEW_MAZE_STR

        controller, holder, units = self.cover_controller(strategy, "SELL")
        grid = np.array([[int(cell) for cell in line]
                         for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        cells = [tuple(map(int, p)) for p in zip(*np.where(grid == 2))
                 if runtime.side_of_pos(p) == side]
        holder.pos = (23, 21)
        units[1].pos, units[2].pos = (23, 20), (24, 21)
        lurker = units[3]
        lurker.pos = cells[0]
        units[4].pos = (23, 18)
        other_side = "A" if side == "B" else "B"
        controller.env.assignment[lurker.name] = (side, "INFO", "OPPOSITE_SCOUT")
        controller.env._fake_group_names = {lurker.name}
        controller.env._fake_sides = lambda: (side, other_side)
        controller.env.targets = {c.name: c.pos for c in units}
        controller.game = NS(grid=grid, chars=units, battle_tick=1,
                             last_engagements=[], target_plant_pos=(8, 3) if side == "B" else (7, 40))
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller._plant_commit_pos = None
        state = {"grid": grid, "chars": units, "is_planted": False}
        return controller, holder, lurker, state

    def test_clear_lurk_site_releases_main_and_retargets_both_fake_directions(self):
        for strategy, side in (("DEFAULT", "A"), ("DEFAULT", "B"),
                               ("FAKE_A_TO_B", "A"), ("FAKE_B_TO_A", "B")):
            with self.subTest(strategy=strategy, side=side):
                controller, holder, lurker, state = self.clear_site_controller(strategy, side)
                self.assertIsNone(controller.lurk_coordination_result(holder, state))
                self.assertEqual(controller.env.target_site, side)
                target = controller.game.target_plant_pos
                self.assertEqual(runtime.side_of_pos(target), side)
                self.assertEqual(int(state["grid"][target]), 2)
                self.assertNotEqual(target, lurker.pos)
                self.assertEqual(controller.env.targets[holder.name], target)
                self.assertIsNone(controller._fake_main_wait_result(holder, state))
                self.assertFalse(controller._fake_contact_confirmed)
                # A later Macro decision may not immediately abandon the reunion.
                controller._maybe_decide_macro(10)
                controller._retarget_real_plant_position(f"{side}_SPLIT")
                self.assertEqual(controller.game.target_plant_pos, target)

    def test_lurk_approach_and_dead_lurker_do_not_release_main(self):
        for alive, pos in ((True, (23, 18)), (False, (7, 40))):
            with self.subTest(alive=alive):
                controller, holder, lurker, state = self.clear_site_controller()
                lurker.pos, lurker.is_alive = pos, alive
                self.assertEqual(controller.lurk_coordination_result(holder, state),
                                 (list(holder.pos), "MOVE"))
                self.assertIsNone(getattr(controller, "_lurk_reunion", None))

    def test_visible_or_revealed_site_defender_prevents_clearance(self):
        for revealed in (False, True):
            with self.subTest(revealed=revealed):
                controller, holder, lurker, state = self.clear_site_controller()
                defender = NS(name="defender", team="D", is_alive=True,
                              pos=(lurker.pos[0], lurker.pos[1] + 1), revealed=revealed)
                state["chars"].append(defender)
                with patch.object(runtime, "_has_los", return_value=not revealed):
                    self.assertEqual(controller.lurk_coordination_result(holder, state),
                                     (list(holder.pos), "MOVE"))

    def test_hidden_defenders_do_not_supply_clearance_information(self):
        controller, holder, lurker, state = self.clear_site_controller()
        state["chars"].append(NS(name="hidden", team="D", is_alive=True,
                                 pos=(lurker.pos[0], lurker.pos[1] + 1)))
        with patch.object(runtime, "_has_los", return_value=False):
            self.assertIsNone(controller.lurk_coordination_result(holder, state))
        self.assertEqual(controller.env.target_site, "B")

    def test_opposite_site_defender_does_not_block_clear_lurk_site(self):
        controller, holder, _lurker, state = self.clear_site_controller()
        state["chars"].append(NS(name="opposite", team="D", is_alive=True,
                                 pos=(8, 3), revealed=True))
        self.assertIsNone(controller.lurk_coordination_result(holder, state))
        self.assertEqual(controller.env.target_site, "B")

    def test_clear_lurk_site_respects_existing_opponent_site_restriction(self):
        controller, holder, _lurker, state = self.clear_site_controller()
        from gc_v1.opponent_site_gc import TOUYAMA_ROSTER

        controller.game.defender_roster = list(TOUYAMA_ROSTER)
        self.assertEqual(controller.lurk_coordination_result(holder, state),
                         (list(holder.pos), "MOVE"))

    def test_iq_wrapper_passes_clear_lurk_site_to_learned_carry(self):
        from iq_controller_adapter import IQAwareController

        macro, holder, _lurker, state = self.clear_site_controller()
        macro._sync_tick_once = lambda state: None
        wrapper = GhostChampionsV1AttackerController.__new__(GhostChampionsV1AttackerController)
        wrapper.macro_controller = macro
        wrapper.carry = NS(positioning_version=1)
        wrapper.escort = NS(positioning_version=2)
        macro.game.current_round = 1
        macro.game.is_planted = False
        macro.game.spike_pos = None
        macro.game.planted_pos = None
        macro.game.round_timer = 180
        macro.game.detonate_timer = 0
        for char in state["chars"]:
            char.hp = 100

        def set_game(view):
            wrapper.game = view
            macro.set_game(view)

        wrapper.set_game = set_game
        adapter = IQAwareController(wrapper)
        game = macro.game
        adapter.set_game(game)
        observed_targets = []

        def carry_action(char, perceived_state):
            observed_targets.append(perceived_state["target_plant_pos"])
            return ([22, 21], "MOVE")

        with patch.object(base_gc.GhostChampionsV1AttackerController, "decide_move",
                          side_effect=carry_action):
            self.assertEqual(adapter.decide_move(holder, state), ([22, 21], "MOVE"))
        self.assertEqual(len(observed_targets), 1)
        self.assertEqual(runtime.side_of_pos(observed_targets[0]), "B")
        self.assertEqual(game.target_plant_pos, observed_targets[0])

    def test_real_macro_sync_keeps_reunion_and_round_reset_restores_lurk_gate(self):
        controller, holder, lurker, state = self.clear_site_controller("FAKE_B_TO_A", "B")
        old_env = controller.env
        controller.env = training.MacroEnv()
        controller.env.reset(forced_curriculum_mode="FREE")
        controller.env.attackers = state["chars"]
        controller.env.current_strategy = old_env.current_strategy
        controller.env.assignment = old_env.assignment
        controller.env.targets = old_env.targets
        controller.env._fake_group_names = old_env._fake_group_names
        controller.env._fake_sides = old_env._fake_sides
        controller._last_real_tick = None
        controller._fake_peak_key = None
        controller._fake_peak_stages = {}
        controller._fake_peak_targets = {}
        controller._attacker_wait_ticks = {}
        controller._lurk_route_state = {}
        controller._sync_tick_once(state)
        target = controller.game.target_plant_pos
        self.assertEqual(runtime.side_of_pos(target), "B")
        for tick in (2, 3, 6):
            controller.game.battle_tick = tick
            controller._sync_tick_once(state)
            self.assertEqual(controller.game.target_plant_pos, target)
            self.assertEqual(controller.env.targets[holder.name], target)
            self.assertEqual(controller.env.targets[lurker.name], lurker.pos)
        # A carrier already planting at another legal cell keeps its commitment.
        controller._plant_commit_pos = lurker.pos
        controller.game.battle_tick = 7
        controller._sync_tick_once(state)
        self.assertEqual(controller.game.target_plant_pos, lurker.pos)
        controller.reset_round()
        self.assertIsNone(controller._lurk_reunion)
        self.assertIsNone(controller._plant_commit_pos)
        controller.env.current_strategy = "DEFAULT"
        controller.env.assignment = {lurker.name: ("B", "INFO", "OPPOSITE_SCOUT")}
        lurker.pos = (23, 18)
        self.assertEqual(controller.lurk_coordination_result(holder, state),
                         (list(holder.pos), "MOVE"))

    def test_default_lurk_keeps_advancing_when_model_waits(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        lurker = unit("lurker", (1, 1))
        controller.env = NS(
            current_strategy="DEFAULT",
            assignment={"lurker": ("A", "LURK", "OPPOSITE_SCOUT_LURK")},
            targets={"lurker": (1, 3)},
        )
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        with patch.object(runtime, "_bfs_next_step", side_effect=lambda g, s, t, o: (1, 2)):
            result = controller._default_lurk_progress_result(
                lurker, {"grid": np.zeros((48, 48), dtype=np.int8), "chars": [lurker]}
            )
        self.assertEqual(result, ([1, 2], "MOVE"))

    def test_lurk_advances_to_new_waypoints_instead_of_stalling_at_first_target(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        lurker = unit("lurker", (1, 1))
        controller.env = NS(
            current_strategy="DEFAULT",
            assignment={"lurker": ("A", "LURK", "OPPOSITE_SCOUT_LURK")},
            targets={"lurker": (1, 3)},
        )
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        with patch.object(runtime, "_bfs_next_step", side_effect=lambda g, s, t, o: t):
            controller._default_lurk_progress_result(
                lurker, {"grid": np.zeros((48, 48), dtype=np.int8), "chars": [lurker]}
            )
            lurker.pos = (1, 3)
            result = controller._default_lurk_progress_result(
                lurker, {"grid": np.zeros((48, 48), dtype=np.int8), "chars": [lurker]}
            )
        self.assertNotEqual(controller._lurk_route_state["lurker"]["target"], (1, 3))
        self.assertNotEqual(result[0], list(lurker.pos))

    def test_carrier_plants_immediately_on_legal_site_cell(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        holder = unit("holder", (2, 2), True)
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[2, 2] = 2
        self.assertEqual(
            controller._carrier_site_plant_result(
                holder, holder, {"grid": grid, "chars": [holder]}
            ),
            ([2, 2], "PLANT"),
        )

    def test_carrier_in_site_area_routes_to_plant_cell_without_entry_waypoint(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        holder = unit("holder", (2, 2), True)
        grid = np.zeros((6, 6), dtype=np.int8)
        grid[2, 4] = 2
        with (patch.object(runtime, "side_of_pos", return_value="A"),
              patch.object(runtime, "_site_cells", return_value=[(2, 2)])):
            result = controller._carrier_site_plant_result(
                holder, holder, {"grid": grid, "chars": [holder]}
            )
        self.assertEqual(result, ([2, 3], "MOVE"))

    def test_real_map_lurker_reaches_site_through_iq_wrapper_without_rejoining(self):
        from iq_controller_adapter import IQAwareController
        from map_data import NEW_MAZE_STR

        grid = np.array([[int(cell) for cell in line]
                         for line in NEW_MAZE_STR.strip().splitlines()], dtype=np.int8)
        cases = [("FAKE_A_TO_B", "A"), ("FAKE_B_TO_A", "B"),
                 ("DEFAULT", "A"), ("DEFAULT", "B")]
        for strategy, side in cases:
            with self.subTest(strategy=strategy, side=side):
                holder = unit("Absol", (23, 21), True)
                lurker = unit("lurker", (23, 18))
                main = unit("main", (23, 20))
                chars = [holder, main, lurker]
                for char in chars:
                    char.hp = 100
                macro = runtime.LearningAttackerMacroGCController.__new__(
                    runtime.LearningAttackerMacroGCController
                )
                other_side = "B" if side == "A" else "A"
                macro.env = NS(
                    current_strategy=strategy, _fake_phase="SELL",
                    _fake_group_names={lurker.name},
                    _fake_sides=lambda: (side, other_side),
                    assignment={lurker.name: (side, "INFO", "OPPOSITE_SCOUT")},
                    targets={lurker.name: (15, 4) if side == "A" else (18, 39)},
                )
                macro._fake_contact_key = None
                macro._fake_contact_confirmed = False
                macro._sync_tick_once = lambda state: None
                wrapper = GhostChampionsV1AttackerController.__new__(
                    GhostChampionsV1AttackerController
                )
                wrapper.carry = NS(positioning_version=1)
                wrapper.escort = NS(positioning_version=2)
                wrapper.macro_controller = macro

                def set_game(view):
                    wrapper.game = view
                    macro.set_game(view)

                wrapper.set_game = set_game
                game = NS(grid=grid, chars=chars, battle_tick=0, current_round=1,
                          last_engagements=[], is_planted=False, spike_pos=None,
                          planted_pos=None, target_plant_pos=(8, 3) if other_side == "A" else (7, 40),
                          round_timer=180, detonate_timer=0)
                adapter = IQAwareController(wrapper)
                adapter.set_game(game)
                site = [tuple(map(int, p)) for p in zip(*np.where(grid == 2))
                        if runtime.side_of_pos(p) == side]
                shortest = min(runtime._bfs_distance(grid, lurker.pos, p) for p in site)
                first_site_los = None
                with patch.object(base_gc.GhostChampionsV1AttackerController,
                                  "decide_move", side_effect=AssertionError("pre-contact policy fallback")):
                    for tick in range(1, shortest + 8):
                        game.battle_tick = tick
                        state = {"grid": grid, "chars": chars, "is_planted": False,
                                 "round_timer": 180 - tick}
                        if tick == 3:
                            # A phase/role refresh must not recall the lurker.
                            macro.env._fake_phase = "ROTATE"
                            macro.env.assignment[lurker.name] = (other_side, "SITE", "MAIN")
                            macro.env.targets[lurker.name] = game.target_plant_pos
                        for waiting in (holder, main):
                            result = adapter.decide_move(waiting, state)
                            self.assertEqual(tuple(result[0]), waiting.pos)
                        result = adapter.decide_move(lurker, state)
                        destination = tuple(result[0])
                        self.assertEqual(abs(destination[0] - lurker.pos[0])
                                         + abs(destination[1] - lurker.pos[1]), 1)
                        self.assertNotEqual(int(grid[destination]), 1)
                        lurker.pos = destination
                        if first_site_los is None and any(runtime._has_los(grid, destination, p) for p in site):
                            first_site_los = tick
                        if int(grid[destination]) == 2:
                            break
                    else:
                        self.fail("lurker never reached its assigned opposite site")
                self.assertIsNotNone(first_site_los)
                self.assertLessEqual(tick, shortest + 4)
                self.assertEqual(runtime.side_of_pos(lurker.pos), side)
                defender = NS(name="defender", team="D", pos=site[0])
                game.last_engagements = [(lurker, defender)]
                self.assertIsNone(macro.lurk_coordination_result(holder, state))

    def test_blocked_lurker_keeps_ownership_instead_of_escort_reunion(self):
        controller, holder, units = self.cover_controller("DEFAULT")
        lurker = units[3]
        lurker.pos = (2, 2)
        controller.env.assignment[lurker.name] = ("A", "LURK", "OPPOSITE_SCOUT_LURK")
        controller.env.targets = {lurker.name: (2, 4)}
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        grid = np.ones((6, 6), dtype=np.int8)
        grid[2, 2] = 0
        grid[2, 4] = 2
        self.assertEqual(controller.lurk_coordination_result(
            lurker, {"grid": grid, "chars": [lurker, holder]}
        ), ([2, 2], "MOVE"))

    def test_fake_strategy_cannot_switch_to_split_before_lurk_contact(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        controller.env = NS(
            current_strategy="FAKE_A_TO_B",
            _fake_group_names={"lurker"},
            _fake_phase="SELL",
            strategy_age=0,
            build_observation=lambda: np.zeros(runtime.OBS_DIM, dtype=np.float32),
            action_mask=lambda: np.ones(runtime.N_ACTIONS, dtype=bool),
        )
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller._fake_peak_key = None
        controller._fake_peak_stages = {}
        controller._fake_peak_targets = {}
        controller._last_macro_decision_tick = None
        controller.device = runtime.torch.device("cpu")
        q = runtime.torch.zeros((1, runtime.N_ACTIONS))
        q[0, runtime.STRATEGY_TO_INDEX["A_SPLIT"]] = 1.0
        controller.model = lambda _obs: q
        controller._maybe_decide_macro(1)
        self.assertEqual(controller.env.current_strategy, "FAKE_A_TO_B")

    def test_initial_macro_choice_is_not_locked_to_reset_default(self):
        controller, _holder, units = self.cover_controller("DEFAULT")
        controller.env.assignment[units[3].name] = ("B", "INFO", "OPPOSITE_SCOUT")
        controller.env.strategy_age = 0
        controller.env.build_observation = lambda: np.zeros(runtime.OBS_DIM, dtype=np.float32)
        controller.env.action_mask = lambda: np.ones(runtime.N_ACTIONS, dtype=bool)
        controller.env._apply_strategy_assignments = lambda *args, **kwargs: None
        controller._opening_macro_decided = False
        controller._last_macro_decision_tick = None
        controller.device = runtime.torch.device("cpu")
        controller.verbose = False
        controller._retarget_real_plant_position = lambda strategy: None
        q = runtime.torch.zeros((1, runtime.N_ACTIONS))
        q[0, runtime.STRATEGY_TO_INDEX["FAKE_B_TO_A"]] = 1.0
        controller.model = lambda _obs: q
        controller._maybe_decide_macro(1)
        self.assertEqual(controller.env.current_strategy, "FAKE_B_TO_A")

    def test_macro_path_routes_around_an_occupied_waypoint(self):
        grid = np.zeros((7, 7), dtype=np.int8)
        start, goal = (1, 1), (1, 3)
        next_pos = runtime._bfs_next_step(grid, start, goal, {goal})

        self.assertEqual(next_pos, (1, 2))

    def test_fake_lurker_is_forced_toward_its_forward_waypoint(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        lurker = unit("lurker", (1, 1))
        controller.env = NS(
            current_strategy="FAKE_A_TO_B",
            _fake_phase="SELL",
            _fake_group_names={"lurker"},
            _fake_sides=lambda: ("A", "B"),
            targets={"lurker": (1, 3)},
        )
        controller.game = NS(last_engagements=[])
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller._fake_peak_key = None
        controller._fake_peak_stages = {}
        controller._fake_peak_targets = {}

        result = controller._fake_sell_progress_result(
            lurker,
            {"grid": np.zeros((48, 48), dtype=np.int8), "chars": [lurker]},
        )

        self.assertEqual(result, ([1, 2], "MOVE"))

    def test_fake_phase_cannot_rotate_before_lurk_contact(self):
        controller = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController
        )
        lurker = unit("lurker", (12, 12))
        defender = NS(name="defender", team="D", is_alive=True, pos=(12, 13))
        phase_updates = []
        controller.env = NS(
            current_strategy="FAKE_A_TO_B",
            _fake_phase="SELL",
            _fake_group_names={lurker.name},
            attackers=[],
            tick=0,
            macro_step=0,
            _living_attackers=lambda: [],
            _update_fake_option_phase=lambda: phase_updates.append("updated"),
            _update_tactical_history=lambda: None,
        )
        controller.game = NS(battle_tick=1, last_engagements=[])
        controller._last_real_tick = None
        controller._fake_contact_key = None
        controller._fake_contact_confirmed = False
        controller._sync_attackers = lambda state: None
        controller._update_information_from_real_game = lambda state: None
        controller._maybe_decide_macro = lambda tick: None

        controller._sync_tick_once({})
        self.assertEqual(phase_updates, [])

        controller.game.battle_tick = 2
        controller.game.last_engagements = [(lurker, defender)]
        controller._sync_tick_once({})
        self.assertEqual(phase_updates, ["updated"])

    def test_wrapper_applies_fake_wait_before_learned_policy_returns(self):
        controller = GhostChampionsV1AttackerController.__new__(
            GhostChampionsV1AttackerController
        )
        holder = unit("holder", (5, 5), True)
        lurker = unit("lurker", (12, 12))
        controller.game = None
        controller.carry = NS(positioning_version=1)
        controller.escort = NS(positioning_version=2)
        controller.macro_controller = NS(
            env=NS(current_strategy="FAKE_A_TO_B", _fake_phase="SELL"),
            _sync_tick_once=lambda state: None,
            _fake_main_wait_result=lambda char, state: (list(char.pos), "MOVE"),
        )

        with patch.object(
            base_gc.GhostChampionsV1AttackerController,
            "decide_move",
            return_value=([5, 6], "MOVE"),
        ):
            result = controller.decide_move(
                holder,
                {"chars": [holder, lurker], "is_planted": False},
            )

        self.assertEqual(result, ([5, 5], "MOVE"))

    def test_wrapper_applies_default_lurk_gate_before_learned_carry_returns(self):
        controller = GhostChampionsV1AttackerController.__new__(
            GhostChampionsV1AttackerController
        )
        holder = unit("holder", (5, 5), True)
        controller.game = None
        controller.carry = NS(positioning_version=1)
        controller.escort = NS(positioning_version=2)
        controller.macro_controller = NS(
            env=NS(current_strategy="DEFAULT"),
            _sync_tick_once=lambda state: None,
            _lurk_group_names=lambda: {"lurker"},
            _fake_main_wait_result=lambda char, state: (list(char.pos), "MOVE"),
        )
        with patch.object(
            base_gc.GhostChampionsV1AttackerController,
            "decide_move",
            return_value=([5, 6], "MOVE"),
        ):
            result = controller.decide_move(
                holder, {"chars": [holder], "is_planted": False}
            )
        self.assertEqual(result, ([5, 5], "MOVE"))

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
        self.assertEqual(controller.env.build_observation().shape, (training.ROSTER_OBS_DIM,))

    def test_defender_recon_waits_for_attacker_arrival_window(self):
        controller = defender_opening.LearningDefenderOpeningMacroGCRuntime.__new__(
            defender_opening.LearningDefenderOpeningMacroGCRuntime
        )
        controller.tick = 10
        controller._runtime_current_ability = "RECON"
        controller._runtime_ready_tick = {"RECON": 0}
        controller._runtime_delay = {"RECON": 0}
        controller.plans = {
            "RECON": NS(origin=(5, 1), target=(5, 5))
        }
        grid = np.zeros((20, 20), dtype=np.int32)
        far = unit("far", (19, 19))
        far.team = "A"
        near = unit("near", (5, 11))
        near.team = "A"

        self.assertEqual(
            controller._policy_execution_action({"grid": grid, "chars": [far]}),
            defender_opening.EXEC_WAIT,
        )
        self.assertEqual(
            controller._policy_execution_action({"grid": grid, "chars": [near]}),
            defender_opening.EXECUTE,
        )

    def test_short_simulation_keeps_finite_observations(self):
        for strategy in ("DEFAULT", "A_SPLIT", "FAKE_A_TO_B"):
            env = training.MacroEnv()
            env.reset()
            env.current_strategy = strategy
            env._apply_strategy_assignments(strategy, initial=True)
            for _ in range(8):
                obs, reward, done, _ = env.step(training.STRATEGY_TO_INDEX[strategy])
                self.assertEqual(obs.shape, (training.ROSTER_OBS_DIM,))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward))
                if done:
                    break


if __name__ == "__main__":
    unittest.main()
