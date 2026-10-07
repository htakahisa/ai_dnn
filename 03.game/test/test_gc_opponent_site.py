"""Final-site rules must follow the full defending roster, not an AI label."""

from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

from party_presets import get_preset
from gc_v1.opponent_site_gc import (
    A_SITE_CELLS, attack_plant_cells, enforce_attack_target, forced_attack_site,
)
from gc_v1.positioning_gc import parse_grid
from map_data import NEW_MAZE_STR
from test_gc_spike_recovery import controller, BaseGC, MacroGC
from test_ultimate_system import make_character, UltimateTestGame
import gc_v1.learning_attacker_macro_gc_runtime as runtime


class OpponentSiteTests(unittest.TestCase):
    def fixture(self):
        grid = parse_grid(NEW_MAZE_STR)
        actor = make_character("Absol", "A", (7, 40))
        actor.has_spike = True
        game = NS(grid=grid, chars=[actor], is_planted=False, round_timer=100,
                  defender_roster=list(get_preset("Touyama Gaming").players),
                  target_plant_pos=(7, 40), last_engagements=[], battle_tick=1)
        return game, actor

    def test_roster_order_and_team_name_or_ai_do_not_matter(self):
        game, _ = self.fixture()
        game.defender_roster.reverse()
        game.defender_team_name = "Custom label"
        game.opponent_ai = "default"
        self.assertEqual(forced_attack_site(game), "A")

    def test_named_touyama_with_different_roster_is_not_forced(self):
        game, _ = self.fixture()
        game.defender_team_name = "Touyama Gaming"
        game.defender_roster = list(get_preset("Omoko Gaming").players)
        self.assertIsNone(forced_attack_site(game))
        self.assertIn((7, 40), attack_plant_cells(game, game.grid))
        game.defender_roster = list(get_preset("Touyama Gaming").players)[:-1] + ["Absol"]
        self.assertIsNone(forced_attack_site(game))

    def test_player_keys_are_compared_by_display_name(self):
        from run_competition_manager import TeamPlayerKey
        game, _ = self.fixture()
        game.defender_roster = [TeamPlayerKey(name, "different-team-id")
                                for name in game.defender_roster]
        self.assertEqual(forced_attack_site(game), "A")

    def test_fallback_roster_includes_dead_players_and_side_swap_changes_rule(self):
        game, _ = self.fixture()
        names = game.defender_roster
        game.defender_roster = None
        game.chars = [NS(name=n, team="D", is_alive=False) for n in names]
        self.assertEqual(forced_attack_site(game), "A")
        game.defender_roster = list(get_preset("Ghost Champions").players)
        self.assertIsNone(forced_attack_site(game))

    def test_perception_view_uses_full_public_roster(self):
        game, actor = self.fixture()
        view = NS(real_game=game, chars=[actor], defender_roster=["Tortlilyan"])
        self.assertEqual(forced_attack_site(view), "A")

    def test_shared_target_is_a_even_if_carrier_is_already_on_b(self):
        game, actor = self.fixture()
        state = {"grid": game.grid, "chars": [actor], "is_planted": False,
                 "target_plant_pos": (7, 40)}
        enforce_attack_target(game, state)
        self.assertIn(game.target_plant_pos, A_SITE_CELLS)
        self.assertEqual(state["target_plant_pos"], game.target_plant_pos)
        self.assertTrue(all(p in A_SITE_CELLS for p in attack_plant_cells(game, game.grid)))

    def test_base_model_cannot_plant_b_but_can_plant_a(self):
        game, actor = self.fixture()
        ctrl = controller(BaseGC)
        ctrl.game = game
        ctrl.carry.decide_move.side_effect = lambda c, s: (list(c.pos), "PLANT")
        state = {"grid": game.grid, "chars": game.chars, "is_planted": False}
        self.assertEqual(ctrl.decide_move(actor, state)[1], "MOVE")
        actor.pos = list(attack_plant_cells(game, game.grid)[0])
        self.assertEqual(ctrl.decide_move(actor, state)[1], "PLANT")

    def test_macro_final_boundary_rejects_b_plant(self):
        game, actor = self.fixture()
        ctrl = controller(MacroGC)
        ctrl.game = game
        ctrl._decide_macro_move = Mock(return_value=(list(actor.pos), "PLANT"))
        state = {"grid": game.grid, "chars": game.chars, "is_planted": False}
        self.assertEqual(ctrl.decide_move(actor, state)[1], "MOVE")

    def test_lurk_and_precontact_hold_are_not_redirected(self):
        game, actor = self.fixture()
        ctrl = controller(MacroGC)
        ctrl.game = game
        state = {"grid": game.grid, "chars": game.chars, "is_planted": False}
        hold = (list(actor.pos), "MOVE")
        self.assertIs(ctrl._restrict_plant_result(actor, state, hold), hold)
        actor.has_spike = False
        lurk = ([7, 39], "MOVE")
        self.assertIs(ctrl._restrict_plant_result(actor, state, lurk), lurk)

    def test_pickup_on_b_routes_to_a_instead_of_planting_locally(self):
        game, actor = self.fixture()
        actor.has_spike = False
        ctrl = controller(BaseGC)
        ctrl.game = game
        state = {"grid": game.grid, "chars": game.chars, "is_planted": False,
                 "spike_pos": tuple(actor.pos), "round_timer": 100}
        ctrl.decide_move(actor, state)
        actor.has_spike = True
        state["spike_pos"] = None
        self.assertEqual(ctrl.decide_move(actor, state)[1], "MOVE")
        self.assertIn(ctrl.spike_recovery.plan["plant"], A_SITE_CELLS)
        self.assertGreater(ctrl.spike_recovery.last_status["required_ticks"], 4)

    def macro_fixture(self, strategy):
        game, actor = self.fixture()
        macro = runtime.LearningAttackerMacroGCController.__new__(
            runtime.LearningAttackerMacroGCController)
        macro.env = runtime.MacroEnv()
        macro.env.reset()
        macro.env.current_strategy = strategy
        macro.env._apply_strategy_assignments(strategy)
        macro.game = game
        macro.device = runtime.torch.device("cpu")
        macro.verbose = False
        macro._last_macro_decision_tick = None
        macro._opening_macro_decided = True
        macro._lurk_plan = None
        macro._plant_commit_pos = None
        macro._lurk_group_names = lambda: set(macro.env._fake_group_names) if "FAKE" in strategy else set()
        macro._observe_fake_group_contact = lambda: False
        macro.env.action_mask = lambda: np.ones(runtime.N_ACTIONS, dtype=bool)
        return macro, game, actor

    def test_b_bound_dqn_options_are_masked(self):
        macro, game, _ = self.macro_fixture("A_RUSH")
        values = runtime.torch.zeros((1, runtime.N_ACTIONS))
        values[0, runtime.STRATEGY_TO_INDEX["B_RUSH"]] = 100
        values[0, runtime.STRATEGY_TO_INDEX["A_SPLIT"]] = 10
        macro.model = lambda obs: values
        macro._maybe_decide_macro(1)
        self.assertEqual(macro.env.current_strategy, "A_SPLIT")
        self.assertEqual(macro.env.target_site, "A")
        self.assertIn(game.target_plant_pos, A_SITE_CELLS)

    def test_empty_site_filtered_mask_falls_back_to_a_default(self):
        macro, _, _ = self.macro_fixture("A_RUSH")
        mask = np.zeros(runtime.N_ACTIONS, dtype=bool)
        mask[runtime.STRATEGY_TO_INDEX["B_RUSH"]] = True
        macro.env.action_mask = lambda: mask
        macro.model = lambda obs: runtime.torch.zeros((1, runtime.N_ACTIONS))
        macro._maybe_decide_macro(1)
        self.assertEqual(macro.env.current_strategy, "DEFAULT")
        self.assertEqual(macro.env.target_site, "A")

    def test_other_roster_can_still_choose_b(self):
        macro, game, _ = self.macro_fixture("A_RUSH")
        game.defender_roster = list(get_preset("Omoko Gaming").players)
        values = runtime.torch.zeros((1, runtime.N_ACTIONS))
        values[0, runtime.STRATEGY_TO_INDEX["B_RUSH"]] = 100
        macro.model = lambda obs: values
        macro._maybe_decide_macro(1)
        self.assertEqual(macro.env.current_strategy, "B_RUSH")
        self.assertEqual(macro.env.target_site, "B")

    def test_default_main_a_keeps_opposite_scout_b(self):
        macro, _, _ = self.macro_fixture("DEFAULT")
        macro.env.target_site = "B"
        values = runtime.torch.zeros((1, runtime.N_ACTIONS))
        values[0, runtime.STRATEGY_TO_INDEX["DEFAULT"]] = 10
        macro.model = lambda obs: values
        macro._maybe_decide_macro(1)
        self.assertEqual(macro.env._default_main_side, "A")
        self.assertEqual(macro.env._default_opposite_side, "B")
        self.assertEqual(macro.env.target_site, "A")

    def test_locked_old_fake_is_changed_to_b_fake_a_finish(self):
        macro, _, _ = self.macro_fixture("FAKE_A_TO_B")
        macro.model = lambda obs: runtime.torch.zeros((1, runtime.N_ACTIONS))
        macro._maybe_decide_macro(1)
        self.assertEqual(macro.env.current_strategy, "FAKE_B_TO_A")
        self.assertEqual(macro.env._fake_real_side, "A")
        self.assertEqual(macro.env._fake_side, "B")

    def test_deadline_and_site_entry_fallbacks_never_plant_b(self):
        macro, game, actor = self.macro_fixture("A_RUSH")
        state = {"grid": game.grid, "chars": [actor], "is_planted": False,
                 "target_plant_pos": (7, 3), "round_timer": 2}
        self.assertFalse(macro._holder_on_plant_cell(actor, state))
        self.assertIsNone(macro._carrier_site_plant_result(actor, actor, state))
        result = macro._hard_plant_deadline_result(actor, actor, state)
        self.assertEqual(result[1], "MOVE")
        self.assertIn(game.target_plant_pos, A_SITE_CELLS)

    def test_actual_engine_recovered_b_spike_finishes_plant_on_a(self):
        game, actor = self.fixture()
        physical = UltimateTestGame(*game.grid.shape)
        physical.grid = game.grid
        physical.chars = game.chars
        physical.defender_roster = game.defender_roster
        ctrl = controller(BaseGC)
        physical.attacker_controller = ctrl
        ctrl.set_game(physical)
        actor.has_spike = False
        physical.spike_pos = tuple(actor.pos)
        physical.move_character(actor)  # Commit before actual automatic pickup.
        actor.has_spike = True
        physical.spike_pos = None
        for _ in range(90):
            physical.move_character(actor)
            if physical.is_planted:
                break
        self.assertTrue(physical.is_planted)
        self.assertIn(physical.planted_pos, A_SITE_CELLS)


if __name__ == "__main__":
    unittest.main()
